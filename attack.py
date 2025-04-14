import os
import sys
import numpy as np

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from models import *
from utils import progress_bar
from torch.autograd import Variable

from differential_evolution import differential_evolution

parser = argparse.ArgumentParser(description='One pixel attack with PyTorch')
parser.add_argument('--model', default='ResNet101', help='The target model')
parser.add_argument('--pixels', default=1, type=int, help='The number of pixels that can be perturbed.')
parser.add_argument('--maxiter', default=100, type=int, help='The maximum number of iteration in the DE algorithm.')
parser.add_argument('--popsize', default=400, type=int, help='The number of adversarial examples in each iteration.')
parser.add_argument('--samples', default=100, type=int, help='The number of image samples to attack.')
parser.add_argument('--targeted', action='store_true', help='Set this switch to test for targeted attacks.')
parser.add_argument('--save', default='./results/results.pkl', help='Save location for the results with pickle.')
parser.add_argument('--verbose', action='store_true', help='Print out additional information every iteration.')

args = parser.parse_args()


def perturb_image(xs, img):
    """
    Perturbs the given image tensor by changing pixel values as specified in xs.
    xs: array of shape [population_size, 5 * pixels] or [5 * pixels].
    img: a torch.Tensor of shape [3, H, W].
    Returns a batch of perturbed images.
    """
    if xs.ndim < 2:
        xs = np.array([xs])
    batch = len(xs)
    # Expand img to (batch_size, 3, H, W) so that each row of xs can perturb one copy
    imgs = img.repeat(batch, 1, 1, 1)
    xs = xs.astype(int)

    count = 0
    for x in xs:
        pixels = np.split(x, len(x) // 5)
        for pixel in pixels:
            x_pos, y_pos, r, g, b = pixel
            # Scale from [0,255] to the normalized range, using CIFAR means/stds
            imgs[count, 0, x_pos, y_pos] = (r / 255.0 - 0.4914) / 0.2023
            imgs[count, 1, x_pos, y_pos] = (g / 255.0 - 0.4822) / 0.1994
            imgs[count, 2, x_pos, y_pos] = (b / 255.0 - 0.4465) / 0.2010
        count += 1

    return imgs


def predict_classes(xs, img, target_class, net, minimize=True):
    """
    Returns the predicted probability of `target_class` for each perturbed sample.
    If minimize=True, we return P(target_class). Otherwise, we return 1 - P(target_class).
    """
    perturbed_images = perturb_image(xs, img.clone()).cuda()

    with torch.no_grad():  # Modern PyTorch usage instead of Variable(., volatile=True)
        logits = net(perturbed_images)
        probs = F.softmax(logits, dim=1)

    # Probability of the target_class
    probs_target = probs[:, target_class].cpu().numpy()
    return probs_target if minimize else (1.0 - probs_target)


def attack_success(x, img, target_class, net, targeted_attack=False, verbose=False):
    """
    Checks if applying x as a perturbation leads to a successful attack (target hit in targeted mode,
    or misclassification in untargeted mode).
    """
    perturbed_image = perturb_image(x, img.clone()).cuda()

    with torch.no_grad():
        logits = net(perturbed_image)
        probs = F.softmax(logits, dim=1)
        probs = probs[0].cpu().numpy()

    predicted_class = np.argmax(probs)

    if verbose:
        print("Confidence for target class {} = {:.4f}".format(target_class, probs[target_class]))

    if targeted_attack:
        return predicted_class == target_class
    else:
        return predicted_class != target_class


def attack(img, label, net, target=None, pixels=1, maxiter=75, popsize=400, verbose=False):
    """
    Runs a differential evolution-based attack on a single image.
    img: shape [1, 3, H, W]
    label: ground-truth label
    target: label to target if targeted attack, else None
    """
    targeted_attack = (target is not None)
    target_class = target if targeted_attack else label

    # We treat the image as 32x32 for CIFAR, so bounds must match that
    # Each pixel = (x_pos, y_pos, R, G, B) → 5 values
    bounds = [(0, 32), (0, 32), (0, 255), (0, 255), (0, 255)] * pixels

    # popmul is just a multiplier
    popmul = max(1, popsize // len(bounds))

    predict_fn = lambda xs: predict_classes(xs, img, target_class, net, minimize=(not targeted_attack))
    callback_fn = lambda x, convergence: attack_success(x, img, target_class, net, targeted_attack, verbose)

    # Initialize population around random pixel values
    inits = np.zeros([popmul * len(bounds), len(bounds)])
    for init in inits:
        for i in range(pixels):
            init[i * 5 + 0] = np.random.rand() * 32
            init[i * 5 + 1] = np.random.rand() * 32
            init[i * 5 + 2] = np.random.normal(128, 127)
            init[i * 5 + 3] = np.random.normal(128, 127)
            init[i * 5 + 4] = np.random.normal(128, 127)

    attack_result = differential_evolution(
        predict_fn,
        bounds,
        maxiter=maxiter,
        popsize=popmul,
        recombination=1,
        atol=-1,
        callback=callback_fn,
        polish=False,
        init=inits
    )

    # After DE finishes, check the best solution
    perturbed = perturb_image(attack_result.x, img)
    perturbed = perturbed.cuda()
    with torch.no_grad():
        logits = net(perturbed)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    predicted_class = np.argmax(probs)

    # Evaluate success
    if targeted_attack:
        success = (predicted_class == target_class)
    else:
        success = (predicted_class != label)

    if success:
        return 1, attack_result.x.astype(int)
    return 0, [None]


def attack_all(net, loader, pixels=1, targeted=False, maxiter=75, popsize=400, verbose=False):
    """
    Runs attacks on a batch of samples from loader (CIFAR-10).
    """
    correct = 0
    success = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # Check if net already misclassifies -> skip
        with torch.no_grad():
            logits = net(inputs)
            _, pred = logits.max(1)

        if targets.item() != pred.item():
            continue

        correct += 1
        # Convert to numpy for indexing
        label_np = targets.cpu().numpy()

        # If not targeted -> [None], else check all classes
        target_classes = [None] if not targeted else range(10)

        for tgt in target_classes:
            if targeted and tgt == label_np[0]:
                continue

            flag, x = attack(inputs, label_np[0], net, tgt, pixels=pixels, maxiter=maxiter,
                             popsize=popsize, verbose=verbose)
            success += flag

            if targeted:
                # if targeted, success_rate out of 9 * correct
                success_rate = float(success) / (9 * correct)
            else:
                success_rate = float(success) / correct

            if flag == 1:
                print("success rate: %.4f (%d/%d) [pixel: (x=%d,y=%d), (R,G,B)=(%d,%d,%d)]"
                      % (success_rate, success, correct, x[0], x[1], x[2], x[3], x[4]))

        if correct == args.samples:
            break

    return float(success) / correct if correct != 0 else 0.0


def main():
    print("==> Loading data and model...")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=2)

    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog',
                   'frog', 'horse', 'ship', 'truck']

    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7', weights_only=False)
    net = checkpoint['net']
    net.cuda()
    cudnn.benchmark = True

    print("==> Starting attack...")

    results = attack_all(net, testloader, pixels=args.pixels, targeted=args.targeted,
                         maxiter=args.maxiter, popsize=args.popsize, verbose=args.verbose)
    print("Final success rate: %.4f" % results)


if __name__ == '__main__':
    main()
