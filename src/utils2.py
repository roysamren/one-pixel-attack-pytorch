def init_params(net):
    '''Init layer parameters (Conv2d, BatchNorm2d, Linear).'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')  # underscore version
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)


# Attempt to get terminal width, fallback if it fails
try:
    rows, term_width_str = os.popen('stty size', 'r').read().split()
    term_width = int(term_width_str)
except:
    term_width = 80

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    """
    A simple progress bar to mimic xlua.progress from Torch.
    """
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    sys.stdout.write('=' * cur_len)
    sys.stdout.write('>')
    sys.stdout.write('.' * rest_len)
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)
    msg_str = ''.join(L)
    sys.stdout.write(msg_str)

    # Fill the remaining width with spaces
    remaining = term_width - int(TOTAL_BAR_LENGTH) - len(msg_str) - 3
    if remaining > 0:
        sys.stdout.write(' ' * remaining)

    # Move back to center
    back_len = term_width - int(TOTAL_BAR_LENGTH / 2) + 2
    for _ in range(back_len):
        sys.stdout.write('\b')

    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds // 3600 // 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds // 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds // 60)
    secondsf = int(seconds - minutes * 60)
    millis = int((seconds - secondsf) * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f