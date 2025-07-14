"""
A slight modification to SciPy's implementation of differential evolution.
...
"""

import numpy as np
from scipy.optimize import OptimizeResult, minimize
#from scipy.optimize.optimize import _status_message
from scipy._lib._util import check_random_state
import warnings
np.float_ = np.float64
_status_message = {
    'success': 'Optimization terminated successfully.',
    'maxiter': 'Maximum number of iterations has been exceeded.',
    'maxfev': 'Maximum number of function evaluations has been exceeded.'
}

__all__ = ['differential_evolution']

_MACHEPS = np.finfo(np.float64).eps


def differential_evolution(func, bounds, args=(),
                           strategy='best1bin',
                           maxiter=1000,
                           popsize=15,
                           tol=0.01,
                           mutation=(0.5, 1),
                           recombination=0.7,
                           seed=None,
                           callback=None,
                           disp=False,
                           polish=True,
                           init='latinhypercube',
                           atol=0):
    """
    Find the global minimum of a multivariate function using differential evolution.
    (Code shortened for brevity; see original for full docstring.)
    """

    solver = DifferentialEvolutionSolver(func, bounds, args=args,
                                         strategy=strategy, maxiter=maxiter,
                                         popsize=popsize, tol=tol,
                                         mutation=mutation,
                                         recombination=recombination,
                                         seed=seed, polish=polish,
                                         callback=callback,
                                         disp=disp, init=init, atol=atol)
    return solver.solve()


class DifferentialEvolutionSolver(object):
    """
    This class implements the differential evolution solver.
    (Code shortened for brevity; see original for docstring.)
    """

    _binomial = {'best1bin': '_best1',
                 'randtobest1bin': '_randtobest1',
                 'currenttobest1bin': '_currenttobest1',
                 'best2bin': '_best2',
                 'rand2bin': '_rand2',
                 'rand1bin': '_rand1'}

    _exponential = {'best1exp': '_best1',
                    'rand1exp': '_rand1',
                    'randtobest1exp': '_randtobest1',
                    'currenttobest1exp': '_currenttobest1',
                    'best2exp': '_best2',
                    'rand2exp': '_rand2'}

    __init_error_msg = ("The population initialization method must be one of "
                        "'latinhypercube' or 'random', or an array of shape "
                        "(M, N) where N is the number of parameters and M>5")

    def __init__(self, func, bounds, args=(),
                 strategy='best1bin', maxiter=1000, popsize=15,
                 tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None,
                 maxfun=np.inf, callback=None, disp=False, polish=True,
                 init='latinhypercube', atol=0):
        if strategy in self._binomial:
            self.mutation_func = getattr(self, self._binomial[strategy])
        elif strategy in self._exponential:
            self.mutation_func = getattr(self, self._exponential[strategy])
        else:
            raise ValueError("Please select a valid mutation strategy")
        self.strategy = strategy

        self.callback = callback
        self.polish = polish
        self.tol = tol
        self.atol = atol

        self.scale = mutation
        if (not np.all(np.isfinite(mutation)) or
                np.any(np.array(mutation) >= 2) or
                np.any(np.array(mutation) < 0)):
            raise ValueError('The mutation constant must be in [0,2).')

        self.dither = None
        if hasattr(mutation, '__iter__') and len(mutation) > 1:
            self.dither = [mutation[0], mutation[1]]
            self.dither.sort()

        self.cross_over_probability = recombination
        self.func = func
        self.args = args

        self.limits = np.array(bounds, dtype='float').T
        if (np.size(self.limits, 0) != 2 or
                not np.all(np.isfinite(self.limits))):
            raise ValueError('bounds should be a sequence of (min, max) pairs.')

        if maxiter is None:
            maxiter = 1000
        self.maxiter = maxiter
        if maxfun is None:
            maxfun = np.inf
        self.maxfun = maxfun

        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])

        self.parameter_count = np.size(self.limits, 1)
        self.random_number_generator = check_random_state(seed)

        # population size
        self.num_population_members = max(5, popsize * self.parameter_count)
        self.population_shape = (self.num_population_members, self.parameter_count)

        self._nfev = 0
        if isinstance(init, str):
            if init == 'latinhypercube':
                self.init_population_lhs()
            elif init == 'random':
                self.init_population_random()
            else:
                raise ValueError(self.__init_error_msg)
        else:
            self.init_population_array(init)

        self.disp = disp

    def init_population_lhs(self):
        rng = self.random_number_generator
        segsize = 1.0 / self.num_population_members
        samples = segsize * rng.random_sample(self.population_shape)
        samples += np.linspace(0., 1., self.num_population_members, endpoint=False)[:, np.newaxis]
        self.population = np.zeros_like(samples)
        for j in range(self.parameter_count):
            order = rng.permutation(range(self.num_population_members))
            self.population[:, j] = samples[order, j]
        self.population_energies = (np.ones(self.num_population_members) * np.inf)
        self._nfev = 0

    def init_population_random(self):
        rng = self.random_number_generator
        self.population = rng.random_sample(self.population_shape)
        self.population_energies = (np.ones(self.num_population_members) * np.inf)
        self._nfev = 0

    def init_population_array(self, init):
        popn = np.asfarray(init)
        if (popn.shape[0] < 5 or popn.shape[1] != self.parameter_count or len(popn.shape) != 2):
            raise ValueError("Population shape must be (M, N) with M>4, N=#params")
        self.population = np.clip(self._unscale_parameters(popn), 0, 1)
        self.num_population_members = popn.shape[0]
        self.population_shape = (self.num_population_members, self.parameter_count)
        self.population_energies = (np.ones(self.num_population_members) * np.inf)
        self._nfev = 0

    @property
    def x(self):
        return self._scale_parameters(self.population[0])

    @property
    def convergence(self):
        return (np.std(self.population_energies) /
                (np.abs(np.mean(self.population_energies)) + _MACHEPS))

    def solve(self):
        nit, warning_flag = 0, False
        status_message = _status_message['success']
        if np.all(np.isinf(self.population_energies)):
            self._calculate_population_energies()

        for nit in range(1, self.maxiter + 1):
            try:
                next(self)
            except StopIteration:
                warning_flag = True
                status_message = _status_message['maxfev']
                break

            if self.disp:
                print("differential_evolution step %d: f(x)= %g" %
                      (nit, self.population_energies[0]))

            convergence = self.convergence
            if (self.callback and
                    self.callback(self._scale_parameters(self.population[0]),
                                  convergence=self.tol / convergence) is True):
                warning_flag = True
                status_message = 'callback function requested early stop'
                break

            intol = (np.std(self.population_energies) <=
                     self.atol + self.tol * np.abs(np.mean(self.population_energies)))
            if warning_flag or intol:
                break
        else:
            status_message = _status_message['maxiter']
            warning_flag = True

        DE_result = OptimizeResult(
            x=self.x,
            fun=self.population_energies[0],
            nfev=self._nfev,
            nit=nit,
            message=status_message,
            success=(warning_flag is not True)
        )

        if self.polish:
            result = minimize(self.func,
                              np.copy(DE_result.x),
                              method='L-BFGS-B',
                              bounds=self.limits.T,
                              args=self.args)
            self._nfev += result.nfev
            DE_result.nfev = self._nfev
            if result.fun < DE_result.fun:
                DE_result.fun = result.fun
                DE_result.x = result.x
                DE_result.jac = result.jac
                self.population_energies[0] = result.fun
                self.population[0] = self._unscale_parameters(result.x)

        return DE_result

    def _calculate_population_energies(self):
        itersize = max(0, min(len(self.population), self.maxfun - self._nfev + 1))
        candidates = self.population[:itersize]
        parameters = np.array([self._scale_parameters(c) for c in candidates])
        energies = self.func(parameters, *self.args)
        self.population_energies[:itersize] = energies
        self._nfev += itersize

        minval = np.argmin(self.population_energies)
        lowest_energy = self.population_energies[minval]
        self.population_energies[minval], self.population_energies[0] = (
            self.population_energies[0],
            lowest_energy
        )
        self.population[[0, minval], :] = self.population[[minval, 0], :]

    def __iter__(self):
        return self

    def __next__(self):
        if np.all(np.isinf(self.population_energies)):
            self._calculate_population_energies()

        if self.dither is not None:
            self.scale = (self.random_number_generator.rand() *
                          (self.dither[1] - self.dither[0]) + self.dither[0])

        itersize = max(0, min(self.num_population_members, self.maxfun - self._nfev + 1))

        trials = np.array([self._mutate(c) for c in range(itersize)])
        for trial in trials:
            self._ensure_constraint(trial)

        parameters = np.array([self._scale_parameters(trial) for trial in trials])
        energies = self.func(parameters, *self.args)
        self._nfev += itersize

        for candidate, (energy, trial) in enumerate(zip(energies, trials)):
            if energy < self.population_energies[candidate]:
                self.population[candidate] = trial
                self.population_energies[candidate] = energy
                if energy < self.population_energies[0]:
                    self.population_energies[0] = energy
                    self.population[0] = trial

        return self.x, self.population_energies[0]

    def next(self):
        # Python 2 fallback, can be removed in Py3-only code
        return self.__next__()

    def _scale_parameters(self, trial):
        return self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2

    def _unscale_parameters(self, parameters):
        return (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5

    def _ensure_constraint(self, trial):
        mask = (trial < 0) | (trial > 1)
        trial[mask] = self.random_number_generator.rand(np.count_nonzero(mask))

    def _mutate(self, candidate):
        rng = self.random_number_generator
        fill_point = rng.randint(0, self.parameter_count)
        if self.strategy in ['currenttobest1exp', 'currenttobest1bin']:
            bprime = self.mutation_func(candidate, self._select_samples(candidate, 5))
        else:
            bprime = self.mutation_func(self._select_samples(candidate, 5))

        if self.strategy in self._binomial:
            crossovers = rng.rand(self.parameter_count) < self.cross_over_probability
            crossovers[fill_point] = True
            trial = np.where(crossovers, bprime, self.population[candidate])
            return trial
        else:
            trial = np.copy(self.population[candidate])
            i = 0
            while i < self.parameter_count and rng.rand() < self.cross_over_probability:
                trial[fill_point] = bprime[fill_point]
                fill_point = (fill_point + 1) % self.parameter_count
                i += 1
            return trial

    def _select_samples(self, candidate, number_samples):
        idxs = list(range(self.num_population_members))
        idxs.remove(candidate)
        self.random_number_generator.shuffle(idxs)
        idxs = idxs[:number_samples]
        return idxs

    def _best1(self, samples):
        r0, r1 = samples[:2]
        return (self.population[0] +
                self.scale * (self.population[r0] - self.population[r1]))

    def _rand1(self, samples):
        r0, r1, r2 = samples[:3]
        return (self.population[r0] +
                self.scale * (self.population[r1] - self.population[r2]))

    def _randtobest1(self, samples):
        r0, r1, r2 = samples[:3]
        bprime = np.copy(self.population[r0])
        bprime += self.scale * (self.population[0] - bprime)
        bprime += self.scale * (self.population[r1] - self.population[r2])
        return bprime

    def _currenttobest1(self, candidate, samples):
        r0, r1 = samples[:2]
        bprime = (self.population[candidate] +
                  self.scale * (self.population[0] - self.population[candidate] +
                                self.population[r0] - self.population[r1]))
        return bprime

    def _best2(self, samples):
        r0, r1, r2, r3 = samples[:4]
        return (self.population[0] + self.scale *
                (self.population[r0] + self.population[r1] -
                 self.population[r2] - self.population[r3]))

    def _rand2(self, samples):
        r0, r1, r2, r3, r4 = samples[:5]
        return (self.population[r0] + self.scale *
                (self.population[r1] + self.population[r2] -
                 self.population[r3] - self.population[r4]))
