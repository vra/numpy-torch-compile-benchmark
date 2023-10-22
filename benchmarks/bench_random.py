from .common import Benchmark

import numpy as np
from torch import compile as c

try:
    from numpy.random import Generator
except ImportError:
    pass


class Randint(Benchmark):

    def time_randint_fast(self):
        np.random.randint(0, 2**30, size=10**5)

    def time_randint_fast_th(self):
        c(np.random.randint)(0, 2**30, size=10**5)

    def time_randint_slow(self):
        np.random.randint(0, 2**30 + 1, size=10**5)

    def time_randint_slow_th(self):
        c(np.random.randint)(0, 2**30 + 1, size=10**5)


class Randint_dtype(Benchmark):
    high = {
        'bool': 1,
        'uint8': 2**7,
        'uint16': 2**15,
        'uint32': 2**31,
        'uint64': 2**63
        }

    param_names = ['dtype']
    params = ['bool', 'uint8', 'uint16', 'uint32', 'uint64']

    def setup(self, name):
        from numpy.lib import NumpyVersion
        if NumpyVersion(np.__version__) < '1.11.0.dev0':
            raise NotImplementedError

    def time_randint_fast(self, name):
        high = self.high[name]
        np.random.randint(0, high, size=10**5, dtype=name)

    def time_randint_fast_th(self, name):
        high = self.high[name]
        c(np.random.randint)(0, high, size=10**5, dtype=name)

    def time_randint_slow(self, name):
        high = self.high[name]
        np.random.randint(0, high + 1, size=10**5, dtype=name)

    def time_randint_slow_th(self, name):
        high = self.high[name]
        c(np.random.randint)(0, high + 1, size=10**5, dtype=name)


class Permutation(Benchmark):
    def setup(self):
        self.n = 10000
        self.a_1d = np.random.random(self.n)
        self.a_2d = np.random.random((self.n, 2))

    def time_permutation_1d(self):
        np.random.permutation(self.a_1d)

    def time_permutation_1d_th(self):
        c(np.random.permutation)(self.a_1d)

    def time_permutation_2d(self):
        np.random.permutation(self.a_2d)

    def time_permutation_2d_th(self):
        c(np.random.permutation)(self.a_2d)

    def time_permutation_int(self):
        np.random.permutation(self.n)

    def time_permutation_int_th(self):
        c(np.random.permutation)(self.n)
