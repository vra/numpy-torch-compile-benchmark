from .common import Benchmark, TYPES1, get_squares_

import numpy as np
from torch import compile as c



class MeshGrid(Benchmark):
    """ Benchmark meshgrid generation
    """
    params = [[16, 32],
              [2, 3, 4],
              ['ij', 'xy'], TYPES1]
    param_names = ['size', 'ndims', 'ind', 'ndtype']
    timeout = 10

    def setup(self, size, ndims, ind, ndtype):
        self.grid_dims = [(np.random.ranf(size)).astype(ndtype) for
                          x in range(ndims)]

    def time_meshgrid(self, size, ndims, ind, ndtype):
        np.meshgrid(*self.grid_dims, indexing=ind)

    def time_meshgrid_th(self, size, ndims, ind, ndtype):
        c(np.meshgrid)(*self.grid_dims, indexing=ind)


class Create(Benchmark):
    """ Benchmark for creation functions
    """
    params = [[16, 512, (32, 32)],
              TYPES1]
    param_names = ['shape', 'npdtypes']
    timeout = 10

    def setup(self, shape, npdtypes):
        values = get_squares_()
        self.xarg = values.get(npdtypes)[0]

    def time_full(self, shape, npdtypes):
        np.full(shape, self.xarg[1], dtype=npdtypes)

    def time_full_th(self, shape, npdtypes):
        c(np.full)(shape, self.xarg[1], dtype=npdtypes)

    def time_full_like(self, shape, npdtypes):
        np.full_like(self.xarg, self.xarg[0])

    def time_full_like_th(self, shape, npdtypes):
        c(np.full_like)(self.xarg, self.xarg[0])

    def time_ones(self, shape, npdtypes):
        np.ones(shape, dtype=npdtypes)

    def time_ones_th(self, shape, npdtypes):
        c(np.ones)(shape, dtype=npdtypes)

    def time_ones_like(self, shape, npdtypes):
        np.ones_like(self.xarg)

    def time_ones_like_th(self, shape, npdtypes):
        c(np.ones_like)(self.xarg)

    def time_zeros(self, shape, npdtypes):
        np.zeros(shape, dtype=npdtypes)

    def time_zeros_th(self, shape, npdtypes):
        c(np.zeros)(shape, dtype=npdtypes)

    def time_zeros_like(self, shape, npdtypes):
        np.zeros_like(self.xarg)

    def time_zeros_like_th(self, shape, npdtypes):
        c(np.zeros_like)(self.xarg)

    def time_empty(self, shape, npdtypes):
        np.empty(shape, dtype=npdtypes)

    def time_empty_th(self, shape, npdtypes):
        c(np.empty)(shape, dtype=npdtypes)

    def time_empty_like(self, shape, npdtypes):
        np.empty_like(self.xarg)

    def time_empty_like_th(self, shape, npdtypes):
        c(np.empty_like)(self.xarg)


class UfuncsFromDLP(Benchmark):
    """ Benchmark for creation functions
    """
    params = [[16, 32, (16, 16), (64, 64)],
              TYPES1]
    param_names = ['shape', 'npdtypes']
    timeout = 10

    def setup(self, shape, npdtypes):
        values = get_squares_()
        self.xarg = values.get(npdtypes)[0]

    def time_from_dlpack(self, shape, npdtypes):
        np.from_dlpack(self.xarg)

    def time_from_dlpack_th(self, shape, npdtypes):
        c(np.from_dlpack)(self.xarg)
