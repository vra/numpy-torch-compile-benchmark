from .common import Benchmark, TYPES1

import numpy as np
from torch import compile as c


class PutMask(Benchmark):
    params = [
        [True, False],
        TYPES1 + ["O", "i,O"]]
    param_names = ["values_is_scalar", "dtype"]

    def setup(self, values_is_scalar, dtype):
        if values_is_scalar:
            self.vals = np.array(1., dtype=dtype)
        else:
            self.vals = np.ones(1000, dtype=dtype)

        self.arr = np.ones(1000, dtype=dtype)

        self.dense_mask = np.ones(1000, dtype="bool")
        self.sparse_mask = np.zeros(1000, dtype="bool")

    def time_dense(self, values_is_scalar, dtype):
        np.putmask(self.arr, self.dense_mask, self.vals)

    def time_dense_th(self, values_is_scalar, dtype):
        c(np.putmask)(self.arr, self.dense_mask, self.vals)

    def time_sparse(self, values_is_scalar, dtype):
        np.putmask(self.arr, self.sparse_mask, self.vals)

    def time_sparse_th(self, values_is_scalar, dtype):
        c(np.putmask)(self.arr, self.sparse_mask, self.vals)


class Put(Benchmark):
    params = [
        [True, False],
        TYPES1 + ["O", "i,O"]]
    param_names = ["values_is_scalar", "dtype"]

    def setup(self, values_is_scalar, dtype):
        if values_is_scalar:
            self.vals = np.array(1., dtype=dtype)
        else:
            self.vals = np.ones(1000, dtype=dtype)

        self.arr = np.ones(1000, dtype=dtype)
        self.indx = np.arange(1000, dtype=np.intp)

    def time_ordered(self, values_is_scalar, dtype):
        np.put(self.arr, self.indx, self.vals)

    def time_ordered_th(self, values_is_scalar, dtype):
        c(np.put)(self.arr, self.indx, self.vals)
