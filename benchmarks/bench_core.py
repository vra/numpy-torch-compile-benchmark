# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import numpy as np
import torch
from torch import compile as c


class Benchmark:
    pass


class TimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """

    def setup(self):
        self.d = {}
        self.x = np.random(300, 300)
        self.y = np.random(300, 300)

    def time_numpy_matmul(self):
        z = np.matmul(self.x, self.y)

    def time_otrch_matmul(self):
        func = torch.compile(np.matmul)
        z = func(self.x, self.y)


class Core(Benchmark):
    def setup(self):
        self.l100 = range(100)
        self.l50 = range(50)
        self.float_l1000 = [float(i) for i in range(1000)]
        self.float64_l1000 = [np.float64(i) for i in range(1000)]
        self.int_l1000 = list(range(1000))
        self.l = [np.arange(1000), np.arange(1000)]
        self.l_view = [memoryview(a) for a in self.l]
        self.l10x10 = np.ones((10, 10))
        self.float64_dtype = np.dtype(np.float64)

    def time_array_1(self):
        np.array(1)

    def time_array_1_th(self):
        torch.compile(np.array)(1)

    def time_array_empty(self):
        np.array([])

    def time_array_empty_th(self):
        torch.compile(np.array)([])

    def time_array_l1(self):
        np.array([1])

    def time_array_l1_th(self):
        torch.compile(np.array)([1])

    def time_array_l100(self):
        np.array(self.l100)

    def time_array_l100_th(self):
        torch.compile(np.array)(self.l100)

    def time_array_float_l1000(self):
        np.array(self.float_l1000)

    def time_array_float_l1000_th(self):
        torch.compile(np.array)(self.float_l1000)

    def time_array_float_l1000_dtype(self):
        np.array(self.float_l1000, dtype=self.float64_dtype)

    def time_array_float_l1000_dtype_th(self):
        torch.compile(np.array)(self.float_l1000, dtype=self.float64_dtype)

    def time_array_float64_l1000(self):
        np.array(self.float64_l1000)

    def time_array_float64_l1000_th(self):
        torch.compile(np.array)(self.float64_l1000)

    def time_array_int_l1000(self):
        np.array(self.int_l1000)

    def time_array_int_l1000_th(self):
        torch.compile(np.array)(self.int_l1000)

    def time_array_l(self):
        np.array(self.l)

    def time_array_l_th(self):
        torch.compile(np.array)(self.l)

    def time_array_l_view(self):
        np.array(self.l_view)

    def time_array_l_view_th(self):
        torch.compile(np.array)(self.l_view)

    def time_can_cast(self):
        np.can_cast(self.l10x10, self.float64_dtype)

    def time_can_cast_th(self):
        torch.compile(np.can_cast)(self.l10x10, self.float64_dtype)

    def time_can_cast_same_kind(self):
        np.can_cast(self.l10x10, self.float64_dtype, casting="same_kind")

    def time_can_cast_same_kind_th(self):
        torch.compile(np.can_cast)(self.l10x10, self.float64_dtype, casting="same_kind")

    def time_vstack_l(self):
        np.vstack(self.l)

    def time_vstack_l_th(self):
        c(np.vstack)(self.l)

    def time_hstack_l(self):
        np.hstack(self.l)

    def time_hstack_l_th(self):
        c(np.hstack)(self.l)

    def time_dstack_l(self):
        np.dstack(self.l)

    def time_dstack_l_th(self):
        c(np.dstack)(self.l)

    def time_arange_100(self):
        np.arange(100)

    def time_arange_100_th(self):
        c(np.arange)(100)

    def time_zeros_100(self):
        np.zeros(100)

    def time_zeros_100_th(self):
        c(np.zeros)(100)

    def time_ones_100(self):
        np.ones(100)

    def time_ones_100_th(self):
        c(np.ones)(100)

    def time_empty_100(self):
        np.empty(100)

    def time_empty_100_th(self):
        c(np.empty)(100)

    def time_empty_like(self):
        np.empty_like(self.l10x10)

    def time_empty_like_th(self):
        c(np.empty_like)(self.l10x10)

    def time_eye_100(self):
        np.eye(100)

    def time_eye_100_th(self):
        c(np.eye)(100)

    def time_identity_100(self):
        np.identity(100)

    def time_identity_100_th(self):
        c(np.identity)(100)

    def time_eye_3000(self):
        np.eye(3000)

    def time_eye_3000_th(self):
        c(np.eye)(3000)

    def time_identity_3000(self):
        np.identity(3000)

    def time_identity_3000_th(self):
        c(np.identity)(3000)

    def time_diag_l100(self):
        np.diag(self.l100)

    def time_diag_l100_th(self):
        c(np.diag)(self.l100)

    def time_diagflat_l100(self):
        np.diagflat(self.l100)

    def time_diagflat_l100_th(self):
        c(np.diagflat)(self.l100)

    def time_diagflat_l50_l50(self):
        np.diagflat([self.l50, self.l50])

    def time_diagflat_l50_l50_th(self):
        c(np.diagflat)([self.l50, self.l50])

    def time_triu_l10x10(self):
        np.triu(self.l10x10)

    def time_triu_l10x10_th(self):
        c(np.triu)(self.l10x10)

    def time_tril_l10x10(self):
        np.tril(self.l10x10)

    def time_tril_l10x10_th(self):
        c(np.tril)(self.l10x10)

    def time_triu_indices_500(self):
        np.triu_indices(500)

    def time_triu_indices_500_th(self):
        c(np.triu_indices)(500)

    def time_tril_indices_500(self):
        np.tril_indices(500)

    def time_tril_indices_500_th(self):
        c(np.tril_indices)(500)




class CorrConv(Benchmark):
    params = [
        [50, 1000, int(1e5)],
        [10, 100, 1000, int(1e4)],
        ["valid", "same", "full"],
    ]
    param_names = ["size1", "size2", "mode"]

    def setup(self, size1, size2, mode):
        self.x1 = np.linspace(0, 1, num=size1)
        self.x2 = np.cos(np.linspace(0, 2 * np.pi, num=size2))

    def time_correlate(self, size1, size2, mode):
        np.correlate(self.x1, self.x2, mode=mode)

    def time_correlate_th(self, size1, size2, mode):
        c(np.correlate)(self.x1, self.x2, mode=mode)

    def time_convolve(self, size1, size2, mode):
        np.convolve(self.x1, self.x2, mode=mode)

    def time_convolve_th(self, size1, size2, mode):
        c(np.convolve)(self.x1, self.x2, mode=mode)


class CountNonzero(Benchmark):
    param_names = ["numaxes", "size", "dtype"]
    params = [
        [1, 2, 3],
        [100, 10000, 1000000],
        [bool, np.int8, np.int16, np.int32, np.int64, str, object],
    ]

    def setup(self, numaxes, size, dtype):
        self.x = np.arange(numaxes * size).reshape(numaxes, size)
        self.x = (self.x % 3).astype(dtype)

    def time_count_nonzero(self, numaxes, size, dtype):
        np.count_nonzero(self.x)

    def time_count_nonzero_th(self, numaxes, size, dtype):
        c(np.count_nonzero)(self.x)

    def time_count_nonzero_axis(self, numaxes, size, dtype):
        np.count_nonzero(self.x, axis=self.x.ndim - 1)

    def time_count_nonzero_axis_th(self, numaxes, size, dtype):
        c(np.count_nonzero)(self.x, axis=self.x.ndim - 1)


class PackBits(Benchmark):
    param_names = ["dtype"]
    params = [[bool, np.uintp]]

    def setup(self, dtype):
        self.d = np.ones(10000, dtype=dtype)
        self.d2 = np.ones((200, 1000), dtype=dtype)

    def time_packbits(self, dtype):
        np.packbits(self.d)

    def time_packbits_th(self, dtype):
        c(np.packbits)(self.d)

    def time_packbits_little(self, dtype):
        np.packbits(self.d, bitorder="little")

    def time_packbits_little_th(self, dtype):
        c(np.packbits)(self.d, bitorder="little")

    def time_packbits_axis0(self, dtype):
        np.packbits(self.d2, axis=0)

    def time_packbits_axis0_th(self, dtype):
        c(np.packbits)(self.d2, axis=0)

    def time_packbits_axis1(self, dtype):
        np.packbits(self.d2, axis=1)

    def time_packbits_axis1_th(self, dtype):
        c(np.packbits)(self.d2, axis=1)


class UnpackBits(Benchmark):
    def setup(self):
        self.d = np.ones(10000, dtype=np.uint8)
        self.d2 = np.ones((200, 1000), dtype=np.uint8)

    def time_unpackbits(self):
        np.unpackbits(self.d)

    def time_unpackbits_th(self):
        c(np.unpackbits)(self.d)

    def time_unpackbits_little(self):
        np.unpackbits(self.d, bitorder="little")

    def time_unpackbits_little_th(self):
        c(np.unpackbits)(self.d, bitorder="little")

    def time_unpackbits_axis0(self):
        np.unpackbits(self.d2, axis=0)

    def time_unpackbits_axis0_th(self):
        c(np.unpackbits)(self.d2, axis=0)

    def time_unpackbits_axis1(self):
        np.unpackbits(self.d2, axis=1)

    def time_unpackbits_axis1_th(self):
        c(np.unpackbits)(self.d2, axis=1)

    def time_unpackbits_axis1_little(self):
        np.unpackbits(self.d2, bitorder="little", axis=1)

    def time_unpackbits_axis1_little_th(self):
        c(np.unpackbits)(self.d2, bitorder="little", axis=1)


class Indices(Benchmark):
    def time_indices(self):
        np.indices((1000, 500))

    def time_indices_th(self):
        c(np.indices)((1000, 500))

