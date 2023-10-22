"""Benchmarks for `numpy.lib`."""


from .common import Benchmark

import numpy as np
from torch import compile as c


class Pad(Benchmark):
    param_names = ["shape", "pad_width", "mode"]
    params = [
        # Shape of the input arrays
        [(2 ** 22,), (1024, 1024), (256, 128, 1),
         (4, 4, 4, 4), (1, 1, 1, 1, 1)],
        # Tested pad widths
        [1, 8, (0, 32)],
        # Tested modes: mean, median, minimum & maximum use the same code path
        #               reflect & symmetric share a lot of their code path
        ["constant", "edge", "linear_ramp", "mean", "reflect", "wrap"],
    ]

    def setup(self, shape, pad_width, mode):
        # Make sure to fill the array to make the OS page fault
        # in the setup phase and not the timed phase
        self.array = np.full(shape, fill_value=1, dtype=np.float64)

    def time_pad(self, shape, pad_width, mode):
        np.pad(self.array, pad_width, mode)

    def time_pad_th(self, shape, pad_width, mode):
        c(np.pad)(self.array, pad_width, mode)


class Nan(Benchmark):
    """Benchmarks for nan functions"""

    param_names = ["array_size", "percent_nans"]
    params = [
            # sizes of the 1D arrays
            [200, int(2e5)],
            # percent of np.nan in arrays
            [0, 0.1, 2., 50., 90.],
            ]

    def setup(self, array_size, percent_nans):
        np.random.seed(123)
        # produce a randomly shuffled array with the
        # approximate desired percentage np.nan content
        base_array = np.random.uniform(size=array_size)
        base_array[base_array < percent_nans / 100.] = np.nan
        self.arr = base_array

    def time_nanmin(self, array_size, percent_nans):
        np.nanmin(self.arr)

    def time_nanmin_th(self, array_size, percent_nans):
        c(np.nanmin)(self.arr)

    def time_nanmax(self, array_size, percent_nans):
        np.nanmax(self.arr)

    def time_nanmax_th(self, array_size, percent_nans):
        c(np.nanmax)(self.arr)

    def time_nanargmin(self, array_size, percent_nans):
        np.nanargmin(self.arr)

    def time_nanargmin_th(self, array_size, percent_nans):
        c(np.nanargmin)(self.arr)

    def time_nanargmax(self, array_size, percent_nans):
        np.nanargmax(self.arr)

    def time_nanargmax_th(self, array_size, percent_nans):
        c(np.nanargmax)(self.arr)

    def time_nansum(self, array_size, percent_nans):
        np.nansum(self.arr)

    def time_nansum_c(self, array_size, percent_nans):
        c(np.nansum)(self.arr)

    def time_nanprod(self, array_size, percent_nans):
        np.nanprod(self.arr)

    def time_nanprod_th(self, array_size, percent_nans):
        c(np.nanprod)(self.arr)

    def time_nancumsum(self, array_size, percent_nans):
        np.nancumsum(self.arr)

    def time_nancumsum_th(self, array_size, percent_nans):
        c(np.nancumsum)(self.arr)

    def time_nancumprod(self, array_size, percent_nans):
        np.nancumprod(self.arr)

    def time_nancumprod_th(self, array_size, percent_nans):
        c(np.nancumprod)(self.arr)

    def time_nanmean(self, array_size, percent_nans):
        np.nanmean(self.arr)

    def time_nanmean_th(self, array_size, percent_nans):
        c(np.nanmean)(self.arr)

    def time_nanvar(self, array_size, percent_nans):
        np.nanvar(self.arr)

    def time_nanvar_th(self, array_size, percent_nans):
        c(np.nanvar)(self.arr)

    def time_nanstd(self, array_size, percent_nans):
        np.nanstd(self.arr)

    def time_nanstd_th(self, array_size, percent_nans):
        c(np.nanstd)(self.arr)

    def time_nanmedian(self, array_size, percent_nans):
        np.nanmedian(self.arr)

    def time_nanmedian_th(self, array_size, percent_nans):
        c(np.nanmedian)(self.arr)

    def time_nanquantile(self, array_size, percent_nans):
        np.nanquantile(self.arr, q=0.2)

    def time_nanquantile_th(self, array_size, percent_nans):
        c(np.nanquantile)(self.arr, q=0.2)

    def time_nanpercentile(self, array_size, percent_nans):
        np.nanpercentile(self.arr, q=50)

    def time_nanpercentile_th(self, array_size, percent_nans):
        c(np.nanpercentile)(self.arr, q=50)


class Unique(Benchmark):
    """Benchmark for np.unique with np.nan values."""

    param_names = ["array_size", "percent_nans"]
    params = [
        # sizes of the 1D arrays
        [200, int(2e5)],
        # percent of np.nan in arrays
        [0, 0.1, 2., 50., 90.],
    ]

    def setup(self, array_size, percent_nans):
        np.random.seed(123)
        # produce a randomly shuffled array with the
        # approximate desired percentage np.nan content
        base_array = np.random.uniform(size=array_size)
        n_nan = int(percent_nans * array_size)
        nan_indices = np.random.choice(np.arange(array_size), size=n_nan)
        base_array[nan_indices] = np.nan
        self.arr = base_array

    def time_unique_values(self, array_size, percent_nans):
        np.unique(self.arr, return_index=False,
                  return_inverse=False, return_counts=False)

    def time_unique_values_th(self, array_size, percent_nans):
        c(np.unique)(self.arr, return_index=False,
                  return_inverse=False, return_counts=False)

    def time_unique_counts(self, array_size, percent_nans):
        np.unique(self.arr, return_index=False,
                  return_inverse=False, return_counts=True)

    def time_unique_counts_th(self, array_size, percent_nans):
        c(np.unique)(self.arr, return_index=False,
                  return_inverse=False, return_counts=True)

    def time_unique_inverse(self, array_size, percent_nans):
        np.unique(self.arr, return_index=False,
                  return_inverse=True, return_counts=False)

    def time_unique_inverse_th(self, array_size, percent_nans):
        c(np.unique)(self.arr, return_index=False,
                  return_inverse=True, return_counts=False)

    def time_unique_all(self, array_size, percent_nans):
        np.unique(self.arr, return_index=True,
                  return_inverse=True, return_counts=True)

    def time_unique_all_th(self, array_size, percent_nans):
        c(np.unique)(self.arr, return_index=True,
                  return_inverse=True, return_counts=True)


class Isin(Benchmark):
    """Benchmarks for `numpy.isin`."""

    param_names = ["size", "highest_element"]
    params = [
        [10, 100000, 3000000],
        [10, 10000, int(1e8)]
    ]

    def setup(self, size, highest_element):
        self.array = np.random.randint(
                low=0, high=highest_element, size=size)
        self.in_array = np.random.randint(
                low=0, high=highest_element, size=size)

    def time_isin(self, size, highest_element):
        np.isin(self.array, self.in_array)

    def time_isin_th(self, size, highest_element):
        c(np.isin)(self.array, self.in_array)
