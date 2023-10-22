from .common import Benchmark

import numpy as np
from torch import compile as c


class Block(Benchmark):
    params = [1, 10, 100]
    param_names = ['size']

    def setup(self, n):
        self.a_2d = np.ones((2 * n, 2 * n))
        self.b_1d = np.ones(2 * n)
        self.b_2d = 2 * self.a_2d

        self.a = np.ones(3 * n)
        self.b = np.ones(3 * n)

        self.one_2d = np.ones((1 * n, 3 * n))
        self.two_2d = np.ones((1 * n, 3 * n))
        self.three_2d = np.ones((1 * n, 6 * n))
        self.four_1d = np.ones(6 * n)
        self.five_0d = np.ones(1 * n)
        self.six_1d = np.ones(5 * n)
        # avoid np.zeros's lazy allocation that might cause
        # page faults during benchmark
        self.zero_2d = np.full((2 * n, 6 * n), 0)

        self.one = np.ones(3 * n)
        self.two = 2 * np.ones((3, 3 * n))
        self.three = 3 * np.ones(3 * n)
        self.four = 4 * np.ones(3 * n)
        self.five = 5 * np.ones(1 * n)
        self.six = 6 * np.ones(5 * n)
        # avoid np.zeros's lazy allocation that might cause
        # page faults during benchmark
        self.zero = np.full((2 * n, 6 * n), 0)

    def time_block_simple_row_wise(self, n):
        np.block([self.a_2d, self.b_2d])

    def time_block_simple_row_wise_th(self, n):
        c(np.block)([self.a_2d, self.b_2d])

    def time_block_simple_column_wise(self, n):
        np.block([[self.a_2d], [self.b_2d]])

    def time_block_simple_column_wise_th(self, n):
        c(np.block)([[self.a_2d], [self.b_2d]])

    def time_block_complicated(self, n):
        np.block([[self.one_2d, self.two_2d],
                  [self.three_2d],
                  [self.four_1d],
                  [self.five_0d, self.six_1d],
                  [self.zero_2d]])

    def time_block_complicated_th(self, n):
        c(np.block)([[self.one_2d, self.two_2d],
                  [self.three_2d],
                  [self.four_1d],
                  [self.five_0d, self.six_1d],
                  [self.zero_2d]])

    def time_nested(self, n):
        np.block([
            [
                np.block([
                   [self.one],
                   [self.three],
                   [self.four]
                ]),
                self.two
            ],
            [self.five, self.six],
            [self.zero]
        ])

    def time_nested_th(self, n):
        c(np.block)([
            [
                np.block([
                   [self.one],
                   [self.three],
                   [self.four]
                ]),
                self.two
            ],
            [self.five, self.six],
            [self.zero]
        ])

    def time_no_lists(self, n):
        np.block(1)
        np.block(np.eye(3 * n))

    def time_no_lists_th(self, n):
        c(np.block)(1)
        c(np.block)(np.eye(3 * n))


class Block2D(Benchmark):
    params = [[(16, 16), (64, 64), (256, 256), (1024, 1024)],
              ['uint8', 'uint16', 'uint32', 'uint64'],
              [(2, 2), (4, 4)]]
    param_names = ['shape', 'dtype', 'n_chunks']

    def setup(self, shape, dtype, n_chunks):

        self.block_list = [
             [np.full(shape=[s//n_chunk for s, n_chunk in zip(shape, n_chunks)],
                     fill_value=1, dtype=dtype) for _ in range(n_chunks[1])]
            for _ in range(n_chunks[0])
        ]

    def time_block2d(self, shape, dtype, n_chunks):
        np.block(self.block_list)

    def time_block2d_th(self, shape, dtype, n_chunks):
        c(np.block)(self.block_list)


class Kron(Benchmark):
    """Benchmarks for Kronecker product of two arrays"""

    def setup(self):
        self.large_arr = np.random.random((10,) * 4)
        self.large_mat = np.asmatrix(np.random.random((100, 100)))
        self.scalar = 7

    def time_arr_kron(self):
        np.kron(self.large_arr, self.large_arr)

    def time_arr_kron_th(self):
        c(np.kron)(self.large_arr, self.large_arr)

    def time_scalar_kron(self):
        np.kron(self.large_arr, self.scalar)

    def time_scalar_kron_th(self):
        c(np.kron)(self.large_arr, self.scalar)

    def time_mat_kron(self):
        np.kron(self.large_mat, self.large_mat)

    def time_mat_kron_th(self):
        c(np.kron)(self.large_mat, self.large_mat)
