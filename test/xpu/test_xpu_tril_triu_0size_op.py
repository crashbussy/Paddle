# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle.base import core

paddle.enable_static()


class XPUTestTrilTriu_ZeroSize(OpTest):
    def setUp(self):
        self.init_config()
        self.__class__.use_xpu = True
        self.place = paddle.XPUPlace(0)

        if self.get_size(self.shape) == 0:
            self.x = np.empty(self.shape, dtype=self.dtype)
        else:
            self.x = np.random.random(self.shape).astype(self.dtype)

        self.inputs = {'X': self.x}
        self.attrs = {
            'diagonal': self.diagonal,
            'lower': self.lower,
        }

        if self.lower:
            self.out = np.tril(self.x, self.diagonal)
        else:
            self.out = np.triu(self.x, self.diagonal)

    def get_size(self, shape):
        size = 1
        for dim in shape:
            size *= dim
        return size

    def init_config(self):
        self.dtype = np.float32
        self.shape = (0, 5)
        self.diagonal = 0
        self.lower = False  # True 表示 tril，False 表示 triu

    def test_check_output(self):
        self.check_output_with_place(self.place, check_pir=True)

    def test_check_grad(self):
        if self.get_size(self.shape) == 0:
            return
        self.check_grad_with_place(self.place, ['X'], 'Out', check_pir=True)


class XPUTestTril_ZeroSize1(XPUTestTrilTriu_ZeroSize):
    def init_config(self):
        self.dtype = np.float32
        self.shape = (0, 5)
        self.diagonal = 0
        self.lower = True


class XPUTestTril_ZeroSize2(XPUTestTrilTriu_ZeroSize):
    def init_config(self):
        self.dtype = np.float32
        self.shape = (5, 0)
        self.diagonal = -1
        self.lower = True


class XPUTestTriu_ZeroSize1(XPUTestTrilTriu_ZeroSize):
    def init_config(self):
        self.dtype = np.float32
        self.shape = (2, 0, 3)
        self.diagonal = 2
        self.lower = False


class XPUTestTriu_ZeroSize2(XPUTestTrilTriu_ZeroSize):
    def init_config(self):
        self.dtype = np.float32
        self.shape = (1, 0, 0, 4)
        self.diagonal = -2
        self.lower = False


class XPUTestZeroAllDims(XPUTestTrilTriu_ZeroSize):
    def init_config(self):
        self.dtype = np.float32
        self.shape = (0, 0, 0)
        self.diagonal = 1
        self.lower = np.random.choice([True, False])



if __name__ == '__main__':
    unittest.main()
