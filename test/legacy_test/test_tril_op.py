import os
import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle.base import core

class TestTrilOp(OpTest):
    def set_args(self):
        self.x_shape = (0, 3)
        self.diagonal = 0
        self.dtype = 'float32'

    def setUp(self):
        paddle.enable_static()
        self.set_args()
        self.op_type = "tril"
        self.python_api = paddle.tril

        x_np = np.random.random(self.x_shape).astype(self.dtype)
        self.inputs = {'X': x_np}
        self.attrs = {'diagonal': self.diagonal}

        out_np = np.tril(x_np, k=self.diagonal)
        self.outputs = {'Out': out_np}

    def test_check_output(self):
        self.check_output(check_pir=True)

# ------------------ 0-Size Tensor 测试 ------------------
class TestTril_ZeroShape_0x3(TestTrilOp):
    def set_args(self):
        self.x_shape = (0, 3)
        self.diagonal = 0
        self.dtype = 'float32'

class TestTril_ZeroShape_3x0(TestTrilOp):
    def set_args(self):
        self.x_shape = (3, 0)
        self.diagonal = 0
        self.dtype = 'float32'

class TestTril_ZeroShape_0x0(TestTrilOp):
    def set_args(self):
        self.x_shape = (0, 0)
        self.diagonal = 0
        self.dtype = 'float32'

# ------------------ 参数 diagonal 测试 ------------------
class TestTril_Diagonal_1(TestTrilOp):
    def set_args(self):
        self.x_shape = (0, 3)
        self.diagonal = 1
        self.dtype = 'float32'

class TestTril_Diagonal_N1(TestTrilOp):
    def set_args(self):
        self.x_shape = (3, 0)
        self.diagonal = -1
        self.dtype = 'float32'

# ------------------ 高维张量测试 ------------------
class TestTril_HighDim_2x0x4(TestTrilOp):
    def set_args(self):
        self.x_shape = (2, 0, 4)
        self.diagonal = 0
        self.dtype = 'float32'

class TestTril_HighDim_0x2x4(TestTrilOp):
    def set_args(self):
        self.x_shape = (0, 2, 4)
        self.diagonal = -1
        self.dtype = 'float32'

# ------------------ 异常测试 ------------------
class TestTril_Exception_1DInput(unittest.TestCase):
    def test_1d_input(self):
        x = paddle.rand((0,))
        with self.assertRaises(ValueError):
            paddle.tril(x)

class TestTril_Exception_DiagTypeError(unittest.TestCase):
    def test_diag_not_int(self):
        x = paddle.rand((0, 3))
        with self.assertRaises(TypeError):
            paddle.tril(x, diagonal='a')

# ------------------ 数据类型测试 ------------------
class TestTril_DType_Float16(TestTrilOp):
    def set_args(self):
        self.x_shape = (0, 3)
        self.diagonal = 0
        self.dtype = 'float16'

class TestTril_DType_Float64(TestTrilOp):
    def set_args(self):
        self.x_shape = (3, 0)
        self.diagonal = 1
        self.dtype = 'float64'

# ------------------ 设备兼容性测试 ------------------
class TestTril_Device_CpuGpu(unittest.TestCase):
    def test_cpu_gpu(self):
        places = []
        if paddle.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        places.append(paddle.CPUPlace())

        for place in places:
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.zeros([0, 3], dtype='float32', place=place)
                result = paddle.tril(x, diagonal=0)
                self.assertTrue(result.place == place)

# ------------------ 大规模张量测试 ------------------
class TestTril_LargeTensor(TestTrilOp):
    def set_args(self):
        self.x_shape = (1000, 1000)
        self.diagonal = 0
        self.dtype = 'float32'

if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
