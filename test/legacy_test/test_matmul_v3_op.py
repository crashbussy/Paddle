import paddle
import numpy as np

# 所有测试用例
test_cases = [
    # 格式: (x_shape, y_shape, expected_out_shape, dtype)
    ((0, 100, 1), (0, 1, 40), (0, 100, 40), "float64"),
    ((0, 100, 1), (0, 1, 4), (0, 100, 4), "float64"),
    ((0, 100, 1), (1, 1, 40), (0, 100, 40), "float64"),
    ((0, 100, 1), (1, 1, 4), (0, 100, 4), "float64"),
    ((0, 12, 197, 197), (0, 12, 197, 64), (0, 12, 197, 64), "float16"),
    ((0, 12, 197, 197), (0, 12, 197, 64), (0, 12, 197, 64), "float32"),
    ((1, 0, 1), (1, 1, 40), (1, 0, 40), "float64"),
    ((1, 0, 1), (1, 1, 4), (1, 0, 4), "float64"),
    ((1, 100, 1), (0, 1, 40), (0, 100, 40), "float64"),
    ((1, 100, 1), (0, 1, 4), (0, 100, 4), "float64"),
    ((1, 100, 1), (1, 1, 0), (1, 100, 0), "float64"),
    ((112, 0, 197, 197), (112, 0, 197, 64), (112, 0, 197, 64), "float16"),
    ((112, 0, 197, 197), (112, 0, 197, 64), (112, 0, 197, 64), "float32"),
    ((112, 12, 0, 197), (112, 12, 197, 64), (112, 12, 0, 64), "float16"),
    ((112, 12, 0, 197), (112, 12, 197, 64), (112, 12, 0, 64), "float32"),
    ((112, 12, 197, 197), (112, 12, 197, 0), (112, 12, 197, 0), "float16"),
    ((112, 12, 197, 197), (112, 12, 197, 0), (112, 12, 197, 0), "float32"),
]

def run_all_matmul_tests():
    for idx, (x_shape, y_shape, expected_shape, dtype) in enumerate(test_cases):
        print(f"\nTest {idx+1} begin: paddle.Tensor.matmul(Tensor({x_shape}), Tensor({y_shape}), dtype={dtype})")
        try:
            x = paddle.zeros(x_shape, dtype=dtype)
            y = paddle.zeros(y_shape, dtype=dtype)
            result = x.matmul(y)

            if result.shape != expected_shape:
                raise AssertionError(
                    f"[accuracy error] paddle.Tensor.matmul(Tensor({x_shape}), Tensor({y_shape}), dtype={dtype})\n\n"
                    f"Not equal to tolerance rtol=0.01, atol=0.01\n\n"
                    f"(shapes {result.shape}, {expected_shape} mismatch)\n"
                    f" x: array([], shape={x.shape}, dtype={dtype})\n"
                    f" y: array([], shape={y.shape}, dtype={dtype})"
                )
            else:
                print(f"[PASS] Shape is correct: {result.shape}")
        except Exception as e:
            print(f"{e}")

if __name__ == "__main__":
    run_all_matmul_tests()
