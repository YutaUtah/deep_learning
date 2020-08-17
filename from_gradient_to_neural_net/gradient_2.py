import numpy as np


def _numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 値を元に戻す

    return grad

def function_2(x):
    return x[0]**2 + x[1]**2

array = _numerical_gradient(function_2, np.array([3.0, 4.0]))

def gradient_descent(f, init_x, lr = 0.01, step_num=100):
    x = init_x

    # x is array with two elements
    # grad is also array with two elements
    for i in range(step_num):
        grad = _numerical_gradient(function_2, x)
        x -= lr * grad

    return x

array = _numerical_gradient(function_2, np.array([3.0, 4.0]))

