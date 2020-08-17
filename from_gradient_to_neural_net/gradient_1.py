import numpy as np
import matplotlib.pyplot as plt


# getting derivative. arg: f is function x is variable
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / 2*h

# defining function f with respect to x variable f(x) = 0.01x^2 + 0.1x
def function_1(x):
    return 0.01*x**2 + 0.1*x


def tangent_line(f, x):
    d = numerical_diff(f, x)
    y = f(x) - d*x
    return lambda t: d*t + y



x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()
