import math

def dichotomy_method(a, b, eps, func):
    f_calls_number = 0
    N = 0
    while (b - a) > eps:

        x_mid = (a + b) / 2
        x1 = x_mid - eps/10
        x2 = x_mid + eps/10

        f_x1 = func(x1)
        f_x2 = func(x2)
        f_calls_number += 2

        if f_x1 < f_x2:
            b = x_mid
        else:
            a = x_mid
        N += 1

    min_x = (a + b) / 2
    min_value = func(min_x)
    return min_x





