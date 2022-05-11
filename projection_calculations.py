import numpy as np


def solvingLineEquation(a, b, c):
    if b[1] - a[1] == 0:
        xd = c[0]
        yd = a[1]
    elif b[0] - a[0] == 0:
        xd = a[0]
        yd = c[1]
    else:
        k = (b[1] - a[1]) / (b[0] - a[0])
        xd = (k ** 2 * a[0] + k * (c[1] - a[1]) + c[0]) / (k ** 2 + 1)
        yd = -(xd - c[0]) / k + c[1]
    return np.array([xd, yd])


def getPointOfProjectionOnSegment(a, b, c):
    d = solvingLineEquation(a, b, c)
    print("Проецируем:")
    print("a = ", a)
    print("b = ", b)
    if b[0] < a[0]:
        a, b = b, a
    if a[0] <= d[0] <= b[0]:
        print("попадание между")
        return d
    elif d[0] < a[0]:
        print("левее чем надо")
        return a
    elif b[0] < d[0]:
        print("правее чем надо")
        return b
