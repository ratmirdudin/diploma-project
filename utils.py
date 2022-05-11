import numpy as np


def isDoublesEqual(number_one, number_two):
    return abs(number_one - number_two) < 10 ** (-13)


def isDoubleArraysEqual(c, d):
    return isDoublesEqual(c[0], d[0]) and isDoublesEqual(c[1], d[1])


def sf_polygon(psi, v):
    N = len(v)
    maxx = np.array(v[0]).dot(psi)
    for i in range(1, N):
        tmp = np.array(v[i]).dot(psi)
        if tmp > maxx:
            maxx = tmp
    return maxx


def u_opt_polygon(psi, v):
    N = len(v)
    maxx = np.array(v[0]).dot(psi)
    ui = v[0]
    for i in range(1, N):
        tmp = np.array(v[i]).dot(psi)
        if tmp > maxx:
            maxx = tmp
            ui = v[i]
    return ui


def u_opt_quad(psi, b):
    print(psi)
    ui = np.zeros(2)
    if psi[0] > 0:
        ui[0] = b[0]
    elif psi[0] < 0:
        ui[0] = (-b[0])
    else:
        ui[0] = 0

    if psi[1] > 0:
        ui[1] = b[1]
    elif psi[1] < 0:
        ui[1] = (-b[1])
    else:
        ui[1] = 0
    return ui
