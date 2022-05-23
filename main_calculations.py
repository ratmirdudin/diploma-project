import math

import numpy as np
import scipy.integrate as integrate
from scipy import linalg as linalg

from projection_calculations import getPointOfProjectionOnSegment
from utils import isDoubleArraysEqual


def coveredReachabilitySet(expm_transpose, sf_M0, sf_U, t0, T, N, K, total_calcs_one, update_pbar_signal):
    progress = 0
    calculate_percent = lambda current: math.floor(current * 100 / total_calcs_one)

    alpha = np.zeros(N)
    for i in range(N):
        alpha[i] = 2 * np.pi * i / N

        progress = progress + 1

    update_pbar_signal.emit(calculate_percent(progress))
    psi = np.zeros((N, 2))
    for i in range(N):
        psi[i] = np.array([np.cos(alpha[i]), np.sin(alpha[i])])

        progress = progress + 1

    update_pbar_signal.emit(calculate_percent(progress))

    t_g = np.linspace(t0, T, K)
    f_g = np.zeros(K)
    expmTransA = expm_transpose(T - t0)
    c = np.zeros(N)

    expmTransposePreCalculated = []
    for j in range(K):
        expmTransposePreCalculated.append(expm_transpose(T - t_g[j]))

    for i in range(N):
        for j in range(K):
            f_g[j] = sf_U(expmTransposePreCalculated[j].dot(psi[i]))

            progress = progress + 1

        c[i] = sf_M0(expmTransA.dot(psi[i])) + integrate.trapz(f_g, t_g)

        update_pbar_signal.emit(calculate_percent(progress))

    x = np.zeros((N, 2))
    for i in range(-1, N - 1):
        a = np.array([[psi[i][0], psi[i][1]],
                      [psi[i + 1][0], psi[i + 1][1]]])
        b = np.array([c[i], c[i + 1]])
        x[i] = linalg.solve(a, b)

        progress = progress + 1

    update_pbar_signal.emit(calculate_percent(progress))

    return x


def unique(x):
    N = len(x)
    uniqueX = np.empty((0, 2), np.double)
    for i in range(N):
        flag = True
        for j in range(i + 1, N):
            if isDoubleArraysEqual(x[i], x[j]):
                flag = False

        if flag:
            uniqueX = np.append(uniqueX, np.array([x[i]]), axis=0)
    return uniqueX


def findOptimalCoordsOfM0(psi_t0, x0):
    maxx = psi_t0.dot(x0[0])
    x0_opt = x0[0]
    for i in range(1, len(x0)):
        tmp = psi_t0.dot(x0[i])
        if tmp > maxx:
            maxx = tmp
            x0_opt = x0[i]
    return x0_opt


def minimizeFunctional(J, x):
    N = len(x)
    minJ = J(x[0])
    x_opt = x[0]
    for i in range(1, N):
        tmp = J(x[i])
        if tmp < minJ:
            minJ = tmp
            x_opt = x[i]
    return minJ, x_opt


def minimizeFunctionalForLinearProblem(J, x):
    minJ, x_opt = minimizeFunctional(J, x)

    return minJ, x_opt


def minimizeQuadraticFunctional(J, x, y_bar):
    test_x = unique(x)
    minJ_one, x_opt_one = minimizeFunctional(J, test_x)

    x_opt_two = findSecondDifferentMinimumOfFunctional(J, test_x, minJ_one, x_opt_one)

    if x_opt_two[0] < x_opt_one[0]:
        x_opt_one, x_opt_two = x_opt_two, x_opt_one
    d = getPointOfProjectionOnSegment(x_opt_one, x_opt_two, y_bar)
    return J(d), np.array(d), x_opt_one, x_opt_two


def getTwoDifferentMinimumsInsteadOfFirstMinOfFunctional(J, x, minJ_one, x_opt_one):
    x_opt_two = findSecondDifferentMinimumOfFunctional(J, x, minJ_one, x_opt_one)

    if x_opt_two[0] < x_opt_one[0]:
        x_opt_one, x_opt_two = x_opt_two, x_opt_one
    return np.array([[x_opt_one[0], x_opt_two[0]], [x_opt_one[1], x_opt_two[1]]])


def findSecondDifferentMinimumOfFunctional(J, x, minJ_one, x_opt_one):
    N = len(x)
    minJ_two = minJ_one
    x_opt_two = x_opt_one
    flag = False
    index = 0
    for i in range(0, N):
        if not isDoubleArraysEqual(x[i], x_opt_one):
            minJ_two = J(x[i])
            x_opt_two = x[i]
            index = i
            flag = True
            break
    if flag:
        for i in range(index + 1, N):
            if not isDoubleArraysEqual(x[i], x_opt_one):
                tmp = J(x[i])
                if tmp < minJ_two:
                    minJ_two = tmp
                    x_opt_two = x[i]
    return x_opt_two


def optimalControl(expm_transpose, u_opt_func, t0, T, psi_T, N, K, M, update_pbar_signal):
    progress = 0
    total_calculations = N + N + N * K + N + M
    calculate_percent = lambda current: math.floor((N + N + N * K + N + current) * 100 / total_calculations)

    psi = np.zeros((M, 2))
    u = np.zeros((M, 2))
    t_g = np.linspace(t0, T, M)
    for i in range(M):
        tmp = expm_transpose(T - t_g[i])
        psi[i] = tmp.dot(psi_T)
        u[i] = u_opt_func(psi[i])

        progress = progress + 1

        update_pbar_signal.emit(calculate_percent(progress))

    return u, psi


def optimalControlForAZeroMatrix(expm_transpose, u_opt_func_A_zero, t0, T, psi_T, N, K, M, update_pbar_signal):
    progress = 0
    total_calculations = N + N + N * K + N + M
    calculate_percent = lambda current: math.floor((N + N + N * K + N + current) * 100 / total_calculations)

    psi = np.zeros((M, 2))
    u = np.zeros((M, 2))
    t_g = np.linspace(t0, T, M)
    for i in range(M):
        tmp = expm_transpose(T - t_g[i])
        psi[i] = tmp.dot(psi_T)

        u[i] = u_opt_func_A_zero(t_g[i])

        progress = progress + 1

        update_pbar_signal.emit(calculate_percent(progress))

    return u, psi


def directEulerSolve(f1, f2, t0, T, x0_opt, u_opt):
    M = len(u_opt)
    h = (T - t0) / (M - 1)

    x1 = np.zeros(M)
    x2 = np.zeros(M)
    x1[0] = x0_opt[0]
    x2[0] = x0_opt[1]
    for i in range(M - 1):
        x1[i + 1] = x1[i] + h * f1(x1[i], x2[i], u_opt[i][0])
        x2[i + 1] = x2[i] + h * f2(x1[i], x2[i], u_opt[i][1])
    return [x1, x2]


def backEulerSolve(f1, f2, t0, T, x_T_opt, u_opt):
    M = len(u_opt)
    h = (T - t0) / (M - 1)

    x1 = np.zeros(M)
    x2 = np.zeros(M)
    x1[M - 1] = x_T_opt[0]
    x2[M - 1] = x_T_opt[1]
    for i in range(M - 1, 0, -1):
        x1[i - 1] = x1[i] - h * f1(x1[i], x2[i], u_opt[i][0])
        x2[i - 1] = x2[i] - h * f2(x1[i], x2[i], u_opt[i][1])
    return [x1, x2]
