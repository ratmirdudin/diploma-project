import math

import numpy as np
import scipy.integrate as integrate
from scipy import linalg as linalg


def covered_reachability_set(expm, expm_transpose, sf_M0, sf_U, t0, T, N, K, M, update_pbar_signal):
    progress = 0
    total_calculations = N + N + N * K + N + M
    calculate_percent = lambda current: math.floor(current * 100 / total_calculations)

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
    # expmA = expm(T - t0).dot(np.array([1, -1]))
    expmTransA = expm_transpose(T - t0)
    c = np.zeros(N)

    expmTransposePreCalculated = []
    for j in range(K):
        expmTransposePreCalculated.append(expm_transpose(T - t_g[j]))

    for i in range(N):
        for j in range(K):
            f_g[j] = sf_U(expmTransposePreCalculated[j].dot(psi[i]))

            # transpose = expm_transpose(T - t_g[j])
            # f_g[j] = sf_U(transpose.dot(psi[i]))

            # exponTrans = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            # transpose = exponTrans(T - t_g[j])
            # f_g[j] = 1.1 * linalg.norm(transpose.dot(psi[i]))

            progress = progress + 1

        c[i] = sf_M0(expmTransA.dot(psi[i])) + integrate.trapz(f_g, t_g)

        update_pbar_signal.emit(calculate_percent(progress))

    x = np.zeros((N, 2))
    for i in range(-1, N - 1):
        a = np.array([[psi[i][0], psi[i][1]], [psi[i + 1][0], psi[i + 1][1]]])
        b = np.array([c[i], c[i + 1]])
        x[i] = linalg.solve(a, b)

        progress = progress + 1

    update_pbar_signal.emit(calculate_percent(progress))

    return x


def minimize_functional(J, x_T, N):
    minJ = J(x_T[0])
    x_T_opt = x_T[0]
    for i in range(1, N):
        tmp = J(x_T[i])
        if tmp < minJ:
            minJ = tmp
            x_T_opt = x_T[i]
    return minJ, x_T_opt


def optimal_control(expm_transpose, u_opt_func, t0, T, q_bar, N, K, M, update_pbar_signal):
    progress = 0
    total_calculations = N + N + N * K + N + M
    calculate_percent = lambda current: math.floor((N + N + N * K + N + current) * 100 / total_calculations)

    psi = np.zeros((M, 2))
    u = np.zeros((M, 2))
    t_g = np.linspace(t0, T, M)
    for i in range(M):
        tmp = expm_transpose(T - t_g[i])
        psi[i] = tmp.dot(q_bar)
        u[i] = u_opt_func(psi[i])

        progress = progress + 1

        update_pbar_signal.emit(calculate_percent(progress))

    return u


def euler_solve(t0, T, A, x_T_opt, u, M):
    h = (T - t0) / (M - 1)

    x1 = np.zeros(M)
    x2 = np.zeros(M)
    f1 = lambda x1, x2, u1: A[0][0] * x1 + A[0][1] * x2 + u1
    f2 = lambda x1, x2, u2: A[1][0] * x1 + A[1][1] * x2 + u2

    x1[M - 1] = x_T_opt[0]
    x2[M - 1] = x_T_opt[1]

    for i in range(M - 1, 0, -1):
        x1[i - 1] = x1[i] - h * f1(x1[i], x2[i], u[i][0])
        x2[i - 1] = x2[i] - h * f2(x1[i], x2[i], u[i][1])
    # x1[0] = 0
    # x2[0] = 0
    # for i in range(M - 1):
    #     x1[i + 1] = x1[i] + h * f1(x1[i], x2[i], u[i][0])
    #     x2[i + 1] = x2[i] + h * f2(x1[i], x2[i], u[i][1])
    return x1, x2
