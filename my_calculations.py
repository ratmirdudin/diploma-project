import math

import numpy as np
import scipy.integrate as integrate
from scipy import linalg as linalg


def covered_reachability_set(expm, expm_transpose, sf, t0, T, x0, N, K, update_pbar_signal):
    progress = 0
    total_calculations = N + N + N * K + N
    calculate_percent = lambda current, total: math.floor(current * 85 / total)

    alpha = np.zeros(N)
    for i in range(N):
        alpha[i] = 2 * np.pi * i / N  # Тут вопрос, T-t0 или 2*Pi ????

        progress = progress + 1

    update_pbar_signal.emit(calculate_percent(progress, total_calculations))
    psi = np.zeros((N, 2))
    for i in range(N):
        psi[i] = np.array([np.cos(alpha[i]), np.sin(alpha[i])])

        progress = progress + 1

    update_pbar_signal.emit(calculate_percent(progress, total_calculations))

    x_g = np.linspace(t0, T, K)
    f_g = np.zeros(K)
    expmA_dot_x0 = expm(T - t0).dot(x0)
    c = np.zeros(N)
    for i in range(N):
        for j in range(K):
            f_g[j] = sf(expm_transpose(T - x_g[j]).dot(psi[i]))

            progress = progress + 1

        c[i] = expmA_dot_x0.dot(psi[i]) + integrate.trapz(f_g, x_g)

        update_pbar_signal.emit(calculate_percent(progress, total_calculations))

    x = np.zeros((N, 2))
    for i in range(-1, N - 1):
        a = np.array([[psi[i][0], psi[i][1]], [psi[i + 1][0], psi[i + 1][1]]])
        b = np.array([c[i], c[i + 1]])
        x[i] = linalg.solve(a, b)

        progress = progress + 1

    update_pbar_signal.emit(calculate_percent(progress, total_calculations))

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


def optimal_control(expm_transpose, u_opt_func, t0, T, q_bar, M, update_pbar_signal):
    progress = 0
    total_calculations = M
    calculate_percent = lambda current, total: 85 + math.floor(current * 15 / total)

    psi = np.zeros((M, 2))
    u = np.zeros((M, 2))
    t_g = np.linspace(t0, T, M)
    for i in range(M):
        tmp = expm_transpose(T - t_g[i])
        psi[i] = tmp.dot(q_bar)

        u[i] = u_opt_func(psi[i])

        progress = progress + 1

        update_pbar_signal.emit(calculate_percent(progress, total_calculations))

    return u


def euler_solve(t0, T, x0, A, u, M):
    h = (T - t0) / (M - 1)

    x1 = np.zeros(M)
    x2 = np.zeros(M)

    x1[0] = x0[0]
    x2[0] = x0[1]

    f1 = lambda x1, x2, u1: A[0][0] * x1 + A[0][1] * x2 + u1
    f2 = lambda x1, x2, u2: A[1][0] * x1 + A[1][1] * x2 + u2

    for i in range(M - 1):
        x1[i + 1] = x1[i] + h * f1(x1[i], x2[i], u[i][0])
        x2[i + 1] = x2[i] + h * f2(x1[i], x2[i], u[i][1])
    return x1, x2
