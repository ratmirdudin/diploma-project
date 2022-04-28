import sys
import time

import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal, QThread, Qt
from PyQt5.QtGui import QCursor, QDoubleValidator, QIntValidator
from numpy import double
from scipy import linalg

import test
from my_calculations import covered_reachability_set, minimize_functional, optimal_control, euler_solve


class ThreadClass(QThread):
    update_pbar_signal = pyqtSignal(int)
    get_data_signal = pyqtSignal(object)
    finishing = pyqtSignal()

    def __init__(self, target, *args, parent=None):
        super(ThreadClass, self).__init__(parent)
        self.__target = target
        self.__args = args

    def run(self):
        print('Starting thread...')
        self.__target(*self.__args, self.update_pbar_signal, self.get_data_signal)
        self.finishing.emit()

    def stop(self):
        print('Stopping thread...')
        self.terminate()


class App(QtWidgets.QMainWindow, test.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.label_q1y.setVisible(False)
        self.lineEdit_q1y.setVisible(False)

        self.label_q2y.setVisible(False)
        self.lineEdit_q2y.setVisible(False)

        self.btn_cancel_calculate_plot.setVisible(False)

        self.lineEdit_x1T.setReadOnly(True)
        self.lineEdit_x2T.setReadOnly(True)

        self.lineEdit_q1y.setReadOnly(True)
        self.lineEdit_q2y.setReadOnly(True)
        self.lineEdit_jmin.setReadOnly(True)

        self.thread = {}

        intValidator = QIntValidator()
        intValidator.setLocale(QtCore.QLocale("en_US"))
        self.lineEdit_N.setValidator(intValidator)
        self.lineEdit_K.setValidator(intValidator)
        self.lineEdit_M.setValidator(intValidator)

        doubleValidator = QDoubleValidator()
        doubleValidator.setLocale(QtCore.QLocale("en_US"))

        self.lineEdit_A00.setValidator(doubleValidator)
        self.lineEdit_A01.setValidator(doubleValidator)
        self.lineEdit_A10.setValidator(doubleValidator)
        self.lineEdit_A11.setValidator(doubleValidator)

        self.lineEdit_x10.setValidator(doubleValidator)
        self.lineEdit_x20.setValidator(doubleValidator)

        self.lineEdit_T.setValidator(doubleValidator)

        self.lineEdit_c1.setValidator(doubleValidator)
        self.lineEdit_c2.setValidator(doubleValidator)
        self.lineEdit_y1.setValidator(doubleValidator)
        self.lineEdit_y2.setValidator(doubleValidator)
        self.lineEdit_r.setValidator(doubleValidator)
        self.lineEdit_a.setValidator(doubleValidator)
        self.lineEdit_b.setValidator(doubleValidator)

        self.btn_cancel_calculate_plot.clicked.connect(self.cancel_calculations)
        self.btn_calculate_plot.clicked.connect(self.calculate_graph)

    def cancel_calculations(self):
        self.thread[1].stop()
        self.btn_cancel_calculate_plot.setVisible(False)
        self.btn_calculate_plot.setVisible(True)
        self.update_progressbar(0)
        self.setCursor(QCursor(Qt.ArrowCursor))

    def calculate_graph(self):
        x0 = np.array([double(self.lineEdit_x10.text()), double(self.lineEdit_x20.text())])
        A = np.array([[double(self.lineEdit_A00.text()), double(self.lineEdit_A01.text())],
                      [double(self.lineEdit_A10.text()), double(self.lineEdit_A11.text())]])

        J = None
        c = None
        y_bar = None
        q_bar = None
        match self.tabWidget_functional.currentIndex():
            case 0:
                c = np.array([double(self.lineEdit_c1.text()), double(self.lineEdit_c2.text())])
                J = lambda x_T: x_T.dot(c)
                q_bar = lambda tmp: -c
                print("Линейный")
            case 1:
                y_bar = np.array([double(self.lineEdit_y1.text()), double(self.lineEdit_y2.text())])
                J = lambda x_T: 1 / 2 * linalg.norm(x_T - y_bar) ** 2  # терминальный функционал
                q_bar = lambda x_T_opt: y_bar - x_T_opt
                print("Квадратичный")

        sf = None
        u_opt_func = None
        match self.tabWidget_control.currentIndex():
            case 0:
                r = double(self.lineEdit_r.text())
                sf = lambda psi: abs(r) * linalg.norm(psi)
                u_opt_func = lambda psi: psi / linalg.norm(psi)
                print("Круг/точка")
            case 1:
                a = double(self.lineEdit_a.text())
                b = double(self.lineEdit_b.text())
                sf = lambda psi: abs(a) * abs(psi[0]) + abs(b) * abs(psi[1])  # опорная функция для управления

                def tmp(psi):
                    ui = np.zeros(2)
                    if psi[0] > 0:
                        ui[0] = a
                    elif psi[0] < 0:
                        ui[0] = (-a)
                    else:
                        ui[0] = 0

                    if psi[1] > 0:
                        ui[1] = b
                    elif psi[1] < 0:
                        ui[1] = (-b)
                    else:
                        ui[1] = 0
                    return ui

                u_opt_func = lambda psi: tmp(psi)
                print("Прямоугольник/отрезок")

        if J is None or sf is None:
            print("Ошибка: J или sf не определена")
            return

        params = {
            "N": int(self.lineEdit_N.text()),
            "M": int(self.lineEdit_M.text()),
            "K": int(self.lineEdit_K.text()),
            "T": double(self.lineEdit_T.text()),
            "A": A,
            "x0": x0,
            "y_bar": y_bar,
            "c": c,
            "J": J,
            "sf": sf,
            "q_bar": q_bar,
            "u_opt_func": u_opt_func
        }

        self.thread[1] = ThreadClass(self.long_running_task, params, parent=None)
        self.thread[1].start()

        self.thread[1].update_pbar_signal.connect(self.update_progressbar)
        self.thread[1].get_data_signal.connect(self.getting_data)

        self.btn_calculate_plot.setVisible(False)
        self.btn_cancel_calculate_plot.setVisible(True)
        self.setCursor(QCursor(Qt.WaitCursor))

        self.thread[1].finishing.connect(lambda: self.setCursor(QCursor(Qt.ArrowCursor)))
        self.thread[1].finishing.connect(lambda: self.btn_cancel_calculate_plot.setVisible(False))
        self.thread[1].finishing.connect(lambda: self.btn_calculate_plot.setVisible(True))
        self.thread[1].finishing.connect(self.thread[1].stop)

    def getting_data(self, data):
        # {"minJ": minJ,
        #  "x_T_opt": x_T_opt,
        #  "q_bar": q_bar,
        #  "x_T": [plot_x, plot_y],
        #  "u_opt": u_opt,
        #  "x": [x1, x2],
        #  "y_bar": y_bar,
        #  "x0": x0,
        #  "N": N}

        x_T = data.get("x_T")
        x_T_opt = data.get("x_T_opt")
        x = data.get("x")
        x0 = data.get("x0")
        y_bar = data.get("y_bar")
        c = data.get("c")

        M = data.get("M")
        T = data.get("T")
        t_g = np.linspace(0, T, M)
        u_opt = data.get("u_opt")

        self.draw_canvas(x_T, x_T_opt, x, x0, y_bar, c, t_g, u_opt)

        convert = lambda t: str('{:.7f}'.format(t))

        q_bar = data.get("q_bar")
        if q_bar is not None:
            self.lineEdit_q1y.setText(convert(q_bar[0]))
            self.lineEdit_q2y.setText(convert(q_bar[1]))

        minJ = data.get("minJ")
        if minJ is not None:
            self.lineEdit_jmin.setText(convert(minJ))

        if x_T_opt is not None:
            self.lineEdit_x1T.setText(convert(x_T_opt[0]))
            self.lineEdit_x2T.setText(convert(x_T_opt[1]))

    def update_progressbar(self, cnt):
        self.pbar.setValue(cnt)

    def draw_canvas(self, x_T, x_T_opt, x, x0, y_bar, c, t_g, u_opt):
        main_MPL = self.MplWidget.canvas
        main_MPL.axes.clear()
        main_MPL.axes.set_xlabel("$x_{1}$", size=13)
        main_MPL.axes.set_ylabel("$x_{2}$", size=13)
        main_MPL.axes.axis('equal')
        main_MPL.axes.grid(color='grey',  # цвет линий
                           linewidth=0.5,  # толщина
                           linestyle='--')  # начертание

        if x0 is not None:
            main_MPL.axes.plot(x0[0], x0[1], 'o', label=r'$x_{0}$')
            # main_MPL.axes.text(x0[0], x0[1] - 1, r'$x_{0}$', size=14)

        if y_bar is not None:
            main_MPL.axes.plot(y_bar[0], y_bar[1], 'o', label=r'$\bar{y}$')
            # main_MPL.axes.text(y_bar[0], y_bar[1] - 1.3, r'$\bar{y}$', size=14)
        elif c is not None:
            main_MPL.axes.plot(c[0], c[1], 'o', label=r'$c$')
            # main_MPL.axes.text(c[0], c[1] - 1.3, r'$c$', size=14)

        if x_T is not None:
            main_MPL.axes.plot(x_T[0], x_T[1], label='x(T)')

        if x_T_opt is not None:
            main_MPL.axes.plot(x_T_opt[0], x_T_opt[1], 'ok', label=r'$x_{*}(T)$')
            # main_MPL.axes.text(x_T_opt[0] - 1.4, x_T_opt[1] - 1.6, r'$x_{*}(T)$', size=14)

        if x is not None:
            main_MPL.axes.plot(x[0], x[1], label=r'$x$')

        u1_MPL = self.MplWidget_u1.canvas
        u1_MPL.axes.clear()
        u2_MPL = self.MplWidget_u2.canvas
        u2_MPL.axes.clear()
        if u_opt is not None:
            u1_MPL.axes.set_xlabel("$t$", size=13)
            u1_MPL.axes.set_ylabel("$u_{1}$", size=13)
            u1_MPL.axes.plot(t_g, u_opt[0], label=r'$u_{1}(t)$')
            u1_MPL.axes.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.1, 1.15), shadow=True)

            u2_MPL.axes.set_xlabel("$t$", size=13)
            u2_MPL.axes.set_ylabel("$u_{2}$", size=13)
            u2_MPL.axes.plot(t_g, u_opt[1], label=r'$u_{2}(t)$')
            u2_MPL.axes.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.1, 1.15), shadow=True)

        # pos = main_MPL.axes.get_position()
        # main_MPL.axes.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
        main_MPL.axes.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.1, 1.15), shadow=True)
        main_MPL.draw()

        u1_MPL.draw()
        u2_MPL.draw()

    @staticmethod
    def long_running_task(params, update_pbar_signal, get_data_signal):
        print("long running task ran")
        N = params.get("N")  # Количество вершин для множества достижимости
        K = params.get("K")  # Разбиение для метода трапеций
        M = params.get("M")  # Численное решение задачи коши
        t0 = 0
        T = params.get("T")
        x0 = params.get("x0")
        A = params.get("A")
        J = params.get("J")
        sf = params.get("sf")

        c = params.get("c")
        y_bar = params.get("y_bar")
        q_bar_func = params.get("q_bar")
        u_opt_func = params.get("u_opt_func")

        expm = lambda t: linalg.expm(t * A)

        A_transpose = np.transpose(A)
        expm_transpose = lambda t: linalg.expm(t * A_transpose)

        # ---------START---------
        start_time = time.time()
        x_T = covered_reachability_set(expm, expm_transpose, sf, t0, T, x0, N, K, update_pbar_signal)
        end_time1 = float('{:.3f}'.format(time.time() - start_time))

        start_time = time.time()
        minJ, x_T_opt = minimize_functional(J, x_T, N)
        q_bar = q_bar_func(x_T_opt)
        end_time2 = float('{:.3f}'.format(time.time() - start_time))

        start_time = time.time()
        u_opt = optimal_control(expm_transpose, u_opt_func, t0, T, q_bar, M, update_pbar_signal)
        end_time3 = float('{:.3f}'.format(time.time() - start_time))

        start_time = time.time()
        x1, x2 = euler_solve(t0, T, x0, A, u_opt, M)
        end_time4 = float('{:.3f}'.format(time.time() - start_time))

        end_time = float('{:.6f}'.format(end_time1 + end_time2 + end_time3 + end_time4))
        # -------------------------END-------------------------

        print("minJ = {}".format(minJ))
        print("x_T_opt = {}".format(x_T_opt))
        print("q_bar = {}".format(q_bar))
        print("1) Время = {} секунд".format(end_time1))
        print("2) Время = {} секунд".format(end_time2))
        print("3) Время = {} секунд".format(end_time3))
        print("4) Время = {} секунд".format(end_time4))
        print("Общее время = {} секунд".format(end_time))
        # print("Стартую тестовые вычисления")
        # start_time = time.time()
        # test_func(expm_transpose, t0, T, q_bar, x0, A, M)
        # print("Время тестовых вычислений = {:.3f} секунд".format(time.time() - start_time))
        print("\n\n")

        plot_x = np.zeros(N + 1)
        plot_y = np.zeros(N + 1)
        for i in range(N):
            plot_x[i], plot_y[i] = x_T[i][0], x_T[i][1]
        plot_x[N], plot_y[N] = x_T[0][0], x_T[0][1]

        plot_u1 = np.zeros(M)
        plot_u2 = np.zeros(M)
        for i in range(M):
            plot_u1[i], plot_u2[i] = u_opt[i][0], u_opt[i][1]

        get_data_signal.emit({"minJ": minJ,
                              "x_T_opt": x_T_opt,
                              "q_bar": q_bar,
                              "x_T": [plot_x, plot_y],
                              "u_opt": [plot_u1, plot_u2],
                              "x": [x1, x2],
                              "y_bar": y_bar,
                              "c": c,
                              "x0": x0,
                              "N": N,
                              "M": M,
                              "T": T
                              })


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
