import sys
import time

import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal, QThread, Qt
from PyQt5.QtGui import QCursor, QDoubleValidator, QIntValidator
from numpy import double
from scipy import linalg

import design_three
from my_calculations import covered_reachability_set, minimize_functional, optimal_control, euler_solve


class ThreadClass(QThread):
    update_pbar_signal = pyqtSignal(int)
    draw_plot_signal = pyqtSignal(object)
    finishing = pyqtSignal()

    def __init__(self, target, *args, parent=None):
        super(ThreadClass, self).__init__(parent)
        self.__target = target
        self.__args = args

    def run(self):
        print('Starting thread...')
        self.__target(*self.__args, self.update_pbar_signal, self.draw_plot_signal)
        self.finishing.emit()

    def stop(self):
        print('Stopping thread...')
        self.terminate()


class App(QtWidgets.QMainWindow, design_three.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

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
        self.lineEdit_y1.setValidator(doubleValidator)
        self.lineEdit_y2.setValidator(doubleValidator)

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
        y_bar = np.array([double(self.lineEdit_y1.text()), double(self.lineEdit_y2.text())])

        params = {
            "N": int(self.lineEdit_N.text()),
            "M": int(self.lineEdit_M.text()),
            "K": int(self.lineEdit_K.text()),
            "T": double(self.lineEdit_T.text()),
            "A": A,
            "x0": x0,
            "y_bar": y_bar
        }
        self.thread[1] = ThreadClass(self.long_running_task, params, parent=None)
        self.thread[1].start()

        self.thread[1].update_pbar_signal.connect(self.update_progressbar)
        self.thread[1].draw_plot_signal.connect(self.getting_data)

        self.btn_calculate_plot.setVisible(False)
        self.btn_cancel_calculate_plot.setVisible(True)
        self.setCursor(QCursor(Qt.WaitCursor))

        self.thread[1].finishing.connect(lambda: self.setCursor(QCursor(Qt.ArrowCursor)))
        self.thread[1].finishing.connect(lambda: self.btn_cancel_calculate_plot.setVisible(False))
        self.thread[1].finishing.connect(lambda: self.btn_calculate_plot.setVisible(True))
        self.thread[1].finishing.connect(self.thread[1].stop)

    def update_progressbar(self, cnt):
        self.pbar.setValue(cnt)

    def draw_canvas(self, x_T, x_T_opt, x, x0, y_bar):
        canvas = self.MplWidget.canvas
        canvas.axes.clear()
        canvas.axes.set_xlabel("$x_{1}$", size=13)
        canvas.axes.set_ylabel("$x_{2}$", size=13)
        canvas.axes.axis('equal')
        canvas.axes.grid(color='grey',  # цвет линий
                         linewidth=1,  # толщина
                         linestyle='--')  # начертание
        canvas.axes.plot(x_T[0], x_T[1])
        canvas.axes.plot(x[0], x[1])

        canvas.axes.plot(x0[0], x0[1], 'o')
        canvas.axes.text(x0[0], x0[1] - 1, r'$x_{0}$', size=14)

        canvas.axes.plot(x_T_opt[0], x_T_opt[1], 'ok')
        canvas.axes.text(x_T_opt[0] - 1.4, x_T_opt[1] - 1.6, r'$x_{*}(T)$', size=14)

        canvas.axes.plot(y_bar[0], y_bar[1], 'o')
        canvas.axes.text(y_bar[0], y_bar[1] - 1.3, r'$\bar{y}$', size=14)
        canvas.draw()

    def getting_data(self, data):
        # {"minJ": minJ,
        #  "x_T_opt": x_T_opt,
        #  "q_bar": q_bar,
        #  "x_T": [plot_x, plot_y],
        #  "u": u,
        #  "x": [x1, x2],
        #  "y_bar": y_bar,
        #  "x0": x0,
        #  "N": N}

        x_T = data.get("x_T")
        x_T_opt = data.get("x_T_opt")
        x = data.get("x")
        N = data.get("N")
        x0 = data.get("x0")
        y_bar = data.get("y_bar")

        self.draw_canvas(x_T, x_T_opt, x, x0, y_bar)

        convert = lambda t: str('{:.7f}'.format(t))

        q_bar = data.get("q_bar")

        self.lineEdit_q1y.setText(convert(q_bar[0]))
        self.lineEdit_q2y.setText(convert(q_bar[1]))

        minJ = data.get("minJ")
        self.lineEdit_jmin.setText(convert(minJ))

        self.lineEdit_x1T.setText(convert(x_T_opt[0]))
        self.lineEdit_x2T.setText(convert(x_T_opt[1]))

    @staticmethod
    def long_running_task(params, update_pbar_signal, get_data_signal):
        print("long running task ran")
        N = params.get("N")  # Количество вершин для множества достижимости
        K = params.get("K")  # Разбиение для метода трапеций
        M = params.get("M")  # численное решение задачи коши
        t0 = 0
        T = params.get("T")
        x0 = params.get("x0")
        A = params.get("A")

        sf = lambda u: abs(u[0]) + abs(u[1])  # опорная функция для управления
        y_bar = params.get("y_bar")
        J = lambda x_T: 1 / 2 * linalg.norm(x_T - y_bar) ** 2  # терминальный функционал

        expm = lambda t: linalg.expm(t * A)

        A_transpose = np.transpose(A)
        expm_transpose = lambda t: linalg.expm(t * A_transpose)

        # ---------START---------
        start_time = time.time()
        x_T = covered_reachability_set(expm, expm_transpose, sf, t0, T, x0, N, K, update_pbar_signal)
        end_time1 = float('{:.3f}'.format(time.time() - start_time))

        start_time = time.time()
        minJ, x_T_opt = minimize_functional(J, x_T, N)
        q_bar = y_bar - x_T_opt
        end_time2 = float('{:.3f}'.format(time.time() - start_time))

        start_time = time.time()
        u = optimal_control(expm_transpose, t0, T, q_bar, M, update_pbar_signal)
        end_time3 = float('{:.3f}'.format(time.time() - start_time))

        start_time = time.time()
        x1, x2 = euler_solve(t0, T, x0, A, u, M)
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
        print("\n\n")

        plot_x = np.zeros(N + 1)
        plot_y = np.zeros(N + 1)
        for i in range(N):
            plot_x[i], plot_y[i] = x_T[i][0], x_T[i][1]
        plot_x[N], plot_y[N] = x_T[0][0], x_T[0][1]

        get_data_signal.emit({"minJ": minJ,
                              "x_T_opt": x_T_opt,
                              "q_bar": q_bar,
                              "x_T": [plot_x, plot_y],
                              "u": u,
                              "x": [x1, x2],
                              "y_bar": y_bar,
                              "x0": x0,
                              "N": N
                              })


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
