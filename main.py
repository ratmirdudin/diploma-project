import sys
import time

import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal, QThread, Qt
from PyQt5.QtGui import QCursor, QDoubleValidator, QIntValidator
from numpy import double
from scipy import linalg

import design
import help
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


class App(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.tabWidget_initSet.setCurrentIndex(0)
        self.tabWidget_control.setCurrentIndex(0)
        self.tabWidget_functional.setCurrentIndex(0)
        self.tabWidget_plots.setCurrentIndex(0)

        text_solve_M0 = self.tabWidget_initSet.tabText(self.tabWidget_initSet.currentIndex())
        text_solve_U = self.tabWidget_control.tabText(self.tabWidget_control.currentIndex())
        text_solve_J = self.tabWidget_functional.tabText(self.tabWidget_functional.currentIndex())
        self.label_solve_M0.setText(text_solve_M0)
        self.label_solve_U.setText(text_solve_U)
        self.label_solve_J.setText(text_solve_J)

        # Временно выключим
        self.btn_help.setVisible(False)

        self.btn_cancel_calculate_plot.setVisible(False)

        self.lineEdit_x1T.setReadOnly(True)
        self.lineEdit_x2T.setReadOnly(True)
        self.lineEdit_jmin.setReadOnly(True)

        self.thread = {}

        # Валидаторы
        self.intValidator = QIntValidator()
        self.intValidator.setLocale(QtCore.QLocale("en_US"))

        self.doubleValidator = QDoubleValidator()
        self.doubleValidator.setLocale(QtCore.QLocale("en_US"))

        # Основные параметры
        self.lineEdit_N.setValidator(self.intValidator)
        self.lineEdit_K.setValidator(self.intValidator)
        self.lineEdit_M.setValidator(self.intValidator)
        self.lineEdit_N.textChanged[str].connect(lambda: self.on_changed(333, 333))
        self.lineEdit_K.textChanged[str].connect(lambda: self.on_changed(333, 333))
        self.lineEdit_M.textChanged[str].connect(lambda: self.on_changed(333, 333))
        self.lineEdit_A00.setValidator(self.doubleValidator)
        self.lineEdit_A01.setValidator(self.doubleValidator)
        self.lineEdit_A10.setValidator(self.doubleValidator)
        self.lineEdit_A11.setValidator(self.doubleValidator)
        self.lineEdit_A00.textChanged[str].connect(lambda: self.on_changed(333, 333))
        self.lineEdit_A01.textChanged[str].connect(lambda: self.on_changed(333, 333))
        self.lineEdit_A10.textChanged[str].connect(lambda: self.on_changed(333, 333))
        self.lineEdit_A11.textChanged[str].connect(lambda: self.on_changed(333, 333))
        self.lineEdit_t0.setValidator(self.doubleValidator)
        self.lineEdit_T.setValidator(self.doubleValidator)
        self.lineEdit_t0.textChanged[str].connect(lambda: self.on_changed(333, 333))
        self.lineEdit_T.textChanged[str].connect(lambda: self.on_changed(333, 333))

        # Параметры начального множества
        self.lineEdit_initSet_r.setValidator(self.doubleValidator)
        self.lineEdit_initSet_g1.setValidator(self.doubleValidator)
        self.lineEdit_initSet_g2.setValidator(self.doubleValidator)
        self.lineEdit_initSet_a1.setValidator(self.doubleValidator)
        self.lineEdit_initSet_a2.setValidator(self.doubleValidator)
        self.lineEdit_initSet_r.textChanged[str].connect(lambda: self.on_changed(333, 333))
        self.lineEdit_initSet_g1.textChanged[str].connect(lambda: self.on_changed(333, 333))
        self.lineEdit_initSet_g2.textChanged[str].connect(lambda: self.on_changed(333, 333))
        self.lineEdit_initSet_a1.textChanged[str].connect(lambda: self.on_changed(333, 333))
        self.lineEdit_initSet_a2.textChanged[str].connect(lambda: self.on_changed(333, 333))

        # Параметры функционала
        self.lineEdit_functional_c1.setValidator(self.doubleValidator)
        self.lineEdit_functional_c2.setValidator(self.doubleValidator)
        self.lineEdit_functional_y1.setValidator(self.doubleValidator)
        self.lineEdit_functional_y2.setValidator(self.doubleValidator)
        self.lineEdit_functional_c1.textChanged[str].connect(lambda: self.on_changed(333, 333))
        self.lineEdit_functional_c2.textChanged[str].connect(lambda: self.on_changed(333, 333))
        self.lineEdit_functional_y1.textChanged[str].connect(lambda: self.on_changed(333, 333))
        self.lineEdit_functional_y2.textChanged[str].connect(lambda: self.on_changed(333, 333))

        # Параметры управления
        self.lineEdit_control_r.setValidator(self.doubleValidator)
        self.lineEdit_control_b1.setValidator(self.doubleValidator)
        self.lineEdit_control_b2.setValidator(self.doubleValidator)
        self.lineEdit_control_r.textChanged[str].connect(lambda: self.on_changed(333, 333))
        self.lineEdit_control_b1.textChanged[str].connect(lambda: self.on_changed(333, 333))
        self.lineEdit_control_b2.textChanged[str].connect(lambda: self.on_changed(333, 333))

        self.tabWidget_initSet.tabBarClicked.connect(lambda currentTab: self.on_changed(0, currentTab))
        self.tabWidget_control.tabBarClicked.connect(lambda currentTab: self.on_changed(1, currentTab))
        self.tabWidget_functional.tabBarClicked.connect(lambda currentTab: self.on_changed(2, currentTab))

        self.btn_help.clicked.connect(self.open_help_dialog)
        self.btn_cancel_calculate_plot.clicked.connect(self.cancel_calculations)
        self.btn_calculate_plot.clicked.connect(self.calculate_graph)

        self.menu.addAction("Выход", self.action_clicked)
        self.menu_2.addAction("О программе", self.action_clicked)

    @QtCore.pyqtSlot()
    def action_clicked(self):
        action = self.sender()
        if action.text() == "Выход":
            self.close()
        elif action.text() == "О программе":
            self.open_help_dialog()

    def open_help_dialog(self):
        self.window = QtWidgets.QDialog()
        self.ui = help.Ui_Dialog()
        self.ui.setupUi(self.window)
        self.window.show()

    def on_changed(self, tab, current):
        match tab:
            case 0:
                self.tabWidget_initSet.setCurrentIndex(current)
                text_solve_M0 = self.tabWidget_initSet.tabText(current)
                self.label_solve_M0.setText(text_solve_M0)
            case 1:
                self.tabWidget_control.setCurrentIndex(current)
                text_solve_U = self.tabWidget_control.tabText(current)
                self.label_solve_U.setText(text_solve_U)
            case 2:
                self.tabWidget_functional.setCurrentIndex(current)
                text_solve_J = self.tabWidget_functional.tabText(current)
                self.label_solve_J.setText(text_solve_J)

        styles = "background-color: rgb(255, 0, 0);\ncolor: rgb(255, 255, 255);"
        flag = True
        text = [self.lineEdit_N.text(),
                self.lineEdit_K.text(),
                self.lineEdit_M.text(),
                self.lineEdit_A00.text(),
                self.lineEdit_A01.text(),
                self.lineEdit_A10.text(),
                self.lineEdit_A11.text(),
                self.lineEdit_t0.text(),
                self.lineEdit_T.text(),
                ]

        match self.tabWidget_initSet.currentIndex():
            case 0:
                text.append(self.lineEdit_initSet_r.text())
                text.append(self.lineEdit_initSet_g1.text())
                text.append(self.lineEdit_initSet_g2.text())
            case 1:
                text.append(self.lineEdit_initSet_a1.text())
                text.append(self.lineEdit_initSet_a2.text())

        match self.tabWidget_control.currentIndex():
            case 0:
                text.append(self.lineEdit_control_r.text())
            case 1:
                text.append(self.lineEdit_control_b1.text())
                text.append(self.lineEdit_control_b2.text())

        match self.tabWidget_functional.currentIndex():
            case 0:
                text.append(self.lineEdit_functional_c1.text())
                text.append(self.lineEdit_functional_c2.text())
            case 1:
                text.append(self.lineEdit_functional_y1.text())
                text.append(self.lineEdit_functional_y2.text())

        for txt in text:
            if not (txt and txt.strip()):
                styles = "background-color: rgb(160, 160, 160);\ncolor: rgb(0, 0, 0);"
                flag = False
                break

        self.btn_calculate_plot.setStyleSheet(styles)
        self.btn_calculate_plot.setEnabled(flag)

    def cancel_calculations(self):
        self.thread[1].stop()
        self.btn_cancel_calculate_plot.setVisible(False)
        self.btn_calculate_plot.setVisible(True)
        self.update_progressbar(0)
        self.setCursor(QCursor(Qt.ArrowCursor))

    def calculate_graph(self):
        # print("Начальное множество: {}".format(self.label_solve_M0.text()))
        # print("Управление: {}".format(self.label_solve_U.text()))
        # print("Функционал: {}".format(self.label_solve_J.text()))

        A = np.array([[double(self.lineEdit_A00.text()), double(self.lineEdit_A01.text())],
                      [double(self.lineEdit_A10.text()), double(self.lineEdit_A11.text())]])

        sf_M0 = None
        x0 = None
        match self.tabWidget_initSet.currentIndex():
            case 0:
                print("Начальное множество: Круг/точка")
                r1 = double(self.lineEdit_initSet_r.text())
                g1 = double(self.lineEdit_initSet_g1.text())
                g2 = double(self.lineEdit_initSet_g2.text())
                sf_M0 = lambda psi: g1 * psi[0] + g2 * psi[1] + abs(r1) * linalg.norm(psi)

                if r1 == 0:
                    x0 = [g1, g2]
                else:
                    phi = np.linspace(0, 2 * np.pi, 100)
                    x = g1 + r1 * np.cos(phi)
                    y = g2 + r1 * np.sin(phi)
                    x0 = [x, y]
            case 1:
                print("Начальное множество: Прямоугольник/отрезок")
                a1 = double(self.lineEdit_initSet_a1.text())
                a2 = double(self.lineEdit_initSet_a2.text())
                sf_M0 = lambda psi: abs(a1) * abs(psi[0]) + abs(a2) * abs(psi[1])
                x0 = [[-a1, a1, a1, -a1, -a1], [a2, a2, -a2, -a2, a2]]

        sf_U = None
        u_opt_func = None
        match self.tabWidget_control.currentIndex():
            case 0:
                print("Управление: Круг")
                r2 = double(self.lineEdit_control_r.text())
                sf_U = lambda psi: abs(r2) * linalg.norm(psi)
                u_opt_func = lambda psi: psi / (r2 * linalg.norm(psi))
            case 1:
                print("Управление: Прямоугольник/отрезок")
                b1 = double(self.lineEdit_control_b1.text())
                b2 = double(self.lineEdit_control_b2.text())
                sf_U = lambda psi: abs(b1) * abs(psi[0]) + abs(b2) * abs(psi[1])  # опорная функция для управления

                def tmp(psi):
                    ui = np.zeros(2)
                    if psi[0] > 0:
                        ui[0] = b1
                    elif psi[0] < 0:
                        ui[0] = (-b1)
                    else:
                        ui[0] = 0

                    if psi[1] > 0:
                        ui[1] = b2
                    elif psi[1] < 0:
                        ui[1] = (-b2)
                    else:
                        ui[1] = 0
                    return ui

                u_opt_func = lambda psi: tmp(psi)

        J = None
        c = None
        y_bar = None
        q_bar = None
        match self.tabWidget_functional.currentIndex():
            case 0:
                print("Функционал: Линейный")
                c = np.array([double(self.lineEdit_functional_c1.text()), double(self.lineEdit_functional_c2.text())])
                J = lambda x_T: x_T.dot(c)
                q_bar = lambda tmp: -c
            case 1:
                print("Функционал: Квадратичный")
                y_bar = np.array(
                    [double(self.lineEdit_functional_y1.text()), double(self.lineEdit_functional_y2.text())])
                J = lambda x_T: 1 / 2 * linalg.norm(x_T - y_bar) ** 2  # терминальный функционал
                q_bar = lambda x_T_opt: y_bar - x_T_opt

        if (J or sf_U or sf_M0 or q_bar or u_opt_func) is None:
            print("Ошибка: некоторые функции не определена")
            return

        params = {
            "N": int(self.lineEdit_N.text()),
            "K": int(self.lineEdit_K.text()),
            "M": int(self.lineEdit_M.text()),
            "t0": double(self.lineEdit_t0.text()),
            "T": double(self.lineEdit_T.text()),
            "A": A,
            "y_bar": y_bar,
            "c": c,
            "J": J,
            "sf_M0": sf_M0,
            "sf_U": sf_U,
            "q_bar": q_bar,
            "u_opt_func": u_opt_func,
            "x0": x0,
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
        x_T = data.get("x_T")
        x_T_opt = data.get("x_T_opt")
        x = data.get("x")
        # Получаем None
        x0 = data.get("x0")

        y_bar = data.get("y_bar")
        c = data.get("c")

        M = data.get("M")
        t0 = data.get("t0")
        T = data.get("T")
        t_g = np.linspace(t0, T, M)
        u_opt = data.get("u_opt")

        self.draw_canvas(x_T, x_T_opt, x, x0, y_bar, c, t_g, u_opt)

        convert = lambda t: str('{:.6f}'.format(t))

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
            if np.size(x0) == 2:
                main_MPL.axes.plot(x0[0], x0[1], 'o', label=r'$M_{0}$')
            else:
                main_MPL.axes.plot(x0[0], x0[1], label=r'$M_{0}$')

        if y_bar is not None:
            main_MPL.axes.plot(y_bar[0], y_bar[1], 'o', label=r'$y$')
        elif c is not None:
            main_MPL.axes.plot(c[0], c[1], 'o', label=r'$c$')

        if x_T is not None:
            main_MPL.axes.plot(x_T[0], x_T[1], label=r'$x(T)$')

        if x_T_opt is not None:
            main_MPL.axes.plot(x_T_opt[0], x_T_opt[1], 'ok', label=r'$x_{*}(T)$')

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
        t0 = params.get("t0")
        T = params.get("T")
        A = params.get("A")

        sf_M0 = params.get("sf_M0")
        x0 = params.get("x0")

        sf_U = params.get("sf_U")
        u_opt_func = params.get("u_opt_func")

        c = params.get("c")
        y_bar = params.get("y_bar")
        J = params.get("J")
        q_bar_func = params.get("q_bar")

        expm = lambda t: linalg.expm(t * A)

        A_transpose = np.transpose(A)
        expm_transpose = lambda t: linalg.expm(t * A_transpose)

        # ---------START---------
        start_time = time.time()
        x_T = covered_reachability_set(expm, expm_transpose, sf_M0, sf_U, t0, T, N, K, update_pbar_signal)
        end_time1 = float('{:.3f}'.format(time.time() - start_time))

        start_time = time.time()
        minJ, x_T_opt = minimize_functional(J, x_T, N)
        q_bar = q_bar_func(x_T_opt)
        end_time2 = float('{:.3f}'.format(time.time() - start_time))

        start_time = time.time()
        u_opt = optimal_control(expm_transpose, u_opt_func, t0, T, q_bar, M, update_pbar_signal)
        end_time3 = float('{:.3f}'.format(time.time() - start_time))

        start_time = time.time()
        x1, x2 = euler_solve(t0, T, A, x_T_opt, u_opt, M)
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
                              "N": N,
                              "M": M,
                              "T": T,
                              "t0": t0,
                              "x0": x0,
                              })


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
