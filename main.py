import sys
import time

import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal, QThread, Qt
from PyQt5.QtGui import QCursor, QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem
from numpy import double
from scipy import linalg

import design
import help
from data_executor import save_data, load_data
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
        print('Запускаю поток...')
        self.__target(*self.__args, self.update_pbar_signal, self.get_data_signal)
        self.finishing.emit()

    def stop(self):
        print('Останавливаю поток...')
        self.terminate()


class App(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.ui_help = help.Ui_Dialog()
        self.window = QtWidgets.QDialog()
        self.setupUi(self)

        self.tabWidget_initSet.setCurrentIndex(2)
        self.tabWidget_control.setCurrentIndex(0)
        self.tabWidget_functional.setCurrentIndex(0)
        self.tabWidget_plots.setCurrentIndex(0)

        text_solve_M0 = self.tabWidget_initSet.tabText(self.tabWidget_initSet.currentIndex())
        text_solve_U = self.tabWidget_control.tabText(self.tabWidget_control.currentIndex())
        text_solve_J = self.tabWidget_functional.tabText(self.tabWidget_functional.currentIndex())
        self.label_solve_M0.setText(text_solve_M0)
        self.label_solve_U.setText(text_solve_U)
        self.label_solve_J.setText(text_solve_J)

        self.btn_cancel_calculate.setVisible(False)

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
        self.lineEdit_N.textChanged[str].connect(lambda: self.onChanged(999, 999))
        self.lineEdit_K.textChanged[str].connect(lambda: self.onChanged(999, 999))
        self.lineEdit_M.textChanged[str].connect(lambda: self.onChanged(999, 999))
        self.lineEdit_A00.setValidator(self.doubleValidator)
        self.lineEdit_A01.setValidator(self.doubleValidator)
        self.lineEdit_A10.setValidator(self.doubleValidator)
        self.lineEdit_A11.setValidator(self.doubleValidator)
        self.lineEdit_A00.textChanged[str].connect(lambda: self.onChanged(999, 999))
        self.lineEdit_A01.textChanged[str].connect(lambda: self.onChanged(999, 999))
        self.lineEdit_A10.textChanged[str].connect(lambda: self.onChanged(999, 999))
        self.lineEdit_A11.textChanged[str].connect(lambda: self.onChanged(999, 999))
        self.lineEdit_t0.setValidator(self.doubleValidator)
        self.lineEdit_T.setValidator(self.doubleValidator)
        self.lineEdit_t0.textChanged[str].connect(lambda: self.onChanged(999, 999))
        self.lineEdit_T.textChanged[str].connect(lambda: self.onChanged(999, 999))

        # Параметры начального множества
        self.lineEdit_initSet_r.setValidator(self.doubleValidator)
        self.lineEdit_initSet_g1.setValidator(self.doubleValidator)
        self.lineEdit_initSet_g2.setValidator(self.doubleValidator)
        self.lineEdit_initSet_a1.setValidator(self.doubleValidator)
        self.lineEdit_initSet_a2.setValidator(self.doubleValidator)
        self.lineEdit_initSet_r.textChanged[str].connect(lambda: self.onChanged(999, 999))
        self.lineEdit_initSet_g1.textChanged[str].connect(lambda: self.onChanged(999, 999))
        self.lineEdit_initSet_g2.textChanged[str].connect(lambda: self.onChanged(999, 999))
        self.lineEdit_initSet_a1.textChanged[str].connect(lambda: self.onChanged(999, 999))
        self.lineEdit_initSet_a2.textChanged[str].connect(lambda: self.onChanged(999, 999))

        # Параметры функционала
        self.lineEdit_functional_c1.setValidator(self.doubleValidator)
        self.lineEdit_functional_c2.setValidator(self.doubleValidator)
        self.lineEdit_functional_y1.setValidator(self.doubleValidator)
        self.lineEdit_functional_y2.setValidator(self.doubleValidator)
        self.lineEdit_functional_c1.textChanged[str].connect(lambda: self.onChanged(999, 999))
        self.lineEdit_functional_c2.textChanged[str].connect(lambda: self.onChanged(999, 999))
        self.lineEdit_functional_y1.textChanged[str].connect(lambda: self.onChanged(999, 999))
        self.lineEdit_functional_y2.textChanged[str].connect(lambda: self.onChanged(999, 999))

        # Параметры управления
        self.lineEdit_control_r.setValidator(self.doubleValidator)
        self.lineEdit_control_b1.setValidator(self.doubleValidator)
        self.lineEdit_control_b2.setValidator(self.doubleValidator)
        self.lineEdit_control_r.textChanged[str].connect(lambda: self.onChanged(999, 999))
        self.lineEdit_control_b1.textChanged[str].connect(lambda: self.onChanged(999, 999))
        self.lineEdit_control_b2.textChanged[str].connect(lambda: self.onChanged(999, 999))

        self.tabWidget_initSet.tabBarClicked.connect(lambda currentTab: self.onChanged(0, currentTab))
        self.tabWidget_control.tabBarClicked.connect(lambda currentTab: self.onChanged(1, currentTab))
        self.tabWidget_functional.tabBarClicked.connect(lambda currentTab: self.onChanged(2, currentTab))

        self.btn_cancel_calculate.clicked.connect(self.cancelCalculations)
        self.btn_calculate.clicked.connect(self.calculateAll)
        self.btn_clear.clicked.connect(self.clearCalculations)
        self.btn_initSet_addRow.clicked.connect(self.addRowToTable)
        self.btn_initSet_deleteRow.clicked.connect(self.deleteRowFromTable)

        self.menu.addAction("Сохранить файл", self.actionClicked)
        self.menu.addAction("Загрузить файл", self.actionClicked)
        self.menu.addAction("Выход", self.actionClicked)
        self.menu_2.addAction("О программе", self.actionClicked)

        self.checkBox.setChecked(False)
        self.checkBox.setEnabled(False)

        self.styles_buttonCalculate = "background-color: rgb(107, 142, 35);\ncolor: rgb(255, 255, 255);"
        self.styles_buttonClear = "background-color: rgb(255, 0, 0);\ncolor: rgb(255, 255, 255);"
        self.styles_buttonDisabled = "background-color: rgb(160, 160, 160);\ncolor: rgb(0, 0, 0);"
        self.btn_clear.setEnabled(False)
        self.btn_clear.setStyleSheet(self.styles_buttonDisabled)

        self.params = None
        self.calculations = None

        self.settingPlots()

        #  Сеттим тестовые параметры для таблицы
        # Треугольник
        # self.tableWidget_InitSet.setItem(0, 0, QTableWidgetItem("2"))
        # self.tableWidget_InitSet.setItem(0, 1, QTableWidgetItem("-5"))
        # self.tableWidget_InitSet.setItem(1, 0, QTableWidgetItem("-6"))
        # self.tableWidget_InitSet.setItem(1, 1, QTableWidgetItem("1"))
        # self.tableWidget_InitSet.setItem(2, 0, QTableWidgetItem("6"))
        # self.tableWidget_InitSet.setItem(2, 1, QTableWidgetItem("-2"))
        self.tableWidget_InitSet.setRowCount(6)
        self.tableWidget_InitSet.setItem(0, 0, QTableWidgetItem("4.5"))
        self.tableWidget_InitSet.setItem(0, 1, QTableWidgetItem("5.5"))

        self.tableWidget_InitSet.setItem(1, 0, QTableWidgetItem("6"))
        self.tableWidget_InitSet.setItem(1, 1, QTableWidgetItem("8"))

        self.tableWidget_InitSet.setItem(2, 0, QTableWidgetItem("9"))
        self.tableWidget_InitSet.setItem(2, 1, QTableWidgetItem("9"))

        self.tableWidget_InitSet.setItem(3, 0, QTableWidgetItem("8"))
        self.tableWidget_InitSet.setItem(3, 1, QTableWidgetItem("4.5"))

        self.tableWidget_InitSet.setItem(4, 0, QTableWidgetItem("3"))
        self.tableWidget_InitSet.setItem(4, 1, QTableWidgetItem("1"))

        self.tableWidget_InitSet.setItem(5, 0, QTableWidgetItem("2"))
        self.tableWidget_InitSet.setItem(5, 1, QTableWidgetItem("1"))

        if self.tableWidget_InitSet.rowCount() >= 3:
            self.btn_initSet_deleteRow.setEnabled(True)
        else:
            self.btn_initSet_deleteRow.setEnabled(False)

    @QtCore.pyqtSlot()
    def actionClicked(self):
        action = self.sender()
        if action.text() == "Сохранить файл":
            print("Сохранение...")
            filename = self.saveFileDialog()
            if filename is not None and len(filename.split(".")) == 2:
                if not filename.endswith(".data"):
                    filename = filename + ".data"

                if self.params is None:
                    self.params = self.getStartParams()

                calcs = {}
                if self.checkBox.isChecked():
                    calcs = self.calculations
                save_data({"startParams": self.params, "calculations": calcs}, filename)
                print("Сохранено")
            else:
                print("Сохранение отменено!")

        elif action.text() == "Загрузить файл":
            print("Загрузка")
            filename = self.openFileNameDialog()
            if filename is not None and filename.endswith(".data"):
                data = load_data(filename)
                calcs = data.get("calculations")
                self.calculations = calcs
                x_T = calcs.get("x_T")
                x_T_opt = calcs.get("x_T_opt")
                x = calcs.get("x")
                x0 = calcs.get("x0")
                y_bar = calcs.get("y_bar")
                c = calcs.get("c")
                t_g = calcs.get("t_g")
                u_opt = calcs.get("u_opt")
                minJ = calcs.get("minJ")

                if calcs is not None:
                    self.drawCanvas(x_T, x_T_opt, x, x0, y_bar, c, t_g, u_opt)

                convert = lambda t: str('{:.6f}'.format(t))

                if minJ is not None:
                    self.lineEdit_jmin.setText(convert(minJ))
                else:
                    self.lineEdit_jmin.setText("")

                if x_T_opt is not None:
                    self.lineEdit_x1T.setText(convert(x_T_opt[0]))
                    self.lineEdit_x2T.setText(convert(x_T_opt[1]))
                else:
                    self.lineEdit_x1T.setText("")
                    self.lineEdit_x2T.setText("")

                pbarCount = calcs.get("pbarCount")
                flag = False
                if pbarCount is not None:
                    self.pbar.setValue(100)
                    flag = True
                else:
                    self.pbar.setValue(0)
                self.params = data.get("startParams")
                self.setStartParams(self.params)
                self.checkBox.setChecked(flag)
                self.checkBox.setEnabled(flag)

                self.btn_clear.setEnabled(flag)
                if flag:
                    self.btn_clear.setStyleSheet(self.styles_buttonClear)
                else:
                    self.btn_clear.setStyleSheet(self.styles_buttonDisabled)

                print("Загружено")
            else:
                print("Загрузка отменена!")

        elif action.text() == "Выход":
            print("Выход")
            self.close()

        elif action.text() == "О программе":
            print("О программе")
            self.openHelpDialog()

    def addRowToTable(self):
        rowCount = self.tableWidget_InitSet.rowCount()
        self.tableWidget_InitSet.setRowCount(rowCount + 1)
        if rowCount >= 3:
            self.btn_initSet_deleteRow.setEnabled(True)

    def deleteRowFromTable(self):
        rowCount = self.tableWidget_InitSet.rowCount()
        if rowCount == 4:
            self.btn_initSet_deleteRow.setEnabled(False)
        self.tableWidget_InitSet.setRowCount(rowCount - 1)

    def clearCalculations(self):
        self.calculations = None
        self.btn_clear.setEnabled(False)
        self.btn_clear.setStyleSheet(self.styles_buttonDisabled)
        self.checkBox.setChecked(False)
        self.checkBox.setEnabled(False)
        self.pbar.setValue(0)
        self.lineEdit_x1T.setText("")
        self.lineEdit_x2T.setText("")
        self.lineEdit_jmin.setText("")
        self.drawCanvas(None, None, None, None, None, None, None, None)

    def setStartParams(self, params):
        A = params.get("A")
        self.lineEdit_A00.setText(str(A[0][0]))
        self.lineEdit_A01.setText(str(A[0][1]))
        self.lineEdit_A10.setText(str(A[1][0]))
        self.lineEdit_A11.setText(str(A[1][1]))

        self.lineEdit_t0.setText(str(params.get("t0")))
        self.lineEdit_T.setText(str(params.get("T")))

        currentOne = None

        r1 = params.get("r1")
        if r1 is not None:
            currentOne = 0
            self.lineEdit_initSet_r.setText(str(r1))
        g = params.get("g")
        if g is not None:
            self.lineEdit_initSet_g1.setText(str(g[0]))
            self.lineEdit_initSet_g2.setText(str(g[1]))
        a = params.get("a")
        if a is not None:
            currentOne = 1
            self.lineEdit_initSet_a1.setText(str(a[0]))
            self.lineEdit_initSet_a2.setText(str(a[1]))
        self.tabWidget_initSet.setCurrentIndex(currentOne)
        text_solve_M0 = self.tabWidget_initSet.tabText(currentOne)
        self.label_solve_M0.setText(text_solve_M0)

        currentTwo = None
        r2 = params.get("r2")
        if r2 is not None:
            currentTwo = 0
            self.lineEdit_control_r.setText(str(r2))
        b = params.get("b")
        if b is not None:
            currentTwo = 1
            self.lineEdit_control_b1.setText(str(b[0]))
            self.lineEdit_control_b2.setText(str(b[1]))
        self.tabWidget_control.setCurrentIndex(currentTwo)
        text_solve_U = self.tabWidget_control.tabText(currentTwo)
        self.label_solve_U.setText(text_solve_U)

        currentThree = None
        c = params.get("c")
        if c is not None:
            currentThree = 0
            self.lineEdit_functional_c1.setText(str(c[0]))
            self.lineEdit_functional_c2.setText(str(c[1]))
        y_bar = params.get("y_bar")
        if y_bar is not None:
            currentThree = 1
            self.lineEdit_functional_y1.setText(str(y_bar[0]))
            self.lineEdit_functional_y2.setText(str(y_bar[1]))
        self.tabWidget_functional.setCurrentIndex(currentThree)
        text_solve_J = self.tabWidget_functional.tabText(currentThree)
        self.label_solve_J.setText(text_solve_J)

        self.lineEdit_N.setText(str(params.get("N")))
        self.lineEdit_K.setText(str(params.get("K")))
        self.lineEdit_M.setText(str(params.get("M")))

    def getStartParams(self):
        # Получаем параметры с форм
        A = np.array([[double(self.lineEdit_A00.text()), double(self.lineEdit_A01.text())],
                      [double(self.lineEdit_A10.text()), double(self.lineEdit_A11.text())]])

        t0 = double(self.lineEdit_t0.text())
        T = double(self.lineEdit_T.text())
        startParams = {
            "A": A,
            "t0": t0,
            "T": T,
        }

        match self.tabWidget_initSet.currentIndex():
            case 0:
                startParams["r1"] = double(self.lineEdit_initSet_r.text())
                startParams["g"] = np.array([double(self.lineEdit_initSet_g1.text()),
                                             double(self.lineEdit_initSet_g2.text())])
            case 1:
                startParams["a"] = np.array([double(self.lineEdit_initSet_a1.text()),
                                             double(self.lineEdit_initSet_a2.text())])

        match self.tabWidget_control.currentIndex():
            case 0:
                startParams["r2"] = double(self.lineEdit_control_r.text())
            case 1:
                startParams["b"] = np.array([double(self.lineEdit_control_b1.text()),
                                             double(self.lineEdit_control_b2.text())])

        match self.tabWidget_functional.currentIndex():
            case 0:
                startParams["c"] = np.array([double(self.lineEdit_functional_c1.text()),
                                             double(self.lineEdit_functional_c2.text())])
            case 1:
                startParams["y_bar"] = np.array([double(self.lineEdit_functional_y1.text()),
                                                 double(self.lineEdit_functional_y2.text())])

        startParams["N"] = int(self.lineEdit_N.text())
        startParams["K"] = int(self.lineEdit_K.text())
        startParams["M"] = int(self.lineEdit_M.text())
        return startParams

    def saveFileDialog(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Сохранить данные", "examples",
                                                  "Файл с данными (*.data)")
        return filename

    def openFileNameDialog(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Выбрать файл для загрузки", "examples",
                                                  "Файл с данными (*.data)")
        return filename

    def openHelpDialog(self):
        self.ui_help.setupUi(self.window)
        self.window.show()

    def onChanged(self, tab, current):
        self.checkBox.setChecked(False)
        self.checkBox.setEnabled(False)
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

        styles = self.styles_buttonCalculate
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

        self.btn_calculate.setStyleSheet(styles)
        self.btn_calculate.setEnabled(flag)

    def cancelCalculations(self):
        self.thread[1].stop()
        self.btn_cancel_calculate.setVisible(False)
        self.btn_calculate.setVisible(True)

        pbar = 0
        if self.calculations is not None:
            self.btn_clear.setEnabled(True)
            self.btn_clear.setStyleSheet(self.styles_buttonClear)
            pbar = 100
        self.updateProgressbar(pbar)

        self.setCursor(QCursor(Qt.ArrowCursor))

    def getParamsFromFormsForSolving(self):
        A = np.array([[double(self.lineEdit_A00.text()), double(self.lineEdit_A01.text())],
                      [double(self.lineEdit_A10.text()), double(self.lineEdit_A11.text())]])
        params = {
            "A": A,
            "t0": double(self.lineEdit_t0.text()),
            "T": double(self.lineEdit_T.text()),
            "N": int(self.lineEdit_N.text()),
            "K": int(self.lineEdit_K.text()),
            "M": int(self.lineEdit_M.text()),
        }

        match self.tabWidget_initSet.currentIndex():
            case 0:
                r1 = double(self.lineEdit_initSet_r.text())
                g = np.array([double(self.lineEdit_initSet_g1.text()),
                              double(self.lineEdit_initSet_g2.text())])
                print("Начальное множество: Круг/точка с параметрами r={}, g={}".format(r1, g))
                params["sf_M0"] = lambda psi: g[0] * psi[0] + g[1] * psi[1] + abs(r1) * linalg.norm(psi)

                if r1 == 0:
                    params["x0"] = g
                else:
                    phi = np.linspace(0, 2 * np.pi, 100)
                    x = g[0] + r1 * np.cos(phi)
                    y = g[1] + r1 * np.sin(phi)
                    params["x0"] = [x, y]
            case 1:
                a = np.array([double(self.lineEdit_initSet_a1.text()),
                              double(self.lineEdit_initSet_a2.text())])
                print("Начальное множество: Прямоугольник/отрезок ограниченный нер-вами u1 <= {}, u2 <={}".format(a[0],
                                                                                                                  a[1]))
                params["sf_M0"] = lambda psi: abs(a[0]) * abs(psi[0]) + abs(a[1]) * abs(psi[1])
                params["x0"] = [[-a[0], a[0], a[0], -a[0], -a[0]], [a[1], a[1], -a[1], -a[1], a[1]]]
            case 2:
                rowCount = self.tableWidget_InitSet.rowCount()
                v1 = np.zeros(rowCount)
                v2 = np.zeros(rowCount)
                for i in range(rowCount):
                    widgetItemX = self.tableWidget_InitSet.item(i, 0)
                    if widgetItemX and widgetItemX.text:
                        v1[i] = double(widgetItemX.text())
                    widgetItemY = self.tableWidget_InitSet.item(i, 1)
                    if widgetItemY and widgetItemY.text:
                        v2[i] = double(widgetItemY.text())
                print("Начальное множество: многоугольник")
                print("x = {}\ny = {}".format(v1, v2))

                def tmp_m0_func(psi):
                    maxx = np.array([v1[0], v2[0]]).dot(psi)
                    for j in range(1, rowCount):
                        tmp = np.array([v1[j], v2[j]]).dot(psi)
                        if tmp >= maxx:
                            maxx = tmp
                    return maxx

                params["sf_M0"] = lambda psi: tmp_m0_func(psi)
                params["x0"] = [np.append(v1, v1[0]), np.append(v2, v2[0])]

        match self.tabWidget_control.currentIndex():
            case 0:
                r2 = double(self.lineEdit_control_r.text())
                print("Управление: Круг радиуса {}".format(r2))
                params["sf_U"] = lambda psi: abs(r2) * linalg.norm(psi)
                params["u_opt_func"] = lambda psi: psi * abs(r2) / linalg.norm(psi)
            case 1:
                b = np.array([double(self.lineEdit_control_b1.text()),
                              double(self.lineEdit_control_b2.text())])
                print("Управление: Прямоугольник/отрезок ограниченный нер-вами u1 <= {}, u2 <={}".format(b[0], b[1]))

                def tmp_u_func(psi):
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

                params["sf_U"] = \
                    lambda psi: abs(b[0]) * abs(psi[0]) + abs(b[1]) * abs(psi[1])
                params["u_opt_func"] = lambda psi: tmp_u_func(psi)

        match self.tabWidget_functional.currentIndex():
            case 0:
                c = np.array([double(self.lineEdit_functional_c1.text()),
                              double(self.lineEdit_functional_c2.text())])
                print("Функционал: Линейный с вектором c = {}".format(c))
                params["c"] = c
                params["J"] = lambda x_T: x_T.dot(c)
                params["q_bar_func"] = lambda x_T_opt: -c
            case 1:
                y_bar = np.array([double(self.lineEdit_functional_y1.text()),
                                  double(self.lineEdit_functional_y2.text())])
                print("Функционал: Квадратичный с вектором y = {}".format(y_bar))
                params["y_bar"] = y_bar
                params["J"] = lambda x_T: 1 / 2 * linalg.norm(x_T - y_bar) ** 2
                params["q_bar_func"] = lambda x_T_opt: y_bar - x_T_opt

        return params

    def calculateAll(self):
        # def testing():
        # L = 10
        # maximums = np.zeros(rowCount)
        # psi = np.array([random.uniform(-99, 99), random.uniform(-99, 99)])
        # for i in range(-1, rowCount - 1):
        #     x = np.linspace(a1[i], a1[i + 1], L)
        #     y = np.zeros(L)
        #     for j in range(L):
        #         y[j] = test.lineEquationInX([a1[i], a2[i]], [a1[i + 1], a2[i + 1]], x[j])
        #
        #     maxx = np.array([x[0], y[0]]).dot(psi)
        #     maxIter = 0
        #     for j in range(1, L):
        #         tmp = np.array([x[j], y[j]]).dot(psi)
        #         # print("j = {}, tmp = {}".format(j, tmp))
        #         if tmp > maxx:
        #             maxIter = j
        #             maxx = tmp
        #     maximums[i + 1] = maxx
        #     # print("x: {}".format(x))
        #     # print("y: {}".format(y))
        #     # print("maximums[{}]: {}".format(i + 1, maximums[i + 1]))
        #     # print("maxIter = {}, maxIter == 0 is {}, maxIter == {} is {}".
        #     format(maxIter, maxIter == 0, L - 1, maxIter == L - 1))
        #     if not (maxIter == 0 or maxIter == L - 1):
        #         return True
        #     # print("\n\n")
        # print(maximums)
        # return False

        params = self.getParamsFromFormsForSolving()
        self.thread[1] = ThreadClass(self.startLongRunningTask, params, parent=None)
        self.thread[1].start()

        self.thread[1].update_pbar_signal.connect(self.updateProgressbar)
        self.thread[1].get_data_signal.connect(self.gettingResults)

        self.btn_clear.setEnabled(False)
        self.btn_clear.setStyleSheet(self.styles_buttonDisabled)
        self.btn_calculate.setVisible(False)
        self.btn_cancel_calculate.setVisible(True)
        self.setCursor(QCursor(Qt.WaitCursor))

        self.thread[1].finishing.connect(lambda: self.setCursor(QCursor(Qt.ArrowCursor)))
        self.thread[1].finishing.connect(lambda: self.btn_cancel_calculate.setVisible(False))
        self.thread[1].finishing.connect(lambda: self.btn_calculate.setVisible(True))
        self.thread[1].finishing.connect(self.thread[1].stop)

    def gettingResults(self, data):

        x_T = data.get("x_T")
        x_T_opt = data.get("x_T_opt")
        x = data.get("x")
        x0 = data.get("x0")

        y_bar = data.get("y_bar")
        c = data.get("c")

        M = data.get("M")
        t0 = data.get("t0")
        T = data.get("T")
        t_g = np.linspace(t0, T, M)
        u_opt = data.get("u_opt")
        minJ = data.get("minJ")

        self.calculations = {
            "x0": x0,
            "c": c,
            "y_bar": y_bar,
            "x_T": x_T,
            "x_T_opt": x_T_opt,
            "x": x,
            "minJ": minJ,
            "u_opt": u_opt,
            "t_g": t_g,
            "pbarCount": 100,
        }

        self.btn_clear.setEnabled(True)
        self.btn_clear.setStyleSheet(self.styles_buttonClear)

        self.params = self.getStartParams()

        # print("Стартовые параметры задачи:\n{}\n\n\n".format(self.params))
        # print("Результаты вычисления:\n{}\n\n\n".format(self.calculations))

        self.drawCanvas(x_T, x_T_opt, x, x0, y_bar, c, t_g, u_opt)

        convert = lambda t: str('{:.6f}'.format(t))

        if minJ is not None:
            self.lineEdit_jmin.setText(convert(minJ))

        if x_T_opt is not None:
            self.lineEdit_x1T.setText(convert(x_T_opt[0]))
            self.lineEdit_x2T.setText(convert(x_T_opt[1]))

        self.checkBox.setChecked(True)
        self.checkBox.setEnabled(True)

    def updateProgressbar(self, cnt):
        self.pbar.setValue(cnt)

    def settingPlots(self):
        main_MPL = self.MplWidget.canvas
        main_MPL.axes.clear()
        main_MPL.axes.set_xlabel(r"$x_{1}$", size=13)
        main_MPL.axes.set_ylabel(r"$x_{2}$", size=13)
        main_MPL.axes.axis('equal')
        main_MPL.axes.grid(color='grey',  # цвет линий
                           linewidth=0.5,  # толщина
                           linestyle='--')  # начертание

        u1_MPL = self.MplWidget_u1.canvas
        u1_MPL.axes.clear()
        u1_MPL.axes.set_xlabel(r"$t$", size=13)
        u1_MPL.axes.set_ylabel(r"$u_{1}$", size=13)

        u2_MPL = self.MplWidget_u2.canvas
        u2_MPL.axes.clear()
        u2_MPL.axes.set_xlabel(r"$t$", size=13)
        u2_MPL.axes.set_ylabel(r"$u_{2}$", size=13)

        main_MPL.draw()
        u1_MPL.draw()
        u2_MPL.draw()

    def drawCanvas(self, x_T, x_T_opt, x, x0, y_bar, c, t_g, u_opt):
        self.settingPlots()
        main_MPL = self.MplWidget.canvas

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
            main_MPL.axes.plot(x[0], x[1], label=r'$x(t)$')
            main_MPL.axes.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.1, 1.15), shadow=True)

        u1_MPL = self.MplWidget_u1.canvas
        u2_MPL = self.MplWidget_u2.canvas
        if u_opt is not None:
            u1_MPL.axes.plot(t_g, u_opt[0], label=r'$u_{1}(t)$')
            u1_MPL.axes.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.1, 1.15), shadow=True)

            u2_MPL.axes.plot(t_g, u_opt[1], label=r'$u_{2}(t)$')
            u2_MPL.axes.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.1, 1.15), shadow=True)

        main_MPL.draw()
        u1_MPL.draw()
        u2_MPL.draw()

    @staticmethod
    def startLongRunningTask(params, update_pbar_signal, get_data_signal):
        print("Запускаю долгие вычисления в другом потоке")
        N = params.get("N")  # Количество вершин для множества достижимости
        K = params.get("K")  # Разбиение для метода трапеций
        M = params.get("M")  # Численное решение задачи коши
        A = params.get("A")
        t0 = params.get("t0")
        T = params.get("T")

        sf_M0 = params.get("sf_M0")
        x0 = params.get("x0")

        sf_U = params.get("sf_U")
        u_opt_func = params.get("u_opt_func")

        c = params.get("c")
        y_bar = params.get("y_bar")
        J = params.get("J")
        q_bar_func = params.get("q_bar_func")

        expm = lambda t: linalg.expm(t * A)

        A_transpose = np.transpose(A)
        expm_transpose = lambda t: linalg.expm(t * A_transpose)

        # ---------Старт---------
        start_time = time.time()
        x_T = covered_reachability_set(expm, expm_transpose, sf_M0, sf_U, t0, T, N, K, M, update_pbar_signal)
        end_time1 = float('{:.3f}'.format(time.time() - start_time))

        start_time = time.time()
        minJ, x_T_opt = minimize_functional(J, x_T, N)
        q_bar = q_bar_func(x_T_opt)
        end_time2 = float('{:.3f}'.format(time.time() - start_time))

        start_time = time.time()
        u_opt = optimal_control(expm_transpose, u_opt_func, t0, T, q_bar, N, K, M, update_pbar_signal)
        end_time3 = float('{:.3f}'.format(time.time() - start_time))

        start_time = time.time()
        x1, x2 = euler_solve(t0, T, A, x_T_opt, u_opt, M)
        end_time4 = float('{:.3f}'.format(time.time() - start_time))

        end_time = float('{:.6f}'.format(end_time1 + end_time2 + end_time3 + end_time4))
        # ----------------------------------Завершение----------------------------------

        print("minJ = {}".format(minJ))
        print("x_T_opt = {}".format(x_T_opt))
        print("q_bar = {}".format(q_bar))
        print("1) Время = {} секунд".format(end_time1))
        print("2) Время = {} секунд".format(end_time2))
        print("3) Время = {} секунд".format(end_time3))
        print("4) Время = {} секунд".format(end_time4))
        print("Общее время = {} секунд".format(end_time))
        print("\n\n")

        plot_x_T1 = np.zeros(N + 1)
        plot_x_T2 = np.zeros(N + 1)
        for i in range(N):
            plot_x_T1[i], plot_x_T2[i] = x_T[i][0], x_T[i][1]
        plot_x_T1[N], plot_x_T2[N] = x_T[0][0], x_T[0][1]

        plot_u1 = np.zeros(M)
        plot_u2 = np.zeros(M)
        for i in range(M):
            plot_u1[i], plot_u2[i] = u_opt[i][0], u_opt[i][1]

        get_data_signal.emit({"minJ": minJ,
                              "x_T_opt": x_T_opt,
                              "q_bar": q_bar,
                              "x_T": [plot_x_T1, plot_x_T2],
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
