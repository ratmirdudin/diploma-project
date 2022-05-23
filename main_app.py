import sys

import matplotlib.patches as mpatches
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
from main_calculations import coveredReachabilitySet, optimalControl, \
    minimizeFunctionalForLinearProblem, minimizeQuadraticFunctional, \
    getTwoDifferentMinimumsInsteadOfFirstMinOfFunctional, \
    directEulerSolve, backEulerSolve, findOptimalCoordsOfM0, optimalControlForAZeroMatrix
from utils import isDoublesEqual, sf_polygon, u_opt_quad, u_opt_polygon, isDoubleArraysEqual


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

        self.tabWidget_initSet.setCurrentIndex(0)
        self.tabWidget_control.setCurrentIndex(2)
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

        self.btn_initSet_addRow.clicked.connect(self.addRowToInitSetTable)
        self.btn_initSet_deleteRow.clicked.connect(self.deleteRowFromInitSetTable)

        self.btn_control_addRow.clicked.connect(self.addRowToControlTable)
        self.btn_control_deleteRow.clicked.connect(self.deleteRowFromControlTable)

        self.menu.addAction("Сохранить файл", self.actionClicked)
        self.menu.addAction("Загрузить файл", self.actionClicked)
        self.menu.addAction("Выход", self.actionClicked)
        # self.menu_2.addAction("О программе", self.actionClicked)

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
        # self.tableWidget_initSet.setRowCount(3)
        # self.tableWidget_InitSet.setItem(0, 0, QTableWidgetItem("2"))
        # self.tableWidget_InitSet.setItem(0, 1, QTableWidgetItem("-5"))
        # self.tableWidget_InitSet.setItem(1, 0, QTableWidgetItem("-6"))
        # self.tableWidget_InitSet.setItem(1, 1, QTableWidgetItem("1"))
        # self.tableWidget_InitSet.setItem(2, 0, QTableWidgetItem("6"))
        # self.tableWidget_InitSet.setItem(2, 1, QTableWidgetItem("-2"))

        # Заполняем многоугольник начального множества
        self.tableWidget_initSet.setRowCount(5)

        self.tableWidget_initSet.setItem(0, 0, QTableWidgetItem("6"))
        self.tableWidget_initSet.setItem(0, 1, QTableWidgetItem("8"))

        self.tableWidget_initSet.setItem(1, 0, QTableWidgetItem("9"))
        self.tableWidget_initSet.setItem(1, 1, QTableWidgetItem("9"))

        self.tableWidget_initSet.setItem(2, 0, QTableWidgetItem("8"))
        self.tableWidget_initSet.setItem(2, 1, QTableWidgetItem("4.5"))

        self.tableWidget_initSet.setItem(3, 0, QTableWidgetItem("3"))
        self.tableWidget_initSet.setItem(3, 1, QTableWidgetItem("1"))

        self.tableWidget_initSet.setItem(4, 0, QTableWidgetItem("2"))
        self.tableWidget_initSet.setItem(4, 1, QTableWidgetItem("1"))

        if self.tableWidget_initSet.rowCount() > 3:
            self.btn_initSet_deleteRow.setEnabled(True)
        else:
            self.btn_initSet_deleteRow.setEnabled(False)

        # Заполняем многоугольник управления
        self.tableWidget_control.setRowCount(4)

        self.tableWidget_control.setItem(0, 0, QTableWidgetItem("-1"))
        self.tableWidget_control.setItem(0, 1, QTableWidgetItem("-1"))

        self.tableWidget_control.setItem(1, 0, QTableWidgetItem("-1"))
        self.tableWidget_control.setItem(1, 1, QTableWidgetItem("1"))

        self.tableWidget_control.setItem(2, 0, QTableWidgetItem("1"))
        self.tableWidget_control.setItem(2, 1, QTableWidgetItem("1"))

        self.tableWidget_control.setItem(3, 0, QTableWidgetItem("1"))
        self.tableWidget_control.setItem(3, 1, QTableWidgetItem("-1"))

        # Пятиугольник управления
        # Заполняем многоугольник управления
        # self.tableWidget_control.setRowCount(5)
        #
        # self.tableWidget_control.setItem(0, 0, QTableWidgetItem("-2"))
        # self.tableWidget_control.setItem(0, 1, QTableWidgetItem("3"))
        #
        # self.tableWidget_control.setItem(1, 0, QTableWidgetItem("1"))
        # self.tableWidget_control.setItem(1, 1, QTableWidgetItem("3"))
        #
        # self.tableWidget_control.setItem(2, 0, QTableWidgetItem("3"))
        # self.tableWidget_control.setItem(2, 1, QTableWidgetItem("-1"))
        #
        # self.tableWidget_control.setItem(3, 0, QTableWidgetItem("0"))
        # self.tableWidget_control.setItem(3, 1, QTableWidgetItem("-3"))
        #
        # self.tableWidget_control.setItem(4, 0, QTableWidgetItem("-3"))
        # self.tableWidget_control.setItem(4, 1, QTableWidgetItem("-1"))

        if self.tableWidget_control.rowCount() > 3:
            self.btn_control_deleteRow.setEnabled(True)
        else:
            self.btn_control_deleteRow.setEnabled(False)

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
                    pass
                    # self.drawCanvas(x_T, x_T_opt, x, x0, y_bar, c, t_g, u_opt)

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

    def addRowToInitSetTable(self):
        rowCount = self.tableWidget_initSet.rowCount()
        self.tableWidget_initSet.setRowCount(rowCount + 1)
        if rowCount >= 3:
            self.btn_initSet_deleteRow.setEnabled(True)

    def addRowToControlTable(self):
        rowCount = self.tableWidget_control.rowCount()
        self.tableWidget_control.setRowCount(rowCount + 1)
        if rowCount >= 3:
            self.btn_control_deleteRow.setEnabled(True)

    def deleteRowFromInitSetTable(self):
        rowCount = self.tableWidget_initSet.rowCount()
        if rowCount == 4:
            self.btn_initSet_deleteRow.setEnabled(False)
        self.tableWidget_initSet.setRowCount(rowCount - 1)

    def deleteRowFromControlTable(self):
        rowCount = self.tableWidget_control.rowCount()
        if rowCount == 4:
            self.btn_control_deleteRow.setEnabled(False)
        self.tableWidget_control.setRowCount(rowCount - 1)

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
        self.drawCanvas(None, None, None, None, None, None, None, None, None, None, None, None, None, None)

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
        v = params.get("v")
        if v is not None:
            currentOne = 2
            rowCount = len(v[0])
            self.tableWidget_initSet.setRowCount(rowCount)
            for i in range(rowCount):
                self.tableWidget_initSet.setItem(i, 0, QTableWidgetItem(str(v[0][i])))
                self.tableWidget_initSet.setItem(i, 1, QTableWidgetItem(str(v[1][i])))

        if currentOne is not None:
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
        if currentTwo is not None:
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
        if currentThree is not None:
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
            case 2:
                rowCount = self.tableWidget_initSet.rowCount()
                v1 = np.zeros(rowCount)
                v2 = np.zeros(rowCount)
                for i in range(rowCount):
                    widgetItemX = self.tableWidget_initSet.item(i, 0)
                    if widgetItemX and widgetItemX.text:
                        v1[i] = double(widgetItemX.text())
                    widgetItemY = self.tableWidget_initSet.item(i, 1)
                    if widgetItemY and widgetItemY.text:
                        v2[i] = double(widgetItemY.text())
                startParams["v"] = np.array([v1, v2])

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

        # Выбор начального множества M0
        typeOfInitSet = self.tabWidget_initSet.currentIndex()
        params["typeOfInitSet"] = typeOfInitSet
        if typeOfInitSet == 0:
            r1 = double(self.lineEdit_initSet_r.text())
            g = np.array([[double(self.lineEdit_initSet_g1.text()),
                           double(self.lineEdit_initSet_g2.text())]])
            print("Начальное множество: Круг/точка с параметрами:\nr={}, g={}".format(r1, g))
            params["sf_M0"] = lambda psi: g[0][0] * psi[0] + g[0][1] * psi[1] + abs(r1) * linalg.norm(psi)

            if r1 == 0:
                params["x0"] = g
            else:
                phi = np.linspace(0, 2 * np.pi, 100)
                x = g[0][0] + r1 * np.cos(phi)
                y = g[0][1] + r1 * np.sin(phi)
                xy = np.zeros((len(x), 2))
                for i in range(len(x)):
                    xy[i] = [x[i], y[i]]
                params["x0"] = xy
        elif typeOfInitSet == 1:
            a = np.array([double(self.lineEdit_initSet_a1.text()),
                          double(self.lineEdit_initSet_a2.text())])
            print("Начальное множество: Прямоугольник/отрезок ограниченный нер-вами:\nu1 <= {}, u2 <={}".format(a[0],
                                                                                                                a[1]))
            params["sf_M0"] = lambda psi: abs(a[0]) * abs(psi[0]) + abs(a[1]) * abs(psi[1])
            params["x0"] = np.array([[-a[0], a[1]], [a[0], a[1]], [a[0], -a[1]], [-a[0], -a[1]]])
        elif typeOfInitSet == 2:
            table = self.tableWidget_initSet
            rowCount = table.rowCount()
            v = np.zeros((rowCount, 2))
            for i in range(rowCount):
                widgetItemX = table.item(i, 0)
                if widgetItemX and widgetItemX.text:
                    v[i][0] = double(widgetItemX.text())
                widgetItemY = table.item(i, 1)
                if widgetItemY and widgetItemY.text:
                    v[i][1] = double(widgetItemY.text())
            print("Начальное множество: многоугольник")
            print("v = {}".format(v))

            params["sf_M0"] = lambda psi: sf_polygon(psi, v)
            params["x0"] = v

        # Выбор управления U
        typeOfControl = self.tabWidget_control.currentIndex()
        params["typeOfControl"] = typeOfControl
        if typeOfControl == 0:
            r2 = double(self.lineEdit_control_r.text())
            print("Управление: Круг радиуса {}".format(r2))
            params["sf_U"] = lambda psi: abs(r2) * linalg.norm(psi)
            params["u_opt_func"] = lambda psi: psi * abs(r2) / linalg.norm(psi)
        elif typeOfControl == 1:
            b = np.array([double(self.lineEdit_control_b1.text()),
                          double(self.lineEdit_control_b2.text())])
            print("Управление: Прямоугольник/отрезок ограниченный нер-вами:\nu1 <= {}, u2 <={}".format(b[0], b[1]))
            # НАДО ДОБАВИТЬ ПО АНАЛОГИИ С W
            params["sf_U"] = \
                lambda psi: abs(b[0]) * abs(psi[0]) + abs(b[1]) * abs(psi[1])
            params["u_opt_func"] = lambda psi: u_opt_quad(psi, b)

        elif typeOfControl == 2:
            table = self.tableWidget_control
            rowCount = table.rowCount()
            w = np.zeros((rowCount, 2))
            for i in range(rowCount):
                widgetItemX = table.item(i, 0)
                if widgetItemX and widgetItemX.text:
                    w[i][0] = double(widgetItemX.text())
                widgetItemY = table.item(i, 1)
                if widgetItemY and widgetItemY.text:
                    w[i][1] = double(widgetItemY.text())
            print("Начальное множество: многоугольник")
            print("w = {}".format(w))
            params["w"] = w
            params["sf_U"] = lambda psi: sf_polygon(psi, w)
            params["u_opt_func"] = lambda psi: u_opt_polygon(psi, w)

        # Выбор функционала J
        typeOfFunctional = self.tabWidget_functional.currentIndex()
        params["typeOfFunctional"] = typeOfFunctional
        if typeOfFunctional == 0:
            c = np.array([double(self.lineEdit_functional_c1.text()),
                          double(self.lineEdit_functional_c2.text())])
            print("Функционал: Линейный с вектором c = {}".format(c))
            params["c"] = c
            params["J"] = lambda x_T: x_T.dot(c)
            params["psi_T_func"] = lambda x_T_opt: -c

        elif typeOfFunctional == 1:
            y_bar = np.array([double(self.lineEdit_functional_y1.text()),
                              double(self.lineEdit_functional_y2.text())])
            print("Функционал: Квадратичный с вектором y = {}".format(y_bar))
            params["y_bar"] = y_bar
            params["J"] = lambda x_T: linalg.norm(x_T - y_bar) ** 2 * 1 / 2
            params["psi_T_func"] = lambda x_T_opt: -2 * (x_T_opt - y_bar) * 1 / 2

        return params

    def calculateAll(self):
        params = self.getParamsFromFormsForSolving()
        self.thread[1] = ThreadClass(self.startLongRunningCalculations, params, parent=None)
        self.thread[1].start()

        self.thread[1].update_pbar_signal.connect(self.updateProgressbar)
        self.thread[1].get_data_signal.connect(self.handleResultsAfterCalculations)

        self.btn_clear.setEnabled(False)
        self.btn_clear.setStyleSheet(self.styles_buttonDisabled)
        self.btn_calculate.setVisible(False)
        self.btn_cancel_calculate.setVisible(True)
        self.setCursor(QCursor(Qt.WaitCursor))

        self.thread[1].finishing.connect(lambda: self.setCursor(QCursor(Qt.ArrowCursor)))
        self.thread[1].finishing.connect(lambda: self.btn_cancel_calculate.setVisible(False))
        self.thread[1].finishing.connect(lambda: self.btn_calculate.setVisible(True))
        self.thread[1].finishing.connect(self.thread[1].stop)

    def handleResultsAfterCalculations(self, data):
        fillsArea = data.get("fillsArea")

        typeOfControl = data.get("typeOfControl")

        x_T = data.get("x_T")
        x_T_opt = data.get("x_T_opt")
        x_T_opt_line = data.get("x_T_opt_line")
        x = data.get("x")
        optimalTrajectories = data.get("optimalTrajectories")
        x0 = data.get("x0")

        y_bar = data.get("y_bar")
        c = data.get("c")

        t0 = data.get("t0")
        T = data.get("T")
        u_opt = data.get("u_opt")
        t_g = np.linspace(t0, T, len(u_opt[0]))

        minJ = data.get("minJ")

        psi_t0 = data.get("psi_t0")
        psi_T = data.get("psi_T")

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

        self.drawCanvas(x_T, x_T_opt, x_T_opt_line, x, optimalTrajectories, x0, y_bar, c, t_g, u_opt, typeOfControl,
                        psi_t0, psi_T,
                        fillsArea)

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
        main_MPL.axes.grid(
            linewidth=0.5,  # толщина
            linestyle='--',  # начертание
            color='grey',  # цвет линий
        )

        # Оси и прочие другие настройки для осей
        # main_MPL.axes.spines['left'].set_position('center')
        # main_MPL.axes.spines['bottom'].set_position('center')
        # main_MPL.axes.spines['top'].set_visible(False)
        # main_MPL.axes.spines['right'].set_visible(False)

        u1_MPL = self.MplWidget_u1.canvas
        u1_MPL.axes.clear()
        u1_MPL.axes.set_xlabel(r"$t$", size=13)
        u1_MPL.axes.set_ylabel(r"$u_{1}$", size=13)
        u1_MPL.axes.grid(
            linewidth=0.5,  # толщина
            linestyle='--',  # начертание
            color='grey',  # цвет линий
        )

        u2_MPL = self.MplWidget_u2.canvas
        u2_MPL.axes.clear()
        u2_MPL.axes.set_xlabel(r"$t$", size=13)
        u2_MPL.axes.set_ylabel(r"$u_{2}$", size=13)
        u2_MPL.axes.grid(
            linewidth=0.5,  # толщина
            linestyle='--',  # начертание
            color='grey',  # цвет линий
        )

        main_MPL.draw()
        u1_MPL.draw()
        u2_MPL.draw()

    def drawCanvas(self, x_T, x_T_opt, x_T_opt_line, x, optimalTrajectories, x0, y_bar, c, t_g, u_opt, typeOfControl,
                   psi_t0, psi_T,
                   fillsArea):
        self.settingPlots()
        main_MPL = self.MplWidget.canvas

        if x_T is not None:
            main_MPL.axes.plot(x_T[0], x_T[1], label=r'$X(T)$', color="tab:green", linewidth=2)

        if y_bar is not None:
            main_MPL.axes.plot(y_bar[0], y_bar[1], 'o', label=r'$y$', color="tab:orange")
        elif c is not None:
            maxCoords = [x_T[0][0], x_T[1][0]]
            maxx = np.array([c[0], c[1]]).dot(np.array([x_T[0][0], x_T[1][0]]))
            for i in range(1, len(x_T[0])):
                tmp = np.array([c[0], c[1]]).dot(np.array([x_T[0][i], x_T[1][i]]))
                if tmp > maxx:
                    maxx = tmp
                    maxCoords = [x_T[0][i], x_T[1][i]]

            main_MPL.axes.quiver(maxCoords[0], maxCoords[1], 2 * c[0], 2 * c[1], units='xy', color="tab:orange")
            main_MPL.axes.plot(maxCoords[0], maxCoords[1], label=r'$c$', color="tab:orange", linewidth=3)

        if x0 is not None:
            if np.size(x0) == 4:
                main_MPL.axes.plot(x0[0][0], x0[1][0], 'o', label=r'$M_{0}$', color="tab:blue", markersize=5)
            else:
                main_MPL.axes.plot(x0[0], x0[1], label=r'$M_{0}$', color="tab:blue")

        # КОСТЫЛЬ******************************************
        x_T_opt_line = None

        if x_T_opt_line is not None:
            main_MPL.axes.plot(x_T_opt_line[0], x_T_opt_line[1],
                               label=r"Опор. мн-во $Х(Т)$" + "\n" + r"в напр. $\psi(T)$",
                               color="tab:red", linewidth=3)
        if x_T_opt is not None and x_T_opt_line is None:
            main_MPL.axes.plot(x_T_opt[0], x_T_opt[1], 'o', color="tab:red", markersize=5)

        # КОСТЫЛЬ 2 ******************************************
        fillsArea = None

        if x is not None:
            if fillsArea is not None:
                main_MPL.axes.plot(x[0][0], x[1][0], label=r"$\psi(t_0), \psi(T)$", color="black", linewidth=3)
            else:
                main_MPL.axes.plot(x[0][0], x[1][0], label=r"$\psi(t_0),$" "\n" + r"$\psi(T)$", color="black",
                                   linewidth=3)
            main_MPL.axes.plot(x[0], x[1], label=r'$x_{*}(t)$', color="tab:red")

            if optimalTrajectories is not None:
                for i in range(1, len(optimalTrajectories)):
                    xplt = optimalTrajectories[i]
                    main_MPL.axes.plot(xplt[0], xplt[1], color="tab:red")

            handles, labels = main_MPL.axes.get_legend_handles_labels()
            if fillsArea is not None and t_g[len(t_g) - 1] > 1:
                main_MPL.axes.fill(fillsArea[0], fillsArea[1], facecolor='tab:gray')
                gray_patch = mpatches.Patch(color='tab:gray', label="Обл. опт." + "\n" + "траекторий")
                handles.append(gray_patch)
            main_MPL.axes.quiver(x[0][0], x[1][0], psi_t0[0], psi_t0[1], units='xy', color="black")
            main_MPL.axes.quiver(x[0][len(x[0]) - 1], x[1][len(x[1]) - 1], psi_T[0], psi_T[1], units='xy',
                                 color="black")
            main_MPL.axes.legend(handles=handles, loc='upper right', fontsize=10, bbox_to_anchor=(1.1, 1.15),
                                 shadow=True)

        u1_MPL = self.MplWidget_u1.canvas
        u2_MPL = self.MplWidget_u2.canvas
        if u_opt is not None:
            if typeOfControl == 1 or typeOfControl == 2:
                u_plot1 = []
                t_g_plot1 = []
                u_plot2 = []
                t_g_plot2 = []
                for i in range(len(u_opt[0]) - 1):
                    if u_opt[0][i] != u_opt[0][i + 1]:
                        # u1_MPL.axes.plot([t_g[i], t_g[i + 1]], [u_opt[0][i], u_opt[0][i + 1]], '--', color="blue")
                        u_plot1 = np.append(u_plot1, np.nan)
                        t_g_plot1 = np.append(t_g_plot1, np.nan)
                    else:
                        u_plot1 = np.append(u_plot1, u_opt[0][i])
                        t_g_plot1 = np.append(t_g_plot1, t_g[i])

                    if u_opt[1][i] != u_opt[1][i + 1]:
                        # u2_MPL.axes.plot([t_g[i], t_g[i + 1]], [u_opt[1][i], u_opt[1][i + 1]], '--', color="blue")
                        u_plot2 = np.append(u_plot2, np.nan)
                        t_g_plot2 = np.append(t_g_plot2, np.nan)
                    else:
                        u_plot2 = np.append(u_plot2, u_opt[1][i])
                        t_g_plot2 = np.append(t_g_plot2, t_g[i])

                if double(self.lineEdit_control_b1.text()) != 0:
                    u1_MPL.axes.plot(t_g_plot1, u_plot1, color="blue", label=r'$u_{1*}(t)$', linewidth=2.5)
                    # u1_MPL.axes.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.1, 1.15), shadow=True)

                if double(self.lineEdit_control_b2.text()) != 0:
                    u2_MPL.axes.plot(t_g_plot2, u_plot2, color="blue", label=r'$u_{2*}(t)$', linewidth=2.5)
                    # u2_MPL.axes.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.1, 1.15), shadow=True)
            else:
                u1_MPL.axes.plot(t_g, u_opt[0], color="blue", label=r'$u_{1*}(t)$', linewidth=2.5)
                # u1_MPL.axes.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.1, 1.15), shadow=True)

                u2_MPL.axes.plot(t_g, u_opt[1], color="blue", label=r'$u_{2*}(t)$', linewidth=2.5)
                # u2_MPL.axes.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.1, 1.15), shadow=True)

        main_MPL.draw()
        u1_MPL.draw()
        u2_MPL.draw()

    @staticmethod
    def startLongRunningCalculations(params, update_pbar_signal, get_data_signal):
        print("Запускаю долгие вычисления в другом потоке")

        typeOfFunctional = params.get("typeOfFunctional")
        typeOfControl = params.get("typeOfControl")
        typeOfInitSet = params.get("typeOfInitSet")
        N = params.get("N")  # Количество вершин для множества достижимости
        K = params.get("K")  # Разбиение для метода трапеций
        M = params.get("M")  # Численное решение задачи коши
        A = params.get("A")

        f1 = lambda coord11, coord12, uOptimal1: A[0][0] * coord11 + A[0][1] * coord12 + uOptimal1
        f2 = lambda coord21, coord22, uOptimal2: A[1][0] * coord21 + A[1][1] * coord22 + uOptimal2

        t0 = params.get("t0")
        T = params.get("T")

        sf_M0 = params.get("sf_M0")
        x0 = params.get("x0")

        sf_U = params.get("sf_U")
        u_opt_func = params.get("u_opt_func")

        c = params.get("c")
        y_bar = params.get("y_bar")
        J = params.get("J")
        psi_T_func = params.get("psi_T_func")

        A_transpose = np.transpose(A)
        expm_transpose = lambda t: linalg.expm(t * A_transpose)

        # ------------------------------------Старт------------------------------------
        print("---------------------------------Старт---------------------------------")
        solutions = {
            "typeOfControl": typeOfControl,
            "t0": t0,
            "T": T,
            "c": c,
            "y_bar": y_bar,
        }

        total_calcs_one = N + N + N * K + N + M
        x_T = coveredReachabilitySet(expm_transpose, sf_M0, sf_U, t0, T, N, K, total_calcs_one, update_pbar_signal)
        N = len(x_T)
        if typeOfFunctional == 0:
            # Если задача линейная, то:
            psi_T = psi_T_func(None)
            solutions["psi_T"] = psi_T

            if typeOfInitSet == 0 and typeOfControl != 0 \
                    and A[0][0] == 0 and A[0][1] == 0 and A[1][0] == 0 and A[1][1] == 0:
                w = params.get("w")
                x_T_i = np.zeros((len(w), 2))
                for i in range(len(w)):
                    x_T_i[i] = x0[0] + T * w[i]

                minn = J(x_T_i[0])
                index_one = 0
                for i in range(1, len(x_T_i)):
                    tmp = J(x_T_i[i])
                    if tmp < minn:
                        minn = tmp
                        index_one = i

                index_two = index_one
                for i in range(0, len(x_T_i)):
                    tmp = J(x_T_i[i])
                    if isDoublesEqual(minn, tmp) and i != index_one:
                        index_two = i
                        break
                print("x[{}] = {}".format(index_one, x_T_i[index_one]))
                print("x[{}] = {}".format(index_two, x_T_i[index_two]))

                fillsAreaX = [x0[0][0], x_T_i[index_one][0], x_T_i[index_two][0]]
                fillsAreaY = [x0[0][1], x_T_i[index_one][1], x_T_i[index_two][1]]
                solutions["fillsArea"] = [fillsAreaX, fillsAreaY]

                alpha_t_array = []

                # lambd = linalg.norm(x_T_opt - x_T_i[index_one]) / linalg.norm(x_T_opt - x_T_i[index_two])

                alpha_t_array.append(lambda _: 1)
                # alpha_t_array.append(lambda ttt: 1 if (ttt >= T / 4) else 0)
                # alpha_t_array.append(lambda _: 1 / 3)
                # alpha_t_array.append(lambda ttt: np.sin(2.2 * np.pi * ttt / T))

                u_opt_array = []
                for i in range(len(alpha_t_array)):
                    u_opt_func = lambda t: alpha_t_array[i](t) * w[index_one] + (1 - alpha_t_array[i](t)) * w[index_two]
                    u_opt, psi = optimalControlForAZeroMatrix(expm_transpose, u_opt_func, t0, T, psi_T, N, K, M,
                                                              update_pbar_signal)
                    u_opt_array.append(u_opt)

                optimalTrajectories = []
                for i in range(len(u_opt_array)):
                    print("i = ", i)
                    x = directEulerSolve(f1, f2, t0, T, x0[0], u_opt_array[i])
                    optimalTrajectories.append(x)
                solutions["optimalTrajectories"] = optimalTrajectories
                solutions["x"] = optimalTrajectories[0]

                x_T_opt = x_T_i[index_one]
                solutions["x_T_opt"] = x_T_opt
                minJ = J(x_T_i[index_one])
                solutions["minJ"] = minJ

                solutions["psi_t0"] = psi_T

                x_T_opt_line = np.array(
                    [[x_T_i[index_one][0], x_T_i[index_two][0]], [x_T_i[index_one][1], x_T_i[index_two][1]]])
                solutions["x_T_opt_line"] = x_T_opt_line

            else:
                minJ, x_T_opt = minimizeFunctionalForLinearProblem(J, x_T)
                u_opt, psi = optimalControl(expm_transpose, u_opt_func, t0, T, psi_T, N, K, M, update_pbar_signal)

                psi_t0 = psi[0]
                solutions["psi_t0"] = psi_t0

                # ЛИНЕЙНЫЙ ФУНКЦИОНАЛ
                # Находим координаты нашего отрезка оптимальных решений
                if isDoublesEqual(c[0], 0) or isDoublesEqual(c[1], 0):
                    x_T_opt_line = getTwoDifferentMinimumsInsteadOfFirstMinOfFunctional(J, x_T, minJ, x_T_opt)
                    solutions["x_T_opt_line"] = x_T_opt_line
                # Пройтись по x0 и найти максимальное скалярное произведение psi_t0 и x0[i] => x0_opt
                x0_opt = findOptimalCoordsOfM0(psi_t0, x0)
                x = directEulerSolve(f1, f2, t0, T, x0_opt, u_opt)
                solutions["x"] = x
                len1 = len(x[0])
                x_T_opt = [x[0][len1 - 1], x[1][len1 - 1]]
                solutions["x_T_opt"] = x_T_opt
                solutions["minJ"] = J(np.array(x_T_opt))

        else:
            # Если задача квадратичная, то:
            minJ, x_T_opt, x_opt_one, x_opt_two = minimizeQuadraticFunctional(J, x_T, y_bar)

            psi_T = psi_T_func(x_T_opt)
            solutions["psi_T"] = psi_T

            # особый случай квадратичной
            if typeOfInitSet == 0 and typeOfControl != 0 \
                    and A[0][0] == 0 and A[0][1] == 0 and A[1][0] == 0 and A[1][1] == 0:
                w = params.get("w")
                x_T_i = np.zeros((len(w), 2))
                for i in range(len(w)):
                    x_T_i[i] = x0[0] + T * w[i]

                index_one = 0
                index_two = 0
                for i in range(len(x_T_i)):
                    if isDoubleArraysEqual(x_opt_one, x_T_i[i]):
                        index_one = i
                    elif isDoubleArraysEqual(x_opt_two, x_T_i[i]):
                        index_two = i

                if index_one > index_two:
                    index_one, index_two = index_two, index_one

                x2 = (x_T_opt[0] + x_T_opt[1] + x0[0][0] - x0[0][1]) / 2
                y2 = x2 - x0[0][0] + x0[0][1]

                x3 = (x_T_opt[0] - x_T_opt[1] + x0[0][0] + x0[0][1]) / 2
                y3 = x3 - x_T_opt[0] + x_T_opt[1]

                fillsAreaX = [x0[0][0], x2, x_T_opt[0], x3]
                fillsAreaY = [x0[0][1], y2, x_T_opt[1], y3]
                solutions["fillsArea"] = [fillsAreaX, fillsAreaY]

                lambd = linalg.norm(x_T_opt - x_T_i[index_one]) / linalg.norm(x_T_opt - x_T_i[index_two])

                alpha_t_array = []

                alpha_t_array.append(lambda _: 1 / (1 + lambd))
                # alpha_t_array.append(lambda ttt: (1 - ttt) if (ttt <= 1) else 0)
                # alpha_t_array.append(lambda ttt: 0 if (ttt <= 3 / 2) else 1)
                # alpha_t_array.append(lambda ttt: abs(ttt - 1) ** 3)
                u_opt_array = []
                for i in range(len(alpha_t_array)):
                    u_opt_func = lambda t: alpha_t_array[i](t) * w[index_one] + (1 - alpha_t_array[i](t)) * w[index_two]
                    u_opt, psi = optimalControlForAZeroMatrix(expm_transpose, u_opt_func, t0, T, psi_T, N, K, M,
                                                              update_pbar_signal)
                    u_opt_array.append(u_opt)

                optimalTrajectories = []
                for i in range(len(u_opt_array)):
                    print("i = ", i)
                    x = backEulerSolve(f1, f2, t0, T, x_T_opt, u_opt_array[i])
                    optimalTrajectories.append(x)
                solutions["optimalTrajectories"] = optimalTrajectories
                solutions["x"] = optimalTrajectories[0]
            else:

                u_opt, psi = optimalControl(expm_transpose, u_opt_func, t0, T, psi_T, N, K, M, update_pbar_signal)
                x = backEulerSolve(f1, f2, t0, T, x_T_opt, u_opt)
                solutions["x"] = x

            psi_t0 = psi[0]
            solutions["psi_t0"] = psi_t0

            solutions["x_T_opt"] = x_T_opt
            solutions["minJ"] = J(x_T_opt)

        print("-------------------------------Завершение-------------------------------")
        # ----------------------------------Завершение----------------------------------

        # print("minJ = {}".format(minJ))
        # print("x_T_opt = {}".format(x_T_opt))
        # print("psi_t0 = {}".format(psi_t0))
        print("psi_T = {}".format(psi_T))
        print("\n\n")

        plot_x01 = np.zeros(len(x0) + 1)
        plot_x02 = np.zeros(len(x0) + 1)
        for i in range(len(x0)):
            plot_x01[i], plot_x02[i] = x0[i][0], x0[i][1]
        plot_x01[len(x0)], plot_x02[len(x0)] = x0[0][0], x0[0][1]
        solutions["x0"] = [plot_x01, plot_x02]

        plot_x_T1 = np.zeros(N + 1)
        plot_x_T2 = np.zeros(N + 1)
        for i in range(N):
            plot_x_T1[i], plot_x_T2[i] = x_T[i][0], x_T[i][1]
        plot_x_T1[N], plot_x_T2[N] = x_T[0][0], x_T[0][1]
        solutions["x_T"] = [plot_x_T1, plot_x_T2]

        plot_u1 = np.zeros(M)
        plot_u2 = np.zeros(M)
        for i in range(M):
            plot_u1[i], plot_u2[i] = u_opt[i][0], u_opt[i][1]
        solutions["u_opt"] = [plot_u1, plot_u2]

        get_data_signal.emit(solutions)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
