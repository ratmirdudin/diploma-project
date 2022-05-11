import matplotlib
from PyQt5.QtWidgets import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure

matplotlib.use('QT5Agg')


class MyNavigationToolbar(NavigationToolbar):
    NavigationToolbar.toolitems = (
        ('Home', 'Сбросить изменения', 'home', 'home'),
        ('Back', 'Назад', 'back', 'back'),
        ('Forward', 'Вперед', 'forward', 'forward'),
        (None, None, None, None),
        ('Pan', 'ЛКМ для перемещения, ПКМ для увеличения/уменьшения', 'move', 'pan'),
        ('Zoom', 'ЛКМ для увеличения, ПКМ уменьшения выделенной области', 'zoom_to_rect', 'zoom'),
        # ('Subplots', 'Настройка графиков', 'subplots', 'configure_subplots'),
        (None, None, None, None),
        ('Save', 'Сохранить график', 'filesave', 'save_figure'),
    )


class MplWidget(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        self.canvas = FigureCanvas(Figure())
        self.canvas.figure.tight_layout()
        self.canvas.figure.subplots_adjust(0.15, 0.15, 0.90, 0.90)  # left,bottom,right,top

        self.toolbar = MyNavigationToolbar(self.canvas, self)

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.toolbar)
        vertical_layout.addWidget(self.canvas)

        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)
