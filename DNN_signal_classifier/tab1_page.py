from PyQt6.QtWidgets import QWidget, QTableWidgetItem, QVBoxLayout
from PyQt6.QtGui import QPixmap
from PyQt6 import uic
import pyqtgraph as pg
from pubsub import pub

class Tab1Page(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi("UI/tab1_page.ui", self)
        self.init_ui()
        pub.subscribe(self.update_signal, 'signal_update')

    def init_ui(self):
        # pyqtgraph 그래프 설정
        self.graph_widget = pg.PlotWidget()
        self.graph_widget.setYRange(0, 100)
        self.graph_widget.setBackground('w')
        self.graph_widget.setMouseEnabled(x=False, y=False)
        self.graph_widget.getPlotItem().hideButtons()

        self.attention_bar = pg.BarGraphItem(x=[0.4], height=[0], width=0.3, brush='r')
        self.meditation_bar = pg.BarGraphItem(x=[1], height=[0], width=0.3, brush='g')
        self.graph_widget.addItem(self.attention_bar)
        self.graph_widget.addItem(self.meditation_bar)
        self.graph_widget.getPlotItem().getAxis('bottom').setTicks([[(0.4, 'Attention'), (1, 'Meditation')]])
        self.chart_widget = self.findChild(QWidget, "chartWidget")
        self.layout = QVBoxLayout(self.chart_widget)
        self.layout.addWidget(self.graph_widget)

    def update_signal(self, signal):
        for i in range(10):
            self.signal_table.setItem(0, i, QTableWidgetItem(str(signal[i])))

        self.attention_bar.setOpts(height=[signal[0]])
        self.meditation_bar.setOpts(height=[signal[1]])

