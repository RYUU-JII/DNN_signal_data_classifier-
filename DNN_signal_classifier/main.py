import sys
import warnings
import eeg_signal
from pubsub import pub
from tab1_page import Tab1Page
from tab2_page import Tab2Page

warnings.filterwarnings("ignore", category=DeprecationWarning)
from signal_dialog import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap
from PyQt6 import uic
import numpy as np

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("UI/MainWindow.ui", self)
        self.init_ui()
        self.EEG_SIG = None
        self.port_name = "COM3"
        self.latest_data = {}
        self.signal_np = np.zeros(11)
        self.sig_run = False
        self.data_keys = [
            'attention', 'meditation', 'delta', 'theta',
            'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta',
            'lowGamma', 'midGamma', 'poorSignal'
        ]

    def init_ui(self):
        self.tabs = self.findChild(QTabWidget, "tabWidget")
        self.tab1 = Tab1Page()
        self.tab2 = Tab2Page()
        self.tabs.addTab(self.tab1, "신호 모니터")
        self.tabs.addTab(self.tab2, "데이터 수집")

        self.action_connect.triggered.connect(self.port_setting)
        self.btn_signal_connect.clicked.connect(self.btn_fnc_signal_connect)
        self.btn_signal_stop.clicked.connect(self.btn_fnc_signal_stop)

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.signal_handle)

        self.signal_image = QPixmap('Image/UI_image/signal_loss.png')
        self.signal_strength.setPixmap(self.signal_image)

    def port_setting(self):
        dialog = SettingDialog(self)
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            self.port_name = dialog.get_selected_port()
            print(f"포트: {self.port_name}")
            self.lineEdit_port.setText(self.port_name)
        else:
            print("연결 취소")

    def show_error(self, message):  # 에러 출력
        # QMessageBox.critical(self, "Error", message)
        print("에러:", message)

    def btn_fnc_signal_connect(self):
        if not self.EEG_SIG:
            try:
                self.EEG_SIG = eeg_signal.NeuroPyWorker(port=self.port_name)
                self.EEG_SIG.emit_signal.connect(self.store_latest_data) # 신호 수신시 실행
                self.EEG_SIG.error_signal.connect(self.show_error) # 에러 발생시 실행
                self.EEG_SIG.start()
                self.sig_run = True
                self.tab2.toggle_run_sig(self.sig_run)
                self.update_timer.start(1000)

            except Exception as e:
                self.show_error(f"연결 실패: {str(e)}")

    def btn_fnc_signal_stop(self):
        if self.EEG_SIG:
            self.EEG_SIG.stop()
            self.EEG_SIG = None
            self.update_timer.stop()
            self.sig_run = False
            self.tab2.toggle_run_sig(self.sig_run)
            self.signal_strength_change(200)

    def store_latest_data(self, data):
        """스레드에서 보내는 신호를 임시로 저장함, 동기화 오류를 피하기 위해 사용"""
        self.latest_data = data

    def signal_handle(self):
        """타이머가 작동되는 주기에 맞춰 신호 관리. self.latest_data를 참조하는 형태"""
        for index, key in enumerate(self.data_keys):
            self.signal_np[index] = self.latest_data.get(key, 0)
        pub.sendMessage('signal_update', signal = self.signal_np)
        self.signal_strength_change(self.signal_np[10])

    def signal_strength_change(self, poorSignal):
        # poorSignal은 0~200 값을 가지며, 0일 때 신호가 가장 안정적이나, 직관성을 위해 조정함
        scaled_value = int(100 - (poorSignal / 2))
        self.signal_str_label.setText(str(scaled_value))

        if scaled_value == 0:
            self.signal_image = QPixmap('Image/UI_image/signal_loss.png')
        elif scaled_value == 100:
            self.signal_image = QPixmap('Image/UI_image/signal_full.png')
        else:
            self.signal_image = QPixmap('Image/UI_image/signal_weak.png')
        self.signal_strength.setPixmap(self.signal_image)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())
