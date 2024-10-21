from PyQt6 import uic
from PyQt6.QtWidgets import QDialog, QMessageBox
import serial.tools.list_ports

class SettingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi("UI/connect_setting_dialog.ui", self)
        self.setWindowTitle("포트 설정")
        self.current_port = None

        self.btn_port_set.setEnabled(False)  # 초기 set 버튼 비활성화
        self.btn_port_set.clicked.connect(self.btn_fnc_port_set)

        self.device_list.itemSelectionChanged.connect(self.btn_fnc_set_btn_toggle)  # 선택 시 상태 변경
        self.btn_list_refresh.clicked.connect(self.btn_fnc_device_list_refresh)
        self.btn_fnc_device_list_refresh()

    def btn_fnc_set_btn_toggle(self):
        self.btn_port_set.setEnabled(True)
        self.current_port = self.device_list.currentItem().text().split(" - ")[0]
        print(f"선택된 포트: {self.current_port}")

    def btn_fnc_port_set(self):
        if self.current_port:
            self.accept()
        else:
            QMessageBox.warning(self, "알림", "포트를 선택해 주세요")

    def btn_fnc_device_list_refresh(self):
        self.device_list.clear()
        ports = serial.tools.list_ports.comports()
        if not ports:
            QMessageBox.information(self, "알림", "연결된 직렬 포트가 없습니다.")
        else:
            for port in ports:
                self.device_list.addItem(f"{port.device} - {port.description}")

    def get_selected_port(self):
        return self.current_port