import serial
from PyQt6.QtCore import QThread, pyqtSignal

class NeuroPyWorker(QThread):
    emit_signal = pyqtSignal(dict)  # UI로 데이터 전송용 시그널
    error_signal = pyqtSignal(str)  # 에러 메시지 전달용 시그널

    def __init__(self, port, baudRate=57600, timeout=1, parent=None):
        super().__init__(parent)
        self.port = port
        self.baudRate = baudRate
        self.timeout = timeout
        self.srl = None
        self.threadRun = False

    def run(self):
        """스레드 시작 시 실행될 메인 루프."""
        try:
            self.srl = serial.Serial(self.port, self.baudRate, timeout=self.timeout)
            print(f"{self.port}에 연결.")
        except serial.SerialException as e:
            self.error_signal.emit(f"시리얼 연결 실패: {e}")
            return

        self.threadRun = True

        while self.threadRun:
            try:
                data = self.read_packet()
                if data:
                    self.emit_signal.emit(data) # 새 데이터를 UI에 전달
            except Exception as e:
                self.error_signal.emit(f"데이터 읽기 실패: {e}")

        self.close_serial_port()

    def read_packet(self):
        """패킷을 읽고 파싱."""
        try:
            # 헤더(aa aa) 확인
            p1 = self.srl.read(1).hex()
            p2 = self.srl.read(1).hex()
            while p1 != 'aa' or p2 != 'aa':
                p1 = p2
                p2 = self.srl.read(1).hex()

            # 페이로드 길이와 데이터 읽기
            payload = []
            checksum = 0
            payloadLength = int(self.srl.read(1).hex(), 16)

            for _ in range(payloadLength):
                tempPacket = self.srl.read(1).hex()
                payload.append(tempPacket)
                checksum += int(tempPacket, 16)

            # 체크섬
            checksum = ~checksum & 0x000000ff
            if checksum != int(self.srl.read(1).hex(), 16):
                raise ValueError("체크섬 오류.")

            return self.parse_payload(payload)
        except Exception as e:
            raise RuntimeError(f"패킷 읽기 중 오류 발생: {e}")

    def parse_payload(self, payload):
        """데이터 파싱 및 딕셔너리로 반환."""
        data = {}
        i = 0

        try:
            while i < len(payload):
                code = payload[i]
                if code == '02':
                    i += 1
                    data['poorSignal'] = int(payload[i], 16)
                elif code == '04':
                    i += 1
                    data['attention'] = int(payload[i], 16)
                elif code == '05':
                    i += 1
                    data['meditation'] = int(payload[i], 16)
                elif code == '16':
                    i += 1
                    data['blinkStrength'] = int(payload[i], 16)
                elif code == '80':
                    i += 2
                    val0 = int(payload[i], 16)
                    i += 1
                    rawValue = val0 * 256 + int(payload[i], 16)
                    if rawValue > 32768:
                        rawValue -= 65536
                    #data['rawValue'] = rawValue
                elif code == '83':
                    i += 1
                    data.update(self.parse_eeg(payload, i))
                    i += 24  # 다음 코드로 이동
                i += 1
        except IndexError as e:
            raise ValueError(f"데이터 파싱 중 인덱스 오류: {e}")

        return data

    def parse_eeg(self, payload, index):
        """EEG 데이터를 파싱하여 딕셔너리로 반환."""
        eeg_data = {}
        labels = ['delta', 'theta', 'lowAlpha', 'highAlpha',
                  'lowBeta', 'highBeta', 'lowGamma', 'midGamma']

        try:
            for label in labels:
                val0 = int(payload[index], 16)
                val1 = int(payload[index + 1], 16)
                val2 = int(payload[index + 2], 16)
                eeg_data[label] = val0 * 65536 + val1 * 256 + val2
                index += 3
        except IndexError as e:
            raise ValueError(f"EEG 데이터 파싱 오류: {e}")

        return eeg_data

    def stop(self):
        self.threadRun = False  # 루프 중단
        self.quit()  # 이벤트 루프 종료
        self.wait()  # 스레드 종료 대기
        self.close_serial_port()  # 시리얼 포트 닫기

    def close_serial_port(self):
        if self.srl and self.srl.is_open:
            self.srl.close()
            print(f"{self.port} 포트가 닫혔습니다.")
