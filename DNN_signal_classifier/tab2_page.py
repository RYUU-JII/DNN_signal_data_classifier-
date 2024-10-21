import numpy as np
from PyQt6 import uic
import pickle
import os
from PyQt6.QtWidgets import QFileDialog, QWidget, QMessageBox
from PyQt6.QtGui import QPixmap
import random
from pubsub import pub

class Tab2Page(QWidget):
    BUFFER_LIMIT = 100

    def __init__(self):
        super().__init__()
        uic.loadUi("UI/tab2_page.ui", self)
        pub.subscribe(self.update_signal, 'signal_update')

        # 이미지 로드 경로 설정
        self.image_dir_red = "Image/Model/red"
        self.image_dir_green = "Image/Model/green"
        self.load_image()

        # UI 설정
        self.image_label.setPixmap(QPixmap("Image/UI_image/brain.png"))
        self.image_duration = 0
        self.label_state = np.array(['RED', 'GREEN'])
        self.label = None

        # 데이터 버퍼 초기화
        self.file_path = None
        self.data_buffer = np.empty((0, 10))
        self.label_buffer = []  # 라벨 버퍼를 1D 리스트로 설정
        self.run = False
        self.sig_run = False

        # 버튼 이벤트 연결
        self.btn_open.clicked.connect(self.open_file)
        self.btn_create_new.clicked.connect(self.create_new_pickle)
        self.btn_read.clicked.connect(self.read_data)
        self.btn_toggle_run.clicked.connect(self.toggle_run_correcting)

    def read_data(self):
        """피클 파일 읽기"""
        try:
            with open(self.file_path, 'rb') as f:
                data = pickle.load(f)
            X = np.array(data.get("X", []))
            Y = np.array(data.get("Y", []))
            print(X.shape)
            print(Y.shape)
            print("사이즈", X.size, Y.size)
        except Exception as e:
            print(f"오류: {e}")

    def load_image(self):
        """이미지 파일 로드"""
        self.images_red = [f for f in os.listdir(self.image_dir_red)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.images_green = [f for f in os.listdir(self.image_dir_green)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def toggle_run_sig(self, state):
        self.sig_run = state

    def update_signal(self, signal):
        """신호 업데이트 및 데이터 버퍼에 추가"""
        if not self.run:
            return  # 학습이 진행 중이 아니면 패스

        poorSignal = signal[10]

        if poorSignal == 0:
            data = np.array([signal[:10]])  # 10개의 특성 값
            lbl = 0 if self.label == 'RED' else 1  # 단일 값 라벨

            if self.image_duration <= 0:
                self.image_duration = random.randint(10, 15)
                self.label = random.choice(self.label_state)

                image_path = (self.image_dir_red if self.label == 'RED'
                              else self.image_dir_green)
                image = os.path.join(image_path, random.choice(os.listdir(image_path)))
                self.image_label.setPixmap(QPixmap(image))

            # 데이터와 라벨 버퍼에 추가
            self.data_buffer = np.vstack([self.data_buffer, data])
            self.label_buffer.append(lbl)  # 1D 리스트에 라벨 추가
            print("lbl:", lbl, self.label)

        self.image_duration -= 1

    def toggle_run_correcting(self):
        """학습 실행/중단 토글"""
        if not self.file_path:
            self.path_not_found()
            return
        elif not self.sig_run:
            QMessageBox.warning(None, "알림", "신호 없음.")
            return

        if not self.run:
            self.run = True
            self.btn_toggle_run.setText("실행중")
        else:
            self.run = False
            self.btn_toggle_run.setText("시작")
            self.image_label.setPixmap(QPixmap("Image/UI_image/brain.png"))
            self.image_duration = 0
            self.label = None
            print("학습 중단: 데이터 저장 중...")
            self.save_data()

    def open_file(self):
        """파일 열기 다이얼로그"""
        if self.run:
            self.now_running()
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "파일 열기", "pickle/data_pkl", "피클 파일 (*.pkl);;모든 파일 (*)"
        )

        if file_path:
            self.file_path = file_path
            file_name = os.path.basename(self.file_path)
            self.current_pkl.setText(file_name)
            print(f"선택된 파일 이름: {file_name}")

    def create_new_pickle(self):
        """새로운 피클 파일 생성"""
        if self.run:
            self.now_running()
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "새 피클 파일 생성", "pickle/data_pkl", "피클 파일 (*.pkl)"
        )

        if file_path:
            if not file_path.endswith(".pkl"):
                file_path += ".pkl"

            initial_data = {"X": np.empty((0, 10)), "Y": np.empty((0,))}

            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(initial_data, f)
                self.file_path = file_path
                print(f"새로운 피클 파일 생성: {self.file_path}")
            except Exception as e:
                print(f"파일 생성 중 오류 발생: {e}")

    def now_running(self):
        QMessageBox.warning(None, "경고", "학습 진행 중!")

    def path_not_found(self):
        QMessageBox.warning(None, "알림", "경로가 없음.")

    def save_data(self):
        """데이터 저장"""
        if not self.file_path:
            self.path_not_found()
            return

        try:
            # 파일이 존재하면 로드, 없으면 빈 데이터 생성
            if os.path.exists(self.file_path):
                with open(self.file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                data = {"X": np.empty((0, 10)), "Y": np.empty((0,))}

            # 새로운 데이터 추가
            new_X = np.array(self.data_buffer)
            new_Y = np.array(self.label_buffer).flatten()

            # 병합 전 데이터 길이 확인 및 조정
            if data["X"].shape[0] != data["Y"].shape[0]:
                min_length = min(data["X"].shape[0], data["Y"].shape[0])
                data["X"] = data["X"][:min_length]
                data["Y"] = data["Y"][:min_length]

            # 데이터 병합
            data["X"] = np.vstack([data["X"], new_X])
            data["Y"] = np.concatenate([data["Y"], new_Y])

            # 파일 저장
            with open(self.file_path, 'wb') as f:
                pickle.dump(data, f)

            # 버퍼 초기화
            self.data_buffer = np.empty((0, 10))
            self.label_buffer = []
            print(f"데이터 저장 완료: {self.file_path}")

        except Exception as e:
            print(f"데이터 저장 중 오류 발생: {e}")

