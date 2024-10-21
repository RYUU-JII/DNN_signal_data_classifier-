import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import pickle
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 정규화
with open('pickle/data_pkl/converted.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']
Y = data['Y']

# Z-스코어 정규화
scaler_standard = StandardScaler()
X_standard = scaler_standard.fit_transform(X)

# 학습 및 검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X_standard, Y, test_size=0.2, random_state=42)

# 텐서 변환 및 DataLoader 생성
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                            torch.tensor(y_val, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

# 클래스 불균형 가중치 계산 및 손실 함수 정의
class_counts = np.bincount(Y)
class_weights = 1. / class_counts
weights = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=weights)

# 2. 모델 정의
class EnhancedSplitNet(nn.Module):
    def __init__(self, weight_factor=2.0, dropout1=0.2, dropout2=0.4):
        super(EnhancedSplitNet, self).__init__()
        self.subnet1 = nn.Sequential(
            nn.Linear(2, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(16, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 8),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout1)
        )
        self.subnet2 = nn.Sequential(
            nn.Linear(8, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout2)
        )
        self.fc = nn.Linear(8 + 16, 2)
        self.weight_factor = weight_factor

    def forward(self, x):
        x1 = x[:, :2]
        x2 = x[:, 2:]
        out1 = self.subnet1(x1)
        out2 = self.subnet2(x2)
        out = torch.cat((out1 * self.weight_factor, out2), dim=1)
        return self.fc(out)

# 모델 초기화 및 옵티마이저 설정
model = EnhancedSplitNet(weight_factor=10, dropout1=0.1, dropout2=0.3)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Early Stopping 클래스 정의
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

early_stopping = EarlyStopping(patience=10)

# 4. 학습 루프
epochs = 100
train_losses, val_losses, val_accuracies = [], [], []

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for X_batch, y_batch in train_loader:  # 배치 단위로 데이터를 가져옴
        outputs = model(X_batch)  # 순전파, X_batch를 모델에 입력하여 예측값을 계산
        loss = criterion(outputs, y_batch)  # 손실 계산: 예측값과 실제값 y_batch를 사용하여 손실 계산

        optimizer.zero_grad()  # 기울기 초기화: 이전 배치의 기울기 값을 모두 초기화
        loss.backward()  # 역전파: 손실을 기준으로 모델의 파라미터에 대한 기울기 계산
        optimizer.step()  # 파라미터 업데이트: optimizer가 계산된 기울기를 사용해 모델의 파라미터를 업데이트
        train_loss += loss.item()  # 현재 배치의 손실 값을 train_loss에 더해줌

    model.eval()
    val_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            val_outputs = model(X_batch)
            loss = criterion(val_outputs, y_batch)
            val_loss += loss.item()

            predictions = torch.argmax(val_outputs, dim=1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    val_accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_accuracy)

    print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss / len(train_loader):.4f}, '
          f'Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}, '
          f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

    scheduler.step(val_loss / len(val_loader))
    early_stopping(val_loss / len(val_loader))

    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

# 5. 손실 및 정확도 그래프 그리기
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Curve')
ax1.legend()

ax2.plot(val_accuracies, label='Validation Accuracy', color='g')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Validation Accuracy Curve')
ax2.legend()

plt.show()

# 6. 혼동 행렬과 성능 지표 시각화
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', ax=ax[0])
ax[0].set_title('Confusion Matrix')

ax[1].text(0.1, 0.8, f'Precision: {precision:.4f}', fontsize=12)
ax[1].text(0.1, 0.6, f'Recall: {recall:.4f}', fontsize=12)
ax[1].text(0.1, 0.4, f'F1-Score: {f1:.4f}', fontsize=12)
ax[1].axis('off')

plt.show()


