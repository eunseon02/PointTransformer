import matplotlib.pyplot as plt
import re

# txt 파일 경로
file_path = 'voxels_torch.txt'

# 손실 데이터를 저장할 리스트
epochs = []
train_loss = []
val_loss = []

# 파일 읽기
with open(file_path, 'r') as file:
    for line in file:
        # 에포크, Train Loss, Validation Loss 추출
        match = re.match(r'Epoch \[(\d+)/\d+\] - Train Loss: ([\d.]+), Validation Loss: ([\d.]+)', line)
        if match:
            epochs.append(int(match.group(1)))
            train_loss.append(float(match.group(2)))
            val_loss.append(float(match.group(3)))

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('loss_plot.png')
