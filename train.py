from Bio import SeqIO
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import os

# 固定序列长度
MAX_LEN = 100
BASE_DICT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# 独热编码函数
def one_hot_encode(seq, max_len=MAX_LEN):
    encoded = np.zeros((max_len, 4))
    for i, base in enumerate(seq[:max_len]):
        if base in BASE_DICT:
            encoded[i, BASE_DICT[base]] = 1
    return encoded

# 读取FASTA文件并编码
def load_fasta_as_dataset(fasta_path, label):
    sequences, labels = [], []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).upper()
        encoded = one_hot_encode(seq)
        sequences.append(encoded)
        labels.append(label)
    return sequences, labels

# 文件路径和标签（正例为1，负例为0）
base_path = "D:/24252/生物物理/cnn"
file_label_map = {
    os.path.join(base_path, "promoter_train_pos.fasta"): 1,
    os.path.join(base_path, "promoter_train_neg.fasta"): 0,
    os.path.join(base_path, "promoter_valid_pos.fasta"): 1,
    os.path.join(base_path, "promoter_valid_neg.fasta"): 0,
}

# 加载数据
X_all, y_all = [], []
for path, label in file_label_map.items():
    if os.path.exists(path):
        X_part, y_part = load_fasta_as_dataset(path, label)
        X_all.extend(X_part)
        y_all.extend(y_part)
    else:
        print(f"文件未找到: {path}")

X_tensor = torch.tensor(np.array(X_all), dtype=torch.float32)
y_tensor = torch.tensor(np.array(y_all), dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)

# 划分 80% 训练集，20% 测试集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 定义简单CNN模型
class SimpleDNA_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool1d(2)
        conv_output_len = (MAX_LEN - 5 + 1) // 2
        self.fc1 = nn.Linear(16 * conv_output_len, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

# 模型训练（CPU）
model = SimpleDNA_CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for X_batch, y_batch in train_loader:
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 测试集评估
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
print(f"✅ 测试集准确率: {correct / total:.2%}")

# 保存模型
save_dir = os.path.join("D:/24252/生物物理", "cnn")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "simple_dna_cnn.pt")
torch.save(model.state_dict(), save_path)
print(f"✅ 模型已保存到：{save_path}")

