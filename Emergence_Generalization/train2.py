import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time

# 指定要求片段
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
print(f"Script directory: {script_dir}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 固定随机种子
torch.manual_seed(42)
np.random.seed(42)

# 模数
P = 97

# One-hot编码函数
def one_hot_encode(data, num_classes=P):
    return np.eye(num_classes)[data.astype(int)]

# 生成数据集：a, b in [0, P-1], target = (a + b) % P
def generate_data(n_samples, is_train=True):
    if is_train:
        # 训练集：随机子集，覆盖~30%以促grokking
        indices = np.random.choice(P*P, n_samples, replace=False)
        a = indices // P
        b = indices % P
    else:
        # 验证/测试：全新随机
        a = np.random.randint(0, P, n_samples)
        b = np.random.randint(0, P, n_samples)
    # One-hot编码
    a_onehot = one_hot_encode(a)
    b_onehot = one_hot_encode(b)
    inputs = np.concatenate([a_onehot, b_onehot], axis=1).astype(np.float32)
    targets = ((a + b) % P).astype(np.int64)
    return inputs, targets

# 数据集大小（减小训练集）
n_train = 3000
n_val = 1000
n_test = 1000

train_inputs, train_targets = generate_data(n_train, is_train=True)
val_inputs, val_targets = generate_data(n_val, is_train=False)
test_inputs, test_targets = generate_data(n_test, is_train=False)

# DataLoader
train_dataset = TensorDataset(torch.from_numpy(train_inputs), torch.from_numpy(train_targets))
val_dataset = TensorDataset(torch.from_numpy(val_inputs), torch.from_numpy(val_targets))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# 简单MLP模型（输入dim=2*P）
class ModularNet(nn.Module):
    def __init__(self, input_dim=2*P, hidden_dim=512, num_classes=P):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ModularNet().to(device)
criterion = nn.CrossEntropyLoss()
# 加weight_decay
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

# 训练函数
def train_epoch(loader, model, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / len(loader), correct / total

# 评估函数
def evaluate(loader, model, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return total_loss / len(loader), correct / total

# 测试函数
def test(model, test_inputs, test_targets, device):
    model.eval()
    with torch.no_grad():
        test_inputs = torch.from_numpy(test_inputs).to(device)
        test_targets = torch.from_numpy(test_targets).to(device)
        outputs = model(test_inputs)
        _, predicted = outputs.max(1)
        correct = predicted.eq(test_targets).sum().item()
        return correct / len(test_targets)

# 开始计时
start_time = time.time()

# 训练循环（增epochs）
epochs = 5000
train_losses, train_accs = [], []
val_losses, val_accs = [], []
fc1_norms, fc2_norms = [], []  # 权重范数列表

for epoch in range(epochs):
    train_loss, train_acc = train_epoch(train_loader, model, optimizer, criterion, device)
    val_loss, val_acc = evaluate(val_loader, model, criterion, device)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # 每100 epoch记录权重范数（Frobenius norm）
    if (epoch + 1) % 100 == 0:
        fc1_norm = torch.norm(model.fc1.weight).item()
        fc2_norm = torch.norm(model.fc2.weight).item()
        fc1_norms.append(fc1_norm)
        fc2_norms.append(fc2_norm)
        print(f'Epoch {epoch+1}: Train Loss {train_loss:.4f}, Acc {train_acc:.4f} | Val Loss {val_loss:.4f}, Acc {val_acc:.4f} | FC1 Norm {fc1_norm:.4f}, FC2 Norm {fc2_norm:.4f}')

# 结束计时
end_time = time.time()
elapsed_time = end_time - start_time
print(f'Total training time: {elapsed_time:.2f} seconds')

# 测试泛化
test_acc = test(model, test_inputs, test_targets, device)
print(f'Test Accuracy (Generalization): {test_acc:.4f}')

# 保存模型
model_path = os.path.join(script_dir, 'model.pth')
torch.save(model.state_dict(), model_path)
print(f'Model saved to: {model_path}')

# 绘制acc/loss曲线
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.axhline(y=1.0, color='r', linestyle='--', label='Perfect Gen.')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curves (Emergence Visible)')

plt.tight_layout()
plot_path = os.path.join(script_dir, 'training_curves.png')
plt.savefig(plot_path)
plt.close()
print(f'Plot saved to: {plot_path}')

# 绘制权重范数变化曲线（每100 epoch）
epochs_sampled = list(range(100, epochs+1, 100))
plt.figure(figsize=(8, 5))
plt.plot(epochs_sampled, fc1_norms, label='FC1 Weight Norm')
plt.plot(epochs_sampled, fc2_norms, label='FC2 Weight Norm')
plt.xlabel('Epoch')
plt.ylabel('Frobenius Norm')
plt.legend()
plt.title('Hidden Layer Weight Norms Over Epochs (Emergence in Changes)')
weight_plot_path = os.path.join(script_dir, 'weight_norms.png')
plt.savefig(weight_plot_path)
plt.close()
print(f'Weight norms plot saved to: {weight_plot_path}')