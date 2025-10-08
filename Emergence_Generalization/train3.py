import os
import time
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# ---- 环境与目录设置 ----
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
print(f"Script directory: {script_dir}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- 数据准备 ----
def get_mnist_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST(root=script_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(root=script_dir, train=False, download=True, transform=transform)
    # 为了加快实验，可以只取子集
    # train = Subset(train, list(range(10000)))
    # test = Subset(test, list(range(2000)))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# ---- 模型定义 ----
class SimpleNet(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(28*28, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 10)
        # 辅助任务：我们额外加一个分支从中间层预测偶数/奇数（binary task）
        self.aux = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.relu(self.fc1(x))
        out_main = self.fc2(h)
        out_aux = self.aux(h)
        return out_main, out_aux

# ---- 训练与评估 ----
def train_one_epoch(model, train_loader, optimizer, criterion_main, criterion_aux, alpha_aux=0.5):
    model.train()
    total_loss = 0.0
    correct_main = 0
    correct_aux = 0
    total = 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out_main, out_aux = model(x)
        loss_main = criterion_main(out_main, y)
        # 构造辅助标签：偶 / 奇数
        y_aux = (y % 2).long()
        loss_aux = criterion_aux(out_aux, y_aux)
        loss = loss_main + alpha_aux * loss_aux
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        # 计算主任务精度
        _, pred_main = torch.max(out_main, dim=1)
        correct_main += (pred_main == y).sum().item()
        # 计算辅助任务精度
        _, pred_aux = torch.max(out_aux, dim=1)
        correct_aux += (pred_aux == y_aux).sum().item()

    return total_loss / total, correct_main / total, correct_aux / total

def eval_model(model, data_loader, criterion_main, criterion_aux, alpha_aux=0.5):
    model.eval()
    total_loss = 0.0
    correct_main = 0
    correct_aux = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            out_main, out_aux = model(x)
            loss_main = criterion_main(out_main, y)
            y_aux = (y % 2).long()
            loss_aux = criterion_aux(out_aux, y_aux)
            loss = loss_main + alpha_aux * loss_aux
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            _, pred_main = torch.max(out_main, dim=1)
            correct_main += (pred_main == y).sum().item()
            _, pred_aux = torch.max(out_aux, dim=1)
            correct_aux += (pred_aux == y_aux).sum().item()
    return total_loss / total, correct_main / total, correct_aux / total

# ---- 主流程 ----
def run_experiment(hidden_dim=64, epochs=10, alpha_aux=0.5, lr=1e-3):
    train_loader, test_loader = get_mnist_dataloaders()
    model = SimpleNet(hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_main = nn.CrossEntropyLoss()
    criterion_aux = nn.CrossEntropyLoss()

    history = {
        'train_loss': [], 'train_acc_main': [], 'train_acc_aux': [],
        'test_loss': [], 'test_acc_main': [], 'test_acc_aux': []
    }

    start_time = time.time()
    for ep in range(1, epochs+1):
        tl, ta_main, ta_aux = train_one_epoch(model, train_loader, optimizer, criterion_main, criterion_aux, alpha_aux)
        vloss, vacc_main, vacc_aux = eval_model(model, test_loader, criterion_main, criterion_aux, alpha_aux)
        print(f"Ep {ep}: train_loss={tl:.4f}, train_main={ta_main:.4f}, train_aux={ta_aux:.4f} | test_main={vacc_main:.4f}, test_aux={vacc_aux:.4f}")
        history['train_loss'].append(tl)
        history['train_acc_main'].append(ta_main)
        history['train_acc_aux'].append(ta_aux)
        history['test_loss'].append(vloss)
        history['test_acc_main'].append(vacc_main)
        history['test_acc_aux'].append(vacc_aux)
    elapsed = time.time() - start_time
    print(f"Total training time: {elapsed:.2f} seconds")
    history['elapsed'] = elapsed

    # 保存模型
    model_path = os.path.join(script_dir, f"model_hd{hidden_dim}_ep{epochs}.pt")
    torch.save(model.state_dict(), model_path)
    print("Saved model to", model_path)
    # 保存历史记录
    hist_path = os.path.join(script_dir, f"history_hd{hidden_dim}_ep{epochs}.json")
    with open(hist_path, 'w') as f:
        json.dump(history, f)
    print("Saved history to", hist_path)

    # 画图保存
    # 主任务精度 + 辅助任务精度对比图
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_acc_aux'], label='train_aux')
    plt.plot(history['test_acc_aux'], label='test_aux')
    plt.plot(history['train_acc_main'], label='train_main')
    plt.plot(history['test_acc_main'], label='test_main')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    fig_path = os.path.join(script_dir, f"acc_curve_hd{hidden_dim}_ep{epochs}.png")
    plt.savefig(fig_path)
    print("Saved figure to", fig_path)
    plt.close()

    return history

if __name__ == '__main__':
    # 你可以改动 hidden_dim, epochs 做扫描实验
    # 比如 hidden_dim = 16,32,64,128；epochs = 5,10,20 等
    hds = [16, 32, 64, 128, 256]
    all_hist = {}
    for hd in hds:
        print("=== Experiment: hidden_dim =", hd)
        hist = run_experiment(hidden_dim=hd, epochs=10, alpha_aux=0.5)
        all_hist[hd] = hist
    # 最终把所有 hist 保存一下
    save_all = os.path.join(script_dir, "all_hist.json")
    with open(save_all, 'w') as f:
        json.dump({str(k): v for k, v in all_hist.items()}, f)
    print("Saved all histories to", save_all)
