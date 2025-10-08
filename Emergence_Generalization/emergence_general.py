import os
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ---- 环境与目录设置 ----
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
print(f"Script directory: {script_dir}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- 辅助函数：设定随机种子 ----
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---- 模型基类接口 ----
class EmergenceModel(nn.Module):
    """
    这个类是一个接口 base class，期望子类实现：
     - forward：返回 (out_main, out_aux)
     - 可选子模块／结构由子类定义
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        返回主任务输出和辅助任务输出（两个张量）。
        """
        raise NotImplementedError

# ---- 训练与评估函数 ----
def train_one_epoch(model, train_loader, optimizer, criterion_main, criterion_aux, alpha_aux=1.0):
    model.train()
    total_loss = 0.0
    total = 0
    correct_main = 0
    correct_aux = 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out_main, out_aux = model(x)
        loss_main = criterion_main(out_main, y)
        y_aux = get_aux_labels(y)  # 这个函数在你的实验脚本中定义
        loss_aux = criterion_aux(out_aux, y_aux)
        loss = loss_main + alpha_aux * loss_aux
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size
        _, pm = torch.max(out_main, dim=1)
        correct_main += (pm == y).sum().item()
        _, pa = torch.max(out_aux, dim=1)
        correct_aux += (pa == y_aux).sum().item()

    return total_loss / total, correct_main / total, correct_aux / total

def eval_model(model, data_loader, criterion_main, criterion_aux, alpha_aux=1.0):
    model.eval()
    total_loss = 0.0
    total = 0
    correct_main = 0
    correct_aux = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            out_main, out_aux = model(x)
            loss_main = criterion_main(out_main, y)
            y_aux = get_aux_labels(y)
            loss_aux = criterion_aux(out_aux, y_aux)
            loss = loss_main + alpha_aux * loss_aux

            bs = x.size(0)
            total_loss += loss.item() * bs
            total += bs
            _, pm = torch.max(out_main, dim=1)
            correct_main += (pm == y).sum().item()
            _, pa = torch.max(out_aux, dim=1)
            correct_aux += (pa == y_aux).sum().item()

    return total_loss / total, correct_main / total, correct_aux / total

# ---- 主实验驱动函数 ----
def run_one_setting(model_cls, model_kwargs,
                    train_loader, test_loader,
                    optimizer_cls, optimizer_kwargs,
                    criterion_main, criterion_aux,
                    alpha_aux,
                    epochs,
                    seed=0,
                    save_prefix="exp"):
    """
    为一个超参数组合跑训练 + 测试，记录历史，保存模型与图表。
    返回 history dict。
    """
    set_seed(seed)
    model = model_cls(**model_kwargs).to(device)
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

    history = {
        'train_loss': [], 'train_acc_main': [], 'train_acc_aux': [],
        'test_loss': [], 'test_acc_main': [], 'test_acc_aux': []
    }

    t0 = time.time()
    for ep in range(1, epochs+1):
        tr_l, tr_am, tr_aa = train_one_epoch(model, train_loader, optimizer,
                                             criterion_main, criterion_aux, alpha_aux)
        te_l, te_am, te_aa = eval_model(model, test_loader, criterion_main, criterion_aux, alpha_aux)
        print(f"[{save_prefix}] ep={ep} train_l={tr_l:.4f} tr_main={tr_am:.4f} tr_aux={tr_aa:.4f} | "
              f"te_main={te_am:.4f} te_aux={te_aa:.4f}")
        history['train_loss'].append(tr_l)
        history['train_acc_main'].append(tr_am)
        history['train_acc_aux'].append(tr_aa)
        history['test_loss'].append(te_l)
        history['test_acc_main'].append(te_am)
        history['test_acc_aux'].append(te_aa)
    elapsed = time.time() - t0
    print(f"[{save_prefix}] elapsed time = {elapsed:.2f} s")
    history['elapsed'] = elapsed

    # 保存模型参数
    model_path = os.path.join(script_dir, f"{save_prefix}_model.pt")
    torch.save(model.state_dict(), model_path)
    print("Saved model to", model_path)
    # 保存 history
    hist_path = os.path.join(script_dir, f"{save_prefix}_history.json")
    with open(hist_path, 'w') as f:
        json.dump(history, f)
    print("Saved history to", hist_path)

    # 画图：主任务 & 辅助任务准确率
    plt.figure(figsize=(6, 4))
    plt.plot(history['train_acc_main'], label='train_main')
    plt.plot(history['test_acc_main'], label='test_main')
    plt.plot(history['train_acc_aux'], label='train_aux')
    plt.plot(history['test_acc_aux'], label='test_aux')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    fig_path = os.path.join(script_dir, f"{save_prefix}_acc.png")
    plt.savefig(fig_path)
    print("Saved acc figure to", fig_path)
    plt.close()

    return history

def sweep_experiments(param_grid, base_name,
                      model_cls, model_base_kwargs,
                      optimizer_cls, optimizer_base_kwargs,
                      train_loader, test_loader,
                      criterion_main, criterion_aux,
                      alpha_aux,
                      epochs,
                      seeds=[0]):
    """
    扫描多个超参数组合（param_grid 是 dict，每个 key 对应一个可扫参数列表）。
    param_grid 例如： {'hidden_dim': [32, 64, 128], 'lr': [1e-3, 5e-4]}
    base_name 是目录前缀标识（如 “cifar_exp”）。
    其他参数为模型类、优化器类、数据集、损失等。
    """
    all_hist = {}
    import itertools
    keys = list(param_grid.keys())
    for vals in itertools.product(*(param_grid[k] for k in keys)):
        cfg = dict(zip(keys, vals))
        name = base_name + "_" + "_".join(f"{k}{v}" for k, v in cfg.items())
        # 构造模型 / 优化器参数
        mk = dict(model_base_kwargs)
        ok = dict(optimizer_base_kwargs)
        # 假设 model_kwargs 接受 hidden_dim, …；optimizer_kwargs 接受 lr 等
        mk.update({k: cfg[k] for k in keys if k in model_base_kwargs})
        ok.update({k: cfg[k] for k in keys if k in optimizer_base_kwargs})
        for seed in seeds:
            run_name = f"{name}_s{seed}"
            print("=== Running", run_name)
            hist = run_one_setting(model_cls, mk,
                                   train_loader, test_loader,
                                   optimizer_cls, ok,
                                   criterion_main, criterion_aux,
                                   alpha_aux,
                                   epochs,
                                   seed,
                                   save_prefix=run_name)
            all_hist[run_name] = hist
    # 最后保存所有 hist
    all_hist_path = os.path.join(script_dir, f"{base_name}_all_hist.json")
    with open(all_hist_path, 'w') as f:
        json.dump(all_hist, f)
    print("Saved all hist to", all_hist_path)
    return all_hist

# ---- 你在脚本中需要补充 / 实现的部分 ----

def get_aux_labels(y):
    """
    给定主任务标签 y（Tensor），返回辅助任务标签（Tensor）；
    例如 y_aux = y % 2（偶/奇分类），或者更复杂的变换。
    这个函数要在你自己的脚本里定义以符合你设计的辅助任务。
    """
    # 示例（偶奇分类）：
    return (y % 2).long()

# ---- 下边写一个示例 main，仅作为演示 ----
if __name__ == '__main__':
    # 示例用 MNIST 数据集 + 上面工具包来跑扫描实验
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # 数据
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=script_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=script_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    # 模型类示例（你也可以改写这个类或用自己的）：
    class SimpleEmerModel(EmergenceModel):
        def __init__(self, hidden_dim=64):
            super().__init__()
            self.fc1 = nn.Linear(28*28, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, 10)
            self.aux = nn.Linear(hidden_dim, 2)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            h = self.relu(self.fc1(x))
            o1 = self.fc2(h)
            o2 = self.aux(h)
            return o1, o2

    # 损失函数 & 优化器
    criterion_main = nn.CrossEntropyLoss()
    criterion_aux = nn.CrossEntropyLoss()
    optimizer_cls = optim.Adam
    optimizer_base_kwargs = {'lr': 1e-3}

    # 扫描参数
    param_grid = {
        'hidden_dim': [16, 32, 64, 128],
        'lr': [1e-3, 5e-4]
    }
    model_base_kwargs = {'hidden_dim': None}  # None 表示占位

    all_hist = sweep_experiments(param_grid, base_name="mnist_exp",
                                 model_cls=SimpleEmerModel, model_base_kwargs=model_base_kwargs,
                                 optimizer_cls=optimizer_cls, optimizer_base_kwargs=optimizer_base_kwargs,
                                 train_loader=train_loader, test_loader=test_loader,
                                 criterion_main=criterion_main, criterion_aux=criterion_aux,
                                 alpha_aux=0.5,
                                 epochs=10,
                                 seeds=[0,1])
    print("Done all experiments.")
