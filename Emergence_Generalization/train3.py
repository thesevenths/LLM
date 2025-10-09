import os
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# ---- 环境设置 ----
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
print(f"Script directory: {script_dir}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ModularDataset(torch.utils.data.Dataset):
    def __init__(self, m=97, train=True, frac_train=0.5, mode='add'):
        self.m = m
        self.pairs = [(a, b) for a in range(m) for b in range(m)]
        random.shuffle(self.pairs)
        split = int(len(self.pairs) * frac_train)
        if train:
            self.pairs = self.pairs[:split]
        else:
            self.pairs = self.pairs[split:]
        self.inputs = torch.tensor(self.pairs, dtype=torch.long)
        if mode == 'add':
            self.labels = (self.inputs[:,0] + self.inputs[:,1]) % m
        elif mode == 'mul':
            self.labels = (self.inputs[:,0] * self.inputs[:,1]) % m
        else:
            raise ValueError("Unknown mode")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

class ModularNet(nn.Module):
    def __init__(self, m=97, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(m, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, m)

    def forward(self, x):
        a = x[:,0]
        b = x[:,1]
        ea = self.embed(a)
        eb = self.embed(b)
        h = torch.cat([ea, eb], dim=1)
        h2 = self.relu(self.fc1(h))
        out = self.fc2(h2)
        return out

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0
    total_loss = 0.0
    correct = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x) #logits.shape：(batch_size, m)，这里比如logits.shape=(128, 97)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total += x.size(0)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
    return total_loss / total, correct / total

def eval_one_epoch(model, loader, criterion):
    model.eval()
    total = 0
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total += x.size(0)
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
    return total_loss / total, correct / total

def detect_grokking_point(test_acc_list, threshold=0.9, min_epoch=10):
    for i in range(min_epoch, len(test_acc_list)):
        if test_acc_list[i] >= threshold and test_acc_list[i-1] < threshold:
            return i
    return None

def train_grokking_full(m=97, hidden_dim=128, epochs=500, lr=1e-3, weight_decay=0.0, seed=0, mode='add'):
    set_seed(seed)
    train_ds = ModularDataset(m=m, train=True, frac_train=0.5, mode=mode)
    test_ds = ModularDataset(m=m, train=False, frac_train=0.5, mode=mode)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

    model = ModularNet(m=m, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'w_norm_fc1': [], 'delta_w_norm_fc1': [], 'grad_norm_fc1': [], 'cosine_fc1': []
    }

    prev_w_fc1 = None
    prev_delta_fc1 = None

    t0 = time.time()
    for ep in range(1, epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion)
        vl, va = eval_one_epoch(model, test_loader, criterion)

        history['train_loss'].append(tl)
        history['train_acc'].append(ta)
        history['test_loss'].append(vl)
        history['test_acc'].append(va)

        # 记录 fc1 权重 norm
        w_fc1 = model.fc1.weight.detach().cpu().numpy()
        w_norm = np.linalg.norm(w_fc1)
        history['w_norm_fc1'].append(w_norm)

        # 计算 delta & cosine
        if prev_w_fc1 is not None:
            delta = w_fc1 - prev_w_fc1
            delta_norm = np.linalg.norm(delta)
            history['delta_w_norm_fc1'].append(delta_norm)
            if prev_delta_fc1 is not None:
                cos_sim = np.dot(delta.flatten(), prev_delta_fc1.flatten()) / (
                    (np.linalg.norm(delta) * np.linalg.norm(prev_delta_fc1)) + 1e-12
                )
            else:
                cos_sim = 0.0
            history['cosine_fc1'].append(cos_sim)
            prev_delta_fc1 = delta.copy()
        else:
            history['delta_w_norm_fc1'].append(0.0)
            history['cosine_fc1'].append(0.0)
            prev_delta_fc1 = None

        prev_w_fc1 = w_fc1.copy()

        # 梯度 norm
        grad_norm = 0.0
        if model.fc1.weight.grad is not None:
            grad_norm = float(model.fc1.weight.grad.detach().cpu().norm().item())
        history['grad_norm_fc1'].append(grad_norm)

        if ep % 100 == 0 or ep == 1:
            print(f"Ep {ep}: train_acc={ta:.4f}, test_acc={va:.4f}, w_norm={w_norm:.3f}, grad_norm={grad_norm:.3f}, cos={history['cosine_fc1'][-1]:.3f}")

    elapsed = time.time() - t0
    history['elapsed'] = elapsed
    print("Total elapsed:", elapsed)

    grok_pt = detect_grokking_point(history['test_acc'], threshold=0.9, min_epoch=10)
    print("Grokking point:", grok_pt)

    # 保存
    model_path = os.path.join(script_dir, f"grok_mod_full_m{m}_hd{hidden_dim}_lr{lr}_wd{weight_decay}.pt")
    torch.save(model.state_dict(), model_path)
    hist_path = os.path.join(script_dir, f"grok_mod_full_m{m}_hd{hidden_dim}_lr{lr}_wd{weight_decay}_hist.json")
    with open(hist_path, 'w') as f:
        json.dump(history, f)
    print("Saved history to", hist_path)

    # 可视化
    epochs_range = list(range(1, epochs + 1))
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    axs[0,0].plot(epochs_range, history['train_acc'], label='train_acc')
    axs[0,0].plot(epochs_range, history['test_acc'], label='test_acc')
    if grok_pt:
        axs[0,0].axvline(grok_pt, color='red', linestyle='--', label='grok_pt')
    axs[0,0].legend()
    axs[0,0].set_title('Accuracy')

    axs[0,1].plot(epochs_range, history['w_norm_fc1'], label='w_norm_fc1')
    axs[0,1].legend()
    axs[0,1].set_title('Weight Norm (fc1)')

    axs[1,0].plot(epochs_range, history['delta_w_norm_fc1'], label='delta_w_norm_fc1')
    axs[1,0].plot(epochs_range, history['grad_norm_fc1'], label='grad_norm_fc1')
    axs[1,0].legend()
    axs[1,0].set_title('Δ Weight Norm & Grad Norm')

    gap = np.array(history['train_acc']) - np.array(history['test_acc'])
    axs[1,1].plot(epochs_range, gap, label='train-test gap')
    axs[1,1].legend()
    axs[1,1].set_title('Generalization Gap')

    axs[2,0].plot(epochs_range, history['cosine_fc1'], label='cosine_fc1')
    axs[2,0].legend()
    axs[2,0].set_title('Cosine of Delta Direction Change')

    axs[2,1].axis('off')

    fig.tight_layout()
    fig_path = os.path.join(script_dir, f"grok_mod_full_m{m}_hd{hidden_dim}_lr{lr}_wd{weight_decay}_fig.png")
    plt.savefig(fig_path)
    print("Saved figure to", fig_path)
    plt.close()

    return history, grok_pt

if __name__ == '__main__':
    dims = [32, 64, 128]
    lrs = [1e-3, 5e-4]
    wds = [0.0, 1e-4]
    all_hist = {}
    grok_points = {}
    for hd in dims:
        for lr in lrs:
            for wd in wds:
                print("=== Running hd =", hd, "lr =", lr, "wd =", wd)
                hist, grok_pt = train_grokking_full(m=97, hidden_dim=hd, epochs=500, lr=lr, weight_decay=wd, seed=0, mode='add')
                key = f"m{97}_hd{hd}_lr{lr}_wd{wd}"
                all_hist[key] = hist
                grok_points[key] = grok_pt
    all_path = os.path.join(script_dir, "grokking_all_full_hist.json")
    with open(all_path, 'w') as f:
        json.dump({'hist': all_hist, 'grok_pt': grok_points}, f)
    print("Saved all histories to", all_path)