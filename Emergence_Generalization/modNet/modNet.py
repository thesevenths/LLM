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
    def __init__(self, m=97, train=True, frac_train=0.5, mode='add', pairs=None):
        """
        If `pairs` is provided it should already be the subset (list of (a,b))
        for this dataset (i.e. already split). Otherwise the dataset will
        construct and split pairs internally (legacy behavior).
        """
        self.m = m
        # If caller provided a pre-sliced pairs list, use it directly.
        if pairs is not None:
            self.pairs = list(pairs)
        else:
            self.pairs = [(a, b) for a in range(m) for b in range(m)]
            random.shuffle(self.pairs)
            split = int(len(self.pairs) * frac_train)
            if train:
                self.pairs = self.pairs[:split]
            else:
                self.pairs = self.pairs[split:]

        self.inputs = torch.tensor(self.pairs, dtype=torch.long)
        if mode == 'add':
            self.labels = (self.inputs[:, 0] + self.inputs[:, 1]) % m
        elif mode == 'mul':
            self.labels = (self.inputs[:, 0] * self.inputs[:, 1]) % m
        else:
            raise ValueError("Unknown mode")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

class ModularNet(nn.Module):
    def __init__(self, m=97, hidden_dim=128):
        super().__init__()
        # 模仿 transformers 的 embedding 层
        self.embed = nn.Embedding(m, hidden_dim) #直接查索引查表，不需要one-hot
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, m)

    def forward(self, x):
        a = x[:,0]
        b = x[:,1]
        ea = self.embed(a) # shape: (batch_size, hidden_dim), 每个输入数字转成hidden_dim
        eb = self.embed(b)
        h = torch.cat([ea, eb], dim=1)
        h2 = self.relu(self.fc1(h))
        out = self.fc2(h2)
        return out


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

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

def detect_grokking_point(
    test_acc_list,
    threshold=0.9,
    min_epoch=10,
    consecutive_epochs=3,      # 要求连续 N 轮达标
    smooth_window=None         # 滑动平均窗口大小（如 5），设为 None 则不平滑
):
    """
    Detect the epoch at which grokking occurs.

    Args:
        test_acc_list (list of float): Test accuracy per epoch.
        threshold (float): Accuracy threshold to consider as "generalized".
        min_epoch (int): Ignore epochs before this (avoid early noise).
        consecutive_epochs (int): Require this many consecutive epochs above threshold.
        smooth_window (int or None): If not None, apply moving average smoothing.

    Returns:
        int or None: Epoch index (1-based in caller's context) where grokking starts,
                     or None if not detected.
    """
    if len(test_acc_list) < min_epoch + consecutive_epochs:
        return None

    # 可选：滑动平均平滑
    if smooth_window is not None and smooth_window > 1:
        import numpy as np
        acc = np.array(test_acc_list)
        # 使用 'valid' 卷积实现滑动平均
        smoothed = np.convolve(acc, np.ones(smooth_window) / smooth_window, mode='same')
        # 边界用原始值填充（可选）
        smoothed[:smooth_window//2] = acc[:smooth_window//2]
        smoothed[-(smooth_window//2):] = acc[-(smooth_window//2):]
        acc_to_use = smoothed.tolist()
    else:
        acc_to_use = test_acc_list

    # 从 min_epoch 开始检查连续达标
    for i in range(min_epoch, len(acc_to_use) - consecutive_epochs + 1):
        # 检查从 i 开始的 consecutive_epochs 轮是否都 ≥ threshold
        if all(acc_to_use[j] >= threshold for j in range(i, i + consecutive_epochs)):
            # 确保前一轮（i-1）未达标（可选，保留跃升语义）
            if i == 0 or acc_to_use[i - 1] < threshold:
                return i  # 返回首次连续达标起始 epoch
            else:
                # 如果前一轮也达标，说明更早已 grok，继续向前找
                continue

    return None

def train_grokking_full(m=97, hidden_dim=128, epochs=500, lr=1e-3, weight_decay=0.0, seed=0, mode='add'):
    set_seed(seed)

    # Create all pairs once and split deterministically to avoid overlap/leakage.
    all_pairs = [(a, b) for a in range(m) for b in range(m)]
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * 0.5)
    train_pairs = all_pairs[:split_idx]
    test_pairs = all_pairs[split_idx:]

    train_ds = ModularDataset(m=m, mode=mode, pairs=train_pairs)
    test_ds = ModularDataset(m=m, mode=mode, pairs=test_pairs)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

    model = ModularNet(m=m, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # === 新增：用于计算 grad_cosine_sim 的梯度缓冲区 ===
    grad_buffer = []  # 存最近的梯度 (fc1.weight.grad)
    buffer_size = 5

    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'w_norm_fc1': [],
        'delta_w_norm_fc1': [],
        'grad_norm_fc1': [],
        'grad_cosine_sim': [],      # 
        'feature_diversity': [],    # hidden layer 特征多样性；奇异值接近（如 S ≈ [1,1.1,0.99,...]），说明表示均匀分布在多个方向上 → high diverity → high entropy
    }

    prev_w_fc1 = None

    t0 = time.time()
    for ep in range(1, epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion)
        vl, va = eval_one_epoch(model, test_loader, criterion)

        history['train_loss'].append(tl)
        history['train_acc'].append(ta)
        history['test_loss'].append(vl)
        history['test_acc'].append(va)

        # --- 权重相关 ---
        w_fc1 = model.fc1.weight.detach().cpu().numpy()
        w_norm = np.linalg.norm(w_fc1)
        history['w_norm_fc1'].append(w_norm)

        if prev_w_fc1 is not None:
            delta = w_fc1 - prev_w_fc1
            delta_norm = np.linalg.norm(delta)
            history['delta_w_norm_fc1'].append(delta_norm)
        else:
            history['delta_w_norm_fc1'].append(0.0)
        prev_w_fc1 = w_fc1.copy()

        # --- 梯度相关 ---
        grad_norm = 0.0
        current_grad = None
        if model.fc1.weight.grad is not None:
            g = model.fc1.weight.grad.detach().cpu().numpy()
            current_grad = g.copy() # shape: (hidden_dim, hidden_dim*2)
            grad_norm = float(np.linalg.norm(g))
        history['grad_norm_fc1'].append(grad_norm)

        # --- 梯度方向一致性 (grad_cosine_sim) ---
        # 不同 epoch 之间梯度方向的一致性，反映训练动态是否稳定/趋同
        grad_cos_sim = 0.0
        if current_grad is not None:
            grad_buffer.append(current_grad.flatten())
            if len(grad_buffer) > buffer_size:
                grad_buffer.pop(0)
            if len(grad_buffer) >= 2:
                grads_mat = np.stack(grad_buffer)  # [T, D] T个历史向量的梯度
                norms = np.linalg.norm(grads_mat, axis=1, keepdims=True) + 1e-12
                cos_matrix = (grads_mat @ grads_mat.T) / (norms @ norms.T) # cosin相似度矩阵 [T, T], 两两计算cosine相似度
                # 取上三角均值（不含对角）
                triu_mask = np.triu(np.ones_like(cos_matrix), k=1).astype(bool) #上三角矩阵掩码
                grad_cos_sim = float(cos_matrix[triu_mask].mean()) #平均cosine相似度
        history['grad_cosine_sim'].append(grad_cos_sim)

        # --- 特征多样性 (feature_diversity) ---
        # 用一个 batch 的隐藏层输出 h2 计算奇异值熵
        feature_div = 0.0
        try:
            model.eval()
            with torch.no_grad():
                # 取一个 batch
                x_batch, _ = next(iter(train_loader))
                x_batch = x_batch.to(device)
                a = x_batch[:, 0]
                b = x_batch[:, 1]
                ea = model.embed(a)
                eb = model.embed(b)
                h = torch.cat([ea, eb], dim=1)
                h2 = model.relu(model.fc1(h))  # [B, hidden_dim]
                h2_np = h2.cpu().numpy()
                # SVD
                U, S, Vt = np.linalg.svd(h2_np, full_matrices=False)
                # 归一化奇异值
                S_norm = S / (S.sum() + 1e-12)
                # 熵
                entropy = -np.sum(S_norm * np.log(S_norm + 1e-12))
                feature_div = float(entropy)
        except Exception as e:
            feature_div = 0.0
        history['feature_diversity'].append(feature_div)

        # --- 打印全部指标 ---
        if ep % 100 == 0 or ep == 1:
            print(
                f"Ep {ep:4d} | "
                f"train_acc={ta:.4f}, test_acc={va:.4f} | "
                f"train_loss={tl:.4f}, test_loss={vl:.4f} | "
                f"w_norm={w_norm:.3f} | "
                f"Δw_norm={history['delta_w_norm_fc1'][-1]:.3e} | "
                f"grad_norm={grad_norm:.3f} | "
                f"grad_cos_sim={grad_cos_sim:.3f} | "
                f"feature_div={feature_div:.3f}"
            )

    elapsed = time.time() - t0
    history['elapsed'] = elapsed
    print("Total elapsed:", elapsed)

    # grok_pt = detect_grokking_point(history['test_acc'], threshold=0.9, min_epoch=10)
    grok_pt = detect_grokking_point(
        history['test_acc'],
        threshold=0.95,
        min_epoch=50,
        consecutive_epochs=5   # 连续 5 轮 ≥95% 才算 grok，防止抖动、偶然达标
)
    print("Grokking point:", grok_pt)

    # --- 保存 ---
    model_path = os.path.join(script_dir, f"grok_mod_full_m{m}_hd{hidden_dim}_lr{lr}_wd{weight_decay}.pt")
    torch.save(model.state_dict(), model_path)
    hist_path = os.path.join(script_dir, f"grok_mod_full_m{m}_hd{hidden_dim}_lr{lr}_wd{weight_decay}_hist.json")
    with open(hist_path, 'w') as f:
        json.dump(history, f, cls=NumpyEncoder)
    print("Saved history to", hist_path)

    # --- 可视化（新增 feature_diversity 和 grad_cosine_sim）---
    epochs_range = list(range(1, epochs + 1))
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))

    axs[0,0].plot(epochs_range, history['train_acc'], label='train')
    axs[0,0].plot(epochs_range, history['test_acc'], label='test')
    if grok_pt:
        axs[0,0].axvline(grok_pt, color='red', linestyle='--', label='grok')
    axs[0,0].set_title('Accuracy')
    axs[0,0].legend()

    axs[0,1].plot(epochs_range, history['train_loss'], label='train')
    axs[0,1].plot(epochs_range, history['test_loss'], label='test')
    axs[0,1].set_title('Loss')
    axs[0,1].legend()

    axs[0,2].plot(epochs_range, history['w_norm_fc1'], 'g-')
    axs[0,2].set_title('Weight Norm (fc1)')

    axs[1,0].plot(epochs_range, history['delta_w_norm_fc1'], 'r-', label='ΔW_norm')
    axs[1,0].set_yscale('log')
    axs[1,0].set_title('Δ Weight Norm (log)')

    axs[1,1].plot(epochs_range, history['grad_norm_fc1'], 'm-', label='Grad Norm')
    axs[1,1].set_title('Gradient Norm')

    axs[1,2].plot(epochs_range, history['grad_cosine_sim'], 'c-')
    axs[1,2].set_title('Grad Cosine Similarity')

    axs[2,0].plot(epochs_range, history['feature_diversity'], 'y-')
    axs[2,0].set_title('Feature Diversity (SVD Entropy)')

    gap = np.array(history['train_acc']) - np.array(history['test_acc'])
    axs[2,1].plot(epochs_range, gap, 'k-')
    axs[2,1].set_title('Generalization Gap')

    axs[2,2].axis('off')

    fig.tight_layout()
    fig_path = os.path.join(script_dir, f"grok_mod_full_m{m}_hd{hidden_dim}_lr{lr}_wd{weight_decay}_fig.png")
    plt.savefig(fig_path, dpi=150)
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
                hist, grok_pt = train_grokking_full(m=97, hidden_dim=hd, epochs=1000, lr=lr, weight_decay=wd, seed=0, mode='add')
                key = f"m{97}_hd{hd}_lr{lr}_wd{wd}"
                all_hist[key] = hist
                grok_points[key] = grok_pt
    all_path = os.path.join(script_dir, "grokking_all_full_hist.json")
    with open(all_path, 'w') as f:
        json.dump({'hist': all_hist, 'grok_pt': grok_points}, f, cls=NumpyEncoder)
    print("Saved all histories to", all_path)