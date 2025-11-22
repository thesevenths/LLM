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
        self.m = m
        if pairs is not None:
            self.pairs = list(pairs)
        else:
            self.pairs = [(a,b) for a in range(m) for b in range(m)]
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
    for x,y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total += x.size(0)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
    return total_loss/total, correct/total

def eval_one_epoch(model, loader, criterion):
    model.eval()
    total = 0
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total += x.size(0)
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
    return total_loss/total, correct/total

def detect_grokking_point(
    test_acc_list,
    threshold=0.9,
    min_epoch=10,
    consecutive_epochs=3,
    smooth_window=None
):
    if len(test_acc_list) < min_epoch + consecutive_epochs:
        return None
    if smooth_window is not None and smooth_window > 1:
        acc = np.array(test_acc_list)
        smoothed = np.convolve(acc, np.ones(smooth_window)/smooth_window, mode='same')
        smoothed[:smooth_window//2] = acc[:smooth_window//2]
        smoothed[-(smooth_window//2):] = acc[-(smooth_window//2):]
        acc_to_use = smoothed.tolist()
    else:
        acc_to_use = test_acc_list
    for i in range(min_epoch, len(acc_to_use)-consecutive_epochs+1):
        if all(acc_to_use[j] >= threshold for j in range(i, i+consecutive_epochs)):
            if i == 0 or acc_to_use[i-1] < threshold:
                return i
            else:
                continue
    return None

def train_grokking_full(m=97, hidden_dim=128, epochs=500, lr=1e-3, weight_decay=0.0, seed=0, mode='add'):
    set_seed(seed)
    all_pairs = [(a,b) for a in range(m) for b in range(m)]
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

    # 阈值设定（改进版）
    T1_delta_w = 5e-3                # 较低阈值
    T1_grad_cos = 0.03               # 降低至 ~0.03
    tau_cov = 0.6                    # 覆盖度阈值
    thr_test_acc_for_stage3 = 0.95
    min_epoch_for_stage3 = 50
    consecutive_for_stage3 = 5
    rem_dirs_threshold_for_stage3 = 5   # “剩余方向”低于这个视为特征基本覆盖

    # 候选方向 U_np — 使用傅里叶基方向为例（两向量拼接）
    K = hidden_dim
    D = hidden_dim * 2
    U_np = np.zeros((K, D), dtype=np.float32)
    for k in range(K):
        # 假设每个方向为 sin+cos 模型（简化示例）
        vec = np.sin(2*np.pi * (k+1) * np.arange(D) / D) + np.cos(2*np.pi * (k+1) * np.arange(D) / D)
        U_np[k] = vec / (np.linalg.norm(vec) + 1e-12)

    grad_buffer = []
    buffer_size = 5

    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'w_norm_fc1': [], 'delta_w_norm_fc1': [],
        'grad_norm_fc1': [], 'grad_cosine_sim': [],
        'feature_diversity': [], 'stage': [],
        'num_covered_dirs': [], 'num_remaining_dirs': [],
        'node_similarity_mean': []    # 新增　节点间相似度监控
    }
    prev_w_fc1 = None

    current_stage = 1
    stage_start = {1: 1}

    t0 = time.time()
    for ep in range(1, epochs+1):
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion)
        vl, va = eval_one_epoch(model, test_loader, criterion)

        history['train_loss'].append(tl)
        history['train_acc'].append(ta)
        history['test_loss'].append(vl)
        history['test_acc'].append(va)

        # 权重变化
        w_fc1 = model.fc1.weight.detach().cpu().numpy()
        w_norm = np.linalg.norm(w_fc1)
        history['w_norm_fc1'].append(w_norm)
        if prev_w_fc1 is not None:
            delta = w_fc1 - prev_w_fc1
            delta_norm = np.linalg.norm(delta)
        else:
            delta_norm = 0.0
        history['delta_w_norm_fc1'].append(delta_norm)
        prev_w_fc1 = w_fc1.copy()

        # 梯度相关
        grad_norm = 0.0
        current_grad = None
        if model.fc1.weight.grad is not None:
            g = model.fc1.weight.grad.detach().cpu().numpy()
            current_grad = g.copy()
            grad_norm = float(np.linalg.norm(g))
        history['grad_norm_fc1'].append(grad_norm)

        grad_cos_sim = 0.0
        if current_grad is not None:
            grad_buffer.append(current_grad.flatten())
            if len(grad_buffer) > buffer_size:
                grad_buffer.pop(0)
            if len(grad_buffer) >= 2:
                grads_mat = np.stack(grad_buffer)
                norms = np.linalg.norm(grads_mat, axis=1, keepdims=True) + 1e-12
                cos_matrix = (grads_mat @ grads_mat.T) / (norms @ norms.T)
                triu_mask = np.triu(np.ones_like(cos_matrix), k=1).astype(bool)
                grad_cos_sim = float(cos_matrix[triu_mask].mean())
        history['grad_cosine_sim'].append(grad_cos_sim)

        # 特征多样性
        feature_div = 0.0
        try:
            model.eval()
            with torch.no_grad():
                x_batch, _ = next(iter(train_loader))
                x_batch = x_batch.to(device)
                a = x_batch[:,0]; b = x_batch[:,1]
                ea = model.embed(a); eb = model.embed(b)
                h = torch.cat([ea, eb], dim=1)
                h2 = model.relu(model.fc1(h))
                h2_np = h2.cpu().numpy()
                U_svd, S_svd, Vt_svd = np.linalg.svd(h2_np, full_matrices=False)
                S_norm = S_svd / (S_svd.sum() + 1e-12)
                entropy = -np.sum(S_norm * np.log(S_norm + 1e-12))
                feature_div = float(entropy)
        except Exception:
            feature_div = 0.0
        history['feature_diversity'].append(feature_div)

        # 候选方向覆盖度监控
        w_flat = w_fc1.reshape(w_fc1.shape[0], -1)  # (hidden_dim × D)
        W_norms = np.linalg.norm(w_flat, axis=1, keepdims=True) + 1e-12
        U_norms = np.linalg.norm(U_np, axis=1, keepdims=True) + 1e-12
        cos_mat = (w_flat @ U_np.T) / (W_norms * U_norms.T)
        max_cos_per_dir = cos_mat.max(axis=0)
        covered = (max_cos_per_dir >= tau_cov)
        num_covered = int(covered.sum())
        num_remaining = int(K - num_covered)
        history['num_covered_dirs'].append(num_covered)
        history['num_remaining_dirs'].append(num_remaining)

        # 隐藏节点间相似度监控（行与行的余弦平均）
        W_norms_rows = np.linalg.norm(w_flat, axis=1, keepdims=True) + 1e-12
        cos_node_mat = (w_flat @ w_flat.T) / (W_norms_rows * W_norms_rows.T)
        # mask upper off‐diagonal
        triu_mask2 = np.triu(np.ones_like(cos_node_mat), k=1).astype(bool)
        if triu_mask2.any():
            node_sim_mean = float(cos_node_mat[triu_mask2].mean())
        else:
            node_sim_mean = 0.0
        history['node_similarity_mean'].append(node_sim_mean)

        # 滑动平均（监控平滑指标）
        if ep >= 10:
            avg_delta_w = np.mean(history['delta_w_norm_fc1'][-10:])
            avg_grad_cos = np.mean(history['grad_cosine_sim'][-10:])
        else:
            avg_delta_w = delta_norm
            avg_grad_cos = grad_cos_sim

        # 阶段判定逻辑
        history['stage'].append(current_stage)
        if current_stage == 1:
            if (avg_delta_w > T1_delta_w) and (avg_grad_cos > T1_grad_cos):
                current_stage = 2
                stage_start[2] = ep
                print(f"*** Enter Stage II at epoch {ep}")
        elif current_stage == 2:
            # 当 “测试准确率连续高” 或 “剩余方向数低” 之一满足时进入阶段 III
            grok_pt = detect_grokking_point(history['test_acc'],
                                            threshold=thr_test_acc_for_stage3,
                                            min_epoch=0,
                                            consecutive_epochs=consecutive_for_stage3)
            if (grok_pt is not None and grok_pt <= ep) or (num_remaining <= rem_dirs_threshold_for_stage3):
                current_stage = 3
                stage_start[3] = ep
                print(f"*** Enter Stage III at epoch {ep}")

        if ep % 100 == 0 or ep == 1:
            print(f"Ep {ep:4d} | train_acc={ta:.4f}, test_acc={va:.4f} | ΔW_norm={delta_norm:.3e} | "
                  f"avgΔW={avg_delta_w:.3e} | grad_cos={grad_cos_sim:.3f} | avg_grad_cos={avg_grad_cos:.3f} | "
                  f"feat_div={feature_div:.3f} | rem_dirs={num_remaining} | node_sim={node_sim_mean:.3f} | stage={current_stage}")

    elapsed = time.time() - t0
    history['elapsed'] = elapsed
    print("Total elapsed:", elapsed)
    print("Stage start epochs:", stage_start)

    grok_pt = detect_grokking_point(history['test_acc'],
                                    threshold=thr_test_acc_for_stage3,
                                    min_epoch=50,
                                    consecutive_epochs=consecutive_for_stage3)
    print("Grokking point:", grok_pt)

    model_path = os.path.join(script_dir,
        f"grok_mod_full_m{m}_hd{hidden_dim}_lr{lr}_wd{weight_decay}.pt")
    torch.save(model.state_dict(), model_path)
    hist_path = os.path.join(script_dir,
        f"grok_mod_full_m{m}_hd{hidden_dim}_lr{lr}_wd{weight_decay}_hist.json")
    with open(hist_path, 'w') as f:
        json.dump({'history': history, 'stage_start': stage_start}, f, cls=NumpyEncoder)
    print("Saved history to", hist_path)

    # 可视化
    epochs_range = list(range(1, epochs+1))
    fig, axs = plt.subplots(4, 2, figsize=(14, 16))

    axs[0,0].plot(epochs_range, history['train_acc'], label='train')
    axs[0,0].plot(epochs_range, history['test_acc'], label='test')
    for st, ep0 in stage_start.items():
        axs[0,0].axvline(ep0, linestyle='--', label=f'Stage {st} start')
    axs[0,0].set_title('Accuracy')
    axs[0,0].legend()

    axs[0,1].plot(epochs_range, history['train_loss'], label='train')
    axs[0,1].plot(epochs_range, history['test_loss'], label='test')
    axs[0,1].set_title('Loss')
    axs[0,1].legend()

    axs[1,0].plot(epochs_range, history['w_norm_fc1'], 'g-')
    axs[1,0].set_title('Weight Norm (fc1)')

    axs[1,1].plot(epochs_range, history['delta_w_norm_fc1'], 'r-', label='ΔW_norm')
    axs[1,1].set_yscale('log')
    axs[1,1].set_title('Δ Weight Norm (log)')

    axs[2,0].plot(epochs_range, history['grad_cosine_sim'], 'c-', label='grad_cos')
    axs[2,0].plot(epochs_range, np.convolve(history['grad_cosine_sim'], np.ones(10)/10, mode='same'), 'c--', label='smooth grad_cos')
    axs[2,0].set_title('Grad Cosine Similarity')
    axs[2,0].legend()

    axs[2,1].plot(epochs_range, history['feature_diversity'], 'y-')
    axs[2,1].set_title('Feature Diversity (SVD Entropy)')

    axs[3,0].plot(epochs_range, history['num_remaining_dirs'], 'b-')
    axs[3,0].set_title('Remaining Candidate Feature Directions')

    axs[3,1].plot(epochs_range, history['node_similarity_mean'], 'm-')
    axs[3,1].set_title('Node Similarity Mean (hidden units)')

    fig.tight_layout()
    fig_path = os.path.join(script_dir,
        f"grok_mod_full_m{m}_hd{hidden_dim}_lr{lr}_wd{weight_decay}_fig.png")
    plt.savefig(fig_path, dpi=150)
    print("Saved figure to", fig_path)
    plt.close()

    return history, stage_start, grok_pt

if __name__ == '__main__':
    dims = [32, 64, 128]
    lrs = [1e-3, 5e-4]
    wds = [0.0, 1e-4]
    all_hist = {}
    stage_starts = {}
    grok_points = {}
    for hd in dims:
        for lr in lrs:
            for wd in wds:
                print("=== Running hd =", hd, "lr =", lr, "wd =", wd)
                hist, stage_start, grok_pt = train_grokking_full(
                    m=97, hidden_dim=hd, epochs=1000, lr=lr, weight_decay=wd,
                    seed=0, mode='add')
                key = f"m{97}_hd{hd}_lr{lr}_wd{wd}"
                all_hist[key] = hist
                stage_starts[key] = stage_start
                grok_points[key] = grok_pt
    all_path = os.path.join(script_dir, "grokking_all_full_hist.json")
    with open(all_path, 'w') as f:
        json.dump({
            'hist': all_hist,
            'stage_starts': stage_starts,
            'grok_pt': grok_points
        }, f, cls=NumpyEncoder)
    print("Saved all histories to", all_path)
