import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import random
import time
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
print(f"Script directory: {script_dir}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 超参 (优: 低wd, 超长, scheduler)
P = 97
hidden_dim = 512
lr = 0.01
wd = 1e-5  # 低wd促III
epochs = 150000
eval_every = 5000

# 数据: train=300
train_size = 300
test_size = 840
torch.manual_seed(42); random.seed(42)
a_vals, b_vals = torch.meshgrid(torch.arange(P), torch.arange(P), indexing='ij')
all_idx = torch.randperm(P*P)
train_idx = all_idx[:train_size]
test_idx = all_idx[train_size:train_size + test_size]
X_train = torch.stack([a_vals.flatten()[train_idx], b_vals.flatten()[train_idx]], dim=1).float().to(device)
y_train = ((a_vals + b_vals) % P).flatten()[train_idx].to(device)
X_test = torch.stack([a_vals.flatten()[test_idx], b_vals.flatten()[test_idx]], dim=1).float().to(device)
y_test = ((a_vals + b_vals) % P).flatten()[test_idx].to(device)
print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

# 模型: tanh
class GrokkingMLP(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.lin1 = nn.Linear(2, hidden_dim, bias=True)
        self.lin2 = nn.Linear(hidden_dim, P, bias=True)
    
    def forward(self, x):
        h = torch.tanh(self.lin1(x))
        return self.lin2(h)

model = GrokkingMLP(hidden_dim=hidden_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

# 存储
logs = {
    'epoch': [], 'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [],
    'W1_norm': [], 'W1_delta_norm': [],
    'grad_norm': [], 'grad_cosine_sim': [],
    'stage': []
}
prev_W1 = None
grad_buffer = deque(maxlen=10)

def compute_grad_cosine_similarity(grads_list):
    if len(grads_list) < 2: return 0.0
    vecs = torch.stack(grads_list).view(len(grads_list), -1)
    norms = torch.norm(vecs, dim=1, keepdim=True)
    cos_sim = torch.mm(vecs, vecs.t()) / (norms @ norms.t() + 1e-8)
    mask = torch.triu(torch.ones_like(cos_sim), diagonal=1).bool()
    return cos_sim[mask].mean().item()

start_time = time.time()

print("Starting training...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    X_train_noisy = X_train + torch.randn_like(X_train) * 0.01
    logits_train = model(X_train_noisy)
    loss = criterion(logits_train, y_train)
    loss.backward()
    grad_W1 = model.lin1.weight.grad.clone().detach()
    grad_buffer.append(grad_W1.cpu())
    
    # MuOn++ (adapt mom if grad low)
    grad_n = torch.norm(grad_W1).item()
    for param_group in optimizer.param_groups:
        param_group['momentum'] = 0.995 if grad_n < 0.01 else 0.9  # strong adapt
    optimizer.step()
    scheduler.step()

    if epoch % eval_every == 0 or epoch == epochs - 1:
        model.eval()
        with torch.no_grad():
            logits_train = model(X_train)
            train_loss = criterion(logits_train, y_train).item()
            pred_train = logits_train.argmax(1)
            train_acc = (pred_train == y_train).float().mean().item()
            logits_test = model(X_test)
            test_loss = criterion(logits_test, y_test).item()
            pred_test = logits_test.argmax(1)
            test_acc = (pred_test == y_test).float().mean().item()

            W1 = model.lin1.weight.data
            W1_norm = torch.norm(W1).item()
            W1_delta_norm = 0.0 if prev_W1 is None else torch.norm(W1 - prev_W1).item()
            prev_W1 = W1.clone()
            grad_norm = torch.norm(grad_W1).item()
            grad_cos_sim = compute_grad_cosine_similarity(list(grad_buffer))

            cos_thresh = 0.5
            norm_thresh = 3.0

            logs['epoch'].append(epoch)
            logs['train_loss'].append(train_loss)
            logs['test_loss'].append(test_loss)
            logs['train_acc'].append(train_acc)
            logs['test_acc'].append(test_acc)
            logs['W1_norm'].append(W1_norm)
            logs['W1_delta_norm'].append(W1_delta_norm)
            logs['grad_norm'].append(grad_norm)
            logs['grad_cosine_sim'].append(grad_cos_sim)

            stage = "Unknown"
            if epoch < 10000 and W1_delta_norm < 1e-3:
                stage = "I: Lazy"
            elif test_acc < 0.9 and grad_cos_sim > cos_thresh and W1_norm > norm_thresh:
                stage = "II: Independent"
            elif train_acc >= 0.99 and test_acc > 0.05 and grad_norm > 0.01:  # 宽test捕
                stage = "III: Interactive"
            else:
                stage = "Transition"
            logs['stage'].append(stage)

            print(f"Ep {epoch:6d} | Train {train_acc:.4f} | Test {test_acc:.4f} | LossT {train_loss:.4f} LossV {test_loss:.4f} | Stage {stage}")

end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time/60:.2f} min")

# 绘图+保存
epochs_plot = logs['epoch']
fig, axs = plt.subplots(2, 3, figsize=(16, 10))

axs[0,0].plot(epochs_plot, logs['train_acc'], 'b-', label='Train')
axs[0,0].plot(epochs_plot, logs['test_acc'], 'r-', label='Test')
for i in range(1, len(logs['stage'])):
    if logs['stage'][i] != logs['stage'][i-1]:
        axs[0,0].axvline(x=epochs_plot[i], color='k', linestyle='--', alpha=0.5)
axs[0,0].set_title('Acc with Stages')
axs[0,0].set_ylabel('Accuracy')
axs[0,0].legend()

axs[0,1].plot(epochs_plot, logs['W1_norm'], 'g-')
axs[0,1].set_title('||W1||')

axs[0,2].plot(epochs_plot, np.log10(np.maximum(logs['W1_delta_norm'], 1e-10)), 'r-')
axs[0,2].set_title('log Δ||W1||')

axs[1,0].plot(epochs_plot, logs['grad_norm'], 'm-')
axs[1,0].set_title('||∇W1||')

axs[1,1].plot(epochs_plot, logs['grad_cosine_sim'], 'c-')
axs[1,1].set_title('Grad Cos Sim')

stage_map = {"I: Lazy": 1, "II: Independent": 2, "III: Interactive": 3, "Transition": 0, "Unknown": -1}
stage_vals = [stage_map.get(s, -1) for s in logs['stage']]
axs[1,2].plot(epochs_plot, stage_vals, 'ko-')
axs[1,2].set_yticks(list(stage_map.values()), list(stage_map.keys()))
axs[1,2].set_title('Detected Stage')

plt.tight_layout()
plot_path = os.path.join(script_dir, 'grok_final.png')
plt.savefig(plot_path, dpi=150)
plt.show()
print(f"Plot saved: {plot_path}")

# 模型保存
try:
    from safetensors.torch import save_file
    model_path = os.path.join(script_dir, 'grok_final.safetensors')
    save_file(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")
except ImportError:
    model_path = os.path.join(script_dir, 'grok_final.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Safetensors no; saved: {model_path}")

print(f"\nFinal Train: {logs['train_acc'][-1]:.4f}, Test: {logs['test_acc'][-1]:.4f}")
print("Stages sample: ", logs['stage'][:5] + ['...'] + [logs['stage'][-1]])