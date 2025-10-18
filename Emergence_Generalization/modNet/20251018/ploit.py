import json
import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
print(f"Script directory: {script_dir}")

hist_path = os.path.join(script_dir, "grok_mod_full_m97_hd128_lr0.001_wd0.0001_hist.json")
with open(hist_path, 'r') as f:
    data = json.load(f)

history = data['history']
stage_start = data['stage_start']

epochs = list(range(1, len(history['train_acc']) + 1))

# 从 history 提取指标
train_acc = np.array(history['train_acc'])
test_acc  = np.array(history['test_acc'])
delta_w   = np.array(history['delta_w_norm_fc1'])
grad_cos  = np.array(history['grad_cosine_sim'])
feat_div  = np.array(history['feature_diversity'])
rem_dirs  = np.array(history['num_remaining_dirs'])
node_sim  = np.array(history['node_similarity_mean'])

# 计算滑动平均，比如 window = 10
window = 10
smooth_delta_w = np.convolve(delta_w, np.ones(window)/window, mode='same')
smooth_grad_cos  = np.convolve(grad_cos,  np.ones(window)/window, mode='same')

# —— 绘图 ——
plt.figure(figsize=(12, 8))

plt.subplot(2,2,1)
plt.plot(epochs, train_acc, label='train_acc')
plt.plot(epochs, test_acc,  label='test_acc')
for s, ep0 in stage_start.items():
    plt.axvline(ep0, linestyle='--', label=f"Stage {s} start")
plt.title("Accuracy over epochs")
plt.legend()

plt.subplot(2,2,2)
plt.plot(epochs, smooth_delta_w, label='smooth ΔW_norm')
plt.plot(epochs, smooth_grad_cos,  label='smooth grad_cos')
plt.title("Weight & Gradient Metrics")
plt.legend()

plt.subplot(2,2,3)
plt.plot(epochs, feat_div, label='feature diversity')
plt.plot(epochs, rem_dirs, label='remaining dirs')
plt.title("Feature Diversity & Remaining Feature Directions")
plt.legend()

plt.subplot(2,2,4)
plt.plot(epochs, node_sim, label='node similarity mean')
plt.title("Hidden Nodes Similarity")
plt.legend()

plt.tight_layout()
plt.show()

# plt_path = os.path.join(script_dir, "grok_mod_m97_hd128_lr0.001_wd0.0001_history.png")
# plt.savefig(plt_path, dpi=150)