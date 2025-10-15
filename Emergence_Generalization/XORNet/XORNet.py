import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

# --- Get the directory where this script is located ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- 1. Prepare XOR Data ---
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# --- 2. Define and Train the Model ---
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 12)
        self.fc2 = nn.Linear(12, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = XORNet()
# criterion = nn.MSELoss()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=1.0)

# Training loop with detailed logging
print("Starting training...")
for epoch in range(600):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    # --- Log hidden layer stats every 1000 epochs ---
    if epoch % 60 == 0 or epoch == 599:
        with torch.no_grad(): # Disable gradient tracking for logging
            w = model.fc1.weight
            b = model.fc1.bias
            w_grad = model.fc1.weight.grad
            b_grad = model.fc1.bias.grad
            
            # Calculate norms
            w_norm = torch.norm(w, p='fro').item() # Frobenius norm for matrix
            if w_grad is not None:
                w_grad_norm = torch.norm(w_grad, p='fro').item()
            else:
                w_grad_norm = 0.0
                
            print(f"\n--- Epoch {epoch} ---")
            print(f"Loss: {loss.item():.6f}")
            print(f"Hidden Weight (fc1.weight):\n{w}")
            print(f"Weight Norm (L2/Fro): {w_norm:.4f}")
            print(f"Hidden Bias (fc1.bias): {b}")
            print(f"Weight Gradient (fc1.weight.grad):\n{w_grad}")
            print(f"Gradient Norm (L2/Fro): {w_grad_norm:.4f}")

# --- 3. Create a grid for decision boundary ---
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.from_numpy(grid).float()

# Make predictions on the grid
with torch.no_grad():
    model.eval()
    pred_grid = model(grid_tensor).numpy().reshape(xx.shape)

# --- 4. Plot the result ---
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, pred_grid, levels=50, cmap='RdBu', alpha=0.8)

# Plot the XOR data points
plt.scatter([0, 1], [0, 1], c='blue', s=200, marker='o', edgecolors='k', linewidth=2, label='Class 0')
plt.scatter([0, 1], [1, 0], c='red', s=200, marker='s', edgecolors='k', linewidth=2, label='Class 1')

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('XOR Problem: Decision Boundary of a Trained Neural Network')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# --- Save the plot to the script's directory ---
plot_path = os.path.join(SCRIPT_DIR, 'xor_decision_boundary.png')
plt.savefig(plot_path, dpi=150)
print(f"\nDecision boundary plot saved as '{plot_path}'.")

# Print final predictions
with torch.no_grad():
    final_pred = model(X)
    print("\nFinal Predictions:")
    for i in range(4):
        print(f"Input: {X[i].tolist()} → Target: {int(y[i].item())}, Prediction: {final_pred[i].item():.4f}")

# === Save as safetensors to the script's directory ===
from safetensors.torch import save_file
state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
model_path = os.path.join(SCRIPT_DIR, 'xor_model.safetensors')
save_file(state_dict_cpu, model_path)
print(f"✅ Model saved in safetensors format at '{model_path}'.")

# === Print final hidden layer parameters ===
print("\n=== Final Hidden Layer Parameters ===")
print("fc1.weight:\n", model.fc1.weight)
print("fc1.bias: ", model.fc1.bias)
print("Weight L2 Norm: ", torch.norm(model.fc1.weight, p='fro').item())