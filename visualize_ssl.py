"""
Visualize SSL embeddings using t-SNE, colored by market regime (volatility level).
Saves: results/tsne_regime.png
Also saves: results/t-SNE_SSL_Embeddings.png (colored by 30-day future direction)
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
from src.ssl.model import SSLModel

# Setup paths
project_root = Path(__file__).resolve().parent
results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)

# Load data
print("Loading data...")
windows = np.load(project_root / "data" / "processed" / "windows_train.npy")
labels_train = np.load(project_root / "data" / "processed" / "labels_train.npy")
future_direction_labels = labels_train
if len(windows) != len(future_direction_labels):
    raise ValueError(
        f"windows_train.npy and labels_train.npy mismatch: "
        f"{len(windows)} vs {len(future_direction_labels)}"
    )
windows_tensor = torch.FloatTensor(windows)

# Load trained SSL model
print("Loading SSL model...")
checkpoint = torch.load(project_root / "models" / "ssl_best_model.pt", map_location='cpu')
model = SSLModel(input_dim=windows.shape[-1])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get embeddings
print("Extracting embeddings...")
with torch.no_grad():
    embeddings = []
    for w in windows_tensor:
        emb = model.encode(w.unsqueeze(0))
        embeddings.append(emb.squeeze(0).cpu().numpy())
embeddings = np.array(embeddings)

# Compute t-SNE
print("Computing t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
emb_2d = tsne.fit_transform(embeddings)

# ========================================
# Create volatility regime labels
# ========================================
print("Computing volatility regimes...")
volatilities = []
for window in windows:
    # Calculate volatility as std of returns within each window
    prices = window[:, 0]  # Assuming price is in column 0
    returns = np.diff(prices) / (prices[:-1] + 1e-8)
    vol = np.std(returns)
    volatilities.append(vol)

volatilities = np.array(volatilities)

# Create 3 volatility regimes: Low, Medium, High
vol_33 = np.percentile(volatilities, 33)
vol_66 = np.percentile(volatilities, 66)

regime_labels = np.zeros(len(volatilities), dtype=int)
regime_labels[volatilities <= vol_33] = 0  # Low
regime_labels[(volatilities > vol_33) & (volatilities <= vol_66)] = 1  # Medium
regime_labels[volatilities > vol_66] = 2  # High

regime_names = ['Low Volatility', 'Medium Volatility', 'High Volatility']
regime_colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red

# ========================================
# Plot 1: t-SNE colored by volatility regime
# ========================================
print("Creating volatility regime visualization...")
plt.figure(figsize=(10, 8))
plt.style.use('seaborn-v0_8-whitegrid')

for i, (name, color) in enumerate(zip(regime_names, regime_colors)):
    mask = regime_labels == i
    plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], 
                c=color, label=name, alpha=0.6, s=50, edgecolors='white', linewidth=0.5)

plt.xlabel("t-SNE Component 1", fontsize=12)
plt.ylabel("t-SNE Component 2", fontsize=12)
plt.title("t-SNE of SSL Embeddings Colored by Market Regime", fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()

output_regime = results_dir / "tsne_regime.png"
plt.savefig(output_regime, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_regime}")
plt.close()

# ========================================
# Plot 2: t-SNE colored by future price direction
# ========================================
print("Creating future direction visualization...")
price_labels = future_direction_labels

plt.figure(figsize=(10, 8))
plt.style.use('seaborn-v0_8-whitegrid')

colors_price = ['#3498db', '#e74c3c']  # Blue for down, Red for up
labels_price = ['Future Down', 'Future Up']

for i, (name, color) in enumerate(zip(labels_price, colors_price)):
    mask = price_labels == i
    plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], 
                c=color, label=name, alpha=0.6, s=50, edgecolors='white', linewidth=0.5)

plt.xlabel("t-SNE Component 1", fontsize=12)
plt.ylabel("t-SNE Component 2", fontsize=12)
plt.title("t-SNE of SSL Embeddings Colored by 30-Day Future Direction", fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()

output_price = results_dir / "t-SNE_SSL_Embeddings.png"
plt.savefig(output_price, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_price}")
plt.close()

print("\nDone! Generated visualizations:")
print(f"  - {output_regime}")
print(f"  - {output_price}")
