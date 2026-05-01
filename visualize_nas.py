"""
Visualize NAS architecture weights evolution over training epochs.
Saves: results/nas_arch_weights.png
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
project_root = Path(__file__).resolve().parent
results_path = project_root / "results" / "arch_weights_history.json"
output_path = project_root / "results" / "nas_arch_weights.png"

# Load architecture weights history
with open(results_path, "r") as f:
    arch_weights_history = json.load(f)

# Get operation names and number of epochs
ops = list(arch_weights_history[0].keys())
epochs = range(1, len(arch_weights_history) + 1)

# Create figure with improved styling
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-whitegrid')

# Define colors for each operation
colors = {
    'conv1d': '#2ecc71',           # Green - winner
    'depthwise_separable_conv1d': '#3498db',  # Blue
    'dilated_conv1d': '#9b59b6',   # Purple
    'tcn': '#e74c3c'               # Red - lowest
}

# Plot each operation's weight over epochs
for op in ops:
    weights = [aw[op] * 100 for aw in arch_weights_history]  # Convert to percentage
    color = colors.get(op, 'gray')
    label_name = op.replace('_', ' ').title()
    plt.plot(epochs, weights, label=label_name, linewidth=2.5, color=color)

# Add horizontal reference line at 25% (uniform distribution)
plt.axhline(y=25, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Uniform (25%)')

# Styling
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Architecture Weight (%)", fontsize=12)
plt.title("Evolution of Architecture Weights During NAS", fontsize=14, fontweight='bold')
plt.legend(loc='center right', fontsize=10)
plt.xlim(1, len(epochs))
plt.ylim(20, 28)

# Add annotation for final values
final_weights = arch_weights_history[-1]
for op, weight in final_weights.items():
    label_name = op.replace('_', ' ').title()
    print(f"{label_name}: {weight*100:.2f}%")

plt.tight_layout()

# Save figure
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")

plt.show()