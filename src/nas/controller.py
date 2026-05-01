# Fixed version of src/nas/controller.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NASController(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=32, num_classes=2, 
                 temperature=5.0,  # Start high for exploration
                 temp_min=0.1,     # End low for exploitation  
                 temp_decay_schedule='cosine',  # Better annealing
                 max_epochs=75):   # Needed for schedule
        super().__init__()
        self.temperature = temperature
        self.temp_min = temp_min
        self.temp_decay_schedule = temp_decay_schedule
        self.max_epochs = max_epochs
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Only TCN, Dilated Conv1D, Conv1D, and Depthwise Separable Conv1D operations
        self.ops = nn.ModuleDict({
            'tcn': TCNBlock(input_dim, hidden_dim),
            'dilated_conv1d': DilatedConv1DBlock(input_dim, hidden_dim, dilation=2),
            'conv1d': SmallConv1DBlock(input_dim, hidden_dim, kernel_size=3, causal=True),
            'depthwise_separable_conv1d': DepthwiseSeparableConv1DBlock(input_dim, hidden_dim, kernel_size=3),
        })
        
        # Architecture parameters
        self.alpha = nn.Parameter(torch.zeros(len(self.ops)))
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    def arch_parameters(self):
        """Returns architecture parameters for optimization."""
        return [self.alpha]
    
    def get_weights(self):
        """Returns operation weights using Gumbel Softmax for differentiable search."""
        return F.gumbel_softmax(self.alpha, tau=self.temperature, hard=False)
    
    def forward(self, x):
        """Forward pass with weighted sum of operations."""
        # x is expected to be 2D: (batch_size, input_dim) from SSL encoder
        weights = self.get_weights()
        batch_size = x.size(0)
        outputs = []
        
        # Apply each operation
        for i, (name, op) in enumerate(self.ops.items()):
            try:
                out = op(x)  # Each operation handles its own dimension requirements
                outputs.append(out)
            except Exception as e:
                print(f"Error in operation {name}: {e}")
                print(f"Input shape to operation {name}: {x.shape}")
                import traceback; traceback.print_exc()
                # Create fallback output with correct dimensions
                out = torch.zeros(batch_size, self.hidden_dim, device=x.device)
                outputs.append(out)
        
        # Combine outputs using architecture weights
        final_output = sum(w * out for w, out in zip(weights, outputs))
        
        # Apply classifier
        logits = self.classifier(final_output)
        return logits
    
    def get_arch_info(self):
        """Returns current architecture information."""
        weights = F.softmax(self.alpha, dim=-1)
        arch_info = {
            name: float(weight)
            for name, weight in zip(self.ops.keys(), weights)
        }
        return arch_info

    def discretize(self):
        """Returns discrete architecture choices."""
        with torch.no_grad():
            weights = F.softmax(self.alpha, dim=-1)
            max_weight_idx = weights.argmax().item()
            return list(self.ops.keys())[max_weight_idx]
    
    def get_temperature(self, epoch, max_epochs=None, temp_max=None, temp_min=None, schedule=None):
        """Temperature annealing schedule."""
        if max_epochs is None:
            max_epochs = self.max_epochs
        if temp_max is None:
            temp_max = self.temperature
        if temp_min is None:
            temp_min = self.temp_min
        if schedule is None:
            schedule = self.temp_decay_schedule
        if schedule == 'cosine':
            return temp_min + (temp_max - temp_min) * (1 + np.cos(np.pi * epoch / max_epochs)) / 2
        elif schedule == 'exponential':
            return temp_max * (temp_min / temp_max) ** (epoch / max_epochs)
        else:
            return self.temperature  # fallback: fixed

    def set_temperature(self, epoch):
        """Update self.temperature according to the schedule and epoch."""
        self.temperature = self.get_temperature(epoch)


# --- NAS Operations ---

class TCNBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(nn.Conv1d(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size,
                                    padding=dilation * (kernel_size - 1) // 2, dilation=dilation))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        # x: (batch, features) or (batch, seq, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        x = x.transpose(1, 2)  # (batch, features, seq)
        out = self.network(x)
        out = self.global_pool(out).squeeze(-1)
        return out

class DilatedConv1DBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, dilation=2):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        out = self.conv(x)
        out = self.relu(out)
        out = self.global_pool(out).squeeze(-1)
        return out

class SmallConv1DBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, causal=True):
        super().__init__()
        self.causal = causal
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=(kernel_size-1) if causal else (kernel_size-1)//2)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        out = self.conv(x)
        if self.causal:
            # After transpose, x.size(2) is the sequence length
            out = out[:, :, :x.size(2)]  # Remove extra padding for causal conv
        out = self.relu(out)
        out = self.global_pool(out).squeeze(-1)
        return out

class DepthwiseSeparableConv1DBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv1d(input_dim, input_dim, kernel_size, groups=input_dim, padding=(kernel_size-1)//2)
        # Pointwise convolution
        self.pointwise = nn.Conv1d(input_dim, hidden_dim, 1)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.relu(out)
        out = self.global_pool(out).squeeze(-1)
        return out