import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NASController(nn.Module):
    def __init__(
        self,
        input_dim=128,
        hidden_dim=64,
        num_classes=2,
        temperature=5.0,
        temp_min=0.1,
        temp_decay_schedule="cosine",
        max_epochs=75,
    ):
        super().__init__()
        self.initial_temperature = temperature
        self.temperature = temperature
        self.temp_min = temp_min
        self.temp_decay_schedule = temp_decay_schedule
        self.max_epochs = max_epochs
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.ops = nn.ModuleDict({
            "tcn": TCNBlock(input_dim, hidden_dim),
            "dilated_conv1d": DilatedConv1DBlock(input_dim, hidden_dim, dilation=2),
            "conv1d": SmallConv1DBlock(input_dim, hidden_dim, kernel_size=3, causal=True),
            "depthwise_separable_conv1d": DepthwiseSeparableConv1DBlock(
                input_dim, hidden_dim, kernel_size=3
            ),
        })
        self.alpha = nn.Parameter(torch.zeros(len(self.ops)))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def arch_parameters(self):
        return [self.alpha]

    def get_weights(self, mode=None):
        mode = mode or ("sample" if self.training else "softmax")
        if mode == "sample":
            return F.gumbel_softmax(self.alpha, tau=self.temperature, hard=False)
        if mode == "softmax":
            return F.softmax(self.alpha, dim=-1)
        if mode == "discrete":
            weights = torch.zeros_like(self.alpha)
            weights[torch.argmax(self.alpha)] = 1.0
            return weights
        raise ValueError(f"Unknown NAS weight mode: {mode}")

    def forward(self, x, mode=None):
        if x.dim() != 3:
            raise ValueError(
                f"NASController expects sequence features (batch, seq, features), got {tuple(x.shape)}"
            )
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected feature dim {self.input_dim}, got {x.size(-1)}")

        weights = self.get_weights(mode)
        outputs = [op(x) for op in self.ops.values()]
        final_output = sum(w * out for w, out in zip(weights, outputs))
        return self.classifier(final_output)

    def get_arch_info(self):
        weights = F.softmax(self.alpha, dim=-1)
        return {name: float(weight) for name, weight in zip(self.ops.keys(), weights)}

    def discretize(self):
        with torch.no_grad():
            return list(self.ops.keys())[torch.argmax(self.alpha).item()]

    def get_temperature(self, epoch, max_epochs=None, temp_max=None, temp_min=None, schedule=None):
        max_epochs = self.max_epochs if max_epochs is None else max_epochs
        temp_max = self.initial_temperature if temp_max is None else temp_max
        temp_min = self.temp_min if temp_min is None else temp_min
        schedule = self.temp_decay_schedule if schedule is None else schedule
        if max_epochs <= 1:
            return temp_min
        progress = min(max(epoch, 0), max_epochs - 1) / (max_epochs - 1)
        if schedule == "cosine":
            return temp_min + (temp_max - temp_min) * (1 + np.cos(np.pi * progress)) / 2
        if schedule == "exponential":
            return temp_max * (temp_min / temp_max) ** progress
        return self.temperature

    def set_temperature(self, epoch):
        self.temperature = float(self.get_temperature(epoch))


class TCNBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            padding = dilation * (kernel_size - 1) // 2
            layers.extend([
                nn.Conv1d(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            ])
        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.global_pool(self.network(x)).squeeze(-1)


class DilatedConv1DBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, dilation=2):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=dilation, dilation=dilation)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = torch.relu(self.conv(x))
        return self.global_pool(out).squeeze(-1)


class SmallConv1DBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, causal=True):
        super().__init__()
        self.causal = causal
        self.conv = nn.Conv1d(
            input_dim,
            hidden_dim,
            kernel_size,
            padding=(kernel_size - 1) if causal else (kernel_size - 1) // 2,
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.conv(x)
        if self.causal:
            out = out[:, :, :x.size(2)]
        out = torch.relu(out)
        return self.global_pool(out).squeeze(-1)


class DepthwiseSeparableConv1DBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.depthwise = nn.Conv1d(
            input_dim,
            input_dim,
            kernel_size,
            groups=input_dim,
            padding=(kernel_size - 1) // 2,
        )
        self.pointwise = nn.Conv1d(input_dim, hidden_dim, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = torch.relu(self.pointwise(self.depthwise(x)))
        return self.global_pool(out).squeeze(-1)
