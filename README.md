# SSL-NAS: Self-Supervised Learning with Neural Architecture Search for Financial Time Series

A unified framework that combines **Self-Supervised Learning (SSL)** pre-training with **Differentiable Neural Architecture Search (NAS)** for financial time series classification in **low-data regimes**.

---

## Overview

Financial time series prediction is challenging due to non-stationarity, low signal-to-noise ratios, and the scarcity of labeled data. This framework addresses these challenges through a two-stage pipeline:

1. **SSL Pre-training** — Learn universal temporal representations from unlabeled financial data using contrastive learning (InfoNCE loss) with domain-specific augmentations.
2. **NAS-based Fine-tuning** — Automatically discover the optimal downstream architecture via differentiable architecture search using Gumbel-Softmax relaxation, while the pre-trained encoder remains frozen.

### Key Features

- **Low-data regime focus**: Designed for scenarios with limited labeled financial data (~600 training windows from 1 year of daily data across 5 stocks).
- **Finance-aware augmentations**: Calibrated noise injection, volatility scaling, magnitude warping, and trend-preserving transformations that respect the statistical properties of financial time series.
- **Bilevel NAS optimization**: Architecture parameters and model weights are optimized alternately on validation and training sets respectively, following the DARTS paradigm.
- **Causal design**: Time-series splits with no look-ahead bias — the scaler is fit on training data only.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SSL-NAS Framework                           │
│                                                                 │
│  Stage 1: Self-Supervised Pre-training                          │
│  ┌───────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ Raw OHLCV │───▶│ Augmentation │───▶│ Bidirectional LSTM   │  │
│  │  Windows  │    │  (2 views)   │    │ Encoder + Projector  │  │
│  └───────────┘    └──────────────┘    └──────────────────────┘  │
│                                          │ InfoNCE Loss         │
│                                                                 │
│  Stage 2: Differentiable Architecture Search                    │
│  ┌───────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Frozen   │───▶│  NAS Search  │───▶│    Classifier        │  │
│  │  Encoder  │    │  Space:      │    │    (2-class)         │  │
│  │ (128-dim) │    │  • Conv1D    │    └──────────────────────┘  │
│  └───────────┘    │  • TCN       │                              │
│                   │  • Dilated   │                              │
│                   │  • DepthSep  │                              │
│                   └──────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

### NAS Search Space

| Operation | Description |
|---|---|
| **Conv1D** | Causal 1D convolution (kernel=3) with global average pooling |
| **TCN** | Temporal Convolutional Network with 2 layers and exponential dilation |
| **Dilated Conv1D** | Dilated convolution (dilation=2) for larger receptive field |
| **Depthwise Separable Conv1D** | Parameter-efficient factored convolution |

---

## Project Structure

```
SSL_NAS_ARC/
├── run_ssl.py                       # Entry point: SSL pre-training
├── run_nas.py                       # Entry point: NAS training
├── evaluate.py                      # Evaluation on test set
├── financial_considerations.py      # Feature engineering, regime detection, financial metrics
├── print_final_architecture.py      # Print the NAS-selected architecture
├── visualize_ssl.py                 # t-SNE visualization of SSL embeddings
├── visualize_nas.py                 # NAS architecture weight evolution plots
├── notebooks/
│   └── datacollection.py            # Data collection and preprocessing pipeline
├── src/
│   ├── ssl/
│   │   ├── model.py                 # SSL model (Bidirectional LSTM + Projector)
│   │   └── ssl_training.py          # Contrastive learning training loop
│   ├── nas/
│   │   ├── controller.py            # NAS controller with Gumbel-Softmax
│   │   ├── nas_training.py          # Bilevel NAS optimization trainer
│   │   └── ssl_encoder.py           # SSL encoder loading utilities
│   └── utils/
│       └── augmentations.py         # Finance-specific time series augmentations
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/kulshreshthaakshay/SSL_NAS_ARC.git
cd SSL_NAS_ARC

# Create and activate a virtual environment
python -m venv ssl-nas-env
# Windows
ssl-nas-env\Scripts\activate
# Linux/Mac
source ssl-nas-env/bin/activate

# Install dependencies
pip install torch torchvision numpy pandas scikit-learn matplotlib tqdm yfinance scipy arch hmmlearn
```

### Dependencies

| Package | Purpose |
|---|---|
| `torch` | Deep learning framework |
| `numpy`, `pandas` | Data manipulation |
| `scikit-learn` | Metrics, preprocessing, t-SNE |
| `yfinance` | Financial data download |
| `scipy` | Interpolation for augmentations |
| `arch` | GARCH volatility modeling |
| `hmmlearn` | Market regime detection (HMM) |
| `matplotlib` | Visualization |
| `tqdm` | Progress bars |

---

## Usage

The pipeline follows three sequential stages:

### 1. Data Collection

```bash
python notebooks/datacollection.py
```

Downloads 1 year (2022) of daily OHLCV data for 5 stocks (AAPL, JPM, XOM, JNJ, WMT), engineers 14 financial features (returns, RSI, MACD, Bollinger Bands, GARCH volatility, VWAP, etc.), and creates normalized 30-day sliding windows split into train/val/test sets.

### 2. SSL Pre-training

```bash
python run_ssl.py
```

Trains the bidirectional LSTM encoder using InfoNCE contrastive loss with finance-appropriate augmentations. Saves the best model checkpoint to `models/ssl_best_model.pt`.

### 3. NAS Training

```bash
python run_nas.py
```

Loads the frozen SSL encoder and performs differentiable architecture search over the candidate operations using bilevel optimization. The architecture parameters are updated on the validation set, while the model weights are updated on the training set. Saves the best model to `models/nas_best_model.pt`.

### 4. Evaluation

```bash
python evaluate.py
```

Evaluates the trained SSL-NAS pipeline on the held-out test set and reports classification metrics (accuracy, precision, recall, F1, AUC-ROC) and financial metrics (Sharpe ratio, maximum drawdown, directional accuracy).

### 5. Visualization

```bash
python visualize_ssl.py            # t-SNE of SSL embeddings by market regime
python visualize_nas.py            # Architecture weight evolution over epochs
python print_final_architecture.py # Print the selected architecture
```

---

## Results

### Classification Metrics

| Metric | Value |
|---|---|
| Accuracy | 0.76 |
| Precision | 0.78 |
| Recall | 0.76 |
| F1-Score | 0.76 |
| AUC-ROC | 0.82 |

### Financial Metrics

| Metric | Value |
|---|---|
| Sharpe Ratio | 8.13 |
| Max Drawdown | −1.44% |
| Directional Accuracy | 76.47% |

### NAS Architecture Selection

The differentiable search converged to **Causal Conv1D** (26.40%) as the preferred operation, followed by Depthwise Separable Conv1D (26.28%), Dilated Conv1D (26.13%), and TCN (21.19%).

---

## Methodology Details

### Self-Supervised Pre-training

- **Encoder**: 2-layer Bidirectional LSTM (hidden_dim=64, output_dim=128)
- **Projector**: 2-layer MLP with LayerNorm (128 → 64 → 64)
- **Loss**: InfoNCE contrastive loss with temperature τ=0.07
- **Augmentations**: Composite of calibrated noise, feature dropout, volatility scaling, magnitude warping, and outlier injection (each applied with p=0.5)

### Neural Architecture Search

- **Search Strategy**: DARTS-style continuous relaxation with Gumbel-Softmax
- **Temperature Schedule**: Cosine annealing from τ=5.0 → 0.1 over training
- **Bilevel Optimization**: Architecture params (Adam, lr=4e-4) on validation set; model weights (Adam, lr=1e-3) on training set
- **Entropy Regularization**: Penalty of 0.01 on architecture weight entropy to encourage decisive selection

### Data Pipeline

- **Stocks**: AAPL, JPM, XOM, JNJ, WMT (5 sectors for generalization)
- **Period**: Jan 2022 — Jan 2023 (low-data regime)
- **Features**: Returns, log-returns, RSI, MACD, Bollinger Bands, realized volatility, GARCH volatility, momentum, volume z-score, OBV, VWAP (14 features)
- **Windows**: 30-day sliding windows
- **Split**: 60% train / 20% val / 20% test (temporal, no shuffling)
- **Normalization**: MinMaxScaler fit on training data only (no look-ahead bias)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ssl_nas_arc_2026,
  title={Self-Supervised Learning with Neural Architecture Search for Financial Time Series Classification in Low-Data Regimes},
  author={Kulshreshtha, Akshay},
  year={2026}
}
```

---

## License

This project is for academic and research purposes. Please contact the author for commercial use inquiries.
