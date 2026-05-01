import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from pathlib import Path
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.ssl.model import SSLModel
from src.utils.augmentations import TimeSeriesAugmentor
from financial_considerations import MarketRegimeDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_project_root():
    """Get absolute path to project root."""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent

def train_ssl(
    batch_size=32,
    epochs=75,
    lr=1e-3,
    temperature=0.07,  # Standard InfoNCE temperature
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """Train SSL model with contrastive learning."""
    logger.info(f"Training on device: {device}")
    logger.info(f"Hyperparameters: batch_size={batch_size}, epochs={epochs}, lr={lr}, temp={temperature}")
    
    # Load and prepare data
    try:
        project_root = get_project_root()
        # Use training data for SSL pretraining
        data_path = project_root / "data" / "processed" / "windows_train.npy"
        
        # Create directories if they don't exist
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if data exists
        if not data_path.exists():
            logger.error(f"Data file not found at {data_path}")
            logger.info("Please run notebooks/datacollection.py first to generate the data")
            return
            
        windows = np.load(data_path)
        logger.info(f"Loaded training data shape: {windows.shape}")
        
        # Get input dimensions from data
        if len(windows.shape) == 3:
            seq_len, input_dim = windows.shape[1], windows.shape[2]
        else:
            raise ValueError(f"Expected 3D data (batch, seq_len, features), got shape {windows.shape}")
        
        logger.info(f"Detected input dimensions: sequence_length={seq_len}, features={input_dim}")
        logger.info(f"Training samples: {windows.shape[0]} (LOW-DATA REGIME)")
        
        dataset = TensorDataset(torch.FloatTensor(windows))
        # Windows compatibility: set num_workers=0 on Windows
        num_workers = 0 if os.name == 'nt' else 4
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        logger.info(f"Successfully loaded data from {data_path}")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # Initialize model with correct input dimensions
    model = SSLModel(input_dim=input_dim, temp=temperature).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    augmentor = TimeSeriesAugmentor(seed=42)  # Fixed seed for reproducibility

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Encoder output dim: 128 (bidirectional LSTM)")
    logger.info(f"Projection dim: 64")

    # Training loop
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            x = batch[0].to(device)
            
            try:
                # Create augmented views using composite transform
                x1 = torch.stack([
                    torch.FloatTensor(augmentor.composite_transform(w.cpu().numpy(), p=0.5))
                    for w in x
                ]).to(device)
                x2 = torch.stack([
                    torch.FloatTensor(augmentor.composite_transform(w.cpu().numpy(), p=0.5))
                    for w in x
                ]).to(device)

                # Forward pass
                z1, z2 = model(x1, x2)
                loss = model.info_nce_loss(z1, z2)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Update progress
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                logger.error(f"Input shape: {x.shape}")
                raise e
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch results
        avg_loss = total_loss / len(loader)
        logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            # Create models directory if it doesn't exist
            model_dir = get_project_root() / "models"
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / "ssl_best_model.pt"
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'input_dim': input_dim,
                'seq_len': seq_len,
                'temperature': temperature,
            }, model_path)
            logger.info(f"✅ Saved best model checkpoint (loss: {best_loss:.4f}) to {model_path}")

    logger.info(f"Training completed. Best loss: {best_loss:.4f}")
    return model

def visualize_embeddings(model, windows, regime_labels, device='cpu', sample_size=500):
    """
    Visualize SSL embeddings using t-SNE, colored by market regime.
    Args:
        model: Trained SSLModel
        windows: np.ndarray of shape (N, seq_len, features)
        regime_labels: np.ndarray of shape (N,)
        device: torch device
        sample_size: number of samples to plot
    """
    model.eval()
    idx = np.random.choice(len(windows), min(sample_size, len(windows)), replace=False)
    sample_windows = torch.FloatTensor(windows[idx]).to(device)
    
    with torch.no_grad():
        # Get SSL embeddings (use encode method)
        embeddings = model.encode(sample_windows).cpu().numpy()
    
    # t-SNE
    logger.info("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=regime_labels[idx], 
                         cmap='viridis', alpha=0.7, s=20)
    plt.colorbar(scatter, label='Market Regime')
    plt.title('t-SNE of SSL Embeddings Colored by Market Regime', fontsize=14)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    save_path = results_dir / 't-SNE_SSL_Embeddings.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved t-SNE visualization to {save_path}")
    plt.show()

if __name__ == "__main__":
    # Train SSL model
    model = train_ssl(
        batch_size=32,
        epochs=75,
        lr=1e-3,
        temperature=0.07
    )
    
    # After training, visualize embeddings
    if model is not None:
        try:
            project_root = get_project_root()
            data_path = project_root / "data" / "processed" / "windows_train.npy"
            windows = np.load(data_path)
            
            # Load best model
            model_path = project_root / "models" / "ssl_best_model.pt"
            checkpoint = torch.load(model_path, map_location='cpu')
            input_dim = checkpoint.get('input_dim', windows.shape[2])
            model = SSLModel(input_dim=input_dim)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            logger.info("Generating market regime labels for visualization...")
            # Detect regimes using returns (assuming first feature contains price-related info)
            # This is a simplified approach - adjust based on your actual features
            try:
                # Calculate returns from the data
                # Assuming normalized data, we'll use the first feature as proxy
                close_proxy = windows[:, -1, 0]  # Last timestep, first feature
                returns = np.diff(close_proxy)
                returns = np.concatenate([[0], returns])  # Pad to match length
                
                detector = MarketRegimeDetector()
                regimes = detector.detect_regimes(returns)
                
                # Visualize
                visualize_embeddings(model, windows, regimes, device='cpu', 
                                   sample_size=min(500, len(windows)))
            except Exception as e:
                logger.warning(f"Could not generate regime visualization: {e}")
                logger.info("Skipping visualization - this is optional")
                
        except Exception as e:
            logger.error(f"Error in embedding visualization: {e}")