import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import json
import os
from src.ssl.model import SSLModel
from src.nas.controller import NASController

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NASTrainer:
    def __init__(
        self,
        ssl_checkpoint_path="models/ssl_best_model.pt",
        batch_size=32,
        w_lr=1e-3,  # weight learning rate
        a_lr=4e-4,  # architecture learning rate
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.ssl_checkpoint_path = ssl_checkpoint_path  # Store for later use
        self.device = device
        self.batch_size = batch_size
        
        # Load pretrained SSL encoder
        self.encoder, self.ssl_info = self._load_pretrained_encoder(ssl_checkpoint_path)
        
        # Get encoder output dimension
        encoder_dim = self._get_encoder_output_dim()
        
        # Initialize NAS controller with correct dimensions
        self.controller = NASController(
            input_dim=encoder_dim,
            hidden_dim=64,
            num_classes=2,
            temperature=5.0
        ).to(device)
        
        # Setup optimizers for bilevel optimization
        arch_params = list(self.controller.arch_parameters())
        arch_param_ids = set(id(p) for p in arch_params)
        weight_params = [p for n, p in self.controller.named_parameters() if id(p) not in arch_param_ids]

        self.w_optimizer = optim.Adam(weight_params, lr=w_lr)
        self.a_optimizer = optim.Adam(arch_params, lr=a_lr)
        
        logger.info(f"Initialized NAS controller with encoder dim: {encoder_dim}")
        
        self.arch_weights_history = []  # Add this line to store weights history
        
    def _load_pretrained_encoder(self, checkpoint_path):
        """Load pretrained SSL encoder with proper wrapper."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Get model info from checkpoint
            input_dim = checkpoint.get('input_dim', 5)
            
            # Initialize SSL model with correct dimensions
            ssl_model = SSLModel(input_dim=input_dim).to(self.device)
            ssl_model.load_state_dict(checkpoint['model_state_dict'])
            ssl_model.eval()
            
            # Create encoder wrapper that only does encoding (not projection)
            class SSLEncoderWrapper(nn.Module):
                def __init__(self, ssl_model):
                    super().__init__()
                    self.ssl_model = ssl_model
                    
                def forward(self, x):
                    """Use SSL model's encode method for feature extraction."""
                    return self.ssl_model.encode(x)
            
            encoder = SSLEncoderWrapper(ssl_model)
            
            # Freeze encoder parameters
            for param in encoder.parameters():
                param.requires_grad = False
            
            logger.info(f"Loaded SSL encoder from {checkpoint_path}")
            logger.info(f"SSL model trained for {checkpoint.get('epoch', 'unknown')} epochs")
            
            return encoder, checkpoint
            
        except Exception as e:
            logger.error(f"Error loading SSL checkpoint: {e}")
            logger.error("Make sure you have trained the SSL model first using run_ssl.py")
            raise
    
    def _get_encoder_output_dim(self):
        """Determine the output dimension of the SSL encoder."""
        try:
            # Get dimensions from SSL checkpoint
            input_dim = self.ssl_info.get('input_dim', 5)
            seq_len = self.ssl_info.get('seq_len', 60)
            
            # Create dummy input
            dummy_input = torch.randn(1, seq_len, input_dim).to(self.device)
            
            with torch.no_grad():
                output = self.encoder(dummy_input)
                encoder_dim = output.shape[-1]
            
            logger.info(f"Detected encoder output dimension: {encoder_dim}")
            return encoder_dim
            
        except Exception as e:
            logger.warning(f"Could not determine encoder output dim: {e}")
            # Fallback: bidirectional LSTM with hidden_dim=64 gives output_dim=128
            logger.info("Using fallback encoder dimension: 128")
            return 128
    
    def train(
        self,
        train_data,
        valid_data,
        epochs=73,
        early_stopping_patience=10
    ):
        """
        Train NAS controller using proper bilevel optimization.
        """
        # Prepare data loaders
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=self.batch_size, shuffle=True)
        
        best_valid_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Phase 1: Update architecture parameters using validation set
            self.controller.train()
            arch_loss = self._update_architecture_params(valid_loader)
            
            # Phase 2: Update model weights using training set
            self.controller.train()
            weight_loss, train_acc = self._update_weight_params(train_loader)
            
            # Validation phase
            self.controller.eval()
            valid_loss, valid_acc = self._validate(valid_loader)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"Arch Loss: {arch_loss:.4f}, Weight Loss: {weight_loss:.4f}")
            logger.info(f"Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}")
            
            # Log architecture weights
            arch_info = self.controller.get_arch_info()
            sorted_arch = sorted(arch_info.items(), key=lambda x: x[1], reverse=True)
            logger.info(f"Architecture Weights: {dict(sorted_arch)}")

            # Save architecture weights for visualization
            self.arch_weights_history.append(arch_info.copy())

            # Early stopping based on validation accuracy
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                patience_counter = 0
                self._save_checkpoint(epoch, valid_loss, valid_acc)
            else:
                patience_counter += 1
                
            # Commented out early stopping
            # if patience_counter >= early_stopping_patience:
            #     logger.info("Early stopping triggered")
            #     break
        
        # Log final architecture and accuracy
        final_arch = self.controller.discretize()
        logger.info(f"\n=== Final Evaluation Results ===")
        logger.info(f"Selected architecture: {final_arch}")
        logger.info(f"Best validation accuracy: {best_valid_acc:.4f}")
        
        # Compute and display final F1 score
        self.controller.eval()
        all_preds = []
        all_labels = []
        valid_loader = DataLoader(valid_data, batch_size=self.batch_size)
        
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(self.device), y.to(self.device)
                features = self.encoder(x)
                if features.dim() == 3:
                    features = features[:, -1, :]
                logits = self.controller(features)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        # Calculate and display final metrics
        final_acc = accuracy_score(all_labels, all_preds)
        final_f1 = f1_score(all_labels, all_preds, average='weighted')
        final_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        final_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        # For AUC, need predicted probabilities and binary labels
        try:
            # Get predicted probabilities for class 1
            all_probs = []
            for x, y in valid_loader:
                x = x.to(self.device)
                features = self.encoder(x)
                if features.dim() == 3:
                    features = features[:, -1, :]
                logits = self.controller(features)
                probs = torch.softmax(logits, dim=1)[:, 1]  # Probability for class 1
                all_probs.extend(probs.cpu().numpy())
            final_auc = roc_auc_score(all_labels, all_probs)
        except Exception as e:
            final_auc = None
            logger.warning(f"Could not compute AUC: {e}")

        logger.info(f"Final validation accuracy: {final_acc:.4f}")
        logger.info(f"Final validation F1 score: {final_f1:.4f}")
        logger.info(f"Final validation precision: {final_precision:.4f}")
        logger.info(f"Final validation recall: {final_recall:.4f}")
        if final_auc is not None:
            logger.info(f"Final validation AUC: {final_auc:.4f}")
        logger.info("================================")
       
        # Save architecture weights history as JSON after training to absolute path
        results_dir = Path(r"E:\Projects\ssl_nas_arc\SSL_NAS_ARC\results")
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / "arch_weights_history.json"
        with open(results_path, "w") as f:
            json.dump(self.arch_weights_history, f, indent=2)
        logger.info(f"Saved architecture weights history to {results_path}")
            
    def _update_architecture_params(self, valid_loader):
        """Update architecture parameters using validation set."""
        total_loss = 0
        steps = 0
        for batch in valid_loader:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            # Get SSL features
            with torch.no_grad():
                features = self.encoder(x)
            # Update architecture parameters
            self.a_optimizer.zero_grad()
            logits = self.controller(features)
            loss = nn.CrossEntropyLoss()(logits, y)
            # --- Architecture Regularization: Entropy penalty to encourage decisive weights ---
            alpha_weights = torch.softmax(self.controller.alpha, dim=-1)
            entropy = -torch.sum(alpha_weights * torch.log(alpha_weights + 1e-8))
            entropy_reg = 0.01  # You can tune this value
            loss = loss - entropy_reg * entropy
            # -------------------------------------------------------------------------------
            loss.backward()
            self.a_optimizer.step()
            total_loss += loss.item()
            steps += 1
            # Limit steps to avoid overfitting on validation set
            if steps >= 30:
                break
        return total_loss / steps if steps > 0 else 0

    # Removed unused architecture_loss function - entropy regularization is already applied in _update_architecture_params
    
    def _update_weight_params(self, train_loader):
        """Update model weights using training set."""
        total_loss = 0
        all_preds = []
        all_labels = []
        steps = 0
        
        for batch in tqdm(train_loader, desc="Training Weights"):
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            
            # Get SSL features - ensure proper dimension
            with torch.no_grad():
                features = self.encoder(x)
                if features.dim() == 3:  # If encoder returns sequence
                    features = features[:, -1, :]  # Take last timestep
                assert features.size(1) == self.controller.input_dim, \
                    f"Expected {self.controller.input_dim} features, got {features.size(1)}"
            
            # Update weight parameters
            self.w_optimizer.zero_grad()
            logits = self.controller(features)
            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()
            
            # Modified gradient clipping
            weight_params = []
            for name, param in self.controller.named_parameters():
                if name not in ['alpha']:  # Exclude architecture parameters
                    weight_params.append(param)
            
            torch.nn.utils.clip_grad_norm_(weight_params, max_norm=1.0)
            
            self.w_optimizer.step()
            
            # Collect predictions
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            total_loss += loss.item()
            steps += 1
        
        train_acc = accuracy_score(all_labels, all_preds)
        return total_loss / steps if steps > 0 else 0, train_acc
    def _validate(self, valid_loader):
        """Validate model performance."""
        total_loss = 0
        all_preds = []
        all_labels = []
        steps = 0
        
        with torch.no_grad():
            for batch in valid_loader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                # Get SSL features with proper dimension handling
                features = self.encoder(x)
                if features.dim() == 3:  # If encoder returns sequence
                    features = features[:, -1, :]  # Take last timestep
                
                logits = self.controller(features)
                loss = nn.CrossEntropyLoss()(logits, y)
                
                # Collect predictions
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
                total_loss += loss.item()
                steps += 1
        
        # Calculate validation accuracy
        valid_acc = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / steps if steps > 0 else float('inf')
        
        return avg_loss, valid_acc
    
    def _save_checkpoint(self, epoch, valid_loss, valid_acc):
        """Save model checkpoint."""
        try:
            Path("models").mkdir(exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.controller.state_dict(),
                'w_optimizer_state_dict': self.w_optimizer.state_dict(),
                'a_optimizer_state_dict': self.a_optimizer.state_dict(),
                'valid_loss': valid_loss,
                'valid_acc': valid_acc,
                'arch_info': self.controller.get_arch_info(),
                'best_arch': self.controller.discretize(),
                'ssl_checkpoint_path': self.ssl_checkpoint_path
            }
            torch.save(checkpoint, 'models/nas_best_model.pt')
            logger.info(f"Saved NAS checkpoint - Acc: {valid_acc:.4f}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def prepare_financial_data(self, window_path=None, val_size=0.2):
        """
        Prepare financial data with multiple task types.
        Now uses pre-split train/val data to avoid look-ahead bias.
        """
        try:
            project_root = Path(__file__).parent.parent.parent
            
            # Load pre-split train and validation data
            train_path = project_root / "data" / "processed" / "windows_train.npy"
            val_path = project_root / "data" / "processed" / "windows_val.npy"
            
            if not train_path.exists() or not val_path.exists():
                logger.error(f"Train or val data not found!")
                logger.error(f"Expected: {train_path} and {val_path}")
                logger.info("Please run notebooks/datacollection.py first")
                raise FileNotFoundError("Training/validation data not found")
            
            # Load windows
            train_windows = np.load(train_path)
            val_windows = np.load(val_path)
            
            logger.info(f"Loaded train windows: {train_windows.shape}")
            logger.info(f"Loaded val windows: {val_windows.shape}")
            
            # Create multiple types of labels for robust evaluation
            train_labels = self._create_financial_labels(train_windows)
            val_labels = self._create_financial_labels(val_windows)
            
            # Use price direction as primary task for now
            train_primary = train_labels['price_direction']
            val_primary = val_labels['price_direction']
            
            # Check label distribution
            unique_train, counts_train = np.unique(train_primary, return_counts=True)
            unique_val, counts_val = np.unique(val_primary, return_counts=True)
            logger.info(f"Train label distribution: {dict(zip(unique_train, counts_train))}")
            logger.info(f"Val label distribution: {dict(zip(unique_val, counts_val))}")
            
            # Create datasets
            train_data = TensorDataset(
                torch.FloatTensor(train_windows),
                torch.LongTensor(train_primary)
            )
            valid_data = TensorDataset(
                torch.FloatTensor(val_windows),
                torch.LongTensor(val_primary)
            )
            
            logger.info(f"Prepared training set: {len(train_data)} samples")
            logger.info(f"Prepared validation set: {len(valid_data)} samples")
            
            return train_data, valid_data
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def _create_financial_labels(self, windows):
        """Create multiple types of financial labels."""
        labels = {}
        
        # 1. Price direction (comparing first and last price)
        price_direction = []
        for window in windows:
            first_price = window[0, 0]  # Assuming first feature is price-related
            last_price = window[-1, 0]
            price_direction.append(1 if last_price > first_price else 0)
        
        # 2. Volatility level (based on price std)
        volatility_level = []
        price_stds = []
        for window in windows:
            prices = window[:, 0]
            price_std = np.std(prices)
            price_stds.append(price_std)
        
        # Use median as threshold for high/low volatility
        vol_threshold = np.median(price_stds)
        volatility_level = [1 if std > vol_threshold else 0 for std in price_stds]
        
        # 3. Trend strength (based on linear trend slope)
        trend_strength = []
        trend_slopes = []
        for window in windows:
            prices = window[:, 0]
            x = np.arange(len(prices))
            # Simple linear regression slope
            slope = np.polyfit(x, prices, 1)[0]
            trend_slopes.append(abs(slope))
        
        # Use median as threshold for strong/weak trend
        slope_threshold = np.median(trend_slopes)
        trend_strength = [1 if slope > slope_threshold else 0 for slope in trend_slopes]
        
        labels['price_direction'] = np.array(price_direction)
        labels['volatility_level'] = np.array(volatility_level)
        labels['trend_strength'] = np.array(trend_strength)
        
        logger.info("Created financial labels:")
        for task, task_labels in labels.items():
            unique, counts = np.unique(task_labels, return_counts=True)
            logger.info(f"  {task}: {dict(zip(unique, counts))}")
        
        return labels
if __name__ == "__main__":
    # Initialize trainer
    trainer = NASTrainer(
        ssl_checkpoint_path="models/ssl_best_model.pt",
        batch_size=32,
        w_lr=1e-3,
        a_lr=4e-4
    )
    
    # Prepare data with financial-specific labels (uses pre-split data)
    train_data, valid_data = trainer.prepare_financial_data()
    
    # Train the NAS controller
    trainer.train(
        train_data=train_data,
        valid_data=valid_data,
        epochs=73,
        early_stopping_patience=10
    )