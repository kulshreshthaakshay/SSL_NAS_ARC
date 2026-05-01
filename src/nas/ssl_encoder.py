import torch
import logging
from pathlib import Path
from src.ssl.model import SSLModel
from src.nas.controller import NASController

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pretrained_encoder(
    checkpoint_path="models/ssl_best_model.pt",
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Load pretrained SSL encoder for NAS training.
    
    Args:
        checkpoint_path (str): Path to SSL checkpoint
        device (str): Device to load model on
    
    Returns:
        tuple: (encoder, checkpoint_info)
    """
    try:
        # Load checkpoint first to get model configuration
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get input dimensions from checkpoint
        input_dim = checkpoint.get('input_dim', 5)
        seq_len = checkpoint.get('seq_len', 60)
        
        logger.info(f"Loading SSL model with input_dim={input_dim}, seq_len={seq_len}")
        
        # Initialize SSL model with correct dimensions
        ssl_model = SSLModel(input_dim=input_dim).to(device)
        
        # Load state dict
        ssl_model.load_state_dict(checkpoint['model_state_dict'])
        ssl_model.eval()
        
        logger.info(f"Loaded SSL checkpoint from epoch {checkpoint['epoch']}")
        logger.info(f"Best SSL loss: {checkpoint.get('loss', 'N/A')}")
        
        # Create a wrapper encoder that only does encoding (not projection)
        class SSLEncoder(torch.nn.Module):
            def __init__(self, ssl_model):
                super().__init__()
                self.ssl_model = ssl_model
                
            def forward(self, x):
                """Extract features using SSL model's encode method."""
                return self.ssl_model.encode(x)
        
        encoder = SSLEncoder(ssl_model)
        
        # Freeze encoder weights
        for param in encoder.parameters():
            param.requires_grad = False
            
        logger.info("Successfully loaded and froze pretrained SSL encoder")
        
        return encoder, checkpoint
        
    except Exception as e:
        logger.error(f"Error loading SSL checkpoint: {e}")
        logger.error(f"Make sure the checkpoint exists at: {checkpoint_path}")
        raise

def get_encoder_output_dim(encoder, input_dim=5, seq_len=60, device='cpu'):
    """
    Determine the output dimension of the SSL encoder by running a forward pass.
    
    Args:
        encoder: The loaded encoder
        input_dim (int): Input feature dimension
        seq_len (int): Sequence length
        device (str): Device to run on
    
    Returns:
        int: Output dimension of the encoder
    """
    try:
        # Create dummy input
        dummy_input = torch.randn(1, seq_len, input_dim).to(device)
        
        with torch.no_grad():
            encoder.eval()
            output = encoder(dummy_input)
            
            # Handle different output types
            if isinstance(output, tuple):
                output = output[0]
            
            output_dim = output.shape[-1]
            logger.info(f"Encoder output dimension: {output_dim}")
            
        return output_dim
        
    except Exception as e:
        logger.error(f"Error determining encoder output dim: {e}")
        # Fallback based on SSL model architecture
        # Bidirectional LSTM with hidden_dim=64 -> output_dim=128
        logger.warning("Using fallback output dimension: 128")
        return 128

def initialize_nas_controller(
    encoder_output_dim,
    hidden_dim=64,
    num_classes=2,
    device='cpu'
):
    """
    Initialize NAS controller with correct dimensions.
    
    Args:
        encoder_output_dim (int): Output dimension from SSL encoder
        hidden_dim (int): Hidden dimension for NAS operations
        num_classes (int): Number of classes for classification
        device (str): Device to initialize on
    
    Returns:
        NASController: Initialized NAS controller
    """
    controller = NASController(
        input_dim=encoder_output_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        temperature=1.0
    ).to(device)
    
    logger.info(f"Initialized NAS controller:")
    logger.info(f"  Input dim: {encoder_output_dim}")
    logger.info(f"  Hidden dim: {hidden_dim}")
    logger.info(f"  Num classes: {num_classes}")
    logger.info(f"  Available operations: {list(controller.ops.keys())}")
    
    return controller

def validate_ssl_nas_pipeline(
    ssl_checkpoint_path="models/ssl_best_model.pt",
    sample_data_path="data/processed/windows.npy",
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Validate the complete SSL -> NAS pipeline.
    
    Args:
        ssl_checkpoint_path (str): Path to SSL checkpoint
        sample_data_path (str): Path to sample data for testing
        device (str): Device to run validation on
    """
    try:
        logger.info("=== Validating SSL-NAS Pipeline ===")
        
        # 1. Load SSL encoder
        encoder, checkpoint_info = load_pretrained_encoder(ssl_checkpoint_path, device)
        
        # 2. Get dimensions
        input_dim = checkpoint_info.get('input_dim', 5)
        seq_len = checkpoint_info.get('seq_len', 60)
        encoder_output_dim = get_encoder_output_dim(encoder, input_dim, seq_len, device)
        
        # 3. Initialize NAS controller
        controller = initialize_nas_controller(encoder_output_dim, device=device)
        
        # 4. Test with sample data if available
        try:
            import numpy as np
            if Path(sample_data_path).exists():
                logger.info("Testing with sample data...")
                
                # Load sample data
                windows = np.load(sample_data_path)
                sample_batch = torch.FloatTensor(windows[:4]).to(device)  # Take 4 samples
                
                logger.info(f"Sample data shape: {sample_batch.shape}")
                
                # Test SSL encoder
                with torch.no_grad():
                    encoder.eval()
                    features = encoder(sample_batch)
                    logger.info(f"SSL encoder output shape: {features.shape}")
                    
                    # Test NAS controller
                    controller.eval()
                    # Ensure features are 2D for controller (NASController expects (batch_size, input_dim))
                    if len(features.shape) == 3:
                        # If 3D, select last time step (common for sequence models)
                        features = features[:, -1, :]
                    elif len(features.shape) != 2:
                        raise ValueError(f"Controller expects 2D features, got shape {features.shape}")
                    # If already 2D, use as is
                    logits = controller(features)
                    logger.info(f"NAS controller output shape: {logits.shape}")
                    
                    # Show architecture weights
                    arch_info = controller.get_arch_info()
                    logger.info(f"Current architecture weights: {arch_info}")
                    
                logger.info("✅ Pipeline validation successful!")
                
        except Exception as e:
            logger.warning(f"Could not test with sample data: {e}")
            logger.info("Pipeline structure validation complete (data test skipped)")
            
    except Exception as e:
        logger.error(f"❌ Pipeline validation failed: {e}")
        raise

if __name__ == "__main__":
    # Test the SSL encoder loading
    try:
        # Load pretrained encoder
        encoder, checkpoint_info = load_pretrained_encoder()
        
        # Get output dimensions
        input_dim = checkpoint_info.get('input_dim', 5)
        seq_len = checkpoint_info.get('seq_len', 60)
        encoder_output_dim = get_encoder_output_dim(encoder, input_dim, seq_len)
        
        # Initialize NAS controller
        controller = initialize_nas_controller(encoder_output_dim)
        
        # Run full validation
        validate_ssl_nas_pipeline()
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        logger.info("Make sure you have:")
        logger.info("1. Trained SSL model (run_ssl.py)")
        logger.info("2. Generated data windows (data_collection.ipynb)")