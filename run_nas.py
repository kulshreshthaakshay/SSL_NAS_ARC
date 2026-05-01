import logging
from src.nas.nas_training import NASTrainer
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize trainer
    trainer = NASTrainer(
        ssl_checkpoint_path="models/ssl_best_model.pt",
        batch_size=32,
        w_lr=1e-3,
        a_lr=1e-4
    )
    
    # Prepare data with financial-specific labels
    train_data, valid_data = trainer.prepare_financial_data(  # Fixed method name
        window_path="data/processed/windows.npy",
        val_size=0.2
    )
    
    # Train NAS
    trainer.train(
        train_data=train_data,
        valid_data=valid_data,
        epochs=50,
        early_stopping_patience=10
    )