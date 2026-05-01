import logging
from pathlib import Path
from src.nas.nas_training import NASTrainer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    project_root = Path(__file__).parent
    
    # Initialize trainer
    trainer = NASTrainer(
        ssl_checkpoint_path=str(project_root / "models" / "ssl_best_model.pt"),
        batch_size=32,
        w_lr=1e-3,
        a_lr=3e-4
    )
    
    # Prepare data with financial-specific labels
    weight_data, arch_data, valid_data = trainer.prepare_financial_data()
    
    # Train NAS
    trainer.train(
        weight_data=weight_data,
        arch_data=arch_data,
        valid_data=valid_data,
        epochs=50,
        early_stopping_patience=10
    )
