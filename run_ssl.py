import logging
from src.ssl.ssl_training import train_ssl

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Train SSL model
    train_ssl(
        batch_size=32,
        epochs=30,
        lr=1e-3,
        temperature=0.07
    )
