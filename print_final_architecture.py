import torch
from src.nas.nas_training import NASTrainer

trainer = NASTrainer(
    ssl_checkpoint_path="models/ssl_best_model.pt",
    batch_size=32,
    w_lr=1e-3,
    a_lr=1e-4
)
# Load the best NAS model
checkpoint = torch.load("models/nas_best_model.pt", map_location='cpu')
trainer.controller.load_state_dict(checkpoint['model_state_dict'])
final_arch = trainer.controller.discretize()
print(f"Final selected architecture: {final_arch}")