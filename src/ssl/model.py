# Create `src/ssl/model.py`
import torch
import torch.nn as nn
import torch.nn.functional as F

class SSLModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, temp=0.07, proj_dim=64):
        super().__init__()
        self.temperature = temp
        
        # Encoder for financial time series
        self.encoder = nn.Sequential(
            nn.LSTM(input_dim, hidden_dim, num_layers=2, 
                   dropout=0.1, batch_first=True, bidirectional=True),
            nn.LayerNorm([hidden_dim * 2])
        )
        
        # Projection head with layer normalization
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        
    def encode(self, x):
        # Get LSTM output
        output, (hidden, _) = self.encoder[0](x)
        # Use last hidden state from both directions
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        # Apply layer normalization
        hidden = self.encoder[1](hidden)
        return hidden
        
    def forward(self, x1, x2):
        # Forward pass through encoder and projector
        z1 = self.projector(self.encode(x1))
        z2 = self.projector(self.encode(x2))
        
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        return z1, z2
    
    def info_nce_loss(self, z1, z2):
        batch_size = z1.shape[0]
        
        # Concatenate representations
        z = torch.cat([z1, z2], dim=0)
        
        # Compute similarities
        sim = torch.mm(z, z.t().contiguous()) / self.temperature
        
        # Create masks for positive pairs
        sim_ij = torch.diag(sim, batch_size)
        sim_ji = torch.diag(sim, -batch_size)
        positive_pairs = torch.cat([sim_ij, sim_ji], dim=0)
        
        # Create mask for negative pairs
        negative_pairs = sim[~torch.eye(2*batch_size, dtype=bool)].view(2*batch_size, -1)
        
        # Compute loss
        numerator = torch.exp(positive_pairs)
        denominator = torch.sum(torch.exp(negative_pairs), dim=1)
        loss = -torch.log(numerator / denominator)
        loss = torch.mean(loss)
        
        return loss