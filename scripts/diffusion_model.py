"""
Copyright © Rowan Ramamurthy, 2025.
This software is provided "as-is," without any express or implied warranty. 
In no event shall the authors be held liable for any damages arising from 
the use of this software.

Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it 
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

Author: Rowan Ramamurthy
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt, cos, pi
from model_base import BaseConversionModel

class SpeechFeatureNormalizer:
    """Handles feature scaling for speech parameters"""
    def __init__(self, feature_ranges=None):
        self.feature_stats = {}
        self.fitted = False
        
    def fit(self, features: np.ndarray):
        self.feature_stats = {
            'mean': features.mean(axis=0),
            'std': features.std(axis=0),
            'min': features.min(axis=0),
            'max': features.max(axis=0)
        }
        self.fitted = True
        
    def normalize(self, features: np.ndarray) -> np.ndarray:
        assert self.fitted, "Normalizer not fitted!"
        return (features - self.feature_stats['mean']) / self.feature_stats['std']
    
    def denormalize(self, norm_features: np.ndarray) -> np.ndarray:
        return norm_features * self.feature_stats['std'] + self.feature_stats['mean']

class ResidualBlock(nn.Module):
    """Basic residual block with accent conditioning"""
    def __init__(self, dim, accent_embed_dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.acc_proj = nn.Linear(accent_embed_dim, dim*2)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, accent_emb):
        scale, shift = torch.chunk(self.acc_proj(accent_emb), 2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return F.relu(x + self.fc(x))

class DiffusionModel(BaseConversionModel):
    def __init__(self, num_accents=10, feature_dim=80, 
                 embed_dim=128, time_dim=256, num_steps=1000):
        super().__init__(num_accents, feature_dim)
        
        # Network components
        self.accent_embed = nn.Embedding(num_accents, embed_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.blocks = nn.ModuleList([
            ResidualBlock(feature_dim, embed_dim)
            for _ in range(4)
        ])
        
        self.final_fc = nn.Linear(feature_dim, feature_dim)
        
        # Diffusion parameters
        self.num_steps = num_steps
        self.betas = self._cosine_schedule()
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # Feature normalization
        self.normalizer = SpeechFeatureNormalizer()

    def _cosine_schedule(self):
        """Cosine noise schedule"""
        s = 0.008
        steps = self.num_steps + 1
        x = torch.linspace(0, self.num_steps, steps)
        f_t = torch.cos(((x / self.num_steps) + s) / (1 + s) * pi * 0.5) ** 2
        alpha_bar = f_t / f_t[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return torch.clip(betas, 0.0001, 0.02)

    def forward(self, x, t, accent_ids):
        # Embeddings
        t_embed = self.time_embed(self._timestep_embedding(t))
        acc_embed = self.accent_embed(accent_ids)
        
        # Merge embeddings
        h = x + t_embed.unsqueeze(1)
        
        # Process through residual blocks
        for block in self.blocks:
            h = block(h, acc_embed)
            
        return self.final_fc(h)

    def _timestep_embedding(self, t):
        """Create sinusoidal timestep embeddings"""
        half_dim = self.time_embed[0].in_features // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

    def loss_fn(self, x0, accent_ids):
        """Calculate diffusion loss with random timesteps"""
        batch_size = x0.size(0)
        t = torch.randint(0, self.num_steps, (batch_size,), device=x0.device)
        
        noise = torch.randn_like(x0)
        alpha_bar = self.alpha_bars[t].unsqueeze(-1).unsqueeze(-1)
        
        noisy = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
        pred_noise = self(noisy, t, accent_ids)
        
        return F.mse_loss(pred_noise, noise)

    def convert(self, input_params, target_accent):
        self.eval()
        with torch.no_grad():
            # Normalize input
            normalized = self.normalizer.normalize(input_params)
            x_t = torch.tensor(normalized).float().unsqueeze(0)
            
            # Diffusion process
            for t in reversed(range(self.num_steps)):
                t_tensor = torch.full((1,), t, dtype=torch.long)
                acc_tensor = torch.tensor([target_accent], dtype=torch.long)
                
                pred_noise = self(x_t, t_tensor, acc_tensor)
                alpha_t = self.alphas[t]
                alpha_bar_t = self.alpha_bars[t]
                
                if t > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = torch.zeros_like(x_t)
                
                x_t = (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * pred_noise) / sqrt(alpha_t)
                x_t += sqrt(1 - alpha_t) * noise
            
            # Denormalize and return
            converted = self.normalizer.denormalize(x_t.squeeze(0).numpy())
            return converted