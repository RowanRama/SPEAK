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
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn

class BaseConversionModel(ABC, nn.Module):
    """
    Abstract base class for accent conversion models
    Converts speech parameters based on target accent ID
    """
    def __init__(self, num_accents: int, feature_dim: int):
        super().__init__()
        self.num_accents = num_accents
        self.feature_dim = feature_dim

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, accent_ids: torch.Tensor) -> torch.Tensor:
        """
        Neural network forward pass for denoising
        """
        pass

    @abstractmethod
    def convert(self, input_params: np.ndarray, target_accent: int) -> np.ndarray:
        """
        Args:
            input_params: Input speech parameters [shape: (n_frames, n_features)]
            target_accent: Integer representing target accent ID
            
        Returns:
            converted_params: Converted speech parameters
        """
        pass

    @abstractmethod
    def loss_fn(self, x0: torch.Tensor, accent_ids: torch.Tensor) -> torch.Tensor:
        """
        Calculate diffusion loss
        """
        pass
