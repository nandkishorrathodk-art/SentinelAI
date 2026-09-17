import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Splits an image into patches and embeds them via a 2D convolution.
    
    This acts as the first layer in Vision Transformers (ViTs), projecting 
    (B, C, H, W) images to (B, N, D) sequences of visual tokens.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            Tensor of shape (B, N, D) where N = H*W / P^2 and D = embed_dim
        """
        B, C, H, W = x.shape
        # Transform x -> (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x
