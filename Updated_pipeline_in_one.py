# ====================================
# PyTorch and Core Libraries
# ====================================
import os
import math
import argparse
from functools import partial

# ====================================
# Data Handling
# ====================================
import numpy as np
import pandas as pd
import nibabel as nib

# ====================================
# Deep Learning
# ====================================
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

# ====================================
# Metrics and Progress
# ====================================
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# ====================================
# UTILITIES from udip_vit_merged.py
# ====================================

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Truncated normal initialization helper function."""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization."""
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# ====================================
# POSITIONAL EMBEDDINGS from udip_vit_merged.py
# ====================================

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate 1D sinusoidal positional embeddings from grid positions."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

def get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=False):
    """Generate 3D sinusoidal positional embeddings."""
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(grid_h, grid_d, grid_w)

    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim/6)*2)

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]
    
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

# ====================================
# PATCH EMBEDDING from udip_vit_merged.py
# ====================================

class PatchEmbed3D(nn.Module):
    """3D Volume to Patch Embedding."""
    def __init__(self, patch_size=14, tubelet_size=16, in_chans=1, embed_dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        # Use (tubelet, patch, patch) ordering so temporal/depth dimension comes first
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(patch_size, tubelet_size,patch_size),
            #kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(patch_size,tubelet_size,  patch_size)
            #stride=(tubelet_size, patch_size, patch_size)
        )

    def forward(self, x, **kwargs):
        # The UDIP model expects (B, C, D, H, W)
        # Our data loader provides (B, D, H, W), so we add the channel dim
        x = x.unsqueeze(1)  # Add channel dimension if needed
        B, C, T, H, W = x.shape

        #x = x.permute(0, 1, 3, 2, 4)  # (B, T, H, W, C) for Conv3d
        
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# ====================================
# ATTENTION AND MLP MODULES from udip_vit_merged.py
# ====================================

class MLP(nn.Module):
    """Multi-Layer Perceptron."""
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0., use_sdpa=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        # Try to use SDPA, but fall back if it's not available (e.g., older PyTorch)
        self.use_sdpa = use_sdpa and hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_sdpa:
            # scaled_dot_product_attention does not return attention weights
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob)
            attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 grid_size=None, grid_depth=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False, mask=None):
        y, attn = self.attn(self.norm1(x), mask=mask)
        if return_attention:
            return attn
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x

# ====================================
# VISION TRANSFORMER ENCODER from udip_vit_merged.py
# ====================================

class VisionTransformer(nn.Module):
    """Vision Transformer encoder with optional non-zero patch optimization."""
    def __init__(self, img_size=182, patch_size=14, num_frames=224, tubelet_size=16,
                 in_chans=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
                 norm_layer=nn.LayerNorm, init_std=0.02, out_layers=None,
                 uniform_power=False, non_zero_patch_opt=True, **kwargs):
        super().__init__()
        self.non_zero_patch_opt = non_zero_patch_opt
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers

        self.input_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size

        self.grid_size = self.input_size // self.patch_size
        self.grid_depth = self.num_frames // self.tubelet_size

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, tubelet_size=tubelet_size,
            in_chans=in_chans, embed_dim=embed_dim)
        
        self.num_patches = self.grid_size * self.grid_depth * self.grid_size

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                act_layer=nn.GELU, grid_size=self.grid_size, grid_depth=self.grid_depth,
                attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        # Initialize weights
        self._init_pos_embed(self.pos_embed.data)
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        """Initialize positional embeddings with sine-cosine."""
        embed_dim = pos_embed.size(-1)
        sincos = get_3d_sincos_pos_embed(embed_dim, self.grid_size, self.grid_depth, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        """Initialize module weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        """Rescale transformer block weights for better initialization."""
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def non_zero_patch(self, x, y):
        """Filter out patches that are all zeros across the batch (match reference)."""
        #non_zero_mask = y.view(
            #-1, self.grid_depth, self.tubelet_size, self.grid_size, self.patch_size,
            #self.grid_size, self.patch_size).sum((2,4,6)) != 0
        non_zero_mask = y.view(
            -1,  self.grid_size, self.patch_size,self.grid_depth, self.tubelet_size,
            self.grid_size, self.patch_size).sum((2,4,6)) != 0
        non_zero_mask = non_zero_mask.view(-1, self.grid_size * self.grid_depth * self.grid_size)
        batch_mask = non_zero_mask.max(0)[0]
        x = x[:, batch_mask, :]
        return x, batch_mask, non_zero_mask

    def forward(self, x, y):
        """Forward pass through vision transformer (match reference)."""
        x = self.patch_embed(x)
        x += self.pos_embed
        if self.non_zero_patch_opt:
            x, batch_mask, non_zero_patch = self.non_zero_patch(x, y)
        else:
            batch_mask = torch.ones(x.shape[1], dtype=torch.bool, device=x.device)
            non_zero_patch = torch.ones((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
        for blk in self.blocks:
            x = blk(x)
        if self.norm is not None:
            x = self.norm(x)
        return x, batch_mask, non_zero_patch


# ====================================
# DECODER from udip_vit_merged.py
# ====================================

class Decoder(nn.Module):
    """UDIP-style decoder with positional token concatenation."""
    def __init__(self, img_size=182, patch_size=14, num_frames=224, tubelet_size=16,
                 embed_dim=384, decoder_embed_dim=192, depth=4, num_heads=6,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0.0, norm_layer=nn.LayerNorm, init_std=0.02):
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.input_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        
        self.num_patches = (self.input_size // self.patch_size) ** 2 * (self.num_frames // self.tubelet_size)
        self.grid_size = self.input_size // self.patch_size
        self.grid_depth = self.num_frames // self.tubelet_size
        
        # Positional embedding for decoder
        self.pos_emb = nn.Parameter(
            torch.zeros(1, self.num_patches, self.decoder_embed_dim), requires_grad=False)
        self._init_pos_embed(self.pos_emb.data)
        
        # Decoder layers
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.decoder_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                act_layer=nn.GELU, grid_size=self.grid_size, grid_depth=self.grid_depth,
                attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(self.decoder_embed_dim)
        self.decoder_embed = nn.Linear(self.embed_dim, self.decoder_embed_dim, bias=True)
        
        # Output projection
        patch_numel = self.patch_size ** 2 * self.tubelet_size
        self.decoder_pred = nn.Linear(self.decoder_embed_dim, patch_numel, bias=True)
        
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        """Rescale block weights."""
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_pos_embed(self, pos_embed):
        """Initialize positional embeddings."""
        embed_dim = pos_embed.size(-1)
        sincos = get_3d_sincos_pos_embed(embed_dim, self.grid_size, self.grid_depth, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def unpatchify(self, x, batch_mask, imgs):
        """Reconstruct image from patches."""
        B = x.shape[0]
        # Create a full tensor of zeros for all patches
        full_patches = torch.zeros(B, self.num_patches, x.shape[-1], device=x.device)
        full_patches.fill_(imgs[0,0,0,0])  # Fill with a constant value (e.g., zero)
        # Place the predicted patches into the correct positions
        full_patches[:, batch_mask, :] = x
        
        # Reshape to image dimensions
        x = full_patches.view(
            B,  self.grid_size,self.grid_depth, self.grid_size, 
             self.patch_size,self.tubelet_size, self.patch_size
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, self.input_size, self.num_frames, self.input_size)
        return x

    def forward(self, x, batch_mask):
        """Forward pass through decoder."""
        # Get positional embeddings for active patches
        pos_emb = self.pos_emb[:, batch_mask, :].repeat(x.shape[0], 1, 1)
        num_latent = x.shape[1]
        
        # Project encoder output
        x = self.decoder_embed(x)
        
        # Concatenate memory tokens with positional tokens (UDIP strategy)
        x = torch.cat([x, pos_emb], dim=1)
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        if self.norm is not None:
            x = self.norm(x)
        
        # Predict from positional tokens only
        x = self.decoder_pred(x[:, num_latent:, :])
        return x

# ====================================
# MAIN MODEL WRAPPER (replaces engine_AE)
# ====================================

class UDIPViT_engine(nn.Module):
    """
    Wrapper for UDIP-ViT components, designed to be a drop-in replacement for engine_AE.
    It handles the single-modality case and matches the forward pass of udip_vit_merged.py.
    """
    def __init__(self, lr, patch_size=14, tubelet_size=16,
                 img_size=182, num_frames=224, in_chans=1, 
                 encoder_embed_dim=384, decoder_embed_dim=192, 
                 encoder_depth=12, decoder_depth=12, num_heads=6, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
                 norm_layer=nn.LayerNorm, init_std=0.02, non_zero_patch_opt=True,
                 # Unused args for compatibility
                 concat_modalities=False, use_modality='T1', use_sincos_pos_embed=True, 
                 use_patchwise_loss=True):
        super().__init__()
        self.lr = lr # for optimizer
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.image_size = img_size
        self.num_frames = num_frames
        
        self.grid_size = self.image_size // self.patch_size
        self.grid_depth = self.num_frames // self.tubelet_size
        
        # Encoder
        self.encoder = VisionTransformer(
            img_size, patch_size, num_frames, tubelet_size, in_chans, encoder_embed_dim,
            encoder_depth, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate,
            attn_drop_rate, norm_layer, init_std, None, False, non_zero_patch_opt)
        
        # Decoder
        self.decoder = Decoder(
            img_size=img_size, patch_size=patch_size, num_frames=num_frames, tubelet_size=tubelet_size,
            embed_dim=encoder_embed_dim, decoder_embed_dim=decoder_embed_dim, 
            depth=decoder_depth, num_heads=num_heads)

    def forward_loss(self, pred, imgs, batch_mask, non_zero_mask):
        target = imgs.view(\
        (-1, self.grid_size, self.patch_size, self.grid_depth, self.tubelet_size,\
         self.grid_size, self.patch_size)).permute(0,1,3,5,2,4,6).reshape((-1, 13*14*13, 14*16*14))
        
        
        # Select only active patches
        target = target[:, batch_mask, :]
        mask = non_zero_mask[:, batch_mask]
        
        # MSE loss
        loss = ((target - pred)**2).mean(-1)
        return (loss * mask).sum() / mask.sum()
    
    def forward(self, imgs, y):
        """Forward pass; expects x_T1 shape (B,H,D,W) like reference UDIP pipeline."""

        # Encode (imgs kept in original ordering expected by patch_embed; y_for_mask used only for zero-patch masking)
        latent, batch_mask, non_zero_mask = self.encoder(imgs, y)
        
        # Average pool latent representations
        compute_pool = latent.mean(1, keepdim=True)
        
        # Decode
        pred = self.decoder(compute_pool, batch_mask)
        
        # Compute loss
        loss = self.forward_loss(pred, imgs, batch_mask, non_zero_mask)
        
        return compute_pool, loss, pred, batch_mask

      

# Validation function for the new architecture
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
from tqdm import tqdm

def validate_one_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_samples = 0
    
    val_pbar = tqdm(dataloader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for batch in val_pbar:
            x_T1, mask = batch
            x_T1 = x_T1.to(device)
            # x_T2 is loaded by dataset but not used by the new model
            mask = mask.to(device)

            # Model forward pass returns loss directly
            _, loss, pred_patches, batch_mask = model(x_T1,mask)
            
            # The loss is already the mean over the batch
            total_loss += loss.item()
            val_pbar.set_postfix({'Val Loss': f'{loss.item():.6f}'})

            # Unpatchify for metrics calculation
            m = model.module if hasattr(model, 'module') else model
            recon_T1 = m.decoder.unpatchify(pred_patches, batch_mask,x_T1)

            # Metrics per sample
            for i in range(x_T1.shape[0]):
                #gt_T1 = x_T1[i].permute(1,0,2).cpu().numpy() # (D, H, W)
                gt_T1 = x_T1[i].cpu().numpy() # (H, D, W)
                pred_T1 = recon_T1[i].detach().cpu().numpy() # (D, H, W)
                #msk_3d = mask[i].permute(1,0,2).cpu().numpy().astype(bool) # (D, H, W)
                msk_3d = mask[i].cpu().numpy().astype(bool) # (H, D, W)
                if msk_3d.sum() > 0:
                    gt_T1_masked = gt_T1[msk_3d]
                    pred_T1_masked = pred_T1[msk_3d]
                    dr_T1 = gt_T1_masked.max() - gt_T1_masked.min()
                    psnr_T1 = compare_psnr(gt_T1_masked, pred_T1_masked, data_range=dr_T1) if dr_T1 > 0 else 0.0
                    try:
                        dr_T1_full = gt_T1.max() - gt_T1.min()
                        ssim_T1 = compare_ssim(gt_T1, pred_T1, data_range=dr_T1_full,mask=msk_3d) if dr_T1_full > 0 else 0.0
                    except Exception:
                        ssim_T1 = 0.0
                    total_psnr += psnr_T1
                    total_ssim += ssim_T1
                n_samples += 1
                
    # Calculate average loss per sample processed by this GPU
    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / n_samples if n_samples > 0 else 0.0
    avg_ssim = total_ssim / n_samples if n_samples > 0 else 0.0
    
    # Synchronize validation metrics across all GPUs
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        device = next(model.parameters()).device
        
        loss_tensor = torch.tensor(avg_loss, device=device)
        psnr_tensor = torch.tensor(avg_psnr, device=device)
        ssim_tensor = torch.tensor(avg_ssim, device=device)
        
        torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(psnr_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(ssim_tensor, op=torch.distributed.ReduceOp.SUM)
        
        avg_loss = loss_tensor.item() / world_size
        avg_psnr = psnr_tensor.item() / world_size
        avg_ssim = ssim_tensor.item() / world_size
    
    return avg_loss, avg_psnr, avg_ssim

# ====================================
# DATASET
# ====================================
import pandas as pd
import nibabel as nib



class aedataset(torch.utils.data.Dataset):
    def __init__(self, datafile, modality):
        """
        Args:
            datafile (type: csv or list): the datafile mentioning the location of images or a list of file locations.
            modality (type: string): column containing location of modality of interest in the datafile.
            transforms (type: pytorch specific transforms): to add channel to the image and convert to tensor.
        Returns:
            img [torch tensor]: img file normalized 
            mask [torch tensor]: mask excluding background
            img_name [string]: name of the image
        """
        self.datafile = pd.read_csv(datafile)
        self.unbiased_brain = self.datafile[modality]

    def __len__(self):
        return len(self.unbiased_brain)

    def __getitem__(self, idxx=int):
        img_name = self.unbiased_brain[idxx]
        img = nib.load(img_name)
        img = img.get_fdata()
        img = torch.from_numpy(img)
        img = torch.nn.functional.pad(img, (0,0,3,3,0,0)) # padding image from 182x218x182 to 182x224x182
        # padding needs to be done before normalization
        mask = img != 0
        img = (img - img[img != 0].mean()) / img[img != 0].std()
        img = img.type(torch.float)
        #mask = mask.int()
        return img, mask

    
# ====================================
# MAIN TRAINING SCRIPT
# ====================================
if __name__ == "__main__":
    import os
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.data.distributed import DistributedSampler
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train DeepENDO ViT model')
    parser.add_argument('--resume', type=str, help='path to checkpoint to resume from')
    parser.add_argument('--start-epoch', type=int, default=0, help='start epoch (default: 0)')
    args = parser.parse_args()

    # Set PyTorch memory allocator configuration
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
    
    # CUDA settings for better memory management
    torch.backends.cudnn.benchmark = True
    
    # DDP setup
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    
    print(f"Environment variables: RANK={rank}, WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
    
    if world_size > 1:
        print(f"Initializing process group with rank={rank}, world_size={world_size}")
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        is_main_process = (rank == 0)
    else:
        print("Not running in DDP mode - environment variables not found")
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print("CUDA not available, using CPU")
        is_main_process = True

    print(f'Rank: {rank}, Using CUDA device: {torch.cuda.current_device() if torch.cuda.is_available() else "CPU"}')
    
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Override for reproducibility
    
    # Create ViT model with improved architecture
    AE_model = UDIPViT_engine(
        lr=0.001,
        patch_size=14,
        tubelet_size=16,
        img_size=182,
        num_frames=224,
        in_chans=1,
        encoder_embed_dim=128,
        decoder_embed_dim=64,
        encoder_depth=12,
        decoder_depth=12,
        num_heads=8,
        non_zero_patch_opt=True,
        use_patchwise_loss=True,
    )
    AE_model = AE_model.to(device)
    print(f"Model successfully loaded on device: {device}")
    
    if dist.is_available() and dist.is_initialized():
        AE_model = DDP(AE_model, device_ids=[local_rank], find_unused_parameters=True) # find_unused_parameters=True to handle unused parameters
            
    optimizer = torch.optim.AdamW(AE_model.parameters(), lr=AE_model.module.lr if hasattr(AE_model, 'module') else AE_model.lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    #from torch.optim.lr_scheduler import StepLR
    #scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    # Resume from checkpoint if specified
    start_epoch = args.start_epoch
    best_val_loss = float('inf')
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            #start_epoch = checkpoint['epoch']
            start_epoch = checkpoint.get('epoch', 0)
            #best_val_loss = checkpoint['best_val_loss']
            
            if hasattr(AE_model, 'module'):
                AE_model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                AE_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Robust optimizer state restore (handle missing 'step' or shape mismatches)
            opt_state = checkpoint.get('optimizer_state_dict')
            def _sanitize_opt_state(state_dict, optimizer):
                if not state_dict or 'state' not in state_dict:
                    return state_dict
                import torch as _torch
                # Ensure param_groups contain necessary hyperparams
                if 'param_groups' in state_dict:
                    default_group = optimizer.param_groups[0]
                    required_keys = ['betas','lr','eps','weight_decay']
                    for g in state_dict['param_groups']:
                        for k in required_keys:
                            if k not in g and k in default_group:
                                g[k] = default_group[k]
                        # Adam/AdamW require betas tuple
                        if 'betas' not in g:
                            g['betas'] = default_group.get('betas', (0.9,0.999))
                        # Clamp malformed betas
                        if isinstance(g['betas'], list):
                            g['betas'] = tuple(g['betas'])
                        if not (isinstance(g['betas'], tuple) and len(g['betas'])==2):
                            g['betas'] = (0.9, 0.999)
                for pid, p_state in state_dict['state'].items():
                    if not isinstance(p_state, dict):
                        continue
                    # Ensure step exists and is a tensor (Adam(W) expects tensor here in newer versions)
                    if 'step' not in p_state:
                        p_state['step'] = _torch.tensor(0.)
                    elif not _torch.is_tensor(p_state['step']):
                        try:
                            p_state['step'] = _torch.as_tensor(p_state['step'], dtype=_torch.float32)
                        except Exception:
                            p_state['step'] = _torch.tensor(0.)
                    # If exp_avg / exp_avg_sq missing, initialize lazily later by dropping entry
                    if 'exp_avg' not in p_state or 'exp_avg_sq' not in p_state:
                        # Easiest safe fallback: clear this param's state so optimizer will re-init
                        state_dict['state'][pid] = {}
                return state_dict
            try:
                opt_state = _sanitize_opt_state(opt_state, optimizer)
                optimizer.load_state_dict(opt_state)
            except KeyError as e:
                print(f"[Resume] Warning: Optimizer state missing keys ({e}). Reinitializing optimizer state.")
            except Exception as e:
                print(f"[Resume] Warning: Failed to load optimizer state ({e}). Continuing with fresh optimizer.")
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Loaded checkpoint. Resuming from epoch {start_epoch}")

    # Initialize gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    dir_name = "/data484_4/txia2/DeepENDO/training/T1_128/output/vit_t1_fixed_replicate3/"
    os.makedirs(dir_name, exist_ok=True)

    if is_main_process:
        writer = SummaryWriter(log_dir=os.path.join(dir_name, "tb_logs"))
    else:
        writer = None

    # DataLoaders
    train_dataset = aedataset(
        datafile="/data4012/kpatel38/backups/autoencoder_ethnicity/train_mixed_ethnicity.csv",
        modality="T1_unbiased_linear"
        
    )
    val_dataset = aedataset(
        datafile="/data4012/kpatel38/backups/autoencoder_ethnicity/val_mixed_ethnicity.csv",
        modality="T1_unbiased_linear"
    )
    
    batch_size = 4
    num_workers = 4
    
    if dist.is_available() and dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, pin_memory=True,
            num_workers=num_workers, sampler=train_sampler, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, pin_memory=True,
            num_workers=num_workers, sampler=val_sampler, drop_last=False
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, pin_memory=True, 
            num_workers=num_workers, shuffle=True, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, pin_memory=True,
            num_workers=num_workers, shuffle=False, drop_last=False
        )

    # Training Loop
    num_epochs = 300
    for epoch in range(start_epoch, num_epochs):
        if dist.is_available() and dist.is_initialized():
            train_sampler.set_epoch(epoch)
            
        AE_model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for i, (x_T1, mask) in enumerate(pbar):
            x_T1 = x_T1.to(device)
            
            with torch.cuda.amp.autocast():
                _, loss, _, _ = AE_model(x_T1,mask)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            running_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
            
            if writer and is_main_process:
                writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_loader) + i)

        avg_train_loss = running_loss / len(train_loader)
        if writer and is_main_process:
            writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
            writer.add_scalar('LR/train', optimizer.param_groups[0]['lr'], epoch)
        
        scheduler.step()

        # Validation
        avg_val_loss, avg_psnr, avg_ssim = validate_one_epoch(AE_model, val_loader, device)
        if writer and is_main_process:
            writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
            writer.add_scalar('Metrics/val_psnr', avg_psnr, epoch)
            writer.add_scalar('Metrics/val_ssim', avg_ssim, epoch)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Val PSNR: {avg_psnr:.4f}, Val SSIM: {avg_ssim:.4f}")

        # Save checkpoint
        if is_main_process:
            is_best = avg_val_loss < best_val_loss
            best_val_loss = min(avg_val_loss, best_val_loss)
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': AE_model.module.state_dict() if hasattr(AE_model, 'module') else AE_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'avg_train_loss': avg_train_loss,
                'avg_val_loss': avg_val_loss,
                'avg_psnr': avg_psnr,
                'avg_ssim': avg_ssim
            }
            
            # Always save latest checkpoint
            save_path = os.path.join(dir_name, 'latest_checkpoint.pth')
            torch.save(checkpoint, save_path)
            
            # Save best model
            if is_best:
                best_path = os.path.join(dir_name, 'best_model.pth')
                torch.save(checkpoint, best_path)
                print(f"New best model saved at epoch {epoch+1} with val_loss: {avg_val_loss:.6f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                epoch_save_path = os.path.join(dir_name, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save(checkpoint, epoch_save_path)
                print(f"Checkpoint saved at epoch {epoch+1}: {epoch_save_path}")

    if writer and is_main_process:
        writer.close()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    print("Training finished.")
