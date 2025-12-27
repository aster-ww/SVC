import numpy as np
import torch
from torch import nn
from functools import reduce
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from model.performer_pytorch import Performer
import os
from typing import Optional, Dict, List
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm.auto import tqdm
from model.utils import *

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class Gene2VecPositionalEmbedding(nn.Module):
    def __init__(self, gene2vec_weight, dim, add_dim = 0):
        super().__init__()

        gene2vec_weight = nn.Linear(200, dim)(gene2vec_weight) ##n_gene * dim
        gene2vec_weight = torch.cat((torch.zeros((1+add_dim, gene2vec_weight.shape[1])), gene2vec_weight), axis=0)
        self.emb = nn.Embedding.from_pretrained(gene2vec_weight)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device) ##n_gene
        return self.emb(t)

class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, *args, **kwargs):
        return self.val


class SVC(nn.Module):
    def __init__(self, *, 
                 gene2vec_weight,
                 image_size = 12, 
                 dim = 384,
                 depth = 12,
                 heads = 12,
                 cell_position = True,
                 cell_morphology = True,
                 nuclear_morphology = True,
                 use_cell_identity = True,
                 cell_morphology_dim = (48,48),
                 nuclear_morphology_dim = (48,48),
                 cell_identity_dim = 32,
                 dim_head = 64, 
                 emb_dropout = 0.,
                 local_attn_heads = 0,
                 local_window_size = 256,
                 causal = False,
                 ff_mult = 4,
                 nb_features = None,
                 feature_redraw_interval = 1000,
                 reversible = False,
                 ff_chunks = 1,
                 ff_glu = False,
                 ff_dropout = 0.,
                 attn_dropout = 0.,
                 generalized_attention = False,
                 kernel_fn = nn.ReLU(),
                 use_scalenorm = False,
                 use_rezero = False,
                 cross_attend = False,
                 no_projection = False,
                 tie_embed = False,                
                 auto_check_redraw = True,
                 g2v_position_emb=True,
                 qkv_bias = False
            ):
        super().__init__()
        patch_height, patch_width = pair(image_size)
        patch_dim = patch_height * patch_width
        
        ### each gene map is a patch. 
        # Input: n * c * h * w  
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h w -> b c (h w)'),
            nn.LayerNorm(patch_dim),   
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        if g2v_position_emb:
            add_dim = 0
            if cell_morphology:
                add_dim +=1
            if nuclear_morphology:
                add_dim +=1
            if use_cell_identity:
                add_dim +=1
            self.pos_emb = Gene2VecPositionalEmbedding(gene2vec_weight, dim, add_dim = add_dim)
            self.layer_pos_emb = Always(None)
        
        else:
            self.pos_emb = torch.zeros_like
            self.layer_pos_emb = Always(None)
        self.add_dim = add_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))### initialize the mask token as zero

        if cell_position:
            self.pos_token = nn.Linear(2, dim)  ##batch, 2 -> batch, dim     
        else:
            self.pos_token = nn.Parameter(torch.randn(1, 1, dim)) 

        if cell_morphology:
            self.cell_morphology_token = nn.Sequential(
            Rearrange('b h w -> b (h w)'),
            nn.LayerNorm(cell_morphology_dim[0]*cell_morphology_dim[1]),   
            nn.Linear(cell_morphology_dim[0]*cell_morphology_dim[1], dim),
            nn.LayerNorm(dim),
        )
        if nuclear_morphology:
            self.nuclear_morphology_token = nn.Sequential(
            Rearrange('b h w -> b (h w)'),
            nn.LayerNorm(nuclear_morphology_dim[0]*nuclear_morphology_dim[1]),   
            nn.Linear(nuclear_morphology_dim[0]*nuclear_morphology_dim[1], dim),
            nn.LayerNorm(dim),
        )
        if use_cell_identity:
            self.cell_identity_token = nn.Linear(cell_identity_dim, dim)
 

        self.dropout = nn.Dropout(emb_dropout)
        self.performer = Performer(dim, depth, heads, dim_head, local_attn_heads, local_window_size, causal, 
                                    ff_mult, nb_features, feature_redraw_interval, reversible, ff_chunks, 
                                    generalized_attention, kernel_fn, use_scalenorm, use_rezero, ff_glu, 
                                    ff_dropout, attn_dropout, cross_attend, no_projection, auto_check_redraw, qkv_bias)

        self.norm = nn.LayerNorm(dim)

        self.mu_head = nn.Sequential(nn.Linear(dim, patch_dim),
                                    Rearrange('b c (h w) ->b c h w ', h = patch_height, w = patch_width),
                                    )
        self.r_head = nn.Sequential(nn.Linear(dim, patch_dim),
                                    Rearrange('b c (h w) ->b c h w ', h = patch_height, w = patch_width),
                                    )


    def forward(self, img, mask, location, cell_morphology_vec=None, nuclear_morphology_vec=None, cell_identity_vec=None, output_attentions = False):

        x = self.to_patch_embedding(img)

        b, l , _ = x.shape

        mask_token = self.mask_token.expand(b, l, -1)
        w = mask.unsqueeze(-1).type_as(mask_token) # (B, L, 1)
        x = x * (1 - w) + mask_token * w  

        if location is not None:
            pos_tokens = self.pos_token(location).unsqueeze(1)
        else:
            pos_tokens = repeat(self.pos_token, '1 1 d -> b 1 d', b = b)

        if cell_morphology_vec is not None:
            cell_morphology_tokens = self.cell_morphology_token(cell_morphology_vec).unsqueeze(1)

        if nuclear_morphology_vec is not None:
            nuclear_morphology_tokens = self.nuclear_morphology_token(nuclear_morphology_vec).unsqueeze(1)

        if cell_identity_vec is not None:
            cell_identity_tokens = self.cell_identity_token(cell_identity_vec).unsqueeze(1)

        if cell_morphology_vec is None and nuclear_morphology_vec is None and cell_identity_vec is None:
            x = torch.cat((pos_tokens, x), dim=1)    
        elif cell_morphology_vec is not None and nuclear_morphology_vec is not None and cell_identity_vec is not None:
            x = torch.cat((pos_tokens, cell_morphology_tokens, nuclear_morphology_tokens, cell_identity_tokens, x), dim=1)
        elif cell_morphology_vec is not None and nuclear_morphology_vec is not None and cell_identity_vec is None:
            x = torch.cat((pos_tokens, cell_morphology_tokens, nuclear_morphology_tokens, x), dim=1)

        x = x + self.pos_emb(x)
        x = self.dropout(x) ## b * gene * dim
  
  
        # performer layers
        layer_pos_emb = self.layer_pos_emb(x)
        if output_attentions:
            x, attn = self.performer(x, pos_emb = layer_pos_emb, output_attentions = output_attentions)
        else:
            x = self.performer(x, pos_emb = layer_pos_emb)

        encoding = self.norm(x) # Shape: (b, c, dim)

        mu, r = torch.exp(self.mu_head(encoding)), torch.exp(self.r_head(encoding)) # Shape: (b, c, h, w)

        if output_attentions:
            return encoding[:,1+self.add_dim:,:], mu[:,1+self.add_dim:,:,:],r[:,1+self.add_dim:,:,:], attn#, morphology[:,1,:]
        else:
            return encoding[:,1+self.add_dim:,:], mu[:,1+self.add_dim:,:,:], r[:,1+self.add_dim:,:,:]#, morphology[:,1,:]


def save_ckpt(model_name, ckpt_folder, model):
    """
    save checkpoint
    """
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    # torch.save(model, f'{ckpt_folder}{model_name}epochs_{epoch}.pth')
    print(f'Saving model checkpoint to {ckpt_folder}{model_name}.pth')
    torch.save(
        {
            'model_state_dict': model.state_dict(),#.module.
        },
        f'{ckpt_folder}{model_name}.pth'
    )


def train_SVC(
    model: torch.nn.Module,
    train_loader: DataLoader,
    cell_median_train: float,
    device: torch.device,
    num_epochs: int,
    learning_rate: float = 1e-4,
    mask_prob: float = 0.1,
    foreground_expanded: torch.Tensor = None,
    weight_decay: float = 1e-2,
    accum_iter: int = 1,
    ckpt_dir: Optional[str] = None,
    ckpt_name: str = "SVC_pretrain",
    use_cell_identity: bool = True,
    use_epoch_bar: bool = True,
    early_stop_patience: int = 0,     
    early_stop_min_delta: float = 0.0  
) -> List[float]:
    """
    Train SVC with masked gene prediction and negative binomial loss.

    Args:
        model: SVC model. Should return (embeddings, mu, r) in forward.
        train_loader: DataLoader yielding
            (cell_morphology, nuclear_morphology, size_factor,
             location, cell_identity, inputs_ori, inputs).
        device: torch.device, e.g. torch.device("cuda").
        num_epochs: Number of training epochs.
        learning_rate: Initial learning rate.
        mask_prob: Fraction of genes to mask per sample (0 to 1).
        foreground_expanded: Foreground mask broadcastable to loss_pixel0.
        weight_decay: Weight decay for AdamW.
        accum_iter: Gradient accumulation steps.
        ckpt_dir: Directory to save checkpoint. If None, no checkpoint is saved.
        ckpt_name: Prefix for checkpoint file name.
        use_cell_identity: Whether to feed cell identity embeddings.
        early_stop_patience: Number of epochs with no sufficient improvement
                             after which training will be stopped.
        early_stop_min_delta: Minimal loss improvement to be counted as progress.
    """

    model = model.to(device)
    num_batches = len(train_loader)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps = 60, 
        cycle_mult=2.0,      
        max_lr = learning_rate,  
        min_lr = learning_rate/1000       #1e-5     
    )

    
    epoch_losses: List[float] = []

    if foreground_expanded is None:
        background_pixel = torch.tensor(
            [[0,0],[0,1],[0,10],[0,11],
             [1,0],[1,11],
             [10,0],[10,11],
             [11,0],[11,1],[11,10],[11,11]],
            device=device, dtype=torch.long
        )
        foreground = torch.ones((12, 12), dtype=torch.bool, device=device)
        foreground[background_pixel[:, 0], background_pixel[:, 1]] = False
        foreground_expanded = foreground.reshape(1, 1, 12, 12)

    foreground_expanded = foreground_expanded.to(device)
    if use_epoch_bar:
        epoch_bar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    else:
        epoch_bar = range(num_epochs)

    best_loss = float("inf")
    best_state_dict = None
    no_improve_epochs = 0

    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(train_loader, 0):
            if use_cell_identity:
                (
                    inputs_ori,
                    cell_morphology,
                    nuclear_morphology,
                    location,
                    cell_identity,
                ) = batch
            else:
                (
                    inputs_ori,
                    cell_morphology,
                    nuclear_morphology,
                    location,
                ) = batch
                cell_identity = None

            inputs_ori = inputs_ori.to(device).float()
            cell_morphology = cell_morphology.to(device).float()
            nuclear_morphology = nuclear_morphology.to(device).float()
            if cell_identity is not None:
                cell_identity = cell_identity.to(device).float()
            location = location.to(device).float()

            size_factor = inputs_ori.sum((1, 2, 3)) / cell_median_train
            size_factor = size_factor.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            inputs = inputs_ori / size_factor

            with torch.set_grad_enabled(True):
                rand = torch.rand(inputs.shape[0], inputs.shape[1], device=inputs.device)
                num_mask = int(mask_prob * inputs.shape[1])
                _, idx = torch.topk(-rand, num_mask, dim=1)

                mask = torch.zeros(
                    inputs.shape[0],
                    inputs.shape[1],
                    dtype=torch.bool,
                    device=inputs.device,
                )
                mask.scatter_(1, idx, True)

                # Forward
                if use_cell_identity:
                    _, predicts_mu, predicts_r = model(
                        inputs,
                        mask,
                        location,
                        cell_morphology,
                        nuclear_morphology,
                        cell_identity,
                    )
                else:
                    _, predicts_mu, predicts_r = model(
                        inputs,
                        mask,
                        location,
                        cell_morphology,
                        nuclear_morphology,
                    )

                predicts_mu = predicts_mu * size_factor

                loss_pixel0 = negative_binomial_loss(
                    inputs_ori, predicts_mu, predicts_r
                )
                loss_pixel0_cleaned = loss_pixel0 * foreground_expanded

                loss_per_gene = loss_pixel0_cleaned.sum(dim=(-1, -2))
                loss = (loss_per_gene * mask).sum() / mask.sum()

                loss.backward()
                if ((i + 1) % accum_iter == 0) or (i + 1 == num_batches):
                    optimizer.step()
                    optimizer.zero_grad()

            running_loss += loss.item()

        avg_loss = running_loss / (i + 1)
        epoch_losses.append(avg_loss)

        improved = (avg_loss + early_stop_min_delta) < best_loss
        if improved:
            best_loss = avg_loss
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            if early_stop_patience > 0:
                no_improve_epochs += 1

        if early_stop_patience > 0:
            if use_epoch_bar:
                epoch_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "no_improve": no_improve_epochs,
                    "best": f"{best_loss:.4f}",
                })
            else:
                print(f"Epoch {epoch + 1}/ {num_epochs}: loss={avg_loss:.4f}, no_improve={no_improve_epochs}, best={best_loss:.4f}")
        else:
            if use_epoch_bar:
                epoch_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                })
            else:
                print(f"Epoch {epoch + 1}/ {num_epochs}: loss={avg_loss:.4f}")

        scheduler.step()

        if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"with best loss {best_loss:.4f}"
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    if ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok=True)
        save_ckpt(ckpt_name, ckpt_dir, model)

    print("Finished training")
    return epoch_losses
