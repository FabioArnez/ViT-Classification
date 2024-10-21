import torch
import torch.nn as nn
from torch.nn import functional as F
from vit_pytorch import ViT
from transformers import ViTConfig
import pytorch_lightning as pl
import torchmetrics


class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = ViT(image_size = config.image_size,
                         patch_size = config.patch_size,
                         num_classes = config.num_classes,
                         dim = config.hidden_size,
                         depth = config.num_hidden_layers,
                         heads = config.num_attention_heads,
                         mlp_dim = config.intermediate_size,
                         dropout = config.hidden_dropout_prob,
                         emb_dropout = config.attention_probs_dropout_prob)
    
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
            
        self.apply(_init)
        nn.init.constant_(self.model.fc.weight, 0)
        nn.init.constant_(self.model.fc.bias, 0)
    
    def forward(self, x):
        return self.model(x)