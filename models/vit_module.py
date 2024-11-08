from typing import Optional, Callable, Union, List, Dict, Any
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import ViTConfig
import pytorch_lightning as pl
import torchmetrics
from torch.optim.lr_scheduler import CosineAnnealingLR
from .vit_model import VisionTransformer
from datetime import datetime, date


class ViTConfigExtended():
    def __init__(self, model_config):
        super().__init__()
        if model_config.model_type not in ['ViT-B', 'ViT-L', 'ViT-H', 'custom']:
            raise ValueError(f'`transforms_type` value is not supported. Got "{model_config.model_type}" value.')
        self.vit_model_type = model_config.model_type

        if self.vit_model_type == 'ViT-B':
            self.num_hidden_layers = 12
            self.num_attention_heads = 12
        elif self.vit_model_type == 'ViT-L':
            self.num_hidden_layers = 24
            self.num_attention_heads = 16
        elif self.vit_model_type == 'ViT-H':
            self.num_hidden_layers = 32
            self.num_attention_heads = 16
        else: # custom
            self.num_hidden_layers = model_config.num_hidden_layers
            self.num_attention_heads = model_config.num_attention_heads

        self.hidden_size = model_config.hidden_size
        self.patch_size = model_config.patch_size
        self.intermediate_size = model_config.intermediate_size
        self.hidden_act = model_config.hidden_activation
        self.hidden_dropout_prob = model_config.hidden_dropout_prob
        self.attention_probs_dropout_prob = model_config.attention_probs_dropout_prob
        self.initializer_range = model_config.initializer_range
        self.layer_norm_eps = model_config.layer_norm_eps
        self.is_encoder_decoder = model_config.is_encoder_decoder
        self.image_size = model_config.image_size
        self.num_channels = model_config.input_channels
        self.num_classes = model_config.num_classes
        self.optimizer_lr = model_config.optimizer_lr
        self.optimizer_weight_decay = model_config.optimizer_weight_decay
        
        if model_config.loss_fn not in ['nll', 'cross_entropy', 'focal', 'custom']:
            raise ValueError(f'`loss_fn` value is not supported. Got "{model_config.loss_fn}" value.')
        self.loss_fn = model_config.loss_fn

class VisionTransformerModule(pl.LightningModule):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        vit_config = ViTConfigExtended(model_config=self.config.model)
        self.model = VisionTransformer(vit_config)
        self.train_acc = torchmetrics.Accuracy(task="multiclass",
                                               num_classes=self.config.model.num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass",
                                             num_classes=self.config.model.num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass",
                                              num_classes=self.config.model.num_classes)
        self.loss_fn = self._get_loss_fn(self.config.model.loss_fn)
        self.save_hyperparameters()

    def _get_loss_fn(self, loss_type) -> Any:
        if loss_type == "nll":
            loss_fn = nn.NLLLoss(reduction='mean')
        elif loss_type == "cross_entropy":
            loss_fn = nn.CrossEntropyLoss(reduction='mean')
        # elif loss_type == "focal":
        #     loss_fn = FocalLoss(gamma=4.0, reduction='mean')
        else:
            raise ValueError(f' Loss function `{loss_type}` value is not supported, use a valid loss function')

        return loss_fn

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),  # specify your model (neural network) parameters (weights)
                                     lr=self.config.model.optimizer_lr,  # learning rate
                                     weight_decay=self.config.model.optimizer_weight_decay,  # L2 penalty regularizer
                                     eps=1e-7)  # adds numerical numerical stability (avoids division by 0)
        lr_scheduler = {"scheduler": CosineAnnealingLR(optimizer,
                                            T_max=self.config.model.lr_scheduler.cosine_anneal.T_max,
                                            eta_min=self.config.model.lr_scheduler.cosine_anneal.eta_min)}
        # lr_scheduler = {"scheduler": CosineAnnealingLR(optimizer,
        #                                                T_max=400,
        #                                                eta_min=1e-5)}
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        sample, label = batch
        pred = self.model(sample)  # make a prediction (forward-pass)
        train_loss = self.loss_fn(pred, label)  # compute loss
        train_accuracy = self.train_acc(pred, label)  # compute accuracy
        
        self.log_dict({"Train loss": train_loss,
                       "Train accuracy": train_accuracy},
                       on_step=False, on_epoch=True, prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        sample, label = batch
        pred = self.model(sample)
        valid_loss = self.loss_fn(pred, label)
        valid_accuracy = self.val_acc(pred, label)
        
        self.log_dict({"Validation loss": valid_loss,
                       "Validation accuracy": valid_accuracy},
                      on_step=False, on_epoch=True, prog_bar=True)        

        return valid_loss

    def test_step(self, batch, batch_idx):
        sample, label = batch
        pred = self.model(sample)
        test_loss = self.loss_fn(pred, label)
        test_accuracy = self.val_acc(pred, label)
        
        self.log_dict({"Test loss": test_loss,
                       "Test accuracy": test_accuracy},
                      on_step=False, on_epoch=True, prog_bar=True)   
        return test_loss


def object_to_dict(obj: object) -> Dict[str, Any]:
    """Convert a class object to a dictionary, handling nested objects and special types."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, (list, tuple)):
        return [object_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: object_to_dict(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):
        return object_to_dict(obj.__dict__)
    else:
        return str(obj)