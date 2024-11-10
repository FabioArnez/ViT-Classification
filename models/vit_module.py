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
import os


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
        self.qkv_bias = model_config.qkv_bias
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

        if model_config.pretrained_path is not None:
            self.pretrained_path = model_config.pretrained_path

class VisionTransformerModule(pl.LightningModule):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        vit_config = ViTConfigExtended(model_config=self.config.model)
        self.model = VisionTransformer(vit_config)
        if self.config.model.pretrained_path is not None:
            self._load_partial_weights()
        self.train_acc = torchmetrics.Accuracy(task="multiclass",
                                               num_classes=self.config.model.num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass",
                                             num_classes=self.config.model.num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass",
                                              num_classes=self.config.model.num_classes)
        self.loss_fn = self._get_loss_fn(self.config.model.loss_fn)
        self.save_hyperparameters()

    def _load_partial_weights(self):
        ckpt_path = "./pretrained_models/jx_vit_base_patch16_224_in21k-e5005f0a.pth"
        jax_weight_dict = torch.load(ckpt_path)
        jax_weight_dict_new = self._update_weight_dict_keys(jax_weight_dict)
        # for key in jax_weight_dict_new:
        #     self._save_weight_dict(root_path="./pretrained_models/",
        #                            file_name="weights_jax_model_dict_new.txt",
        #                            key=key,
        #                            shape=jax_weight_dict_new[key].shape)
        model_dict = self.model.state_dict()
        state_dict = {k:v for k,v in jax_weight_dict_new.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.model.load_state_dict(model_dict)

    def _update_weight_dict_keys(self, jax_dict):
        new_dict = {}

        def add_item(key, value):
            key = key.replace('blocks', 'model.transformer.layers')
            new_dict[key] = value
            
        for key, value in jax_dict.items():
            if key == 'cls_token':
                key = key.replace('cls_token', 'model.cls_token')
                new_dict[key] = value
            elif 'patch_embed.proj' in key:
                new_key = key.replace('patch_embed.proj', 'model.to_patch_embedding.0')
                add_item(new_key, value)
            elif key == 'pos_embed':
                key = key.replace('pos_embed', 'model.pos_embedding')
                new_dict[key] = value
            elif key == 'norm.weight':
                key = key.replace('norm.weight', 'model.transformer.norm.weight')
                new_dict[key] = value
            elif key == 'norm.bias':
                key = key.replace('norm.bias', 'model.transformer.norm.bias')
                new_dict[key] = value
            
            elif 'norm1' in key:
                new_key = key.replace('norm1', '0.norm')
                add_item(new_key, value)
            elif 'attn.qkv' in key:
                new_key = key.replace('attn.qkv', '0.to_qkv')
                add_item(new_key, value)
            elif 'attn.proj' in key:
                new_key = key.replace('attn.proj', '0.to_out.0')
                add_item(new_key, value)
            elif 'norm2' in key:
                new_key = key.replace('norm2', '1.net.0')
                add_item(new_key, value)
            elif 'mlp.fc1' in key:
                new_key = key.replace('mlp.fc1', '1.net.1')
                add_item(new_key, value)
            elif 'mlp.fc2' in key:
                new_key = key.replace('mlp.fc2', '1.net.4')
                add_item(new_key, value)

        return new_dict
    
    def _save_weight_dict(self, root_path, file_name, key, shape):
        file_path = os.path.join(root_path, file_name)
        fo = open(file_path, "a")
        string = key + '\t' + str(shape) + '\n'
        fo.write(string)
        fo.close()

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
        test_accuracy = self.test_acc(pred, label)
        
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