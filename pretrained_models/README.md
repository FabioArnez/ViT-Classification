# Load Pre-Trained ViT models:

ViT DNNs do not train well on small datasets (e.g., CIFAR-10). Therefore, it is recommended to use a pre-trained model.
<!-- ## Adding Functionality to Load Pre-trained Models -->
If we use the `vit-pytorch` library additional modifications are needed to properly load the weights of the original pre-trained models, e.g., (ViT-B/16). These modifications are described below.


## Modifications in the `vit.py` script
To properly load the parameters of a ViT pre-trained model into a `vit-pytorch` ViT DNN, we need to perform two modifications in the `vit.py`.
To this end, we recommend coping the script `vit.py` in the local project. The modifications are the following:

1. Add the `qkv_bias` parameter in the Attention class and enable the use of the parameters through the `ViT` class.
2. In the `ViT` class use `nn.Conv2d` for the `to_patch_embedding` layer:

    ```python
    self.to_patch_embedding = nn.Sequential(
            # use conv2d to fit the pre-trained weight file
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)
        )
    ```
    instead of:

    ```python
    self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
    ```

    Then, modify the `forward(self, img)` method to match with the `self.to_patch_embedding` layer modifications.
    ```python
    x = self.to_patch_embedding(img) # [B, C, H, W]
        x = x.flatten(2).transpose(1,2) # [B, N, C]
        b, n, _ = x.shape
    ```


## Adding functionality in the `vit_module.py`

Add functionality to load pre-trained weights (ImageNet) to `vit_module.py`:

Code to load weights:
- https://github.com/lucidrains/vit-pytorch/issues/239
- https://github.com/liyiersan/MSA/blob/22243186133369941bb78bbd93e6e2cd04317f66/models/vit.py#L133-L211
- https://github.com/Sebastian-X/vit-pytorch-with-pretrained-weights/tree/master/tools
- https://github.com/Sebastian-X/vit-pytorch-with-pretrained-weights/blob/master/utils/utils.py#L44
links to weights:
- https://github.com/huggingface/pytorch-image-models/releases?page=7
- https://github.com/huggingface/pytorch-image-models/releases/tag/v0.1-vitjx

```
Load the jax->PyTorch(.pth) pretrained weights from timm, note that we remove many unnecessary components (e.g., mlp_head) 
        
Weights can be downloaded from here: https://github.com/huggingface/pytorch-image-models/releases/tag/v0.1-vitjx
    
you can download various pretriained weights and adjust your codes to fit them
ideas from https://github.com/Sebastian-X/vit-pytorch-with-pretrained-weights/blob/master/tools/trans_weight.py

Weights mapping is as follows:
                    -----------Model Parameters------------
    timm_jax_vit_base                           self

    pos_embed                       pos_embedding
    patch_embed.proj.weight         to_patch_embedding.0.weights
    patch_embed.proj.bias           to_patch_embedding.0.bias
    cls_token                       cls_token
    norm.weight                     model.transformer.norm.weight
    norm.bias                       model.transformer.norm.bias

                    -----------Attention Layer-------------
    blocks.0.norm1.weight       model.transformer.layers.0.0.norm.weight
    blocks.0.norm1.bias         model.transformer.layers.0.0.norm.bias
    blocks.0.attn.qkv.weight    model.transformer.layers.0.0.to_qkv.weight
    blocks.0.attn.qkv.bias      model.transformer.layers.0.0.to_qkv.bias
    blocks.0.attn.proj.weight   model.transformer.layers.0.0.to_out.0.weight
    blocks.0.attn.proj.bias     model.transformer.layers.0.0.to_out.0.bias

                    -----------MLP Layer-------------
    blocks.0.norm2.weight       model.transformer.layers.0.1.net.0.weight
    blocks.0.norm2.bias         model.transformer.layers.0.1.net.0.bias
    blocks.0.mlp.fc1.weight     model.transformer.layers.0.1.net.1.weight
    blocks.0.mlp.fc1.bias       model.transformer.layers.0.1.net.1.bias
    blocks.0.mlp.fc2.weight     model.transformer.layers.0.1.net.4.weight
    blocks.0.mlp.fc2.bias       model.transformer.layers.0.1.net.4.bias
            .                                          .
            .                                          .
            .                                          .
        """
```

