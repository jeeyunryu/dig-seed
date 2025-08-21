# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

from models.transformer_layer import (
  get_pad_mask, get_subsequent_mask
)

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class PretrainVisionTransformerEncoderRatio(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0., pretrained_cfg=None,
                 use_learnable_pos_emb=False, use_mean_pooling=False, init_scale=0.001, return_feat_map=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    

    def forward_features(self, x, mask=None):
        H = x.size(2)
        W = x.size(3)

        # input preprocessing
        x = self.patch_embed(x)

        ## replace masked patches with mask_token
        B, N, C = x.shape
        if mask is not None:
            vis_mask = ~mask # 어떤 걸 보이게 할까 (마스킹 제외 패치)
            x = x * vis_mask.unsqueeze(-1) + self.mask_token.expand(B, N, -1) * mask.unsqueeze(-1) # x * vis_mask.unsqueeze(-1) -> 마스킹 된 것 값 제거
        ## add position embedding
            

        Hp = H // self.patch_size
        Wp = W // self.patch_size
            
        pos = _interpolate_pos_embed(self.pos_embed, Hp, Wp)
        pos = pos.type_as(x).to(x.device)
        x = x + pos
            
        # x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        # encoder
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)


        return x

    def forward(self, x, mask=None):

        x = self.forward_features(x, mask)
        x = self.head(x)
        return x
    
class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0., pretrained_cfg=None,
                 use_learnable_pos_emb=False, use_mean_pooling=False, init_scale=0.001, return_feat_map=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    

    def forward_features(self, x, mask=None):
        # H = x.size(2)
        # W = x.size(3)

        # input preprocessing
        x = self.patch_embed(x)

        ## replace masked patches with mask_token
        B, N, C = x.shape
        if mask is not None:
            vis_mask = ~mask # 어떤 걸 보이게 할까 (마스킹 제외 패치)
            x = x * vis_mask.unsqueeze(-1) + self.mask_token.expand(B, N, -1) * mask.unsqueeze(-1) # x * vis_mask.unsqueeze(-1) -> 마스킹 된 것 값 제거
        ## add position embedding
            

        # Hp = H // self.patch_size
        # Wp = W // self.patch_size
            
        # pos = _interpolate_pos_embed(self.pos_embed, Hp, Wp)
        # pos = pos.type_as(x).to(x.device)
        # x = x + pos
            
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        # encoder
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)


        return x

    def forward(self, x, mask=None):

        x = self.forward_features(x, mask)
        x = self.head(x)
        return x

def _interpolate_pos_embed(pos_embed: torch.Tensor, Hp: int, Wp: int):
        """
        pos_embed : (1, Npos, C), Npos = N_old 또는 1+N_old
        반환      : (1, Hp*Wp, C) 또는 (1, 1+Hp*Wp, C) (cls 토큰 있으면 유지)
        """
        
        Bpe, Npos, C = pos_embed.shape
        assert Bpe == 1, "pos_embed batch dim은 보통 1입니다."

        # cls 토큰 유무 자동 판별
        has_cls = False
        N_grid = Npos
        if Npos - 1 > 0:
            # (Npos-1)이 정사각 격자로 해석 가능하면 cls 토큰이 있다고 가정
            rt = int(round((Npos - 1) ** 0.5))
            if rt * rt == (Npos - 1):
                has_cls = True
                N_grid = Npos - 1

        if has_cls:
            cls_pos = pos_embed[:, :1, :]           # (1,1,C)
            pos = pos_embed[:, 1:, :]               # (1,N_grid,C)
        else:
            cls_pos = None
            pos = pos_embed                         # (1,N_grid,C)

        # 원래 그리드 크기 추정
        Gh_old = int(round(N_grid ** 0.5))
        Gw_old = N_grid // Gh_old
        assert Gh_old * Gw_old == N_grid, "pos_embed의 토큰 수가 사각 격자로 안 나눠집니다."
        # (1, Gh_old, Gw_old, C) -> (1, C, Gh_old, Gw_old)
        pos = pos.reshape(1, Gh_old, Gw_old, C).permute(0, 3, 1, 2)
        # 2D 보간
        pos = F.interpolate(pos, size=(Hp, Wp), mode='bicubic', align_corners=False)
        # (1, C, Hp, Wp) -> (1, Hp*Wp, C)
        pos = pos.permute(0, 2, 3, 1).reshape(1, Hp * Wp, C)

        if has_cls:
            pos = torch.cat([cls_pos, pos], dim=1)  # (1, 1+Hp*Wp, C)
        return pos

@register_model
def simmim_vit_tiny_patch4_32x128(pretrained=False, **kwargs):
  model = PretrainVisionTransformerEncoder(
      img_size=(32, 128), patch_size=4, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
      norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
  model.default_cfg = _cfg()
  return model

@register_model
def simmim_vit_small_patch4_32x128(pretrained=False, **kwargs): #* 여기다!! finetuning spot
    # model = PretrainVisionTransformerEncoderRatio(
    # img_size=(32, 128), patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, use_learnable_pos_emb = True,
    # norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = PretrainVisionTransformerEncoder(
        img_size=(32, 128), patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def simmim_vit_base_patch4_32x128(pretrained=False, **kwargs):
  model = PretrainVisionTransformerEncoder(
      img_size=(32, 128), patch_size=4, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
      norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
  model.default_cfg = _cfg()
  return model