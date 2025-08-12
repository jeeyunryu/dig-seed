import torch.nn as nn

from .encoder import *
from .decoder import *
from .attn_decoder import AttentionRecognitionHead
from .embedding_head import Embedding
from models import encoder
from sklearn.decomposition import PCA



class CTCRecModel(nn.Module):
  def __init__(self, args):
    super(CTCRecModel, self).__init__()

    self.encoder = create_encoder(args)
    d_embedding = 512
    self.ctc_classifier = nn.Sequential(nn.Linear(self.encoder.num_features, d_embedding),
                                        nn.LayerNorm(d_embedding, eps=1e-6),
                                        nn.GELU(),
                                        nn.Linear(d_embedding, args.nb_classes + 1))

    # some function and variable should be inherited.
    self.patch_embed = self.encoder.patch_embed
    self.pos_embed = self.encoder.pos_embed

  def no_weight_decay(self):
    skip_weight_decay_list = self.encoder.no_weight_decay()
    return {'encoder.' + item for item in skip_weight_decay_list}

  def get_num_layers(self):
    return self.encoder.get_num_layers()

  def forward(self, x):
    x, tgt, tgt_lens = x
    enc_x = self.encoder(x)

    B, N, C = enc_x.shape
    reshaped_enc_x = enc_x.view(B, *self.encoder.patch_embed.patch_shape, C).mean(1)
    ctc_logit = self.ctc_classifier(reshaped_enc_x)

    return ctc_logit
  
def get_divisible_width(orig_w, target_chunks):
    # target_chunks로 나눠지는 가장 가까운 너비 찾기 (오차 최소화)
    base = orig_w // target_chunks
    rem = orig_w % target_chunks
    if rem == 0:
        return orig_w  # 이미 나눠짐
    # 둘 중 가까운 값을 선택 (내림 or 올림)
    lower = base * target_chunks
    upper = (base + 1) * target_chunks
    if abs(orig_w - lower) <= abs(upper - orig_w):
        return lower
    else:
        return upper
    
def resize_width_to_divisible_len(enc_feat, target_len):
    D, H, W = enc_feat.shape
    W_new = get_divisible_width(W, target_len)
    
    if W == W_new:
      return enc_feat, W
    enc_feat = enc_feat.unsqueeze(0)
    
    resized = F.interpolate(enc_feat, size=(H, W_new), mode='bilinear', align_corners=False)
    resized = resized.squeeze(0)
    
    return resized, W_new

def split_enc_x(enc_x, text_lens):
    B, C, H, W = enc_x.shape # D: dimension!!
    output = []
    for b in range(B):
        enc_feat_b = enc_x[b]
        target_len = int(text_lens[b])

        resized_feat, W_new = resize_width_to_divisible_len(enc_feat_b, target_len)
        chunk_size = W_new // target_len
        split_chars = torch.split(resized_feat, chunk_size, dim=2)  # 균등 분할
        padded_chars = []
        for char_feat in split_chars:
          x_padded = F.pad(char_feat, (0, 32-chunk_size))
          x_padded = x_padded.permute(1, 2, 0)
          x_padded = x_padded.reshape(-1, C)
          padded_chars.append(x_padded)

        split = torch.stack(padded_chars)
        split = split.contiguous()

        output.append(split)
    return output

class AttnRecModel(nn.Module):
  def __init__(self, args):
    super(AttnRecModel, self).__init__()

    self.dig_mode = args.dig_mode

    self.pca = PCA(n_components=300)

    self.all_embed = []
    self.all_label = []
    self.all_imgkeys = []
    self.all_idx = []


    self.encoder = create_encoder(args)
    self.decoder = AttentionRecognitionHead(
                      num_classes=args.nb_classes,
                      in_planes=self.encoder.num_features,
                      sDim=512,
                      attDim=512,
                      max_len_labels=args.max_len, args=args) 

    # some function and variable should be inherited.
    self.patch_embed = self.encoder.patch_embed
    self.pos_embed = self.encoder.pos_embed

    # 1d or 2d features
    self.use_1d_attdec = args.use_1d_attdec
    self.beam_width = getattr(args, 'beam_width', 0)

    if self.dig_mode == 'dig-seed':
      self.embeder = Embedding(256, 384)

  def no_weight_decay(self):
    skip_weight_decay_list = self.encoder.no_weight_decay()
    return {'encoder.' + item for item in skip_weight_decay_list}

  def get_num_layers(self):
    return self.encoder.get_num_layers()
  
  def get_all_embed(self):
    return self.all_embed, self.all_label, self.all_imgkeys, self.all_idx


  def forward(self, x):
    # 원본
    # x, tgt, tgt_lens = x
    # enc_x = self.encoder(x)

    # dec_output, _ = self.decoder((enc_x, tgt, tgt_lens))
    # return dec_output, None, None, None
    x, tgt, tgt_lens, img_key = x
    
    enc_x = self.encoder(x)
    B, N, C = enc_x.shape
    enc_x_2d = enc_x.view(B, *self.encoder.patch_embed.patch_shape, C).permute(0, 3, 1, 2)
    spl_enc_x = split_enc_x(enc_x_2d, tgt_lens-1)

    for idx, enc_x_sample in enumerate(spl_enc_x): 
      N, C, D = enc_x_sample.shape
      for i in range(26):
        if i < N: 
          x = enc_x_sample[i] # one character feature [256, 384] => [1, 256*384] => [1, 300]
          x_avg = x.mean(dim=0, keepdim=True)
          x_avg = x_avg[..., :300]
          self.all_embed.append(x_avg.cpu().numpy())
          labels = tgt[idx][i]
          self.all_label.append(labels.cpu().numpy())
          self.all_imgkeys.append(img_key[idx])
          self.all_idx.append(i)

    if (self.dig_mode == 'dig'):
      dec_output, _ = self.decoder((enc_x, tgt, tgt_lens))
      return dec_output, None, None, None
    else:
      enc_x = enc_x.contiguous()
      embedding_vectors = self.embeder(enc_x)
      dec_output, _ = self.decoder((enc_x, tgt, tgt_lens), embedding_vectors)
      return dec_output, None, None, None, embedding_vectors

    

class RecModel(nn.Module):
  def __init__(self, args):
    super(RecModel, self).__init__()

    self.encoder = create_encoder(args)
    self.decoder = create_decoder(args)
    # if args.decoder_name == 'small_tf_decoder':
    #   d_embedding = 384
    # else:  
    #   d_embedding = 512
    d_embedding = self.decoder.d_embedding
    self.linear_norm = nn.Sequential(
      nn.Linear(self.encoder.num_features, d_embedding),
      nn.LayerNorm(d_embedding),
    )

    # some function and variable should be inherited.
    self.patch_embed = self.encoder.patch_embed
    self.pos_embed = self.encoder.pos_embed

    # target embedding is used in both encoder and decoder
    if hasattr(self.encoder, 'insert_sem'):
      if self.encoder.insert_sem:
        self.trg_word_emb = nn.Embedding(
          args.nb_classes + 1, d_embedding
        )
        self.insert_sem = True
      else:
        self.trg_word_emb = None
        self.insert_sem = False
    else:
      self.trg_word_emb = None
      self.insert_sem = False

    # 1d or 2d features
    self.use_1d_attdec = args.use_1d_attdec
    self.beam_width = getattr(args, 'beam_width', 0)
    
    # add feat projector
    self.use_feat_distill = getattr(args, 'use_feat_distill', False)
    if self.use_feat_distill:
      self.feat_proj = self._build_mlp(3, self.encoder.num_features, 4096, self.encoder.num_features)

  def no_weight_decay(self):
    skip_weight_decay_list = self.encoder.no_weight_decay()
    return {'encoder.' + item for item in skip_weight_decay_list}

  def get_num_layers(self):
    return self.encoder.get_num_layers()

  def forward(self, x):
    
    x, tgt, tgt_lens = x
    if self.insert_sem: #*
      enc_x, rec_score = self.encoder((x, self.trg_word_emb))
    else:
      enc_x = self.encoder(x)
    # maybe a multi-label branch is added.
    if isinstance(enc_x, tuple):
      cls_logit, enc_x, cls_logit_attn_maps = enc_x
    else:
      cls_logit = None
      cls_logit_attn_maps = None
      
    if not self.training:
      tgt = None
      tgt_lens = None
    # only use multi-label loss
    if enc_x is None and cls_logit is not None:
      # no decoder
      return None, cls_logit, None, None
    
    # 1d or 2d features for decoder
    if self.use_1d_attdec:
      B, N, C = enc_x.shape
      enc_x = enc_x.view(B, *self.encoder.patch_embed.patch_shape, C).mean(1)

    dec_in = self.linear_norm(enc_x)

   
    dec_output, dec_attn_maps = self.decoder(dec_in,
                                             dec_in,
                                             targets=tgt,
                                             tgt_lens=tgt_lens,
                                             train_mode=self.training,
                                             cls_query_attn_maps=cls_logit_attn_maps,
                                             trg_word_emb=self.trg_word_emb,
                                             beam_width=self.beam_width,)
    
    # feat distillation
    if self.use_feat_distill and self.training:
      b, l, c = enc_x.shape
      s_feat = self.feat_proj(enc_x.reshape(b*l, c))
      s_feat = s_feat.reshape(b, l, c)
      # s_feat = self.feat_proj(enc_x)
      return dec_output, s_feat
    
    # return dec_output, None, None, None
    return dec_output, None, None, dec_attn_maps
    
    B, len_q, len_k = dec_attn_maps.shape
    # dec_attn_maps = dec_attn_maps.view(B, len_q, *self.patch_embed.patch_shape)
    # dec_attn_maps = dec_attn_maps[:, :, :self.patch_embed.num_patches].view(B, len_q, *self.patch_embed.patch_shape)
    dec_attn_maps = None
    if self.insert_sem:
      return dec_output, rec_score, dec_attn_maps
    if cls_logit is not None:
      # reshpe attn_maps to spatial version
      return dec_output, cls_logit, cls_logit_attn_maps, dec_attn_maps
    else:
      return dec_output, None, None, dec_attn_maps

  def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True, use_conv=False):
      mlp = []
      for l in range(num_layers):
          dim1 = input_dim if l == 0 else mlp_dim
          dim2 = output_dim if l == num_layers - 1 else mlp_dim

          if use_conv:
            mlp.append(nn.Conv1d(dim1, dim2, 1, bias=False))
          else:
            mlp.append(nn.Linear(dim1, dim2, bias=False))

          if l < num_layers - 1:
              mlp.append(nn.BatchNorm1d(dim2))
              mlp.append(nn.ReLU(inplace=True))
          elif last_bn:
              # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
              # for simplicity, we further removed gamma in BN
              mlp.append(nn.BatchNorm1d(dim2, affine=False))

      return nn.Sequential(*mlp)

class MimRecModel(nn.Module):
  def __init__(self, args):
    super(MimRecModel, self).__init__()

    self.encoder = create_encoder(args)
    self.rec_decoder = create_decoder(args)
    self.pix_encoder_to_decoder = nn.Linear(self.encoder.num_features, 192, bias=False)
    self.decoder = nn.Sequential(nn.Linear(192, 192, bias=False),
                                     nn.LayerNorm(192, eps=1e-6),
                                     nn.GELU(),
                                     nn.Linear(192, 48))
    self.use_mim_loss = args.mim_sample_ratio > 0.
    self.use_mim_proj = args.use_mim_proj
    if self.use_mim_proj:
      self.mim_proj = nn.Sequential(nn.Linear(self.encoder.num_features, self.encoder.num_features * 2),
                                nn.LayerNorm(self.encoder.num_features * 2, eps=1e-6),
                                nn.GELU(),
                                nn.Linear(self.encoder.num_features * 2, self.encoder.num_features),
                                nn.LayerNorm(self.encoder.num_features, eps=1e-6),)
      # self.mim_proj = nn.Sequential(nn.Linear(self.encoder.num_features, self.encoder.num_features),
      #                           nn.LayerNorm(self.encoder.num_features, eps=1e-6),
      #                           nn.GELU(),
      #                           nn.Linear(self.encoder.num_features, self.encoder.num_features),
      #                           nn.LayerNorm(self.encoder.num_features, eps=1e-6),)
    # if args.decoder_name == 'small_tf_decoder':
    #   d_embedding = 384
    # else:  
    #   d_embedding = 512
    d_embedding = self.rec_decoder.d_embedding
    self.linear_norm = nn.Sequential(
      nn.Linear(self.encoder.num_features, d_embedding),
      nn.LayerNorm(d_embedding),
    )

    # some function and variable should be inherited.
    self.patch_embed = self.encoder.patch_embed
    self.pos_embed = self.encoder.pos_embed

  def no_weight_decay(self):
    skip_weight_decay_list = self.encoder.no_weight_decay()
    return {'encoder.' + item for item in skip_weight_decay_list}

  def get_num_layers(self):
    return self.encoder.get_num_layers()

  def forward(self, x):
    out_dict = {}

    if len(x) == 5:
      x, mask, num_mim_samples, tgt, tgt_lens = x
    else:
      x, tgt, tgt_lens = x
      mask = None
      num_mim_samples = 0
    # enc_x = self.encoder(x, mask)

    # mim loss
    if self.use_mim_loss:
      if self.use_mim_proj:
        temp_enc_x = self.encoder(x, mask)
        mim_enc_x = temp_enc_x[:num_mim_samples]
        mim_enc_x = self.mim_proj(mim_enc_x)
        enc_x = torch.cat([mim_enc_x, temp_enc_x[num_mim_samples:]], dim=0)
      else:
        temp_enc_x = self.encoder(x, mask)
        enc_x = temp_enc_x.clone()
      pix_dec_input = self.pix_encoder_to_decoder(temp_enc_x)
      pix_dec_output = self.decoder(pix_dec_input)
      out_dict['pix_pred'] = pix_dec_output
    else:
      enc_x = self.encoder(x, mask)

    # recognition      
    if not self.training:
      tgt = None
      tgt_lens = None

    enc_x = self.linear_norm(enc_x)
    dec_output, dec_attn_maps = self.rec_decoder(enc_x,
                                             enc_x,
                                             targets=tgt,
                                             tgt_lens=tgt_lens,
                                             train_mode=self.training,
                                             cls_query_attn_maps=None,
                                             trg_word_emb=None,)
    out_dict['rec_pred'] = dec_output
    return out_dict
