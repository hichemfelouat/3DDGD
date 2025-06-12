import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        self.eps    = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[None, :, None] * x + self.bias[None, :, None]
            return x

class Stem(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            LayerNorm(out_dim, eps=1e-6, data_format="channels_last"),
        )

    def forward(self, x):
        x = self.nn(x)
        return x

class Tail(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.nn = nn.Sequential(
            LayerNorm(in_dim, eps=1e-6, data_format="channels_last"),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        x = self.nn(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv     = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads=8, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            LayerNorm(dim // 4, eps=1e-6, data_format="channels_last"),
            nn.ReLU(),
            nn.Linear(dim // 4, dim // 4),
            LayerNorm(dim // 4, eps=1e-6, data_format="channels_last"),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
        )

        self.attention = MultiHeadAttention(dim, num_heads)
        self.norm1     = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.norm2     = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.gamma1    = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2    = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x     = self.norm1(x)
        x     = self.attention(x)
        if self.gamma1 is not None:
            x = self.gamma1 * x
        x = input + self.drop_path(x)

        input = x
        x = self.norm2(x)
        x = self.nn(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = input + self.drop_path(x)

        return x

class Net(nn.Module):
    def __init__(self, C_in, C_out, depths=[3, 4, 6, 3], stem_dims=[256, 256, 512], dims=[256, 256, 256, 512],
                 decode_out_dims=[256, 256, 256, 128], drop_path_rate=0., k_eig_list=[485, 64, 16], outputs_at='faces',
                 layer_scale_init_value=1e-6, num_heads=8):
        super().__init__()
        self.C_in       = C_in
        self.k_eig_list = k_eig_list
        self.outputs_at = outputs_at
        self.dims       = dims

        # Stems 
        self.stems = nn.ModuleList()
        for i in range(len(depths)):
            if i != (len(depths) - 1):
                stem = Stem(C_in, stem_dims[i])
            else:
                stem = Stem(dims[-2], dims[-1])
            self.stems.append(stem)

        # Encoders 
        self.encoders = nn.ModuleList()
        dp_rates      = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur           = 0
        for i in range(len(depths)):
            encode_layer = nn.Sequential(
                *[Block(dim=dims[i], num_heads=num_heads, drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])],
            )
            self.encoders.append(encode_layer)
            cur += depths[i]

        # Decoders 
        self.decoders = nn.ModuleList()
        dp_rates      = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur           = 0
        for i in range(len(depths) + 1):
            if i != 0 and i != len(depths):
                decode_layer = nn.Sequential(
                    *[Block(dim=dims[-1], num_heads=num_heads, drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[i-1])],
                    Tail(dims[-1], decode_out_dims[i]),
                )
                cur += depths[i-1]
            elif i == 0:
                decode_layer = nn.Sequential(Tail(dims[-1], decode_out_dims[i]))
            elif i == len(depths):
                decode_layer = nn.Sequential(
                    *[Block(dim=(decode_out_dims[-1] * 4), num_heads=num_heads, drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[-1])],
                    Tail((decode_out_dims[-1] * 4), decode_out_dims[-1]),
                )
                cur += depths[i - 1]
            self.decoders.append(decode_layer)

        # Decoders_cat 
        self.decoders_cat = nn.ModuleList()
        for i in range(len(depths) - 1):
            cat_layer = nn.Sequential(Tail(decode_out_dims[i], decode_out_dims[-1]))
            self.decoders_cat.append(cat_layer)

        # Last layer 
        self.last_layer = nn.Sequential(
            LayerNorm(128, eps=1e-6, data_format="channels_last"),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, C_out),
        )
        

    def forward(self, x_in):

        if x_in.shape[-1] != self.C_in:
            raise ValueError("Input channels must be {}".format(self.C_in))
        if len(x_in.shape) == 2:
            appended_batch_dim = True
            # add a batch dim to all inputs
            x_in = x_in.unsqueeze(0)
        elif len(x_in.shape) == 3:
            appended_batch_dim = False
        else:
            raise ValueError("x_in should be tensor with shape [N,C] or [B,N,C]")

        encoder_feat = list()
        stem_0       = self.stems[0](x_in)
        feat_0       = self.encoders[0](stem_0)
        encoder_feat.append(feat_0)
        
        for i in range(1, len(self.k_eig_list)):
            feat_i = self.encoders[i](encoder_feat[i - 1])
            encoder_feat.append(feat_i)
        stem_i = self.stems[-1](encoder_feat[-1])
        feat_i = self.encoders[-1](stem_i)

        decode_feat = list()
        feat_0      = self.decoders[0](feat_i)
        decode_feat.append(feat_0)
        for i in range(1, (len(self.k_eig_list) + 1)):
            decode_input = torch.cat((encoder_feat[-i], decode_feat[i - 1]), dim=-1)
            feat_i       = self.decoders[i](decode_input)
            decode_feat.append(feat_i)

        decode_cat_feat = list()
        for i in range(len(self.dims) - 1):
            feat_i = self.decoders_cat[i](decode_feat[i])
            decode_cat_feat.append(feat_i)

        x = torch.cat((decode_feat[-1], decode_cat_feat[2], decode_cat_feat[1], decode_cat_feat[0]), dim=-1)
        x = self.decoders[-1](x)

        x = self.last_layer(x)

        x = torch.sigmoid(x)

        return x
