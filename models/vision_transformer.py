# import torch
# from torch import nn, einsum
# from torch.nn import functional as F
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
# import math
#
#
# class PatchEmbedding(nn.Module):
#     def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
#         super().__init__()
#
#         def _make_tuple(x):
#             if not isinstance(x, (list, tuple)):
#                 return (x, x)
#             return x
#
#         img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
#         self.num_patches = (img_size[0] // patch_size[0]) * (
#             img_size[1] // patch_size[1]
#         )
#         self.conv = nn.LazyConv2d(
#             num_hiddens, kernel_size=patch_size, stride=patch_size
#         )
#
#     def forward(self, X):
#         # Output shape: (batch size, no. of patches, no. of channels)
#         return self.conv(X).flatten(2).transpose(1, 2)
#
#
# def masked_softmax(X, valid_lens):
#     """Perform softmax operation by masking elements on the last axis.
#
#     Defined in :numref:`sec_attention-scoring-functions`"""
#     # X: 3D tensor, valid_lens: 1D or 2D tensor
#     def _sequence_mask(X, valid_len, value=0):
#         maxlen = X.size(1)
#         mask = (
#             torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :]
#             < valid_len[:, None]
#         )
#         X[~mask] = value
#         return X
#
#     if valid_lens is None:
#         return nn.functional.softmax(X, dim=-1)
#     else:
#         shape = X.shape
#         if valid_lens.dim() == 1:
#             valid_lens = torch.repeat_interleave(valid_lens, shape[1])
#         else:
#             valid_lens = valid_lens.reshape(-1)
#         # On the last axis, replace masked elements with a very large negative
#         # value, whose exponentiation outputs 0
#         X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
#         return nn.functional.softmax(X.reshape(shape), dim=-1)
#
#
# class DotProductAttention(nn.Module):
#     """Scaled dot product attention.
#
#     Defined in :numref:`subsec_additive-attention`"""
#
#     def __init__(self, dropout, num_heads=None):
#         super().__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.num_heads = num_heads  # To be covered later
#
#     # Shape of queries: (batch_size, no. of queries, d)
#     # Shape of keys: (batch_size, no. of key-value pairs, d)
#     # Shape of values: (batch_size, no. of key-value pairs, value dimension)
#     # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
#     def forward(self, queries, keys, values, valid_lens=None, window_mask=None):
#         d = queries.shape[-1]
#         # Swap the last two dimensions of keys with keys.transpose(1, 2)
#         scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
#         if window_mask is not None:  # To be covered later
#             num_windows = window_mask.shape[0]
#             n, num_queries, num_kv_pairs = scores.shape
#             # Shape of window_mask: (num_windows, no. of queries,
#             # no. of key-value pairs)
#             scores = torch.reshape(
#                 scores,
#                 (
#                     n // (num_windows * self.num_heads),
#                     num_windows,
#                     self.num_heads,
#                     num_queries,
#                     num_kv_pairs,
#                 ),
#             ) + torch.expand_dims(torch.expand_dims(window_mask, 1), 0)
#             scores = torch.reshape(scores, (n, num_queries, num_kv_pairs))
#         self.attention_weights = masked_softmax(scores, valid_lens)
#         return torch.bmm(self.dropout(self.attention_weights), values)
#
#
# class MultiHeadAttention(nn.Module):
#     """Multi-head attention.
#
#     Defined in :numref:`sec_multihead-attention`"""
#
#     def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
#         super().__init__()
#         self.num_heads = num_heads
#         self.attention = DotProductAttention(dropout, num_heads)
#         self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
#         self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
#         self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
#         self.W_o = nn.LazyLinear(num_hiddens, bias=bias)
#
#     def forward(self, queries, keys, values, valid_lens, window_mask=None):
#         # Shape of queries, keys, or values:
#         # (batch_size, no. of queries or key-value pairs, num_hiddens)
#         # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
#         # After transposing, shape of output queries, keys, or values:
#         # (batch_size * num_heads, no. of queries or key-value pairs,
#         # num_hiddens / num_heads)
#         queries = self.transpose_qkv(self.W_q(queries))
#         keys = self.transpose_qkv(self.W_k(keys))
#         values = self.transpose_qkv(self.W_v(values))
#
#         if valid_lens is not None:
#             # On axis 0, copy the first item (scalar or vector) for num_heads
#             # times, then copy the next item, and so on
#             valid_lens = torch.repeat_interleave(
#                 valid_lens, repeats=self.num_heads, dim=0
#             )
#
#         # Shape of output: (batch_size * num_heads, no. of queries,
#         # num_hiddens / num_heads)
#         output = self.attention(queries, keys, values, valid_lens, window_mask)
#         # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
#         output_concat = self.transpose_output(output)
#         return self.W_o(output_concat)
#
#     def transpose_qkv(self, X):
#         """Transposition for parallel computation of multiple attention heads.
#
#         Defined in :numref:`sec_multihead-attention`"""
#         # Shape of input X: (batch_size, no. of queries or key-value pairs,
#         # num_hiddens). Shape of output X: (batch_size, no. of queries or
#         # key-value pairs, num_heads, num_hiddens / num_heads)
#         X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
#         # Shape of output X: (batch_size, num_heads, no. of queries or key-value
#         # pairs, num_hiddens / num_heads)
#         X = X.permute(0, 2, 1, 3)
#         # Shape of output: (batch_size * num_heads, no. of queries or key-value
#         # pairs, num_hiddens / num_heads)
#         return X.reshape(-1, X.shape[2], X.shape[3])
#
#     def transpose_output(self, X):
#         """Reverse the operation of transpose_qkv.
#
#         Defined in :numref:`sec_multihead-attention`"""
#         X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
#         X = X.permute(0, 2, 1, 3)
#         return X.reshape(X.shape[0], X.shape[1], -1)
#
#
# class ViTMLP(nn.Module):
#     def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
#         super().__init__()
#         self.dense1 = nn.LazyLinear(mlp_num_hiddens)
#         self.gelu = nn.GELU()
#         self.dropout1 = nn.Dropout(dropout)
#         self.dense2 = nn.LazyLinear(mlp_num_outputs)
#         self.dropout2 = nn.Dropout(dropout)
#
#     def forward(self, x):
#         return self.dropout2(self.dense2(self.dropout1(self.gelu(self.dense1(x)))))
#
#
# class ViTBlock(nn.Module):
#     def __init__(
#         self,
#         num_hiddens,
#         norm_shape,
#         mlp_num_hiddens,
#         num_heads,
#         dropout,
#         use_bias=False,
#     ):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(norm_shape)
#         self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
#         self.ln2 = nn.LayerNorm(norm_shape)
#         self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)
#
#     def forward(self, X, valid_lens=None):
#         X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
#         return X + self.mlp(self.ln2(X))
#
#
# class ViT(nn.Module):
#     """Vision Transformer."""
#
#     def __init__(
#         self,
#         img_size,
#         patch_size,
#         num_hiddens,
#         mlp_num_hiddens,
#         num_heads,
#         num_blks,
#         emb_dropout,
#         blk_dropout,
#         lr=0.1,
#         use_bias=False,
#         num_classes=10,
#     ):
#         super().__init__()
#         self.patch_embedding = PatchEmbedding(img_size, patch_size, num_hiddens)
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
#         num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
#         # Positional embeddings are learnable
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_steps, num_hiddens))
#         self.dropout = nn.Dropout(emb_dropout)
#         self.blks = nn.Sequential()
#         for i in range(num_blks):
#             self.blks.add_module(
#                 f"{i}",
#                 ViTBlock(
#                     num_hiddens,
#                     num_hiddens,
#                     mlp_num_hiddens,
#                     num_heads,
#                     blk_dropout,
#                     use_bias,
#                 ),
#             )
#         self.head = nn.Sequential(
#             nn.LayerNorm(num_hiddens), nn.Linear(num_hiddens, num_classes)
#         )
#
#     def forward(self, X):
#         X = self.patch_embedding(X)
#         X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
#         X = self.dropout(X + self.pos_embedding)
#         for blk in self.blks:
#             X = blk(X)
#         return self.head(X[:, 0])
import torch
from tensorboard import summary
from torch import nn, einsum
from torch.nn import functional as F
from torch.utils import checkpoint
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
from models import soft_net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.CB = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=3, kernel_size=(3, 3), padding="same"
            ),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=3, out_channels=3, kernel_size=(3, 3), padding="same"
            ),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        self.CL = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=3, kernel_size=(3, 3), padding="same"
            ),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        self.Maxpooling = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        Fm1 = self.CB(x)
        # Fm2 = self.CB(self.Maxpooling(x))
        Fm2 = self.CB(x)
        Fm3 = self.CL(x)
        # print("Fm1,Fm2,Fm3",Fm1.shape, Fm2.shape, Fm3.shape)
        return Fm1, Fm2, Fm3

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        # patch_size = (patch_size, patch_size)
        # self.patch_size = patch_size
        self.patch_size = (patch_size, patch_size)
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        # padding
        # pad if the H and W of the image are not an integral multiple of patch_size
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(
                x,
                (
                    0,
                    self.patch_size[1] - W % self.patch_size[1],
                    0,
                    self.patch_size[0] - H % self.patch_size[0],
                    0,
                    0,
                ),
            )

        # dowmsample an integral multiple of patch_size
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        # pad if the H and W of the input feature map are not multiples of 2
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # Attention! The tensor shape is [B, H, W, C], which is different form the official document
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x


def window_partition(x, window_size: int):
    """
    partition the feature map to non overlapping window base on the window_size
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    return every window to feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
            self,
            dim,
            window_size,
            num_heads,
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = (
                coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2, x3, mask):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x1.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # print("***************************",x.shape) torch.Size([12, 49, 96]) ([12, 49, 192])([12, 49, 384]) ([12, 49, 768])

        # qkv = (
        #     self.qkv(x)
        #     .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        #     .permute(2, 0, 3, 1, 4)
        # )
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # torch.Size([12, 24, 49, 32]) torch.Size([12, 24, 49, 32]) torch.Size([12, 24, 49, 32])
        # q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = self.q_linear(x1).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_linear(x2).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_linear(x3).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # self.fc1 = nn.Linear(in_features, hidden_features)
        # self.act = act_layer()
        # self.drop1 = nn.Dropout(drop)
        # self.fc2 = nn.Linear(hidden_features, out_features)
        # self.drop2 = nn.Dropout(drop)

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop),
        )

    def forward(self, x):
        # x = self.fc1(x)
        # x = self.act(x)
        # x = self.drop1(x)
        # x = self.fc2(x)
        # x = self.drop2(x)
        x = self.mlp(x)
        return x


def drop_path_f(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
            self,
            dim,
            num_heads,
            window_size=7,
            shift_size=0,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert (
                -1 < self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x1, x2, x3, attn_mask):
        H, W = self.H, self.W
        B, L, C = x1.shape
        # print("x1.shape,x2.shape,x3.shape", x1.shape, x2.shape, x3.shape)
        assert L == H * W, "input feature has wrong size"

        shortcut = x1
        # print("x1.shape,x2.shape,x3.shape",x1.shape,x2.shape,x3.shape)
        x1 = self.norm1(x1)
        x1 = x1.view(B, H, W, C)

        x2 = self.norm1(x2)
        x2 = x2.view(B, H, W, C)

        x3 = self.norm1(x3)
        x3 = x3.view(B, H, W, C)

        # print("***************",x.shape) # torch.Size([12, 49, 96])
        # print("qqqqqqqqqqqqqqqqqqqq",x.shape) # torch.Size([12, 7, 7, 96])
        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x1 = F.pad(x1, (0, 0, pad_l, pad_r, pad_t, pad_b))
        x2 = F.pad(x2, (0, 0, pad_l, pad_r, pad_t, pad_b))
        x3 = F.pad(x3, (0, 0, pad_l, pad_r, pad_t, pad_b))

        _, Hp, Wp, _ = x1.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x1 = torch.roll(
                x1, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            shifted_x2 = torch.roll(
                x2, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            shifted_x3 = torch.roll(
                x3, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x1 = x1
            shifted_x2 = x2
            shifted_x3 = x3

            attn_mask = None

        # partition windows
        x_windows1 = window_partition(shifted_x1, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows1 = x_windows1.view(
            -1, self.window_size * self.window_size, C
        )  # [nW*B, Mh*Mw, C]

        x_windows2 = window_partition(shifted_x2, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows2 = x_windows2.view(
            -1, self.window_size * self.window_size, C
        )  # [nW*B, Mh*Mw, C]

        x_windows3 = window_partition(shifted_x3, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows3 = x_windows3.view(
            -1, self.window_size * self.window_size, C
        )  # [nW*B, Mh*Mw, C]
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows1, x_windows2, x_windows3, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, C
        )  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(
            attn_windows, self.window_size, Hp, Wp
        )  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # remove the padding data
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            num_heads,
            window_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # make sure the Hp and Wp are multiples of window_size
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # made the channels the same as feature map, for the convenience of window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(
            -1, self.window_size * self.window_size
        )  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(
            2
        )  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )
        return attn_mask

    def forward(self, x1, x2, x3, H1, W1, H2, W2, H3, W3):
        # print("-----------------------", x1.shape, x2.shape, x3.shape)
        attn_mask1 = self.create_mask(x1, H1, W1)  # [nW, Mh*Mw, Mh*Mw]
        attn_mask2 = self.create_mask(x2, H2, W2)  # [nW, Mh*Mw, Mh*Mw]
        attn_mask3 = self.create_mask(x3, H3, W3)  # [nW, Mh*Mw, Mh*Mw]

        # print("-----------------------", x1.shape, x2.shape, x3.shape)
        for blk in self.blocks:
            blk.H, blk.W = H1, W1
            # print("****blk.h,blk.w****", blk.H, blk.W)
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x1 = checkpoint.checkpoint(blk, x1, attn_mask1)
                x2 = checkpoint.checkpoint(blk, x2, attn_mask2)
                x3 = checkpoint.checkpoint(blk, x3, attn_mask3)

            else:
                # print("***x1.shape****,x2.shape,x3.shape", x1.shape, x2.shape, x3.shape)
                x1 = blk(x1, x2, x3, attn_mask1)
                # x2 = blk(x2, attn_mask2)
                # x3 = blk(x3, attn_mask3)

        if self.downsample is not None:
            x1 = self.downsample(x1, H1, W1)
            x2 = self.downsample(x2, H2, W2)
            x3 = self.downsample(x3, H3, W3)

            H1, W1 = (H1 + 1) // 2, (W1 + 1) // 2
            H2, W2 = (H2 + 1) // 2, (W3 + 1) // 2
            H3, W3 = (H3 + 1) // 2, (W3 + 1) // 2

        return x1, x2, x3, H1, W1, H2, W2, H3, W3


class FMSwinTransformer(nn.Module):
    r"""Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
            https://arxiv.org/pdf/2103.14030
    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
            self,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4 输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_c=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # Attention! The stage here is different form the content in the image form the original document
            # the stage doesn't contain the patch_merging of this stage, but the next stage
            # print("embed_dim, i_layer",embed_dim__96, i_layer__0,1,2,3)
            layers = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]): sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layers)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.apply(self._init_weights)
        self.softnet = soft_net.SOFTNet().to(device)
        self.MFE = soft_net.MFE().to(device)
        # self.MFE = MFE().to(device)
        self.linear = nn.Linear(in_features=3, out_features=1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, L, C]
        Fm1, Fm2, Fm3 = self.MFE(x)
        x1, H1, W1 = self.patch_embed(Fm1)  # torch.Size([12, 49, 96])
        x1 = self.pos_drop(x1)
        # print("x1.shape", x1.shape)
        # for layer in self.layers:
        #     x1, H1, W1 = layer(x1, H1, W1)
        # x1 = self.norm(x1)
        # x1 = self.avgpool(x1.transpose(1, 2))
        # x1 = torch.flatten(x1, 1)
        # x1 = self.head(x1)
        # # ****************************************************
        x2, H2, W2 = self.patch_embed(Fm2)  # torch.Size([12, 49, 96])
        x2 = self.pos_drop(x2)
        # for layer in self.layers:
        #     x2, H2, W2 = layer(x2, H2, W2)
        # x2 = self.norm(x2)
        # x2 = self.avgpool(x2.transpose(1, 2))
        # x2 = torch.flatten(x2, 1)
        # x2 = self.head(x2)
        # # ******************************************************
        x3, H3, W3 = self.patch_embed(Fm3)  # torch.Size([12, 49, 96])
        x3 = self.pos_drop(x3)

        # for layer in self.layers:
        #     x3, H3, W3 = layer(x3, H3, W3)
        # x3 = self.norm(x3)
        # x3 = self.avgpool(x3.transpose(1, 2))
        # x3 = torch.flatten(x3, 1)
        # x3 = self.head(x3)
        # x_123 = torch.cat((x1, x2, x3), 1)
        # *******************************************************************
        # print("11111111111", x.shape)  torch.Size([12, 3, 42, 42])
        # x, H, W = self.patch_embed(x)  # torch.Size([12, 49, 96])
        # x = self.pos_drop(x)  # torch.Size([12, 49, 96])
        # print("x.shape",x.shape)
        # # print("3333333333333333", x.shape)

        # x1, H1, W1 = self.layers[0](x, H, W)
        # x2, H2, W2 = self.layers[1](x1, H1, W1)
        # x3, H3, W3 = self.layers[2](x2, H2, W2)
        # x4, H4, W4 = self.layers[3](x3, H3, W3)
        # print("x1.shape,x2.shape,x3.shape,x4.shape",x1.shape,x2.shape,x3.shape,x4.shape)
        # x1.shape,x2.shape,x3.shape,x4.shape torch.Size([12, 16, 192]) torch.Size([12, 4, 384]) torch.Size([12, 1,
        # 768]) torch.Size([12, 1, 768])
        for layer in self.layers:
            x1, x2, x3, H1, W1, H2, W2, H3, W3 = layer(x1, x2, x3, H1, W1, H2, W2, H3, W3)
        x = self.norm(x1)  # [B, L, C]
        # # print("55555555555555", x.shape) torch.Size([12, 1, 768])
        x = self.avgpool(x.transpose(1, 2))  # [B, C, 1] torch.Size([12, 768, 1])
        x = torch.flatten(x, 1)  # torch.Size([12, 768])
        x = self.head(x)

        # x = torch.cat((x, x_123), 1)
        # x = self.linear(x_123)
        return x

