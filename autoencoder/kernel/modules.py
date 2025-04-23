# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Optional


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class DownBlock(nn.Module):
    """
    A single resolution downsampling block consisting of multiple ResNet blocks,
    optional attention, and an optional downsample layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attn_resolutions: List[int],
        curr_res: int,
        dropout: float = 0.0,
        resamp_with_conv: bool = True
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        self.attn = nn.ModuleList()
        channels = in_channels
        for _ in range(num_res_blocks):
            self.blocks.append(
                ResnetBlock(
                    in_channels=channels,
                    out_channels=out_channels,
                    temb_channels=0,
                    dropout=dropout
                )
            )
            channels = out_channels
            if curr_res in attn_resolutions:
                self.attn.append(AttnBlock(channels))

        self.downsample: Optional[nn.Module] = None
        if resamp_with_conv:
            self.downsample = Downsample(channels, True)
        elif curr_res > 1:
            self.downsample = Downsample(channels, False)

    def forward(self, x: Tensor) -> Tensor:
        for i, block in enumerate(self.blocks):
            x = block(x, None)
            if i < len(self.attn):
                x = self.attn[i](x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class UpBlock(nn.Module):
    """
    A single resolution upsampling block consisting of multiple ResNet blocks,
    optional attention, and an optional upsample layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attn_resolutions: List[int],
        curr_res: int,
        dropout: float = 0.0,
        resamp_with_conv: bool = True
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        self.attn = nn.ModuleList()
        channels = in_channels
        for _ in range(num_res_blocks + 1):
            self.blocks.append(
                ResnetBlock(
                    in_channels=channels,
                    out_channels=out_channels,
                    temb_channels=0,
                    dropout=dropout
                )
            )
            channels = out_channels
            if curr_res in attn_resolutions:
                self.attn.append(AttnBlock(channels))

        self.upsample: Optional[nn.Module] = None
        if resamp_with_conv:
            self.upsample = Upsample(channels, True)

    def forward(self, x: Tensor) -> Tensor:
        for i, block in enumerate(self.blocks):
            x = block(x, None)
            if i < len(self.attn):
                x = self.attn[i](x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        ch: int,
        out_ch: int,
        resolution: int,
        z_channels: int,
        num_res_blocks: int,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        attn_resolutions: List[int] = [],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        double_z: bool = True,
        **ignorekwargs
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # initial conv
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, padding=1)

        # build downsampling stack
        self.down_blocks: nn.ModuleList = nn.ModuleList()
        curr_res = resolution
        in_mult = (1,) + tuple(ch_mult)
        for i, mult in enumerate(ch_mult):
            down = DownBlock(
                in_channels=ch * in_mult[i],
                out_channels=ch * mult,
                num_res_blocks=num_res_blocks,
                attn_resolutions=attn_resolutions,
                curr_res=curr_res,
                dropout=dropout,
                resamp_with_conv=resamp_with_conv
            )
            self.down_blocks.append(down)
            if i < self.num_resolutions - 1:
                curr_res //= 2

        # middle
        mid_ch = ch * ch_mult[-1]
        self.mid_block1 = ResnetBlock(
            in_channels=mid_ch, out_channels=mid_ch, temb_channels=0, dropout=dropout)
        self.mid_attn = AttnBlock(mid_ch)
        self.mid_block2 = ResnetBlock(
            in_channels=mid_ch, out_channels=mid_ch, temb_channels=0, dropout=dropout)

        # end
        self.norm_out = Normalize(mid_ch)
        out_channels_final = 2 * z_channels if double_z else z_channels
        self.conv_out = nn.Conv2d(
            mid_ch, out_channels_final, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv_in(x)
        for block in self.down_blocks:
            h = block(h)
        h = self.mid_block1(h, None)
        h = self.mid_attn(h)
        h = self.mid_block2(h, None)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        out_ch: int,
        ch: int,
        z_channels: int,
        resolution: int,
        num_res_blocks: int,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        attn_resolutions: List[int] = [],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        give_pre_end: bool = False,
        **ignorekwargs
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.give_pre_end = give_pre_end

        # compute lowest resolution
        curr_res = resolution // (2 ** (self.num_resolutions - 1))
        mid_ch = ch * ch_mult[-1]
        self.z_shape: Tuple[int, int, int, int] = (
            1, z_channels, curr_res, curr_res)

        # z to feature map
        self.conv_in = nn.Conv2d(z_channels, mid_ch, kernel_size=3, padding=1)

        # middle
        self.mid_block1 = ResnetBlock(
            in_channels=mid_ch, out_channels=mid_ch, temb_channels=0, dropout=dropout)
        self.mid_attn = AttnBlock(mid_ch)
        self.mid_block2 = ResnetBlock(
            in_channels=mid_ch, out_channels=mid_ch, temb_channels=0, dropout=dropout)

        # build upsampling stack
        self.up_blocks: nn.ModuleList = nn.ModuleList()
        for i in reversed(range(self.num_resolutions)):
            up = UpBlock(
                in_channels=mid_ch,
                out_channels=ch * ch_mult[i],
                num_res_blocks=num_res_blocks,
                attn_resolutions=attn_resolutions,
                curr_res=curr_res,
                dropout=dropout,
                resamp_with_conv=resamp_with_conv
            )
            self.up_blocks.append(up)
            mid_ch = ch * ch_mult[i]
            if i > 0:
                curr_res *= 2

        # end
        self.norm_out = Normalize(mid_ch)
        self.conv_out = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)
        h = self.mid_block1(h, None)
        h = self.mid_attn(h)
        h = self.mid_block2(h, None)
        for block in self.up_blocks:
            h = block(h)
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices
