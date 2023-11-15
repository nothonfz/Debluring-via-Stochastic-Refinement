import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# class Upsample(nn.Module):
#     def __init__(self, dim, p):
#         super().__init__()
#         self.swish = Swish()
#         self.up = nn.Upsample(scale_factor=2, mode="nearest")
#         self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
#         self.dropout = nn.Dropout(p)
#         self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
#
#     def forward(self, x):
#         x1 = self.conv1(self.up(self.swish(x)))
#         x3 = self.conv2(self.dropout(self.swish(x1)))
#         return x3
#
#
# class Downsample(nn.Module):
#     def __init__(self, dim, p):
#         super().__init__()
#         self.swish = Swish()
#         self.conv_down = nn.Conv2d(dim, dim, 3, 2, 1)
#         self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
#         self.dropout = nn.Dropout(p)
#         self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
#
#     def forward(self, x):
#         x1 = self.conv1(self.swish(self.conv_down(self.swish(x))))
#         x3 = self.conv2(self.dropout(self.swish(x1)))
#         return x3


class res_down(nn.Module):
    # 根据补充材料实现的下采样层
    def __init__(self, dim, out_dim, p, noise_level_emb_dim):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, out_dim)
        self.swish = Swish()
        self.conv_down = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.conv1 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.dropout = nn.Dropout(p)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.identity = nn.Conv2d(out_dim, out_dim, 1, 2)

    def forward(self, x, time):
        x1 = self.conv1(self.swish(self.conv_down(self.swish(x))))
        x2 = self.noise_func(x1, time)
        x3 = self.conv2(self.dropout(self.swish(x2)))
        return x3 + self.identity(x)


class res_downNoEmb(nn.Module):
    # 根据补充材料实现的下采样层，但没有噪声步作为输入，作为初始预测器的层
    def __init__(self, dim, out_dim, p):
        super().__init__()
        self.swish = Swish()
        self.conv_down = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.conv1 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.dropout = nn.Dropout(p)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.identity = nn.Conv2d(out_dim, out_dim, 1, 2)

    def forward(self, x,):
        x1 = self.conv1(self.swish(self.conv_down(self.swish(x))))
        x3 = self.conv2(self.dropout(self.swish(x1)))
        return x3 + self.identity(x)


class res_up(nn.Module):
    # 根据补充材料实现的上采样层
    def __init__(self, dim, out_dim, p, noise_level_emb_dim):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, out_dim)
        self.swish = Swish()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(dim, out_dim, 3, 1, 1)
        self.dropout = nn.Dropout(p)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.identity = nn.Conv2d(out_dim, out_dim, 1, 1)

    def forward(self, x, time):
        x1 = self.conv1(self.swish(self.up(self.swish(x))))
        x2 = self.noise_func(x1, time)
        x3 = self.conv2(self.dropout(self.swish(x2)))
        return x3 + self.identity(self.up(x))


class res_upNoEmb(nn.Module):
    # 根据补充材料实现的上采样层，但没有噪声步作为输入，作为初始预测器的层
    def __init__(self, dim, out_dim, p):
        super().__init__()
        self.swish = Swish()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(dim, out_dim, 3, 1, 1)
        self.dropout = nn.Dropout(p)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.identity = nn.Conv2d(out_dim, out_dim, 1, 1)

    def forward(self, x):
        x1 = self.conv1(self.swish(self.up(self.swish(x))))
        x3 = self.conv2(self.dropout(self.swish(x1)))
        return x3 + self.identity(self.up(x))
# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            # 去掉了GroupNorm
            # nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class ResnetBlockNoEmb(nn.Module):
    # 初始预测器的ResBlock，没有噪声步作为条件输入
    def __init__(self, dim, dim_out, dropout=0, norm_groups=32):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)

class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 3, 4),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                if with_noise_level_emb:
                    downs.append(ResnetBlock(
                        pre_channel, channel_mult, noise_level_channel, dropout))
                else:
                    downs.append(ResnetBlockNoEmb(
                        pre_channel, channel_mult, dropout))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                if with_noise_level_emb:
                    downs.append(res_down(pre_channel, pre_channel, dropout, noise_level_channel))
                else:
                    downs.append(res_downNoEmb(pre_channel, pre_channel, dropout))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        if with_noise_level_emb:
            self.mid = nn.ModuleList([
                ResnetBlock(pre_channel, pre_channel, noise_level_channel, dropout=dropout, norm_groups=norm_groups),
                ResnetBlock(pre_channel, pre_channel, noise_level_channel, norm_groups=norm_groups, dropout=dropout)
            ])
        else:
            self.mid = nn.ModuleList([
                ResnetBlockNoEmb(pre_channel, pre_channel, dropout=dropout, norm_groups=norm_groups),
                ResnetBlockNoEmb(pre_channel, pre_channel, norm_groups=norm_groups, dropout=dropout)
            ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                if with_noise_level_emb:
                    ups.append(ResnetBlock(
                        pre_channel + feat_channels.pop(), channel_mult, noise_level_channel, dropout))
                else:
                    ups.append(ResnetBlockNoEmb(
                        pre_channel + feat_channels.pop(), channel_mult, dropout))
                pre_channel = channel_mult
            if not is_last:
                if with_noise_level_emb:
                    ups.append(res_up(pre_channel, pre_channel, dropout, noise_level_channel))
                else:
                    ups.append(res_upNoEmb(pre_channel, pre_channel, dropout))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time=None):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlock):
                x = layer(x, t)
            elif isinstance(layer, res_down):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlock):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlock):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            elif isinstance(layer, ResnetBlockNoEmb):
                x = layer(torch.cat((x, feats.pop()), dim=1))
            elif isinstance(layer, res_up):
                x = layer(x, t)
            else:
                x = layer(x)

        return self.final_conv(x)
