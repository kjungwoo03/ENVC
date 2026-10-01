"""Event and image feature pyramids with 48/64/96-channel codec contexts."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfpsie.submodel import conv, ResBlock
from .vfpsie.sca import Symmetrical_Cross_Modal_Attention
from .vfpsie.utils import warp, resize

CTX_CHANNELS = 48
E2_CHANNELS = 64
E3_CHANNELS = 96

_ENC_CHS = (64, 96, 128, 160)


class ImageEncoder(nn.Module):
    """Extract image features at full, half, quarter, eighth, and sixteenth resolution."""

    def __init__(self, in_channel, init_chs=_ENC_CHS):
        super().__init__()
        self.pyramid0 = nn.Sequential(
            conv(in_channel, CTX_CHANNELS, 3, 1, 1),
            conv(CTX_CHANNELS, CTX_CHANNELS, 3, 1, 1),
        )
        self.pyramid1 = nn.Sequential(
            conv(in_channel, init_chs[0], 3, 2, 1),
            conv(init_chs[0], init_chs[0], 3, 1, 1),
        )
        self.pyramid2 = nn.Sequential(
            conv(init_chs[0], init_chs[1], 3, 2, 1),
            conv(init_chs[1], init_chs[1], 3, 1, 1),
        )
        self.pyramid3 = nn.Sequential(
            conv(init_chs[1], init_chs[2], 3, 2, 1),
            conv(init_chs[2], init_chs[2], 3, 1, 1),
        )
        self.pyramid4 = nn.Sequential(
            conv(init_chs[2], init_chs[3], 3, 2, 1),
            conv(init_chs[3], init_chs[3], 3, 1, 1),
        )

    def forward(self, img):
        f0 = self.pyramid0(img)
        f1 = self.pyramid1(img)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f0, f1, f2, f3, f4


class EventEncoder(nn.Module):
    def __init__(self, in_channel, init_chs=_ENC_CHS):
        super().__init__()
        self.pyramid1 = nn.Sequential(
            conv(in_channel, init_chs[0], 3, 2, 1),
            conv(init_chs[0], init_chs[0], 3, 1, 1),
        )
        self.pyramid2 = nn.Sequential(
            conv(init_chs[0], init_chs[1], 3, 2, 1),
            conv(init_chs[1], init_chs[1], 3, 1, 1),
        )
        self.pyramid3 = nn.Sequential(
            conv(init_chs[1], init_chs[2], 3, 2, 1),
            conv(init_chs[2], init_chs[2], 3, 1, 1),
        )
        self.pyramid4 = nn.Sequential(
            conv(init_chs[2], init_chs[3], 3, 2, 1),
            conv(init_chs[3], init_chs[3], 3, 1, 1),
        )

    def forward(self, event):
        f1 = self.pyramid1(event)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4


class Decoder4(nn.Module):
    """Coarsest stage (1/16 -> 1/8), internal only -- its output feeds
    Decoder3 and is never exposed as an e-value (there is no "e4")."""

    def __init__(self, ch=_ENC_CHS[3], internal_ch=_ENC_CHS[2]):
        super().__init__()
        self.attention = Symmetrical_Cross_Modal_Attention(dim=ch, num_heads=4)
        self.convblock = nn.Sequential(
            conv(ch * 2, ch * 2),
            ResBlock(ch * 2, 32),
            nn.ConvTranspose2d(ch * 2, 2 + internal_ch, 4, 2, 1, bias=True),
        )

    def forward(self, I, E):
        AI, AE = self.attention(I, E)
        f_in = torch.cat([AI, AE], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder3(nn.Module):
    """1/8 -> 1/4, produces e3 (E3_CHANNELS=96, matches DCMVC's g_ch_4x)."""

    def __init__(self, in_ch=_ENC_CHS[2], mid_ch=256, out_ch=E3_CHANNELS):
        super().__init__()
        concat_ch = in_ch * 3 + 2
        self.convblock = nn.Sequential(
            conv(concat_ch, mid_ch),
            ResBlock(mid_ch, 32),
            nn.ConvTranspose2d(mid_ch, 2 + out_ch, 4, 2, 1, bias=True),
        )

    def forward(self, It_, I0, E, up_flow0):
        B, C, H, W = It_.shape
        I0_warp, mask = warp(I0, up_flow0)
        mask = mask.unsqueeze(1).repeat(1, C, 1, 1)
        I0_warp[mask] = It_[mask]
        f_in = torch.cat([It_, I0_warp, E, up_flow0], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder2(nn.Module):
    """1/4 -> 1/2, produces e2 (E2_CHANNELS=64, matches DCMVC's g_ch_2x).
    `in_ch=E3_CHANNELS` (its own It_ argument is e3) is deliberately equal to
    `_ENC_CHS[1]` (this level's own I0/E width) so the masked replace in
    forward() is shape-valid -- see module docstring."""

    def __init__(self, in_ch=E3_CHANNELS, mid_ch=192, out_ch=E2_CHANNELS):
        super().__init__()
        concat_ch = in_ch + _ENC_CHS[1] * 2 + 2
        self.convblock = nn.Sequential(
            conv(concat_ch, mid_ch),
            ResBlock(mid_ch, 32),
            nn.ConvTranspose2d(mid_ch, 2 + out_ch, 4, 2, 1, bias=True),
        )

    def forward(self, It_, I0, E, up_flow0):
        B, C, H, W = It_.shape
        I0_warp, mask = warp(I0, up_flow0)
        mask = mask.unsqueeze(1).repeat(1, C, 1, 1)
        I0_warp[mask] = It_[mask]
        f_in = torch.cat([It_, I0_warp, E, up_flow0], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder1(nn.Module):
    """Decode full-resolution flow, image fusion, blending weights, and event context."""

    def __init__(self, in_ch=E2_CHANNELS, enc_ch=_ENC_CHS[0], mid_ch=128,
                ctx_channels=CTX_CHANNELS):
        super().__init__()
        self.attention = Symmetrical_Cross_Modal_Attention(dim=enc_ch, num_heads=1)
        concat_ch = in_ch + enc_ch * 2 + 2
        self.convblock = nn.Sequential(
            conv(concat_ch, mid_ch),
            ResBlock(mid_ch, 32),
            nn.ConvTranspose2d(mid_ch, 7, 4, 2, 1, bias=True),
        )
        self.ctx_head = nn.Sequential(
            ResBlock(mid_ch, 32),
            nn.ConvTranspose2d(mid_ch, ctx_channels, 4, 2, 1, bias=True),
        )

    def forward(self, It_, I0, E, up_flow0):
        B, C, H, W = It_.shape
        I0_warp, mask = warp(I0, up_flow0)
        mask = mask.unsqueeze(1).repeat(1, C, 1, 1)
        I0_warp[mask] = It_[mask]
        AI, AE = self.attention(I0_warp, E)
        f_in = torch.cat([It_, AE + I0_warp, AI + E, up_flow0], 1)
        feat = self.convblock[1](self.convblock[0](f_in))
        out = self.convblock[2](feat)
        ctx = self.ctx_head(feat)
        return out, ctx


class EventMotionContextGenerator(nn.Module):
    """Predict an RGB frame and contexts at full, half, and quarter resolution."""

    def __init__(self, ctx_channels=CTX_CHANNELS):
        super().__init__()
        self.image_encoder = ImageEncoder(in_channel=3)
        self.event_encoder = EventEncoder(in_channel=10)
        self.decoder4 = Decoder4()
        self.decoder3 = Decoder3()
        self.decoder2 = Decoder2()
        self.decoder1 = Decoder1(ctx_channels=ctx_channels)


    def forward(self, img0, evt0):
        img0_ = img0
        evt0_ = evt0

        I0_0, I0_1, I0_2, I0_3, I0_4 = self.image_encoder(img0_)
        f0_1, f0_2, f0_3, f0_4 = self.event_encoder(evt0_)

        out4 = self.decoder4(I0_4, f0_4)
        up_flow0_4 = out4[:, 0:2]
        synth3 = out4[:, 2:]

        out3 = self.decoder3(synth3, I0_3, f0_3, up_flow0_4)
        up_flow0_3 = out3[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        e3_wide = out3[:, 2:]

        out2 = self.decoder2(e3_wide, I0_2, f0_2, up_flow0_3)
        up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_3, scale_factor=2.0)
        e2_wide = out2[:, 2:]

        out1, ctx = self.decoder1(
            e2_wide, I0_1, f0_1, up_flow0_2
        )
        up_flow0_1 = out1[:, 0:2] + 2.0 * resize(up_flow0_2, scale_factor=2.0)

        img0_fusion = torch.clamp(out1[:, 2:5], 0, 1)
        weight = F.softmax(out1[:, 5:], dim=1)

        img0_warp, mask = warp(img0_, up_flow0_1)
        mask = mask.unsqueeze(1).repeat(1, 3, 1, 1)
        img0_warp[mask] = img0_fusion[mask]

        p_pred = weight[:, 0:1, ...] * img0_fusion + weight[:, 1:2, ...] * img0_warp
        p_pred = torch.clamp(p_pred, 0, 1)

        e2 = e2_wide
        e3 = e3_wide
        return {
            "e1": ctx, "e2": e2, "e3": e3, "p_pred": p_pred,
            "img0_fusion": img0_fusion,
            "up_flow0_1": up_flow0_1,
        }
