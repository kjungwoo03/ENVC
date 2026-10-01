"""Event-conditioned video compression with transmitted residual motion."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .envc_base import ENVCBase, g_ch_1x, g_ch_2x, g_ch_4x
from .entropy_models import BitEstimator
from .layers import (
    DepthConvBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    subpel_conv1x1,
)
from .video_net import (
    ME_Spynet,
    bilineardownsacling,
    flow_warp,
    get_hyper_enc_dec_models,
)


MV_CHANNELS = 64


class ResidualMotionEncoder(nn.Module):
    """DCMVC MvEnc, applied to event-conditioned residual motion."""

    def __init__(self, input_channel=2, channel=MV_CHANNELS, inplace=False):
        super().__init__()
        self.enc_1 = nn.Sequential(
            ResidualBlockWithStride(input_channel, channel, stride=2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
        )
        self.enc_2 = ResidualBlockWithStride(channel, channel, stride=2, inplace=inplace)
        self.adaptor_0 = DepthConvBlock(channel, channel, inplace=inplace)
        self.adaptor_1 = DepthConvBlock(channel * 2, channel, inplace=inplace)
        self.enc_3 = nn.Sequential(
            ResidualBlockWithStride(channel, channel, stride=2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
            nn.Conv2d(channel, channel, 3, stride=2, padding=1),
        )

    def forward(self, delta_motion, ref_delta_feature, quant_step):
        out = self.enc_1(delta_motion)
        out = out * quant_step
        out = self.enc_2(out)
        if ref_delta_feature is None:
            out = self.adaptor_0(out)
        else:
            out = self.adaptor_1(torch.cat((out, ref_delta_feature), dim=1))
        return self.enc_3(out)


class ResidualMotionDecoder(nn.Module):
    """DCMVC MvDec; its image-space output is ``delta_m_hat``."""

    def __init__(self, output_channel=2, channel=MV_CHANNELS, inplace=False):
        super().__init__()
        self.dec_1 = nn.Sequential(
            DepthConvBlock(channel, channel, inplace=inplace),
            ResidualBlockUpsample(channel, channel, 2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
            ResidualBlockUpsample(channel, channel, 2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
        )
        self.dec_2 = ResidualBlockUpsample(channel, channel, 2, inplace=inplace)
        self.dec_3 = nn.Sequential(
            DepthConvBlock(channel, channel, inplace=inplace),
            subpel_conv1x1(channel, output_channel, 2),
        )

    def forward(self, latent, quant_step):
        feature = self.dec_1(latent)
        out = self.dec_2(feature)
        out = out * quant_step
        return self.dec_3(out), feature


class EventConfidenceGate(nn.Module):
    """Predict one spatial confidence map per context scale from events and the reference RGB frame."""

    def __init__(self, event_channels=10, img_channels=3, mid_ch=16, init_bias=4.0):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(event_channels + img_channels, mid_ch, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head1 = nn.Conv2d(mid_ch, 1, 3, padding=1)
        self.head2 = nn.Conv2d(mid_ch, 1, 3, padding=1)
        self.head3 = nn.Conv2d(mid_ch, 1, 3, padding=1)
        for head in (self.head1, self.head2, self.head3):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, init_bias)

    def forward(self, event_voxel, ref_frame, size1, size2, size3):
        feat = self.trunk(torch.cat([event_voxel, ref_frame], dim=1))
        gate3 = torch.sigmoid(self.head3(feat))
        gate2 = torch.sigmoid(
            F.interpolate(self.head2(feat), size=size2, mode="bilinear", align_corners=False)
        )
        gate1 = torch.sigmoid(
            F.interpolate(self.head1(feat), size=size1, mode="bilinear", align_corners=False)
        )
        if gate3.shape[-2:] != size3:
            gate3 = F.interpolate(gate3, size=size3, mode="bilinear", align_corners=False)
        return gate1, gate2, gate3


class ENVC(ENVCBase):
    """Inter-frame codec with an event motion prior and a residual-motion bit estimate."""

    def __init__(self, anchor_num=4, ec_thread=False, stream_part=1, inplace=False):
        super().__init__(
            anchor_num=anchor_num,
            ec_thread=ec_thread,
            stream_part=stream_part,
            inplace=inplace,
        )

        self.mv_z_channel = MV_CHANNELS
        self.bit_estimator_z_mv = BitEstimator(MV_CHANNELS)
        self.optic_flow = ME_Spynet()
        self.mv_encoder = ResidualMotionEncoder(inplace=inplace)
        self.mv_hyper_prior_encoder, self.mv_hyper_prior_decoder = \
            get_hyper_enc_dec_models(MV_CHANNELS, MV_CHANNELS, inplace=inplace)

        self.mv_y_prior_fusion_adaptor_0 = DepthConvBlock(
            MV_CHANNELS, MV_CHANNELS * 2, inplace=inplace
        )
        self.mv_y_prior_fusion_adaptor_1 = DepthConvBlock(
            MV_CHANNELS * 2, MV_CHANNELS * 2, inplace=inplace
        )
        self.mv_y_prior_fusion = nn.Sequential(
            DepthConvBlock(MV_CHANNELS * 2, MV_CHANNELS * 3, inplace=inplace),
            DepthConvBlock(MV_CHANNELS * 3, MV_CHANNELS * 3, inplace=inplace),
        )
        self.mv_y_spatial_prior_adaptor_1 = nn.Conv2d(MV_CHANNELS * 4, MV_CHANNELS * 3, 1)
        self.mv_y_spatial_prior_adaptor_2 = nn.Conv2d(MV_CHANNELS * 4, MV_CHANNELS * 3, 1)
        self.mv_y_spatial_prior_adaptor_3 = nn.Conv2d(MV_CHANNELS * 4, MV_CHANNELS * 3, 1)
        self.mv_y_spatial_prior = nn.Sequential(
            DepthConvBlock(MV_CHANNELS * 3, MV_CHANNELS * 3, inplace=inplace),
            DepthConvBlock(MV_CHANNELS * 3, MV_CHANNELS * 3, inplace=inplace),
            DepthConvBlock(MV_CHANNELS * 3, MV_CHANNELS * 2, inplace=inplace),
        )
        self.mv_decoder = ResidualMotionDecoder(inplace=inplace)

        self.mv_y_q_basic_enc = nn.Parameter(torch.ones((1, MV_CHANNELS, 1, 1)))
        self.mv_y_q_scale_enc = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.mv_y_q_scale_enc_fine = None
        self.mv_y_q_basic_dec = nn.Parameter(torch.ones((1, MV_CHANNELS, 1, 1)))
        self.mv_y_q_scale_dec = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.mv_y_q_scale_dec_fine = None

        self.event_confidence_gate = EventConfidenceGate()
        self.event_gate_enabled = True


    @staticmethod
    def _fine_scale(q_scale):
        anchors = q_scale[:, 0, 0, 0].detach().cpu().numpy()
        return np.exp(np.linspace(np.log(anchors[0]), np.log(anchors[-1]), 64))

    def refresh_q_scales(self):
        self.y_q_scale_enc_fine = self._fine_scale(self.y_q_scale_enc)
        self.y_q_scale_dec_fine = self._fine_scale(self.y_q_scale_dec)
        self.mv_y_q_scale_enc_fine = self._fine_scale(self.mv_y_q_scale_enc)
        self.mv_y_q_scale_dec_fine = self._fine_scale(self.mv_y_q_scale_dec)

    def load_state_dict(self, state_dict, strict=True):
        result = nn.Module.load_state_dict(self, state_dict, strict=strict)
        self.refresh_q_scales()
        return result


    @staticmethod
    def normalize_dpb(dpb):
        return {
            "ref_frame": dpb["ref_frame"],
            "ref_feature": dpb.get("ref_feature"),
            "ref_mv_feature": dpb.get("ref_mv_feature"),
            "ref_y": dpb.get("ref_y"),
            "ref_mv_y": dpb.get("ref_mv_y"),
        }

    def get_q_for_inference(self, q_in_ckpt, q_index):
        mv_enc_scale = self.mv_y_q_scale_enc if q_in_ckpt else self.mv_y_q_scale_enc_fine
        mv_dec_scale = self.mv_y_q_scale_dec if q_in_ckpt else self.mv_y_q_scale_dec_fine
        y_enc_scale = self.y_q_scale_enc if q_in_ckpt else self.y_q_scale_enc_fine
        y_dec_scale = self.y_q_scale_dec if q_in_ckpt else self.y_q_scale_dec_fine
        return (
            self.get_curr_q(mv_enc_scale, self.mv_y_q_basic_enc, q_index=q_index),
            self.get_curr_q(mv_dec_scale, self.mv_y_q_basic_dec, q_index=q_index),
            self.get_curr_q(y_enc_scale, self.y_q_basic_enc, q_index=q_index),
            self.get_curr_q(y_dec_scale, self.y_q_basic_dec, q_index=q_index),
        )

    def mv_prior_param_decoder(self, mv_z_hat, dpb, slice_shape=None):
        params = self.mv_hyper_prior_decoder(mv_z_hat)
        params = self.slice_to_y(params, slice_shape)
        ref_mv_y = dpb["ref_mv_y"]
        if ref_mv_y is None:
            params = self.mv_y_prior_fusion_adaptor_0(params)
        else:
            params = self.mv_y_prior_fusion_adaptor_1(torch.cat((params, ref_mv_y), dim=1))
        return self.mv_y_prior_fusion(params)

    def event_motion_prior(self, dpb, p_pred):
        """Backward motion reproducible from decoded ref + shared events."""
        return self.dec_me(p_pred, dpb["ref_frame"])

    def hybrid_motion_compensation(self, dpb, motion, frame_idx, p_pred, e1, e2, e3, event_voxel):
        warp_frame = flow_warp(dpb["ref_frame"], motion)
        motion2 = bilineardownsacling(motion) / 2
        motion3 = bilineardownsacling(motion2) / 2
        ref1, ref2, ref3 = self.multi_scale_feature_extractor(dpb, frame_idx)
        context1_init = flow_warp(ref1, motion)
        # Rebuild reference features if recurrent feature magnitudes exceed the bound.
        if context1_init.abs().max() > 5.0:
            clean_feature = self.feature_adaptor_I(dpb["ref_frame"])
            ref1, ref2, ref3 = self.feature_extractor(clean_feature)
            context1_init = flow_warp(ref1, motion)
        context1 = self.align(
            ref1, torch.cat((context1_init, warp_frame, motion), dim=1), motion
        )
        context2 = flow_warp(ref2, motion2)
        context3 = flow_warp(ref3, motion3)

        if self.event_gate_enabled:
            gate1, gate2, gate3 = self.event_confidence_gate(
                event_voxel, dpb["ref_frame"],
                size1=context1.shape[-2:], size2=context2.shape[-2:], size3=context3.shape[-2:],
            )
            context1 = context1 + gate1 * self.event_ctx_adaptor1(e1)
            context2 = context2 + gate2 * self.event_ctx_adaptor2(e2)
            context3 = context3 + gate3 * self.event_ctx_adaptor3(e3)
        else:
            context1 = context1 + self.event_ctx_adaptor1(e1)
            context2 = context2 + self.event_ctx_adaptor2(e2)
            context3 = context3 + self.event_ctx_adaptor3(e3)
        context1, context2, context3 = self.context_fusion_net(
            context1, context2, context3
        )
        event_context = self.decoder_side_feature_adaptor(p_pred)
        context1 = self.decoder_context_refine(
            context1, event_context
        )
        # Bound the fused context passed to the residual codec.
        context1 = context1.clamp(-10.0, 10.0)
        context2 = context2.clamp(-10.0, 10.0)
        context3 = context3.clamp(-10.0, 10.0)
        return (
            context1,
            context2,
            context3,
            warp_frame,
        )

    def _motion_forward(self, x, dpb, event_motion, mv_y_q_enc, mv_y_q_dec):
        target_motion = self.optic_flow(x, dpb["ref_frame"])
        delta_motion = target_motion - event_motion
        mv_y = self.mv_encoder(delta_motion, dpb["ref_mv_feature"], mv_y_q_enc)
        mv_y_pad, slice_shape = self.pad_for_y(mv_y)
        mv_z = self.mv_hyper_prior_encoder(mv_y_pad)
        mv_z_for_bit, mv_z_hat = self.quantize_hyperprior(mv_z)
        mv_params = self.mv_prior_param_decoder(mv_z_hat, dpb, slice_shape)
        _, mv_y_q, mv_y_hat, mv_scales_hat = self.forward_four_part_prior(
            mv_y,
            mv_params,
            self.mv_y_spatial_prior_adaptor_1,
            self.mv_y_spatial_prior_adaptor_2,
            self.mv_y_spatial_prior_adaptor_3,
            self.mv_y_spatial_prior,
        )
        delta_motion_hat, mv_feature = self.mv_decoder(mv_y_hat, mv_y_q_dec)
        final_motion = event_motion + delta_motion_hat
        return (
            target_motion,
            delta_motion,
            delta_motion_hat,
            final_motion,
            mv_feature,
            mv_y_hat,
            mv_y_q,
            mv_scales_hat,
            mv_z_for_bit,
        )

    def _estimate_bpp_fp32(self, y_q, scales, z_for_bit, bit_estimator, pixel_num):
        """Estimate latent rates in FP32 to avoid unstable half-precision CDF calculations."""
        with torch.autocast(device_type=y_q.device.type, enabled=False):
            bpp_y = self.get_y_laplace_bits(
                y_q.float(), scales.float()
            ).sum((1, 2, 3)) / pixel_num
            bpp_z = self.get_z_bits(
                z_for_bit.float(), bit_estimator
            ).sum((1, 2, 3)) / pixel_num
        return bpp_y, bpp_z

    def _full_forward(
        self,
        x,
        ref_frame,
        ref_feature,
        ref_mv_feature,
        ref_y,
        ref_mv_y,
        p_pred,
        e1,
        e2,
        e3,
        event_motion,
        event_voxel,
        mv_y_q_enc,
        mv_y_q_dec,
        y_q_enc,
        y_q_dec,
        frame_idx,
    ):
        dpb = {
            "ref_frame": ref_frame,
            "ref_feature": ref_feature,
            "ref_mv_feature": ref_mv_feature,
            "ref_y": ref_y,
            "ref_mv_y": ref_mv_y,
        }
        motion_out = self._motion_forward(x, dpb, event_motion, mv_y_q_enc, mv_y_q_dec)
        final_motion = motion_out[3]
        contexts = self.hybrid_motion_compensation(
            dpb, final_motion, frame_idx, p_pred, e1, e2, e3, event_voxel
        )
        context1, context2, context3 = contexts[:3]
        y = self.contextual_encoder(x, context1, context2, context3, y_q_enc)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.contextual_hyper_prior_encoder(y_pad)
        z_for_bit, z_hat = self.quantize_hyperprior(z)
        params = self.res_prior_param_decoder(z_hat, dpb, context3, slice_shape)
        _, y_q, y_hat, scales_hat = self.forward_four_part_prior(
            y,
            params,
            self.y_spatial_prior_adaptor_1,
            self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3,
            self.y_spatial_prior,
        )
        x_hat, feature = self.get_recon_and_feature(
            y_hat, context1, context2, context3, y_q_dec
        )
        return (*motion_out, *contexts[3:], x_hat, feature, y_hat, y_q, scales_hat, z_for_bit)


    @torch.no_grad()
    def forward_one_frame(
        self,
        x,
        dpb,
        event_voxel,
        q_in_ckpt=False,
        q_index=None,
        frame_idx=0,
    ):
        dpb = self.normalize_dpb(dpb)
        p_pred, e1, e2, e3 = self.run_front_end(dpb, event_voxel)
        event_motion = self.event_motion_prior(dpb, p_pred)
        mv_q_enc, mv_q_dec, y_q_enc, y_q_dec = self.get_q_for_inference(
            q_in_ckpt, q_index
        )

        args = (
            x,
            dpb["ref_frame"],
            dpb["ref_feature"],
            dpb["ref_mv_feature"],
            dpb["ref_y"],
            dpb["ref_mv_y"],
            p_pred,
            e1,
            e2,
            e3,
            event_motion,
            event_voxel,
            mv_q_enc,
            mv_q_dec,
            y_q_enc,
            y_q_dec,
            frame_idx,
        )
        outs = self._full_forward(*args)

        (
            target_motion,
            delta_motion,
            delta_hat,
            final_motion,
            mv_feature,
            mv_y_hat,
            mv_y_q,
            mv_scales,
            mv_z_for_bit,
            final_warp,
            x_hat,
            feature,
            y_hat,
            y_q,
            scales,
            z_for_bit,
        ) = outs

        pixel_num = x.shape[-2] * x.shape[-1]
        bpp_mv_y, bpp_mv_z = self._estimate_bpp_fp32(
            mv_y_q, mv_scales, mv_z_for_bit, self.bit_estimator_z_mv, pixel_num
        )
        bpp_y, bpp_z = self._estimate_bpp_fp32(
            y_q, scales, z_for_bit, self.bit_estimator_z, pixel_num
        )
        bpp = bpp_mv_y + bpp_mv_z + bpp_y + bpp_z

        return {
            "x_hat": x_hat,
            "p_pred": p_pred,
            "event_motion": event_motion,
            "target_motion": target_motion,
            "delta_motion": delta_motion,
            "delta_motion_hat": delta_hat,
            "final_motion": final_motion,
            "warp_frame": final_warp,
            "bpp_mv_y": bpp_mv_y,
            "bpp_mv_z": bpp_mv_z,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
            "bpp": bpp,
            "dpb": {
                "ref_frame": x_hat,
                "ref_feature": feature,
                "ref_mv_feature": mv_feature,
                "ref_y": y_hat,
                "ref_mv_y": mv_y_hat,
            },
            "bit": bpp.sum() * pixel_num,
            "bit_y": bpp_y.sum() * pixel_num,
            "bit_z": bpp_z.sum() * pixel_num,
            "bit_mv_y": bpp_mv_y.sum() * pixel_num,
            "bit_mv_z": bpp_mv_z.sum() * pixel_num,
        }

    def encode_decode(
        self,
        x,
        dpb,
        event_voxel,
        q_in_ckpt,
        q_index,
        output_path=None,
        pic_width=None,
        pic_height=None,
        frame_idx=0,
    ):
        del pic_width, pic_height
        if output_path is not None:
            raise NotImplementedError(
                "ENVC supports entropy-estimated rates only; output_path must be None."
            )
        out = self.forward_one_frame(
            x,
            dpb,
            event_voxel,
            q_in_ckpt=q_in_ckpt,
            q_index=q_index,
            frame_idx=frame_idx,
        )
        return {
            "dpb": out["dpb"],
            "bit": out["bit"].item(),
            "bit_y": out["bit_y"].item(),
            "bit_z": out["bit_z"].item(),
            "bit_mv_y": out["bit_mv_y"].item(),
            "bit_mv_z": out["bit_mv_z"].item(),
            "encoding_time": 0,
            "decoding_time": 0,
        }
