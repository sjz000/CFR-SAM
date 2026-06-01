# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .common import LayerNorm2d

from utils import point_nms
from mmcv.ops import ModulatedDeformConv2d
from mmcv.cnn import build_norm_layer
import math

class MLFusion(nn.Module):
    def __init__(self, norm, act):
        super().__init__()

        self.attn_conv = nn.ModuleList()
        for i in range(4):
            self.attn_conv.append(nn.Sequential(
                nn.Conv2d(256, 256, 1, bias=False),
                norm(256),
                act(),
            ))

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_list):

        for i in range(4):
            x = feature_list[i]
            attn = self.attn_conv[i](x)
            attn = self.pool(attn)
            attn = self.sigmoid(attn)

            x = attn * x + x
            feature_list[i] = x

        return feature_list[0] + feature_list[1] + feature_list[2] + feature_list[3]


class MEEM(nn.Module):
    def __init__(self, in_dim, hidden_dim, width, norm, act):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            norm(hidden_dim),
            nn.Sigmoid()
        )

        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

        self.mid_conv = nn.ModuleList()
        self.edge_enhance = nn.ModuleList()
        for i in range(width - 1):
            self.mid_conv.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                norm(hidden_dim),
                nn.Sigmoid()
            ))
            self.edge_enhance.append(EdgeEnhancer(hidden_dim, norm, act))

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * width, in_dim, 1, bias=False),
            norm(in_dim),
            act()
        )

    def forward(self, x):
        mid = self.in_conv(x)

        out = mid
        # print(out.shape)

        for i in range(self.width - 1):
            mid = self.pool(mid)
            mid = self.mid_conv[i](mid)

            out = torch.cat([out, self.edge_enhance[i](mid)], dim=1)

        out = self.out_conv(out)

        return out


class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm, act):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            norm(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, norm='BN', groups=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False,
                              groups=groups)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class DeformLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 deconv_kernel=4,
                 deconv_stride=2,
                 deconv_padding=1,
                 deconv_out_padding=0,
                 num_groups=1,
                 deform_groups=1,
                 dilation=1,
                 norm_cfg=dict(type='BN'),
                 with_upsample=True):
        super(DeformLayer, self).__init__()
        self.with_upsample = with_upsample
        self.deform_groups = deform_groups
        self.kernel_size = kernel_size

        # Offset and mask generator
        offset_channels = 3 * kernel_size * kernel_size  # offset_x/y + mask
        self.dcn_offset = nn.Conv2d(
            in_channels,
            offset_channels * deform_groups,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            dilation=dilation)

        # Deformable Convolution
        self.dcn = ModulatedDeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            deform_groups=deform_groups)

        # Normalization after DCN
        self.dcn_bn = build_norm_layer(norm_cfg, out_channels)[1]

        # Optional upsampling
        if self.with_upsample:
            self.up_sample = nn.ConvTranspose2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=deconv_kernel,
                stride=deconv_stride,
                padding=deconv_padding,
                output_padding=deconv_out_padding,
                bias=False)
            self._deconv_init()
            self.up_bn = build_norm_layer(norm_cfg, out_channels)[1]

        self.relu = nn.ReLU(inplace=True)

        self.init_weights()

    def init_weights(self):
        for m in [self.dcn]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.dcn_offset.weight, 0)
        nn.init.constant_(self.dcn_offset.bias, 0)

    def _deconv_init(self):
        """Initialize the up-sample conv transpose weights like bilinear."""
        w = self.up_sample.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]

    def forward(self, x):
        offset_mask = self.dcn_offset(x)
        offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((offset_x, offset_y), dim=1)
        mask = mask.sigmoid()
        x = self.dcn(x, offset, mask)
        x = self.dcn_bn(x)
        x = self.relu(x)

        if self.with_upsample:
            x_up = self.up_sample(x)
            x_up = self.up_bn(x_up)
            x_up = self.relu(x_up)
            return x, x_up
        else:
            return x

class LSE(nn.Module):
    def __init__(self,in_dim,inter_channels):
        super().__init__()
        self.cbn1=ConvBNReLU(in_dim//2+in_dim,inter_channels//2,1,padding=0,norm=nn.BatchNorm2d)
        self.dcn3 = DeformLayer(in_dim//2+in_dim, inter_channels//2, 3,with_upsample=False)
        self.conv1 = nn.Conv2d(2,1,kernel_size=7,padding=3)
        self.conv2 = nn.Conv2d(2,1,kernel_size=7,padding=3)
        self.cbn2=ConvBNReLU(inter_channels,inter_channels,1,padding=0,norm=nn.BatchNorm2d)
    def forward(self,img,feat4):
        # feat4 = F.interpolate(feat4,size=(img.shape[2]//2,img.shape[3]//2),mode='bilinear',align_corners=False)
        # img = F.interpolate(img,size=(img.shape[2]//2,img.shape[3]//2),mode='bilinear',align_corners=False)
        x1,x2 = torch.chunk(feat4,2,dim=1)
        x1 = self.cbn1(torch.cat([img,x1],dim=1))
        x2 = self.dcn3(torch.cat([img,x2],dim=1))

        x1_mean = torch.mean(x1,dim=1,keepdim=True)
        x1_max,_  = torch.max(x1,dim=1,keepdim=True)
        x1_attn = self.conv1(torch.sigmoid(torch.cat([x1_mean,x1_max],dim=1)))

        x2_mean = torch.mean(x2,dim=1,keepdim=True)
        x2_max,_  = torch.max(x2,dim=1,keepdim=True)
        x2_attn = self.conv2(torch.sigmoid(torch.cat([x2_mean,x2_max],dim=1)))
        x1 = x1*x1_attn+x1
        x2 = x2*x2_attn+x2

        y  = torch.cat([x1,x2],dim=1)
        z = self.cbn2(y)
        return z

class ERM(nn.Module):
    def __init__(self,in_channels,inter_channels):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,inter_channels,1,1,0)
        self.conv2 = nn.Conv2d(inter_channels,in_channels,1,1,0)
        self.dcn = DeformLayer(inter_channels, inter_channels, 3,with_upsample=False)
        self.norm = nn.BatchNorm2d(in_channels)
    def forward(self,edge,body):
        edge = torch.sigmoid(edge)
        edge = edge*body
        edge = self.norm(self.conv2(self.dcn(self.conv1(edge))))
        # edge = self.norm(self.act(edge))
        return edge+body

class DetailEnhancement(nn.Module):
    def __init__(self, img_dim, feature_dim, norm, act):
        super().__init__()
        self.img_in_conv = nn.Sequential(
            nn.Conv2d(3, img_dim, 3, padding=1, bias=False),
            norm(img_dim),
            act()
        )
        self.img_er = MEEM(img_dim, img_dim // 2, 4, norm, act)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim , 32, 3, padding=1, bias=False),
            norm(32),
            act(),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            norm(16),
            act(),
        )

        self.out_conv = nn.Conv2d(16, 1, 1)
        self.edge_conv = nn.Sequential(
            ConvBNReLU(feature_dim, 16, 1, padding=0, norm=norm),
            norm(16),
            act(),
            nn.Conv2d(16, 1, 1, 1, padding=0))
        self.LSE = LSE(feature_dim, feature_dim)
        self.edge_rec = ERM(feature_dim, feature_dim // 2)
        self.feature_upsample = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
        )

    def forward(self, img, feature, b_feature, cell_nums):
        feature = torch.cat([feature, b_feature], dim=1)
        feature = self.feature_upsample(feature)

        img_feature = self.img_in_conv(img)
        img_feature = self.img_er(img_feature) + img_feature
        img_feature = torch.repeat_interleave(img_feature, cell_nums, dim=0)

        edge_feature = self.LSE(img_feature, feature)
        x_edge = self.edge_conv(edge_feature)
        x_out = self.edge_rec(edge_feature, feature)

        #out_feature = torch.cat([feature, img_feature], dim=1)
        out_feature = self.fusion_conv(x_out)
        out = self.out_conv(out_feature)

        return out, x_edge

class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
            self,
            image_encoder: ImageEncoderViT,
            prompt_encoder: PromptEncoder,
            mask_decoder: MaskDecoder,
            num_classes: int,
            multimask: bool,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.multimask = multimask
        self.deep_feautre_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.deep_out_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1)
        )

        self.fusion_block = MLFusion(norm=nn.BatchNorm2d, act=nn.ReLU)

        self.detail_enhance = DetailEnhancement(img_dim=32, feature_dim=32, norm=nn.BatchNorm2d, act=nn.ReLU)

    @staticmethod
    def get_anchor_points(images, space):
        bs, _, h, w = images.shape

        anchors = np.stack(
            np.meshgrid(
                np.arange(np.ceil(w / space)) + 0.5,
                np.arange(np.ceil(h / space)) + 0.5
            ), axis=-1) * space

        anchors = torch.from_numpy(anchors).float().to(images.device)
        return anchors.repeat(bs, 1, 1, 1).unsqueeze(-2)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
            self,
            images,
            prompt_points=None,
            prompt_labels=None,
            cell_nums=None,
            only_det=False
    ):
        # image_embeddings, outputs = self.image_encoder(images)
        features_list = self.image_encoder(images)


        img_feature = self.fusion_block(features_list)
        deep_feature = self.deep_feautre_conv(img_feature.contiguous())
        outputs = {}
        if prompt_points is not None:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(prompt_points, prompt_labels),
                boxes=None,
                masks=None,
            )

            low_res_masks, iou_predictions, feature = self.mask_decoder(
            # low_res_masks, iou_predictions, cls_predictions = self.mask_decoder(
                image_embeddings=img_feature,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                cell_nums=cell_nums,
                multimask_output=self.multimask
            )

            values, indices = torch.max(iou_predictions, dim=1)
            iou_predictions = values

            #images = torch.repeat_interleave(images, cell_nums, dim=0)
            deep_feature = torch.repeat_interleave(deep_feature, cell_nums, dim=0)
            refine_mask, pred_edge = self.detail_enhance(images, feature, deep_feature, cell_nums)

            if self.multimask:
                low_res_masks = low_res_masks[torch.arange(len(iou_predictions)), indices].unsqueeze(1)

            masks = F.interpolate(
                low_res_masks,
                images.shape[-2:],
                mode="bilinear",
                align_corners=False)[:, 0]

            outputs.update(
                pred_masks=masks,
                pred_ious=iou_predictions,
                pred_refine_masks=refine_mask.squeeze(1),
                edge = pred_edge
                # pred_logs=cls_predictions
            )

        return outputs

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            # input_size: Tuple[int, ...],
            # original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        # masks = masks[..., : input_size[0], : input_size[1]]
        # masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))  # 填充右边和下边
        return x


# Define block
class ResidualBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResidualBlock, self).__init__()

        # TODO: 3x3 convolution -> relu
        # the input and output channel number is channel_num
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, 3, padding=1),
            LayerNorm2d(channel_num),
            nn.GELU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, 3, padding=1),
            LayerNorm2d(channel_num),
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        # TODO: forward
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + residual
        out = self.gelu(x)
        return out
