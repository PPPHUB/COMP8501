import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, List

from .mobilenetv3 import MobileNetV3LargeEncoder
from .resnet import ResNet50Encoder
from .lraspp import LRASPP
from .decoder import RecurrentDecoder, Projection
from .fast_guided_filter import FastGuidedFilterRefiner
from .deep_guided_filter import DeepGuidedFilterRefiner
from .u2netencoder import U2NET
from .decoder import ConvGRU
class MattingNetwork(nn.Module):
    def __init__(self,
                 variant: str = 'mobilenetv3',
                 refiner: str = 'deep_guided_filter',
                 pretrained_backbone: bool = False):
        super().__init__()
        assert variant in ['mobilenetv3', 'resnet50','encoder2']
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        self.variant=variant
        if variant == 'mobilenetv3':
            if variant == 'mobilenetv3':
                self.backbone = MobileNetV3LargeEncoder(pretrained_backbone)
                self.aspp = LRASPP(960, 128)
                self.decoder = RecurrentDecoder([16, 24, 40, 128], [80, 40, 32, 16])

        if variant == 'encoder2':

            self.backbone = U2NET()# init the U-encoder block  part and read the pre-trained model.
            self.backbone.load_state_dict(torch.load("/content/drive/MyDrive/u2net_human_seg.pth") )
            self.aspp = LRASPP(512, 512)
            self.decoder = RecurrentDecoder([64, 128, 256,512], [128, 64, 32, 16])
            self.skipconn= U2NET()# for sikp connection
            self.skipconn.load_state_dict(torch.load("/content/drive/MyDrive/u2net_human_seg.pth") )
            self.gruskip=[ConvGRU(i) for i in [64, 128, 256,512]]
        else:
            self.backbone = ResNet50Encoder(pretrained_backbone)
            self.aspp = LRASPP(2048, 256)
            self.decoder = RecurrentDecoder([64, 256, 512, 256], [128, 64, 32, 16])
            
        self.project_mat = Projection(16, 4)
        self.project_seg = Projection(16, 1)

        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()
        
    def forward(self,
                src: Tensor,
                r1: Optional[Tensor] = None,
                r2: Optional[Tensor] = None,
                r3: Optional[Tensor] = None,
                r4: Optional[Tensor] = None,
                c1: Optional[Tensor] = None,
                c2: Optional[Tensor] = None,
                c3: Optional[Tensor] = None,
                c4: Optional[Tensor] = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):
        
        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            src_sm = src



        if self.variant!="encoder2":
            f1, f2, f3, f4 = self.backbone(src_sm)
        else:
            f0,f1, f2, f3, f4 = self.backbone(src_sm)# have different output.
            if hasattr(self,"h1"):
                f1+=self.gruskip[0]( self.h1)
                f2+=self.gruskip[1]( self.h2)
                f3+=self.gruskip[2]( self.h3)
                f4+=self.gruskip[3]( self.h4)


        f4 = self.aspp(f4)

        hid, *rec = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4,c1,c2,c3,c4,f0)
        if self.variant=="encoder2"
            self.h1,self.h2,self.h3,self.h4=self.skipconn(hid)
        if not segmentation_pass:
            fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
            if downsample_ratio != 1:
                fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
            fgr = fgr_residual + src
            fgr = fgr.clamp(0., 1.)
            pha = pha.clamp(0., 1.)
            return [fgr, pha, *rec]
        else:
            seg = self.project_seg(hid)
            return [seg, *rec]

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x
