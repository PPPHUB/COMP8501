import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Optional

class RecurrentDecoder(nn.Module):
    def __init__(self, feature_channels, decoder_channels):
        super().__init__()
        if len(feature_channels)!=5:
            self.avgpool = AvgPool()
            self.decode4 = BottleneckBlock(feature_channels[3])
            self.decode3 = UpsamplingBlock(feature_channels[3], feature_channels[2], 3, decoder_channels[0])
            self.decode2 = UpsamplingBlock(decoder_channels[0], feature_channels[1], 3, decoder_channels[1])
            self.decode1 = UpsamplingBlock(decoder_channels[1], feature_channels[0], 3, decoder_channels[2])
            self.decode0 = OutputBlock(decoder_channels[2], 3, decoder_channels[3])


    def forward(self,
                s0: Tensor, f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor,
                r1: Optional[Tensor], r2: Optional[Tensor],
                r3: Optional[Tensor], r4: Optional[Tensor],c1: Optional[Tensor], c2: Optional[Tensor],
                c3: Optional[Tensor], c4: Optional[Tensor],f0: Optional[Tensor]):

        s1, s2, s3 = self.avgpool(s0)
        print(f4.shape)
        x4, r4,c4 = self.decode4(f4, r4,c4)
        print("x4",x4.shape)
        print("x4",f3.shape)
        print("x4",s3.shape)
        x3, r3,c3 = self.decode3(x4.unsqueeze(0), f3.unsqueeze(0), s3, r3,c3)
        x2, r2,c2 = self.decode2(x3, f2, s2, r2,c2)
        x1, r1,c1 = self.decode1(x2, f1, s1, r1,c1)


        print("x1",x1.shape)
        x0 = self.decode0(x1, s0)
        print("x0",x0.shape)
        return x0, r1, r2, r3, r4,c1,c2,c3,c4
    

class AvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool2d(2, 2, count_include_pad=False, ceil_mode=True)
        
    def forward_single_frame(self, s0):
        s1 = self.avgpool(s0)
        s2 = self.avgpool(s1)
        s3 = self.avgpool(s2)
        return s1, s2, s3
    
    def forward_time_series(self, s0):
        B, T = s0.shape[:2]
        s0 = s0.flatten(0, 1)
        s1, s2, s3 = self.forward_single_frame(s0)
        s1 = s1.unflatten(0, (B, T))
        s2 = s2.unflatten(0, (B, T))
        s3 = s3.unflatten(0, (B, T))
        return s1, s2, s3
    
    def forward(self, s0):
        if s0.ndim == 5:
            return self.forward_time_series(s0)
        else:
            return self.forward_single_frame(s0)


class BottleneckBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.gru = ConvGRU(channels // 2)
        
    def forward(self, x, r: Optional[Tensor],c: Optional[Tensor]):
        a, b = x.split(self.channels // 2, dim=-3)
        b, r,c = self.gru(b, r,c)
        x = torch.cat([a, b], dim=-3)
        return x, r,c

    
class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, src_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.gru = ConvGRU(out_channels // 2)

    def forward_single_frame(self, x, f, s, r: Optional[Tensor],c: Optional[Tensor]):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        print("x",x.shape)
        print("f",f.shape)
        print(s.shape,"s")
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        a, b = x.split(self.out_channels // 2, dim=1)
        b, r,c = self.gru(b, r,c)
        x = torch.cat([a, b], dim=1)
        return x, r,c
    
    def forward_time_series(self, x, f, s, r: Optional[Tensor],c: Optional[Tensor]):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        f = f.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        x = x.unflatten(0, (B, T))
        a, b = x.split(self.out_channels // 2, dim=2)
        b, r,c = self.gru(b, r,c)
        x = torch.cat([a, b], dim=2)
        return x, r,c
    
    def forward(self, x, f, s, r: Optional[Tensor],c: Optional[Tensor]):
        if x.ndim == 5:
            return self.forward_time_series(x, f, s, r,c)
        else:
            return self.forward_single_frame(x, f, s, r,c)


class OutputBlock(nn.Module):
    def __init__(self, in_channels, src_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        
    def forward_single_frame(self, x, s):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        return x
    
    def forward_time_series(self, x, s):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        x = x.unflatten(0, (B, T))
        return x
    
    def forward(self, x, s):
        if x.ndim == 5:
            return self.forward_time_series(x, s)
        else:
            return self.forward_single_frame(x, s)


class ConvGRU(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        self.channels = channels
        self.ih = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding),
            nn.Sigmoid()
        )
        self.hh = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size, padding=padding),
            nn.Tanh()
        )
        self.ft = nn.Sequential(
            nn.Conv2d(channels * 2, channels , kernel_size, padding=padding),
            nn.Sigmoid()
        )
        self.it = nn.Sequential(
            nn.Conv2d(channels * 2, channels , kernel_size, padding=padding),
            nn.Sigmoid()
        )
        self.ct = nn.Sequential(
            nn.Conv2d(channels * 2, channels , kernel_size, padding=padding),
            nn.Tanh()
        )
        self.ot = nn.Sequential(
            nn.Conv2d(channels * 2, channels , kernel_size, padding=padding),
            nn.Tanh()
        )
        self.tanh=nn.Tanh()
    def forward_single_frame(self, x, h,c):
        it=self.it(torch.cat([h,x], dim=1))
        ft=self.ft(torch.cat([h,x], dim=1))
        ct=self.ct(torch.cat([h,x], dim=1))
        c=ft*c+it*ct
        ot=self.ot(torch.cat([h,x], dim=1))
        h=ot*self.tanh(c)
        return h, h,c
    #def forward_single_frame(self, x, h):
     ##   r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
    #    c = self.hh(torch.cat([x, r * h], dim=1))
    #    h = (1 - z) * h + z * c

      #  return h, h

    def forward_time_series(self, x, h,c):
        o = []
        for xt in x.unbind(dim=1):
            ot, h,c = self.forward_single_frame(xt, h,c)
            o.append(ot)
        o = torch.stack(o, dim=1)
        return o, h,c
        
    def forward(self, x, h: Optional[Tensor],c: Optional[Tensor]):
        if h is None:
            h = torch.zeros((x.size(0), x.size(-3), x.size(-2), x.size(-1)),
                            device=x.device, dtype=x.dtype)
        if c is None :
            c = torch.zeros((x.size(0), x.size(-3), x.size(-2), x.size(-1)),
                            device=x.device, dtype=x.dtype)
        if x.ndim == 5:
            return self.forward_time_series(x, h,c)
        else:
            return self.forward_single_frame(x, h,c)


class Projection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward_single_frame(self, x):
        return self.conv(x)
    
    def forward_time_series(self, x):
        B, T = x.shape[:2]
        return self.conv(x.flatten(0, 1)).unflatten(0, (B, T))
        
    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)
    