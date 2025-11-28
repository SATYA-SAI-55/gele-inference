# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# -------------------------
# PVT Backbone wrapper
# -------------------------
class PVTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # features_only=True returns feature maps from stages
        self.backbone = timm.create_model("pvt_v2_b2", features_only=True, pretrained=True)

    def forward(self, x):
        # returns list of feature maps, e.g. [f1, f2, f3, f4]
        return self.backbone(x)


# -------------------------
# Helpers
# -------------------------
def channel_shuffle(x, groups):
    B, C, H, W = x.size()
    assert C % groups == 0, "channels must be divisible by groups"
    x = x.view(B, groups, C // groups, H, W)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    return x.view(B, C, H, W)


# D_SWSAM with groups=reduction to match checkpoint
class D_SWSAM(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.conv_h = nn.Conv2d(in_channels, in_channels, (1, 3), padding=(0, 1), groups=in_channels)
        self.conv_v = nn.Conv2d(in_channels, in_channels, (3, 1), padding=(1, 0), groups=in_channels)
        self.conv_d1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.conv_d2 = nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2, groups=in_channels)
        self.conv_reduce = nn.Conv2d(in_channels * 4, in_channels, 1)

        # Use the same grouped conv as in your training notebook:
        self.shuffle = nn.Conv2d(in_channels, in_channels, 1, groups=reduction)

        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, max(in_channels // reduction, 1), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_channels // reduction, 1), 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.conv_h(x)
        v = self.conv_v(x)
        d1 = self.conv_d1(x)
        d2 = self.conv_d2(x)
        concat = torch.cat([h, v, d1, d2], dim=1)
        fused = self.conv_reduce(concat)
        shuffled = self.shuffle(fused)
        attn = self.spatial(shuffled)
        return x * attn


# SWSAM with groups=reduction to match checkpoint
class SWSAM(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.shuffle = nn.Conv2d(in_channels, in_channels, 1, groups=reduction)
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, max(in_channels // reduction, 1), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_channels // reduction, 1), 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        shuffled = self.shuffle(x)
        attn = self.spatial(shuffled)
        return x * attn

# -------------------------
# KTM (knowledge transfer) - memory safe version
# -------------------------
class KTM(nn.Module):
    def __init__(self, in_channels1, in_channels2, inter_channels=64, kv_pool=1):
        """
        kv_pool: if >1, keys/values are pooled by this factor (reduces memory for attention)
        """
        super().__init__()
        self.inter_channels = inter_channels
        self.kv_pool = max(1, kv_pool)

        self.query_conv = nn.Conv2d(in_channels1, inter_channels, 1)
        self.key_conv = nn.Conv2d(in_channels2, inter_channels, 1)
        self.value_conv = nn.Conv2d(in_channels2, inter_channels, 1)
        self.out_conv = nn.Conv2d(inter_channels, in_channels1, 1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        """
        x1: target feature to be enhanced, shape [B, C1, H, W]
        x2: source feature to extract knowledge from, shape [B, C2, H2, W2]
        """
        B, _, H, W = x1.shape

        # Optionally reduce key/value spatial size for memory safety
        if self.kv_pool > 1:
            x2_pool = F.adaptive_avg_pool2d(x2, (max(1, x2.shape[2] // self.kv_pool),
                                                max(1, x2.shape[3] // self.kv_pool)))
        else:
            x2_pool = x2

        Q = self.query_conv(x1).view(B, self.inter_channels, -1).permute(0, 2, 1)  # [B, Nq, C]
        K = self.key_conv(x2_pool).view(B, self.inter_channels, -1)                 # [B, C, Nk]
        V = self.value_conv(x2_pool).view(B, self.inter_channels, -1).permute(0, 2, 1)  # [B, Nk, C]

        # Attention: [B, Nq, Nk]
        attn = torch.bmm(Q, K)  # may be large if Nq*Nk big
        attn = self.softmax(attn)

        out = torch.bmm(attn, V)  # [B, Nq, C]
        out = out.permute(0, 2, 1).contiguous().view(B, self.inter_channels, H, W)
        out = self.out_conv(out)
        return x1 + out


# -------------------------
# GeleNet full model
# -------------------------
class GeleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = PVTBackbone()

        # The PVT-v2-b2 feature map channels (timm typical outputs) are something like:
        # [64, 128, 320, 512] — match what you had in the notebook
        self.dswsam = D_SWSAM(64)
        # set kv_pool=2 to reduce attention memory
        self.ktm = KTM(in_channels1=128, in_channels2=320, inter_channels=64, kv_pool=2)
        self.swsam = SWSAM(512)

        # reduce channel -> 64 for fusion
        self.reduce1 = nn.Conv2d(64, 64, 1)
        self.reduce2 = nn.Conv2d(128, 64, 1)
        self.reduce3 = nn.Conv2d(512, 64, 1)

        # small decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        f1, f2, f3, f4 = self.backbone(x)  # list of 4 tensors

        # apply attention modules
        f1 = self.dswsam(f1)
        f2 = self.ktm(f2, f3)
        f4 = self.swsam(f4)

        # channel reduce
        f1 = self.reduce1(f1)
        f2 = self.reduce2(f2)
        f4 = self.reduce3(f4)

        # upsample to f1 spatial size
        f2 = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(f4, size=f1.shape[2:], mode='bilinear', align_corners=True)

        fused = f1 + f2 + f4
        out = self.decoder(fused)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)  # resize to input
        return out


# convenience loader for state dicts
def load_model(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeleNet().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
