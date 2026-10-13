"""Baseline segmentation networks for matched D10 fetal-heart comparisons.

Four architectures are provided for the AISH SOTA comparison:

- ``unet``     U-Net with an ImageNet-pretrained ResNet-50 encoder.
- ``attunet``  Attention U-Net (attention-gated skips, same R50 encoder).
- ``unetpp``   UNet++ nested decoder (same R50 encoder).
- ``swinunet`` Swin-Unet (pure Transformer encoder-decoder, Swin-T sizing).

Conventions matched to the TransUNet protocol: grayscale 1-channel input,
384x384 training resolution, 12 output classes (background + 11 structures).
The CNN encoders reuse torchvision's ImageNet ResNet-50 weights; the first
convolution is adapted to one channel by averaging the pretrained RGB kernels,
mirroring how TransUNet's R50 hybrid consumes the same backbone family.

Swin-Unet uses window size 6 so that every stage resolution at a 384 input
(96/48/24/12) is divisible by the window; the official Swin-Unet release uses
window 7 at 224. An optional ImageNet-1K Swin-T checkpoint can be loaded with
``swin_pretrained``; encoder blocks whose names/shapes match are kept.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# ---------------------------------------------------------------------------
# Shared ResNet-50 encoder (ImageNet-1K, grayscale input)
# ---------------------------------------------------------------------------

RESNET_STAGES = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")


class ResNet50Encoder(nn.Module):
    """Returns feature pyramid [64, 256, 512, 1024, 2048] at strides 2..32."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = torchvision.models.resnet50(weights=weights)
        if pretrained:
            rgb_weight = resnet.conv1.weight.data  # (64, 3, 7, 7)
            resnet.conv1.weight.data = rgb_weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
        else:
            resnet.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.stem(x)
        f0 = x  # stride-2 stem feature
        x = self.maxpool(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f0, f1, f2, f3, f4]


class ConvBlock(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2, as in the original U-Net."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class SegHead(nn.Module):
    """Classification head with a final x2 upsample so logits match the input
    resolution, as TransUNet's segmentation head does."""

    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, num_classes, 1),
        )

    def forward(self, x):
        return self.head(x)


class FullResSegHead(SegHead):
    def forward(self, x):
        x = self.head(x)
        return F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNetResNet50(nn.Module):
    decoder_channels = (1024, 512, 256, 64)

    def __init__(self, num_classes=12, pretrained=True):
        super().__init__()
        self.encoder = ResNet50Encoder(pretrained)
        self.up4 = UpBlock(2048, 1024)
        self.up3 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up1 = UpBlock(256, 64)
        self.head = FullResSegHead(64, num_classes)

    def forward(self, x):
        f0, f1, f2, f3, f4 = self.encoder(x)
        d4 = self.up4(f4, f3)
        d3 = self.up3(d4, f2)
        d2 = self.up2(d3, f1)
        d1 = self.up1(d2, f0)
        return self.head(d1)


# ---------------------------------------------------------------------------
# Attention U-Net (Oktay et al., 2018)
# ---------------------------------------------------------------------------

class AttentionGate(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.w_gate = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch)
        )
        self.w_skip = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        # gate: coarse feature; bring it to the skip resolution before gating
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        a = self.relu(self.w_gate(gate) + self.w_skip(skip))
        return skip * self.psi(a)


class AttentionUNetResNet50(nn.Module):
    def __init__(self, num_classes=12, pretrained=True):
        super().__init__()
        self.encoder = ResNet50Encoder(pretrained)
        self.up4 = UpBlock(2048, 1024)
        self.up3 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up1 = UpBlock(256, 64)
        self.gate4 = AttentionGate(2048, 1024, 512)
        self.gate3 = AttentionGate(1024, 512, 256)
        self.gate2 = AttentionGate(512, 256, 128)
        self.gate1 = AttentionGate(256, 64, 32)
        self.head = FullResSegHead(64, num_classes)

    def forward(self, x):
        f0, f1, f2, f3, f4 = self.encoder(x)
        d4 = self.up4(f4, self.gate4(f4, f3))
        d3 = self.up3(d4, self.gate3(d4, f2))
        d2 = self.up2(d3, self.gate2(d3, f1))
        d1 = self.up1(d2, self.gate1(d2, f0))
        return self.head(d1)


# ---------------------------------------------------------------------------
# UNet++ (Zhou et al., 2018), deep supervision disabled: read-out from X^{0,4}
# ---------------------------------------------------------------------------

class NestedUNetResNet50(nn.Module):
    enc_channels = (64, 256, 512, 1024, 2048)

    def __init__(self, num_classes=12, pretrained=True):
        super().__init__()
        c = self.enc_channels
        self.encoder = ResNet50Encoder(pretrained)
        # upsample path i -> i-1 maps c[i+1] to c[i]
        self.up = nn.ModuleList(
            [
                nn.ConvTranspose2d(c[i + 1], c[i], kernel_size=2, stride=2)
                for i in range(4)
            ]
        )
        # nodes[i][j], j >= 1: input = c[i] (upsampled X^{i-1,j-1}) + c[i] (X^{i,j-1})
        self.nodes = nn.ModuleDict()
        for i in range(5):
            for j in range(1, 5 - i):
                self.nodes[f"x{i}_{j}"] = ConvBlock(c[i] * 2, c[i])
        self.head = FullResSegHead(c[0], num_classes)

    def _up(self, index, x):
        x = self.up[index](x)
        return x

    def forward(self, x):
        c = self.enc_channels
        f = self.encoder(x)  # f[0..4]
        # X[i][0] are the encoder features.
        nodes = {f"x{i}_0": f[i] for i in range(5)}
        for j in range(1, 5):
            for i in range(5 - j):
                upsampled = self._up(i, nodes[f"x{i + 1}_{j - 1}"])
                if upsampled.shape[-2:] != nodes[f"x{i}_{j - 1}"].shape[-2:]:
                    upsampled = F.interpolate(
                        upsampled,
                        size=nodes[f"x{i}_{j - 1}"].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                nodes[f"x{i}_{j}"] = self.nodes[f"x{i}_{j}"](
                    torch.cat([nodes[f"x{i}_{j - 1}"], upsampled], dim=1)
                )
        return self.head(nodes["x0_4"])


# ---------------------------------------------------------------------------
# Swin-Unet (Cao et al., 2021), Swin-T sizing, window 6, 384 input
# ---------------------------------------------------------------------------

def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)


def window_reverse(windows, window_size, h, w):
    b, _, _, c = windows.shape
    b = b // ((h // window_size) * (w // window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, c)


def swin_attention_mask(resolution, window_size, shift_size, device):
    if shift_size == 0:
        return None
    img_mask = torch.zeros((1, resolution, resolution, 1), device=device)
    slices = (
        (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        if resolution > window_size
        else (slice(0, None),)
    )
    count = 0
    for h in slices:
        for w in slices:
            img_mask[:, h, w, :] = count
            count += 1
    windows = window_partition(img_mask, window_size).view(-1, window_size * window_size)
    attn_mask = windows.unsqueeze(1) - windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )
    return attn_mask


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size), torch.arange(window_size), indexing="ij"
            )
        ).flatten(1)  # 2 x n
        relative = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + window_size - 1
        index = relative[..., 0] * (2 * window_size - 1) + relative[..., 1]
        self.register_buffer("relative_position_index", index, persistent=False)

    def forward(self, x, mask=None):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias_index = self.relative_position_index.view(-1)
        bias = self.relative_position_bias_table[bias_index].view(n, n, -1)
        attn = attn + bias.permute(2, 0, 1)
        if mask is not None:
            n_windows = mask.shape[0]
            attn = attn.view(b // n_windows, n_windows, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class SwinBlock(nn.Module):
    def __init__(self, dim, resolution, num_heads, window_size, shift_size):
        super().__init__()
        if resolution <= window_size:
            window_size, shift_size = resolution, 0
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x, resolution):
        h = w = resolution
        b, n, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        mask = swin_attention_mask(resolution, self.window_size, self.shift_size, x.device)
        windows = window_partition(x, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, c)
        attended = self.attn(windows, mask)
        x = window_reverse(
            attended.view(-1, self.window_size, self.window_size, c), self.window_size, h, w
        )
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(b, n, c) + shortcut
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, resolution):
        h = w = resolution
        b, n, c = x.shape
        x = x.view(b, h // 2, 2, w // 2, 2, c).permute(0, 1, 3, 4, 2, 5).reshape(b, (h // 2) * (w // 2), 4 * c)
        return self.reduction(self.norm(x))


class PatchExpand(nn.Module):
    """x2 resolution up-expansion: dim -> dim/2 channels via Linear(dim, 2*dim)
    followed by pixel_shuffle(2), matching the official Swin-Unet layer."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.expand = nn.Linear(dim, dim * 2, bias=False)

    def forward(self, x, resolution):
        h = w = resolution
        x = self.expand(self.norm(x))
        b, n, c = x.shape
        x = x.view(b, h, w, c)
        x = x.permute(0, 3, 1, 2)  # b c h w
        x = F.pixel_shuffle(x, 2)
        return x.flatten(2).transpose(1, 2), resolution * 2


class SwinUnet(nn.Module):
    """Swin-Unet with the official Swin-T backbone (embed 96, depths 2-2-6-2,
    ~28M parameters) so the ImageNet checkpoint loads.

    Window size follows the input: window 6 divides 96/48/24/12 at 384 input
    (checkpoint bias tables are window-7 and are re-initialised); window 7 is
    the official 224-pixel setting and divides 56/28/14/7 exactly, so the
    checkpoint bias tables load unchanged."""

    embed_dim = 96
    depths = (2, 2, 6, 2)
    num_heads = (3, 6, 12, 24)

    def __init__(self, num_classes=12, pretrained=True, swin_pretrained=None, img_size=384):
        super().__init__()
        d = self.embed_dim
        self.window_size = 7 if img_size == 224 else 6
        stem_resolution = img_size // 4
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, d, kernel_size=4, stride=4, bias=False),
            nn.GroupNorm(1, d),  # LayerNorm-equivalent over channels for conv features
        )
        self.layers = nn.ModuleList()
        for i, depth in enumerate(self.depths):
            resolution = stem_resolution // (2 ** i)
            blocks = nn.ModuleList(
                [
                    SwinBlock(
                        d * (2 ** i),
                        resolution,
                        self.num_heads[i],
                        self.window_size,
                        self.window_size // 2 if (bi % 2 == 1 and resolution > self.window_size) else 0,
                    )
                    for bi in range(depth)
                ]
            )
            downsample = PatchMerging(d * (2 ** i)) if i < 3 else None
            self.layers.append(nn.ModuleDict({"blocks": blocks, "downsample": downsample}))
        self.norm = nn.LayerNorm(d * 8)

        # Decoder: three skip-connected expands plus two final expands.
        self.dec_norms = nn.ModuleList()
        self.dec_reduces = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.expands = nn.ModuleList()
        dec_dims = [d * 8, d * 4, d * 2, d]
        for i, dim in enumerate(dec_dims[:-1]):
            self.expands.append(PatchExpand(dim))
            # expand output has dim//2 channels; concat with same-width skip
            # gives dim channels again, then reduce back to dim//2
            self.dec_reduces.append(nn.Linear(dim, dim // 2, bias=False))
            self.dec_norms.append(nn.LayerNorm(dim))
        # two final expands without skips bring 1/4-scale features to full res
        self.expands.append(PatchExpand(d))
        self.expands.append(PatchExpand(d // 2))
        self.final_norm = nn.LayerNorm(d // 4)
        self.head = nn.Linear(d // 4, num_classes)

        if swin_pretrained:
            self.load_swin_pretrained(swin_pretrained)

    def load_swin_pretrained(self, path):
        """Load an official Swin-T ImageNet checkpoint where shapes allow.

        Keys are dropped when the window size (bias tables), resolution, or the
        decoder/head (absent in ImageNet pretraining) differ. Shape-mismatched
        tensors are skipped rather than raising, so any checkpoint variant can
        be attempted; the run log reports what was actually applied.
        """
        state = torch.load(path, map_location="cpu")
        state = state.get("model", state)
        own = self.state_dict()
        applied, skipped = [], []
        for key, value in state.items():
            if key.endswith(("relative_position_bias_table", "relative_position_index", "attn_mask")):
                skipped.append((key, "window-size dependent"))
                continue
            if key.startswith("patch_embed.proj."):
                new_key = key.replace("patch_embed.proj.", "patch_embed.0.")
                if value.ndim == 4 and value.shape[1] == 3:
                    value = value.mean(dim=1, keepdim=True)  # RGB -> grayscale
            elif key.startswith("patch_embed.norm."):
                new_key = key.replace("patch_embed.norm.", "patch_embed.1.")
            else:
                new_key = key
            # Official checkpoint names the MLP linears fc1/fc2; ours is a
            # plain Sequential (mlp.0 / mlp.2).
            new_key = new_key.replace("mlp.fc1.", "mlp.0.").replace("mlp.fc2.", "mlp.2.")
            if new_key not in own or own[new_key].shape != value.shape:
                skipped.append((key, "missing or shape mismatch"))
                continue
            own[new_key] = value
            applied.append(new_key)
        self.load_state_dict(own, strict=True)
        print(
            f"[swinunet] pretrained={path}: applied {len(applied)}/{len(state)} tensors, "
            f"skipped {len(skipped)} (window-size dependent / decoder / shape mismatch)"
        )

    def forward(self, x):
        x = self.patch_embed(x)  # b d h w
        skips = []
        current = x.flatten(2).transpose(1, 2)
        resolution = x.shape[-1]
        for i, layer in enumerate(self.layers):
            for block in layer["blocks"]:
                current = block(current, resolution)
            skips.append(current)
            if layer["downsample"] is not None:
                current = layer["downsample"](current, resolution)
                resolution //= 2
        current = self.norm(current)

        # decoder with skips x0, x1, x2
        for i in range(3):
            current, resolution = self.expands[i](current, resolution)
            skip = skips[2 - i]
            current = self.dec_norms[i](torch.cat([current, skip], dim=-1))
            current = self.dec_reduces[i](current)
            for block in self.layers[2 - i]["blocks"]:
                current = block(current, resolution)
        # two final expands without skips bring 1/4-scale features to full res
        current, resolution = self.expands[3](current, resolution)
        current, resolution = self.expands[4](current, resolution)
        x = self.final_norm(current)
        logits = self.head(x)  # b n num_classes
        b = logits.shape[0]
        return logits.view(b, resolution, resolution, -1).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

BASELINE_ARCHS = ("unet", "attunet", "unetpp", "swinunet")


def build_baseline(arch, num_classes=12, img_size=384, pretrained=True, swin_pretrained=None):
    if arch == "unet":
        return UNetResNet50(num_classes=num_classes, pretrained=pretrained)
    if arch == "attunet":
        return AttentionUNetResNet50(num_classes=num_classes, pretrained=pretrained)
    if arch == "unetpp":
        return NestedUNetResNet50(num_classes=num_classes, pretrained=pretrained)
    if arch == "swinunet":
        return SwinUnet(
            num_classes=num_classes,
            pretrained=pretrained,
            swin_pretrained=swin_pretrained,
            img_size=img_size,
        )
    raise ValueError(f"unknown baseline arch: {arch}; expected one of {BASELINE_ARCHS}")
