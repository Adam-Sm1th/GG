"""
Gaussian Noise Watermarking Autoencoder  v5.0 (Strong Attack / Large AE)
========================================================================
更强的 latent proxy 攻击 + 更大容量的条件自编码器。

核心变化：
  1. Encoder 从两层卷积升级为 FiLM 条件残差网络，消息同时以 dense map 和 style 向量注入。
  2. Decoder 从轻量 CNN 升级为空间域 + Haar 频域双分支，多尺度残差读取 256 bit。
  3. AttackLayer 变成组合攻击：通道漂移、模糊、裁剪缩放、频带衰减、局部擦除、
     通道 dropout、低频噪声、STE 量化和 masked Gaussian。
  4. 仍然强制输出近似 N(0,1)，避免 Stable Diffusion 初始 latent 统计漂移。
"""

import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ─────────────────────────────────────────────────────────────
#  Shared blocks
# ─────────────────────────────────────────────────────────────

def _num_groups(channels, max_groups=32):
    for groups in (32, 16, 8, 4, 2, 1):
        if groups <= max_groups and channels % groups == 0:
            return groups
    return 1


def _standardize_latent(x, eps=1e-6):
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    std = x.std(dim=[1, 2, 3], keepdim=True, unbiased=False)
    return (x - mean) / (std + eps)


def haar_dwt(x):
    """Differentiable Haar DWT: [B,C,H,W] -> four [B,C,H/2,W/2] bands."""
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    LL = (x00 + x01 + x10 + x11) * 0.25
    HL = (x00 - x01 + x10 - x11) * 0.25
    LH = (x00 + x01 - x10 - x11) * 0.25
    HH = (x00 - x01 - x10 + x11) * 0.25
    return LL, HL, LH, HH


def haar_idwt(LL, HL, LH, HH):
    B, C, H, W = LL.shape
    out = torch.empty(B, C, H * 2, W * 2, device=LL.device, dtype=LL.dtype)
    out[:, :, 0::2, 0::2] = LL + HL + LH + HH
    out[:, :, 0::2, 1::2] = LL - HL + LH - HH
    out[:, :, 1::2, 0::2] = LL + HL - LH - HH
    out[:, :, 1::2, 1::2] = LL - HL - LH + HH
    return out


class FiLMResBlock(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(channels), channels)
        self.norm2 = nn.GroupNorm(_num_groups(channels), channels)
        self.act = nn.SiLU(inplace=False)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.affine = nn.Sequential(
            nn.SiLU(inplace=False),
            nn.Linear(cond_dim, channels * 2),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.35))
        nn.init.zeros_(self.affine[-1].weight)
        nn.init.zeros_(self.affine[-1].bias)

    def forward(self, x, cond):
        h = self.norm1(x)
        gamma, beta = self.affine(cond).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        h = h * (1.0 + 0.1 * gamma) + 0.1 * beta
        h = self.conv1(self.act(h))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h * self.res_scale


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_num_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.35))

    def forward(self, x):
        return x + self.net(x) * self.res_scale


# ─────────────────────────────────────────────────────────────
#  Encoder — FiLM 条件残差大模型 + 强制正态化
# ─────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    noise  : [B, 16, 64, 64]
    msg    : [B, 256]  float {0,1}
    output : [B, 16, 64, 64]
    """
    def __init__(self, noise_ch=16, msg_len=256, spatial=64,
                 width=128, cond_dim=512):
        super().__init__()
        if spatial % 4 != 0:
            raise ValueError("spatial must be divisible by 4.")

        self.noise_ch = noise_ch
        self.spatial = spatial
        self.width = width
        self.base_spatial = spatial // 4

        self.msg_proj = nn.Sequential(
            nn.Linear(msg_len, 1024),
            nn.SiLU(inplace=True),
            nn.Linear(1024, width * self.base_spatial * self.base_spatial),
            nn.SiLU(inplace=True),
        )

        self.stem = nn.Sequential(
            nn.Conv2d(noise_ch + width, width, 3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(width), width),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResBlock(width),
            ResBlock(width),
            ResBlock(width),
            ResBlock(width),
        )
        self.to_residual = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(width), width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, noise_ch, 3, padding=1, bias=True),
            nn.Tanh(),
        )

        # 强攻击需要更足的起始能量；真正幅度仍由 L_amp 和 AMP_MARGIN 约束。
        self.alpha = nn.Parameter(torch.tensor(0.12))
        nn.init.normal_(self.to_residual[-2].weight, std=0.01)
        nn.init.zeros_(self.to_residual[-2].bias)

    def forward(self, noise, msg):
        B, C, H, W = noise.shape
        msg_feat = self.msg_proj(msg).view(B, self.width, self.base_spatial, self.base_spatial)
        msg_feat = F.interpolate(msg_feat, size=(H, W), mode="bilinear", align_corners=False)

        x = self.stem(torch.cat([noise, msg_feat], dim=1))
        x = self.blocks(x)
        residual = self.to_residual(x)

        alpha = torch.clamp(self.alpha, 0.01, 0.35)
        encoded = noise + alpha * residual
        mean = encoded.mean(dim=[1, 2, 3], keepdim=True)
        std = encoded.std(dim=[1, 2, 3], keepdim=True, unbiased=False)
        return (encoded - mean) / (std + 1e-7)
        # return encoded  # 由 loss 强制归一化，保持梯度流畅，攻击层也更直接地作用在原始空间。


# ─────────────────────────────────────────────────────────────
#  Attack Layer — 多攻击组合，STE / 可微近似
# ─────────────────────────────────────────────────────────────

class AttackLayer(nn.Module):
    """
    Latent-space proxy for: image distortion -> VAE encode -> diffusion inverse.

    这里不是模拟单个 JPEG，而是训练期随机组合多种破坏，让 Decoder 学到更宽的
    吸收域。所有连续部分保留梯度，量化用 STE。
    """
    def __init__(self, threshold=0.5, std_min=0.0, std_max=0.50,
                 quant_max=0.14, blur_max=0.75, resize_min=0.45,
                 gain_max=0.18, bias_max=0.12, freq_drop_max=0.65,
                 cutout_max=0.35, channel_drop_max=0.22,
                 coarse_noise_max=0.25, max_shift=5):
        super().__init__()
        self.threshold = threshold
        self.std_min = std_min
        self.std_max = std_max
        self.attack_std = std_min
        self.attack_level = 0.0
        self.quant_max = quant_max
        self.blur_max = blur_max
        self.resize_min = resize_min
        self.gain_max = gain_max
        self.bias_max = bias_max
        self.freq_drop_max = freq_drop_max
        self.cutout_max = cutout_max
        self.channel_drop_max = channel_drop_max
        self.coarse_noise_max = coarse_noise_max
        self.max_shift = max_shift

    def set_strength(self, epoch, total):
        t = min(max(epoch / max(total, 1), 0.0), 1.0)
        t = 0.5 - 0.5 * math.cos(math.pi * t)
        self.attack_level = t
        self.attack_std = self.std_min + t * (self.std_max - self.std_min)

    @staticmethod
    def _ste_quantize(x, step):
        quantized = torch.round(x / step) * step
        return x + (quantized - x).detach()

    def _resample_attack(self, y, level):
        B, C, H, W = y.shape
        scale = self.resize_min + (1.0 - self.resize_min) * random.random()
        aspect = 0.85 + 0.30 * random.random()
        new_h = max(8, min(H, int(H * scale * aspect)))
        new_w = max(8, min(W, int(W * scale / aspect)))
        top = random.randint(0, H - new_h) if H > new_h else 0
        left = random.randint(0, W - new_w) if W > new_w else 0
        cropped = y[:, :, top:top + new_h, left:left + new_w]
        mode = "bicubic" if random.random() < 0.45 else "bilinear"
        y = F.interpolate(cropped, size=(H, W), mode=mode, align_corners=False)
        if random.random() < 0.35 * level:
            y = F.interpolate(y, size=(max(8, H // 2), max(8, W // 2)),
                              mode="bilinear", align_corners=False)
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        return y

    def _frequency_attack(self, y, level):
        LL, HL, LH, HH = haar_dwt(y)
        drop = self.freq_drop_max * level * (0.35 + 0.65 * random.random())
        ll_blur = F.avg_pool2d(LL, kernel_size=3, stride=1, padding=1)
        LL = LL.lerp(ll_blur, 0.25 * drop)

        # JPEG/VAE 往往先伤高频，随机不等比例压三个细节频段。
        bands = []
        for band in (HL, LH, HH):
            atten = 1.0 - drop * (0.55 + 0.45 * random.random())
            bands.append(band * max(0.05, atten))
        return haar_idwt(LL, bands[0], bands[1], bands[2])

    def _cutout_attack(self, y, level):
        B, C, H, W = y.shape
        max_frac = max(0.02, self.cutout_max * level)
        mask = torch.ones(B, 1, H, W, device=y.device, dtype=y.dtype)
        num_boxes = 1 if random.random() < 0.7 else 2
        for b in range(B):
            for _ in range(num_boxes):
                box_h = max(2, int(H * max_frac * (0.35 + 0.65 * random.random())))
                box_w = max(2, int(W * max_frac * (0.35 + 0.65 * random.random())))
                top = random.randint(0, max(H - box_h, 0))
                left = random.randint(0, max(W - box_w, 0))
                mask[b, :, top:top + box_h, left:left + box_w] = 0.0
        coarse = torch.randn(B, C, 4, 4, device=y.device, dtype=y.dtype)
        fill = F.interpolate(coarse, size=(H, W), mode="bilinear", align_corners=False)
        fill = fill * (0.35 + self.attack_std)
        return y * mask + fill * (1.0 - mask)

    def forward(self, x):
        if self.attack_std < 1e-6 and self.attack_level < 1e-6:
            return x
        level = self.attack_level

        y = x
        B, C, H, W = y.shape

        gain_span = self.gain_max * level
        bias_span = self.bias_max * level
        gain = 1.0 + (torch.rand(B, C, 1, 1, device=y.device, dtype=y.dtype) * 2.0 - 1.0) * gain_span
        bias = (torch.rand(B, C, 1, 1, device=y.device, dtype=y.dtype) * 2.0 - 1.0) * bias_span
        y = y * gain + bias

        if random.random() < 0.50 * level:
            shift = max(1, int(self.max_shift * level))
            y = torch.roll(y, shifts=(random.randint(-shift, shift), random.randint(-shift, shift)), dims=(2, 3))

        if random.random() < 0.90 * level:
            kernel = random.choice((3, 5, 7))
            blurred = F.avg_pool2d(y, kernel_size=kernel, stride=1, padding=kernel // 2)
            blend = self.blur_max * level * (0.25 + 0.75 * random.random())
            y = y.lerp(blurred, blend)

        if random.random() < 0.85 * level:
            y = self._resample_attack(y, level)

        if random.random() < 0.80 * level:
            y = self._frequency_attack(y, level)

        if random.random() < 0.55 * level:
            y = self._cutout_attack(y, level)

        drop_prob = self.channel_drop_max * level
        if drop_prob > 1e-4 and random.random() < 0.55 * level:
            keep = (torch.rand(B, C, 1, 1, device=y.device, dtype=y.dtype) > drop_prob).to(y.dtype)
            repl = torch.randn(B, C, 4, 4, device=y.device, dtype=y.dtype)
            repl = F.interpolate(repl, size=(H, W), mode="bilinear", align_corners=False)
            y = y * keep + repl * (1.0 - keep) * (0.25 + self.attack_std)

        coarse_amp = self.coarse_noise_max * level
        if coarse_amp > 1e-6:
            coarse = torch.randn(B, C, 8, 8, device=y.device, dtype=y.dtype)
            coarse = F.interpolate(coarse, size=(H, W), mode="bilinear", align_corners=False)
            y = y + coarse * coarse_amp

        q_step = self.quant_max * level * (0.35 + 0.65 * random.random())
        if q_step > 1e-6:
            y = self._ste_quantize(y, q_step)

        soft_mask = torch.sigmoid(-(x.abs() - self.threshold) * 8.0)
        y = y + soft_mask * self.attack_std * torch.randn_like(x)

        return y


# ─────────────────────────────────────────────────────────────
#  Decoder — 空间域 + Haar 频域双分支大模型
# ─────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    def __init__(self, noise_ch=16, msg_len=256, width=128):
        super().__init__()
        in_ch = noise_ch * 4
        c1, c2, c3, c4 = width, width * 2, width * 3, width * 4

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(c1), c1),
            nn.SiLU(inplace=True),
            ResBlock(c1),

            # 32x32 -> 16x16
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_num_groups(c2), c2),
            nn.SiLU(inplace=True),
            ResBlock(c2),

            # 16x16 -> 8x8
            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_num_groups(c3), c3),
            nn.SiLU(inplace=True),
            ResBlock(c3),

            # 8x8 -> 4x4
            nn.Conv2d(c3, c4, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_num_groups(c4), c4),
            nn.SiLU(inplace=True),
            ResBlock(c4),

            # 4x4 -> 2x2
            nn.Conv2d(c4, c4, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_num_groups(c4), c4),
            nn.SiLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(c4),
            nn.Linear(c4, msg_len)
        )

    def forward(self, x):
        bands = torch.cat(haar_dwt(x), dim=1)
        feat = self.net(bands).flatten(1)
        return self.head(feat)


# ─────────────────────────────────────────────────────────────
#  Loss
# ─────────────────────────────────────────────────────────────

def compute_losses(original, encoded, logits, target,
                   lam_bit=1.0, lam_mean=0.0, lam_var=0.0,
                   lam_amp=0.5, amp_margin=0.2,
                   lam_margin=0.15, logit_margin=2.0):
    
    l_bit  = F.binary_cross_entropy_with_logits(logits, target)

    with torch.no_grad():
        bit_acc = ((logits > 0).float() == target).float().mean().item()

    # 因为 Encoder 已经强制 Normalization 了，这两项自然为 0，不再参与梯度
    l_mean = encoded.mean() ** 2 
    l_var  = (encoded.var() - 1.0) ** 2
    
    # 【修改 4】缩小 margin，要求残差更隐蔽
    l_amp  = F.relu((encoded - original).abs() - amp_margin).mean()
    
    target_sign = target.mul(2.0).sub(1.0)
    l_margin = F.relu(logit_margin - logits * target_sign).mean()

    total = (lam_bit * l_bit + lam_mean * l_mean + lam_var * l_var
             + lam_amp * l_amp + lam_margin * l_margin)

    return dict(total=total, l_bit=l_bit, l_mean=l_mean,
                l_var=l_var, l_amp=l_amp, l_margin=l_margin, bit_acc=bit_acc)


# def compute_losses(original, encoded, logits, target,
#                    lam_bit=1.0, lam_mean=0.0, lam_var=0.0,
#                    lam_amp=0.5, amp_margin=0.2,
#                    lam_margin=0.15, logit_margin=2.0):
    
#     l_bit  = F.binary_cross_entropy_with_logits(logits, target)

#     with torch.no_grad():
#         bit_acc = ((logits > 0).float() == target).float().mean().item()

#     # 因为 Encoder 已经强制 Normalization 了，这两项自然为 0，不再参与梯度
#     l_mean = encoded.mean() ** 2 
#     l_var  = (encoded.var() - 1.0) ** 2
    
#     # 【修改 4】缩小 margin，要求残差更隐蔽
#     l_amp  = F.relu((encoded - original).abs() - amp_margin).mean()
    
#     target_sign = target.mul(2.0).sub(1.0)
#     l_margin = F.relu(logit_margin - logits * target_sign).mean()

#     total = (lam_bit * l_bit + lam_mean * l_mean + lam_var * l_var
#              + lam_margin * l_margin)

#     return dict(total=total, l_bit=l_bit, l_mean=l_mean,
#                 l_var=l_var, l_amp=l_amp, l_margin=l_margin, bit_acc=bit_acc)


# ─────────────────────────────────────────────────────────────
#  训练
# ─────────────────────────────────────────────────────────────

def train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    BATCH_SIZE    = 16
    EPOCHS        = 150000
    LR            = 6e-4
    MSG_LEN       = 256
    NOISE_CH      = 16
    SPATIAL       = 64
    STEPS         = 4

    LAM_BIT    = 1.0
    LAM_MEAN   = 0.0  # 已弃用
    LAM_VAR    = 0.0  # 已弃用
    LAM_AMP_WARMUP = 0.08
    LAM_AMP    = 0.45
    AMP_MARGIN = 0.24

    WARMUP_EPOCHS = 1500
    ATK_RAMP_EPOCHS = 8000
    AMP_RAMP_EPOCHS = 4000
    ATK_STD_MAX   = 0.50

    SAVE_DIR  = os.path.join(base_dir, "weights_strong_v5.1_nowhitelamp")
    LOG_EVERY = 100

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)

    encoder = Encoder(NOISE_CH, MSG_LEN, SPATIAL).to(device)
    attack  = AttackLayer(
        threshold=0.5,
        std_min=0.0,
        std_max=ATK_STD_MAX,
        quant_max=0.14,
        blur_max=0.75,
        resize_min=0.45,
        gain_max=0.18,
        bias_max=0.12,
        freq_drop_max=0.65,
        cutout_max=0.35,
        channel_drop_max=0.22,
        coarse_noise_max=0.25,
        max_shift=5,
    )
    decoder = Decoder(NOISE_CH, MSG_LEN).to(device)

    params    = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(params, lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_acc = 0.0

    print(f"Device     : {device}")
    print(f"Parameters : {sum(p.numel() for p in params):,} "
          f"(Enc {sum(p.numel() for p in encoder.parameters()):,} / "
          f"Dec {sum(p.numel() for p in decoder.parameters()):,})")
    print(f"Warmup     : {WARMUP_EPOCHS} epochs (no attack)")
    print(f"Attack     : combo latent proxy, ramp {ATK_RAMP_EPOCHS} epochs, std max {ATK_STD_MAX}")
    print(f"Save dir   : {SAVE_DIR}/")
    print("=" * 76)
    print(f"{'Ep':>6} {'Loss':>7} {'L_bit':>6} {'L_amp':>6} {'Acc%':>7} {'α':>6} {'std':>5} {'lvl':>5}")
    print("=" * 76)

    for epoch in range(EPOCHS):
        encoder.train()
        decoder.train()

        if epoch < WARMUP_EPOCHS:
            attack.attack_std = 0.0
            attack.attack_level = 0.0
        else:
            attack.set_strength(epoch - WARMUP_EPOCHS, ATK_RAMP_EPOCHS)

        if epoch < WARMUP_EPOCHS:
            lam_amp = LAM_AMP_WARMUP
        else:
            t_amp = min((epoch - WARMUP_EPOCHS) / max(AMP_RAMP_EPOCHS, 1), 1.0)
            lam_amp = LAM_AMP_WARMUP + t_amp * (LAM_AMP - LAM_AMP_WARMUP)

        acc_sum = loss_sum = 0.0
        last_L  = None
        optimizer.zero_grad()

        for _ in range(STEPS):
            noise  = torch.randn(BATCH_SIZE, NOISE_CH, SPATIAL, SPATIAL).to(device)
            target = torch.randint(0, 2, (BATCH_SIZE, MSG_LEN)).float().to(device)

            encoded  = encoder(noise, target)
            attacked = attack(encoded)
            logits   = decoder(attacked)

            L = compute_losses(noise, encoded, logits, target,
                               LAM_BIT, LAM_MEAN, LAM_VAR, lam_amp, AMP_MARGIN)
            (L["total"] / STEPS).backward()
            loss_sum += L["total"].item() / STEPS
            acc_sum  += L["bit_acc"]      / STEPS
            last_L    = L

        nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optimizer.step()
        scheduler.step()

        is_best = ""
        if epoch >= WARMUP_EPOCHS and acc_sum > best_acc:
            best_acc = acc_sum
            torch.save(encoder.state_dict(), f"{SAVE_DIR}/encoder_best.pth")
            torch.save(decoder.state_dict(), f"{SAVE_DIR}/decoder_best.pth")
            is_best = " ★"

        if (epoch + 1) % LOG_EVERY == 0 or is_best:
            a = torch.clamp(encoder.alpha, 0.01, 0.35).item()
            print(f"{epoch+1:>6} {loss_sum:>7.4f} "
                  f"{last_L['l_bit'].item():>6.4f} "
                  f"{last_L['l_amp'].item():>6.4f} "
                  f"{acc_sum*100:>6.2f}% "
                  f"{a:>6.4f} {attack.attack_std:>5.3f} {attack.attack_level:>5.2f}"
                  f"{is_best}")

    torch.save(encoder.state_dict(), f"{SAVE_DIR}/encoder_final.pth")
    torch.save(decoder.state_dict(), f"{SAVE_DIR}/decoder_final.pth")
    print("=" * 76)
    print(f"训练完成。最佳 bit accuracy : {best_acc*100:.2f}%")
    print(f"权重保存在 '{SAVE_DIR}/'")


def sanity_check():
    print("Sanity check …")
    B, C, H = 2, 16, 64

    x = torch.randn(B, C, H, H)
    LL, HL, LH, HH = haar_dwt(x)
    y = haar_idwt(LL, HL, LH, HH)
    dwt_err = (x - y).abs().max().item()
    print(f"  haar err : {dwt_err:.2e}  {'✓' if dwt_err < 1e-6 else '✗'}")
    assert dwt_err < 1e-6

    enc = Encoder(C, 256, H)
    atk = AttackLayer(std_min=0.0, std_max=0.50)
    dec = Decoder(C, 256)
    params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
    print(f"  params   : {params:,} (Enc {sum(p.numel() for p in enc.parameters()):,} / "
          f"Dec {sum(p.numel() for p in dec.parameters()):,})")

    noise  = torch.randn(B, C, H, H)
    msg    = torch.randint(0, 2, (B, 256)).float()

    encoded = enc(noise, msg)
    encoded.retain_grad()
    atk.set_strength(7000, 8000)
    attacked = atk(encoded)
    (attacked * encoded.detach()).mean().backward()
    grad_norm = encoded.grad.abs().mean().item()

    encoded2 = enc(noise, msg)
    attacked2 = atk(encoded2)
    logits = dec(attacked2)
    L = compute_losses(noise, encoded2, logits, msg)

    print(f"  encoded  : {encoded2.shape}  mean={encoded2.mean():.4f}  var={encoded2.var():.4f}")
    print(f"  logits   : {logits.shape}")
    print(f"  atk grad : {grad_norm:.2e}  {'✓' if grad_norm > 1e-9 else '✗ 断流!'}")
    print(f"  L_bit    : {L['l_bit'].item():.4f}  L_amp : {L['l_amp'].item():.4f}")
    print(f"  Acc      : {L['bit_acc']*100:.1f}%")
    assert grad_norm > 1e-9
    assert abs(encoded2.mean().item()) < 1e-4, "均值未被强制归零！"
    assert abs(encoded2.var(unbiased=False).item() - 1.0) < 1e-2, "方差未被强制归一！"
    print("Sanity check passed ✓\n")


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        sanity_check()
    else:
        sanity_check()
        train()