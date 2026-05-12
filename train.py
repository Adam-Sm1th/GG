"""
Gaussian Noise Watermarking Autoencoder  v4.1 (Stable Diffusion Safe Fix)
============================================
针对 Stable Diffusion 生成崩溃问题进行核心修复。

核心思路变化：
  1. 维持短梯度的空间调制架构。
  2. 【关键修复】Encoder 输出前进行强制白化（Whitening），保证严格的 N(0, 1) 统计特性。
  3. 【关键修复】大幅压制 alpha 的上限，防止引入过强的低频结构特征。
  4. 移除多余的 L_mean 和 L_var 损失约束（由架构直接保证）。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ─────────────────────────────────────────────────────────────
#  Encoder — 显式调制 + 强制正态化设计
# ─────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    noise  : [B, 16, 64, 64]
    msg    : [B, 256]  float {0,1}
    output : [B, 16, 64, 64]
    """
    def __init__(self, noise_ch=16, msg_len=256, spatial=64):
        super().__init__()
        self.noise_ch = noise_ch
        self.spatial  = spatial

        self.msg_embed = nn.Sequential(
            nn.Linear(msg_len, noise_ch * spatial * spatial // 16),
            nn.Tanh(),
        )

        self.modulator = nn.Sequential(
            nn.Conv2d(noise_ch * 2, noise_ch * 2, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(noise_ch * 2, noise_ch, 3, padding=1, bias=True),
            nn.Tanh(),
        )

        # JPEG + inverse 会明显吃掉弱高频信号，空间域需要稍微给足一点初始能量。
        self.alpha = nn.Parameter(torch.tensor(0.08))

        nn.init.normal_(self.modulator[0].weight, std=0.01)
        nn.init.normal_(self.modulator[2].weight, std=0.01)

    def forward(self, noise, msg):
        B, C, H, W = noise.shape

        msg_feat = self.msg_embed(msg)                                         
        msg_feat = msg_feat.view(B, self.noise_ch, H // 4, W // 4)
        msg_feat = F.interpolate(msg_feat, size=(H, W), mode='bilinear', align_corners=False)

        x = torch.cat([noise, msg_feat], dim=1)                               
        residual = self.modulator(x)                                           

        # 上限仍然保守，但比 0.15 更适合做 JPEG 鲁棒训练；最终幅度由 L_amp 压住。
        alpha = torch.clamp(self.alpha, 0.01, 0.25)
        encoded = noise + alpha * residual

        # 【修改 3】强制白化 (Forced Whitening)
        # 强制让每个 batch 的每个样本，整体符合完美的均值 0，方差 1
        # 彻底阻断任何让 SD 发散的统计学漂移
        mean = encoded.mean(dim=[1, 2, 3], keepdim=True)
        std = encoded.std(dim=[1, 2, 3], keepdim=True)
        encoded = (encoded - mean) / (std + 1e-7)

        return encoded


# ─────────────────────────────────────────────────────────────
#  Attack Layer — 重参数化，梯度连续
# ─────────────────────────────────────────────────────────────

class AttackLayer(nn.Module):
    """
    Latent-space proxy for: image JPEG -> VAE encode -> diffusion inverse.

    真实 JPEG 不是直接作用在 initial latent 上，所以这里只做可微/STE 近似：
    低通、重采样、量化、通道增益/偏置漂移，再叠加 masked Gaussian。
    """
    def __init__(self, threshold=0.5, std_min=0.0, std_max=0.35,
                 quant_max=0.08, blur_max=0.45,
                 resize_min=0.65, gain_max=0.12, bias_max=0.08):
        super().__init__()
        self.threshold  = threshold
        self.std_min    = std_min
        self.std_max    = std_max
        self.attack_std = std_min
        self.attack_level = 0.0
        self.quant_max = quant_max
        self.blur_max = blur_max
        self.resize_min = resize_min
        self.gain_max = gain_max
        self.bias_max = bias_max

    def set_strength(self, epoch, total):
        t = min(max(epoch / max(total, 1), 0.0), 1.0)
        self.attack_level = t
        self.attack_std = self.std_min + t * (self.std_max - self.std_min)

    @staticmethod
    def _ste_quantize(x, step):
        quantized = torch.round(x / step) * step
        return x + (quantized - x).detach()

    @staticmethod
    def _standardize(x):
        mean = x.mean(dim=[1, 2, 3], keepdim=True)
        std = x.std(dim=[1, 2, 3], keepdim=True)
        return (x - mean) / (std + 1e-6)

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

        if torch.rand((), device=y.device).item() < 0.85:
            kernel = 3 if torch.rand((), device=y.device).item() < 0.7 else 5
            blurred = F.avg_pool2d(y, kernel_size=kernel, stride=1, padding=kernel // 2)
            blend = self.blur_max * level * (0.35 + 0.65 * torch.rand((), device=y.device).item())
            y = y.lerp(blurred, blend)

        if torch.rand((), device=y.device).item() < 0.70:
            scale = self.resize_min + (1.0 - self.resize_min) * torch.rand((), device=y.device).item()
            new_h = max(8, int(H * scale))
            new_w = max(8, int(W * scale))
            y = F.interpolate(y, size=(new_h, new_w), mode="bilinear", align_corners=False)
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)

        q_step = self.quant_max * level * (0.5 + torch.rand((), device=y.device).item())
        if q_step > 1e-6:
            y = self._ste_quantize(y, q_step)

        soft_mask = torch.sigmoid(-(x.abs() - self.threshold) * 8.0)
        eps       = torch.randn_like(x)
        y = y + soft_mask * self.attack_std * eps

        # Inversion 后 latent 的整体统计会漂，但 SD 初始噪声仍应近似 N(0,1)。
        return self._standardize(y)


# ─────────────────────────────────────────────────────────────
#  Decoder — 轻量设计
# ─────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    def __init__(self, noise_ch=16, msg_len=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(noise_ch, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(512, msg_len)

    def forward(self, x):
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True)
        x = (x - mean) / (std + 1e-6)
        feat = self.net(x).flatten(1)
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


# ─────────────────────────────────────────────────────────────
#  训练
# ─────────────────────────────────────────────────────────────

def train():
    BATCH_SIZE    = 32
    EPOCHS        = 150000
    LR            = 1e-3
    MSG_LEN       = 256
    NOISE_CH      = 16
    SPATIAL       = 64
    STEPS         = 4

    LAM_BIT    = 1.0
    LAM_MEAN   = 0.0  # 已弃用
    LAM_VAR    = 0.0  # 已弃用
    LAM_AMP_WARMUP = 0.10
    LAM_AMP    = 0.75
    AMP_MARGIN = 0.22

    WARMUP_EPOCHS = 1000
    ATK_RAMP_EPOCHS = 5000
    AMP_RAMP_EPOCHS = 3000
    ATK_STD_MAX   = 0.35

    SAVE_DIR  = "./weights"
    LOG_EVERY = 100

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)

    encoder = Encoder(NOISE_CH, MSG_LEN, SPATIAL).to(device)
    attack  = AttackLayer(threshold=0.5, std_min=0.0, std_max=ATK_STD_MAX)
    decoder = Decoder(NOISE_CH, MSG_LEN).to(device)

    params    = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(params, lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_acc = 0.0

    print(f"Device     : {device}")
    print(f"Parameters : {sum(p.numel() for p in params):,}")
    print(f"Warmup     : {WARMUP_EPOCHS} epochs (no attack)")
    print(f"Save dir   : {SAVE_DIR}/")
    print("=" * 76)
    print(f"{'Ep':>6} {'Loss':>7} {'L_bit':>6} {'L_amp':>6} {'Acc%':>7} {'α':>6} {'atk':>5}")
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
            a = torch.clamp(encoder.alpha, 0.01, 0.25).item()
            print(f"{epoch+1:>6} {loss_sum:>7.4f} "
                  f"{last_L['l_bit'].item():>6.4f} "
                  f"{last_L['l_amp'].item():>6.4f} "
                  f"{acc_sum*100:>6.2f}% "
                  f"{a:>6.4f} {attack.attack_std:>5.3f}"
                  f"{is_best}")

    torch.save(encoder.state_dict(), f"{SAVE_DIR}/encoder_final.pth")
    torch.save(decoder.state_dict(), f"{SAVE_DIR}/decoder_final.pth")
    print("=" * 76)
    print(f"训练完成。最佳 bit accuracy : {best_acc*100:.2f}%")
    print(f"权重保存在 '{SAVE_DIR}/'")


def sanity_check():
    print("Sanity check …")
    B, C, H = 2, 16, 64

    enc = Encoder(C, 256, H)
    atk = AttackLayer(std_min=0.0, std_max=0.35)
    dec = Decoder(C, 256)

    noise  = torch.randn(B, C, H, H)
    msg    = torch.randint(0, 2, (B, 256)).float()

    encoded = enc(noise, msg)
    encoded.retain_grad()
    atk.attack_std = 0.3
    atk.attack_level = 0.8
    attacked = atk(encoded)
    (attacked[:, :, :16, :16] * encoded.detach()[:, :, :16, :16]).mean().backward()
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
    assert abs(encoded2.var().item() - 1.0) < 1e-2, "方差未被强制归一！"
    print("Sanity check passed ✓\n")


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        sanity_check()
    else:
        sanity_check()
        train()
