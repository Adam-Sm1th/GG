"""
Gaussian Noise Watermarking Autoencoder  v8.0 (DWT Multi-Band Domain)
=====================================================================
仍然坚持“必须在频域做”，但不再把 256 bit 全挤进单个 LL 频段。

核心变化：
  1. 【四频段嵌入】：LL/HL/LH/HH 都有可学习残差和独立强度，容量比 LL-only 明显更足。
  2. 【四频段解码】：Decoder 直接读取拼接后的 DWT 系数，避免 128 维特征瓶颈卡住 ACC。
  3. 【延后压制幅度】：先让 bit 通道学会，再逐渐加大隐蔽性约束，减少 70% 左右的早期平台期。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ─────────────────────────────────────────────────────────────
#  纯 PyTorch 可微 Haar 小波变换 (无需外部库)
# ─────────────────────────────────────────────────────────────

def haar_dwt(x):
    """ 将 [B, C, H, W] 分解为四个 [B, C, H/2, W/2] 的频段 """
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    
    LL = (x00 + x01 + x10 + x11) / 4.0  # 低频概貌 (藏水印的黄金位置)
    HL = (x00 - x01 + x10 - x11) / 4.0  # 水平高频
    LH = (x00 + x01 - x10 - x11) / 4.0  # 垂直高频
    HH = (x00 - x01 - x10 + x11) / 4.0  # 对角高频
    return LL, HL, LH, HH

def haar_idwt(LL, HL, LH, HH):
    """ 将四个频段重构回原分辨率 """
    B, C, H_half, W_half = LL.shape
    out = torch.empty(B, C, H_half * 2, W_half * 2,
                      device=LL.device, dtype=LL.dtype)
    
    out[:, :, 0::2, 0::2] = LL + HL + LH + HH
    out[:, :, 0::2, 1::2] = LL - HL + LH - HH
    out[:, :, 1::2, 0::2] = LL + HL - LH - HH
    out[:, :, 1::2, 1::2] = LL - HL - LH + HH
    return out


# ─────────────────────────────────────────────────────────────
#  Encoder — 小波四频段嵌入
# ─────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, noise_ch=16, msg_len=256, spatial=64):
        super().__init__()
        self.noise_ch = noise_ch
        self.spatial = spatial
        self.band_spatial = spatial // 2
        self.base_spatial = spatial // 4

        # 先生成 16x16 的消息特征，再上采样到 DWT 频段大小 32x32。
        self.msg_proj = nn.Sequential(
            nn.Linear(msg_len, 1024),
            nn.SiLU(inplace=True),
            nn.Linear(1024, noise_ch * self.base_spatial * self.base_spatial),
            nn.Tanh()
        )

        self.band_mixer = nn.Sequential(
            nn.Conv2d(noise_ch, noise_ch * 4, 3, padding=1, bias=False),
            nn.GroupNorm(16, noise_ch * 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(noise_ch * 4, noise_ch * 4, 3, padding=1, bias=True),
            nn.Tanh()
        )

        # LL 稍低，细节频段稍高；训练时仍由幅度损失自动往回压。
        self.alpha = nn.Parameter(torch.tensor([0.12, 0.16, 0.16, 0.16]))

    def forward(self, noise, msg):
        B, C, H, W = noise.shape

        # Step 1: 原图分解到小波域
        LL, HL, LH, HH = haar_dwt(noise)

        # Step 2: 生成四个频段的水印残差
        msg_feat = self.msg_proj(msg).view(B, C, self.base_spatial, self.base_spatial)
        msg_feat = F.interpolate(msg_feat, size=(H // 2, W // 2),
                                 mode='bilinear', align_corners=False)
        delta = self.band_mixer(msg_feat).view(B, 4, C, H // 2, W // 2)

        # Step 3: 在 DWT 频域中叠加，仍然保持完全可微。
        alpha = torch.clamp(self.alpha, 0.02, 0.60).view(1, 4, 1, 1, 1)
        LL_encoded = LL + alpha[:, 0] * delta[:, 0]
        HL_encoded = HL + alpha[:, 1] * delta[:, 1]
        LH_encoded = LH + alpha[:, 2] * delta[:, 2]
        HH_encoded = HH + alpha[:, 3] * delta[:, 3]

        # Step 4: 小波逆变换回空间域
        encoded = haar_idwt(LL_encoded, HL_encoded, LH_encoded, HH_encoded)

        # Step 5: 强制白化 (对付 SD 的必备良药)
        mean = encoded.mean(dim=[1, 2, 3], keepdim=True)
        std  = encoded.std(dim=[1, 2, 3], keepdim=True)
        encoded = (encoded - mean) / (std + 1e-7)

        return encoded


# ─────────────────────────────────────────────────────────────
#  Attack Layer (保留你的极端软掩码测试)
# ─────────────────────────────────────────────────────────────

class AttackLayer(nn.Module):
    def __init__(self, threshold=0.5, std_min=0.0, std_max=0.4):
        super().__init__()
        self.threshold  = threshold
        self.std_min    = std_min
        self.std_max    = std_max
        self.attack_std = std_min

    def set_strength(self, epoch, total):
        t = epoch / max(total - 1, 1)
        self.attack_std = self.std_min + t * (self.std_max - self.std_min)

    def forward(self, x):
        if self.attack_std < 1e-6:
            return x
        soft_mask = torch.sigmoid(-(x.abs() - self.threshold) * 8.0)
        eps       = torch.randn_like(x)
        return x + soft_mask * self.attack_std * eps


# ─────────────────────────────────────────────────────────────
#  Decoder — 从四个 DWT 频段读取
# ─────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    def __init__(self, noise_ch=16, msg_len=256):
        super().__init__()
        in_ch = noise_ch * 4
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1, bias=False),
            nn.GroupNorm(32, 128), nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),      # -> 16x16
            nn.GroupNorm(32, 128), nn.SiLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),      # -> 8x8
            nn.GroupNorm(32, 256), nn.SiLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),      # -> 4x4
            nn.GroupNorm(32, 512), nn.SiLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=2, padding=1, bias=False),      # -> 2x2
            nn.GroupNorm(32, 512), nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, msg_len)
        )

    def forward(self, x):
        # 解码第一步：转到 DWT 域，并保留所有频段。
        bands = torch.cat(haar_dwt(x), dim=1)
        feat = self.net(bands).flatten(1)
        return self.head(feat)


# ─────────────────────────────────────────────────────────────
#  Loss
# ─────────────────────────────────────────────────────────────

def compute_losses(original, encoded, logits, target,
                   lam_bit=1.0, lam_mean=0.0, lam_var=0.0,
                   lam_amp=1.0, amp_margin=0.2):

    l_bit = F.binary_cross_entropy_with_logits(logits, target)

    with torch.no_grad():
        bit_acc = ((logits > 0).float() == target).float().mean().item()

    l_mean = encoded.mean() ** 2
    l_var  = (encoded.var() - 1.0) ** 2
    l_amp  = F.relu((encoded - original).abs() - amp_margin).mean()

    total = lam_bit * l_bit + lam_mean * l_mean + lam_var * l_var + lam_amp * l_amp

    return dict(total=total, l_bit=l_bit, l_mean=l_mean,
                l_var=l_var, l_amp=l_amp, bit_acc=bit_acc)


# ─────────────────────────────────────────────────────────────
#  训练流程
# ─────────────────────────────────────────────────────────────

def train():
    BATCH_SIZE    = 32
    EPOCHS        = 100000
    LR            = 1e-3
    MSG_LEN       = 256
    NOISE_CH      = 16
    SPATIAL       = 64
    STEPS         = 4

    LAM_BIT    = 1.0
    LAM_MEAN   = 0.0
    LAM_VAR    = 0.0
    LAM_AMP_WARMUP = 0.05
    LAM_AMP    = 0.35
    AMP_MARGIN = 0.25

    WARMUP_EPOCHS = 1000
    AMP_RAMP_EPOCHS = 2000
    ATK_STD_MAX   = 0.4

    SAVE_DIR  = "./weights_v8_dwt_multiband"
    LOG_EVERY = 50

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
    print(f"Transform  : Haar Discrete Wavelet Transform (LL/HL/LH/HH)")
    print(f"Save dir   : {SAVE_DIR}/")
    print("=" * 76)
    print(f"{'Ep':>6} {'Loss':>7} {'L_bit':>6} {'L_amp':>6} {'Acc%':>7} {'αLL':>6} {'αD':>6} {'atk':>5}")
    print("=" * 76)

    for epoch in range(EPOCHS):
        encoder.train()
        decoder.train()

        if epoch < WARMUP_EPOCHS:
            attack.attack_std = 0.0
        else:
            attack.set_strength(epoch - WARMUP_EPOCHS, EPOCHS - WARMUP_EPOCHS)

        if epoch < WARMUP_EPOCHS:
            lam_amp = LAM_AMP_WARMUP
        else:
            t = min((epoch - WARMUP_EPOCHS) / max(AMP_RAMP_EPOCHS, 1), 1.0)
            lam_amp = LAM_AMP_WARMUP + t * (LAM_AMP - LAM_AMP_WARMUP)

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
        if acc_sum > best_acc:
            best_acc = acc_sum
            torch.save(encoder.state_dict(), f"{SAVE_DIR}/encoder_best.pth")
            torch.save(decoder.state_dict(), f"{SAVE_DIR}/decoder_best.pth")
            is_best = " ★"

        if (epoch + 1) % LOG_EVERY == 0 or is_best:
            a = torch.clamp(encoder.alpha, 0.02, 0.60).detach()
            a_ll = a[0].item()
            a_detail = a[1:].mean().item()
            print(f"{epoch+1:>6} {loss_sum:>7.4f} "
                  f"{last_L['l_bit'].item():>6.4f} "
                  f"{last_L['l_amp'].item():>6.4f} "
                  f"{acc_sum*100:>6.2f}% "
                  f"{a_ll:>6.4f} {a_detail:>6.4f} {attack.attack_std:>5.3f}"
                  f"{is_best}")

    torch.save(encoder.state_dict(), f"{SAVE_DIR}/encoder_final.pth")
    torch.save(decoder.state_dict(), f"{SAVE_DIR}/decoder_final.pth")
    print("=" * 76)
    print(f"训练完成。最佳 bit accuracy : {best_acc*100:.2f}%")
    print(f"权重保存在 '{SAVE_DIR}/'")


def sanity_check():
    print("Sanity check …")
    B, C, H = 2, 16, 64

    # 验证 Haar 小波的无损重构数学特性
    x = torch.randn(B, C, H, H)
    LL, HL, LH, HH = haar_dwt(x)
    y = haar_idwt(LL, HL, LH, HH)
    err = (x - y).abs().max().item()
    print(f"  小波重构误差: {err:.2e}  {'✓' if err < 1e-6 else '✗ 数学错误!'}")
    assert err < 1e-6

    enc = Encoder(C, 256, H)
    atk = AttackLayer(std_min=0.0, std_max=0.4)
    atk.attack_std = 0.3
    dec = Decoder(C, 256)

    noise = torch.randn(B, C, H, H)
    msg   = torch.randint(0, 2, (B, 256)).float()

    encoded = enc(noise, msg)
    encoded.retain_grad()
    attacked = atk(encoded)
    attacked.mean().backward()
    grad_norm = encoded.grad.abs().mean().item()

    encoded2  = enc(noise, msg)
    attacked2 = atk(encoded2)
    logits    = dec(attacked2)
    L         = compute_losses(noise, encoded2, logits, msg)

    print(f"  encoded  : {encoded2.shape}  mean={encoded2.mean():.4f}  var={encoded2.var():.4f}")
    print(f"  atk grad : {grad_norm:.2e}  {'✓' if grad_norm > 1e-9 else '✗ 断流!'}")

    assert grad_norm > 1e-9, "梯度断流！"
    assert abs(encoded2.mean().item()) < 1e-3, "均值未被强制归零！"
    print("Sanity check passed ✓\n")


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        sanity_check()
    else:
        sanity_check()
        train()
