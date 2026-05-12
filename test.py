import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

# 强制使用无界面后端
import matplotlib
matplotlib.use('Agg') 

from train import Encoder, Decoder, AttackLayer

def load_model(weights_path, device):
    NOISE_CH, MSG_LEN, SPATIAL = 16, 256, 64
    encoder = Encoder(NOISE_CH, MSG_LEN, SPATIAL).to(device)
    decoder = Decoder(NOISE_CH, MSG_LEN).to(device)
    
    enc_path = os.path.join(weights_path, "encoder_best.pth")
    dec_path = os.path.join(weights_path, "decoder_best.pth")
    
    if os.path.exists(enc_path) and os.path.exists(dec_path):
        encoder.load_state_dict(torch.load(enc_path, map_location=device))
        decoder.load_state_dict(torch.load(dec_path, map_location=device))
        print(f"成功加载最佳权重: {weights_path}")
    else:
        print("未找到权重文件！")
    
    encoder.eval()
    decoder.eval()
    return encoder, decoder

def test_robustness(encoder, decoder, device):
    print("\n--- 鲁棒性测试 (Robustness Sweep) ---")
    noise_levels = np.linspace(0.0, 1.0, 11)
    results = []
    BATCH_SIZE, NOISE_CH, SPATIAL, MSG_LEN = 64, 16, 64, 256

    with torch.no_grad():
        for std in noise_levels:
            noise = torch.randn(BATCH_SIZE, NOISE_CH, SPATIAL, SPATIAL).to(device)
            msg = torch.randint(0, 2, (BATCH_SIZE, MSG_LEN)).float().to(device)
            encoded = encoder(noise, msg)
            # 模拟攻击
            attacked = encoded + torch.randn_like(encoded) * std
            logits = decoder(attacked)
            acc = ((logits > 0).float() == msg).float().mean().item()
            results.append(acc)
            print(f"Noise Std: {std:.2f} | Bit Accuracy: {acc*100:>6.2f}%")

    # 绘图并保存
    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, results, marker='o', color='b')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guess')
    plt.xlabel("Attack Noise Std Deviation")
    plt.ylabel("Bit Accuracy")
    plt.title("Robustness to Gaussian Noise")
    plt.grid(True)
    plt.legend()
    plt.savefig("robustness_curve.png") # 保存图片
    print("鲁棒性曲线已保存至: robustness_curve.png")
    return noise_levels, results

def visualize_stats(encoder, device):
    print("\n--- 统计分布检查 ---")
    noise = torch.randn(1, 16, 64, 64).to(device)
    msg = torch.randint(0, 2, (1, 256)).float().to(device)
    
    with torch.no_grad():
        encoded = encoder(noise, msg)
    
    orig_np = noise.cpu().numpy().flatten()
    enc_np = encoded.cpu().numpy().flatten()
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(orig_np, bins=50, alpha=0.5, label='Original')
    plt.hist(enc_np, bins=50, alpha=0.5, label='Encoded')
    plt.title("Distribution Histogram")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    diff = (encoded - noise)[0, 0].cpu().numpy()
    plt.imshow(diff, cmap='RdBu')
    plt.colorbar()
    plt.title("Modulation Residual (Channel 0)")
    
    plt.savefig("distribution_stats.png") # 保存图片
    print("分布统计图已保存至: distribution_stats.png")

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WEIGHTS_DIR = "./weights"

    enc, dec = load_model(WEIGHTS_DIR, DEVICE)
    visualize_stats(enc, DEVICE)
    test_robustness(enc, dec, DEVICE)