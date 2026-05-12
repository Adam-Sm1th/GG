import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
import random

# 导入你自定义的反演 Pipeline 和 水印模型
from src.inversion.inverse_diffusion3 import InversionDiffusion3Pipeline
from train import Encoder, Decoder, compute_losses

def main():
    # ==========================================
    # 1. 基础配置
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16  # SD3.5 必须用 FP16 跑以节省显存
    
    # 水印与生成参数
    NOISE_CH = 16
    MSG_LEN = 256
    SPATIAL = 64
    num_inference_steps = 28
    guidance_scale = 4.5
    
    # 微调超参数 (使用较小的学习率)
    LR = 2e-4
    EPOCHS = 5000  # 真实反演很慢，500步微调通常足够见效
    GRAD_ACCUM_STEPS = 4  # 梯度累加，弥补 Batch Size = 1 的问题
    
    weights_dir = "./weights"
    finetune_dir = "./weights_finetuned"
    os.makedirs(finetune_dir, exist_ok=True)

    # ==========================================
    # 2. 加载 SD3.5 与 水印模型
    # ==========================================
    print("⏳ 正在加载 InversionDiffusion3Pipeline (这部分不参与梯度更新)...")
    pipe = InversionDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", 
        torch_dtype=weight_dtype
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.eval()
    pipe.transformer.eval()

    print("⏳ 正在加载预训练的 Encoder 和 Decoder...")
    encoder = Encoder(NOISE_CH, MSG_LEN, SPATIAL).to(device).to(weight_dtype)
    decoder = Decoder(NOISE_CH, MSG_LEN).to(device).to(weight_dtype)

    # 加载你在基础阶段训练好的权重
    encoder.load_state_dict(torch.load(os.path.join(weights_dir, "encoder_best.pth"), map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(weights_dir, "decoder_best.pth"), map_location=device))
    
    encoder.train()
    decoder.train()

    # 优化器配置
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(params, lr=5e-5, eps=1e-4) # 增大 Adam 的 epsilon

    # 准备一组多样的提示词，让模型适应不同的 CFG 扭曲路径
    prompt_pool = [
        "A beautiful futuristic city at sunset, cyberpunk style",
        "A cute fluffy cat sitting on a wooden table, 4k",
        "Portrait of a warrior princess, fantasy concept art",
        "Macro photography of a glowing neon insect",
        "A serene landscape with mountains and a calm lake"
    ]

    # ==========================================
    # 3. 开始真实对抗微调循环
    # ==========================================
    print(f"\n🚀 开始真实反演微调 (STE残差注入法) | 目标 Epochs: {EPOCHS}...")
    best_acc = 0.0
    optimizer.zero_grad()

    with tqdm(total=EPOCHS, desc="Fine-tuning") as pbar:
        for epoch in range(1, EPOCHS + 1):
            
            # 1. 生成初始数据 (使用 weight_dtype 保证精度一致)
            msg = torch.randint(0, 2, (1, MSG_LEN)).to(device).to(weight_dtype)
            init_noise = torch.randn(1, NOISE_CH, SPATIAL, SPATIAL).to(device).to(weight_dtype)
            prompt = random.choice(prompt_pool)

            # 2. 【有梯度】Encoder 嵌入水印
            encoded_latents = encoder(init_noise, msg)

            # 3. 【无梯度】运行真实 SD 生成与反演
            with torch.no_grad():
                # --- 生成阶段 ---
                gen_output = pipe(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=512, width=512,
                    latents=encoded_latents.clone().detach(),
                    output_type="latent" # 为了加速微调，直接输出 Latents (如果需要模拟 PNG 压缩，可改为 image 并过 VAE)
                )
                generated_latents = gen_output.images
                
                # --- 反演阶段 ---
                # 注意：pipe内部操作可能会改变精度，拿到结果后强制转回 weight_dtype
                reversed_latents = pipe.naive_forward_diffusion(
                    latents=generated_latents,
                    num_inference_steps=num_inference_steps
                ).to(weight_dtype)

            # ========================================================
            # 💡 核心黑科技：Straight-Through Estimator (STE)
            # 这里的 attacked_latents 的【值】完全等于真实的 reversed_latents。
            # 但在调用 .backward() 时，梯度会被 detach() 拦截掉真实的 SD 链路，
            # 直接顺着前面的加号，原封不动地传给 encoded_latents，从而更新 Encoder！
            # ========================================================
            attacked_latents = encoded_latents + (reversed_latents - encoded_latents).detach()

            # 4. 【有梯度】Decoder 解析与 Loss 计算
            logits = decoder(attacked_latents)
            
            # 使用原有的 Loss 计算逻辑（这里为了简化，我们手写核心约束）
            l_bit = F.binary_cross_entropy_with_logits(logits, msg)
            
            # 惩罚 Encoder 改动过大 (AMP Margin)
            amp_margin = 0.15
            l_amp = F.relu((encoded_latents - init_noise).abs() - amp_margin).mean()
            
            # 总 Loss (加大对提取正确率的权重)
            loss = 2.0 * l_bit + 1.0 * l_amp
            
            # 梯度累加，防止 batch size 为 1 导致梯度震荡
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            # 计算准确率 (仅用于记录)
            with torch.no_grad():
                extracted_bits = (logits > 0).float()
                acc = (extracted_bits == msg).float().mean().item()

            if epoch % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # 记录与保存
            pbar.set_postfix({"Loss": f"{loss.item()*GRAD_ACCUM_STEPS:.4f}", "Acc": f"{acc*100:.1f}%"})
            pbar.update(1)

            # 定期保存最好的权重
            if acc > best_acc and epoch > 10:  # 稍微稳定后再开始保存
                best_acc = acc
                torch.save(encoder.state_dict(), os.path.join(finetune_dir, "encoder_ft_best.pth"))
                torch.save(decoder.state_dict(), os.path.join(finetune_dir, "decoder_ft_best.pth"))
                
            # 每 50 步打印一次详情
            if epoch % 50 == 0:
                tqdm.write(f"Epoch {epoch} | L_bit: {l_bit.item():.4f} | L_amp: {l_amp.item():.4f} | Acc: {acc*100:.2f}%")

    # 保存最终状态
    torch.save(encoder.state_dict(), os.path.join(finetune_dir, "encoder_ft_final.pth"))
    torch.save(decoder.state_dict(), os.path.join(finetune_dir, "decoder_ft_final.pth"))
    print("\n✅ 微调完成！权重已保存至:", finetune_dir)
    print(f"🌟 微调期间观察到的最高单图准确率: {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()