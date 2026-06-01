"""
Phase 2 Fine-tuning for Watermarking Autoencoder (Real SD 3.5 Pipeline)
========================================================================
结合了 Phase 1 的自编码器和 Phase 2 的真实推理管线。
使用 STE (Straight-Through Estimator) 绕过不可导的 DiT 生成、物理攻击和反演过程，
使 Encoder 和 Decoder 能够在真实的 SD3.5 潜空间传输分布上进行端到端的共同微调。
"""

import os
import json
import time
import argparse
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from datasets import load_dataset
from torchvision import transforms

# 导入一阶段模型和相关工具
from train import Encoder, Decoder, compute_losses
from src import utils
from src.inversion.inverse_diffusion3 import InversionDiffusion3Pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 Fine-tuning with Real SD3.5 Pipeline")
    
    # --- 基础与路径配置 ---
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--pretrained_dir", type=str, default="./weights_strong_v5", help="一阶段预训练权重路径")
    parser.add_argument("--save_dir", type=str, default="./weights_finetuned", help="二阶段微调权重保存路径")
    parser.add_argument("--dataset_name", type=str, default="Gustavosta/Stable-Diffusion-Prompts")
    parser.add_argument("--max_train_samples", type=int, default=2000, help="限制训练集大小，设为0则使用全量数据（7万多条极其耗时）")
    
    # --- 训练超参 ---
    parser.add_argument("--batch_size", type=int, default=1, help="受限于扩散模型推理显存，通常设为 1 或 2")
    parser.add_argument("--accumulate_grad_batches", type=int, default=8, help="梯度累加步数，弥补小 Batch Size")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5, help="二阶段微调学习率应较小，防止在复杂流形中震荡")
    parser.add_argument("--freeze_encoder", action="store_true", help="是否冻结 Encoder 仅微调 Decoder（不加此参数则为共同微调）")
    
    # --- 水印与 SD 维度 ---
    parser.add_argument("--noise_ch", type=int, default=16)
    parser.add_argument("--msg_len", type=int, default=256)
    parser.add_argument("--spatial", type=int, default=64)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    
    # --- 推理与反演参数 ---
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--inverse_solver", type=str, default="fixed_point")
    parser.add_argument("--inverse_fixpoint_iters", type=int, default=10)
    parser.add_argument("--disable_t5", action="store_true", help="跳过 T5 文本编码器以节省显存")
    
    # --- 物理攻击分布参数 (utils.image_distortion 所需) ---
    parser.add_argument('--jpeg_ratio', type=int, default=50)
    parser.add_argument('--random_crop_ratio', type=float, default=None)
    parser.add_argument('--random_drop_ratio', type=float, default=None)
    parser.add_argument('--gaussian_blur_r', type=int, default=None)
    parser.add_argument('--median_blur_k', type=int, default=None)
    parser.add_argument('--resize_ratio', type=float, default=None)
    parser.add_argument('--gaussian_std', type=float, default=None)
    parser.add_argument('--sp_prob', type=float, default=None)
    parser.add_argument('--brightness_factor', type=float, default=None)

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16  # SD3.5 推理用 FP16 降低显存压力
    ae_dtype = torch.float32      # 自编码器保持 FP32 以稳定梯度
    
    os.makedirs(args.save_dir, exist_ok=True)

    # ==========================================
    # 1. 加载管道与数据集
    # ==========================================
    print("⏳ 正在加载 SD3.5 Pipeline...")
    pipe_kwargs = dict(torch_dtype=weight_dtype)
    if args.disable_t5:
        pipe_kwargs.update(text_encoder_3=None, tokenizer_3=None)
        print("✅ 已禁用 T5 Text Encoder 以节省显存")
    
    pipe = InversionDiffusion3Pipeline.from_pretrained(args.model_id, **pipe_kwargs).to(device)
    pipe.set_progress_bar_config(disable=True)
    
    print(f"⏳ 正在加载数据集: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")
    
    # 数据集截断逻辑 (避免跑 400 个小时)
    if args.max_train_samples is not None and args.max_train_samples > 0:
        actual_samples = min(args.max_train_samples, len(dataset))
        dataset = dataset.select(range(actual_samples))
        print(f"✅ 已截断数据集至前 {actual_samples} 条 (加速微调验证)。如需全量训练，请设置 --max_train_samples 0")
    
    # 图像预处理 (将 [0, 255] PIL 转换为 [-1, 1] 的 Tensor，用于反演输入)
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) 
    ])

    # ==========================================
    # 2. 加载一阶段自编码器并配置优化器
    # ==========================================
    print("⏳ 正在加载并初始化水印自编码器...")
    encoder = Encoder(args.noise_ch, args.msg_len, args.spatial).to(device).to(ae_dtype)
    decoder = Decoder(args.noise_ch, args.msg_len).to(device).to(ae_dtype)
    
    # 请确保 pretrained_dir 目录下有这两份权重
    encoder.load_state_dict(torch.load(os.path.join(args.pretrained_dir, "encoder_final.pth"), map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(args.pretrained_dir, "decoder_final.pth"), map_location=device))
    
    if args.freeze_encoder:
        print("❄️ Encoder 已冻结。仅微调 Decoder。")
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        params_to_opt = list(decoder.parameters())
    else:
        print("🔥 共同微调模式已开启：Encoder 与 Decoder 将通过 STE 同步更新。")
        encoder.train()
        params_to_opt = list(encoder.parameters()) + list(decoder.parameters())
    
    decoder.train()
    optimizer = optim.AdamW(params_to_opt, lr=args.lr, weight_decay=1e-4)
    
    # 二阶段 Loss 权重配置（由于经过了真实扩散，不可见性约束 L_amp 可以适度放宽）
    LAM_BIT = 1.0
    LAM_AMP = 0.2  
    AMP_MARGIN = 0.24
    
    # ==========================================
    # 3. 二阶段微调核心循环
    # ==========================================
    print("\n🚀 开始 Phase 2 真实链路微调...")
    global_step = 0
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        optimizer.zero_grad()
        
        # 随机打乱数据集索引
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        pbar = tqdm(total=len(indices) // args.batch_size, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i in range(0, len(indices), args.batch_size):
            batch_indices = indices[i:i + args.batch_size]
            prompts = [dataset[idx].get('Prompt', dataset[idx].get('text', '')) for idx in batch_indices]
            current_bs = len(prompts)
            
            # 采样初始噪声与随机消息
            init_noise = torch.randn(current_bs, args.noise_ch, args.spatial, args.spatial).to(device, dtype=ae_dtype)
            target_msg = torch.randint(0, 2, (current_bs, args.msg_len)).to(device, dtype=ae_dtype)
            
            # --- [步骤 A：前向编码] ---
            if args.freeze_encoder:
                with torch.no_grad():
                    z_w = encoder(init_noise, target_msg)
            else:
                z_w = encoder(init_noise, target_msg)
            
            # --- [步骤 B：真实 SD3.5 链路 (梯度截断区)] ---
            with torch.no_grad():
                z_w_fp16 = z_w.to(weight_dtype)
                inverted_latents_list = []
                
                for b_idx in range(current_bs):
                    single_prompt = prompts[b_idx]
                    single_z_w = z_w_fp16[b_idx:b_idx+1]
                    atk_seed = random.randint(0, 100000)
                    
                    # B.1 真实前向生成
                    gen_output = pipe(
                        prompt=single_prompt,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        height=args.height,
                        width=args.width,
                        latents=single_z_w
                    )
                    image = gen_output.images[0]
                    
                    # B.2 真实物理攻击 (在 args 中指定了才会触发对应攻击)
                    noised_image = utils.image_distortion(image, atk_seed, args)
                    
                    # B.3 真实反演提取潜变量
                    img_tensor = img_transform(noised_image).unsqueeze(0).to(device, dtype=weight_dtype)
                    latents = pipe.get_image_latents(img_tensor, sample=False)
                    
                    z_w_inv = pipe.naive_forward_diffusion(
                        latents=latents,
                        prompt="",  # 反演使用空 prompt 保持一致性
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=1.0,
                        inverse_solver=args.inverse_solver,
                        inverse_fixpoint_iters=args.inverse_fixpoint_iters
                    )
                    inverted_latents_list.append(z_w_inv)
                
                z_w_inv_batch = torch.cat(inverted_latents_list, dim=0).to(ae_dtype)
            
            # --- [步骤 C：STE (Straight-Through Estimator)] ---
            # 共同微调的核心：使得不可导链路的输出可以往回传梯度给 Encoder
            if not args.freeze_encoder:
                z_w_diff = z_w_inv_batch.detach() + z_w - z_w.detach()
            else:
                z_w_diff = z_w_inv_batch.detach()
            
            # --- [步骤 D：解码与 Loss 计算] ---
            logits = decoder(z_w_diff)
            
            loss_dict = compute_losses(
                original=init_noise, 
                encoded=z_w, 
                logits=logits, 
                target=target_msg,
                lam_bit=LAM_BIT, 
                lam_amp=LAM_AMP, 
                amp_margin=AMP_MARGIN
            )
            
            # 损失缩放（用于梯度累加）
            loss = loss_dict["total"] / args.accumulate_grad_batches
            loss.backward()
            
            epoch_loss += loss_dict["total"].item()
            epoch_acc += loss_dict["bit_acc"]
            
            # 梯度累加更新步
            if (global_step + 1) % args.accumulate_grad_batches == 0:
                torch.nn.utils.clip_grad_norm_(params_to_opt, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            global_step += 1
            
            # --- 进度条显示更新 (修复了 tqdm 重复打印的问题) ---
            pbar.set_postfix({
                "Loss": f"{loss_dict['total'].item():.4f}", 
                "BitAcc": f"{loss_dict['bit_acc']*100:.2f}%"
            }, refresh=False)  # 关键修复：禁止在这里刷新屏幕
            pbar.update(1)     # 由 update 统一刷新，保持只有一行
            
        pbar.close()
        
        # --- Epoch 总结与保存 ---
        avg_acc = epoch_acc / (len(indices) / args.batch_size)
        print(f"Epoch {epoch+1} 结束 | 平均 Loss: {epoch_loss / (len(indices)/args.batch_size):.4f} | 平均准确率: {avg_acc*100:.2f}%")
        
        if avg_acc > best_acc:
            best_acc = avg_acc
            if not args.freeze_encoder:
                torch.save(encoder.state_dict(), os.path.join(args.save_dir, "encoder_ft_best.pth"))
            torch.save(decoder.state_dict(), os.path.join(args.save_dir, "decoder_ft_best.pth"))
            print(f"🌟 已保存最佳权重！(当前最高准确率: {best_acc*100:.2f}%)")
            
        # 每个 Epoch 结束也保存一份最新权重
        if not args.freeze_encoder:
            torch.save(encoder.state_dict(), os.path.join(args.save_dir, "encoder_ft_latest.pth"))
        torch.save(decoder.state_dict(), os.path.join(args.save_dir, "decoder_ft_latest.pth"))

if __name__ == "__main__":
    main()