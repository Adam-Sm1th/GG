import os
import argparse
import torch
import lpips  # 需要 pip install lpips
from tqdm import tqdm
from torchvision import transforms
from datasets import load_dataset
import itertools

# 导入你原有的自定义模块
from src.inversion.inverse_diffusion3 import InversionDiffusion3Pipeline

def load_watermark_classes(watermark_impl):
    if watermark_impl == "spatial":
        from train import Encoder, Decoder
    elif watermark_impl == "freq":
        from train_freq import Encoder, Decoder
    else:
        raise ValueError(f"Unsupported watermark_impl: {watermark_impl}")
    return Encoder, Decoder

def canonicalize_model_id(model_id):
    lower_model_id = model_id.lower()
    if lower_model_id.startswith("stabilityai/stable-diffusion-3.5-"):
        return lower_model_id
    return model_id

def parse_args():
    parser = argparse.ArgumentParser(description="Measure Generation Diversity (Pairwise LPIPS)")
    
    # --- 核心测量参数 ---
    parser.add_argument("--num_prompts", type=int, default=10, help="测试的提示词数量")
    parser.add_argument("--images_per_prompt", type=int, default=5, help="每个提示词生成的图片数(用于计算两两差异)")
    
    # --- 模型与路径配置 (与你的原始代码保持一致) ---
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--weights_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights_strong_v5"))
    parser.add_argument("--watermark_impl", type=str, choices=["freq", "spatial"], default="spatial")
    parser.add_argument("--memory_mode", type=str, choices=["auto", "cuda", "model_cpu_offload", "sequential_cpu_offload"], default="cuda")
    parser.add_argument("--dataset_name", type=str, default="Gustavosta/Stable-Diffusion-Prompts")
    
    # --- 水印维度配置 ---
    parser.add_argument("--noise_ch", type=int, default=16)
    parser.add_argument("--msg_len", type=int, default=256)
    parser.add_argument("--spatial", type=int, default=64)
    
    # --- 生成超参 ---
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--base_seed", type=int, default=42)
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16

    print("⏳ 正在加载 LPIPS 模型 (AlexNet后端，标准配置)...")
    # LPIPS 默认推荐使用 alexnet 后端来计算感知距离
    lpips_vgg = lpips.LPIPS(net='alex').to(device)
    
    # 图像预处理 (LPIPS 需要 [-1, 1] 范围的 Tensor)
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) 
    ])

    print(f"⏳ 正在加载提示词数据集 ({args.dataset_name})...")
    dataset = load_dataset(args.dataset_name, split="train")
    actual_num_prompts = min(args.num_prompts, len(dataset))
    
    print("⏳ 正在加载 InversionDiffusion3Pipeline...")
    pipe = InversionDiffusion3Pipeline.from_pretrained(
        canonicalize_model_id(args.model_id),
        torch_dtype=weight_dtype
    )
    if args.memory_mode == "cuda":
        pipe = pipe.to(device)
    elif args.memory_mode == "model_cpu_offload":
        pipe.enable_model_cpu_offload(device=device)
    pipe.set_progress_bar_config(disable=True) 

    print("⏳ 正在加载 Watermark Encoder (仅需编码器参与生成)...")
    Encoder, _ = load_watermark_classes(args.watermark_impl)
    encoder = Encoder(args.noise_ch, args.msg_len, args.spatial).to(device).to(weight_dtype)
    encoder.load_state_dict(torch.load(os.path.join(args.weights_dir, "encoder_final.pth"), map_location=device))
    encoder.eval()

    all_prompts_lpips = []

    print(f"\n🚀 开始计算 Diversity (共测 {actual_num_prompts} 个提示词，每个生成 {args.images_per_prompt} 张图)...")
    
    with tqdm(total=actual_num_prompts) as pbar:
        for prompt_id in range(actual_num_prompts):
            current_prompt = dataset[prompt_id].get('Prompt', dataset[prompt_id].get('text', ''))
            
            prompt_images_tensors = []
            
            # 为当前提示词生成多张不同 Seed 的图片
            for img_idx in range(args.images_per_prompt):
                # 确保每次生成的 Seed 都不同
                current_seed = args.base_seed + (prompt_id * 100) + img_idx
                torch.manual_seed(current_seed)
                
                # 随机生成 watermark message 和初始噪声
                msg_bits = torch.randint(0, 2, (1, args.msg_len)).to(device).to(weight_dtype)
                init_noise = torch.randn(1, args.noise_ch, args.spatial, args.spatial).to(device).to(weight_dtype)

                with torch.no_grad():
                    # 潜入水印特征
                    initial_latents = encoder(init_noise, msg_bits)
                    
                    # 生成图像
                    generated_output = pipe(
                        prompt=current_prompt,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        height=args.height,
                        width=args.width,
                        latents=initial_latents 
                    )
                
                image = generated_output.images[0]
                # 转换为 LPIPS 需要的格式 [-1, 1] 并在 batch 维度扩展
                img_tensor = img_transform(image).unsqueeze(0).to(device)
                prompt_images_tensors.append(img_tensor)
            
            # 计算这 N 张图的两两 LPIPS 距离 (N_choose_2 对)
            pairwise_distances = []
            for img1, img2 in itertools.combinations(prompt_images_tensors, 2):
                with torch.no_grad():
                    # LPIPS 返回的形状是 (1,1,1,1)，用 .item() 转为标量
                    dist = lpips_vgg(img1, img2).item()
                    pairwise_distances.append(dist)
            
            # 计算当前 Prompt 的平均 LPIPS
            avg_prompt_lpips = sum(pairwise_distances) / len(pairwise_distances)
            all_prompts_lpips.append(avg_prompt_lpips)
            
            tqdm.write(f"Prompt {prompt_id} | Pairwise LPIPS: {avg_prompt_lpips:.4f} | Text: {current_prompt[:30]}...")
            pbar.update(1)

    # 汇总全局平均 Diversity
    final_diversity_score = sum(all_prompts_lpips) / len(all_prompts_lpips)
    
    print("\n" + "="*40)
    print("📊 DIVERSITY (LPIPS) REPORT")
    print("="*40)
    print(f"Total Prompts Tested:  {actual_num_prompts}")
    print(f"Images per Prompt:     {args.images_per_prompt}")
    print(f"Final Diversity Score: {final_diversity_score:.4f}")
    print("="*40)
    print("📝 参考指标: SD v2.1 基线约 0.707，值越接近基线代表多样性保持越好。")

if __name__ == "__main__":
    main()