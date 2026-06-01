import os
import json
import time
import argparse
import torch
from tqdm import tqdm
from datasets import load_dataset
from src.inversion.inverse_diffusion3 import InversionDiffusion3Pipeline

def canonicalize_model_id(model_id):
    lower_model_id = model_id.lower()
    if lower_model_id.startswith("stabilityai/stable-diffusion-3.5-"):
        return lower_model_id
    return model_id

def parse_args():
    parser = argparse.ArgumentParser(description="Minimal SD3.5 Inversion Pipeline (Latent Fixed)")
    
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--memory_mode", type=str, choices=["auto", "cuda", "model_cpu_offload", "sequential_cpu_offload"], default="auto")
    parser.add_argument("--disable_t5", action="store_true", help="Skip SD3/3.5 T5 text encoder to reduce RAM/VRAM use")
    parser.add_argument("--output_dir", type=str, default="./output_minimal", help="Base directory for output")
    parser.add_argument("--dataset_name", type=str, default="Gustavosta/Stable-Diffusion-Prompts")
    
    # 实验超参
    parser.add_argument("--noise_ch", type=int, default=16, help="Latent channels (16 for SD3/3.5)")
    parser.add_argument("--spatial", type=int, default=64, help="Spatial dimension of latents")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to process")
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    
    # 反演超参
    parser.add_argument("--inverse_solver", type=str, choices=["fixed_point", "euler", "gradient_descent"], default="fixed_point")
    parser.add_argument("--inverse_fixpoint_iters", type=int, default=20)
    
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()

def main():
    args = parse_args()
    args.model_id = canonicalize_model_id(args.model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16
    
    memory_mode = args.memory_mode
    is_large_model = "large" in args.model_id.lower()
    if memory_mode == "auto":
        memory_mode = "cuda" if (args.disable_t5 or not is_large_model) else "model_cpu_offload"

    experiment_name = f"{args.name}_{int(time.time())}"
    output_base_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(output_base_dir, exist_ok=True)

    with open(os.path.join(output_base_dir, "settings.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

    print(f"⏳ Loading Dataset ({args.dataset_name})...")
    dataset = load_dataset(args.dataset_name, split="train")
    actual_num_images = min(args.num_images, len(dataset))
    
    print("⏳ Loading InversionDiffusion3Pipeline...")
    pipe_kwargs = dict(torch_dtype=weight_dtype)
    if args.disable_t5:
        pipe_kwargs.update(text_encoder_3=None, tokenizer_3=None)
        
    pipe = InversionDiffusion3Pipeline.from_pretrained(args.model_id, **pipe_kwargs)
    
    if memory_mode == "cuda":
        pipe = pipe.to(device)
    elif memory_mode == "model_cpu_offload":
        pipe.enable_model_cpu_offload(device=device)
    elif memory_mode == "sequential_cpu_offload":
        pipe.enable_sequential_cpu_offload(device=device)
        
    pipe.set_progress_bar_config(disable=True)

    print(f"\n🚀 开始极简版反演实验 (绕过 VAE 误差)...")
    with tqdm(total=actual_num_images) as pbar:
        for prompt_id in range(actual_num_images):
            current_prompt = dataset[prompt_id].get('Prompt', dataset[prompt_id].get('text', ''))
            
            torch.manual_seed(args.seed + prompt_id)
            
            sample_dir = os.path.join(output_base_dir, f"sample_{prompt_id:05d}")
            os.makedirs(sample_dir, exist_ok=True)

            # ==========================================
            # 步骤 1: 生成初始噪声 z
            # ==========================================
            z = torch.randn(1, args.noise_ch, args.spatial, args.spatial).to(device, weight_dtype)

            with torch.no_grad():
                # ==========================================
                # 步骤 2: 生成原图，并提取纯净的 Latent
                # ==========================================
                # A. 先生成一次，保存可视化的原图
                img_output = pipe(
                    prompt=current_prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width,
                    latents=z.clone(),
                    output_type="pil"
                )
                img_output.images[0].save(os.path.join(sample_dir, "01_original.png"))

                # B. 用完全相同的条件再跑一次，但强制输出浮点 Latent (绕过 VAE)
                latent_output = pipe(
                    prompt=current_prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width,
                    latents=z.clone(),
                    output_type="latent"  # 💥 核心修改点
                )
                original_latent = latent_output[0] 

                # ==========================================
                # 步骤 3: 纯净 Latent 反演 -> 推导初始噪声 z'
                # ==========================================
                z_prime = pipe.naive_forward_diffusion(
                    original_latent.clone(),
                    prompt="",  # 强制空提示词
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=1.0, # 强制 CFG=1.0
                    inverse_solver=args.inverse_solver,
                    inverse_fixpoint_iters=args.inverse_fixpoint_iters
                )

                # ==========================================
                # 步骤 4: 用反演出的 z' + 空提示词 重建图片
                # ==========================================
                reconstructed_output = pipe(
                    prompt="",  # 强制空提示词
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=1.0, # 强制 CFG=1.0
                    height=args.height,
                    width=args.width,
                    latents=z_prime.clone(),
                    output_type="pil"
                )
                image_reconstructed = reconstructed_output.images[0]
                image_reconstructed.save(os.path.join(sample_dir, "02_reconstructed_empty_prompt.png"))

            tqdm.write(f"Sample {prompt_id:05d} Done")
            pbar.update(1)

    print("\n" + "="*50)
    print(f"✅ 实验完成! 结果保存在: {output_base_dir}")
    print("="*50)

if __name__ == "__main__":
    main()