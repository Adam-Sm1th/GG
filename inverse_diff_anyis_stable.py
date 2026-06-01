import os
import json
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from datasets import load_dataset
from src import utils
from src.inversion.inverse_diffusion3 import InversionDiffusion3Pipeline

def get_cgroup_memory_limit_gib():
    try:
        with open("/sys/fs/cgroup/memory.max", "r", encoding="utf-8") as f:
            value = f.read().strip()
        if value == "max":
            return None
        return int(value) / (1024 ** 3)
    except OSError:
        return None

def canonicalize_model_id(model_id):
    lower_model_id = model_id.lower()
    if lower_model_id.startswith("stabilityai/stable-diffusion-3.5-"):
        return lower_model_id
    return model_id

def parse_args():
    parser = argparse.ArgumentParser(description="SD3.5 Noise Inversion Analysis (Aggregate)")
    
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--memory_mode", type=str, choices=["auto", "cuda", "model_cpu_offload", "sequential_cpu_offload"], default="auto")
    parser.add_argument("--disable_t5", action="store_true", help="Skip SD3/3.5 T5 text encoder to reduce RAM/VRAM use")
    parser.add_argument("--output_dir", type=str, default="./output_analysis", help="Base directory for output")
    parser.add_argument("--dataset_name", type=str, default="Gustavosta/Stable-Diffusion-Prompts")
    
    # --- 实验超参 ---
    parser.add_argument("--noise_ch", type=int, default=16, help="Latent channels (16 for SD3/3.5)")
    parser.add_argument("--spatial", type=int, default=64, help="Spatial dimension of latents")
    # 默认改回 100，你可以随意调
    parser.add_argument("--num_images", type=int, default=1000, help="Number of images to process (Runs for aggregate analysis)")
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    
    # --- 反演超参 ---
    parser.add_argument("--inverse_solver", type=str, choices=["fixed_point", "euler", "gradient_descent"], default="euler")
    parser.add_argument("--inverse_fixpoint_iters", type=int, default=10)
    parser.add_argument("--inverse_prompt_mode", type=str, choices=["empty", "same"], default="empty")
    
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    
    # --- 攻击层 (Distortion Settings) ---
    parser.add_argument('--jpeg_ratio', type=int, default=None)
    parser.add_argument('--random_crop_ratio', type=float, default=None)
    parser.add_argument('--random_drop_ratio', type=float, default=None)
    parser.add_argument('--gaussian_blur_r', type=int, default=None)
    parser.add_argument('--median_blur_k', type=int, default=None)
    parser.add_argument('--resize_ratio', type=float, default=None)
    parser.add_argument('--gaussian_std', type=float, default=None)
    parser.add_argument('--sp_prob', type=float, default=None)
    parser.add_argument('--brightness_factor', type=float, default=None)
    
    return parser.parse_args()

def has_distortion(args):
    return any(val is not None for val in [
        args.jpeg_ratio, args.random_crop_ratio, args.random_drop_ratio,
        args.gaussian_blur_r, args.median_blur_k, args.resize_ratio,
        args.gaussian_std, args.sp_prob, args.brightness_factor
    ])

def save_heatmaps_and_stats(z, z_prime_clean, z_prime_dist, base_dir, channels=16, scale=8):
    dev_dir = os.path.join(base_dir, "deviations")  
    mask_dir = os.path.join(base_dir, "sign_masks") 
    overlay_dir = os.path.join(base_dir, "overlays") 
    z_abs_overlay_dir = os.path.join(base_dir, "z_abs_overlays")
    
    os.makedirs(dev_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(z_abs_overlay_dir, exist_ok=True)
    
    z_prime_main = z_prime_dist if z_prime_dist is not None else z_prime_clean
    
    z_abs = torch.abs(z).squeeze(0).cpu().numpy()
    deviation = torch.abs(z - z_prime_main).squeeze(0).cpu().numpy()
    sign_flipped_main = ((z * z_prime_main) < 0).squeeze(0).cpu().numpy()
    
    H_orig, W_orig = deviation.shape[1], deviation.shape[2]
    border_pattern = np.ones((scale, scale), dtype=bool)
    if scale > 2:
        border_pattern[1:-1, 1:-1] = False 
    tiled_border = np.tile(border_pattern, (H_orig, W_orig))
    cmap_reds = plt.get_cmap("Reds")
    
    for c in range(channels):
        dev_c = deviation[c]
        orig_z_c = z_abs[c]
        flipped_c = sign_flipped_main[c]
        
        dev_scaled = np.repeat(np.repeat(dev_c, scale, axis=0), scale, axis=1)
        orig_z_scaled = np.repeat(np.repeat(orig_z_c, scale, axis=0), scale, axis=1)
        mask_scaled_main = np.repeat(np.repeat(flipped_c, scale, axis=0), scale, axis=1)

        plt.imsave(os.path.join(dev_dir, f"channel_{c:02d}.png"), dev_scaled, cmap="Reds")
        plt.imsave(os.path.join(mask_dir, f"channel_{c:02d}.png"), mask_scaled_main, cmap="gray")

        dev_min, dev_max = dev_scaled.min(), dev_scaled.max()
        norm_dev = (dev_scaled - dev_min) / (dev_max - dev_min) if dev_max > dev_min else np.zeros_like(dev_scaled)
        overlay_dev_img = cmap_reds(norm_dev)
        overlay_dev_img[mask_scaled_main & tiled_border] = [0.0, 0.0, 1.0, 1.0] 
        plt.imsave(os.path.join(overlay_dir, f"channel_{c:02d}.png"), overlay_dev_img)

        z_min, z_max = orig_z_scaled.min(), orig_z_scaled.max()
        norm_z = (orig_z_scaled - z_min) / (z_max - z_min) if z_max > z_min else np.zeros_like(orig_z_scaled)
        overlay_z_img = cmap_reds(norm_z)
        
        if z_prime_dist is not None:
            z_c_tensor = z.squeeze(0).cpu().numpy()[c]
            z_prime_clean_c = z_prime_clean.squeeze(0).cpu().numpy()[c]
            z_prime_dist_c = z_prime_dist.squeeze(0).cpu().numpy()[c]
            
            flipped_clean_c = (z_c_tensor * z_prime_clean_c) < 0
            flipped_dist_c = (z_c_tensor * z_prime_dist_c) < 0
            
            both_c = flipped_clean_c & flipped_dist_c          
            only_clean_c = flipped_clean_c & ~flipped_dist_c   
            only_dist_c = ~flipped_clean_c & flipped_dist_c    
            
            both_border = np.repeat(np.repeat(both_c, scale, axis=0), scale, axis=1) & tiled_border
            only_clean_border = np.repeat(np.repeat(only_clean_c, scale, axis=0), scale, axis=1) & tiled_border
            only_dist_border = np.repeat(np.repeat(only_dist_c, scale, axis=0), scale, axis=1) & tiled_border
            
            overlay_z_img[both_border] = [0.0, 0.0, 0.0, 1.0]        
            overlay_z_img[only_clean_border] = [0.0, 0.0, 1.0, 1.0]  
            overlay_z_img[only_dist_border] = [0.0, 1.0, 0.0, 1.0]   
        else:
            overlay_z_img[mask_scaled_main & tiled_border] = [0.0, 0.0, 0.0, 1.0]

        plt.imsave(os.path.join(z_abs_overlay_dir, f"channel_{c:02d}.png"), overlay_z_img)


def save_aggregate_analysis(fixed_z, all_flip_masks, base_dir, channels=16, scale=8):
    agg_dir = os.path.join(base_dir, "aggregate_analysis")
    never_flipped_dir = os.path.join(agg_dir, "z_abs_never_flipped_overlays")
    always_flipped_dir = os.path.join(agg_dir, "z_abs_always_flipped_overlays")
    freq_dir = os.path.join(agg_dir, "flip_frequency_heatmaps")
    
    os.makedirs(never_flipped_dir, exist_ok=True)
    os.makedirs(always_flipped_dir, exist_ok=True)
    os.makedirs(freq_dir, exist_ok=True)
    
    total_flips = np.sum(all_flip_masks, axis=0) 
    num_runs = len(all_flip_masks)
    
    never_flipped_mask = (total_flips == 0)      
    always_flipped_mask = (total_flips == num_runs) 
    
    z_abs = torch.abs(fixed_z).squeeze(0).cpu().numpy()
    H_orig, W_orig = z_abs.shape[1], z_abs.shape[2]
    
    border_pattern = np.ones((scale, scale), dtype=bool)
    if scale > 2:
        border_pattern[1:-1, 1:-1] = False 
    tiled_border = np.tile(border_pattern, (H_orig, W_orig))
    
    cmap_reds = plt.get_cmap("Reds")
    cmap_viridis = plt.get_cmap("viridis") 
    
    # 文本动态适配了跑图次数
    stats_lines = [
        "=========================================",
        f" Aggregate Analysis ({num_runs} Runs)",
        "=========================================\n"
    ]
    
    total_pixels_per_channel = H_orig * W_orig

    for c in range(channels):
        orig_z_c = z_abs[c]
        orig_z_scaled = np.repeat(np.repeat(orig_z_c, scale, axis=0), scale, axis=1)
        
        z_min, z_max = orig_z_scaled.min(), orig_z_scaled.max()
        norm_z = (orig_z_scaled - z_min) / (z_max - z_min) if z_max > z_min else np.zeros_like(orig_z_scaled)
        base_img = cmap_reds(norm_z)
        
        never_c = never_flipped_mask[c]
        never_scaled = np.repeat(np.repeat(never_c, scale, axis=0), scale, axis=1)
        img_never = base_img.copy()
        img_never[never_scaled & tiled_border] = [0.0, 0.0, 1.0, 1.0] 
        plt.imsave(os.path.join(never_flipped_dir, f"channel_{c:02d}.png"), img_never)
        
        always_c = always_flipped_mask[c]
        always_scaled = np.repeat(np.repeat(always_c, scale, axis=0), scale, axis=1)
        img_always = base_img.copy()
        img_always[always_scaled & tiled_border] = [0.0, 0.0, 0.0, 1.0] 
        plt.imsave(os.path.join(always_flipped_dir, f"channel_{c:02d}.png"), img_always)
        
        freq_c = total_flips[c]
        freq_scaled = np.repeat(np.repeat(freq_c, scale, axis=0), scale, axis=1)
        norm_freq = freq_scaled / num_runs 
        img_freq = cmap_viridis(norm_freq)
        plt.imsave(os.path.join(freq_dir, f"channel_{c:02d}.png"), img_freq)
        
        num_never = np.sum(never_c)
        num_always = np.sum(always_c)
        stats_lines.append(f"--- Channel {c:02d} ---")
        stats_lines.append(f"  [Stable Pixels (0 Flips)]      : {num_never} pixels ({(num_never/total_pixels_per_channel)*100:.2f}%)")
        stats_lines.append(f"  [Dead Pixels ({num_runs} Flips)]    : {num_always} pixels ({(num_always/total_pixels_per_channel)*100:.2f}%)\n")

    with open(os.path.join(agg_dir, "aggregate_statistics.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(stats_lines))


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

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) 
    ])

    print(f"⏳ Loading Dataset ({args.dataset_name})...")
    dataset = load_dataset(args.dataset_name, split="train")
    
    # 修复完毕：使用 args.num_images 读取你的输入
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

    print(f"\n🔒 锁定全局初始噪声 Z (Seed: {args.seed})。即将用 {actual_num_images} 个不同提示词轰炸同一个高斯分布...")
    torch.manual_seed(args.seed)
    fixed_z = torch.randn(1, args.noise_ch, args.spatial, args.spatial).to(device, weight_dtype)
    
    all_flip_masks = []

    print(f"🚀 开始跑 {actual_num_images} 次独立测试...")
    with tqdm(total=actual_num_images) as pbar:
        for prompt_id in range(actual_num_images):
            current_prompt = dataset[prompt_id].get('Prompt', dataset[prompt_id].get('text', ''))
            
            current_seed = args.seed + prompt_id + 100
            torch.manual_seed(current_seed)
            
            sample_dir = os.path.join(output_base_dir, f"sample_{prompt_id:05d}")
            os.makedirs(sample_dir, exist_ok=True)

            z = fixed_z.clone()

            with torch.no_grad():
                generated_output = pipe(
                    prompt=current_prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width,
                    latents=z.clone() 
                )
            
            image = generated_output.images[0]
            image.save(os.path.join(sample_dir, "generated_image_clean.png"))

            inverse_prompt = current_prompt if args.inverse_prompt_mode == "same" else ""
            inverse_guidance_scale = args.guidance_scale if args.inverse_prompt_mode == "same" else 1.0

            image_tensor_clean = img_transform(image).unsqueeze(0).to(device, weight_dtype)
            with torch.no_grad():
                latents_clean = pipe.get_image_latents(image_tensor_clean, sample=False)
                z_prime_clean = pipe.naive_forward_diffusion(
                    latents_clean,
                    prompt=inverse_prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=inverse_guidance_scale,
                    inverse_solver=args.inverse_solver,
                    inverse_fixpoint_iters=args.inverse_fixpoint_iters
                )
            
            z_prime_dist = None
            
            if has_distortion(args):
                noised_image = utils.image_distortion(image, current_seed, args)
                noised_image.save(os.path.join(sample_dir, "generated_image_distorted.png"))

                image_tensor_dist = img_transform(noised_image).unsqueeze(0).to(device, weight_dtype)
                
                with torch.no_grad():
                    latents_dist = pipe.get_image_latents(image_tensor_dist, sample=False)
                    z_prime_dist = pipe.naive_forward_diffusion(
                        latents_dist,
                        prompt=inverse_prompt,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=inverse_guidance_scale,
                        inverse_solver=args.inverse_solver,
                        inverse_fixpoint_iters=args.inverse_fixpoint_iters
                    )

            z_prime_main = z_prime_dist if z_prime_dist is not None else z_prime_clean
            flip_mask = ((z * z_prime_main) < 0).squeeze(0).cpu().numpy()
            all_flip_masks.append(flip_mask)

            save_heatmaps_and_stats(z, z_prime_clean, z_prime_dist, sample_dir, channels=args.noise_ch, scale=8)

            tqdm.write(f"Run {prompt_id+1}/{actual_num_images} Done | Prompt: {current_prompt[:30]}...")
            pbar.update(1)

    print("\n" + "="*40)
    print(f"📊 正在生成 {actual_num_images} 次跑图的全局聚合分析 (Aggregate Analysis)...")
    save_aggregate_analysis(fixed_z, all_flip_masks, output_base_dir, channels=args.noise_ch, scale=8)

    print(f"✅ 实验完美收官! 汇总结果已保存在: {os.path.join(output_base_dir, 'aggregate_analysis')}")
    print("="*40)

if __name__ == "__main__":
    main()