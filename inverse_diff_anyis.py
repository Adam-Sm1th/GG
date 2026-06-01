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
    parser = argparse.ArgumentParser(description="SD3.5 Noise Inversion Analysis with Distortions")
    
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--memory_mode", type=str, choices=["auto", "cuda", "model_cpu_offload", "sequential_cpu_offload"], default="auto")
    parser.add_argument("--disable_t5", action="store_true", help="Skip SD3/3.5 T5 text encoder to reduce RAM/VRAM use")
    parser.add_argument("--output_dir", type=str, default="./output_analysis", help="Base directory for output")
    parser.add_argument("--dataset_name", type=str, default="Gustavosta/Stable-Diffusion-Prompts")
    
    # --- 实验超参 ---
    parser.add_argument("--noise_ch", type=int, default=16, help="Latent channels (16 for SD3/3.5)")
    parser.add_argument("--spatial", type=int, default=64, help="Spatial dimension of latents")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to process")
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    
    # [新增]：是否固定全局初始噪声
    parser.add_argument("--fixed_initial_noise", action="store_true", help="如果开启，所有提示词都将使用同一个固定的初始高斯噪声")
    
    # --- 反演超参 ---
    parser.add_argument("--inverse_solver", type=str, choices=["fixed_point", "euler", "gradient_descent"], default="euler")
    parser.add_argument("--inverse_fixpoint_iters", type=int, default=10)
    parser.add_argument("--inverse_prompt_mode", type=str, choices=["empty", "same"], default="empty")
    
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    
    # --- 攻击层 (Distortion Settings) ---
    parser.add_argument('--jpeg_ratio', type=int, default=20)
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
    """判断是否启用了任何图像攻击参数"""
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
    total_pixels = H_orig * W_orig
    
    border_pattern = np.ones((scale, scale), dtype=bool)
    if scale > 2:
        border_pattern[1:-1, 1:-1] = False 
    tiled_border = np.tile(border_pattern, (H_orig, W_orig))
    
    cmap_reds = plt.get_cmap("Reds")
    
    stats_lines = [
        "=========================================",
        " Channel Statistics Report",
        f" Mode: {'Distorted' if z_prime_dist is not None else 'Clean'} Inversion",
        "=========================================\n"
    ]

    for c in range(channels):
        dev_c = deviation[c]
        orig_z_c = z_abs[c]
        flipped_c = sign_flipped_main[c]
        
        flip_count = np.sum(flipped_c)
        flip_ratio = (flip_count / total_pixels) * 100
        mean_dev = np.mean(dev_c)
        max_dev = np.max(dev_c)
        
        dev_bin_1 = np.sum(dev_c < 0.1) / total_pixels * 100
        dev_bin_2 = np.sum((dev_c >= 0.1) & (dev_c < 0.5)) / total_pixels * 100
        dev_bin_3 = np.sum((dev_c >= 0.5) & (dev_c < 1.0)) / total_pixels * 100
        dev_bin_4 = np.sum(dev_c >= 1.0) / total_pixels * 100
        
        mask_z1 = orig_z_c < 0.5
        mask_z2 = (orig_z_c >= 0.5) & (orig_z_c < 1.0)
        mask_z3 = (orig_z_c >= 1.0) & (orig_z_c < 2.0)
        mask_z4 = orig_z_c >= 2.0

        count_z1, count_z2, count_z3, count_z4 = np.sum(mask_z1), np.sum(mask_z2), np.sum(mask_z3), np.sum(mask_z4)

        z_bin_1 = count_z1 / total_pixels * 100
        z_bin_2 = count_z2 / total_pixels * 100
        z_bin_3 = count_z3 / total_pixels * 100
        z_bin_4 = count_z4 / total_pixels * 100
        
        if flip_count > 0:
            flipped_z_abs = orig_z_c[flipped_c] 
            f_z_bin_1 = np.sum(flipped_z_abs < 0.5) / flip_count * 100
            f_z_bin_2 = np.sum((flipped_z_abs >= 0.5) & (flipped_z_abs < 1.0)) / flip_count * 100
            f_z_bin_3 = np.sum((flipped_z_abs >= 1.0) & (flipped_z_abs < 2.0)) / flip_count * 100
            f_z_bin_4 = np.sum(flipped_z_abs >= 2.0) / flip_count * 100
        else:
            f_z_bin_1 = f_z_bin_2 = f_z_bin_3 = f_z_bin_4 = 0.0

        rate_in_z1 = (np.sum(mask_z1 & flipped_c) / count_z1 * 100) if count_z1 > 0 else 0.0
        rate_in_z2 = (np.sum(mask_z2 & flipped_c) / count_z2 * 100) if count_z2 > 0 else 0.0
        rate_in_z3 = (np.sum(mask_z3 & flipped_c) / count_z3 * 100) if count_z3 > 0 else 0.0
        rate_in_z4 = (np.sum(mask_z4 & flipped_c) / count_z4 * 100) if count_z4 > 0 else 0.0

        stats_lines.append(f"--- Channel {c:02d} ---")
        stats_lines.append(f"  [Sign Flipped Ratio]    : {flip_ratio:.2f}% ({flip_count}/{total_pixels} pixels)")
        stats_lines.append(f"  [Absolute Deviation]    : Mean = {mean_dev:.4f} | Max = {max_dev:.4f}")
        stats_lines.append(f"  [Deviation Dist.]       : <0.1: {dev_bin_1:.1f}% | 0.1~0.5: {dev_bin_2:.1f}% | 0.5~1.0: {dev_bin_3:.1f}% | >=1.0: {dev_bin_4:.1f}%")
        stats_lines.append(f"  [Original Abs(z) Dist.] : <0.5: {z_bin_1:.1f}% | 0.5~1.0: {z_bin_2:.1f}% | 1.0~2.0: {z_bin_3:.1f}% | >=2.0: {z_bin_4:.1f}%")
        stats_lines.append(f"  [Flipped Pixels Abs(z)] : <0.5: {f_z_bin_1:.1f}% | 0.5~1.0: {f_z_bin_2:.1f}% | 1.0~2.0: {f_z_bin_3:.1f}% | >=2.0: {f_z_bin_4:.1f}%")
        stats_lines.append(f"  [Flip Rate within Bin]  : <0.5: {rate_in_z1:.1f}% | 0.5~1.0: {rate_in_z2:.1f}% | 1.0~2.0: {rate_in_z3:.1f}% | >=2.0: {rate_in_z4:.1f}%\n")

        # ---------------- 图像渲染 ----------------
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

        # ====== z_abs_overlays 三色对比逻辑 ======
        z_min, z_max = orig_z_scaled.min(), orig_z_scaled.max()
        norm_z = (orig_z_scaled - z_min) / (z_max - z_min) if z_max > z_min else np.zeros_like(orig_z_scaled)
        overlay_z_img = cmap_reds(norm_z)
        
        if z_prime_dist is not None:
            z_c_tensor = z.squeeze(0).cpu().numpy()[c]
            z_prime_clean_c = z_prime_clean.squeeze(0).cpu().numpy()[c]
            z_prime_dist_c = z_prime_dist.squeeze(0).cpu().numpy()[c]
            
            flipped_clean_c = (z_c_tensor * z_prime_clean_c) < 0
            flipped_dist_c = (z_c_tensor * z_prime_dist_c) < 0
            
            both_c = flipped_clean_c & flipped_dist_c          # 黑圈：始终翻转
            only_clean_c = flipped_clean_c & ~flipped_dist_c   # 蓝圈：仅纯净翻转
            only_dist_c = ~flipped_clean_c & flipped_dist_c    # 绿圈：仅攻击翻转
            
            both_border = np.repeat(np.repeat(both_c, scale, axis=0), scale, axis=1) & tiled_border
            only_clean_border = np.repeat(np.repeat(only_clean_c, scale, axis=0), scale, axis=1) & tiled_border
            only_dist_border = np.repeat(np.repeat(only_dist_c, scale, axis=0), scale, axis=1) & tiled_border
            
            overlay_z_img[both_border] = [0.0, 0.0, 0.0, 1.0]        # 纯黑 Black 
            overlay_z_img[only_clean_border] = [0.0, 0.0, 1.0, 1.0]  # 纯蓝 Blue
            overlay_z_img[only_dist_border] = [0.0, 1.0, 0.0, 1.0]   # 纯绿 Green
        else:
            # 如果没加攻击，默认用黑圈标识翻转
            overlay_z_img[mask_scaled_main & tiled_border] = [0.0, 0.0, 0.0, 1.0]

        plt.imsave(os.path.join(z_abs_overlay_dir, f"channel_{c:02d}.png"), overlay_z_img)

    stats_file_path = os.path.join(base_dir, "statistics.txt")
    with open(stats_file_path, "w", encoding="utf-8") as f:
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

    # ==========================================
    # [新增]：判断是否启用固定的初始噪声
    # ==========================================
    if args.fixed_initial_noise:
        print(f"🔒 启用固定初始噪声 (Seed: {args.seed}). 所有样本的初始 Z 将保持完全一致。")
        torch.manual_seed(args.seed)
        fixed_z = torch.randn(1, args.noise_ch, args.spatial, args.spatial).to(device, weight_dtype)

    print(f"\n🚀 开始批量实验...")
    with tqdm(total=actual_num_images) as pbar:
        for prompt_id in range(actual_num_images):
            current_prompt = dataset[prompt_id].get('Prompt', dataset[prompt_id].get('text', ''))
            
            # 使用针对当前迭代的种子，确保图像攻击、其他随机行为正常进行
            current_seed = args.seed + prompt_id
            torch.manual_seed(current_seed)
            
            sample_dir = os.path.join(output_base_dir, f"sample_{prompt_id:05d}")
            os.makedirs(sample_dir, exist_ok=True)

            # [新增]：根据配置选择获取初始噪声 z 的方式
            if args.fixed_initial_noise:
                z = fixed_z.clone()
            else:
                z = torch.randn(1, args.noise_ch, args.spatial, args.spatial).to(device, weight_dtype)

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
                # 即使固定了初始噪声 z，图像攻击的扰动（裁剪、噪声添加）依然会用到 current_seed
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

            save_heatmaps_and_stats(z, z_prime_clean, z_prime_dist, sample_dir, channels=args.noise_ch, scale=8)

            tqdm.write(f"Sample {prompt_id:05d} Done | Prompt: {current_prompt[:30]}...")
            pbar.update(1)

    print("\n" + "="*40)
    print(f"✅ 实验完成! 结果保存在: {output_base_dir}")
    print("="*40)

if __name__ == "__main__":
    main()