import os
import json
import time
import argparse
import torch
from tqdm import tqdm
from torchvision import transforms

# 辅助函数：Bit 数组转 01 字符串
def tensor_to_str(tensor):
    return "".join([str(int(b.item())) for b in tensor.flatten()])

def load_watermark_classes(watermark_impl):
    if watermark_impl == "spatial":
        from train import Encoder, Decoder
    elif watermark_impl == "freq":
        from train_freq import Encoder, Decoder
    else:
        raise ValueError(f"Unsupported watermark_impl: {watermark_impl}")
    return Encoder, Decoder

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
    parser = argparse.ArgumentParser(description="SD3.5 Watermark Generation and Extraction")
    
    # --- 模型与路径配置 ---
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-large", help="SD3.5 model path or HuggingFace ID")
    parser.add_argument("--weights_dir", type=str, default="./weights", help="Directory containing trained Encoder/Decoder weights")
    parser.add_argument("--watermark_impl", type=str, choices=["freq", "spatial"], default="freq", help="Watermark model implementation to load")
    parser.add_argument("--memory_mode", type=str, choices=["auto", "cuda", "model_cpu_offload", "sequential_cpu_offload"], default="auto", help="Pipeline placement strategy")
    parser.add_argument("--disable_t5", action="store_true", help="Skip SD3/3.5 T5 text encoder to reduce RAM/VRAM use")
    parser.add_argument("--output_dir", type=str, default="./output_images", help="Base directory for output images and results")
    parser.add_argument("--dataset_name", type=str, default="Gustavosta/Stable-Diffusion-Prompts", help="HuggingFace dataset for prompts")
    
    # --- 水印维度配置 ---
    parser.add_argument("--noise_ch", type=int, default=16, help="Latent channels (16 for SD3/3.5)")
    parser.add_argument("--msg_len", type=int, default=256, help="Length of the watermark message bits")
    parser.add_argument("--spatial", type=int, default=64, help="Spatial dimension of latents (e.g., 64 for 512x512 image)")
    
    # --- 生成与反演超参 ---
    parser.add_argument("--num_images", type=int, default=10, help="Total number of images to generate and test")
    parser.add_argument("--num_inference_steps", type=int, default=28, help="Number of steps for both generation and inversion")
    parser.add_argument("--num_inversion_steps", type=int, default=None, help="Number of inversion steps; defaults to num_inference_steps")
    parser.add_argument("--guidance_scale", type=float, default=4.5, help="CFG scale for generation")
    parser.add_argument("--inverse_guidance_scale", type=float, default=None, help="CFG scale for inversion; defaults to 1.0 for empty prompt, generation guidance for same prompt")
    parser.add_argument("--inverse_prompt_mode", type=str, choices=["empty", "same"], default="empty", help="Use empty prompt or generation prompt during inversion")
    parser.add_argument("--inverse_solver", type=str, choices=["fixed_point", "euler", "gradient_descent"], default="fixed_point")
    parser.add_argument("--inverse_fixpoint_iters", type=int, default=10)
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    
      # noise settings
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

def main():
    # ==========================================
    # 1. 基础配置 & 超参加载
    # ==========================================
    args = parse_args()
    from datasets import load_dataset
    from src import utils
    from src.inversion.inverse_diffusion3 import InversionDiffusion3Pipeline

    args.model_id = canonicalize_model_id(args.model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16  # SD3.5 通常使用 FP16 推理
    inversion_steps = args.num_inversion_steps or args.num_inference_steps
    is_large_model = "large" in args.model_id.lower()
    memory_limit_gib = get_cgroup_memory_limit_gib()
    memory_mode = args.memory_mode
    if memory_mode == "auto":
        memory_mode = "cuda" if (args.disable_t5 or not is_large_model) else "model_cpu_offload"

    if (
        is_large_model
        and not args.disable_t5
        and memory_mode in {"model_cpu_offload", "sequential_cpu_offload"}
        and memory_limit_gib is not None
        and memory_limit_gib < 40
    ):
        raise RuntimeError(
            f"SD3.5 Large with T5 + CPU offload needs more host RAM than this container limit "
            f"({memory_limit_gib:.1f} GiB). Use --disable_t5 --memory_mode cuda, or switch to "
            "stabilityai/stable-diffusion-3.5-medium."
        )
    
    # 目录配置
    experiment_name = f"test_run_{int(time.time())}"
    output_base_dir = os.path.join(args.output_dir, experiment_name)
    image_save_path = os.path.join(output_base_dir, "image")
    os.makedirs(image_save_path, exist_ok=True)
    
    # 保存本次运行的超参数配置
    settings = vars(args)
    settings["effective_memory_mode"] = memory_mode
    with open(os.path.join(output_base_dir, "settings.json"), "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)

    # 图像预处理 (将 [0, 255] PIL 转换为 [-1, 1] 的 Tensor)
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) 
    ])

    # ==========================================
    # 2. 加载数据集与模型
    # ==========================================
    print(f"⏳ 正在加载提示词数据集 ({args.dataset_name})...")
    dataset = load_dataset(args.dataset_name, split="train")
    
    actual_num_images = min(args.num_images, len(dataset))
    print(f"✅ 数据集加载完成！总计 {len(dataset)} 条 prompt，本次测试抽取前 {actual_num_images} 条。")

    print("⏳ 正在加载 InversionDiffusion3Pipeline...")
    pipe_kwargs = dict(torch_dtype=weight_dtype)
    if args.disable_t5:
        pipe_kwargs.update(text_encoder_3=None, tokenizer_3=None)
    pipe = InversionDiffusion3Pipeline.from_pretrained(
        args.model_id,
        **pipe_kwargs
    )
    if memory_mode == "cuda":
        pipe = pipe.to(device)
    elif memory_mode == "model_cpu_offload":
        pipe.enable_model_cpu_offload(device=device)
    elif memory_mode == "sequential_cpu_offload":
        pipe.enable_sequential_cpu_offload(device=device)
    print(f"✅ Pipeline memory mode: {memory_mode}")
    if args.disable_t5:
        print("✅ T5 text encoder disabled to save memory")
    pipe.set_progress_bar_config(disable=True) 

    print("⏳ 正在加载 Watermark Encoder & Decoder...")
    Encoder, Decoder = load_watermark_classes(args.watermark_impl)
    encoder = Encoder(args.noise_ch, args.msg_len, args.spatial).to(device).to(weight_dtype)
    decoder = Decoder(args.noise_ch, args.msg_len).to(device)

    encoder.load_state_dict(torch.load(os.path.join(args.weights_dir, "encoder_best.pth"), map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(args.weights_dir, "decoder_best.pth"), map_location=device))
    
    encoder.eval()
    decoder.eval()

    results = {}
    total_start_time = time.time()

    # ==========================================
    # 3. 批量生成与真实反演提取循环
    # ==========================================
    print(f"\n🚀 开始批量测试...")
    with tqdm(total=actual_num_images) as pbar:
        for prompt_id in range(actual_num_images):
            current_prompt = dataset[prompt_id].get('Prompt', dataset[prompt_id].get('text', ''))
            
            current_seed = args.seed + prompt_id
            torch.manual_seed(current_seed)
            
            # 随机生成 msg_len bit 消息
            msg_bits_raw = torch.randint(0, 2, (1, args.msg_len)).to(device)
            msg_bits = msg_bits_raw.to(weight_dtype) 
            
            # 生成标准的初始高斯噪声
            init_noise = torch.randn(1, args.noise_ch, args.spatial, args.spatial).to(device).to(weight_dtype)

            # --- [水印嵌入] ---
            with torch.no_grad():
                initial_latents = encoder(init_noise, msg_bits)

            # --- [前向生成] ---
            with torch.no_grad():
                generated_output = pipe(
                    prompt=current_prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width,
                    latents=initial_latents 
                )
                
            image = generated_output.images[0]
            img_filename = f"{str(prompt_id).zfill(5)}.png"
            image.save(os.path.join(image_save_path, img_filename))
            noised_image = utils.image_distortion(image, args.seed + prompt_id, args)

            # --- [闭环提取] ---
            image_tensor = img_transform(noised_image).unsqueeze(0).to(device).to(weight_dtype)
            
            with torch.no_grad():
                latents = pipe.get_image_latents(image_tensor, sample=False)
                
                inv_start = time.time()
                inverse_prompt = current_prompt if args.inverse_prompt_mode == "same" else ""
                if args.inverse_guidance_scale is None:
                    inverse_guidance_scale = args.guidance_scale if args.inverse_prompt_mode == "same" else 1.0
                else:
                    inverse_guidance_scale = args.inverse_guidance_scale
                reversed_latents = pipe.naive_forward_diffusion(
                    latents=latents,
                    prompt=inverse_prompt,
                    num_inference_steps=inversion_steps,
                    guidance_scale=inverse_guidance_scale,
                    inverse_solver=args.inverse_solver,
                    inverse_fixpoint_iters=args.inverse_fixpoint_iters
                )
                inv_end = time.time()

                # Decoder 很小，用 FP32 能减少 JPEG 后 logits 贴近 0 时的符号抖动。
                logits = decoder(reversed_latents.float())
                
                extracted_bits = (logits > 0).float()
                acc = (extracted_bits == msg_bits_raw.float()).float().mean().item()

            # --- 记录当前图片的数据 ---
            results[prompt_id] = {
                "seed": current_seed,
                "prompt": current_prompt,
                "inversion_time": inv_end - inv_start,
                "embedded_bits": tensor_to_str(msg_bits_raw),
                "extracted_bits": tensor_to_str(extracted_bits),
                "bit_acc": acc
            }

            # 实时更新控制台反馈
            tqdm.write(f"ID {str(prompt_id).zfill(5)} | MsgAcc: {acc:.4f} | Prompt: {current_prompt[:30]}...")
            pbar.update(1)

    # ==========================================
    # 4. 汇总统计与保存 JSON
    # ==========================================
    total_time = time.time() - total_start_time
    total_acc = sum([v["bit_acc"] for k, v in results.items() if isinstance(k, int)]) / actual_num_images
    avg_inv_time = sum([v["inversion_time"] for k, v in results.items() if isinstance(k, int)]) / actual_num_images

    results["summary"] = {
        "dataset_name": args.dataset_name,
        "total_avg_bit_accuracy": total_acc,
        "avg_inversion_time_per_image": avg_inv_time,
        "total_images": actual_num_images,
        "msg_length": args.msg_len,
        "num_inference_steps": args.num_inference_steps,
        "num_inversion_steps": inversion_steps,
        "guidance_scale": args.guidance_scale,
        "memory_mode": memory_mode
    }

    results_path = os.path.join(output_base_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n" + "="*40)
    print("⏱️  TIMING REPORT & SUMMARY")
    print("="*40)
    print(f"Total Execution Time:      {total_time:.2f} seconds")
    print(f"Avg Inversion Time/Image:  {avg_inv_time:.2f} seconds")
    print(f"总平均比特准确率 (MsgAcc): {total_acc:.4f}")
    print(f"图片与结果已保存至:        {output_base_dir}")
    print("="*40)

if __name__ == "__main__":
    main()
