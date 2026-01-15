# Copyright (2024) Bytedance Ltd. and/or its affiliates 
import os
import argparse
import sys
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
import accelerate
import k_diffusion as K
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torchvision import transforms as T

# ================= 配置区域 =================
DATASET_ROOT = "OmniConsistency_dataset"  # 数据集根目录
OUTPUT_FOLDER_NAME = "output-deadiff"     # 结果保存的子文件夹名称

BATCH_SIZE = 1           # 每次生成一张
STEPS = 50
SEED = 123456
# ===========================================

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda().half() # 强制 FP16
    model.eval()
    return model

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale, img_weight):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = []
        if isinstance(uncond[0], list):
            for uc, c in zip(uncond, cond):
                cond_in_temp = []
                for c_tmp, uc_tmp in zip(c, uc):
                    cond_in_temp.append(torch.cat([uc_tmp, c_tmp]) if c_tmp is not None else None)
                cond_in.append(cond_in_temp)
        else:
            for c in cond:
                if isinstance(c, list):
                    cond_in_temp = []
                    for c_tmp, uc in zip(c, uncond):
                        cond_in_temp.append(torch.cat([uc, c_tmp]))
                    cond_in.append(cond_in_temp)
                else:
                    cond_in.append(torch.cat([uncond, c]))
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in, img_weight=img_weight).chunk(2)
        return uncond + (cond - uncond) * cond_scale

class DEADiff(object):
    def __init__(self, config, ckpt):
        config = OmegaConf.load(f"{config}")
        self.model = load_model_from_config(config, f"{ckpt}")
        self.model_wrap = K.external.CompVisDenoiser(self.model)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)

    def generate(self, prompt, image_input, batch_size, ddim_steps, scale, img_weight, seed):
        accelerator = accelerate.Accelerator()
        device = accelerator.device
        seed_everything(seed)
        prompts = batch_size * [prompt]
        
        with torch.no_grad():
            with autocast("cuda"):
                with self.model.ema_scope():
                    subject_text = ["style"] * batch_size
                    
                    # 注意：纯DEADiff逻辑中，image_input既是输入也是风格参考
                    c_encoder_hidden_states = self.model.get_learned_conditioning({
                            'target_text': prompts,
                            'inp_image': 2*(T.ToTensor()(Image.fromarray(image_input).convert('RGB').resize((224, 224)))-0.5).unsqueeze(0).repeat(batch_size, 1,1,1).to('cuda'),
                            'subject_text': subject_text,
                        })
                    
                    uc_encoder_hidden_states = self.model.get_learned_conditioning({
                        'target_text': batch_size * [""], 
                        'subject_text': subject_text
                    })
                    
                    uc, c = uc_encoder_hidden_states, c_encoder_hidden_states
                    shape = [4, 64, 64]
                    
                    sampler = DDIMSampler(self.model)
                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                     conditioning=c,
                                                     batch_size=batch_size,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=[uc, uc],
                                                     img_weight=img_weight)
                    
                    x_samples = self.model.decode_first_stage(samples_ddim)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    
                    n_rows = 2 if batch_size >= 4 else 1
                    grid = make_grid(x_samples, nrow=n_rows)
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    return Image.fromarray(grid.astype(np.uint8))

def process_dataset():
    if not os.path.exists(DATASET_ROOT):
        print(f"Error: Dataset root '{DATASET_ROOT}' not found.")
        return

    # 1. 初始化模型
    print("Initializing Model...")
    inference = DEADiff("configs/inference_deadiff_512x512.yaml", "pretrained/deadiff_v1.ckpt")
    
    # 2. 遍历数据集根目录下的所有子文件夹 (风格文件夹)
    style_dirs = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]
    style_dirs.sort()

    print(f"Found {len(style_dirs)} style folders. Starting processing...")

    for style_idx, style_name in enumerate(style_dirs):
        style_dir_path = os.path.join(DATASET_ROOT, style_name)
        
        src_dir = os.path.join(style_dir_path, "src")
        caption_dir = os.path.join(style_dir_path, "caption")
        output_dir = os.path.join(style_dir_path, OUTPUT_FOLDER_NAME)

        # 检查必要的文件夹是否存在
        if not os.path.exists(src_dir) or not os.path.exists(caption_dir):
            print(f"Skipping {style_name}: 'src' or 'caption' folder missing.")
            continue

        # 创建输出文件夹
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有图片文件
        image_files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        image_files.sort()

        print(f"\nProcessing Style [{style_idx+1}/{len(style_dirs)}]: {style_name} ({len(image_files)} images)")

        for img_file in image_files:
            # 构造完整路径
            img_path = os.path.join(src_dir, img_file)
            
            # 寻找对应的 prompt 文件 (文件名相同，后缀为 .txt)
            base_name = os.path.splitext(img_file)[0]
            txt_file = base_name + ".txt"
            txt_path = os.path.join(caption_dir, txt_file)

            if not os.path.exists(txt_path):
                print(f"  Warning: Prompt file missing for {img_file}, skipping.")
                continue

            # 读取 Prompt
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()

            print(f"  -> Generating: {img_file} | Prompt: {prompt[:30]}...")

            try:
                # 加载图片
                style_pil = Image.open(img_path).convert("RGB")
                style_np = np.array(style_pil)

                # 执行生成
                result_image = inference.generate(
                    prompt=prompt,
                    image_input=style_np,
                    batch_size=BATCH_SIZE,
                    ddim_steps=STEPS,
                    scale=8.0,
                    img_weight=1.0,
                    seed=SEED
                )

                # 保存结果
                # 直接保存为同名文件到 output-deadiff 文件夹下
                save_path = os.path.join(output_dir, img_file)
                
                # 如果是 png 输入，建议保存为 png；如果是 jpg，保存为 jpg 或 png 均可
                # 这里为了质量统一保存为 png
                save_name = os.path.splitext(img_file)[0] + ".png"
                save_path = os.path.join(output_dir, save_name)
                
                result_image.save(save_path)
                
            except Exception as e:
                print(f"  Failed to process {img_file}: {e}")
                import traceback
                traceback.print_exc()

    print("\nAll processing completed!")

if __name__ == "__main__":
    process_dataset()