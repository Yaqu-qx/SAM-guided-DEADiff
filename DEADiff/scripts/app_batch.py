# Copyright (2024) Bytedance Ltd. and/or its affiliates 
import os
import argparse
import sys
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageOps, ImageDraw, ImageFont
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
import accelerate
import k_diffusion as K
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torchvision import transforms as T

# ================= 配置区域 =================
INPUT_FILE = "./myTestSet1220/app_batch_pairs.txt"      # 输入文件路径
OUTPUT_ROOT = "outputs/app_batch_results"              # 【修改】输出根目录
SUMMARY_FILENAME = "final_summary_chart.jpg"           # 【修改】最终汇总大图的文件名

BATCH_SIZE = 1                       # 每组生成几张结果
STEPS = 50
SEED = 123456

# 样式配置
CELL_HEIGHT = 512
TEXT_COL_WIDTH = 400
STYLE_COL_WIDTH = 512
RESULT_COL_WIDTH = 512
FONT_SIZE = 24
BG_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)
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

# === 辅助函数：绘制文字单元格 ===
def draw_text_cell(text, width, height):
    img = Image.new('RGB', (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", FONT_SIZE)
    except:
        font = ImageFont.load_default()

    lines = []
    words = text.split()
    current_line = ""
    chars_per_line = width // (FONT_SIZE // 2) 
    
    current_line = ""
    for word in words:
        if len(current_line + word) <= chars_per_line:
            current_line += word + " "
        else:
            lines.append(current_line)
            current_line = word + " "
    lines.append(current_line)
    
    y_text = (height - len(lines) * (FONT_SIZE + 5)) // 2 
    for line in lines:
        text_w = len(line) * (FONT_SIZE // 2)
        x_text = max(10, (width - text_w) // 2)
        draw.text((x_text, y_text), line, font=font, fill=TEXT_COLOR)
        y_text += FONT_SIZE + 5
    return img

# === 辅助函数：调整图片大小并保持比例填入正方形 ===
def resize_contain(img, target_size):
    img.thumbnail(target_size, Image.BICUBIC)
    background = Image.new('RGB', target_size, (255, 255, 255))
    bg_w, bg_h = target_size
    img_w, img_h = img.size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    background.paste(img, offset)
    return background

# === 【新增】辅助函数：生成合法的文件名 ===
def sanitize_filename(prompt, max_length=50):
    # 只保留字母、数字、下划线、空格、连字符
    safe_name = "".join([c for c in prompt if c.isalnum() or c in " -_"])
    safe_name = safe_name.strip().replace(" ", "_")
    return safe_name[:max_length]

def process_batch_chart():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # 创建输出根目录
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 1. 初始化模型
    inference = DEADiff("configs/inference_deadiff_512x512.yaml", "pretrained/deadiff_v1.ckpt")
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    row_images = [] 

    print(f"Start processing {len(lines)} groups...")

    # 2. 循环处理
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        parts = line.split()
        if len(parts) < 3: continue

        style_path = parts[1]
        prompt = " ".join(parts[2:])

        print(f"Processing [{idx+1}/{len(lines)}]: {prompt[:20]}...")

        if not os.path.exists(style_path):
            print(f"  Style image not found: {style_path}")
            continue

        try:
            # A. 准备 Prompt 图片 
            img_col1 = draw_text_cell(prompt, TEXT_COL_WIDTH, CELL_HEIGHT)
            
            # B. 准备 Style 图片 
            style_pil = Image.open(style_path).convert("RGB")
            img_col2 = resize_contain(style_pil, (STYLE_COL_WIDTH, CELL_HEIGHT))
            
            # C. 生成 Result 图片 
            style_np = np.array(style_pil)
            result_grid = inference.generate(
                prompt=prompt,
                image_input=style_np,
                batch_size=BATCH_SIZE,
                ddim_steps=STEPS,
                scale=8.0,
                img_weight=1.0,
                seed=SEED
            )
            img_col3 = resize_contain(result_grid, (RESULT_COL_WIDTH, CELL_HEIGHT))
            
            # === 【新增】D. 保存单组数据到独立文件夹 ===
            # 生成文件夹名：序号_提示词摘要
            folder_name = f"{idx+1:03d}_{sanitize_filename(prompt)}"
            group_dir = os.path.join(OUTPUT_ROOT, folder_name)
            os.makedirs(group_dir, exist_ok=True)

            # 1. 保存 Prompt 文本
            with open(os.path.join(group_dir, "prompt.txt"), "w", encoding="utf-8") as f:
                f.write(prompt)
            
            # 2. 保存风格原图
            style_pil.save(os.path.join(group_dir, "style_image.jpg"))
            
            # 3. 保存生成的风格迁移结果
            result_grid.save(os.path.join(group_dir, "result_image.jpg"))

            # 4. (可选) 保存 Prompt 的渲染图
            img_col1.save(os.path.join(group_dir, "prompt_render.jpg"))

            print(f"  -> Saved assets to: {group_dir}")

            # E. 拼接这一行 [Prompt | Style | Result] (用于最后的大图)
            margin = 10
            row_w = TEXT_COL_WIDTH + STYLE_COL_WIDTH + RESULT_COL_WIDTH + margin * 2
            row_h = CELL_HEIGHT
            
            row_img = Image.new('RGB', (row_w, row_h), (200, 200, 200)) 
            
            row_img.paste(img_col1, (0, 0))
            row_img.paste(img_col2, (TEXT_COL_WIDTH + margin, 0))
            row_img.paste(img_col3, (TEXT_COL_WIDTH + STYLE_COL_WIDTH + margin * 2, 0))
            
            row_images.append(row_img)

        except Exception as e:
            print(f"  Failed: {e}")
            import traceback
            traceback.print_exc()

    # 3. 最终垂直拼接并保存到 OUTPUT_ROOT
    if row_images:
        print("Stitching final chart...")
        total_h = sum([img.height for img in row_images]) + (len(row_images)-1) * 5 
        max_w = row_images[0].width
        
        final_chart = Image.new('RGB', (max_w, total_h), (255, 255, 255))
        
        current_y = 0
        for img in row_images:
            final_chart.paste(img, (0, current_y))
            current_y += img.height + 5
            
        final_chart_path = os.path.join(OUTPUT_ROOT, SUMMARY_FILENAME)
        final_chart.save(final_chart_path)
        print(f"Successfully saved summary chart to: {final_chart_path}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    process_batch_chart()