# Copyright (2024) Bytedance Ltd. and/or its affiliates 
import os
import argparse
import sys
import subprocess
import cv2
import shutil
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
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
from ldm.controlnet.annotator.util import HWC3, resize_image
from ldm.controlnet.annotator.canny import CannyDetector

# ================= 配置区域 =================
INPUT_FILE = "myTestSet1220/inputs_sam_pairs.txt"         # 输入文件
OUTPUT_ROOT = "outputs/results_sam_batch_results"     # 输出根目录
SUMMARY_FILE = "final_sam_summary.jpg" # 汇总长图文件名

# SAM 配置 (请确认路径)
SAM_PYTHON_PATH = "/public/home/CS290U/panyq2025-CS290U/anaconda3/envs/grounding-sam-2/bin/python"
SAM_SCRIPT_PATH = "/public/home/CS290U/panyq2025-CS290U/project1/run_samdino_inference.py"

# 生成配置
BATCH_SIZE = 1  # 批量处理建议设为1，方便对应 Mask
STEPS = 50
SEED = 42

# 图表样式配置
CELL_HEIGHT = 512
COL_WIDTH_TEXT = 300
COL_WIDTH_IMG = 512
FONT_SIZE = 24
BG_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)
# ===========================================

apply_canny = CannyDetector()

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda().half()
    model.eval()
    return model

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale, img_weight):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        assert isinstance(cond, dict)
        c_crossattn = cond['c_crossattn']
        c_concat = cond['c_concat']
        uc_crossattn = uncond['c_crossattn']
        uc_concat = uncond['c_concat']
        cond_in = []
        if isinstance(uc_crossattn[0], list):
            for uc, c in zip(uc_crossattn, c_crossattn):
                cond_in_temp = []
                for c_tmp, uc_tmp in zip(c, uc):
                    cond_in_temp.append(torch.cat([uc_tmp, c_tmp]) if c_tmp is not None else None)
                cond_in.append(cond_in_temp)
        else:
            for c in c_crossattn:
                if isinstance(c, list):
                    cond_in_temp = []
                    for c_tmp, uc in zip(c, uc_crossattn):
                        cond_in_temp.append(torch.cat([uc, c_tmp]))
                    cond_in.append(cond_in_temp)
                else:
                    cond_in.append(torch.cat([uc_crossattn, c]))
        cond_in = {'c_crossattn': cond_in, 'c_concat': [torch.cat([uc_concat[0], c_concat[0]], 0)]}
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in, img_weight=img_weight).chunk(2)
        return uncond + (cond - uncond) * cond_scale

class DEADiff_Batch(object):
    def __init__(self, config, ckpt, ckpt_controlnet):
        config = OmegaConf.load(f"{config}")
        config.model.params.control_stage_config.params.ckpt_path = ckpt_controlnet
        self.model = load_model_from_config(config, f"{ckpt}")
        self.model_wrap = K.external.CompVisDenoiser(self.model)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)

    def call_sam_external(self, image_pil, prompt):
        temp_input_path = os.path.abspath("temp_sam_input_batch.jpg")
        temp_mask_path = os.path.abspath("temp_sam_mask_batch.png")
        image_pil.save(temp_input_path)

        cmd = [
            SAM_PYTHON_PATH,
            SAM_SCRIPT_PATH,
            "--input_image", temp_input_path,
            "--prompt", prompt,
            "--output_mask", temp_mask_path,
            "--device", "cpu"
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if os.path.exists(temp_mask_path):
                # 重新读取并确保是灰度图
                mask = cv2.imread(temp_mask_path, cv2.IMREAD_GRAYSCALE)
                return mask
        except Exception as e:
            print(f"SAM Error: {e}")
        return None

    def process_single_group(self, content_np, style_np, style_prompt, sam_prompt, seed):
        accelerator = accelerate.Accelerator()
        device = accelerator.device
        seed_everything(seed)
        
        batch_size = BATCH_SIZE
        
        # 1. Canny 预处理
        img_h, img_w, _ = content_np.shape
        detected_map = apply_canny(resize_image(content_np, 384), 100, 200) # Canny Detect
        detected_map = HWC3(detected_map)
        
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(batch_size)], dim=0)
        control = rearrange(control, 'b h w c -> b c h w').clone()

        # 2. Diffusion 推理
        with torch.no_grad(), autocast("cuda"), self.model.ema_scope():
            prompts = [style_prompt] * batch_size
            subject_text = ["style"] * batch_size # 默认为 style 模式
            
            # 风格图处理
            style_tensor = 2*(T.ToTensor()(Image.fromarray(style_np).convert('RGB').resize((224, 224)))-0.5).unsqueeze(0).repeat(batch_size, 1,1,1).to('cuda')

            c_enc = self.model.get_learned_conditioning({
                    'target_text': prompts,
                    'inp_image': style_tensor,
                    'subject_text': subject_text,
            })
            uc_enc = self.model.get_learned_conditioning({
                    'target_text': batch_size * [""], 
                    'subject_text': subject_text
            })
            
            cond = {"c_concat": [control], "c_crossattn": c_enc}
            un_cond = {"c_concat": [control], "c_crossattn": [uc_enc, uc_enc]} # ControlNet 需要双份 uc
            
            # 使用 content_np 的缩放后的尺寸作为 latent shape
            # detected_map 是经过 resize_image 的，尺寸可能是 384x512
            H_det, W_det = detected_map.shape[:2]
            shape = [4, H_det // 8, W_det // 8]

            sampler = DDIMSampler(self.model)
            samples, _ = sampler.sample(S=STEPS,
                                        conditioning=cond,
                                        batch_size=batch_size,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=8.0,
                                        unconditional_conditioning=un_cond,
                                        img_weight=1.0)
            
            x_samples = self.model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            
            # 获取生成的风格图 (PIL)
            # 取第一张
            stylized_pil = T.ToPILImage()(x_samples[0])
            stylized_np = np.array(stylized_pil)

        # 3. SAM 分割与合成
        print(f"  Calling SAM with prompt: '{sam_prompt}'...")
        content_pil = Image.fromarray(content_np)
        mask_np = self.call_sam_external(content_pil, sam_prompt)
        
        final_pil = stylized_pil # 默认是全图风格化
        mask_pil = None

        if mask_np is not None:
            # 尺寸对齐
            target_h, target_w = stylized_np.shape[:2]
            
            # A. 调整 Mask
            mask_resized = cv2.resize(mask_np, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            mask_pil = Image.fromarray(mask_resized) # 用于保存

            # B. 调整原图
            content_resized = cv2.resize(content_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            # C. 合成
            mask_float = mask_resized.astype(np.float32) / 255.0
            mask_3d = np.expand_dims(mask_float, axis=2)
            
            # Final = Stylized * Mask + Content * (1-Mask)
            composite = stylized_np * mask_3d + content_resized * (1 - mask_3d)
            final_pil = Image.fromarray(composite.clip(0, 255).astype(np.uint8))
        else:
            print("  Warning: SAM Mask not found. Using full stylized image.")
            # 造一个全黑 mask 用于显示
            mask_pil = Image.new('L', stylized_pil.size, 0)

        return stylized_pil, mask_pil, final_pil

# === 绘图辅助函数 ===
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
    for word in words:
        if len(current_line + word) <= chars_per_line:
            current_line += word + " "
        else:
            lines.append(current_line)
            current_line = word + " "
    lines.append(current_line)
    
    y_text = (height - len(lines) * (FONT_SIZE + 5)) // 2
    for line in lines:
        draw.text((10, y_text), line, font=font, fill=TEXT_COLOR)
        y_text += FONT_SIZE + 5
    return img

def resize_contain(img, target_size):
    img_copy = img.copy()
    img_copy.thumbnail(target_size, Image.BICUBIC)
    bg = Image.new('RGB', target_size, (255, 255, 255))
    offset = ((target_size[0] - img_copy.width) // 2, (target_size[1] - img_copy.height) // 2)
    bg.paste(img_copy, offset)
    return bg

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # 初始化模型
    inference = DEADiff_Batch(
        "configs/inference_deadiff_control_512x512.yaml", 
        "pretrained/deadiff_v1.ckpt", 
        "pretrained/control_sd15_canny.pth"
    )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    row_images = []

    print(f"Start processing {len(lines)} groups...")

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        
        # 解析输入：ContentPath StylePath StylePrompt... SAMPrompt
        parts = line.split()
        if len(parts) < 4:
            print(f"Skipping line {idx+1}: Format error.")
            continue

        content_path = parts[0]
        style_path = parts[1]
        prompts = " ".join(parts[2:]) # 中间的是风格提示词
        style_prompt = prompts.split("/")[0] # 前面是风格提示词
        sam_prompt = prompts.split("/")[1] # 后面是 SAM 提示词


        print(f"\nProcessing Group {idx+1}:")
        print(f"  Content: {content_path}")
        print(f"  Style: {style_path}")
        print(f"  Style Prompt: {style_prompt}")
        print(f"  SAM Prompt: {sam_prompt}")

        if not os.path.exists(content_path) or not os.path.exists(style_path):
            print("  Error: Image path not found.")
            continue

        try:
            # 加载图片
            content_pil = Image.open(content_path).convert("RGB")
            style_pil = Image.open(style_path).convert("RGB")
            content_np = np.array(content_pil)
            style_np = np.array(style_pil)

            # 推理
            # 返回: 纯风格化图(raw), Mask, 最终合成图
            raw_style_pil, mask_pil, final_pil = inference.process_single_group(
                content_np, style_np, style_prompt, sam_prompt, SEED
            )

            # === 1. 保存单组结果 ===
            group_dir = os.path.join(OUTPUT_ROOT, f"group_{idx+1:03d}")
            os.makedirs(group_dir, exist_ok=True)
            
            # 保存文本信息
            with open(os.path.join(group_dir, "info.txt"), "w") as f:
                f.write(f"Style Prompt: {style_prompt}\nSAM Prompt: {sam_prompt}\n")
            
            # 保存图片
            content_pil.save(os.path.join(group_dir, "content.jpg"))
            style_pil.save(os.path.join(group_dir, "style.jpg"))
            raw_style_pil.save(os.path.join(group_dir, "raw_stylized.jpg"))
            if mask_pil: mask_pil.save(os.path.join(group_dir, "mask.png"))
            final_pil.save(os.path.join(group_dir, "final_result.jpg"))
            
            print(f"  Saved individual files to {group_dir}")

            # === 2. 制作图表行 ===
            # 列：[StylePrompt] [Content] [Style] [SamPrompt] [Mask] [FinalResult]
            
            # 准备图片单元格
            img_style_txt = draw_text_cell(style_prompt, COL_WIDTH_TEXT, CELL_HEIGHT)
            img_content   = resize_contain(content_pil, (COL_WIDTH_IMG, CELL_HEIGHT))
            img_style     = resize_contain(style_pil, (COL_WIDTH_IMG, CELL_HEIGHT))
            img_sam_txt   = draw_text_cell(sam_prompt, COL_WIDTH_TEXT, CELL_HEIGHT)
            
            if mask_pil:
                img_mask = resize_contain(mask_pil, (COL_WIDTH_IMG, CELL_HEIGHT))
            else:
                img_mask = Image.new('RGB', (COL_WIDTH_IMG, CELL_HEIGHT), (0,0,0))
                
            img_final = resize_contain(final_pil, (COL_WIDTH_IMG, CELL_HEIGHT))

            # 拼接
            margin = 5
            row_w = (COL_WIDTH_TEXT * 2) + (COL_WIDTH_IMG * 4) + (margin * 5)
            row_img = Image.new('RGB', (row_w, CELL_HEIGHT), (220, 220, 220))
            
            curr_x = 0
            # 1. Style Prompt
            row_img.paste(img_style_txt, (curr_x, 0)); curr_x += COL_WIDTH_TEXT + margin
            # 2. Content
            row_img.paste(img_content, (curr_x, 0)); curr_x += COL_WIDTH_IMG + margin
            # 3. Style
            row_img.paste(img_style, (curr_x, 0)); curr_x += COL_WIDTH_IMG + margin
            # 4. SAM Prompt
            row_img.paste(img_sam_txt, (curr_x, 0)); curr_x += COL_WIDTH_TEXT + margin
            # 5. Mask
            row_img.paste(img_mask, (curr_x, 0)); curr_x += COL_WIDTH_IMG + margin
            # 6. Result
            row_img.paste(img_final, (curr_x, 0))
            
            row_images.append(row_img)

        except Exception as e:
            print(f"  Processing Failed: {e}")
            import traceback
            traceback.print_exc()

    # === 3. 生成汇总长图 ===
    if row_images:
        print("\nGenerating summary chart...")
        total_h = sum([img.height for img in row_images]) + (len(row_images)-1)*5
        max_w = row_images[0].width
        
        summary = Image.new('RGB', (max_w, total_h), (255, 255, 255))
        y = 0
        for img in row_images:
            summary.paste(img, (0, y))
            y += img.height + 5
            
        summary.save(SUMMARY_FILE)
        print(f"Summary saved to {SUMMARY_FILE}")

if __name__ == "__main__":
    main()