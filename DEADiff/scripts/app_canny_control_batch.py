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
from ldm.controlnet.annotator.util import HWC3, resize_image
from ldm.controlnet.annotator.canny import CannyDetector

# ================= 配置区域 =================
INPUT_FILE = "myTestSet1220/inputs_prompt_style_pairs.txt"              # 输入文件路径
OUTPUT_ROOT = "outputs/app_canny_batch"    # 输出根目录

BATCH_SIZE = 1                       # 每组生成 1 张图
STEPS = 50                           # 采样步数
SEED = 42                            # 随机种子
CFG_SCALE = 8.0                      # 提示词相关性
IMG_WEIGHT = 1.0                     # 风格权重
# ===========================================

apply_canny = CannyDetector()

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

class DEADiff_CannyControl(object):
    def __init__(self, config, ckpt, ckpt_controlnet):
        config = OmegaConf.load(f"{config}")
        config.model.params.control_stage_config.params.ckpt_path = ckpt_controlnet
        self.model = load_model_from_config(config, f"{ckpt}")
        self.model_wrap = K.external.CompVisDenoiser(self.model)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)

    def generate(self, prompt, image_style_input, image_content_input, subject_text, batch_size, ddim_steps, scale, img_weight, seed):
        accelerator = accelerate.Accelerator()
        device = accelerator.device
        seed_everything(seed)

        prompts = batch_size * [prompt]

        with torch.no_grad():
            with autocast("cuda"):
                with self.model.ema_scope():
                    # 默认使用 style 模式提取参考图特征
                    if subject_text is None: subject_text = "style"
                    
                    c_encoder_hidden_states = self.model.get_learned_conditioning({
                            'target_text':prompts,
                            'inp_image': 2*(T.ToTensor()(Image.fromarray(image_style_input).convert('RGB').resize((224, 224)))-0.5).unsqueeze(0).repeat(batch_size, 1,1,1).to('cuda'),
                            'subject_text': [subject_text]*batch_size,
                        })
                    
                    uc_encoder_hidden_states = self.model.get_learned_conditioning({
                        'target_text': batch_size * [""],
                        'subject_text': [subject_text]*batch_size
                    })

                    uc, c = uc_encoder_hidden_states, c_encoder_hidden_states

                    ####### ControlNet 处理 #######
                    img = resize_image(HWC3(image_content_input), 384)
                    H, W, C = img.shape

                    detected_map = apply_canny(img, 100, 200)
                    detected_map = HWC3(detected_map)
                    
                    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
                    control = torch.stack([control for _ in range(batch_size)], dim=0)
                    control = rearrange(control, 'b h w c -> b c h w').clone()

                    cond = {"c_concat": [control], "c_crossattn": c}
                    un_cond = {"c_concat": [control], "c_crossattn": [uc, uc]}
                    
                    shape = [4, H // 8, W // 8]

                    sampler = DDIMSampler(self.model)
                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                     conditioning=cond,
                                                     batch_size=batch_size,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond,
                                                     img_weight=img_weight)
                    
                    x_samples = self.model.decode_first_stage(samples_ddim)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    
                    # 返回第一张生成的图（因为 batch_size=1）
                    x_sample = T.ToPILImage()(x_samples[0])
                    return x_sample

# === 工具函数：清理文件名 ===
def sanitize_filename(prompt, max_length=50):
    safe_name = "".join([c for c in prompt if c.isalnum() or c in " -_"])
    safe_name = safe_name.strip().replace(" ", "_")
    return safe_name[:max_length]

def process_batch():
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到输入文件 {INPUT_FILE}")
        return

    # 1. 初始化模型
    inference = DEADiff_CannyControl(
        "configs/inference_deadiff_control_512x512.yaml", 
        "pretrained/deadiff_v1.ckpt", 
        "pretrained/control_sd15_canny.pth"
    )

    # 2. 读取任务列表
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"开始批量处理，共 {len(lines)} 组数据...")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 3. 循环处理
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        parts = line.split()
        if len(parts) < 3:
            print(f"跳过格式错误行: {line}")
            continue

        # 解析输入: Content_Path  Style_Path  Prompt
        content_path = parts[0]
        style_path = parts[1]
        prompt = " ".join(parts[2:])

        print(f"\n[{idx+1}/{len(lines)}] 处理中: {prompt[:30]}...")

        if not os.path.exists(content_path) or not os.path.exists(style_path):
            print(f"  错误: 图片路径不存在，跳过。")
            continue

        try:
            # 加载图片并转为 Numpy (DEADiff 内部逻辑需要 Numpy)
            content_pil = Image.open(content_path).convert("RGB")
            style_pil = Image.open(style_path).convert("RGB")
            
            content_np = np.array(content_pil)
            style_np = np.array(style_pil)

            # 推理
            result_img = inference.generate(
                prompt=prompt,
                image_style_input=style_np,
                image_content_input=content_np,
                subject_text="style",  # 默认提取参考图的 Style
                batch_size=BATCH_SIZE,
                ddim_steps=STEPS,
                scale=CFG_SCALE,
                img_weight=IMG_WEIGHT,
                seed=SEED
            )

            # === 保存结果 ===
            # 创建独立文件夹: outputs/batch_canny/001_prompt_text
            folder_name = f"{idx+1:03d}_{sanitize_filename(prompt)}"
            save_dir = os.path.join(OUTPUT_ROOT, folder_name)
            os.makedirs(save_dir, exist_ok=True)

            # 1. 保存 Prompt
            with open(os.path.join(save_dir, "prompt.txt"), "w", encoding="utf-8") as f:
                f.write(prompt)

            # 2. 保存原图 (Content)
            content_pil.save(os.path.join(save_dir, "content.jpg"))

            # 3. 保存风格图 (Style)
            style_pil.save(os.path.join(save_dir, "style_ref.jpg"))

            # 4. 保存结果图
            result_img.save(os.path.join(save_dir, "result.png"))

            print(f"  -> 结果已保存至: {save_dir}")

        except Exception as e:
            print(f"  处理失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n批量处理全部完成！")

if __name__ == "__main__":
    process_batch()