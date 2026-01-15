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

# ================= 默认配置 =================
DEFAULT_DATASET_ROOT = "OmniConsistency_dataset"  
OUTPUT_FOLDER_NAME = "output-deadiff"     

BATCH_SIZE = 1           
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

    def generate(self, prompt, image_input, reference_type, batch_size, ddim_steps, scale, img_weight, seed):
        """
        reference_type: "style" or "content"
        """
        accelerator = accelerate.Accelerator()
        device = accelerator.device
        seed_everything(seed)
        prompts = batch_size * [prompt]
        
        with torch.no_grad():
            with autocast("cuda"):
                with self.model.ema_scope():
                    # 根据 mode 参数决定提取图像的什么特征
                    subject_text = [reference_type] * batch_size
                    
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
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description="DEADiff Batch Processing with Flexible Input")
    parser.add_argument("--mode", type=str, required=True, choices=["img_is_style", "img_is_content"],
                        help="模式选择: 'img_is_style' (提取图片风格) 或 'img_is_content' (提取图片内容)")
    parser.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET_ROOT, help="数据集根目录")
    parser.add_argument("--global_image_path", type=str, default=None, 
                        help="[可选] 全局输入图片路径。如果指定，所有生成都将强制使用这张图作为参考，忽略数据集src中的图片。")
    
    args = parser.parse_args()
    
    dataset_root = args.dataset_root
    mode = args.mode
    global_img_path = args.global_image_path

    # 2. 确定 DEADiff 的 subject_text 参数
    if mode == "img_is_style":
        ref_type = "style"
        mode_desc = "[参考图 = 风格] | [提示词 = 内容]"
    else:
        ref_type = "content"
        mode_desc = "[参考图 = 内容] | [提示词 = 风格]"

    print("="*60)
    print(f"运行模式: {mode_desc}")
    if global_img_path:
        print(f"全局参考图已启用: {global_img_path}")
        print("注意: 数据集中的 'src' 图片将被忽略，仅读取 'caption' 文本。")
        
        if not os.path.exists(global_img_path):
            print(f"Error: Global image path '{global_img_path}' not found!")
            return
    else:
        print("参考图来源: 对应数据集子文件夹中的 src 图片")
    print("="*60)

    if not os.path.exists(dataset_root):
        print(f"Error: Dataset root '{dataset_root}' not found.")
        return

    # 3. 初始化模型
    print("Initializing Model...")
    inference = DEADiff("configs/inference_deadiff_512x512.yaml", "pretrained/deadiff_v1.ckpt")

    # 如果有全局图，预先加载并转为 numpy，避免重复 I/O
    global_img_np = None
    if global_img_path:
        try:
            global_img_pil = Image.open(global_img_path).convert("RGB")
            global_img_np = np.array(global_img_pil)
        except Exception as e:
            print(f"Error loading global image: {e}")
            return
    
    # 4. 遍历数据集
    # style_dirs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    # style_dirs.sort()
    style_dirs = ['Macaron']

    print(f"Found {len(style_dirs)} sub-folders in dataset.")

    for style_idx, style_name in enumerate(style_dirs):
        style_dir_path = os.path.join(dataset_root, style_name)
        
        src_dir = os.path.join(style_dir_path, "src")
        caption_dir = os.path.join(style_dir_path, "caption")
        
        # 为了区分普通跑和全局覆盖跑，建议修改输出文件夹名字（可选，这里保持一致也可以）
        if global_img_path:
            output_dir = os.path.join(style_dir_path, f"{OUTPUT_FOLDER_NAME}_global_override")
        else:
            output_dir = os.path.join(style_dir_path, OUTPUT_FOLDER_NAME)

        if not os.path.exists(src_dir) or not os.path.exists(caption_dir):
            continue

        os.makedirs(output_dir, exist_ok=True)
        
        # 我们依然遍历 src 里的文件列表，以此作为“任务列表”的基准
        # 这样能保证生成的数量和文件名与数据集结构一致
        image_files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        image_files.sort()

        print(f"\nProcessing Folder [{style_idx+1}/{len(style_dirs)}]: {style_name} ({len(image_files)} tasks)")

        for img_file in image_files:
            # 获取 Prompt
            base_name = os.path.splitext(img_file)[0]
            txt_file = base_name + ".txt"
            txt_path = os.path.join(caption_dir, txt_file)

            if not os.path.exists(txt_path):
                print(f"  Warning: Missing caption for {img_file}")
                continue

            with open(txt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()

            # === 核心逻辑分支 ===
            try:
                if global_img_np is not None:
                    # 分支A：使用全局指定的图
                    input_np = global_img_np
                    source_desc = "GlobalImage"
                else:
                    # 分支B：使用数据集 src 里的图
                    img_path = os.path.join(src_dir, img_file)
                    style_pil = Image.open(img_path).convert("RGB")
                    input_np = np.array(style_pil)
                    source_desc = img_file

                print(f"  -> [{source_desc}] + '{prompt[:20]}...'")

                # 调用生成
                result_image = inference.generate(
                    prompt=prompt,
                    image_input=input_np,
                    reference_type=ref_type,
                    batch_size=BATCH_SIZE,
                    ddim_steps=STEPS,
                    scale=8.0,
                    img_weight=1.0,
                    seed=SEED
                )

                # 保存
                save_name = os.path.splitext(img_file)[0] + ".png"
                save_path = os.path.join(output_dir, save_name)
                result_image.save(save_path)
                
            except Exception as e:
                print(f"  Failed: {e}")
                import traceback
                traceback.print_exc()

    print("\nAll processing completed!")

if __name__ == "__main__":
    process_dataset()