import os
# 1. 强制屏蔽显卡，确保 CPU 运行
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from contextlib import nullcontext
# 即使不用 cv2.resize，我们后面还是需要 cv2 做其他处理，所以保留导入
import cv2 

import k_diffusion as K
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torchvision import transforms as T
from ldm.controlnet.annotator.util import HWC3
from ldm.controlnet.annotator.canny import CannyDetector

# 初始化 Canny 检测器
apply_canny = CannyDetector()

def load_model_from_config(config, ckpt, verbose=False):
    print(f"[Info] Loading model from {ckpt} to CPU...")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    print('[Info] Model loading done')
    
    model.cpu()
    model.eval()
    return model

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale, img_weight):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        x_in = x_in.float()
        sigma_in = sigma_in.float()
        
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
                    if c_tmp is None:
                        cond_in_temp.append(None)
                    else:
                        cond_in_temp.append(torch.cat([uc_tmp, c_tmp]))
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

    def generate(self, prompt, image_style_input, image_content_input, subject_text, batch_size, ddim, ddim_steps, scale, img_weight, seed):
        if seed == -1:
            seed = np.random.randint(0, 2147483647)
        seed_everything(seed)
        print(f"[Info] Generating with seed: {seed}")

        device = torch.device("cpu")
        
        n_rows = 2 if batch_size >= 2 else 1
        if batch_size < 2: n_rows = 1
        elif batch_size < 5: n_rows = 2
        else: n_rows = 3
        prompts = batch_size * [prompt]

        with torch.no_grad():
            with nullcontext():
                with self.model.ema_scope():
                    # --- 【核心修复：弃用 cv2.resize，改用 PIL】 ---
                    try:
                        # 1. 统一转换为 PIL Image 对象 (这是最安全的数据容器)
                        if isinstance(image_content_input, np.ndarray):
                            pil_content = Image.fromarray(image_content_input).convert("RGB")
                        else:
                            pil_content = image_content_input.convert("RGB")

                        # 2. 计算缩放尺寸 (逻辑与之前相同，只是操作对象变了)
                        resolution = 384
                        W_raw, H_raw = pil_content.size # PIL 是 (W, H)
                        k = float(resolution) / min(H_raw, W_raw)
                        H_target = int(np.round(H_raw * k / 64.0)) * 64
                        W_target = int(np.round(W_raw * k / 64.0)) * 64
                        
                        print(f"[Info] Resizing content image to ({W_target}, {H_target}) using PIL...")
                        
                        # 3. 使用 PIL 进行缩放 (这里不会报错！)
                        # 使用 LANCZOS 算法保证质量
                        pil_resized = pil_content.resize((W_target, H_target), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
                        
                        # 4. 缩放完成后，再转回 Numpy 给 Canny 使用
                        img = np.array(pil_resized)
                        
                        # 双重保险：确保是 uint8 和连续内存
                        img = img.astype(np.uint8)
                        img = np.ascontiguousarray(img)

                    except Exception as e:
                        print(f"[Error] Image processing failed: {e}")
                        import traceback
                        traceback.print_exc()
                        return []
                    
                    # 5. Canny 边缘检测
                    detected_map = apply_canny(img, 100, 200)
                    detected_map = HWC3(detected_map)
                    
                    control = torch.from_numpy(detected_map.copy()).float().to(device) / 255.0
                    control = torch.stack([control for _ in range(batch_size)], dim=0)
                    control = rearrange(control, 'b h w c -> b c h w').clone()

                    # 6. 处理 Style Image
                    if isinstance(image_style_input, Image.Image):
                        style_pil = image_style_input.convert('RGB')
                    else:
                        style_pil = Image.fromarray(image_style_input).convert('RGB')
                        
                    style_pil = style_pil.resize((224, 224))
                    style_tensor = 2 * (T.ToTensor()(style_pil) - 0.5)
                    style_tensor = style_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)

                    # 7. 设置条件
                    uc_encoder_hidden_states = None
                    if scale != 1.0:
                        neg_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"
                        uc_encoder_hidden_states = self.model.get_learned_conditioning({
                            'target_text': batch_size * [neg_prompt], 
                            'subject_text': subject_text
                        })
                    
                    if subject_text == "style & content": subject_text = ["style", "content"]
                    if subject_text == "None": subject_text = None

                    c_encoder_hidden_states = self.model.get_learned_conditioning({
                            'target_text': prompts,
                            'inp_image': style_tensor,
                            'subject_text': [subject_text]*batch_size,
                        })
                    
                    uc, c = uc_encoder_hidden_states, c_encoder_hidden_states
                    cond = {"c_concat": [control], "c_crossattn": c}
                    un_cond = {"c_concat": [control], "c_crossattn": [uc, uc]}
                    
                    shape = [4, H_target // 8, W_target // 8]

                    # 8. 开始采样
                    print(f"[Info] Starting sampling with {ddim_steps} steps (this will be slow on CPU)...")
                    if ddim == 'ddim':
                        sampler = DDIMSampler(self.model)
                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                         conditioning=cond,
                                                         batch_size=batch_size,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=un_cond,
                                                         img_weight=img_weight)
                    else:
                        sigmas = self.model_wrap.get_sigmas(ddim_steps)
                        x = torch.randn([batch_size, *shape], device=device) * sigmas[0]
                        extra_args = {'cond': cond, 'uncond': un_cond, 'cond_scale': scale, 'img_weight': img_weight}
                        samples_ddim = K.sampling.sample_euler_ancestral(self.model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=False)
                    
                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    
                    all_samples = [T.ToPILImage()(x_sample_ddim) for x_sample_ddim in x_samples_ddim]
                    
        return all_samples

# ==========================================
#        配置与执行区域 (Main)
# ==========================================
if __name__ == "__main__":
    # --- 1. 配置路径和参数 ---
    config_path = "configs/inference_deadiff_control_512x512.yaml"
    ckpt_path = "pretrained/deadiff_v1.ckpt"
    controlnet_path = "pretrained/control_sd15_canny.pth"
    
    # 输入图片路径 (请确保这些文件存在)
    style_image_path = "assets/style_images/04(1).jpeg"      
    content_image_path = "assets/content_images/01.jpeg"  
    
    # 输出保存路径
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成参数
    prompt = "best quality, extremely detailed, cyberpunk style" 
    subject_text = "style" 
    batch_size = 1       
    ddim_steps = 20      
    scale = 7.5          
    seed = 42            
    
    # --- 2. 检查文件是否存在 ---
    if not os.path.exists(style_image_path):
        print(f"Error: Style image not found at {style_image_path}")
        exit()
    if not os.path.exists(content_image_path):
        print(f"Error: Content image not found at {content_image_path}")
        exit()

    # --- 3. 加载模型 ---
    print("Initialize Model...")
    inference = DEADiff_CannyControl(config_path, ckpt_path, controlnet_path)
    
    # --- 4. 读取图片 ---
    # 我们直接读取为 PIL 对象传进去，避免在 Main 里转 Numpy 又出问题
    print("Loading images...")
    pil_style = Image.open(style_image_path).convert("RGB")
    pil_content = Image.open(content_image_path).convert("RGB")

    # --- 5. 执行生成 ---
    print("Running generation...")
    start_time = time.time()
    
    results = inference.generate(
        prompt=prompt,
        image_style_input=pil_style,   # 直接传 PIL 对象
        image_content_input=pil_content, # 直接传 PIL 对象
        subject_text=subject_text,
        batch_size=batch_size,
        ddim="ddim",
        ddim_steps=ddim_steps,
        scale=scale,
        img_weight=1.0,
        seed=seed
    )
    
    end_time = time.time()
    print(f"Generation finished in {end_time - start_time:.2f} seconds.")

    # --- 6. 保存结果 ---
    for i, img in enumerate(results):
        save_path = os.path.join(output_dir, f"result_seed{seed}_{i}.png")
        img.save(save_path)
        print(f"Image saved to: {save_path}")