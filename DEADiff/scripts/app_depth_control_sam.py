import subprocess
import os
import cv2
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
from torch import autocast
import accelerate
from safetensors.torch import load_file
import gradio as gr

import k_diffusion as K
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torchvision import transforms as T
from ldm.controlnet.annotator.util import HWC3, resize_image
from ldm.controlnet.annotator.midas import MidasDetector

# ================= 配置区域 =================
# 复用你之前测试成功的路径
SAM_PYTHON_PATH = "/public/home/CS290U/panyq2025-CS290U/anaconda3/envs/grounding-sam-2/bin/python"
SAM_SCRIPT_PATH = "/public/home/CS290U/panyq2025-CS290U/project1/run_samdino_inference.py"
# ===========================================

apply_midas = MidasDetector()

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    print('loading done')
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    # 针对你的 12GB 显存优化
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


class DEADiff_DepthControl(object):
    def __init__(self,
                 config,
                 ckpt,
                 ckpt_controlnet):
        config = OmegaConf.load(f"{config}")
        config.model.params.control_stage_config.params.ckpt_path = ckpt_controlnet
        self.model = load_model_from_config(config, f"{ckpt}")

        self.model_wrap = K.external.CompVisDenoiser(self.model)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)

    # === 新增函数：调用外部 SAM ===
    def call_sam_external(self, image_input, prompt):
        # 1. 保存临时图片供 SAM 读取
        temp_input_path = "temp_sam_input_depth.jpg"
        temp_mask_path = "temp_sam_mask_depth.png"
        
        # image_input 可能是 numpy array (RGB)
        if isinstance(image_input, np.ndarray):
            Image.fromarray(image_input).save(temp_input_path)
        else:
            image_input.save(temp_input_path)

        # 2. 构建命令行命令
        cmd = [
            SAM_PYTHON_PATH,
            SAM_SCRIPT_PATH,
            "--input_image", temp_input_path,
            "--prompt", prompt,
            "--output_mask", temp_mask_path,
            "--device", "cpu" # 强制使用 CPU，避免与 DEADiff 抢显存或报错
        ]

        print("正在调用 SAM 环境进行分割...")
        # 3. 执行命令
        try:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("SAM 输出:", result.stdout)
        except subprocess.CalledProcessError as e:
            print("SAM 调用失败:", e.stderr)
            return None

        # 4. 读取生成的 Mask
        if os.path.exists(temp_mask_path):
            mask = cv2.imread(temp_mask_path, cv2.IMREAD_GRAYSCALE)
            return mask
        else:
            print("错误: SAM 脚本执行完成但没有生成 Mask 文件。")
            return None

    def generate(self, prompt, image_style_input, image_content_input, subject_text, batch_size, ddim, ddim_steps, scale, img_weight, seed, mask_prompt, mask_logic):
        accelerator = accelerate.Accelerator()
        device = accelerator.device
        seed_everything(seed)

        n_rows = 2 if batch_size >= 2 else 1
        if batch_size < 2:
            n_rows = 1
        elif batch_size < 5:
            n_rows = 2
        else:
            n_rows = 3
        prompts = batch_size * [prompt]

        precision_scope = autocast
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    all_samples = list()
                    uc = None
                    if scale != 1.0:
                        uc_encoder_hidden_states = self.model.get_learned_conditioning({'target_text':batch_size * ["over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"], 'subject_text': subject_text})
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    if subject_text == "style & content":
                        subject_text = ["style", "content"]
                    if subject_text == "None":
                        subject_text = None

                    ####### depth control #######
                    image_content_input = HWC3(image_content_input)
                    detected_map = apply_midas(resize_image(image_content_input, 384))
                    detected_map = HWC3(detected_map)
                    img = resize_image(image_content_input, 512)
                    H, W, C = img.shape

                    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

                    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
                    control = torch.stack([control for _ in range(batch_size)], dim=0)
                    control = rearrange(control, 'b h w c -> b c h w').clone()

                    ####### conditioning #######
                    c_encoder_hidden_states = self.model.get_learned_conditioning({
                            'target_text':prompts,
                            'inp_image': 2*(T.ToTensor()(Image.fromarray(image_style_input).convert('RGB').resize((224, 224)))-0.5).unsqueeze(0).repeat(batch_size, 1,1,1).to('cuda'),
                            'subject_text': [subject_text]*batch_size,
                        })
                    uc, c = uc_encoder_hidden_states, c_encoder_hidden_states
                    cond = {"c_concat": [control], "c_crossattn": c}
                    un_cond = {"c_concat": [control], "c_crossattn": [uc, uc]}
                    
                    shape = [4, H // 8, W // 8]

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
                        x = torch.randn([batch_size, *shape], device=device) * sigmas[0] # for GPU draw
                        
                        extra_args = {'cond': cond, 'uncond': un_cond, 'cond_scale': scale, 'img_weight': img_weight}
                        samples_ddim = K.sampling.sample_euler_ancestral(self.model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process)
                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = accelerator.gather(x_samples_ddim)
                    
                    if accelerator.is_main_process:
                        all_samples = [T.ToPILImage()(x_sample_ddim) for x_sample_ddim in x_samples_ddim]
                        grid = make_grid(x_samples_ddim, nrow=n_rows)
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        all_samples.append(Image.fromarray(grid.astype(np.uint8)))

        # ==================== 【新增逻辑】局部重绘合成 ====================
        if mask_prompt and mask_prompt.strip() != "":
            print(f"检测到 Mask 提示词: {mask_prompt}，模式: {mask_logic}")
            
            # 1. 决定用哪张图做 Mask 检测 (使用原图 image_content_input)
            mask = self.call_sam_external(image_content_input, mask_prompt)
            
            if mask is not None:
                if "背景" in mask_logic:
                    print("反转 Mask (处理背景)...")
                    mask = 255 - mask

                # 2. 获取风格化图片的尺寸作为标准
                # Depth 版本的输出通常是 512x512 或者根据原图比例调整后的尺寸
                stylized_img = np.array(all_samples[0])
                target_h, target_w = stylized_img.shape[:2]

                # 3. 调整 Mask 尺寸 -> 强制匹配风格图
                mask_resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

                # 4. 调整原图尺寸 -> 强制匹配风格图
                original_img_resized = cv2.resize(image_content_input, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                # 5. 归一化 Mask 并扩展维度
                mask_float = mask_resized.astype(np.float32) / 255.0
                mask_3d = np.expand_dims(mask_float, axis=2)
                
                # 6. 图像融合
                # Final = Stylized * Mask + Original * (1 - Mask)
                # new_samples = []
                for style_img in all_samples:
                    style_np = np.array(style_img)
                    final_composite = style_np * mask_3d + original_img_resized * (1 - mask_3d)
                    all_samples.append(Image.fromarray(final_composite.clip(0, 255).astype(np.uint8)))
                
                # 还可以把 mask 贴到最后方便调试
                all_samples.append(Image.fromarray(mask_resized))
                # all_samples = new_samples
        # ================================================================

        return all_samples


def interface(bd_inference):
    with gr.Row():
        with gr.Column():
            image_style_input = gr.Image(type="numpy", label="Input Style Image")
            image_content_input = gr.Image(type="numpy", label="Input Content Image")
            prompt = gr.Textbox(label="Prompt",value="best quality, extremely detailed")
            
            # === 新增 Mask 输入框 ===
            mask_text = gr.Textbox(label="局部重绘提示词 (例如: dress, bag)", value="dress")
            mask_logic = gr.Radio(
                choices=["foreground", "background"],
                value="foreground",
                label="SAM 控制模式"
            )
            # ======================

            subject_text = gr.Dropdown(
                    ["None", "style", "content", "style & content"], label="blip qformer文本输入", value="style"
                )
        with gr.Column():
            image_output = gr.Gallery(label="Result", ).style(columns=[2], rows=[2], object_fit="contain", height='auto')
            image_button = gr.Button("Generate")
    with gr.Row():
        batch_size = gr.Slider(1, 8, value=4, step=1, label="出图数量(batch_size)")
        ddim = gr.Radio(
                    choices=["ddim", "Euler a"],
                    label=f"Sampler",
                    interactive=True,
                    value="ddim",
                )
        ddim_steps = gr.Slider(10, 50, value=50, step=1, label="采样步数(Steps)", info="Choose between 10 and 50")
        scale = gr.Slider(5, 15, value=8, step=1, label="描述词关联度(CFG scale)", info="Choose between 5 and 15")
        img_weight = gr.Slider(0, 2, value=1.0, step=0.1, label="img embedding加权权重", info="Choose between 0 and 1")
        seed = gr.Number(value=-1,minimum=-1,step=1,label="随机种子(Seed)",info="input -1 for random generation")

    inputs=[
            prompt,
            image_style_input,
            image_content_input,
            subject_text,
            batch_size,
            ddim,
            ddim_steps,
            scale,
            img_weight,
            seed,
            mask_text, # <--- 记得加这个
            mask_logic
        ]
    
    image_button.click(bd_inference.generate, inputs=inputs, outputs=image_output)


if __name__ == "__main__":
    inference = DEADiff_DepthControl("configs/inference_deadiff_control_512x512.yaml", "pretrained/deadiff_v1.ckpt", "pretrained/control_sd15_depth.pth")
    with gr.Blocks() as demo:
        gr.HTML(
            """
            <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
            <h1 style="font-weight: 900; font-size: 3rem; margin: 0rem">
                DEADiff (Depth Control + Local Stylization)
            </h1>
            </div>
            """)
        gr.Markdown("Create images in any style of a reference image conditioned on the depth map of one content image using this demo.")
        interface(inference)
    demo.queue(max_size=3)
    demo.launch(server_name="0.0.0.0", server_port=8732)