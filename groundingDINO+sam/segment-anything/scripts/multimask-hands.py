import os
import cv2
import numpy as np
import json
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def overlay_masks_on_image(image, masks_data, alpha=0.5):
    """
    将自动生成的 mask 覆盖在原图上。
    masks_data: List of dicts (SamAutomaticMaskGenerator 的输出格式)
    """
    overlay = image.copy()
    
    # 根据面积大小排序，这样小的 mask 会盖在大的上面，看得更清楚
    sorted_anns = sorted(masks_data, key=(lambda x: x['area']), reverse=True)

    # 只要有 mask 就开始画
    if len(sorted_anns) == 0:
        return overlay

    # 为每个 mask 生成随机颜色
    img_h, img_w = image.shape[:2]
    
    # 这一层用于累加颜色
    colored_mask_layer = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    # 这一层用于记录哪里有 mask (用于处理透明度)
    mask_alpha_layer = np.zeros((img_h, img_w), dtype=np.float32)

    for ann in sorted_anns:
        m = ann['segmentation'] # 布尔矩阵
        color_mask = np.random.randint(0, 255, (1, 3)).tolist()[0]
        
        # 将颜色填入对应的 mask 区域
        for c in range(3):
            colored_mask_layer[:, :, c] = np.where(m, color_mask[c], colored_mask_layer[:, :, c])
        
        # 记录 mask 区域
        mask_alpha_layer = np.where(m, 1.0, mask_alpha_layer)

    # 混合原图和颜色层
    # 只有在 mask 区域才应用 alpha 混合，其他区域保持原样
    # 简单的加权混合：
    colored_mask_layer = colored_mask_layer.astype(np.uint8)
    
    # 使用 cv2.addWeighted 全局混合可能会让背景变暗，这里手动混合 mask 区域
    # 逻辑：Result = Original * (1-alpha) + Color * alpha  (仅在 Mask 区域)
    
    mask_indices = mask_alpha_layer > 0
    overlay[mask_indices] = cv2.addWeighted(image[mask_indices], 1 - alpha, colored_mask_layer[mask_indices], alpha, 0)
    
    return overlay

def run_automatic_mask_generation(
    image_dir,
    save_dir,
    sam_checkpoint,
    model_type="vit_h",
    device="cuda",
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    min_mask_region_area=100
):
    """
    全自动分割主函数
    """
    # 1. 创建输出目录
    os.makedirs(save_dir, exist_ok=True)
    masks_vis_dir = os.path.join(save_dir, "overlay_vis")
    json_dir = os.path.join(save_dir, "metadata")
    single_masks_dir = os.path.join(save_dir, "single_masks")
    
    os.makedirs(masks_vis_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(single_masks_dir, exist_ok=True)

    print(f"正在加载模型 {model_type} 到 {device} ...")
    
    # 2. 加载 SAM 模型
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # 3. 初始化自动生成器
    # points_per_side: 采样点数。越大分割越细致（比如手套上的孔），但也越慢。
    # min_mask_region_area: 过滤掉小于这个像素数的微小噪点。
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=min_mask_region_area, 
    )

    # 4. 遍历文件夹
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("未找到图片文件！")
        return

    for fname in image_files:
        img_path = os.path.join(image_dir, fname)
        base_name = os.path.splitext(fname)[0]
        
        print(f"正在处理: {fname} ...")
        
        # 读取并转换图片
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"无法读取: {img_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # === 核心步骤：自动生成 Mask ===
        # 返回的是一个 list，每个元素是一个 dict
        masks = mask_generator.generate(image_rgb)
        
        print(f"  -> 生成了 {len(masks)} 个 Mask")

        # === 5. 保存可视化结果 (Overlay) ===
        overlay_img = overlay_masks_on_image(image_rgb, masks, alpha=0.6)
        overlay_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(masks_vis_dir, f"{base_name}_auto.jpg"), overlay_bgr)

        # === 6. 保存元数据 (JSON) ===
        # 需要把 numpy 类型转为 list 才能存 json
        metadata = []
        for i, m in enumerate(masks):
            # 将 mask 的 boolean 矩阵存为单独的 png，json 里只存文件名和 info
            mask_filename = f"{base_name}_id{i}.png"
            
            # 保存单独的 mask 图片 (黑白图: 0=背景, 255=前景)
            mask_uint8 = (m['segmentation'].astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(single_masks_dir, mask_filename), mask_uint8)

            metadata.append({
                "id": i,
                "area": int(m['area']),
                "bbox": [int(x) for x in m['bbox']], # x, y, w, h
                "predicted_iou": float(m['predicted_iou']),
                "stability_score": float(m['stability_score']),
                "mask_file": mask_filename
            })

        with open(os.path.join(json_dir, f"{base_name}.json"), "w") as f:
            json.dump(metadata, f, indent=4)

    print(f"\n全部处理完成！结果保存在: {save_dir}")

if __name__ == "__main__":
    # === 配置区域 ===
    
    # 输入图片文件夹
    IMAGE_DIR = "../data1213_multi" 
    
    # 结果保存路径
    SAVE_DIR = "output/hands_auto"
    
    # 权重路径
    CHECKPOINT = "checkpoint/sam_vit_h_4b8939.pth"
    
    # 模型类型 (vit_h, vit_l, vit_b)
    MODEL_TYPE = "vit_h"
    
    # 设备 (cuda 或 cpu)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 运行函数
    run_automatic_mask_generation(
        image_dir=IMAGE_DIR,
        save_dir=SAVE_DIR,
        sam_checkpoint=CHECKPOINT,
        model_type=MODEL_TYPE,
        device=DEVICE,
        # 如果觉得分得太碎（孔洞太多），把下面这个数改小（比如 16 或 8）
        # 如果觉得漏了细节，把这个数改大（比如 64）
        points_per_side=16, 
        min_mask_region_area=100 # 忽略小于 100 像素的噪点
    )