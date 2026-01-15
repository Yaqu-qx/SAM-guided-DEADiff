import os
import cv2
import numpy as np
import json
from segment_anything import sam_model_registry, SamPredictor

def overlay_masks_on_image(image, masks, color_list=None, alpha=0.7):
    overlay = image.copy()
    H, W = image.shape[:2]
    if color_list is None:
        color_list = [tuple(np.random.randint(0, 255, size=3).tolist()) for _ in range(len(masks))]
    for i, mask in enumerate(masks):
        color = color_list[i]

        colored = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            colored[..., c][mask] = color[c]

        overlay = cv2.addWeighted(overlay, 1.0, colored, alpha, 0)
    return overlay

def draw_prompt_points(overlay, point_coords, point_labels):
    for (x, y), label in zip(point_coords, point_labels):
        if label == 1:  # 前景点
            color = (255, 0, 0) 
        else:  # 背景点
            color = (0, 0, 255) 
        cv2.circle(overlay, (int(x), int(y)), radius=8, color=color, thickness=-1)
    return overlay

def run_multimask_on_folder(
    image_dir,
    save_dir,
    sam_checkpoint,
    model_type="vit_h",
    device="cuda",
    prompt_point_ratio=(0.5, 0.5, 0.34, 0.34),
    max_masks=3
):
    os.makedirs(save_dir, exist_ok=True)
    masks_out_dir = os.path.join(save_dir, "masks")
    overlay_out_dir = os.path.join(save_dir, "overlay")
    os.makedirs(masks_out_dir, exist_ok=True)
    os.makedirs(overlay_out_dir, exist_ok=True)

    # 加载 SAM + predictor
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)


    for fname in os.listdir(image_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(image_dir, fname)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print("无法读取图片:", img_path)
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        predictor.set_image(image_rgb)

        # 提示点（中心）  
        px = int(w * prompt_point_ratio[0])
        py = int(h * prompt_point_ratio[1])
        # px = 596
        # py = 1015
        point_coords = np.array([[px, py]])
        point_labels = np.array([1]) 

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )

        # 保存每个 mask 图
        base = os.path.splitext(fname)[0]
        # 可保存 scores
        info = {
            "scores": scores.tolist(),
        }
        for i in range(masks.shape[0]):
            mask_i = masks[i].astype(np.uint8) * 255
            mask_fname = f"{base}_mask{i}.png"
            cv2.imwrite(os.path.join(masks_out_dir, mask_fname), mask_i)

        with open(os.path.join(save_dir, base + "_info.json"), "w") as f:
            json.dump(info, f)

        overlay = overlay_masks_on_image(image_rgb, masks, alpha=0.75)
        overlay = draw_prompt_points(overlay, point_coords, point_labels)  # 把提示点画上去

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        overlay_fname = base + "_overlay.png"
        cv2.imwrite(os.path.join(overlay_out_dir, overlay_fname), overlay_bgr)

        print(f"处理 {fname} 完成，输出 {masks.shape[0]} 个 mask")

if __name__ == "__main__":
    # 你要改这几项
    image_dir = "../data1213_multi"
    save_dir = "output/hands-1213"
    sam_checkpoint = "checkpoint/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"  # 或 "cpu"

    run_multimask_on_folder(
        image_dir=image_dir,
        save_dir=save_dir,
        sam_checkpoint=sam_checkpoint,
        model_type=model_type,
        device=device,
        prompt_point_ratio=(0.5, 0.5, 0.34, 0.34),
        max_masks=3
    )
