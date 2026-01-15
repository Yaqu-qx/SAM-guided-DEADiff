import os
import cv2
import numpy as np

# 输入路径
image_dir = "mini_dataset"           
mask_root = "output/masks"              
save_dir = "output/colored_mask2"      
os.makedirs(save_dir, exist_ok=True)

def overlay_masks(image, mask_dir):
    overlay = image.copy()
    count = 0
    for f in os.listdir(mask_dir):
        count += 1
        if not f.endswith(".png"):
            continue
        mask = cv2.imread(os.path.join(mask_dir, f), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        # 随机颜色
        color = np.random.randint(0, 255, size=3, dtype=np.uint8)
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)

        if count >= 10:
            break
    return overlay


for img_file in os.listdir(image_dir):
    if not (img_file.endswith(".jpg") or img_file.endswith(".png") or img_file.endswith(".jpeg")):
        continue
    
    img_path = os.path.join(image_dir, img_file)
    image = cv2.imread(img_path)
    if image is None:
        print(f"跳过无法读取的图片: {img_file}")
        continue
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    base_name = os.path.splitext(img_file)[0]
    mask_dir = os.path.join(mask_root, base_name)

    if not os.path.isdir(mask_dir):
        print(f"未找到掩码文件夹: {mask_dir}，跳过该图片")
        continue

    overlay = overlay_masks(image, mask_dir)

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    save_path = os.path.join(save_dir, base_name + "_overlay.png")
    cv2.imwrite(save_path, overlay_bgr)
    print(f"保存结果: {save_path}")
