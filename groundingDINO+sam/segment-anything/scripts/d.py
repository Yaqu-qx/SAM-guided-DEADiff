import cv2
import numpy as np

def find_fingerpad_point(img_path):
    # ---- 1. 读取图像 ----
    img = cv2.imread(img_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---- 2. 灰度 + 阈值，提取深色手套区域 ----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 深色区域 → 白色（255），背景 → 黑色
    _, th = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # ---- 3. 寻找轮廓 ----
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ---- 4. 找左半边最大的轮廓（即主要的手指区域）----
    h, w = gray.shape
    best = None
    best_area = 0

    for c in cnts:
        x, y, wc, hc = cv2.boundingRect(c)
        if x < w // 2:                 # 只看左半部分（手套所在）
            a = cv2.contourArea(c)
            if a > best_area:
                best_area = a
                best = c

    # ---- 5. 计算轮廓质心（指腹点）----
    M = cv2.moments(best)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # ---- 6. 在图中标红点 ----
    out = rgb.copy()
    cv2.circle(out, (cx, cy), 10, (255, 0, 0), -1)

    # ---- 7. 保存或显示结果 ----
    cv2.imwrite("fingerpad_detected.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    # 输出点坐标
    return cx, cy, out


# ====== 使用示例 ======
img_path = "./hands_dataset/0.jpg"  # 换成你的路径
x, y, out_img = find_fingerpad_point(img_path)
print("指腹中心点坐标:", (x, y))
