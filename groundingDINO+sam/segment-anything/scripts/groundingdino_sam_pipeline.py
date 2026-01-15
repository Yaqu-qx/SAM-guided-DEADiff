"""
groundingdino_sam_pipeline.py
Requirements:
  - GroundingDINO repo installed (follow its README), weights downloaded.
  - segment-anything repo installed, SAM checkpoint downloaded.
  - on CUDA: set device="cuda" (or "cpu")
Usage: edit variables in `if __name__ == "__main__":` then run:
  python groundingdino_sam_pipeline.py
"""
import os
import cv2
import json
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

# GroundingDINO helpers (from their repo demo)
from groundingdino.util.inference import load_model, load_image, predict, annotate  # README example

def visualize_and_save(image_rgb, boxes, phrases, scores, masks_per_box, save_base):
    """
    image_rgb: HxWx3, uint8
    boxes: Nx4 (x0,y0,x1,y1)
    phrases: list of N strings (predicted phrase)
    scores: list/array of N floats (box scores)
    masks_per_box: list of arrays, each array shape (K, H, W) (K candidate masks per box)
    """
    h, w = image_rgb.shape[:2]
    overlay = image_rgb.copy()
    # draw boxes and best mask (take first candidate as default best) and annotate
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = [int(float(v)) for v in box]
        # draw box
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color=(0,255,0), thickness=2)

        # draw phrase + score
        caption = f"{phrases[i]}:{scores[i]:.2f}"
        cv2.putText(overlay, caption, (x0, max(0, y0-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # overlay masks (all candidates, different colors)
        candidates = masks_per_box[i]  # shape (K, H, W) boolean
        for k in range(candidates.shape[0]):
            mask = candidates[k].astype(bool)
            color = tuple(np.random.randint(0,180,size=3).tolist())  # darker color
            colored = np.zeros_like(image_rgb, dtype=np.uint8)
            for c in range(3):
                colored[..., c][mask] = color[c]
            overlay = cv2.addWeighted(overlay, 1.0, colored, 0.7, 0)

    # save overlay and original side-by-side
    combined = np.concatenate([image_rgb, overlay], axis=1)
    os.makedirs(os.path.dirname(save_base), exist_ok=True)
    cv2.imwrite(save_base + "_comparison.png", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    return

def run_pipeline_on_folder(
    image_dir,
    out_dir,
    grounding_cfg, grounding_checkpoint,
    sam_checkpoint, sam_model_type="vit_h",
    device="cuda",
    text_prompts=[],
    box_threshold=0.35,
    text_threshold=0.25,
    max_boxes_per_image=10
):
    os.makedirs(out_dir, exist_ok=True)
    masks_out = os.path.join(out_dir, "masks")
    overlays_out = os.path.join(out_dir, "overlay")
    os.makedirs(masks_out, exist_ok=True)
    os.makedirs(overlays_out, exist_ok=True)

    grounding_model = load_model(grounding_cfg, grounding_checkpoint)
    # load SAM
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    image_files = sorted(os.listdir(image_dir))
    for fname, prompt in zip(image_files, text_prompts):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image_path = os.path.join(image_dir, fname)
        image_source, image = load_image(image_path)  # load_image returns original + processed (per their helper)
        # 'image' is the numpy image for predict(...) in their util
        boxes, logits, phrases = predict(
            model=grounding_model,
            image=image,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        # boxes: (N,4), logits: (N, num_tokens?) - README returns useful info; phrases: list of strings
        # For simplicity: derive per-box scalar score from logits (e.g., max over tokens) or use provided box score if available.
        # Here we use a simple heuristic: if logits is (N, T), pick max across T
        box_scores = np.max(np.abs(logits), axis=1) if logits is not None else np.ones(len(boxes))
        # limit number of boxes
        selected_idx = np.argsort(-box_scores)[:max_boxes_per_image]

        masks_per_box = []
        selected_boxes = []
        selected_phrases = []
        selected_scores = []

        # read original RGB for SAM (load_image returns maybe a transform; ensure you have full-res rgb; try cv2)
        rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        predictor.set_image(rgb)

        for idx in selected_idx:
            box = boxes[idx]  # expected format [x0,y0,x1,y1] in pixel coords
            score = float(box_scores[idx])
            phrase = phrases[idx] if phrases is not None and len(phrases)>idx else text_prompt

            # prepare box for SAM: numpy array shape(1,4) float, format [x0,y0,x1,y1]
            box_np = np.array([box], dtype=float)

            # predict masks for this box; multimask_output True to get multiple candidates
            masks, mask_scores, logits_mask = predictor.predict(
                box=box_np,
                multimask_output=True
            )
            # masks shape: (K, H, W) boolean
            masks_per_box.append(masks)
            selected_boxes.append(box)
            selected_phrases.append(phrase)
            selected_scores.append(score)

            # save each candidate mask as PNG
            base = os.path.splitext(fname)[0]
            for k in range(masks.shape[0]):
                mask_img = (masks[k].astype(np.uint8) * 255)
                mask_path = os.path.join(masks_out, f"{base}_box{idx}_cand{k}.png")
                cv2.imwrite(mask_path, mask_img)

        # save info json
        info = {
            "boxes": [list(map(float,b)) for b in selected_boxes],
            "phrases": selected_phrases,
            "box_scores": selected_scores
        }
        with open(os.path.join(out_dir, os.path.splitext(fname)[0] + "_info.json"), "w") as f:
            json.dump(info, f, indent=2)

        # visualize overlay and save
        vis_base = os.path.join(overlays_out, os.path.splitext(fname)[0])
        visualize_and_save(rgb, selected_boxes, selected_phrases, selected_scores, np.array(masks_per_box), vis_base)

        print(f"[done] {fname}: selected {len(selected_boxes)} boxes")

if __name__ == "__main__":
    # --- 修改这几项 ---
    image_dir = "mini_dataset"
    out_dir = "output/grounded_sam"
    grounding_cfg = "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"   # repo config
    grounding_checkpoint = "../GroundingDINO/weights/groundingdino_swint_ogc.pth"
    sam_checkpoint = "checkpoint/sam_vit_h_4b8939.pth"
    sam_model_type = "vit_h"
    device = "cuda"  # or "cpu"
    prompt_file = "prompts.txt"

    with open(prompt_file, "r") as f:
        prompts = [line.strip() for line in f.readlines()]

    # 你要检测的文本 prompt（可以是多个词，用 '.' 分隔）
    # text_prompt = "doll . bag . mouse . cup . printer . slippers . pothos ."

    run_pipeline_on_folder(
        image_dir=image_dir,
        out_dir=out_dir,
        grounding_cfg=grounding_cfg,
        grounding_checkpoint=grounding_checkpoint,
        sam_checkpoint=sam_checkpoint,
        sam_model_type=sam_model_type,
        device=device,
        text_prompts=prompts,
        box_threshold=0.35,
        text_threshold=0.25,
        max_boxes_per_image=6
    )
