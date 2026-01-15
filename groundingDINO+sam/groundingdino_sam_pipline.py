import argparse
import os
import sys

import numpy as np
import json
import torch
from PIL import Image

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import matplotlib.pyplot as plt


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    # filter output
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0
    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{'value': 0, 'label': 'background'}]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)

def parse_prompts_file(prompts_file_path):
    mapping = {}
    with open(prompts_file_path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            # 支持中文冒号和英文冒号
            sep = None
            if "：" in line:
                sep = "："
            elif ":" in line:
                sep = ":"
            if sep is None:
                raise ValueError(f"prompts 文件第 {i} 行格式不正确（找不到 ':' 或 '：'）：{line}")
            key, prompt = line.split(sep, 1)
            key = key.strip().strip('"').strip("'")
            prompt = prompt.strip()
            if not key:
                raise ValueError(f"prompts 文件第 {i} 行缺少文件名 key：{line}")
            mapping[key] = prompt
    return mapping

if __name__ == "__main__":
    config = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
    grounded_checkpoint = 'GroundingDINO/weights/groundingdino_swint_ogc.pth'
    sam_version = 'vit_h'
    sam_checkpoint = 'segment-anything/checkpoint/sam_vit_h_4b8939.pth'
    input_folder = 'segment-anything/mini_dataset'
    prompts_file = 'prompts.txt'
    output_dir = 'outputs/groundingdino-sam'
    device = 'cpu'
    box_threshold = 0.3
    text_threshold = 0.25
    bert_base_uncased_path = 'bert-base-uncased'

    # 读取 prompts.txt
    prompts_map = parse_prompts_file(prompts_file)

    image_files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    # load model only once
    model = load_model(config, grounded_checkpoint, bert_base_uncased_path, device=device)
    predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

    os.makedirs(output_dir, exist_ok=True)

    # 遍历文件夹下所有图片
    for fname in image_files:
        image_path = os.path.join(input_folder, fname)
        text_prompt = prompts_map[os.path.splitext(fname)[0]]
        print(f"Processing {image_path}...")

        # 每张图片单独保存到子目录
        img_out_dir = os.path.join(output_dir, os.path.splitext(fname)[0])
        os.makedirs(img_out_dir, exist_ok=True)

        # load image
        image_pil, image = load_image(image_path)
        image_pil.save(os.path.join(img_out_dir, "raw_image.jpg"))

        # grounding dino
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, device=device
        )

        # SAM
        image_cv = cv2.imread(image_path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_cv)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_cv.shape[:2]).to(device)

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(device),
            multimask_output=False,
        )

        # 可视化保存
        plt.figure(figsize=(10, 10))
        plt.imshow(image_cv)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)

        plt.axis('off')
        plt.savefig(os.path.join(img_out_dir, "grounded_sam_output.jpg"), bbox_inches="tight", dpi=300, pad_inches=0.0)

        save_mask_data(img_out_dir, masks, boxes_filt, pred_phrases)
