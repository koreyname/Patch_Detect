# -*- coding: utf-8 -*-
"""
对图片随机生成补丁并输出 YOLOv8 数据集（含进度条）
1=纯色补丁；2=磨平马赛克补丁（像素化，来自原图）；3=彩色噪点补丁

目录结构（自动创建）：
  DATASET_ROOT/
    images/train, images/val(可空)
    labels/train, labels/val(可空)

依赖：
  pip install pillow numpy tqdm
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Iterable
import random
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

# ===================== 固定参数（按需修改） =====================
# 输入图片目录
INPUT_DIR  = Path("/root/autodl-tmp/ultralytics-main/datasets/image_road")

# 数据集根目录（会在里面创建 images/train、labels/train 等）
DATASET_ROOT = Path("/root/autodl-tmp/ultralytics-main/dataset/final/pure2")

# 补丁尺寸（像素）
PATCH_WIDTH  = 48
PATCH_HEIGHT = 48

# “磨平马赛克”的像素块大小（越大越粗糙）
MOSAIC_TILE = 16

# 输出图片是否跟随源格式；True=沿用源扩展名，False=统一 .jpg
FOLLOW_SRC_FORMAT = True
JPEG_QUALITY = 95

# YOLO 类别模式：'single' -> 全部类0；'by_type' -> 0=纯色,1=马赛克,2=噪点
CLASS_MODE = "single"
# =============================================================

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

IMAGES_TRAIN = DATASET_ROOT / "images" / "val"
IMAGES_VAL   = DATASET_ROOT / "images" / "train"      # 可为空
LABELS_TRAIN = DATASET_ROOT / "labels" / "val"
LABELS_VAL   = DATASET_ROOT / "labels" / "train"      # 可为空

def _ensure_dirs() -> None:
    for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        d.mkdir(parents=True, exist_ok=True)
    if not INPUT_DIR.exists():
        print(f"[WARN] 输入目录不存在：{INPUT_DIR.resolve()} —— 已为你创建空目录。请放入图片后再运行。")
        INPUT_DIR.mkdir(parents=True, exist_ok=True)

def list_images(dirpath: Path) -> Iterable[Path]:
    for p in sorted(dirpath.glob("**/*")):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p

# --------------------- 补丁生成器 ---------------------
def gen_solid_patch(size: Tuple[int, int]) -> Image.Image:
    w, h = size
    color = tuple(random.randint(0, 255) for _ in range(3))
    arr = np.full((h, w, 3), color, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")

def gen_noise_patch(size: Tuple[int, int]) -> Image.Image:
    w, h = size
    arr = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")

def gen_content_mosaic_from_crop(crop: Image.Image, tile: int) -> Image.Image:
    """像素化：每个 tile 用该 tile 的平均色填充（磨平马赛克）。"""
    if crop.mode != "RGB":
        crop = crop.convert("RGB")
    arr = np.asarray(crop).astype(np.float32)  # (h,w,3)
    h, w = arr.shape[:2]
    tile = max(1, int(tile))

    out = np.empty_like(arr, dtype=np.uint8)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y2 = min(y + tile, h)
            x2 = min(x + tile, w)
            block = arr[y:y2, x:x2]
            mean_color = block.mean(axis=(0, 1))
            out[y:y2, x:x2] = mean_color
    return Image.fromarray(out, mode="RGB")

# --------------------- 位置与贴图 ---------------------
def choose_random_xy(W: int, H: int, w: int, h: int) -> Tuple[int, int]:
    if W < w or H < h:
        raise ValueError(f"图片太小（{W}x{H}）不足以放下补丁（{w}x{h}）。")
    x = random.randint(0, W - w)
    y = random.randint(0, H - h)
    return x, y

def paste_patch_at(img: Image.Image, patch: Image.Image, xy: Tuple[int, int]) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    out = img.copy()
    out.paste(patch, xy)  # 不透明覆盖
    return out

# --------------------- YOLO 标签 ---------------------
def class_id_for(choice: int) -> int:
    if CLASS_MODE == "by_type":
        # 1->0, 2->1, 3->2
        return {1: 0, 2: 1, 3: 2}[choice]
    return 0  # 单类：统一为0

def yolo_bbox(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[float, float, float, float]:
    cx = (x + w / 2.0) / W
    cy = (y + h / 2.0) / H
    nw = w / W
    nh = h / H
    # 保险起见裁剪到[0,1]
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)
    return cx, cy, nw, nh

def write_yolo_label(txt_path: Path, cls: int, bbox: Tuple[float, float, float, float]) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{cls} " + " ".join(f"{v:.6f}" for v in bbox) + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(line)

# --------------------- 交互与处理 ---------------------
def prompt_patch_type() -> int:
    msg = (
        "请选择补丁类型并回车：\n"
        "  1 = 纯色补丁\n"
        "  2 = 磨平马赛克补丁（像素化，来自原图）\n"
        "  3 = 彩色噪点补丁\n"
        ">> "
    )
    while True:
        choice = input(msg).strip()
        if choice in {"1", "2", "3"}:
            return int(choice)
        print("输入无效，请输入 1/2/3。")

def process_one_image(img_path: Path, choice: int) -> Tuple[Path, Path]:
    with Image.open(img_path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        W, H = im.size
        w, h = (PATCH_WIDTH, PATCH_HEIGHT)
        x, y = choose_random_xy(W, H, w, h)

        if choice == 1:
            patch = gen_solid_patch((w, h))
        elif choice == 2:
            crop = im.crop((x, y, x + w, y + h))
            patch = gen_content_mosaic_from_crop(crop, MOSAIC_TILE)
        elif choice == 3:
            patch = gen_noise_patch((w, h))
        else:
            raise ValueError("未知补丁类型")

        out_img = paste_patch_at(im, patch, (x, y))

    # 文件名与扩展名
    stem = f"{img_path.stem}_patched_type{choice}"
    if FOLLOW_SRC_FORMAT:
        ext = img_path.suffix.lower()
        if ext not in VALID_EXTS:
            ext = ".jpg"
    else:
        ext = ".jpg"

    img_out_path = IMAGES_TRAIN / f"{stem}{ext}"
    if ext in {".jpg", ".jpeg"}:
        out_img.save(img_out_path, quality=JPEG_QUALITY)
    else:
        out_img.save(img_out_path)

    # 写 YOLO 标签
    cls = class_id_for(choice)
    bbox = yolo_bbox(x, y, w, h, W, H)
    label_out_path = LABELS_TRAIN / f"{stem}.txt"
    write_yolo_label(label_out_path, cls, bbox)

    return img_out_path, label_out_path

def main() -> int:
    _ensure_dirs()
    images = list(list_images(INPUT_DIR))
    if not images:
        print(f"[INFO] 在 {INPUT_DIR.resolve()} 未找到图片（支持扩展名：{sorted(VALID_EXTS)}）。")
        return 0

    print(f"[INFO] 共找到 {len(images)} 张图片。")
    print(f"[INFO] 输出数据集根目录：{DATASET_ROOT.resolve()}")
    choice = prompt_patch_type()

    ok, fail = 0, 0
    progress = tqdm(images, desc="生成补丁与标签", unit="张", ncols=90)
    for img_path in progress:
        try:
            img_out, lbl_out = process_one_image(img_path, choice)
            ok += 1
            progress.set_postfix_str(f"{img_path.name} -> {img_out.name} | OK:{ok} FAIL:{fail}")
        except Exception as e:
            fail += 1
            progress.write(f"[ERROR] 处理失败：{img_path} —— {e}")

    progress.close()
    print(f"\n[SUMMARY] 成功：{ok}，失败：{fail}")
    print(f"[DATASET] images/train: {IMAGES_TRAIN.resolve()}")
    print(f"[DATASET] labels/train: {LABELS_TRAIN.resolve()}")
    print(f"[HINT] images/val 与 labels/val 保持为空或自行填充验证集。")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断。")
        sys.exit(130)