# -*- coding: utf-8 -*-
"""
自然化补丁生成器（含 YOLO 标注导出 + Train/Val 双份输出）
- 控制台选择 1/2/3：低纹理 / 高纹理 / 混合
- 对同一张源图，执行两次补丁生成：
  * 第一次输出到 images/train 与 labels/train
  * 第二次输出到 images/val   与 labels/val
- YOLO 标签：class x_center y_center width height（归一化到 0~1）
  * 类别编码：低纹理=0，高纹理=1

依赖：opencv-python, numpy
"""

import os
import cv2
import glob
import math
import random
import numpy as np
from typing import Tuple, List

# ===================== 常量配置（按需修改） =====================

INPUT_DIR = "/root/autodl-tmp/ultralytics-main/datasets/image_road"            # 输入目录
OUTPUT_ROOT = "/root/autodl-tmp/ultralytics-main/datasets/final/natual"          # 数据集根目录
SAVE_MASK = False                       # 是否额外保存mask（在各自split目录下）
PATCHES_PER_IMAGE = 1                   # 每张图生成补丁数量（对train/val各自执行）
RANDOM_SEED_BASE = 20240904             # 基础随机种子（val 会在此基础上做偏移，确保两次不同）

# 补丁形状： "circle" | "square" | "squircle"
PATCH_SHAPE = "squircle"

# 补丁尺寸（相对于较短边的比例范围）
PATCH_SIZE_RATIO_MIN = 0.10
PATCH_SIZE_RATIO_MAX = 0.22

# 纹理阈值分位（挑选中心点时用）
LOW_TEXTURE_PERCENTILE = 0.30
HIGH_TEXTURE_PERCENTILE = 0.70

# 纹理图参数
TEXTURE_SMOOTH_K = 9
LAPLACIAN_KSIZE = 3

# 自然化生成参数
NOISE_OCTAVES = 4
NOISE_BASE_STD = 18.0
LOW_TEXTURE_NOISE_SCALE = 0.35
HIGH_TEXTURE_NOISE_SCALE = 1.0
COLOR_JITTER = 6.0
SEAMLESS_MODE = cv2.MIXED_CLONE  # 或 cv2.NORMAL_CLONE

# 如果本地找不到满足条件的中心点，最多随机尝试次数
MAX_RANDOM_TRIES = 200

# 类别定义（用于 YOLO 标签）
CLASS_ID_LOW = 0
CLASS_ID_HIGH = 0
CLASS_NAMES = ["patch"]  # 可选：写入 classes.txt

# ===============================================================


def ensure_dirs(root: str):
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        path = os.path.join(root, sub)
        os.makedirs(path, exist_ok=True)
    # 可选：写 classes.txt
    with open(os.path.join(root, "classes.txt"), "w", encoding="utf-8") as f:
        for name in CLASS_NAMES:
            f.write(name + "\n")


def load_images(input_dir: str):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(input_dir, e)))
    return sorted(files)


def texture_map_var_laplacian(gray: np.ndarray,
                              lap_ksize: int = 3,
                              smooth_k: int = 9) -> np.ndarray:
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=lap_ksize)
    k = (smooth_k if smooth_k % 2 == 1 else smooth_k + 1)
    mean = cv2.boxFilter(lap, ddepth=-1, ksize=(k, k), normalize=True)
    mean2 = cv2.boxFilter(lap * lap, ddepth=-1, ksize=(k, k), normalize=True)
    var = mean2 - mean * mean
    var = np.maximum(var, 0)
    vmin, vmax = np.percentile(var, 1), np.percentile(var, 99)
    var = np.clip((var - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    return var


def multiscale_noise(h: int, w: int, octaves: int = 4) -> np.ndarray:
    base = np.zeros((h, w), np.float32)
    rng = np.random.default_rng()
    for o in range(octaves):
        scale = 2 ** o
        small_h = max(2, h // scale)
        small_w = max(2, w // scale)
        small = rng.random((small_h, small_w)).astype(np.float32)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        weight = 1.0 / (2 ** o)
        base += weight * up
    base = base - base.min()
    base = base / max(base.max(), 1e-6)
    return base


def make_patch_mask(shape: str, size: Tuple[int, int]) -> np.ndarray:
    h, w = size
    mask = np.zeros((h, w), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ry, rx = max(h / 2.0 - 1, 1), max(w / 2.0 - 1, 1)

    if shape == "circle":
        dist = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        mask[dist <= 1.0] = 255
    elif shape == "square":
        pad = 0.07
        py, px = ry * (1 - pad), rx * (1 - pad)
        cond_y = (np.abs(yy - cy) <= py)
        cond_x = (np.abs(xx - cx) <= px)
        mask[cond_y & cond_x] = 255
    else:
        n = 4.0
        dist = (np.abs((yy - cy) / ry) ** n + np.abs((xx - cx) / rx) ** n)
        mask[dist <= 1.0] = 255

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(h, w) * 0.02)
    return np.clip(mask, 0, 255).astype(np.uint8)


def local_color_stats(img: np.ndarray, yx: Tuple[int, int], radius: int = 16):
    h, w = img.shape[:2]
    y, x = yx
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    patch = img[y0:y1, x0:x1, :]
    mean = patch.reshape(-1, 3).mean(axis=0)
    std = patch.reshape(-1, 3).std(axis=0) + 1e-6
    return mean, std


def synth_natural_texture(size: Tuple[int, int],
                          base_mean: np.ndarray,
                          base_std: np.ndarray,
                          noise_scale: float,
                          jitter: float) -> np.ndarray:
    h, w = size
    out = np.zeros((h, w, 3), np.float32)
    g = multiscale_noise(h, w, octaves=NOISE_OCTAVES)
    g = (g - 0.5) * 2.0
    rng = np.random.default_rng()
    for c in range(3):
        shift = rng.uniform(-0.2, 0.2)
        gain = rng.uniform(0.9, 1.1)
        chan = np.clip((g + shift) * gain, -1, 1)
        out[..., c] = chan
    out *= (NOISE_BASE_STD * noise_scale)
    out = (out - out.mean(axis=(0, 1), keepdims=True)) / (out.std(axis=(0, 1), keepdims=True) + 1e-6)
    out = out * base_std.reshape(1, 1, 3) + base_mean.reshape(1, 1, 3)
    jitter_vec = rng.uniform(-jitter, jitter, size=(1, 1, 3)).astype(np.float32)
    out = out + jitter_vec
    return np.clip(out, 0, 255).astype(np.uint8)


def select_center_by_texture(tex: np.ndarray,
                             mode: str,
                             patch_hw: Tuple[int, int]) -> Tuple[int, int] or None:
    h, w = tex.shape
    ph, pw = patch_hw
    y_margin, x_margin = ph // 2 + 2, pw // 2 + 2
    valid = np.zeros_like(tex, dtype=bool)
    valid[y_margin:h - y_margin, x_margin:w - x_margin] = True

    if mode == "low":
        thr = np.quantile(tex[valid], LOW_TEXTURE_PERCENTILE)
        mask = (tex <= thr) & valid
    elif mode == "high":
        thr = np.quantile(tex[valid], HIGH_TEXTURE_PERCENTILE)
        mask = (tex >= thr) & valid
    else:
        mask = valid

    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    idx = random.randrange(0, len(ys))
    return int(ys[idx]), int(xs[idx])


def yolo_normalized_bbox(cx: int, cy: int, pw: int, ph: int, img_w: int, img_h: int):
    """
    给定中心 (cx,cy) 和补丁宽高（像素），输出YOLO归一化 (x_c, y_c, w, h)
    注意：我们选点时已保证不越界，这里直接转换即可。
    """
    x_c = cx / img_w
    y_c = cy / img_h
    w_n = pw / img_w
    h_n = ph / img_h
    # 保底裁剪到 [0,1]
    x_c = min(max(x_c, 0.0), 1.0)
    y_c = min(max(y_c, 0.0), 1.0)
    w_n = min(max(w_n, 0.0), 1.0)
    h_n = min(max(h_n, 0.0), 1.0)
    return x_c, y_c, w_n, h_n


def place_patch_once(img_bgr: np.ndarray, mode: str):
    """
    在单张图上放置一个补丁。
    返回：
      out_img:   放置完成后的图
      mask_full: 与原图同尺寸的uint8 mask
      yolo_row:  (class_id, x_c, y_c, w, h) —— 归一化
    """
    h, w = img_bgr.shape[:2]
    short_side = min(h, w)
    patch_size = int(short_side * random.uniform(PATCH_SIZE_RATIO_MIN, PATCH_SIZE_RATIO_MAX))
    patch_h, patch_w = patch_size, patch_size

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    tex = texture_map_var_laplacian(gray, lap_ksize=LAPLACIAN_KSIZE, smooth_k=TEXTURE_SMOOTH_K)

    center = select_center_by_texture(tex, mode, (patch_h, patch_w))
    if center is None:
        for _ in range(5):
            patch_size = int(short_side * random.uniform(PATCH_SIZE_RATIO_MIN, PATCH_SIZE_RATIO_MAX))
            patch_h, patch_w = patch_size, patch_size
            center = select_center_by_texture(tex, mode, (patch_h, patch_w))
            if center is not None:
                break
        if center is None:
            return None

    cy, cx = center
    mask_local = make_patch_mask(PATCH_SHAPE, (patch_h, patch_w))
    mean, std = local_color_stats(img_bgr, (cy, cx), radius=max(8, patch_size // 8))
    noise_scale = LOW_TEXTURE_NOISE_SCALE if mode == "low" else HIGH_TEXTURE_NOISE_SCALE
    patch_tex = synth_natural_texture((patch_h, patch_w), mean, std, noise_scale, COLOR_JITTER)

    src = np.zeros_like(img_bgr)
    mask_full = np.zeros((h, w), np.uint8)

    y0, y1 = cy - patch_h // 2, cy + (patch_h - patch_h // 2)
    x0, x1 = cx - patch_w // 2, cx + (patch_w - patch_w // 2)
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(h, y1), min(w, x1)

    roi_h, roi_w = y1 - y0, x1 - x0
    src[y0:y1, x0:x1] = patch_tex[:roi_h, :roi_w]
    mask_full[y0:y1, x0:x1] = mask_local[:roi_h, :roi_w]

    center_pt = (cx, cy)
    try:
        out = cv2.seamlessClone(src, img_bgr, mask_full, center_pt, SEAMLESS_MODE)
    except cv2.error:
        alpha = (mask_full.astype(np.float32) / 255.0)[..., None]
        out = (alpha * src + (1 - alpha) * img_bgr).astype(np.uint8)

    # 由于我们选点防越界，bbox 就是补丁的矩形外接框
    x_c, y_c, w_n, h_n = yolo_normalized_bbox(cx, cy, patch_w, patch_h, w, h)
    # 旧：class_id = CLASS_ID_LOW if mode == "low" else CLASS_ID_HIGH
    class_id = 0  # 强制单类标注
    yolo_row = (class_id, x_c, y_c, w_n, h_n)
    return out, mask_full, yolo_row


def run_once_on_image(img_bgr: np.ndarray, choice_mode: str):
    """
    对单张图执行一次补丁流水线；支持多补丁。
    返回：out_img, mask_acc, yolo_rows(list)
    """
    work = img_bgr.copy()
    mask_acc = np.zeros(work.shape[:2], np.uint8)
    yolo_rows: List[Tuple[int, float, float, float, float]] = []

    for k in range(PATCHES_PER_IMAGE):
        # 确定本次补丁模式
        if choice_mode == "1":
            mode = "low"
        elif choice_mode == "2":
            mode = "high"
        else:
            mode = random.choice(["low", "high"])

        success = None
        for _ in range(MAX_RANDOM_TRIES):
            res = place_patch_once(work, mode)
            if res is not None:
                out, mask, yolo_row = res
                work = out
                mask_acc = cv2.bitwise_or(mask_acc, mask)
                yolo_rows.append(yolo_row)
                success = True
                break
        if not success:
            print(f"[!] 放置失败（模式 {mode}），跳过该补丁。")
            break

    return work, mask_acc, yolo_rows


def write_yolo_label(label_path: str, rows: List[Tuple[int, float, float, float, float]]):
    with open(label_path, "w", encoding="utf-8") as f:
        for (cid, xc, yc, w, h) in rows:
            f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def main():
    ensure_dirs(OUTPUT_ROOT)
    img_files = load_images(INPUT_DIR)
    if not img_files:
        print(f"[!] 输入目录为空：{INPUT_DIR}")
        return

    print("请选择生成类型：")
    print("  1. 低纹理区域自然化补丁（class=0）")
    print("  2. 高纹理区域自然化补丁（class=1）")
    print("  3. 混合类型（每个补丁随机低/高，按实际写不同class）")
    choice = input("输入 1/2/3 并回车：").strip()
    if choice not in {"1", "2", "3"}:
        print("[!] 非法选择，退出。")
        return

    print(f"[i] 形状: {PATCH_SHAPE}, 每图补丁数: {PATCHES_PER_IMAGE}")
    print(f"[i] 输入: {INPUT_DIR}")
    print(f"[i] 输出根目录: {OUTPUT_ROOT}")
    print("-----------------------------------------------------")

    for idx, f in enumerate(img_files):
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[!] 读取失败：{f}")
            continue

        base = os.path.splitext(os.path.basename(f))[0]

        # ============ 第一次：写入 train ============
        random.seed(RANDOM_SEED_BASE + idx * 2 + 0)
        np.random.seed(RANDOM_SEED_BASE + idx * 2 + 0)

        out_img_train, mask_train, yolo_rows_train = run_once_on_image(img, choice)

        img_train_path = os.path.join(OUTPUT_ROOT, "images/train", f"{base}_train.png")
        lab_train_path = os.path.join(OUTPUT_ROOT, "labels/train", f"{base}_train.txt")
        cv2.imwrite(img_train_path, out_img_train)
        write_yolo_label(lab_train_path, yolo_rows_train)
        if SAVE_MASK:
            cv2.imwrite(os.path.join(OUTPUT_ROOT, "images/train", f"{base}_train_mask.png"), mask_train)

        # ============ 第二次：写入 val ============ 
        random.seed(RANDOM_SEED_BASE + idx * 2 + 1)
        np.random.seed(RANDOM_SEED_BASE + idx * 2 + 1)

        out_img_val, mask_val, yolo_rows_val = run_once_on_image(img, choice)

        img_val_path = os.path.join(OUTPUT_ROOT, "images/val", f"{base}_val.png")
        lab_val_path = os.path.join(OUTPUT_ROOT, "labels/val", f"{base}_val.txt")
        cv2.imwrite(img_val_path, out_img_val)
        write_yolo_label(lab_val_path, yolo_rows_val)
        if SAVE_MASK:
            cv2.imwrite(os.path.join(OUTPUT_ROOT, "images/val", f"{base}_val_mask.png"), mask_val)

        print(f"[✓] {base}: train/val 均已输出（补丁数 train={len(yolo_rows_train)}, val={len(yolo_rows_val)}）")

    print("-----------------------------------------------------")
    print("[✓] 全部完成。数据集结构如下：")
    print(f"{OUTPUT_ROOT}/")
    print("  images/train/*.png")
    print("  images/val/*.png")
    print("  labels/train/*.txt")
    print("  labels/val/*.txt")
    print("  classes.txt")


if __name__ == "__main__":
    main()
