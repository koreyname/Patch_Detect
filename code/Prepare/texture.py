# -*- coding: utf-8 -*-
"""
按纹理复杂度（Shannon熵）检测低纹理区域：
1) 生成半透明绿色覆盖的图片（images/train）
2) 导出低纹理区域的 YOLO 框（labels/train，多框）
3) 目录结构符合 YOLOv8：images/{train,val}、labels/{train,val}

依赖：
  pip install pillow numpy opencv-python tqdm
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

# ======================== 固定参数（按需修改） ==========================
# 输入图像目录
INPUT_DIR = Path("/root/autodl-tmp/ultralytics-main/datasets/final/natual_high/images/train")

# 数据集根目录（会自动创建 images/train、labels/train；val 可为空）
DATASET_ROOT = Path("/root/autodl-tmp/ultralytics-main/datasets/LowTexture/natual_high")

# YOLO 输出子目录
IMAGES_TRAIN = DATASET_ROOT / "images" / "train"
IMAGES_VAL   = DATASET_ROOT / "images" / "val"    # 可为空
LABELS_TRAIN = DATASET_ROOT / "labels" / "train"
LABELS_VAL   = DATASET_ROOT / "labels" / "val"    # 可为空

# 纹理复杂度（熵图）计算
WIN             = 16     # 熵计算窗口尺寸（像素）
ENT_QUANTILE    = 0.35   # 分位阈值：小于该分位视为“低纹理”
SMOOTH_ENTROPY  = False  # 是否对熵图做轻微平滑（通常不需要）

# 低纹理二值掩膜后处理（形态学）
OPEN_KERNEL     = 3      # 先开运算去噪（像素），<=1 则跳过
CLOSE_KERNEL    = 9      # 再闭运算合并块（像素），<=1 则跳过

# 框筛选
MIN_BOX_AREA_FRAC = 0.001   # 最小框面积占整图比例（过小噪声剔除）
MAX_BOX_NUM        = 200    # 每张图最多导出多少个低纹理框（防止过多）

# 叠加样式
OVERLAY_ALPHA   = 0.35      # 叠加透明度（0~1）
OVERLAY_COLOR   = (0, 255, 0)  # BGR (OpenCV) 绿色

# 输出图片格式
FOLLOW_SRC_FORMAT = True     # True: 跟随源扩展名；False: 统一 .jpg
JPEG_QUALITY      = 95

# 类别定义（单类：lowtex=0）
CLASS_ID = 0

# 支持的输入扩展名
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# ====================================================================


def ensure_dirs() -> None:
    for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        d.mkdir(parents=True, exist_ok=True)
    if not INPUT_DIR.exists():
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[WARN] 输入目录不存在，已创建空目录：{INPUT_DIR.resolve()}")


def list_images(dirpath: Path) -> Iterable[Path]:
    for p in sorted(dirpath.glob("**/*")):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p


def shannon_entropy_blockwise(gray: np.ndarray, win: int) -> np.ndarray:
    """
    按 win×win 滑窗（非重叠）计算块熵，返回形状 (H//win, W//win) 的熵图。
    """
    H, W = gray.shape
    hB, wB = H // win, W // win
    ent = np.zeros((hB, wB), dtype=np.float32)
    for by in range(hB):
        for bx in range(wB):
            patch = gray[by*win:(by+1)*win, bx*win:(bx+1)*win]
            hist, _ = np.histogram(patch, bins=256, range=(0, 256), density=True)
            p = hist + 1e-12  # 防止 log(0)
            ent[by, bx] = float(-(p * np.log2(p)).sum())
    if SMOOTH_ENTROPY:
        ent = cv2.GaussianBlur(ent, (3, 3), 0)
    return ent


def low_texture_mask(ent: np.ndarray, win: int, H: int, W: int, q: float) -> np.ndarray:
    """
    从块熵图得到低纹理掩膜（布尔，分辨率与原图一致）。
    小于分位阈值 q 的块视为低纹理。
    """
    thr = float(np.quantile(ent, q))
    mask_blk = (ent < thr).astype(np.uint8)  # 0/1
    # 把块级掩膜放大到像素级
    mask = np.kron(mask_blk, np.ones((win, win), dtype=np.uint8))
    mask = mask[:H, :W]  # 裁剪到原图大小
    return mask  # dtype=uint8, {0,1}


def postprocess_mask(mask01: np.ndarray) -> np.ndarray:
    """
    对低纹理掩膜做形态学开闭运算以去噪与并块，返回 uint8 0/255 二值图。
    """
    bin_mask = (mask01 * 255).astype(np.uint8)
    if OPEN_KERNEL and OPEN_KERNEL > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (OPEN_KERNEL, OPEN_KERNEL))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, k, iterations=1)
    if CLOSE_KERNEL and CLOSE_KERNEL > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_KERNEL, CLOSE_KERNEL))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, k, iterations=1)
    return bin_mask


def contours_to_boxes(bin_mask: np.ndarray, min_area: int, max_boxes: int) -> List[Tuple[int,int,int,int]]:
    """
    从二值图提取外接矩形框，按面积过滤并返回 [x,y,w,h] 列表（像素）。
    """
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area >= min_area:
            boxes.append((x, y, w, h))
    # 可按面积降序，避免爆量
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes[:max_boxes]


def overlay_green(img_bgr: np.ndarray, mask01: np.ndarray, alpha: float, color_bgr=(0,255,0)) -> np.ndarray:
    """
    对 mask==1 的像素做半透明绿色覆盖；返回 BGR 图（uint8）。
    """
    out = img_bgr.copy()
    m = mask01.astype(bool)
    if not m.any():
        return out
    # 逐像素 alpha 混合
    overlay = np.empty_like(out)
    overlay[:] = np.array(color_bgr, dtype=np.uint8)
    out[m] = (alpha * overlay[m] + (1.0 - alpha) * out[m]).astype(np.uint8)
    return out


def yolo_write_labels(txt_path: Path, boxes_xywh: List[Tuple[int,int,int,int]], W: int, H: int, cls_id: int) -> None:
    """
    写 YOLO txt：每行 "cls cx cy w h"（归一化且保留6位小数）
    """
    lines = []
    for (x, y, w, h) in boxes_xywh:
        cx = (x + w / 2.0) / W
        cy = (y + h / 2.0) / H
        nw = w / W
        nh = h / H
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def process_image(img_path: Path) -> Tuple[Path, Path]:
    """
    返回：输出图片路径、标签路径
    """
    # 读取为 BGR（OpenCV）
    pil = Image.open(img_path).convert("RGB")
    W, H = pil.size
    img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ent = shannon_entropy_blockwise(gray, WIN)
    mask01 = low_texture_mask(ent, WIN, H, W, ENT_QUANTILE)
    bin_mask = postprocess_mask(mask01)

    # 提取框
    min_area_px = int(MIN_BOX_AREA_FRAC * W * H)
    boxes = contours_to_boxes(bin_mask, min_area=min_area_px, max_boxes=MAX_BOX_NUM)

    # 绿色覆盖
    vis_bgr = overlay_green(img_bgr, mask01, OVERLAY_ALPHA, OVERLAY_COLOR)

    # 保存图片
    stem = img_path.stem + "_lowtex"
    if FOLLOW_SRC_FORMAT and img_path.suffix.lower() in VALID_EXTS:
        ext = img_path.suffix.lower()
    else:
        ext = ".jpg"
    img_out_path = IMAGES_TRAIN / f"{stem}{ext}"
    img_out_path.parent.mkdir(parents=True, exist_ok=True)
    if ext in (".jpg", ".jpeg"):
        # 用 cv2.imencode 控制 JPEG 质量
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        ok, buf = cv2.imencode(".jpg", vis_bgr, encode_param)
        if ok:
            img_out_path.write_bytes(buf.tobytes())
        else:
            Image.fromarray(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)).save(img_out_path, quality=JPEG_QUALITY)
    else:
        Image.fromarray(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)).save(img_out_path)

    # 保存标签（允许为空文件或无文件，YOLOv8二者皆可；这里写空文件更直观）
    label_out_path = LABELS_TRAIN / f"{stem}.txt"
    yolo_write_labels(label_out_path, boxes, W, H, CLASS_ID)

    return img_out_path, label_out_path


def main() -> int:
    ensure_dirs()
    images = list(list_images(INPUT_DIR))
    if not images:
        print(f"[INFO] 在 {INPUT_DIR.resolve()} 未找到图片（支持扩展名：{sorted(VALID_EXTS)}）")
        return 0

    print(f"[INFO] 输入：{INPUT_DIR.resolve()}")
    print(f"[INFO] 输出数据集根：{DATASET_ROOT.resolve()}")
    print(f"[INFO] 参数：WIN={WIN}, ENT_Q={ENT_QUANTILE}, OPEN={OPEN_KERNEL}, CLOSE={CLOSE_KERNEL}, "
          f"MIN_BOX_AREA_FRAC={MIN_BOX_AREA_FRAC}, ALPHA={OVERLAY_ALPHA}")

    ok, fail = 0, 0
    for p in tqdm(images, desc="低纹理检测与标注", unit="张", ncols=90):
        try:
            img_out, lbl_out = process_image(p)
            ok += 1
        except Exception as e:
            tqdm.write(f"[ERROR] {p} —— {e}")
            fail += 1

    print(f"\n[SUMMARY] 成功：{ok}，失败：{fail}")
    print(f"[DATASET] images/train: {IMAGES_TRAIN.resolve()}")
    print(f"[DATASET] labels/train: {LABELS_TRAIN.resolve()}")
    print("[HINT] images/val 与 labels/val 可为空，或自行填入验证集。")
    return 0


if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断。")
        sys.exit(130)