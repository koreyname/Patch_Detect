# -*- coding: utf-8 -*-
"""
无标签数据集版：多模型级联 + YOLO整图标注 + detect/undetect 分类
- 仅输出到 images/ 目录树：
    images/detect/   ：保存带框整图
    images/labels/   ：保存像素坐标标注 txt（每行：cls x1 y1 x2 y2 conf model_name）
    images/undetect/ ：未检出原图
- 不再生成 OUTPUT_DIR/{detect,undetect}；metrics 仍在 OUTPUT_DIR
"""
from __future__ import annotations
from pathlib import Path
import shutil
from typing import List, Dict, Tuple
import sys

# =============== 推理配置 ===============
MODELS_DIR   = Path("/root/autodl-tmp/ultralytics-main/code/runs/bests/all")
RECURSIVE    = False
MODEL_PATTERN = "*.pt"
SORT_BY       = "name"   # "name" | "mtime" | "size"
REVERSE_SORT  = False

INPUT_DIR  = Path("/root/autodl-tmp/ultralytics-main/datasets/final/pure/images/val")
OUTPUT_DIR = Path("/root/autodl-tmp/ultralytics-main/Results/tests")  # 仅用于汇总文件

IMGSZ      = 640
CONF_THRES = 0.5
IOU_THRES  = 0.5   # 仅影响NMS；无评测
DEVICE     = 0     # -1=CPU

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 开关：启用预检查（0 字节/读失败快速标记为 undetect）
PRECHECK_IMAGE = True

# images/ 导出根目录（默认放在 OUTPUT_DIR 的同级目录下）
IMAGES_ROOT = OUTPUT_DIR / "images"
LABELS_DIR_NAME = "labels"  # 若你已有固定拼写“lables”，改成 "lables" 即可
# 清理旧的 OUTPUT_DIR/{detect,undetect}
CLEAN_LEGACY_OUTPUT_DIRS = True
# ====================================


def list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    imgs: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
            imgs.append(p)
    return sorted(imgs)


def scan_models(models_dir: Path, pattern: str, recursive: bool,
                sort_by: str = "name", reverse: bool = False) -> List[Path]:
    if not models_dir.exists():
        return []
    files = list(models_dir.rglob(pattern) if recursive else models_dir.glob(pattern))
    files = [f for f in files if f.is_file()]
    def key_fn(p: Path):
        try:
            if sort_by == "mtime": return p.stat().st_mtime
            if sort_by == "size":  return p.stat().st_size
            return p.name
        except Exception:
            return p.name
    files.sort(key=key_fn, reverse=reverse)
    return files


def precheck_readable(img_path: Path) -> Tuple[bool, str]:
    try:
        if img_path.stat().st_size == 0:
            return False, "zero_bytes"
    except Exception as e:
        return False, f"stat_fail:{e}"
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            im.verify()
        return True, ""
    except Exception as e:
        return False, f"pil_verify_fail:{e}"


def safe_infer(model, img_path: Path):
    from ultralytics import YOLO  # noqa
    try:
        return model(
            source=str(img_path),
            imgsz=IMGSZ, conf=CONF_THRES, iou=IOU_THRES,
            device=DEVICE, verbose=False,
        )
    except Exception as e1:
        try:
            from PIL import Image
            import numpy as np
            with Image.open(img_path) as im:
                if im.mode not in ("RGB", "RGBA", "L"):
                    im = im.convert("RGB")
                arr = np.array(im)
            return model(
                source=arr,
                imgsz=IMGSZ, conf=CONF_THRES, iou=IOU_THRES,
                device=DEVICE, verbose=False,
            )
        except Exception as e2:
            raise RuntimeError(f"opencv_read_fail:{e1} | pil_read_fail:{e2}")


def draw_and_save_preds(img_path: Path,
                        preds: List[Tuple[int, float, float, float, float, float]],
                        out_path: Path,
                        model_name: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError("cv2_imread_none")
        COLOR_P = (0, 255, 0)
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        for (pcls, x1, y1, x2, y2, cf) in preds:
            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))
            cv2.rectangle(img, p1, p2, COLOR_P, thickness=2)
            label = f"c{pcls} {cf:.2f} [{model_name}]"
            cv2.putText(img, label, (p1[0], max(0, p1[1] - 5)),
                        FONT, 0.5, COLOR_P, thickness=1, lineType=cv2.LINE_AA)
        cv2.imwrite(str(out_path))
    except Exception:
        from PIL import Image, ImageDraw, ImageFont
        im = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        for (pcls, x1, y1, x2, y2, cf) in preds:
            x1 = int(round(x1)); y1 = int(round(y1))
            x2 = int(round(x2)); y2 = int(round(y2))
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
            text = f"c{pcls} {cf:.2f} [{model_name}]"
            if font:
                draw.text((x1, max(0, y1 - 10)), text, fill=(0, 255, 0), font=font)
            else:
                draw.text((x1, max(0, y1 - 10)), text, fill=(0, 255, 0))
        im.save(out_path)


def write_labels_txt(txt_path: Path,
                     preds: List[Tuple[int, float, float, float, float, float]],
                     model_name: str):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{int(pcls)} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {cf:.4f} {model_name}"
        for (pcls, x1, y1, x2, y2, cf) in preds
    ]
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main():
    from ultralytics import YOLO

    # 仅确保 OUTPUT_DIR 存在以保存 metrics
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # images/ 导出结构
    images_detect_dir   = IMAGES_ROOT / "detect"
    images_labels_dir   = IMAGES_ROOT / LABELS_DIR_NAME
    images_undetect_dir = IMAGES_ROOT / "undetect"
    for d in (images_detect_dir, images_labels_dir, images_undetect_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 清理旧的 OUTPUT_DIR/{detect,undetect}
    if CLEAN_LEGACY_OUTPUT_DIRS:
        for legacy in (OUTPUT_DIR / "detect", OUTPUT_DIR / "undetect"):
            if legacy.exists():
                try:
                    shutil.rmtree(legacy)
                    print(f"[CLEAN] 已删除旧目录: {legacy}")
                except Exception as e:
                    print(f"[WARN] 旧目录删除失败: {legacy} :: {e}")

    model_paths = scan_models(MODELS_DIR, MODEL_PATTERN, RECURSIVE, SORT_BY, REVERSE_SORT)
    if not model_paths:
        print(f"[ERROR] 模型目录无 .pt 文件: {MODELS_DIR}")
        sys.exit(1)
    imgs = list_images(INPUT_DIR)
    if not imgs:
        print(f"[WARN] 输入目录无图片: {INPUT_DIR}")
        sys.exit(0)

    print("[INFO] 本次使用模型顺序：")
    for i, m in enumerate(model_paths, 1):
        print(f"  {i:02d}. {m}")
    (OUTPUT_DIR / "model_order.txt").write_text(
        "\n".join(str(p) for p in model_paths) + "\n", encoding="utf-8"
    )

    model_cache: Dict[Path, YOLO] = {}
    def get_model(w: Path) -> YOLO:
        if w not in model_cache:
            if not w.exists():
                print(f"[ERROR] 模型权重不存在: {w}")
                sys.exit(1)
            model_cache[w] = YOLO(str(w))
        return model_cache[w]

    n_total = 0
    n_detect_img = 0
    bad_files: List[str] = []
    per_model_hits: Dict[str, int] = {p.name: 0 for p in model_paths}
    detected_by: Dict[str, str] = {}

    for img_path in imgs:
        n_total += 1

        # 预检查：读失败 -> 直接丢 images/undetect
        if PRECHECK_IMAGE:
            ok, reason = precheck_readable(img_path)
            if not ok:
                shutil.copy2(img_path, images_undetect_dir / img_path.name)
                bad_files.append(f"{img_path} :: {reason}")
                if n_total % 50 == 0:
                    rate_now = (n_detect_img / n_total * 100.0) if n_total > 0 else 0.0
                    print(f"[INFO] 进度: {n_total}/{len(imgs)} 已处理，当前正确捕获率: {rate_now:.2f}%")
                continue

        # 逐模型级联，取首个产生框的结果
        res_used = None
        res_model_name = "-"
        for w in model_paths:
            model = get_model(w)
            try:
                results = safe_infer(model, img_path)
            except Exception:
                continue
            res = results[0]
            if res.boxes is not None and len(res.boxes) > 0:
                res_used = res
                res_model_name = w.name
                break

        if res_used is not None and res_used.boxes is not None and len(res_used.boxes) > 0:
            xyxy = res_used.boxes.xyxy.cpu().numpy()
            cls  = res_used.boxes.cls.cpu().numpy().astype(int)
            conf = res_used.boxes.conf.cpu().numpy()
            preds = [(int(c), float(x1), float(y1), float(x2), float(y2), float(cf))
                     for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, conf)]

            # 只在 images/ 保存
            out_path_images = images_detect_dir / img_path.name
            draw_and_save_preds(img_path=img_path, preds=preds,
                                out_path=out_path_images, model_name=res_model_name)
            txt_path = images_labels_dir / f"{img_path.stem}.txt"
            write_labels_txt(txt_path=txt_path, preds=preds, model_name=res_model_name)

            n_detect_img += 1
            detected_by[img_path.name] = res_model_name if res_model_name else "-"
            per_model_hits[res_model_name] = per_model_hits.get(res_model_name, 0) + 1
        else:
            shutil.copy2(img_path, images_undetect_dir / img_path.name)

        if n_total % 50 == 0:
            rate_now = (n_detect_img / n_total * 100.0) if n_total > 0 else 0.0
            print(f"[INFO] 进度: {n_total}/{len(imgs)} 已处理，当前正确捕获率: {rate_now:.2f}%")

    # —— 汇总 —— #
    success_rate_pct = (n_detect_img / n_total * 100.0) if n_total > 0 else 0.0
    lines = [
        f"Models dir: {MODELS_DIR}",
        f"Models used: {len(model_paths)}",
        f"Total images: {n_total}",
        f"Detected (>=1 box): {n_detect_img}",
        f"Undetected: {n_total - n_detect_img}",
        f"正确捕获率 (成功捕获/总数): {success_rate_pct:.2f}%",
        "",
        "Per-model first-hit images:",
    ] + [f"  - {m}: {cnt}" for m, cnt in per_model_hits.items()]
    summary = "\n".join(lines) + "\n"

    (OUTPUT_DIR / "metrics.txt").write_text(summary, encoding="utf-8")
    if detected_by:
        pairs = [f"{k} -> {v}" for k, v in sorted(detected_by.items())]
        (OUTPUT_DIR / "detected_by.txt").write_text("\n".join(pairs) + "\n", encoding="utf-8")
    if bad_files:
        (OUTPUT_DIR / "bad_files.txt").write_text("\n".join(bad_files) + "\n", encoding="utf-8")

    print("\n========== 推理完成 ==========")
    print(summary)
    print(f"[EXPORT] 带框整图    -> {images_detect_dir}")
    print(f"[EXPORT] 坐标标签txt -> {images_labels_dir}")
    print(f"[EXPORT] 未检出原图  -> {images_undetect_dir}")
    print(f"汇总指标 -> {OUTPUT_DIR / 'metrics.txt'}")
    if bad_files:
        print(f"[WARN] 有 {len(bad_files)} 个文件读失败，详情见 {OUTPUT_DIR / 'bad_files.txt'}")


if __name__ == "__main__":
    main()
