# -*- coding: utf-8 -*-
"""
单类补丁训练：低纹理先验（lowtex） + YOLO（将原三类补丁统一映射为 1 类：patch=0）
两阶段训练：
  Phase-1：按“补丁框在低纹理中的覆盖率”过采样（放大难样本）
  Phase-2：弱化过采样收敛，抑制低纹理误检

修复点：
- 不再对镜像路径 .resolve()（确保使用 WORK_DS 下的单类标签）
- 当 WORK_DS/images/val 为空时，data.yaml 自动回退到 images/train
- 训练输出固定到 WORK_DS/runs/final/{p1,p2}/
- 可选多卡训练（DEVICE 可设 "0,1"）
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import os

# ==================== 写死路径与参数（按需自行修改） ====================
PATCH_DS = Path("/root/autodl-tmp/ultralytics-main/datasets/final/natual_high")
LOWTEX_DS = Path("/root/autodl-tmp/ultralytics-main/Prepare/LowTexture/natual_high")
WORK_DS = PATCH_DS / "_singleclass_work"

PHASE1_LIST = WORK_DS / "train_weighted_phase1.txt"
PHASE2_LIST = WORK_DS / "train_weighted_phase2.txt"
DATA_YAML   = WORK_DS / "patch.yaml"  # 将写入 nc:1 & names:[patch]

# 过采样权重策略
LAMBDA_P1  = 2.5
GAMMA      = 2.0
LAMBDA_P2  = 0.5
REPEAT_CAP = 5
LOWTEX_SUFFIX = "_lowtex"

# 训练超参
MODEL     = "yolov8n.pt"
EPOCHS_P1 = 100
EPOCHS_P2 = 50
IMGSZ     = 640
BATCH     = 32
DEVICE    = "0"            # 单卡:"0"；双卡:"0,1"
# =====================================================================

# -------------------- 工具函数 --------------------
def read_yolo_boxes(lbl_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not lbl_path.exists():
        return []
    out = []
    for ln in lbl_path.read_text().splitlines():
        p = ln.strip().split()
        if len(p) == 5:
            c = int(float(p[0])); cx, cy, w, h = map(float, p[1:])
            out.append((c, cx, cy, w, h))
    return out

def write_yolo_boxes(lbl_path: Path, boxes: List[Tuple[int, float, float, float, float]]):
    if not boxes:
        return
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for (c,cx,cy,w,h) in boxes]
    lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def yolo_to_xyxy(cx, cy, w, h, W, H):
    x1 = (cx - w/2) * W; y1 = (cy - h/2) * H
    x2 = (cx + w/2) * W; y2 = (cy + h/2) * H
    return max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2)

def area_xyxy(x1,y1,x2,y2): return max(0.0, x2-x1) * max(0.0, y2-y1)

def ioa_a_over_a(ax1,ay1,ax2,ay2, bx1,by1,bx2,by2):
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    inter = area_xyxy(ix1,iy1,ix2,iy2)
    aa = area_xyxy(ax1,ay1,ax2,ay2) + 1e-6
    return inter / aa

def lowtex_label_path_for(img_path: Path) -> Path:
    cand1 = LOWTEX_DS / "labels" / "train" / f"{img_path.stem}{LOWTEX_SUFFIX}.txt"
    if cand1.exists():
        return cand1
    cand2 = LOWTEX_DS / "labels" / "train" / f"{img_path.stem}.txt"
    return cand2

def coverage_for_image(img_path: Path, patch_lbl: Path) -> float:
    patch_boxes = read_yolo_boxes(patch_lbl)
    if not patch_boxes:
        return 0.0
    low_lbl = lowtex_label_path_for(img_path)
    low_boxes = read_yolo_boxes(low_lbl)
    if not low_boxes:
        return 0.0
    with Image.open(img_path) as im:
        W, H = im.size
    low_xyxy = [yolo_to_xyxy(cx,cy,w,h, W,H) for (_,cx,cy,w,h) in low_boxes]
    covs = []
    for (_, cx,cy,w,h) in patch_boxes:
        ax1,ay1,ax2,ay2 = yolo_to_xyxy(cx,cy,w,h, W,H)
        if not low_xyxy:
            covs.append(0.0); continue
        ioas = [ioa_a_over_a(ax1,ay1,ax2,ay2, *lb) for lb in low_xyxy]
        covs.append(max(ioas))
    return float(np.mean(covs)) if covs else 0.0

def weight_from_cover(c: float, lamb: float, gamma: float) -> int:
    rep = int(round(1.0 + lamb * (c ** gamma)))
    return max(1, min(REPEAT_CAP, rep))

# -------------------- 单类工作区构建 --------------------
def ensure_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() and os.path.realpath(dst) == str(src.resolve()):
                return
            dst.unlink()
        os.symlink(src, dst)
    except OSError:
        pass

def remap_label_to_single(src_lbl: Path, dst_lbl: Path):
    boxes = read_yolo_boxes(src_lbl)
    boxes_sc = [(0, cx, cy, w, h) for (_c, cx, cy, w, h) in boxes]
    write_yolo_boxes(dst_lbl, boxes_sc)

def build_singleclass_views():
    """在 WORK_DS 下镜像 images/{train,val} 为软链接，labels/{train,val} 写入单类标注。"""
    for split in ("train", "val"):
        img_src_dir = PATCH_DS / "images" / split
        lbl_src_dir = PATCH_DS / "labels" / split
        img_dst_dir = WORK_DS / "images" / split
        lbl_dst_dir = WORK_DS / "labels" / split
        img_dst_dir.mkdir(parents=True, exist_ok=True)
        lbl_dst_dir.mkdir(parents=True, exist_ok=True)

        if not img_src_dir.exists():
            continue
        imgs = sorted(img_src_dir.glob("*.*"))
        for img in imgs:
            ensure_symlink(img, img_dst_dir / img.name)
            src_lbl = lbl_src_dir / f"{img.stem}.txt"
            dst_lbl = lbl_dst_dir / f"{img.stem}.txt"
            if src_lbl.exists():
                remap_label_to_single(src_lbl, dst_lbl)
            else:
                if dst_lbl.exists():
                    dst_lbl.unlink()

# -------------------- 加权训练清单构建（基于 WORK_DS 镜像路径） --------------------
def build_weighted_list(outfile: Path, lamb: float, gamma: float):
    img_src_dir = PATCH_DS / "images" / "train"
    lbl_src_dir = PATCH_DS / "labels" / "train"
    img_dst_dir = WORK_DS / "images" / "train"

    imgs = sorted(img_src_dir.glob("*.*"))
    lines = []
    miss_low, miss_patch = 0, 0
    for img in imgs:
        patch_lbl = lbl_src_dir / f"{img.stem}.txt"
        if not patch_lbl.exists():
            miss_patch += 1
            # 注意：不要 .resolve()，否则会绕开 WORK_DS 的镜像与单类标签
            lines.append(str(img_dst_dir / img.name)); continue
        c = coverage_for_image(img, patch_lbl)
        rep = weight_from_cover(c, lamb, gamma)
        lines.extend([str(img_dst_dir / img.name)] * rep)
        low_lbl = lowtex_label_path_for(img)
        if not low_lbl.exists(): miss_low += 1
    outfile.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {outfile} 写入 {len(lines)} 行；原始 {len(imgs)} 张；缺低纹理标注 {miss_low}，缺补丁标注 {miss_patch}")

# -------------------- data.yaml 更新（带 val 回退） --------------------
def write_data_yaml(data_yaml: Path, train_entry: str):
    val_dir = WORK_DS / "images" / "val"
    has_val = val_dir.exists() and any(val_dir.glob("*.*"))
    val_entry = "images/val" if has_val else "images/train"
    if not has_val:
        print(f"[WARN] 未发现验证集，已将 val 回退到 {val_entry}")

    content = "\n".join([
        f"path: {WORK_DS}",
        f"train: {train_entry}",
        f"val: {val_entry}",
        "nc: 1",
        "names: [patch]",
        ""
    ])
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text(content, encoding="utf-8")
    print(f"[OK] data.yaml 已写入，train: {train_entry}, val: {val_entry}")

# -------------------- 训练 --------------------
def train(model_w: str, data_yaml: Path, epochs: int, imgsz: int, batch: int,
          device=0, resume=False, run_name="train", amp=True):
    from ultralytics import YOLO
    model = YOLO(model_w)
    return model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,       # "0,1" 多卡
        resume=resume,
        patience=10,
        project=str(WORK_DS / "runs" ),
        name=run_name,
        workers=6,
    #  cache=True,          # 这行会触发 cache='ram'
    #   workers=6,           # 两卡总 8 个线程，减小内存峰值
       cache="disk",        # 改为磁盘缓存，避免 OOM
        amp=amp,
    )


def find_last_best():
    runs = WORK_DS / "runs" 
    if not runs.exists(): return None
    cands = sorted(runs.glob("**/weights/best.pt"), key=os.path.getmtime)
    return str(cands[-1]) if cands else None

# -------------------- 主流程 --------------------
def main():
    build_singleclass_views()

    # Phase-1
    build_weighted_list(PHASE1_LIST, LAMBDA_P1, GAMMA)
    write_data_yaml(DATA_YAML, str(PHASE1_LIST))
    print("\n=== Phase-1 训练开始 ===")
    train(MODEL, DATA_YAML, EPOCHS_P1, IMGSZ, BATCH, DEVICE, run_name="p1")

    best = find_last_best() or MODEL
    print(f"[INFO] Phase-1 best: {best}")

    # Phase-2
    build_weighted_list(PHASE2_LIST, LAMBDA_P2, GAMMA)
    write_data_yaml(DATA_YAML, str(PHASE2_LIST))
    print("\n=== Phase-2 训练开始 ===")
    train(best, DATA_YAML, EPOCHS_P2, IMGSZ, BATCH, DEVICE, run_name="p2")

    print("\n[DONE] 训练完成。请到 _singleclass_work/runs/{p1,p2}/*/weights/ 查看 best.pt")

if __name__ == "__main__":
    main()
