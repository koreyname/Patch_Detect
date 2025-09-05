# -*- coding: utf-8 -*-
"""
多模型级联 + 稳健读图容错 + 逐框评测(TP/FP/FN)
(整图标注输出；分类到 patches/tp|fp|fn)

新增策略（按你的要求）：
1) 按模型顺序对同一张图依次推理与评测；
2) 若某个模型达到 “ONLY-TP”（有TP且无FP/FN），立刻选用该模型结果并进入下一张图；
3) 若所有模型都没有 ONLY-TP：
   - 若存在 “有TP但非ONLY-TP” 的模型，则选用其中“最好”的（优先 TP 数多，其次 F1 高，再次 FP+FN 少，再次 FP 少）；
   - 若所有模型都无TP，则使用“当前结果”（即顺序中的最后一个模型结果）。
4) tp 目录只保存 “纯 TP（ONLY-TP）” 的图片；fp / fn 目录按是否存在 FP / FN 决定是否保存。
5) 进度与最终“准确率(Accuracy)”为逐图口径：ONLY-TP 图像数 / 已处理图像数。
"""
from __future__ import annotations
from pathlib import Path
import shutil
from typing import List, Dict, Tuple, Optional
import sys
import os

# =============== 推理配置 ===============
MODELS_DIR   = Path("/root/autodl-tmp/ultralytics-main/code/runs/bests/all")
RECURSIVE    = False
MODEL_PATTERN = "*.pt"
SORT_BY       = "name"   # "name" | "mtime" | "size"
REVERSE_SORT  = False

INPUT_DIR  = Path("/root/autodl-tmp/ultralytics-main/datasets/final/images/val")
OUTPUT_DIR = Path("/root/autodl-tmp/ultralytics-main/Results/tests")

IMGSZ      = 640
CONF_THRES = 0.5
IOU_THRES  = 0.5
DEVICE     = 0   # -1=CPU

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 开关：启用预检查（0 字节/读失败快速标记）
PRECHECK_IMAGE = True
# 运行前清空输出子目录，防止历史结果残留
CLEAN_OUTPUT_DIRS = True
# ====================================

# =============== 评测配置（与 labels 对齐） ===============
LABELS_DIR = (INPUT_DIR.parent.parent / "labels" / INPUT_DIR.name)

EVAL_ENABLE            = True     # 置 False 可关闭评测（只跑原逻辑）
IOU_MATCH_THRES        = 0.50     # 预测/GT 匹配 IoU 阈值（评测用）
MISSING_GT_IS_NEGATIVE = True     # 若无 .txt 标注，视为负样本(无目标)；置 False 则跳过该图的评测（视作非TP）
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
    """快速检查是否可读：0字节或 PIL.open 失败视为不可读，返回原因字符串。"""
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
    """推理的安全封装：先走常规路径失败则用 PIL 读入 numpy 再推理。"""
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
                arr = np.array(im)  # RGB
            return model(
                source=arr,
                imgsz=IMGSZ, conf=CONF_THRES, iou=IOU_THRES,
                device=DEVICE, verbose=False,
            )
        except Exception as e2:
            raise RuntimeError(f"opencv_read_fail:{e1} | pil_read_fail:{e2}")


# ==================== 评测相关：GT加载、IoU与匹配 ====================
def yolo_label_path(img_path: Path) -> Path:
    rel = img_path.relative_to(INPUT_DIR).with_suffix(".txt")
    return LABELS_DIR / rel


def load_gt_boxes(img_path: Path, img_shape: Tuple[int, int]) -> Optional[List[Tuple[int, float, float, float, float]]]:
    """读取 YOLO 标注，返回 [(cls, x1,y1,x2,y2)]（像素坐标）。若无标注文件返回 None。"""
    H, W = img_shape
    lp = yolo_label_path(img_path)
    if not lp.exists():
        return None
    gts: List[Tuple[int, float, float, float, float]] = []
    with lp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls, cx, cy, w, h = map(float, parts[:5])
            x1 = (cx - w/2) * W
            y1 = (cy - h/2) * H
            x2 = (cx + w/2) * W
            y2 = (cy + h/2) * H
            x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
            y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
            gts.append((int(cls), x1, y1, x2, y2))
    return gts


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(preds, gts, iou_thres=IOU_MATCH_THRES):
    """
    preds: [(cls, x1,y1,x2,y2, conf)]
    gts:   [(cls, x1,y1,x2,y2)]
    返回:
      matches = [(pi, gi, iou)]
      pred_unmatched = [pi]
      gt_unmatched   = [gi]
    """
    matches = []
    if not preds and not gts:
        return matches, [], []
    pairs = []
    for pi, (pcls, px1, py1, px2, py2, _) in enumerate(preds):
        for gi, (gcls, gx1, gy1, gx2, gy2) in enumerate(gts):
            if pcls != gcls:
                continue
            iou = iou_xyxy((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
            if iou >= iou_thres:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True, key=lambda x: x[0])
    used_p, used_g = set(), set()
    for iou, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi)
        matches.append((pi, gi, iou))
    pred_unmatched = [pi for pi in range(len(preds)) if pi not in used_p]
    gt_unmatched   = [gi for gi in range(len(gts))  if gi not in used_g]
    return matches, pred_unmatched, gt_unmatched


# ============== 绘制与分类保存（tp 仅保存“纯 TP”） ==============
def draw_and_save_full_annot(
    img_path: Path,
    preds: List[Tuple[int, float, float, float, float, float]],
    gts: Optional[List[Tuple[int, float, float, float, float]]],
    matches: List[Tuple[int, int, float]],
    pred_unm: List[int],
    gt_unm: List[int],
    out_root: Path,
    model_name: str,
    only_tp: bool,
    has_fp: bool,
    has_fn: bool,
):
    out_all = out_root / "all"
    out_tp  = out_root / "tp"
    out_fp  = out_root / "fp"
    out_fn  = out_root / "fn"

    try:
        import cv2
        img = cv2.imread(str(img_path))  # BGR
        if img is None:
            raise RuntimeError("cv2_imread_none")
        COLOR_TP = (0, 255, 0)
        COLOR_FP = (0, 0, 255)
        COLOR_FN = (0, 255, 255)
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        def put_box_label(x1, y1, x2, y2, color, text):
            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))
            cv2.rectangle(img, p1, p2, color, thickness=2)
            cv2.putText(img, text, (p1[0], max(0, p1[1] - 5)), FONT, 0.5, color, thickness=1, lineType=cv2.LINE_AA)

        match_map = {pi: iou for (pi, gi, iou) in matches}

        for pi, (pcls, x1, y1, x2, y2, cf) in enumerate(preds):
            if pi in match_map:
                iou = match_map[pi]
                text = f"TP c{pcls} {cf:.2f} IoU {iou:.2f} [{model_name}]"
                put_box_label(x1, y1, x2, y2, COLOR_TP, text)
            else:
                text = f"FP c{pcls} {cf:.2f} [{model_name}]"
                put_box_label(x1, y1, x2, y2, COLOR_FP, text)

        if gts is not None:
            for gi in gt_unm:
                gcls, gx1, gy1, gx2, gy2 = gts[gi]
                text = f"FN c{gcls}"
                put_box_label(gx1, gy1, gx2, gy2, COLOR_FN, text)

        out_path_all = out_all / img_path.name
        cv2.imwrite(str(out_path_all), img)

    except Exception:
        from PIL import Image, ImageDraw, ImageFont
        im = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        def draw_box(x1, y1, x2, y2, color, text):
            x1 = int(round(x1)); y1 = int(round(y1))
            x2 = int(round(x2)); y2 = int(round(y2))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            if font:
                draw.text((x1, max(0, y1 - 10)), text, fill=color, font=font)
            else:
                draw.text((x1, max(0, y1 - 10)), text, fill=color)

        match_map = {pi: iou for (pi, gi, iou) in matches}

        for pi, (pcls, x1, y1, x2, y2, cf) in enumerate(preds):
            if pi in match_map:
                iou = match_map[pi]
                draw_box(x1, y1, x2, y2, (0, 255, 0), f"TP c{pcls} {cf:.2f} IoU {iou:.2f} [{model_name}]")
            else:
                draw_box(x1, y1, x2, y2, (255, 0, 0), f"FP c{pcls} {cf:.2f} [{model_name}]")

        if gts is not None:
            for gi in gt_unm:
                gcls, gx1, gy1, gx2, gy2 = gts[gi]
                draw_box(gx1, gy1, gx2, gy2, (255, 255, 0), f"FN c{gcls}")

        out_path_all = out_all / img_path.name
        im.save(out_path_all)

    # —— 分类复制（严格规则：tp 仅“纯 TP”） —— #
    src = (out_root / "all" / img_path.name)
    if only_tp:
        shutil.copy2(src, out_root / "tp" / img_path.name)
    if has_fp:
        shutil.copy2(src, out_root / "fp" / img_path.name)
    if has_fn:
        shutil.copy2(src, out_root / "fn" / img_path.name)


# =====================================================================


def main():
    from ultralytics import YOLO

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    patches_root = OUTPUT_DIR / "patches"

    # —— 清理并重建输出子目录 —— #
    subdirs = [patches_root / "all", patches_root / "tp", patches_root / "fp", patches_root / "fn"]
    if CLEAN_OUTPUT_DIRS:
        for d in subdirs:
            if d.exists():
                shutil.rmtree(d)
    for d in subdirs:
        d.mkdir(parents=True, exist_ok=True)

    if EVAL_ENABLE:
        with (OUTPUT_DIR / "detections.csv").open("w", encoding="utf-8") as f:
            f.write("image,model,status,cls,conf,x1,y1,x2,y2\n")

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

    # 模型缓存
    model_cache: Dict[Path, YOLO] = {}
    def get_model(w: Path) -> YOLO:
        if w not in model_cache:
            if not w.exists():
                print(f"[ERROR] 模型权重不存在: {w}")
                sys.exit(1)
            model_cache[w] = YOLO(str(w))
        return model_cache[w]

    # 统计
    n_total = 0
    n_tp_img = 0  # ONLY-TP 图像数
    bad_files: List[str] = []
    per_model_hits: Dict[str, int] = {p.name: 0 for p in model_paths}  # 记录 ONLY-TP 命中的模型
    detected_by: Dict[str, str] = {}

    tp_sum = fp_sum = fn_sum = 0  # 逐框汇总（来自最终选用的模型）

    # 遍历图片
    for img_path in imgs:
        n_total += 1

        # 预检查
        if PRECHECK_IMAGE:
            ok, reason = precheck_readable(img_path)
            if not ok:
                shutil.copy2(img_path, patches_root / "all" / img_path.name)
                bad_files.append(f"{img_path} :: {reason}")
                if n_total % 100 == 0:
                    acc_now = (n_tp_img / n_total * 100.0) if n_total > 0 else 0.0
                    print(f"[INFO] 进度: {n_total}/{len(imgs)} 已处理，当前准确率(ONLY-TP/样本): {acc_now:.2f}%")
                continue

        # 读取 GT（一次）
        W = H = None
        try:
            from PIL import Image
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception as e:
            # 若读取失败，稍后若有推理结果可用其 orig_shape 兜底
            pass

        gts = None
        if H is not None and W is not None:
            gts = load_gt_boxes(img_path, (H, W))
        else:
            # 若上面没拿到尺寸，先置 None，等第一份结果拿到 orig_shape 再补读
            gts = None

        # 在各模型上评测，寻找 ONLY-TP；若没有，再选最佳
        chosen = None            # 最终选用
        best_non_onlytp = None   # 候选：有TP但非ONLY-TP
        last_candidate = None    # 顺序中的“当前结果”（最后一个）

        def build_candidate(model_name: str, preds, matches, pred_unm, gt_unm):
            tp = len(matches)
            fp = len(pred_unm)
            fn = len(gt_unm)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            has_tp = tp > 0
            has_fp = fp > 0
            has_fn = fn > 0
            only_tp = has_tp and not has_fp and not has_fn
            return {
                "model": model_name,
                "preds": preds,
                "matches": matches,
                "pred_unm": pred_unm,
                "gt_unm": gt_unm,
                "tp": tp, "fp": fp, "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "only_tp": only_tp,
                "has_fp": has_fp,
                "has_fn": has_fn,
            }

        # 若之前没能拿到 W,H，则在首次成功推理后用 orig_shape 再加载 GT
        def ensure_gts_from_result(res_obj):
            nonlocal W, H, gts
            if (W is None or H is None) and (res_obj is not None):
                if hasattr(res_obj, "orig_shape") and res_obj.orig_shape is not None:
                    H, W = res_obj.orig_shape
                    gts = load_gt_boxes(img_path, (H, W)) if (H is not None and W is not None) else None

        for w in model_paths:
            model = get_model(w)
            try:
                results = safe_infer(model, img_path)
            except Exception:
                # 推理失败，记一个空候选
                preds = []
                # 若 gts 仍未知就跳过匹配（空）
                if gts is None:
                    gts_eval = []
                else:
                    gts_eval = gts
                matches, pred_unm, gt_unm = greedy_match(preds, gts_eval, IOU_MATCH_THRES)
                cand = build_candidate(w.name, preds, matches, pred_unm, gt_unm)
                last_candidate = cand
                continue

            res = results[0] if len(results) > 0 else None
            ensure_gts_from_result(res)

            # 组装预测框
            if res is not None and res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls  = res.boxes.cls.cpu().numpy().astype(int)
                conf = res.boxes.conf.cpu().numpy()
                preds = [(int(c), float(x1), float(y1), float(x2), float(y2), float(cf))
                         for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, conf)]
            else:
                preds = []

            # 处理 GT 缺失策略
            gts_eval = gts
            if gts_eval is None:
                if MISSING_GT_IS_NEGATIVE:
                    gts_eval = []
                else:
                    # 跳过该图的评测（按原逻辑：不算 TP），但仍产出可视化
                    gts_eval = None

            if gts_eval is None:
                matches, pred_unm, gt_unm = [], list(range(len(preds))), []
            else:
                matches, pred_unm, gt_unm = greedy_match(preds, gts_eval, IOU_MATCH_THRES)

            cand = build_candidate(w.name, preds, matches, pred_unm, gt_unm)
            last_candidate = cand

            # 规则 1：若出现 ONLY-TP，立刻选用并停止遍历其他模型
            if cand["only_tp"]:
                chosen = cand
                break

            # 否则记录“有TP但非ONLY-TP”的最优候选
            if cand["tp"] > 0:
                if best_non_onlytp is None:
                    best_non_onlytp = cand
                else:
                    # 比较优先级：TP 多 > F1 大 > (FP+FN) 少 > FP 少
                    a, b = best_non_onlytp, cand
                    better = False
                    if b["tp"] > a["tp"]:
                        better = True
                    elif b["tp"] == a["tp"]:
                        if b["f1"] > a["f1"]:
                            better = True
                        elif b["f1"] == a["f1"]:
                            if (b["fp"] + b["fn"]) < (a["fp"] + a["fn"]):
                                better = True
                            elif (b["fp"] + b["fn"]) == (a["fp"] + a["fn"]):
                                if b["fp"] < a["fp"]:
                                    better = True
                    if better:
                        best_non_onlytp = b

        # 遍历结束：若还未选中
        if chosen is None:
            if best_non_onlytp is not None:
                chosen = best_non_onlytp
            else:
                # 所有模型都无 TP，则使用顺序中的“当前结果”（最后一个）
                chosen = last_candidate

        # 若仍然没有候选（极端情况），构造一个空候选以保证流程继续
        if chosen is None:
            chosen = {
                "model": model_paths[-1].name if model_paths else "-",
                "preds": [], "matches": [], "pred_unm": [], "gt_unm": [],
                "tp": 0, "fp": 0, "fn": 0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "only_tp": False, "has_fp": False, "has_fn": False,
            }

        # —— 写 CSV（仅记录最终选用的模型结果） —— #
        if EVAL_ENABLE:
            rows = []
            for pi, (pcls, x1, y1, x2, y2, cf) in enumerate(chosen["preds"]):
                is_tp = any(pi == m[0] for m in chosen["matches"])
                status = "TP" if is_tp else "FP"
                rows.append(f"{img_path.name},{chosen['model']},{status},{pcls},{cf:.4f},{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}")
            for gi in chosen["gt_unm"]:
                # 如果需要写出 FN 的坐标，需要 gts 可用；若 gts_eval 为 None，上面 gt_unm 会为空。
                # 这里按已计算好的 gt_unm 写行（无置信度）
                # 为了安全，这里不反查坐标（已在匹配时用过）
                rows.append(f"{img_path.name},{chosen['model']},FN,0,,,,")  # 简化版；若要坐标可在上面保留 gts_eval
            with (OUTPUT_DIR / "detections.csv").open("a", encoding="utf-8") as f:
                for r in rows:
                    f.write(r + "\n")

        # —— 绘制与分类保存（仅对最终选用的结果） —— #
        draw_and_save_full_annot(
            img_path=img_path,
            preds=chosen["preds"],
            gts=[] if (MISSING_GT_IS_NEGATIVE and gts is None) else gts,
            matches=chosen["matches"],
            pred_unm=chosen["pred_unm"],
            gt_unm=chosen["gt_unm"],
            out_root=patches_root,
            model_name=chosen["model"],
            only_tp=chosen["only_tp"],
            has_fp=chosen["has_fp"],
            has_fn=chosen["has_fn"],
        )

        # —— 统计 —— #
        if chosen["only_tp"]:
            n_tp_img += 1
            detected_by[img_path.name] = chosen["model"]
            per_model_hits[chosen["model"]] = per_model_hits.get(chosen["model"], 0) + 1

        tp_sum += chosen["tp"]
        fp_sum += chosen["fp"]
        fn_sum += chosen["fn"]

        # —— 进度（逐图口径） —— #
        if n_total % 100 == 0:
            acc_now = (n_tp_img / n_total * 100.0) if n_total > 0 else 0.0
            print(f"[INFO] 进度: {n_total}/{len(imgs)} 已处理，当前准确率(ONLY-TP/样本): {acc_now:.2f}%")

    # —— 汇总（逐图 Accuracy + 逐框 Precision/Recall/F1）—— #
    accuracy_pct  = (n_tp_img / n_total * 100.0) if n_total > 0 else 0.0
    precision_pct = (tp_sum / (tp_sum + fp_sum) * 100.0) if (tp_sum + fp_sum) > 0 else 0.0

    lines = [
        f"Models dir: {MODELS_DIR}",
        f"Models used: {len(model_paths)}",
        f"Total images: {n_total}",
        f"Images with ONLY TP (no FP/FN): {n_tp_img}",
        f"Other images (have FP/FN or none): {n_total - n_tp_img}",
        f"Accuracy (ONLY-TP/Images): {accuracy_pct:.2f}%",
        f"Precision (per-box): {precision_pct:.2f}%",
        "",
        "Per-model ONLY-TP images (first-hit in cascade):",
    ] + [f"  - {m}: {cnt}" for m, cnt in per_model_hits.items()]
    summary = "\n".join(lines) + "\n"

    if EVAL_ENABLE:
        precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
        recall    = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
        f1        = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
        summary += (
            "\nEvaluation (per-box, vs. labels):\n"
            f"  TP: {tp_sum}\n  FP: {fp_sum}\n  FN: {fn_sum}\n"
            f"  Precision: {precision:.4f}\n"
            f"  Recall:    {recall:.4f}\n"
            f"  F1:        {f1:.4f}\n"
            f"  IoU match threshold: {IOU_MATCH_THRES}\n"
            f"  Missing-GT policy (MISSING_GT_IS_NEGATIVE): {MISSING_GT_IS_NEGATIVE}\n"
        )

    (OUTPUT_DIR / "metrics.txt").write_text(summary, encoding="utf-8")
    if detected_by:
        pairs = [f"{k} -> {v}" for k, v in sorted(detected_by.items())]
        (OUTPUT_DIR / "detected_by.txt").write_text("\n".join(pairs) + "\n", encoding="utf-8")
    if bad_files:
        (OUTPUT_DIR / "bad_files.txt").write_text("\n".join(bad_files) + "\n", encoding="utf-8")

    print("\n========== 推理完成 ==========")
    print(summary)
    print(f"整图标注(总)  -> {patches_root / 'all'}")
    print(f"分类索引      -> {patches_root}/(tp|fp|fn)")
    print(f"汇总指标      -> {OUTPUT_DIR / 'metrics.txt'}")
    if EVAL_ENABLE:
        print(f"逐补丁明细    -> {OUTPUT_DIR / 'detections.csv'}")
    if bad_files:
        print(f"[WARN] 有 {len(bad_files)} 个文件读/评测失败，详情见 {OUTPUT_DIR / 'bad_files.txt'}")


if __name__ == "__main__":
    main()
