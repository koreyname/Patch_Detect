# -*- coding: utf-8 -*-
"""
视频 -> 级联多模型 -> 帧内首个有检出者生效 -> 输出带框视频
不做评测、不落静态图、不分流目录
用法：python video_detect_cascade.py
"""
from pathlib import Path
import cv2
from typing import List, Dict
from ultralytics import YOLO

# ========== 固定参数（改这里） ==========
MODELS_DIR      = Path("/root/autodl-tmp/ultralytics-main/best")  # 放多份 .pt 的目录
MODEL_PATTERN   = "*.pt"      # 可改成 "best*.pt" 等
RECURSIVE       = False       # 是否递归子目录
SORT_BY         = "name"      # "name" | "mtime" | "size"
REVERSE_SORT    = False       # True 反向（例如 mtime 从新到旧）

VIDEO_IN        = Path("out.mp4")
VIDEO_OUT       = Path("result.mp4")
DEVICE          = 0           # 0/1… 指定GPU，-1=CPU
IMGSZ           = 640
CONF_THRES      = 0.40
IOU_THRES       = 0.50

BOX_THICK       = 2
FONT_SCALE      = 0.6
TXT_THICK       = 1
SHOW_MODEL_TAG  = True        # 在画面左上角标注使用了哪个模型
# =======================================

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

def draw_boxes(frame, boxes_xyxy, clss, confs, names, model_tag: str = ""):
    """在 frame(BGR) 上画框与标签；可选在左上角写当前模型名"""
    # 模型标签
    if SHOW_MODEL_TAG and model_tag:
        cv2.putText(frame, f"Model: {model_tag}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    # 逐框
    for (x1, y1, x2, y2), c, cf in zip(boxes_xyxy, clss, confs):
        p1 = (int(x1), int(y1)); p2 = (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), BOX_THICK)
        cls_name = names.get(int(c), str(int(c))) if isinstance(names, dict) else str(int(c))
        label = f"{cls_name} {float(cf):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TXT_THICK)
        y_text = max(0, p1[1] - 4)
        # 文字背景条
        cv2.rectangle(frame, (p1[0], y_text - th - 4), (p1[0] + tw + 4, y_text + 2), (0, 255, 0), -1)
        cv2.putText(frame, label, (p1[0] + 2, y_text - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), TXT_THICK, cv2.LINE_AA)

def main():
    # 1) 收集模型
    model_paths = scan_models(MODELS_DIR, MODEL_PATTERN, RECURSIVE, SORT_BY, REVERSE_SORT)
    if not model_paths:
        raise SystemExit(f"[ERROR] 未在 {MODELS_DIR} 找到 {MODEL_PATTERN} 匹配的权重。")
    print("[INFO] 级联模型顺序：")
    for i, p in enumerate(model_paths, 1):
        print(f"  {i:02d}. {p}")

    # 2) 打开视频 IO
    if not VIDEO_IN.exists():
        raise SystemExit(f"[ERROR] 视频不存在：{VIDEO_IN}")
    cap = cv2.VideoCapture(str(VIDEO_IN))
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] 无法打开视频：{VIDEO_IN}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if VIDEO_OUT.suffix.lower()==".mp4" else "XVID"))
    writer = cv2.VideoWriter(str(VIDEO_OUT), fourcc, float(src_fps), (W, H))
    if not writer.isOpened():
        raise SystemExit("[ERROR] 无法创建输出视频，尝试更换输出后缀或编码器（mp4v/XVID/MJPG/H264）。")

    # 3) 懒加载模型（第一次用到再载入，避免一次性全占显存）
    model_cache: Dict[Path, YOLO] = {}
    def get_model(p: Path) -> YOLO:
        if p not in model_cache:
            print(f"[LOAD] {p}")
            model_cache[p] = YOLO(str(p))
        return model_cache[p]

    # 4) 帧循环：按级联顺序推理，首个有检出者生效
    idx = 0
    names_cache = {}  # 记住每个模型的 names
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        used_res = None
        used_model_name = ""

        for w in model_paths:
            model = get_model(w)
            # YOLO 可以直接吃 numpy 的 BGR 帧
            results = model(
                source=frame,
                imgsz=IMGSZ, conf=CONF_THRES, iou=IOU_THRES,
                device=DEVICE, verbose=False
            )
            res = results[0]
            n_boxes = int(res.boxes.shape[0]) if res.boxes is not None else 0
            if n_boxes > 0:
                used_res = res
                used_model_name = w.name
                break  # 命中即停

        if used_res is not None:
            boxes = used_res.boxes.xyxy.cpu().numpy()
            clss  = used_res.boxes.cls.cpu().numpy().astype(int)
            confs = used_res.boxes.conf.cpu().numpy()
            # 类名：优先取当前模型的 names
            if used_model_name not in names_cache:
                names_cache[used_model_name] = getattr(model_cache[w], "names", {})
            draw_boxes(frame, boxes, clss, confs, names_cache[used_model_name], model_tag=used_model_name)

        writer.write(frame)
        idx += 1
        if idx % 50 == 0:
            print(f"[INFO] 进度：{idx} 帧")

    cap.release()
    writer.release()
    print(f"[DONE] {VIDEO_OUT}  (共 {idx} 帧, FPS≈{src_fps:.2f}, 尺寸={W}x{H})")

if __name__ == "__main__":
    main()
