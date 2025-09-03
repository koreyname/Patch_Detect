# -*- coding: utf-8 -*-
"""
把一组图片合成为连续视频（所有参数固定在下方常量中）。
仅需：修改常量 -> 运行：python images_to_video_fixed.py
"""
from pathlib import Path
import re
import cv2
import numpy as np
from PIL import Image, ImageOps

# ====================== 固定参数区域（按需改这里） ======================
# 输入：可以是目录（优先）或通配符（如 "data/**/*.jpg"）
INPUT_PATH = "/root/autodl-tmp/ultralytics-main/datasets/final/pure/images/train"      # 例：r"D:\photos" 或 "./data/**/*.jpg"
RECURSIVE = True                      # 若 INPUT_PATH 是目录：是否递归子目录
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
MAX_IMAGES = 0                        # 0 表示不限制，>0 表示最多取前 N 张

# 输出视频
OUTPUT_PATH = "./out.mp4"

# 视频参数
FPS = 30.0
SECONDS_PER_IMAGE = 0.0               # >0：每张图片停留秒数；=0：每张占 1 帧
SIZE = (1920, 1080)                   # 输出分辨率 (宽, 高)；若为 None 则取第一张图片尺寸
FIT_MODE = "contain"                  # "contain"(等比留黑) / "cover"(等比裁剪铺满) / "stretch"(直接拉伸)
FILL_RGB = (0, 0, 0)                  # contain 模式留边颜色
CODEC = "mp4v"                        # 常见：mp4v / XVID / MJPG / H264（依赖本机编解码支持）
# ======================================================================


def natural_key(s):
    """自然排序 key（img2 < img10）"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]

def load_and_orient(path: Path) -> Image.Image:
    """读取并按 EXIF 旋转纠正方向"""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")

def resize_fit(im: Image.Image, target, mode: str, fill=(0, 0, 0)) -> Image.Image:
    tw, th = target
    w, h = im.size
    if mode == "stretch":
        return im.resize((tw, th), Image.LANCZOS)

    scale_contain = min(tw / w, th / h)
    scale_cover = max(tw / w, th / h)

    if mode == "cover":
        nw, nh = int(w * scale_cover), int(h * scale_cover)
        im2 = im.resize((nw, nh), Image.LANCZOS)
        left = (nw - tw) // 2
        top = (nh - th) // 2
        return im2.crop((left, top, left + tw, top + th))
    else:  # contain
        nw, nh = int(w * scale_contain), int(h * scale_contain)
        im2 = im.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGB", (tw, th), fill)
        canvas.paste(im2, ((tw - nw) // 2, (th - nh) // 2))
        return canvas

def gather_images(input_path: str, exts, recursive: bool):
    p = Path(input_path)
    if p.exists() and p.is_dir():
        it = p.rglob("*") if recursive else p.glob("*")
        files = [x for x in it if x.suffix.lower() in exts]
    else:
        # 当作通配符处理（支持 ** 递归）
        files = list(Path().glob(input_path))
        files = [x for x in files if x.suffix.lower() in exts]
    return sorted(files, key=natural_key)

def main():
    # 收集图片
    files = gather_images(INPUT_PATH, EXTS, RECURSIVE)
    if not files:
        raise SystemExit("未找到图片，请检查 INPUT_PATH/EXTS/RECURSIVE 设置。")

    if MAX_IMAGES > 0:
        files = files[:MAX_IMAGES]

    # 决定目标分辨率
    if SIZE is None:
        im0 = load_and_orient(files[0])
        target_size = im0.size  # (w, h)
    else:
        target_size = SIZE

    # 打开视频写入器
    fourcc = cv2.VideoWriter_fourcc(*CODEC)
    writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, FPS, target_size)
    if not writer.isOpened():
        raise SystemExit(f"无法打开 VideoWriter。可尝试更换 CODEC（当前：{CODEC}）或输出后缀。")

    # 决定每张图片重复帧数
    if SECONDS_PER_IMAGE > 0:
        frames_per_image = max(1, int(round(FPS * SECONDS_PER_IMAGE)))
    else:
        frames_per_image = 1

    total_frames = 0
    n = len(files)
    print(f"共 {n} 张图片 -> {OUTPUT_PATH}")
    print(f"FPS={FPS}, 每图帧数={frames_per_image}, 分辨率={target_size}, 模式={FIT_MODE}, 编码={CODEC}")

    for idx, path in enumerate(files, 1):
        im = load_and_orient(path)
        im = resize_fit(im, target_size, mode=FIT_MODE, fill=FILL_RGB)
        frame = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        for _ in range(frames_per_image):
            writer.write(frame)
            total_frames += 1
        if idx % 100 == 0 or idx == n:
            print(f"进度：{idx}/{n}")

    writer.release()
    duration = total_frames / float(FPS)
    print("完成。")
    print(f"帧数：{total_frames}，时长：{duration:.2f}s")

if __name__ == "__main__":
    # 避免某些环境未显式导入 Image
    from PIL import Image
    main()
