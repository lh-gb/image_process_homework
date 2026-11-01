"""Unicode 路径、EXIF 方向与图像导出。"""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw


def load_image(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'找不到图片：{path}')
    with Image.open(path) as im:
        rgb = np.array(ImageOps.exif_transpose(im).convert('RGB'))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix not in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
        raise ValueError(f'不支持的图像扩展名：{suffix}')
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if suffix in ('.jpg', '.jpeg') else []
    ok, data = cv2.imencode(suffix, image, params)
    if not ok:
        raise OSError(f'图像编码失败：{path}')
    data.tofile(str(path))
    return str(path)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', 'utf-8')


def make_grid(images, labels, columns=3, cell_size=(420, 310)):
    w, h = cell_size
    canvas = Image.new('RGB', (w*columns, h*((len(images)+columns-1)//columns)), '#edf0f5')
    draw = ImageDraw.Draw(canvas)
    for i, (arr, label) in enumerate(zip(images, labels)):
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if arr.ndim == 3 else arr
        im = ImageOps.contain(Image.fromarray(rgb), (w-20, h-38))
        x, y = (i % columns)*w, (i // columns)*h
        canvas.paste(im, (x+(w-im.width)//2, y+8+(h-38-im.height)//2))
        draw.text((x+12, y+h-24), str(label), fill='#202938')
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
