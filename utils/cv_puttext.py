import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _resource_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path.cwd()))
    return Path(__file__).resolve().parent.parent


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font_path = _resource_base_dir() / "fonts" / "platech.ttf"
    font_text = ImageFont.truetype(str(font_path), textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=font_text)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    img_path = "result.jpg"
    img = cv2.imread(img_path)
    save_img = cv2ImgAddText(img, "中国加油！", 50, 100, (255, 0, 0), 50)
    cv2.imwrite("save.jpg", save_img)
