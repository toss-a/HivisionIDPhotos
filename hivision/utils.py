#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
精简后的通用图像与数据处理工具
"""
import io
import base64
import numpy as np
import cv2
from PIL import Image


def save_image_dpi_to_bytes(image: np.ndarray, output_image_path: str = None, dpi: int = 300) -> bytes:
    """
    将 OpenCV (BGR/BGRA) 图像数组正确转换并保存为指定 DPI 的 PNG 字节流。
    """
    if len(image.shape) == 3 and image.shape[2] == 4:
        # OpenCV 输出为 BGRA，PIL 期望 RGBA，必须转换通道避免人物皮肤偏蓝
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        pil_img = Image.fromarray(rgba, mode="RGBA")
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # OpenCV 输出为 BGR，PIL 期望 RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb, mode="RGB")
    else:
        pil_img = Image.fromarray(image)

    byte_stream = io.BytesIO()
    pil_img.save(byte_stream, format="PNG", dpi=(dpi, dpi))
    image_bytes = byte_stream.getvalue()

    if output_image_path:
        with open(output_image_path, "wb") as f:
            f.write(image_bytes)

    return image_bytes


def save_image_dpi_to_file(image: np.ndarray, output_image_path: str, dpi: int = 300):
    """
    极速文件直接写入：将 OpenCV (BGR/BGRA) 图像以指定 DPI 保存到目标路径为标准 PNG (RGBA/RGB)
    """
    if len(image.shape) == 3 and image.shape[2] == 4:
        # OpenCV 输出为 BGRA，PIL 期望 RGBA，必须转换通道避免人物皮肤偏蓝
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        pil_img = Image.fromarray(rgba, mode="RGBA")
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # OpenCV 输出为 BGR，PIL 期望 RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb, mode="RGB")
    else:
        pil_img = Image.fromarray(image)

    pil_img.save(output_image_path, format="PNG", dpi=(dpi, dpi))


def bytes_2_base64(img_byte_arr: bytes) -> str:
    """字节流转 base64 字符串"""
    base64_image = base64.b64encode(img_byte_arr).decode("utf-8")
    return "data:image/png;base64," + base64_image


def base64_2_numpy(base64_image: str) -> np.ndarray:
    """base64 字符串解码为 numpy 图像"""
    if base64_image.startswith("data:image"):
        base64_image = base64_image.split(",")[1]
    img_bytes = base64.b64decode(base64_image)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
