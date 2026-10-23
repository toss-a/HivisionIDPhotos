#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于人像分割 Mask 的全人体精准包络检测模块：
- detect_person_matting: 基于人像 Alpha 蒙版的像素级人体全包络（包含衣服每一寸边缘、袖口、下摆，绝不切衣服）
- detect_person_as_face_matting: 证件照模式下的自适应胸像定位
"""
import numpy as np
from hivision.creator.context import Context
from hivision.error import FaceError
from hivision.creator.human_matting import get_modnet_matting, WEIGHTS


def detect_person_matting(image_bgr_or_rgba: np.ndarray, min_area_ratio: float = 0.01):
    """
    基于人像分割 Mask 提取真实的像素级人体外接边界（包含全部衣服边缘，绝不切衣服）：
    - 若输入已包含 Alpha 通道（如已被抠图），耗时 < 1ms，直接提取非零像素边界
    - 若输入为 3 通道原图，调用极致优化的 MODNet 推理提取真实人像 Mask
    返回格式: [{"box": [x1, y1, x2, y2], "score": 1.0}]
    """
    if image_bgr_or_rgba.ndim == 3 and image_bgr_or_rgba.shape[2] == 4:
        alpha = image_bgr_or_rgba[:, :, 3]
    else:
        matted = get_modnet_matting(image_bgr_or_rgba, WEIGHTS["hivision_modnet"])
        alpha = matted[:, :, 3]

    y_idx, x_idx = np.where(alpha > 127)
    if len(x_idx) == 0:
        return []

    x1 = float(np.min(x_idx))
    y1 = float(np.min(y_idx))
    x2 = float(np.max(x_idx))
    y2 = float(np.max(y_idx))

    # 面积过滤，防止极小噪点干扰
    h, w = image_bgr_or_rgba.shape[:2]
    area = (x2 - x1) * (y2 - y1)
    if area < (h * w * min_area_ratio):
        return []

    return [{
        "box": [x1, y1, x2, y2],
        "score": 1.0
    }]


def detect_person_as_face_matting(ctx: Context):
    """
    证件照模式下的 Mask 人像全包络：
    直接依据真实人体轮廓确定定位框，彻底杜绝切衣服和无正脸报错。
    """
    persons = detect_person_matting(ctx.origin_image)
    if not persons:
        raise FaceError("未在画面中检测到有效人像主体", 0)

    x1, y1, x2, y2 = persons[0]["box"]
    pw = x2 - x1
    ph = y2 - y1

    top = int(y1)
    height = int(max(1, ph * 0.45))
    left = int(x1 + pw * 0.15)
    width = int(max(1, pw * 0.70))

    ctx.face["rectangle"] = (left, top, width, height)
    ctx.face["roll_angle"] = 0.0
