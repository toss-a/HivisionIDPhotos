#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于人像分割 Mask 与轻量网络的全人体精准检测模块：
1. detect_person_matting: 基于人像 Alpha 蒙版的像素级人体全包络（包含衣服每一寸边缘、袖口、下摆，绝不切衣服）
2. detect_persons_yolov8: 基于 YOLOv8n 的通用人体目标检测
"""
import os
import cv2
import numpy as np
import onnxruntime as ort
from hivision.utils import get_resource_path, get_onnx_session_options
from hivision.creator.context import Context
from hivision.error import FaceError
from hivision.creator.human_matting import get_modnet_matting, WEIGHTS

_YOLO_PERSON_SESS = None
_YOLO_MODEL_PATH = get_resource_path("hivision", "creator", "weights", "yolov8n.onnx")


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


# 保持兼容别名
detect_person_as_face = detect_person_as_face_matting


def get_yolov8_person_session():
    global _YOLO_PERSON_SESS
    if _YOLO_PERSON_SESS is None:
        if not os.path.exists(_YOLO_MODEL_PATH):
            raise FileNotFoundError(f"未找到 YOLOv8 权重文件: {_YOLO_MODEL_PATH}")
        opts = get_onnx_session_options()
        _YOLO_PERSON_SESS = ort.InferenceSession(
            _YOLO_MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"]
        )
    return _YOLO_PERSON_SESS


def detect_persons_yolov8(image_bgr: np.ndarray, conf_thresh: float = 0.35, nms_thresh: float = 0.45):
    """
    YOLOv8 辅助人体目标检测
    """
    sess = get_yolov8_person_session()
    raw = image_bgr
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    elif raw.shape[2] == 4:
        raw = raw[:, :, :3]

    h0, w0 = raw.shape[:2]
    r = min(640.0 / h0, 640.0 / w0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = (640.0 - new_w) / 2.0, (640.0 - new_h) / 2.0

    resized = cv2.resize(raw, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    blob = (cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    out = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0].T

    person_scores = out[:, 4]
    valid_mask = person_scores > conf_thresh
    if not np.any(valid_mask):
        return []

    boxes = out[valid_mask, 0:4]
    scores = person_scores[valid_mask]

    x = (boxes[:, 0] - boxes[:, 2] / 2.0 - dw) / r
    y = (boxes[:, 1] - boxes[:, 3] / 2.0 - dh) / r
    w = boxes[:, 2] / r
    h = boxes[:, 3] / r

    cv_boxes = [[int(x[i]), int(y[i]), int(w[i]), int(h[i])] for i in range(len(x))]
    indices = cv2.dnn.NMSBoxes(cv_boxes, scores.tolist(), conf_thresh, nms_thresh)

    persons = []
    if len(indices) > 0:
        for i in np.array(indices).flatten():
            bx, by, bw, bh = cv_boxes[i]
            x1 = max(0, bx)
            y1 = max(0, by)
            x2 = min(w0, bx + bw)
            y2 = min(h0, by + bh)
            persons.append({
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(scores[i])
            })

    persons.sort(key=lambda p: p["score"], reverse=True)
    return persons
