#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
人脸检测器（精简优化版）：
- 支持 MTCNN 与 RetinaFace-ResNet50
- 保持模型会话常驻
"""
import os
import cv2
import numpy as np
from .context import Context
from hivision.error import FaceError
from hivision.creator.retinaface import retinaface_detect_faces

try:
    from mtcnnruntime import MTCNN
except ImportError:
    raise ImportError("请确保已安装 mtcnn-runtime: pip install mtcnn-runtime")

_MTCNN = None
_RETINAFACE_SESS = None
from hivision.utils import get_resource_path

_RETINAFACE_MODEL_PATH = get_resource_path(
    "hivision", "creator", "retinaface", "weights", "retinaface-resnet50.onnx"
)


def get_mtcnn_detector():
    global _MTCNN
    if _MTCNN is None:
        _MTCNN = MTCNN()
    return _MTCNN


def detect_face_mtcnn(ctx: Context, scale: int = 2):
    """基于 MTCNN 的人脸检测"""
    detector = get_mtcnn_detector()
    image = cv2.resize(
        ctx.origin_image,
        (ctx.origin_image.shape[1] // scale, ctx.origin_image.shape[0] // scale),
        interpolation=cv2.INTER_AREA,
    )
    faces, landmarks = detector.detect(image, thresholds=[0.8, 0.8, 0.8])

    if len(faces) != 1:
        faces, landmarks = detector.detect(raw_img)
    else:
        for item, param in enumerate(faces[0]):
            faces[0][item] = param * 2

    if len(faces) != 1:
        raise FaceError(f"Expected 1 face, but got {len(faces)}", len(faces))

    left = faces[0][0]
    top = faces[0][1]
    width = faces[0][2] - left + 1
    height = faces[0][3] - top + 1
    ctx.face["rectangle"] = (left, top, width, height)

    landmarks = landmarks[0]
    left_eye = np.array([landmarks[0], landmarks[5]])
    right_eye = np.array([landmarks[1], landmarks[6]])
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    roll_angle = np.degrees(np.arctan2(dy, dx))
    ctx.face["roll_angle"] = roll_angle


def detect_face_retinaface(ctx: Context):
    """基于 RetinaFace-ResNet50 的人脸检测"""
    global _RETINAFACE_SESS

    if not os.path.exists(_RETINAFACE_MODEL_PATH):
        raise FileNotFoundError(f"未找到 RetinaFace 模型: {_RETINAFACE_MODEL_PATH}")

    faces_dets, _RETINAFACE_SESS = retinaface_detect_faces(
        ctx.origin_image,
        _RETINAFACE_MODEL_PATH,
        sess=_RETINAFACE_SESS,
    )

    if len(faces_dets) != 1:
        raise FaceError(f"Expected 1 face, but got {len(faces_dets)}", len(faces_dets))

    face_det = faces_dets[0]
    ctx.face["rectangle"] = (
        face_det[0],
        face_det[1],
        face_det[2] - face_det[0] + 1,
        face_det[3] - face_det[1] + 1,
    )

    landmarks = face_det[5:]
    left_eye = np.array([landmarks[0], landmarks[1]])
    right_eye = np.array([landmarks[2], landmarks[3]])
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    roll_angle = np.degrees(np.arctan2(dy, dx))
    ctx.face["roll_angle"] = roll_angle

_YUNET = None
_YUNET_MODEL_PATH = get_resource_path(
    "hivision", "creator", "weights", "face_detection_yunet_2023mar.onnx"
)


def get_yunet_detector(conf_threshold: float = 0.6, nms_threshold: float = 0.3):
    global _YUNET
    if _YUNET is None:
        if not os.path.exists(_YUNET_MODEL_PATH):
            raise FileNotFoundError(f"未找到 YuNet 模型权重: {_YUNET_MODEL_PATH}")
        _YUNET = cv2.FaceDetectorYN.create(
            model=_YUNET_MODEL_PATH,
            config="",
            input_size=(320, 320),
            score_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            top_k=5000,
        )
    return _YUNET


def detect_face_yunet(ctx: Context, max_side: int = 960):
    """
    基于 OpenCV YuNet 的极速人脸检测（推荐 CPU 推理使用）：
    - 在 AMD 5825U 上仅需 ~10ms，相比 RetinaFace 提速 15~20 倍，仅 227KB
    - 输出严格符合 Context.face 规范（rectangle 与 roll_angle 旋转偏角）
    """
    detector = get_yunet_detector()
    raw_img = ctx.origin_image
    if raw_img.ndim == 2:
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
    elif raw_img.shape[2] == 4:
        raw_img = raw_img[:, :, :3]

    h, w = raw_img.shape[:2]

    # 将大图适度降采样检测以达到极致性能（如 10ms），若未检出则回退原图
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        target_w, target_h = int(w * scale), int(h * scale)
        image = cv2.resize(raw_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        image = raw_img

    img_h, img_w = image.shape[:2]
    detector.setInputSize((img_w, img_h))
    _, faces = detector.detect(image)

    # 若降采样后未检出正好 1 张脸，回退原图尺寸做更细粒度探测
    if (faces is None or len(faces) != 1) and scale != 1.0:
        detector.setInputSize((w, h))
        _, faces = detector.detect(raw_img)
        scale = 1.0

    if faces is None or len(faces) != 1:
        count = 0 if faces is None else len(faces)
        raise FaceError(f"Expected 1 face, but got {count}", count)

    face = faces[0]
    box = face[0:4] / scale
    left = int(round(box[0]))
    top = int(round(box[1]))
    width = int(round(box[2]))
    height = int(round(box[3]))

    # 边界保护
    left = max(0, left)
    top = max(0, top)
    width = min(w - left, max(1, width))
    height = min(h - top, max(1, height))
    ctx.face["rectangle"] = (left, top, width, height)

    # 5 关键点：p0 为视角左眼(人的右眼)，p1 为视角右眼(人的左眼)
    p0 = face[4:6] / scale
    p1 = face[6:8] / scale
    left_eye = p0 if p0[0] < p1[0] else p1
    right_eye = p1 if p0[0] < p1[0] else p0
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    roll_angle = float(np.degrees(np.arctan2(dy, dx)))
    ctx.face["roll_angle"] = roll_angle
