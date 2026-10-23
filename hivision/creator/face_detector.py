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
_RETINAFACE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "retinaface", "weights", "retinaface-resnet50.onnx"
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
        faces, landmarks = detector.detect(ctx.origin_image)
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
