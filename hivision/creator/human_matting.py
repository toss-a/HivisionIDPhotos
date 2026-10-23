#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
人像抠图模块（极致向量化与低延迟优化版）：
- 支持 hivision_modnet, modnet_photographic_portrait_matting, rmbg-1.4
- 保持 ONNX Runtime Session 内存常驻与 CPU 线程自适应优化
- 全流程纯 OpenCV/NumPy 向量化前后处理，剔除慢速 PIL 跨库对象转换与冗余内存拷贝
"""
import os
import cv2
import numpy as np
import onnxruntime
from .context import Context
from hivision.utils import get_resource_path, get_onnx_session_options

WEIGHTS = {
    "hivision_modnet": get_resource_path(
        "hivision", "creator", "weights", "hivision_modnet.onnx"
    ),
    "modnet_photographic_portrait_matting": get_resource_path(
        "hivision",
        "creator",
        "weights",
        "modnet_photographic_portrait_matting.onnx",
    ),
    "rmbg-1.4": get_resource_path(
        "hivision", "creator", "weights", "rmbg-1.4.onnx"
    ),
}

ONNX_DEVICE = onnxruntime.get_device()
ONNX_PROVIDER = (
    "CUDAExecutionProvider" if ONNX_DEVICE == "GPU" else "CPUExecutionProvider"
)

# 内存常驻 Session 缓存
_SESSIONS = {}


def get_cached_onnx_session(model_key: str, checkpoint_path: str):
    """获取或初始化 ONNX 推理会话（持久常驻）"""
    global _SESSIONS
    if model_key in _SESSIONS and _SESSIONS[model_key] is not None:
        return _SESSIONS[model_key]

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"未找到抠图模型权重: {checkpoint_path}")

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ONNX_PROVIDER == "CUDAExecutionProvider"
        else ["CPUExecutionProvider"]
    )
    sess_options = get_onnx_session_options()
    try:
        sess = onnxruntime.InferenceSession(checkpoint_path, sess_options=sess_options, providers=providers)
    except Exception:
        sess = onnxruntime.InferenceSession(checkpoint_path, sess_options=sess_options, providers=["CPUExecutionProvider"])

    _SESSIONS[model_key] = sess
    return sess


def hollow_out_fix(src: np.ndarray) -> np.ndarray:
    """修补抠图区域，补充局部小空洞（优化版：点压缩与加速轮廓计算）"""
    b, g, r, a = cv2.split(src)
    src_bgr = cv2.merge((b, g, r))

    add_area_top = np.zeros((10, a.shape[1]), np.uint8)
    a_padded = np.vstack((add_area_top, a, add_area_top))
    add_area_side = np.zeros((a_padded.shape[0], 10), np.uint8)
    a_padded = np.hstack((add_area_side, a_padded, add_area_side))

    _, a_threshold = cv2.threshold(a_padded, 127, 255, 0)
    a_erode = cv2.erode(
        a_threshold,
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=3,
    )
    contours, _ = cv2.findContours(
        a_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return src
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    a_contour = cv2.drawContours(np.zeros(a_padded.shape, np.uint8), [sorted_contours[0]], -1, 255, 2)
    h, w = a_padded.shape[:2]
    mask = np.zeros([h + 2, w + 2], np.uint8)
    cv2.floodFill(a_contour, mask=mask, seedPoint=(0, 0), newVal=255)
    a_padded = cv2.add(a_padded, 255 - a_contour)
    return cv2.merge((src_bgr, a_padded[10:-10, 10:-10]))


def _image2bgr(input_image: np.ndarray) -> np.ndarray:
    """确保图像为 3 通道 BGR 格式"""
    if input_image.ndim == 2:
        return cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    if input_image.shape[2] == 1:
        return cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    if input_image.shape[2] == 4:
        return input_image[:, :, :3]
    return input_image


def _read_modnet_image(input_image: np.ndarray, ref_size: int = 512):
    """MODNet 极速前处理（纯 NumPy/OpenCV 向量化，零 PIL 跨库分配）"""
    length, width = input_image.shape[:2]
    img_bgr = _image2bgr(input_image)
    resized = cv2.resize(img_bgr, (ref_size, ref_size), interpolation=cv2.INTER_AREA)
    # 归一化: (x - 127.5) / 127.5，并转 NCHW float32
    im = (resized.astype(np.float32) - 127.5) * (1.0 / 127.5)
    im = np.transpose(im, (2, 0, 1))[None, ...]
    return im, width, length


def get_modnet_matting(input_image, checkpoint_path, ref_size=512, model_key="hivision_modnet"):
    sess = get_cached_onnx_session(model_key, checkpoint_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    im, width, length = _read_modnet_image(input_image=input_image, ref_size=ref_size)
    matte = sess.run([output_name], {input_name: im})[0]
    matte = np.squeeze(matte)
    matte = np.clip(matte * 255.0, 0, 255).astype(np.uint8)
    mask = cv2.resize(matte, (width, length), interpolation=cv2.INTER_AREA)

    img_bgr = _image2bgr(input_image)
    b, g, r = cv2.split(img_bgr)
    return cv2.merge((b, g, r, mask))


def get_rmbg_matting(input_image, checkpoint_path, ref_size=1024):
    sess = get_cached_onnx_session("rmbg-1.4", checkpoint_path)
    h, w = input_image.shape[:2]
    img_bgr = _image2bgr(input_image)

    # 极速前处理：RGB 空间 resize + float32 归一化
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (ref_size, ref_size), interpolation=cv2.INTER_LINEAR)
    im_np = (resized.astype(np.float32) - 127.5) * (1.0 / 127.5)
    im_np = np.transpose(im_np, (2, 0, 1))[None, ...]

    result = sess.run(None, {sess.get_inputs()[0].name: im_np})[0]
    result = np.squeeze(result)
    ma = np.max(result)
    mi = np.min(result)
    if ma > mi:
        result = (result - mi) / (ma - mi)

    # 极速后处理：纯 OpenCV 向量化缩放与 BGRA 通道合并，彻底替代慢速 PIL paste
    mask = cv2.resize((result * 255.0).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
    b, g, r = cv2.split(img_bgr)
    return cv2.merge((b, g, r, mask))


def extract_human(ctx: Context):
    matting_image = get_modnet_matting(ctx.processing_image, WEIGHTS["hivision_modnet"], model_key="hivision_modnet")
    ctx.processing_image = hollow_out_fix(matting_image)
    ctx.matting_image = ctx.processing_image.copy()


def extract_human_modnet_photographic_portrait_matting(ctx: Context):
    matting_image = get_modnet_matting(
        ctx.processing_image,
        WEIGHTS["modnet_photographic_portrait_matting"],
        model_key="modnet_photographic_portrait_matting",
    )
    ctx.processing_image = matting_image
    ctx.matting_image = ctx.processing_image.copy()


def extract_human_rmbg(ctx: Context):
    matting_image = get_rmbg_matting(ctx.processing_image, WEIGHTS["rmbg-1.4"])
    ctx.processing_image = matting_image
    ctx.matting_image = ctx.processing_image.copy()
