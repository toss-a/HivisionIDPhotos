#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
人像抠图模块（精简优化版）：
- 支持 hivision_modnet, modnet_photographic_portrait_matting, rmbg-1.4
- 保持 ONNX Runtime Session 内存常驻，避免每次重复从磁盘加载大模型
"""
import os
import cv2
import numpy as np
from PIL import Image
import onnxruntime
from .tensor2numpy import NNormalize, NTo_Tensor, NUnsqueeze
from .context import Context

WEIGHTS = {
    "hivision_modnet": os.path.join(
        os.path.dirname(__file__), "weights", "hivision_modnet.onnx"
    ),
    "modnet_photographic_portrait_matting": os.path.join(
        os.path.dirname(__file__),
        "weights",
        "modnet_photographic_portrait_matting.onnx",
    ),
    "rmbg-1.4": os.path.join(
        os.path.dirname(__file__), "weights", "rmbg-1.4.onnx"
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
    try:
        sess = onnxruntime.InferenceSession(checkpoint_path, providers=providers)
    except Exception:
        sess = onnxruntime.InferenceSession(checkpoint_path, providers=["CPUExecutionProvider"])

    _SESSIONS[model_key] = sess
    return sess


def hollow_out_fix(src: np.ndarray) -> np.ndarray:
    """修补抠图区域，补充局部小空洞"""
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
        a_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return src
    sorted_contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
    a_contour = cv2.drawContours(np.zeros(a_padded.shape, np.uint8), [sorted_contours[0]], -1, 255, 2)
    h, w = a_padded.shape[:2]
    mask = np.zeros([h + 2, w + 2], np.uint8)
    cv2.floodFill(a_contour, mask=mask, seedPoint=(0, 0), newVal=255)
    a_padded = cv2.add(a_padded, 255 - a_contour)
    return cv2.merge((src_bgr, a_padded[10:-10, 10:-10]))


def _image2bgr(input_image):
    if len(input_image.shape) == 2:
        return np.repeat(input_image[:, :, None], 3, axis=2)
    if input_image.shape[2] == 1:
        return np.repeat(input_image, 3, axis=2)
    if input_image.shape[2] == 4:
        return input_image[:, :, :3]
    return input_image


def _read_modnet_image(input_image, ref_size=512):
    im = Image.fromarray(np.uint8(input_image))
    width, length = im.size[0], im.size[1]
    im = np.asarray(im)
    im = _image2bgr(im)
    im = cv2.resize(im, (ref_size, ref_size), interpolation=cv2.INTER_AREA)
    im = NNormalize(im, mean=np.array([0.5, 0.5, 0.5]), std=np.array([0.5, 0.5, 0.5]))
    im = NUnsqueeze(NTo_Tensor(im))
    return im, width, length


def get_modnet_matting(input_image, checkpoint_path, ref_size=512, model_key="hivision_modnet"):
    sess = get_cached_onnx_session(model_key, checkpoint_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    im, width, length = _read_modnet_image(input_image=input_image, ref_size=ref_size)
    matte = sess.run([output_name], {input_name: im})
    matte = (matte[0] * 255).astype("uint8")
    matte = np.squeeze(matte)
    mask = cv2.resize(matte, (width, length), interpolation=cv2.INTER_AREA)

    b, g, r = cv2.split(np.uint8(_image2bgr(input_image)))
    return cv2.merge((b, g, r, mask))


def get_rmbg_matting(input_image, checkpoint_path, ref_size=1024):
    sess = get_cached_onnx_session("rmbg-1.4", checkpoint_path)

    orig_image = Image.fromarray(np.uint8(_image2bgr(input_image)))
    image = orig_image.convert("RGB").resize((ref_size, ref_size), Image.BILINEAR)

    im_np = np.array(image).astype(np.float32)
    im_np = im_np.transpose(2, 0, 1)
    im_np = np.expand_dims(im_np, axis=0)
    im_np = (im_np / 255.0 - 0.5) / 0.5

    result = sess.run(None, {sess.get_inputs()[0].name: im_np})[0]
    result = np.squeeze(result)
    ma = np.max(result)
    mi = np.min(result)
    result = (result - mi) / (ma - mi) if ma > mi else result

    im_array = (result * 255).astype(np.uint8)
    pil_im = Image.fromarray(im_array, mode="L").resize(orig_image.size, Image.BILINEAR)

    new_im = Image.new("RGBA", orig_image.size, (0, 0, 0, 0))
    new_im.paste(orig_image, mask=pil_im)
    return np.array(new_im)


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
