#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2024/9/5 16:45
@File: __init__.py
@Description:
    创建证件照（精简版：仅保留透明背景与证件照裁剪核心逻辑）
"""
import numpy as np
from typing import Tuple
import hivision.creator.utils as U
from .context import Context, ContextHandler, Params, Result
from .human_matting import extract_human
from .face_detector import detect_face_mtcnn
from .photo_adjuster import adjust_photo
import cv2
import time


class IDCreator:
    """
    证件照核心创建类：用于背景透明化抠图与基于人脸的自动裁剪
    """

    def __init__(self):
        self.before_all: ContextHandler = None
        self.after_matting: ContextHandler = None
        self.after_detect: ContextHandler = None
        self.after_all: ContextHandler = None

        # 核心处理器
        self.matting_handler: ContextHandler = extract_human
        self.detection_handler: ContextHandler = detect_face_mtcnn
        self.ctx = None

    def __call__(
        self,
        image: np.ndarray,
        size: Tuple[int, int] = (413, 295),
        change_bg_only: bool = False,
        crop_only: bool = False,
        head_measure_ratio: float = 0.2,
        head_height_ratio: float = 0.45,
        head_top_range: float = (0.12, 0.1),
        face: Tuple[int, int, int, int] = None,
        whitening_strength: int = 0,
        brightness_strength: int = 0,
        contrast_strength: int = 0,
        sharpen_strength: int = 0,
        saturation_strength: int = 0,
        face_alignment: bool = False,
        horizontal_flip: bool = False,
    ) -> Result:
        params = Params(
            size=size,
            change_bg_only=change_bg_only,
            head_measure_ratio=head_measure_ratio,
            head_height_ratio=head_height_ratio,
            head_top_range=head_top_range,
            crop_only=crop_only,
            face=face,
            whitening_strength=whitening_strength,
            brightness_strength=brightness_strength,
            contrast_strength=contrast_strength,
            sharpen_strength=sharpen_strength,
            saturation_strength=saturation_strength,
            face_alignment=face_alignment,
            horizontal_flip=horizontal_flip,
        )

        self.ctx = Context(params)
        ctx = self.ctx
        ctx.processing_image = image
        ctx.processing_image = U.resize_image_esp(
            ctx.processing_image, 2000
        )
        ctx.origin_image = ctx.processing_image.copy()
        if self.before_all:
            self.before_all(ctx)

        # 1. 人像抠图
        if not ctx.params.crop_only:
            self.matting_handler(ctx)
            if self.after_matting:
                self.after_matting(ctx)
        else:
            ctx.matting_image = ctx.processing_image

        # 仅抠图直接返回结果
        if ctx.params.change_bg_only:
            ctx.result = Result(
                standard=ctx.matting_image,
                hd=ctx.matting_image,
                matting=ctx.matting_image,
                clothing_params=None,
                typography_params=None,
                face=None,
            )
            if self.after_all:
                self.after_all(ctx)
            return ctx.result

        # 2. 人脸检测
        self.detection_handler(ctx)
        if self.after_detect:
            self.after_detect(ctx)

        # 2.1 人脸对齐（可选）
        if ctx.params.face_alignment and abs(ctx.face.get("roll_angle", 0)) > 2:
            from hivision.creator.rotation_adjust import rotate_bound_4channels
            b, g, r, a = cv2.split(ctx.matting_image)
            ctx.origin_image, ctx.matting_image, _, _, _, _ = rotate_bound_4channels(
                cv2.merge((b, g, r)),
                a,
                -1 * ctx.face["roll_angle"],
            )
            self.detection_handler(ctx)
            if self.after_detect:
                self.after_detect(ctx)

        # 3. 图像裁切调整
        result_image_hd, result_image_standard, clothing_params, typography_params = (
            adjust_photo(ctx)
        )

        ctx.result = Result(
            standard=result_image_standard,
            hd=result_image_hd,
            matting=ctx.matting_image,
            clothing_params=clothing_params,
            typography_params=typography_params,
            face=ctx.face,
        )
        if self.after_all:
            self.after_all(ctx)

        return ctx.result
