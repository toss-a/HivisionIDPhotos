#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HivisionIDPhotos 高性能精简 API 服务
专为与 Node.js (print-server) 高效对接设计：
- 支持本地文件路径直传模式（零内存拷贝、零 Base64、极速响应）
- 兼容原有 HTTP multipart/form-data 与 base64 模式
- 内存常驻 ONNX 推理会话，告别每次请求重新加载大模型
- 默认监听 17520 端口，可通过 --port 或环境变量 PORT 自定义
"""
import os
import cv2
import argparse
import numpy as np
from typing import Optional
from fastapi import FastAPI, UploadFile, Form, File, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from hivision import IDCreator
from hivision.error import FaceError
from hivision.creator.choose_handler import choose_handler
from hivision.creator.face_detector import get_mtcnn_detector
from hivision.creator.retinaface import retinaface_detect_faces
from hivision.utils import (
    bytes_2_base64,
    base64_2_numpy,
    save_image_dpi_to_bytes,
    save_image_dpi_to_file,
)

app = FastAPI(title="HivisionIDPhotos Fast Service")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

creator = IDCreator()
_RETINAFACE_SESS = None
_RETINAFACE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "hivision",
    "creator",
    "retinaface",
    "weights",
    "retinaface-resnet50.onnx",
)


# ================== 辅助函数 ==================

def _resize_for_detect(image: np.ndarray, max_side: int = 640):
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
    return image, scale


def _pick_main_face(faces, img_w, img_h):
    if not faces:
        return None
    cx_img = img_w / 2
    cy_img = img_h / 2

    def sort_key(face):
        x1, y1, x2, y2 = face["box"]
        area = (x2 - x1) * (y2 - y1)
        cx = x1 + (x2 - x1) / 2
        cy = y1 + (y2 - y1) / 2
        dist_penalty = ((cx - cx_img) / img_w) ** 2 + ((cy - cy_img) / img_h) ** 2
        return (face.get("score", 0), area, -dist_penalty)

    return sorted(faces, key=sort_key, reverse=True)[0]


def _detect_faces(image_bgr: np.ndarray, model: str = "retinaface-resnet50", max_side: int = 640):
    global _RETINAFACE_SESS
    resized, scale = _resize_for_detect(image_bgr, max_side=max_side)
    scale = scale if scale > 0 else 1.0
    faces = []

    if model == "mtcnn":
        detector = get_mtcnn_detector()
        dets, _ = detector.detect(resized)
        if dets is not None:
            for det in dets:
                if len(det) >= 5:
                    x1, y1, x2, y2, score = det[:5]
                    faces.append({
                        "box": [float(x1 / scale), float(y1 / scale), float(x2 / scale), float(y2 / scale)],
                        "score": float(score),
                    })
    else:
        if not os.path.exists(_RETINAFACE_MODEL_PATH):
            raise FileNotFoundError(f"未找到 RetinaFace 权重: {_RETINAFACE_MODEL_PATH}")
        dets, _RETINAFACE_SESS = retinaface_detect_faces(
            resized, _RETINAFACE_MODEL_PATH, sess=_RETINAFACE_SESS
        )
        for det in dets:
            if len(det) >= 5:
                x1, y1, x2, y2, score = det[:5]
                faces.append({
                    "box": [float(x1 / scale), float(y1 / scale), float(x2 / scale), float(y2 / scale)],
                    "score": float(score),
                })
    return faces


def _suggest_crop_box(img_shape, target_height, target_width, face_box=None, padding_ratio=0.25):
    img_h, img_w = img_shape[:2]
    if target_height <= 0 or target_width <= 0:
        target_ratio = img_w / img_h
    else:
        target_ratio = float(target_width) / float(target_height)

    img_ratio = img_w / img_h
    if img_ratio > target_ratio:
        base_h = img_h
        base_w = base_h * target_ratio
    else:
        base_w = img_w
        base_h = base_w / target_ratio

    crop_w, crop_h = base_w, base_h
    cx = img_w / 2
    cy = img_h / 2

    if face_box:
        x1, y1, x2, y2 = face_box
        box_w = x2 - x1
        box_h = y2 - y1
        cx = x1 + box_w / 2
        cy = y1 + box_h / 2

        pad_w = box_w * padding_ratio
        pad_h = box_h * padding_ratio
        need_w = box_w + pad_w * 2
        need_h = box_h + pad_h * 2

        crop_w = max(crop_w, need_w, need_h * target_ratio)
        crop_h = max(crop_h, need_h, need_w / target_ratio)

        if crop_w > img_w:
            crop_w = img_w
            crop_h = crop_w / target_ratio
        if crop_h > img_h:
            crop_h = img_h
            crop_w = crop_h * target_ratio

    half_w = crop_w / 2
    half_h = crop_h / 2
    cx = min(max(cx, half_w), img_w - half_w)
    cy = min(max(cy, half_h), img_h - half_h)

    left = max(0, cx - half_w)
    top = max(0, cy - half_h)

    return {
        "x": int(round(left)),
        "y": int(round(top)),
        "width": int(round(crop_w)),
        "height": int(round(crop_h)),
    }, target_ratio


def _read_image_local_or_bytes(path=None, file_bytes=None, base64_str=None, flags=cv2.IMREAD_COLOR):
    if path and os.path.exists(path):
        img = cv2.imread(path, flags)
        if img is not None:
            return img
    if base64_str:
        return base64_2_numpy(base64_str)
    if file_bytes:
        nparr = np.frombuffer(file_bytes, np.uint8)
        return cv2.imdecode(nparr, flags)
    return None


# ================== Pydantic 请求模型 ==================

class MattingJsonRequest(BaseModel):
    input_path: str
    output_path: Optional[str] = None
    human_matting_model: str = "rmbg-1.4"
    dpi: int = 300


class CropJsonRequest(BaseModel):
    input_path: str
    output_standard_path: Optional[str] = None
    output_hd_path: Optional[str] = None
    height: int = 413
    width: int = 295
    face_detect_model: str = "mtcnn"
    hd: bool = True
    dpi: int = 300
    head_measure_ratio: float = 0.2
    head_height_ratio: float = 0.45
    top_distance_max: float = 0.12
    top_distance_min: float = 0.10


class CropHintJsonRequest(BaseModel):
    input_path: Optional[str] = None
    input_image_base64: Optional[str] = None
    target_height: float = 101.6
    target_width: float = 152.4
    face_detect_model: str = "retinaface-resnet50"
    padding_ratio: float = 0.25
    detect_max_side: int = 640


# ================== API 路由 ==================

@app.get("/")
@app.get("/health")
async def health_check():
    """轻量健康检查端点"""
    return {
        "status": "ok",
        "service": "hivision-idphoto-engine",
        "version": "2.0-optimized"
    }


@app.post("/human_matting")
async def human_matting_endpoint(request: Request):
    """
    人像抠图接口：
    1. 若 Content-Type 为 application/json，走极速本地文件通道（零拷贝）。
    2. 若 Content-Type 为 multipart/form-data，走兼容回退通道。
    """
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        body = await request.json()
        req = MattingJsonRequest(**body)

        if not os.path.exists(req.input_path):
            return JSONResponse({"status": False, "error": f"输入文件不存在: {req.input_path}"}, status_code=400)

        img = cv2.imread(req.input_path, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"status": False, "error": "无法解码输入图片"}, status_code=400)

        choose_handler(creator, req.human_matting_model, None)

        try:
            result = creator(img, change_bg_only=True)
            out_img = cv2.cvtColor(result.standard, cv2.COLOR_RGBA2BGRA)
            h, w = out_img.shape[:2]

            if req.output_path:
                save_image_dpi_to_file(result.standard, req.output_path, dpi=req.dpi)
                return {
                    "status": True,
                    "width": w,
                    "height": h,
                    "output_path": req.output_path,
                }
            else:
                png_bytes = save_image_dpi_to_bytes(out_img, dpi=req.dpi)
                return {
                    "status": True,
                    "width": w,
                    "height": h,
                    "image_base64": bytes_2_base64(png_bytes),
                }
        except FaceError as fe:
            return {"status": False, "error": str(fe)}
        except Exception as exc:
            return JSONResponse({"status": False, "error": str(exc)}, status_code=500)

    # 兼容回退：Form/Multipart 请求
    form = await request.form()
    input_image = form.get("input_image")
    input_base64 = form.get("input_image_base64")
    model = form.get("human_matting_model", "rmbg-1.4")
    dpi = int(form.get("dpi", 300))

    file_bytes = await input_image.read() if input_image else None
    img = _read_image_local_or_bytes(file_bytes=file_bytes, base64_str=input_base64, flags=cv2.IMREAD_COLOR)

    if img is None:
        return {"status": False, "error": "未获取到图片数据"}

    choose_handler(creator, model, None)
    try:
        result = creator(img, change_bg_only=True)
        out_bytes = save_image_dpi_to_bytes(cv2.cvtColor(result.standard, cv2.COLOR_RGBA2BGRA), None, dpi)
        return {
            "status": True,
            "image_base64": bytes_2_base64(out_bytes),
        }
    except FaceError as fe:
        return {"status": False, "error": str(fe)}
    except Exception as exc:
        return {"status": False, "error": str(exc)}


@app.post("/idphoto_crop")
async def idphoto_crop_endpoint(request: Request):
    """
    证件照智能裁剪接口：
    1. 支持 JSON 本地文件通道（零拷贝）。
    2. 支持 Multipart / Base64 回退。
    """
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        body = await request.json()
        req = CropJsonRequest(**body)

        if not os.path.exists(req.input_path):
            return JSONResponse({"status": False, "error": f"输入文件不存在: {req.input_path}"}, status_code=400)

        img = cv2.imread(req.input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return JSONResponse({"status": False, "error": "无法解码输入透明图片"}, status_code=400)

        choose_handler(creator, face_detect_option=req.face_detect_model)
        size = (int(req.height), int(req.width))

        try:
            result = creator(
                img,
                size=size,
                head_measure_ratio=req.head_measure_ratio,
                head_height_ratio=req.head_height_ratio,
                head_top_range=(req.top_distance_max, req.top_distance_min),
                crop_only=True,
            )

            res = {"status": True}
            if req.output_standard_path:
                save_image_dpi_to_file(result.standard, req.output_standard_path, dpi=req.dpi)
                res["output_standard_path"] = req.output_standard_path
            else:
                std_bytes = save_image_dpi_to_bytes(cv2.cvtColor(result.standard, cv2.COLOR_RGBA2BGRA), None, req.dpi)
                res["image_base64_standard"] = bytes_2_base64(std_bytes)

            if req.hd:
                if req.output_hd_path:
                    save_image_dpi_to_file(result.hd, req.output_hd_path, dpi=req.dpi)
                    res["output_hd_path"] = req.output_hd_path
                else:
                    hd_bytes = save_image_dpi_to_bytes(cv2.cvtColor(result.hd, cv2.COLOR_RGBA2BGRA), None, req.dpi)
                    res["image_base64_hd"] = bytes_2_base64(hd_bytes)

            return res
        except FaceError as fe:
            return {"status": False, "error": str(fe)}
        except Exception as exc:
            return JSONResponse({"status": False, "error": str(exc)}, status_code=500)

    # 兼容回退：Form/Multipart 请求
    form = await request.form()
    input_image = form.get("input_image")
    input_base64 = form.get("input_image_base64")
    height = int(form.get("height", 413))
    width = int(form.get("width", 295))
    face_model = form.get("face_detect_model", "mtcnn")
    hd = str(form.get("hd", "true")).lower() == "true"
    dpi = int(form.get("dpi", 300))
    head_measure_ratio = float(form.get("head_measure_ratio", 0.2))
    head_height_ratio = float(form.get("head_height_ratio", 0.45))
    top_distance_max = float(form.get("top_distance_max", 0.12))
    top_distance_min = float(form.get("top_distance_min", 0.10))

    file_bytes = await input_image.read() if input_image else None
    img = _read_image_local_or_bytes(file_bytes=file_bytes, base64_str=input_base64, flags=cv2.IMREAD_UNCHANGED)

    if img is None:
        return {"status": False, "error": "未获取到图片数据"}

    choose_handler(creator, face_detect_option=face_model)
    try:
        result = creator(
            img,
            size=(height, width),
            head_measure_ratio=head_measure_ratio,
            head_height_ratio=head_height_ratio,
            head_top_range=(top_distance_max, top_distance_min),
            crop_only=True,
        )
        std_bytes = save_image_dpi_to_bytes(cv2.cvtColor(result.standard, cv2.COLOR_RGBA2BGRA), None, dpi)
        res = {
            "status": True,
            "image_base64_standard": bytes_2_base64(std_bytes),
        }
        if hd:
            hd_bytes = save_image_dpi_to_bytes(cv2.cvtColor(result.hd, cv2.COLOR_RGBA2BGRA), None, dpi)
            res["image_base64_hd"] = bytes_2_base64(hd_bytes)
        return res
    except FaceError as fe:
        return {"status": False, "error": str(fe)}
    except Exception as exc:
        return {"status": False, "error": str(exc)}


@app.post("/photo_crop_hint")
async def photo_crop_hint_endpoint(request: Request):
    """
    照片智能裁切建议接口（用于根据人脸计算目标比例的裁切框）：
    - 支持 JSON 或 Multipart
    - 快速人脸定位并返回建议裁切框
    """
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        body = await request.json()
        req = CropHintJsonRequest(**body)
        img = _read_image_local_or_bytes(path=req.input_path, base64_str=req.input_image_base64, flags=cv2.IMREAD_COLOR)
        target_height = req.target_height
        target_width = req.target_width
        face_detect_model = req.face_detect_model
        padding_ratio = req.padding_ratio
        detect_max_side = req.detect_max_side
    else:
        form = await request.form()
        input_image = form.get("input_image")
        input_base64 = form.get("input_image_base64")
        file_bytes = await input_image.read() if input_image else None
        img = _read_image_local_or_bytes(file_bytes=file_bytes, base64_str=input_base64, flags=cv2.IMREAD_COLOR)
        target_height = float(form.get("target_height", 101.6))
        target_width = float(form.get("target_width", 152.4))
        face_detect_model = form.get("face_detect_model", "retinaface-resnet50")
        padding_ratio = float(form.get("padding_ratio", 0.25))
        detect_max_side = int(form.get("detect_max_side", 640))

    if img is None:
        return {"status": False, "error": "未获取到有效图片"}

    img_h, img_w = img.shape[:2]
    faces = []
    detect_error = None
    face_detect_model = face_detect_model or "retinaface-resnet50"

    try:
        faces = _detect_faces(img, model=face_detect_model, max_side=detect_max_side)
    except Exception as exc:
        detect_error = str(exc)

    main_face = _pick_main_face(faces, img_w, img_h)
    face_box = None
    if main_face:
        x1, y1, x2, y2 = main_face["box"]
        face_box = [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)]

    crop_box, target_ratio = _suggest_crop_box(
        img.shape, float(target_height), float(target_width), face_box, padding_ratio=max(0, padding_ratio)
    )

    response = {
        "status": True,
        "detected": bool(main_face),
        "model": face_detect_model,
        "image_size": {"width": img_w, "height": img_h},
        "target_ratio": target_ratio,
        "crop_box": crop_box,
        "detect_error": detect_error,
    }

    if main_face:
        response["face_box"] = {
            "x1": int(round(face_box[0])),
            "y1": int(round(face_box[1])),
            "x2": int(round(face_box[2])),
            "y2": int(round(face_box[3])),
            "score": float(main_face.get("score", 0)),
        }

    return response


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="HivisionIDPhotos Fast Service")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"), help="绑定 IP")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "17520")), help="绑定端口")
    args = parser.parse_args()

    print(f"[HIVISION] 启动服务: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
