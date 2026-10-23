#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HivisionIDPhotos 极简高性能 API 服务
专为与 print-server 对接定制：
- 纯 JSON + 本地文件路径通信（零拷贝、零 Base64、极速响应）
- 跨平台支持：Windows 与 Linux 下完全统一
- 内存常驻 ONNX 推理会话与 MTCNN / RetinaFace 检测器
- 支持自动检测与自定义端口 (--port / PORT 环境变量，默认 17520)
"""
import os
import argparse
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from hivision import IDCreator
from hivision.error import FaceError
from hivision.creator.choose_handler import choose_handler
from hivision.creator.face_detector import get_mtcnn_detector, get_yunet_detector
from hivision.creator.retinaface import retinaface_detect_faces
from hivision.utils import save_image_dpi_to_file, get_resource_path

app = FastAPI(title="HivisionIDPhotos Fast Service")

# 跨域支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

creator = IDCreator()
_RETINAFACE_SESS = None
_RETINAFACE_MODEL_PATH = get_resource_path(
    "hivision", "creator", "retinaface", "weights", "retinaface-resnet50.onnx"
)


# ================== 跨平台图像读取与人脸/裁切算法 ==================

def read_image_file(file_path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """
    跨平台安全读取图像：通过 np.fromfile 免疫 Windows/Linux 下中文路径及空格问题
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到输入文件: {file_path}")

    # Linux 下某些特殊文件系统也能直接通过 np.fromfile 读取
    try:
        data = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(data, flags)
    except Exception:
        img = cv2.imread(file_path, flags)

    if img is None:
        raise ValueError(f"无法解码图像文件: {file_path}")
    return img


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

    if model == "yunet":
        detector = get_yunet_detector()
        h_r, w_r = resized.shape[:2]
        detector.setInputSize((w_r, h_r))
        _, dets = detector.detect(resized)
        if dets is not None:
            for det in dets:
                x, y, w_box, h_box = det[0:4]
                score = det[-1]
                x1 = x / scale
                y1 = y / scale
                x2 = (x + w_box) / scale
                y2 = (y + h_box) / scale
                faces.append({
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "score": float(score),
                })
    elif model == "mtcnn":
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


# ================== 纯 JSON 请求模型 ==================

class MattingRequest(BaseModel):
    input_path: str
    output_path: str
    human_matting_model: str = "rmbg-1.4"
    dpi: int = 300


class CropRequest(BaseModel):
    input_path: str
    output_standard_path: str
    output_hd_path: str
    height: int = 413
    width: int = 295
    face_detect_model: str = "mtcnn"
    hd: bool = True
    dpi: int = 300
    head_measure_ratio: float = 0.2
    head_height_ratio: float = 0.45
    top_distance_max: float = 0.12
    top_distance_min: float = 0.10


class CropHintRequest(BaseModel):
    input_path: str
    target_height: float = 101.6
    target_width: float = 152.4
    face_detect_model: str = "retinaface-resnet50"
    padding_ratio: float = 0.25
    detect_max_side: int = 640


# ================== 核心 API 端点 ==================

@app.get("/")
@app.get("/health")
def health_check():
    """轻量健康检查端点"""
    return {
        "status": "ok",
        "service": "hivision-idphoto-engine",
        "version": "2.0-pure-json"
    }


@app.post("/human_matting")
def human_matting_endpoint(req: MattingRequest):
    """
    人像透明抠图：纯本地文件直传，免 Base64、免 Multipart
    """
    try:
        img = read_image_file(req.input_path, cv2.IMREAD_COLOR)
        choose_handler(creator, req.human_matting_model, None)

        result = creator(img, change_bg_only=True)
        h, w = result.standard.shape[:2]

        # 确保输出目录存在
        out_dir = os.path.dirname(req.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        save_image_dpi_to_file(result.standard, req.output_path, dpi=req.dpi)

        return {
            "status": True,
            "width": w,
            "height": h,
            "output_path": req.output_path,
        }
    except FaceError as fe:
        return {"status": False, "error": str(fe)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/idphoto_crop")
def idphoto_crop_endpoint(req: CropRequest):
    """
    证件照人像识别裁剪：直接对透明 PNG 裁剪并输出标准照与高清照
    """
    try:
        img = read_image_file(req.input_path, cv2.IMREAD_UNCHANGED)
        choose_handler(creator, face_detect_option=req.face_detect_model)
        size = (int(req.height), int(req.width))

        result = creator(
            img,
            size=size,
            head_measure_ratio=req.head_measure_ratio,
            head_height_ratio=req.head_height_ratio,
            head_top_range=(req.top_distance_max, req.top_distance_min),
            crop_only=True,
        )

        std_dir = os.path.dirname(req.output_standard_path)
        if std_dir:
            os.makedirs(std_dir, exist_ok=True)
        save_image_dpi_to_file(result.standard, req.output_standard_path, dpi=req.dpi)

        if req.hd and req.output_hd_path:
            hd_dir = os.path.dirname(req.output_hd_path)
            if hd_dir:
                os.makedirs(hd_dir, exist_ok=True)
            save_image_dpi_to_file(result.hd, req.output_hd_path, dpi=req.dpi)

        return {
            "status": True,
            "output_standard_path": req.output_standard_path,
            "output_hd_path": req.output_hd_path if req.hd else None,
        }
    except FaceError as fe:
        return {"status": False, "error": str(fe)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/photo_crop_hint")
def photo_crop_hint_endpoint(req: CropHintRequest):
    """
    智能裁切建议框计算：基于人脸识别计算出目标宽高比最佳安全框
    """
    try:
        img = read_image_file(req.input_path, cv2.IMREAD_COLOR)
        img_h, img_w = img.shape[:2]
        faces = []
        detect_error = None
        face_detect_model = req.face_detect_model or "retinaface-resnet50"

        try:
            faces = _detect_faces(img, model=face_detect_model, max_side=req.detect_max_side)
        except Exception as exc:
            detect_error = str(exc)

        main_face = _pick_main_face(faces, img_w, img_h)
        face_box = None
        if main_face:
            x1, y1, x2, y2 = main_face["box"]
            face_box = [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)]

        crop_box, target_ratio = _suggest_crop_box(
            img.shape, float(req.target_height), float(req.target_width), face_box, padding_ratio=max(0, req.padding_ratio)
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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    import uvicorn

    parser = argparse.ArgumentParser(description="HivisionIDPhotos Fast Service")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"), help="绑定 IP 地址")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "17520")), help="监听端口 (默认 17520)")
    args = parser.parse_args()

    print(f"[HIVISION] 服务启动成功: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
