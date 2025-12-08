from fastapi import FastAPI, UploadFile, Form, File
from hivision import IDCreator
from hivision.error import FaceError
from hivision.creator.layout_calculator import (
    generate_layout_array,
    generate_layout_image,
    generate_mixed_layout_image,
)
from hivision.creator.choose_handler import choose_handler
from hivision.utils import (
    add_background,
    resize_image_to_kb,
    bytes_2_base64,
    base64_2_numpy,
    hex_to_rgb,
    add_watermark,
    save_image_dpi_to_bytes,
    convert_layout_direction,
)
from hivision.creator.retinaface import retinaface_detect_faces
import numpy as np
import cv2
import os
from starlette.middleware.cors import CORSMiddleware
from starlette.formparsers import MultiPartParser

# 设置Starlette表单字段大小限制
MultiPartParser.max_part_size = 10 * 1024 * 1024  # 10MB
# 设置Starlette文件上传大小限制
MultiPartParser.max_file_size = 20 * 1024 * 1024   # 20MB

app = FastAPI()
creator = IDCreator()
retinaface_sess = None
mtcnn_detector = None
RETINAFACE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "hivision",
    "creator",
    "retinaface",
    "weights",
    "retinaface-resnet50.onnx",
)


async def _read_image(input_image, input_image_base64, flags=cv2.IMREAD_COLOR):
    if input_image_base64:
        img = base64_2_numpy(input_image_base64)
    else:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, flags)

    if img is None:
        return img

    if len(img.shape) == 2 and flags == cv2.IMREAD_COLOR:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4 and flags == cv2.IMREAD_COLOR:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

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
    global retinaface_sess, mtcnn_detector

    resized, scale = _resize_for_detect(image_bgr, max_side=max_side)
    scale = scale if scale > 0 else 1.0

    faces = []
    if model == "mtcnn":
        try:
            if mtcnn_detector is None:
                from mtcnnruntime import MTCNN

                mtcnn_detector = MTCNN()
            dets, _ = mtcnn_detector.detect(resized)
        except Exception as exc:
            raise RuntimeError(f"MTCNN 初始化或推理失败: {exc}") from exc

        if dets is None:
            return []

        for det in dets:
            if len(det) < 5:
                continue
            x1, y1, x2, y2, score = det[:5]
            faces.append(
                {
                    "box": [float(x1 / scale), float(y1 / scale), float(x2 / scale), float(y2 / scale)],
                    "score": float(score),
                }
            )
    else:
        if not os.path.exists(RETINAFACE_MODEL_PATH):
            raise FileNotFoundError(f"未找到 RetinaFace 模型权重: {RETINAFACE_MODEL_PATH}")
        try:
            dets, retinaface_sess = retinaface_detect_faces(
                resized, RETINAFACE_MODEL_PATH, sess=retinaface_sess
            )
        except Exception as exc:
            raise RuntimeError(f"RetinaFace 推理失败: {exc}") from exc

        for det in dets:
            if len(det) < 5:
                continue
            x1, y1, x2, y2, score = det[:5]
            faces.append(
                {
                    "box": [float(x1 / scale), float(y1 / scale), float(x2 / scale), float(y2 / scale)],
                    "score": float(score),
                }
            )

    return faces


def _suggest_crop_box(img_shape, target_height, target_width, face_box=None, padding_ratio=0.25):
    """
    尽量保留更多画面：先取在原图内面积最大的目标比例框，再偏向人脸并确保人脸+padding 能容纳。
    """
    img_h, img_w = img_shape[:2]
    if target_height <= 0 or target_width <= 0:
        target_ratio = img_w / img_h
    else:
        target_ratio = float(target_width) / float(target_height)

    # 原图内能容纳的最大目标比例框
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

        # 保证人脸+padding 能放下，同时不小于基础最大取景
        crop_w = max(crop_w, need_w, need_h * target_ratio)
        crop_h = max(crop_h, need_h, need_w / target_ratio)

        # 限制不超过原图
        if crop_w > img_w:
            crop_w = img_w
            crop_h = crop_w / target_ratio
        if crop_h > img_h:
            crop_h = img_h
            crop_w = crop_h * target_ratio

    # 将中心偏向人脸但保持在范围内
    half_w = crop_w / 2
    half_h = crop_h / 2
    cx = min(max(cx, half_w), img_w - half_w)
    cy = min(max(cy, half_h), img_h - half_h)

    left = max(0, cx - half_w)
    top = max(0, cy - half_h)

    crop_box = {
        "x": int(round(left)),
        "y": int(round(top)),
        "width": int(round(crop_w)),
        "height": int(round(crop_h)),
    }
    return crop_box, target_ratio

# 添加 CORS 中间件 解决跨域问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的请求来源
    allow_credentials=True,  # 允许携带 Cookie
    allow_methods=[
        "*"
    ],  # 允许的请求方法，例如：GET, POST 等，也可以指定 ["GET", "POST"]
    allow_headers=["*"],  # 允许的请求头，也可以指定具体的头部
)


# 证件照智能制作接口
@app.post("/idphoto")
async def idphoto_inference(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    height: int = Form(413),
    width: int = Form(295),
    human_matting_model: str = Form("modnet_photographic_portrait_matting"),
    face_detect_model: str = Form("mtcnn"),
    hd: bool = Form(True),
    dpi: int = Form(300),
    face_align: bool = Form(False),
    whitening_strength: int = Form(0),
    head_measure_ratio: float = Form(0.2),
    head_height_ratio: float = Form(0.45),
    top_distance_max: float = Form(0.12),
    top_distance_min: float = Form(0.10),
    brightness_strength: float = Form(0),
    contrast_strength: float = Form(0),
    sharpen_strength: float = Form(0),
    saturation_strength: float = Form(0),
):  
    # 如果传入了base64，则直接使用base64解码
    if input_image_base64:
        img = base64_2_numpy(input_image_base64)
    # 否则使用上传的图片
    else:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # 将BGR转换为RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ------------------- 选择抠图与人脸检测模型 -------------------
    choose_handler(creator, human_matting_model, face_detect_model)

    # 将字符串转为元组
    size = (int(height), int(width))
    try:
        result = creator(
            img,
            size=size,
            head_measure_ratio=head_measure_ratio,
            head_height_ratio=head_height_ratio,
            head_top_range=(top_distance_max, top_distance_min),
            face_alignment=face_align,
            whitening_strength=whitening_strength,
            brightness_strength=brightness_strength,
            contrast_strength=contrast_strength,
            sharpen_strength=sharpen_strength,
            saturation_strength=saturation_strength,
        )
    except FaceError:
        result_message = {"status": False}
    # 如果检测到人脸数量等于1, 则返回标准证和高清照结果（png 4通道图像）
    else:
        result_image_standard_bytes = save_image_dpi_to_bytes(result.standard, None, dpi)
        
        result_message = {
            "status": True,
            "image_base64_standard": bytes_2_base64(result_image_standard_bytes),
        }

        # 如果hd为True, 则增加高清照结果（png 4通道图像）
        if hd:
            result_image_hd_bytes = save_image_dpi_to_bytes(result.hd, None, dpi)
            result_message["image_base64_hd"] = bytes_2_base64(result_image_hd_bytes)

    return result_message


# 人像抠图接口
@app.post("/human_matting")
async def human_matting_inference(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    human_matting_model: str = Form("hivision_modnet"),
    dpi: int = Form(300),
):
    if input_image_base64:
        img = base64_2_numpy(input_image_base64)
    else:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # ------------------- 选择抠图与人脸检测模型 -------------------
    choose_handler(creator, human_matting_model, None)

    try:
        result = creator(
            img,
            change_bg_only=True,
        )
    except FaceError:
        result_message = {"status": False}

    else:
        result_image_standard_bytes = save_image_dpi_to_bytes(cv2.cvtColor(result.standard, cv2.COLOR_RGBA2BGRA), None, dpi)
        result_message = {
            "status": True,
            "image_base64": bytes_2_base64(result_image_standard_bytes),
        }
    return result_message


# 透明图像添加纯色背景接口
@app.post("/add_background")
async def photo_add_background(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    color: str = Form("000000"),
    kb: int = Form(None),
    dpi: int = Form(300),
    render: int = Form(0),
):
    render_choice = ["pure_color", "updown_gradient", "center_gradient"]

    if input_image_base64:
        img = base64_2_numpy(input_image_base64)
    else:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    color = hex_to_rgb(color)
    color = (color[2], color[1], color[0])

    result_image = add_background(
        img,
        bgr=color,
        mode=render_choice[render],
    ).astype(np.uint8)

    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    if kb:
        result_image_bytes = resize_image_to_kb(result_image, None, int(kb), dpi=dpi)
    else:
        result_image_bytes = save_image_dpi_to_bytes(result_image, None, dpi=dpi)

    result_messgae = {
        "status": True,
        "image_base64": bytes_2_base64(result_image_bytes),
    }

    return result_messgae


# 六寸排版照生成接口
@app.post("/generate_layout_photos")
async def generate_layout_photos(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    height: int = Form(413),
    width: int = Form(295),
    kb: int = Form(None),
    dpi: int = Form(300),
    layout_direction: str = Form("auto"),
    crop_line: bool = Form(False),
):
    # try:
    if input_image_base64:
        img = base64_2_numpy(input_image_base64)
    else:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    size = (int(height), int(width))

    # 转换排版方向参数
    layout_direction_internal = convert_layout_direction(layout_direction)
    
    typography_arr, typography_rotate = generate_layout_array(
        input_height=size[0], input_width=size[1],
        layout_direction=layout_direction_internal
    )

    result_layout_image = generate_layout_image(
        img, typography_arr, typography_rotate, height=size[0], width=size[1], crop_line=crop_line
    ).astype(np.uint8)

    result_layout_image = cv2.cvtColor(result_layout_image, cv2.COLOR_RGB2BGR)
    if kb:
        result_layout_image_bytes = resize_image_to_kb(
            result_layout_image, None, int(kb), dpi=dpi
        )
    else:
        result_layout_image_bytes = save_image_dpi_to_bytes(result_layout_image, None, dpi=dpi)
        
    result_layout_image_base64 = bytes_2_base64(result_layout_image_bytes)

    result_messgae = {
        "status": True,
        "image_base64": result_layout_image_base64,
    }

    return result_messgae


# 透明图像添加水印接口
@app.post("/watermark")
async def watermark(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    text: str = Form("Hello"),
    size: int = 20,
    opacity: float = 0.5,
    angle: int = 30,
    color: str = "#000000",
    space: int = 25,
    kb: int = Form(None),
    dpi: int = Form(300),
):
    if input_image_base64:
        img = base64_2_numpy(input_image_base64)
    else:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        result_image = add_watermark(img, text, size, opacity, angle, color, space)

        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        if kb:
            result_image_bytes = resize_image_to_kb(result_image, None, int(kb), dpi=dpi)
        else:
            result_image_bytes = save_image_dpi_to_bytes(result_image, None, dpi=dpi)
        result_image_base64 = bytes_2_base64(result_image_bytes)

        result_messgae = {
            "status": True,
            "image_base64": result_image_base64,
        }
    except Exception as e:
        result_messgae = {
            "status": False,
            "error": str(e),
        }

    return result_messgae


@app.post("/photo_crop_hint")
async def photo_crop_hint(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    target_height: float = Form(101.6),
    target_width: float = Form(152.4),
    face_detect_model: str = Form("retinaface-resnet50"),
    padding_ratio: float = Form(0.25),
    detect_max_side: int = Form(640),
):
    """
    提供给照片打印的裁切建议接口：
    - 默认使用 RetinaFace 人脸检测，可选 mtcnn
    - 先将长边缩放到 detect_max_side 后再检测，降低 CPU 开销
    - 检测失败时返回居中裁切方案
    """
    try:
        img = await _read_image(input_image, input_image_base64, flags=cv2.IMREAD_COLOR)
    except Exception as exc:
        return {"status": False, "error": f"读取图片失败: {exc}"}

    if img is None:
        return {"status": False, "error": "未获取到有效图片"}

    img_h, img_w = img.shape[:2]
    faces = []
    detect_error = None
    face_detect_model = face_detect_model or "retinaface-resnet50"

    try:
        faces = _detect_faces(img, model=face_detect_model, max_side=int(detect_max_side))
    except Exception as exc:
        detect_error = str(exc)

    main_face = _pick_main_face(faces, img_w, img_h)
    face_box = None
    if main_face:
        x1, y1, x2, y2 = main_face["box"]
        face_box = [
            max(0, x1),
            max(0, y1),
            min(img_w, x2),
            min(img_h, y2),
        ]

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


# 设置照片KB值接口(RGB图)
@app.post("/set_kb")
async def set_kb(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    dpi: int = Form(300),
    kb: int = Form(50),
):
    if input_image_base64:
        img = base64_2_numpy(input_image_base64)
    else:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        result_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result_image_bytes = resize_image_to_kb(result_image, None, int(kb), dpi=dpi)
        result_image_base64 = bytes_2_base64(result_image_bytes)

        result_messgae = {
            "status": True,
            "image_base64": result_image_base64,
        }
    except Exception as e:
        result_messgae = {
            "status": False,
            "error": e,
        }

    return result_messgae


# 证件照智能裁剪接口
@app.post("/idphoto_crop")
async def idphoto_crop_inference(
    input_image: UploadFile = File(None),
    input_image_base64: str = Form(None),
    height: int = Form(413),
    width: int = Form(295),
    face_detect_model: str = Form("mtcnn"),
    hd: bool = Form(True),
    dpi: int = Form(300),
    head_measure_ratio: float = Form(0.2),
    head_height_ratio: float = Form(0.45),
    top_distance_max: float = Form(0.12),
    top_distance_min: float = Form(0.10),
):
    if input_image_base64:
        img = base64_2_numpy(input_image_base64)
    else:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)  # 读取图像(4通道)

    # ------------------- 选择抠图与人脸检测模型 -------------------
    choose_handler(creator, face_detect_option=face_detect_model)

    # 将字符串转为元组
    size = (int(height), int(width))
    try:
        result = creator(
            img,
            size=size,
            head_measure_ratio=head_measure_ratio,
            head_height_ratio=head_height_ratio,
            head_top_range=(top_distance_max, top_distance_min),
            crop_only=True,
        )
    except FaceError:
        result_message = {"status": False}
    # 如果检测到人脸数量等于1, 则返回标准证和高清照结果（png 4通道图像）
    else:
        result_image_standard_bytes = save_image_dpi_to_bytes(cv2.cvtColor(result.standard, cv2.COLOR_RGBA2BGRA), None, dpi)
        
        result_message = {
            "status": True,
            "image_base64_standard": bytes_2_base64(result_image_standard_bytes),
        }

        # 如果hd为True, 则增加高清照结果（png 4通道图像）
        if hd:
            result_image_hd_bytes = save_image_dpi_to_bytes(cv2.cvtColor(result.hd, cv2.COLOR_RGBA2BGRA), None, dpi)
            result_message["image_base64_hd"] = bytes_2_base64(result_image_hd_bytes)

    return result_message


# 混合排版接口：4张一寸+2张两寸
@app.post("/generate_mixed_layout_photos")
async def generate_mixed_layout_photos(
    input_image_1_inch: UploadFile = File(None),
    input_image_1_inch_base64: str = Form(None),
    input_image_2_inch: UploadFile = File(None),
    input_image_2_inch_base64: str = Form(None),
    kb: int = Form(None),
    dpi: int = Form(300),
    crop_line: bool = Form(False),
):
    # 读取一寸照片
    if input_image_1_inch_base64:
        img_1_inch = base64_2_numpy(input_image_1_inch_base64)
    else:
        image_bytes = await input_image_1_inch.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_1_inch = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 读取两寸照片
    if input_image_2_inch_base64:
        img_2_inch = base64_2_numpy(input_image_2_inch_base64)
    else:
        image_bytes = await input_image_2_inch.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_2_inch = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 生成混合排版图像
    result_layout_image = generate_mixed_layout_image(
        img_1_inch, img_2_inch, crop_line=crop_line
    ).astype(np.uint8)

    result_layout_image = cv2.cvtColor(result_layout_image, cv2.COLOR_RGB2BGR)
    if kb:
        result_layout_image_bytes = resize_image_to_kb(
            result_layout_image, None, int(kb), dpi=dpi
        )
    else:
        result_layout_image_bytes = save_image_dpi_to_bytes(result_layout_image, None, dpi=dpi)

    result_layout_image_base64 = bytes_2_base64(result_layout_image_bytes)

    result_message = {
        "status": True,
        "image_base64": result_layout_image_base64,
    }

    return result_message


if __name__ == "__main__":
    import uvicorn

    # 在8080端口运行推理服务
    uvicorn.run(app, host="0.0.0.0", port=8080)
