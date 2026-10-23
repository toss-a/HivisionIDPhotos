import os
import cv2
import argparse
import numpy as np
from hivision import IDCreator
from hivision.error import FaceError
from hivision.creator.choose_handler import choose_handler
from hivision.utils import save_image_dpi_to_file

parser = argparse.ArgumentParser(description="HivisionIDPhotos 证件照核心推理命令行工具")
parser.add_argument("-t", "--type", choices=["human_matting", "idphoto_crop"], default="human_matting", help="推理类型")
parser.add_argument("-i", "--input", required=True, help="输入图像路径")
parser.add_argument("-o", "--output", required=True, help="输出图像保存路径")
parser.add_argument("--height", type=int, default=413, help="目标高度(px)")
parser.add_argument("--width", type=int, default=295, help="目标宽度(px)")
parser.add_argument("--matting_model", default="rmbg-1.4", choices=["rmbg-1.4", "hivision_modnet", "modnet_photographic_portrait_matting"])
parser.add_argument("--face_detect_model", default="yolov8n-person", choices=["yolov8n-person", "yunet", "mtcnn", "retinaface-resnet50"])
parser.add_argument("--dpi", type=int, default=300, help="输出 DPI")

args = parser.parse_args()

if not os.path.exists(args.input):
    raise FileNotFoundError(f"找不到输入文件: {args.input}")

creator = IDCreator()

if args.type == "human_matting":
    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    choose_handler(creator, matting_model_option=args.matting_model)
    res = creator(img, change_bg_only=True)
    save_image_dpi_to_file(res.standard, args.output, dpi=args.dpi)
    print(f"[OK] 抠图完成，已保存到: {args.output}")

elif args.type == "idphoto_crop":
    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    choose_handler(creator, face_detect_option=args.face_detect_model)
    res = creator(img, size=(args.height, args.width), crop_only=True)
    save_image_dpi_to_file(res.standard, args.output, dpi=args.dpi)
    print(f"[OK] 裁剪完成，已保存到: {args.output}")
