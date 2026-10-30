from hivision.creator.human_matting import (
    extract_human,
    extract_human_modnet_photographic_portrait_matting,
    extract_human_rmbg,
)
from hivision.creator.face_detector import (
    detect_face_mtcnn,
    detect_face_retinaface,
)

HUMAN_MATTING_MODELS = [
    "hivision_modnet",
    "modnet_photographic_portrait_matting",
    "rmbg-1.4",
]

FACE_DETECT_MODELS = ["retinaface-resnet50", "mtcnn"]


def choose_handler(creator, matting_model_option=None, face_detect_option=None):
    if matting_model_option == "modnet_photographic_portrait_matting":
        creator.matting_handler = extract_human_modnet_photographic_portrait_matting
    elif matting_model_option == "rmbg-1.4":
        creator.matting_handler = extract_human_rmbg
    else:
        creator.matting_handler = extract_human

    if face_detect_option == "mtcnn":
        creator.detection_handler = detect_face_mtcnn
    else:
        # 默认使用 retinaface-resnet50
        creator.detection_handler = detect_face_retinaface
