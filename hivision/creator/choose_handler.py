from hivision.creator.human_matting import (
    extract_human,
    extract_human_modnet_photographic_portrait_matting,
    extract_human_rmbg,
)
from hivision.creator.face_detector import (
    detect_face_mtcnn,
    detect_face_retinaface,
    detect_face_yunet,
)
from hivision.creator.human_detector import detect_person_as_face

HUMAN_MATTING_MODELS = [
    "hivision_modnet",
    "modnet_photographic_portrait_matting",
    "rmbg-1.4",
]

FACE_DETECT_MODELS = ["yolov8n-person", "yunet", "mtcnn", "retinaface-resnet50"]


def choose_handler(creator, matting_model_option=None, face_detect_option=None):
    if matting_model_option == "modnet_photographic_portrait_matting":
        creator.matting_handler = extract_human_modnet_photographic_portrait_matting
    elif matting_model_option == "rmbg-1.4":
        creator.matting_handler = extract_human_rmbg
    else:
        creator.matting_handler = extract_human

    if face_detect_option == "yolov8n-person":
        creator.detection_handler = detect_person_as_face
    elif face_detect_option == "retinaface-resnet50":
        creator.detection_handler = detect_face_retinaface
    elif face_detect_option == "yunet":
        creator.detection_handler = detect_face_yunet
    else:
        creator.detection_handler = detect_face_mtcnn
