import os
import re

import cv2

from config import (
    CAMERA_INDEX,
    CASCADE_PATH,
    FACE_SIZE,
    FRAME_HEIGHT,
    FRAME_WIDTH,
)


class VisionSetupError(RuntimeError):
    """Raised when a required vision component cannot be initialized."""


def sanitize_member_name(name: str) -> str:
    """Replace characters that are problematic in file and folder names."""
    return re.sub(r"[^A-Za-z0-9_\- ]", "_", name).strip()


def ensure_valid_member_name(name: str) -> str:
    cleaned = " ".join(name.split()).strip()
    if not cleaned:
        raise ValueError("Member name cannot be empty.")

    safe_name = sanitize_member_name(cleaned)
    if not safe_name:
        raise ValueError("Member name must include letters or numbers.")

    return cleaned


def create_face_recognizer():
    if not hasattr(cv2, "face"):
        raise VisionSetupError(
            "OpenCV face recognition is unavailable. Install 'opencv-contrib-python'."
        )

    return cv2.face.LBPHFaceRecognizer_create(
        radius=2,
        neighbors=8,
        grid_x=8,
        grid_y=8,
    )


def load_face_cascade():
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        raise VisionSetupError(
            f"Could not load the face detector cascade from '{CASCADE_PATH}'."
        )
    return cascade


def open_camera(camera_index: int = CAMERA_INDEX):
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        cap.release()
        raise VisionSetupError(f"Could not open camera at index {camera_index}.")

    return cap


def prepare_face_image(gray_frame, x: int, y: int, w: int, h: int):
    face_img = gray_frame[y:y + h, x:x + w]
    face_img = cv2.resize(face_img, FACE_SIZE)
    return cv2.equalizeHist(face_img)


def model_file_exists(model_path: str) -> bool:
    return os.path.exists(model_path)
