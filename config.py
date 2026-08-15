import os
import cv2

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "model.yml")
DB_PATH = os.path.join(BASE_DIR, "gym.db")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

# ---- Camera ----
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ---- Enrollment ----
PHOTOS_PER_MEMBER = 5          # how many face samples to capture per member
CAPTURE_DELAY_FRAMES = 8       # frames to wait between each saved photo (avoids near-duplicates)
FACE_SIZE = (200, 200)         # all saved/trained faces are resized to this

# ---- Recognition ----
# LBPH gives a DISTANCE score: lower = more confident match.
# Tune this after real-world testing. Start conservative (lower number).
CONFIDENCE_THRESHOLD = 65
ATTENDANCE_COOLDOWN_SECONDS = 30  # avoid re-logging the same person every frame

os.makedirs(DATASET_DIR, exist_ok=True)
