import os
import cv2
import numpy as np
from config import DATASET_DIR, MODEL_PATH


def load_training_data():
    faces = []
    labels = []

    if not os.path.isdir(DATASET_DIR):
        return faces, labels

    for folder_name in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # folder naming convention: "<member_id>_<name>"
        try:
            member_id = int(folder_name.split("_")[0])
        except ValueError:
            print(f"Skipping folder with unexpected name: {folder_name}")
            continue

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(member_id)

    return faces, labels


def main():
    print("Loading training images from dataset...")
    faces, labels = load_training_data()

    if len(faces) == 0:
        print("No training data found. Enroll at least one member first (enroll.py).")
        return

    print(f"Training on {len(faces)} photos across {len(set(labels))} member(s)...")

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)

    print(f"Model trained and saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
