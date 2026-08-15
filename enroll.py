import os
import cv2
from config import (
    DATASET_DIR, CASCADE_PATH, CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    PHOTOS_PER_MEMBER, FACE_SIZE
)
from database import init_db, add_member

# Pose prompts cycled through during enrollment, one per photo (repeats if
# PHOTOS_PER_MEMBER is larger than this list). Varying pose/angle on purpose
# gives the recognizer more to work with than 15 near-identical shots.
POSE_PROMPTS = [
    "Look straight at the camera",
    "Turn head slightly LEFT",
    "Turn head slightly RIGHT",
    "Tilt head slightly UP",
    "Tilt head slightly DOWN",
    "Look straight, neutral expression",
    "Look straight, slight smile",
    "Turn head a bit more LEFT",
    "Turn head a bit more RIGHT",
]


def main():
    init_db()

    name = input("Enter member name: ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    member_id = add_member(name)
    member_dir = os.path.join(DATASET_DIR, f"{member_id}_{name}")
    os.makedirs(member_dir, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print(f"\nEnrolling '{name}' (ID {member_id}).")
    print("Follow the on-screen pose prompt, then press SPACE to capture that photo.")
    print("Press 'q' to cancel.\n")

    captured = 0

    while captured < PHOTOS_PER_MEMBER:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        prompt = POSE_PROMPTS[captured % len(POSE_PROMPTS)]

        # Header bar for readability
        cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"Photo {captured + 1}/{PHOTOS_PER_MEMBER}: {prompt}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        ready = len(faces) == 1
        status = "Ready - press SPACE" if ready else "Position one face in frame"
        status_color = (0, 255, 0) if ready else (0, 0, 255)
        cv2.putText(frame, status, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        cv2.imshow("Enrollment - SPACE to capture, q to cancel", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Enrollment cancelled.")
            break

        if key == ord(' ') and ready:
            (x, y, w, h) = faces[0]
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, FACE_SIZE)
            face_img = cv2.equalizeHist(face_img)  # normalize lighting for consistent matching
            filename = os.path.join(member_dir, f"img_{captured}.jpg")
            cv2.imwrite(filename, face_img)
            captured += 1
            print(f"Saved photo {captured}/{PHOTOS_PER_MEMBER} ({prompt})")

    cap.release()
    cv2.destroyAllWindows()

    if captured == PHOTOS_PER_MEMBER:
        print(f"\nEnrollment complete for '{name}' (ID {member_id}).")
        print("Run train_model.py to update the recognition model.")
    else:
        print(f"\nEnrollment incomplete ({captured}/{PHOTOS_PER_MEMBER} photos saved).")


if __name__ == "__main__":
    main()
