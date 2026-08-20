import os
import shutil
import cv2
from config import (
    DATASET_DIR, CAMERA_INDEX, PHOTOS_PER_MEMBER
)
from database import add_member, deactivate_member, init_db, mark_dataset_changed
from vision import (
    VisionSetupError,
    ensure_valid_member_name,
    load_face_cascade,
    open_camera,
    prepare_face_image,
    sanitize_member_name,
)


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

    try:
        name = ensure_valid_member_name(input("Enter member name: "))
    except ValueError as exc:
        print(exc)
        return

    member_id = add_member(name)
    safe = sanitize_member_name(name)
    member_dir = os.path.join(DATASET_DIR, f"{member_id}_{safe}")
    os.makedirs(member_dir, exist_ok=True)

    try:
        face_cascade = load_face_cascade()
        cap = open_camera(CAMERA_INDEX)
    except VisionSetupError as exc:
        print(f"Error: {exc}")
        deactivate_member(member_id)
        shutil.rmtree(member_dir, ignore_errors=True)
        return

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

        # Header bar for readability (use frame dimensions, not config)
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
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
            deactivate_member(member_id)
            shutil.rmtree(member_dir, ignore_errors=True)
            break

        if key == ord(' ') and ready:
            (x, y, w, h) = faces[0]
            face_img = prepare_face_image(gray, x, y, w, h)
            filename = os.path.join(member_dir, f"img_{captured}.jpg")
            cv2.imwrite(filename, face_img)
            captured += 1
            print(f"Saved photo {captured}/{PHOTOS_PER_MEMBER} ({prompt})")

    cap.release()
    cv2.destroyAllWindows()

    if captured == PHOTOS_PER_MEMBER:
        mark_dataset_changed()
        print(f"\nEnrollment complete for '{name}' (ID {member_id}).")
        print("Run train_model.py to update the recognition model.")
    else:
        deactivate_member(member_id)
        shutil.rmtree(member_dir, ignore_errors=True)
        print(f"\nEnrollment incomplete ({captured}/{PHOTOS_PER_MEMBER} photos saved).")


if __name__ == "__main__":
    main()
