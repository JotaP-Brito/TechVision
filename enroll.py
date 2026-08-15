import os
import cv2
from config import (
    DATASET_DIR, CASCADE_PATH, CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    PHOTOS_PER_MEMBER, CAPTURE_DELAY_FRAMES, FACE_SIZE
)
from database import init_db, add_member


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
    print("Look at the camera. Move your head slightly between captures.")
    print("Press 'q' to cancel.\n")

    captured = 0
    frame_counter = 0

    while captured < PHOTOS_PER_MEMBER:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f"Captured: {captured}/{PHOTOS_PER_MEMBER}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Enrollment - press q to cancel", frame)

        # Only try to save when exactly one face is visible, and pace captures out
        if len(faces) == 1:
            frame_counter += 1
            if frame_counter >= CAPTURE_DELAY_FRAMES:
                (x, y, w, h) = faces[0]
                face_img = gray[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, FACE_SIZE)
                filename = os.path.join(member_dir, f"img_{captured}.jpg")
                cv2.imwrite(filename, face_img)
                captured += 1
                frame_counter = 0
                print(f"Saved photo {captured}/{PHOTOS_PER_MEMBER}")
        else:
            frame_counter = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Enrollment cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured == PHOTOS_PER_MEMBER:
        print(f"\nEnrollment complete for '{name}' (ID {member_id}).")
        print("Run train_model.py to update the recognition model.")
    else:
        print(f"\nEnrollment incomplete ({captured}/{PHOTOS_PER_MEMBER} photos saved).")


if __name__ == "__main__":
    main()
