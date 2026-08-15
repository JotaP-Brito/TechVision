import os
import time
import cv2
from config import (
    MODEL_PATH, CASCADE_PATH, CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    FACE_SIZE, CONFIDENCE_THRESHOLD, ATTENDANCE_COOLDOWN_SECONDS
)
from database import init_db, get_member_name, log_attendance


def main():
    init_db()

    if not os.path.exists(MODEL_PATH):
        print("No trained model found. Run train_model.py first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    last_logged = {}  # member_id -> last log timestamp, to avoid spamming attendance

    print("Recognition running. Press 'q' to quit.")

    frame_skip = 3  # only run detection/recognition every Nth frame
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        frame_count += 1
        if frame_count % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

            for (x, y, w, h) in faces:
                face_img = gray[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, FACE_SIZE)

                label_id, distance = recognizer.predict(face_img)

                if distance < CONFIDENCE_THRESHOLD:
                    name = get_member_name(label_id) or "Unknown"
                    color = (0, 255, 0)
                    text = f"{name} ({distance:.0f})"

                    now = time.time()
                    if now - last_logged.get(label_id, 0) > ATTENDANCE_COOLDOWN_SECONDS:
                        log_attendance(label_id, distance)
                        last_logged[label_id] = now
                        print(f"ACCESS GRANTED: {name} (confidence distance: {distance:.1f})")
                else:
                    color = (0, 0, 255)
                    text = f"Unknown ({distance:.0f})"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Gym Face Recognition - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
