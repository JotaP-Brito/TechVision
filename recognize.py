import os
import time
import cv2
from config import (
    MODEL_PATH, CAMERA_INDEX, CONFIDENCE_THRESHOLD, ATTENDANCE_COOLDOWN_SECONDS
)
from database import get_member_name, get_training_status, init_db, log_attendance
from vision import (
    VisionSetupError,
    create_face_recognizer,
    load_face_cascade,
    model_file_exists,
    open_camera,
    prepare_face_image,
)


def main():
    init_db()

    if not model_file_exists(MODEL_PATH):
        print("No trained model found. Run train_model.py first.")
        return

    training_status = get_training_status()
    if not training_status["ready"]:
        print(f"Model is out of date: {training_status['reason']}")
        print("Run train_model.py before starting recognition.")
        return

    try:
        recognizer = create_face_recognizer()
        face_cascade = load_face_cascade()
        cap = open_camera(CAMERA_INDEX)
    except VisionSetupError as exc:
        print(f"Error: {exc}")
        return

    recognizer.read(MODEL_PATH)

    last_logged = {}  # member_id -> last log timestamp
    print("Recognition running. Press 'q' to quit.")

    frame_skip = 3
    frame_count = 0
    last_results = []   # store (x, y, w, h, text, color) for smooth drawing

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        frame_count += 1

        if frame_count % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

            last_results = []
            for (x, y, w, h) in faces:
                face_img = prepare_face_image(gray, x, y, w, h)

                label_id, distance = recognizer.predict(face_img)

                # Determine name and color
                name = get_member_name(label_id) if distance < CONFIDENCE_THRESHOLD else None

                if name is not None:
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

                last_results.append((x, y, w, h, text, color))

        # Draw results every frame (even if not a detection frame)
        for (x, y, w, h, text, color) in last_results:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Gym Face Recognition - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
