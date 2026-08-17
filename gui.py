import os
import time
import threading
import sqlite3
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

from config import (
    DATASET_DIR, MODEL_PATH, CASCADE_PATH, CAMERA_INDEX,
    FRAME_WIDTH, FRAME_HEIGHT, PHOTOS_PER_MEMBER, FACE_SIZE,
    CONFIDENCE_THRESHOLD, ATTENDANCE_COOLDOWN_SECONDS, DB_PATH
)
from database import init_db, add_member, deactivate_member, get_all_active_members, get_member_name, log_attendance

# Pose prompts for enrollment (same as enroll.py)
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

class TechVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TechVision - Gym Entry System")
        self.root.geometry("900x700")

        # Initialize database
        init_db()

        # Main layout
        self.create_widgets()

    def create_widgets(self):
        # Top navigation bar
        nav_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        nav_frame.pack(fill=tk.X)

        tk.Button(nav_frame, text="📸 Enroll", command=self.show_enroll_panel,
                  bg="#3498db", fg="white", font=("Arial", 12), padx=20, pady=10).pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(nav_frame, text="🧠 Train", command=self.show_train_panel,
                  bg="#3498db", fg="white", font=("Arial", 12), padx=20, pady=10).pack(side=tk.LEFT, padx=10)
        tk.Button(nav_frame, text="🚪 Recognize", command=self.show_recognize_panel,
                  bg="#3498db", fg="white", font=("Arial", 12), padx=20, pady=10).pack(side=tk.LEFT, padx=10)
        tk.Button(nav_frame, text="📋 Attendance", command=self.show_attendance_panel,
                  bg="#3498db", fg="white", font=("Arial", 12), padx=20, pady=10).pack(side=tk.LEFT, padx=10)
        tk.Button(nav_frame, text="👥 Members", command=self.show_members_panel,
                  bg="#3498db", fg="white", font=("Arial", 12), padx=20, pady=10).pack(side=tk.LEFT, padx=10)

        # Content area
        self.content = tk.Frame(self.root)
        self.content.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                              bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Initialize with enrollment panel
        self.show_enroll_panel()

    def clear_content(self):
        for widget in self.content.winfo_children():
            widget.destroy()

    # ---------- Enrollment Panel ----------
    def show_enroll_panel(self):
        self.clear_content()
        self.current_panel = "enroll"

        frame = tk.Frame(self.content, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(frame, text="Member Enrollment", font=("Arial", 18, "bold")).pack(pady=10)

        # Name entry
        name_frame = tk.Frame(frame)
        name_frame.pack(pady=5)
        tk.Label(name_frame, text="Member Name:").pack(side=tk.LEFT)
        self.name_entry = tk.Entry(name_frame, width=30)
        self.name_entry.pack(side=tk.LEFT, padx=10)

        # Camera display area
        self.video_label = tk.Label(frame, bg="black")
        self.video_label.pack(pady=10)

        # Control buttons
        btn_frame = tk.Frame(frame)
        btn_frame.pack(pady=10)
        self.enroll_start_btn = tk.Button(btn_frame, text="Start Enrollment", command=self.start_enrollment,
                                          bg="#27ae60", fg="white", padx=20, pady=5)
        self.enroll_start_btn.pack(side=tk.LEFT, padx=5)
        self.enroll_cancel_btn = tk.Button(btn_frame, text="Cancel", command=self.cancel_enrollment,
                                           bg="#e74c3c", fg="white", padx=20, pady=5, state=tk.DISABLED)
        self.enroll_cancel_btn.pack(side=tk.LEFT, padx=5)

        self.enroll_status_label = tk.Label(frame, text="", font=("Arial", 12))
        self.enroll_status_label.pack(pady=5)

    def start_enrollment(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a member name.")
            return

        # Create member in database and folder
        self.member_id = add_member(name)
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip()
        self.member_dir = os.path.join(DATASET_DIR, f"{self.member_id}_{safe_name}")
        os.makedirs(self.member_dir, exist_ok=True)

        self.captured = 0
        self.last_capture_time = 0
        self.capture_delay = 1.0  # seconds between auto captures

        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera.")
            return

        self.enroll_start_btn.config(state=tk.DISABLED)
        self.enroll_cancel_btn.config(state=tk.NORMAL)
        self.enroll_status_label.config(text=f"Enrolling {name} - follow the prompts...")

        self.update_enrollment_frame()

    def update_enrollment_frame(self):
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.enroll_status_label.config(text="Failed to read from camera.")
            self.stop_camera()
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        # Draw face rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show current prompt
        if self.captured < PHOTOS_PER_MEMBER:
            prompt = POSE_PROMPTS[self.captured % len(POSE_PROMPTS)]
            # Header bar
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
            cv2.putText(frame, f"Photo {self.captured + 1}/{PHOTOS_PER_MEMBER}: {prompt}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Auto capture if exactly one face detected and delay elapsed
            if len(faces) == 1 and (time.time() - self.last_capture_time) > self.capture_delay:
                (x, y, w, h) = faces[0]
                face_img = gray[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, FACE_SIZE)
                face_img = cv2.equalizeHist(face_img)
                filename = os.path.join(self.member_dir, f"img_{self.captured}.jpg")
                cv2.imwrite(filename, face_img)
                self.captured += 1
                self.last_capture_time = time.time()
                print(f"Captured {self.captured}/{PHOTOS_PER_MEMBER}")

                if self.captured >= PHOTOS_PER_MEMBER:
                    self.enroll_status_label.config(text="Enrollment complete! You can now train the model.")
                    self.stop_camera()
                    self.enroll_start_btn.config(state=tk.NORMAL)
                    self.enroll_cancel_btn.config(state=tk.DISABLED)
                    return
        else:
            # Enrollment finished
            self.stop_camera()
            self.enroll_start_btn.config(state=tk.NORMAL)
            self.enroll_cancel_btn.config(state=tk.DISABLED)
            return

        # Convert to RGB for Tkinter
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Schedule next frame
        self.video_label.after(15, self.update_enrollment_frame)

    def cancel_enrollment(self):
        # Clean up partial data
        if hasattr(self, 'member_id'):
            deactivate_member(self.member_id)
            import shutil
            shutil.rmtree(self.member_dir, ignore_errors=True)
        self.stop_camera()
        self.enroll_status_label.config(text="Enrollment cancelled.")
        self.enroll_start_btn.config(state=tk.NORMAL)
        self.enroll_cancel_btn.config(state=tk.DISABLED)

    def stop_camera(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.video_label.config(image='')

    # ---------- Training Panel ----------
    def show_train_panel(self):
        self.clear_content()
        self.current_panel = "train"

        frame = tk.Frame(self.content, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(frame, text="Model Training", font=("Arial", 18, "bold")).pack(pady=10)
        tk.Label(frame, text="Train the recognition model using the collected photos.").pack(pady=5)

        self.train_btn = tk.Button(frame, text="Train Model", command=self.train_model,
                                   bg="#f39c12", fg="white", padx=30, pady=10, font=("Arial", 12))
        self.train_btn.pack(pady=20)

        self.train_status = tk.Label(frame, text="", font=("Arial", 11))
        self.train_status.pack(pady=5)

    def train_model(self):
        self.train_btn.config(state=tk.DISABLED)
        self.train_status.config(text="Training in progress...")
        # Run in a separate thread to avoid freezing GUI
        threading.Thread(target=self._train_model_thread, daemon=True).start()

    def _train_model_thread(self):
        try:
            import train_model
            # Redirect stdout? Not necessary, but we can capture in future.
            train_model.main()
            self.root.after(0, self._train_complete)
        except Exception as e:
            self.root.after(0, self._train_error, str(e))

    def _train_complete(self):
        self.train_btn.config(state=tk.NORMAL)
        self.train_status.config(text="Model trained successfully!")

    def _train_error(self, msg):
        self.train_btn.config(state=tk.NORMAL)
        self.train_status.config(text=f"Training failed: {msg}")
        messagebox.showerror("Error", f"Training failed:\n{msg}")

    # ---------- Recognition Panel ----------
    def show_recognize_panel(self):
        self.clear_content()
        self.current_panel = "recognize"

        frame = tk.Frame(self.content, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(frame, text="Live Recognition", font=("Arial", 18, "bold")).pack(pady=10)

        # Video area
        self.video_label = tk.Label(frame, bg="black")
        self.video_label.pack(pady=10)

        # Buttons
        btn_frame = tk.Frame(frame)
        btn_frame.pack(pady=10)
        self.recog_start_btn = tk.Button(btn_frame, text="Start Recognition", command=self.start_recognition,
                                         bg="#27ae60", fg="white", padx=20, pady=5)
        self.recog_start_btn.pack(side=tk.LEFT, padx=5)
        self.recog_stop_btn = tk.Button(btn_frame, text="Stop", command=self.stop_recognition,
                                        bg="#e74c3c", fg="white", padx=20, pady=5, state=tk.DISABLED)
        self.recog_stop_btn.pack(side=tk.LEFT, padx=5)

        self.recog_status = tk.Label(frame, text="", font=("Arial", 12))
        self.recog_status.pack(pady=5)

    def start_recognition(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Error", "No trained model found. Please train first.")
            return

        self.recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)
        self.recognizer.read(MODEL_PATH)
        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera.")
            return

        self.last_logged = {}
        self.frame_count = 0
        self.last_results = []

        self.recog_start_btn.config(state=tk.DISABLED)
        self.recog_stop_btn.config(state=tk.NORMAL)
        self.recog_status.config(text="Recognition running...")

        self.update_recognition_frame()

    def update_recognition_frame(self):
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.recog_status.config(text="Failed to read from camera.")
            self.stop_recognition()
            return

        self.frame_count += 1

        if self.frame_count % 3 == 0:  # process every 3rd frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

            self.last_results = []
            for (x, y, w, h) in faces:
                face_img = gray[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, FACE_SIZE)
                face_img = cv2.equalizeHist(face_img)

                label_id, distance = self.recognizer.predict(face_img)
                name = get_member_name(label_id) if distance < CONFIDENCE_THRESHOLD else None

                if name:
                    color = (0, 255, 0)
                    text = f"{name} ({distance:.0f})"
                    now = time.time()
                    if now - self.last_logged.get(label_id, 0) > ATTENDANCE_COOLDOWN_SECONDS:
                        log_attendance(label_id, distance)
                        self.last_logged[label_id] = now
                        print(f"ACCESS GRANTED: {name}")
                else:
                    color = (0, 0, 255)
                    text = f"Unknown ({distance:.0f})"

                self.last_results.append((x, y, w, h, text, color))

        # Draw results (even on skipped frames)
        for (x, y, w, h, text, color) in self.last_results:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Convert and display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Schedule next frame
        self.video_label.after(15, self.update_recognition_frame)

    def stop_recognition(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.video_label.config(image='')
        self.recog_start_btn.config(state=tk.NORMAL)
        self.recog_stop_btn.config(state=tk.DISABLED)
        self.recog_status.config(text="Recognition stopped.")

    # ---------- Attendance Panel ----------
    def show_attendance_panel(self):
        self.clear_content()
        self.current_panel = "attendance"

        frame = tk.Frame(self.content, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(frame, text="Attendance Log", font=("Arial", 18, "bold")).pack(pady=10)

        # Treeview table
        columns = ("id", "name", "timestamp", "confidence")
        self.att_tree = ttk.Treeview(frame, columns=columns, show="headings", height=20)
        self.att_tree.heading("id", text="ID")
        self.att_tree.heading("name", text="Member")
        self.att_tree.heading("timestamp", text="Timestamp")
        self.att_tree.heading("confidence", text="Confidence")
        self.att_tree.column("id", width=50)
        self.att_tree.column("name", width=200)
        self.att_tree.column("timestamp", width=200)
        self.att_tree.column("confidence", width=100)
        self.att_tree.pack(fill=tk.BOTH, expand=True, pady=10)

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.att_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.att_tree.configure(yscrollcommand=scrollbar.set)

        refresh_btn = tk.Button(frame, text="Refresh", command=self.load_attendance,
                                bg="#3498db", fg="white", padx=20, pady=5)
        refresh_btn.pack(pady=5)

        self.load_attendance()

    def load_attendance(self):
        if not hasattr(self, 'att_tree'):
            return
        # Clear existing
        for row in self.att_tree.get_children():
            self.att_tree.delete(row)

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""
            SELECT attendance.id, members.name, attendance.timestamp, attendance.confidence
            FROM attendance
            JOIN members ON attendance.member_id = members.id
            ORDER BY attendance.timestamp DESC
            LIMIT 500
        """)
        rows = cur.fetchall()
        conn.close()

        for row in rows:
            self.att_tree.insert("", tk.END, values=(row["id"], row["name"], row["timestamp"], f"{row['confidence']:.1f}" if row["confidence"] is not None else ""))

    # ---------- Members Panel ----------
    def show_members_panel(self):
        self.clear_content()
        self.current_panel = "members"

        frame = tk.Frame(self.content, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(frame, text="Manage Members", font=("Arial", 18, "bold")).pack(pady=10)

        # Treeview
        columns = ("id", "name", "created_at", "active")
        self.members_tree = ttk.Treeview(frame, columns=columns, show="headings", height=15)
        self.members_tree.heading("id", text="ID")
        self.members_tree.heading("name", text="Name")
        self.members_tree.heading("created_at", text="Created At")
        self.members_tree.heading("active", text="Active")
        self.members_tree.column("id", width=50)
        self.members_tree.column("name", width=200)
        self.members_tree.column("created_at", width=180)
        self.members_tree.column("active", width=80)
        self.members_tree.pack(fill=tk.BOTH, expand=True, pady=10)

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.members_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.members_tree.configure(yscrollcommand=scrollbar.set)

        btn_frame = tk.Frame(frame)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Deactivate", command=self.deactivate_selected,
                  bg="#e67e22", fg="white", padx=15, pady=5).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Reactivate", command=self.reactivate_selected,
                  bg="#27ae60", fg="white", padx=15, pady=5).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Refresh", command=self.load_members,
                  bg="#3498db", fg="white", padx=15, pady=5).pack(side=tk.LEFT, padx=5)

        self.load_members()

    def load_members(self):
        if not hasattr(self, 'members_tree'):
            return
        for row in self.members_tree.get_children():
            self.members_tree.delete(row)

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT id, name, created_at, active FROM members ORDER BY id")
        rows = cur.fetchall()
        conn.close()

        for row in rows:
            active_text = "Yes" if row["active"] else "No"
            self.members_tree.insert("", tk.END, values=(row["id"], row["name"], row["created_at"], active_text))

    def get_selected_member_id(self):
        selection = self.members_tree.selection()
        if not selection:
            messagebox.showerror("Error", "Please select a member.")
            return None
        item = self.members_tree.item(selection[0])
        return item["values"][0]  # ID is first column

    def deactivate_selected(self):
        member_id = self.get_selected_member_id()
        if member_id:
            deactivate_member(member_id)
            messagebox.showinfo("Info", "Member deactivated.\nRemember to delete their folder from dataset/ and retrain the model.")
            self.load_members()

    def reactivate_selected(self):
        member_id = self.get_selected_member_id()
        if member_id:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("UPDATE members SET active = 1 WHERE id = ?", (member_id,))
            conn.commit()
            conn.close()
            messagebox.showinfo("Info", "Member reactivated.")
            self.load_members()

    def on_closing(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = TechVisionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()