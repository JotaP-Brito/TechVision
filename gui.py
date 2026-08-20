import os
import time
import threading
import shutil
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

from config import (
    DATASET_DIR, MODEL_PATH, CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, PHOTOS_PER_MEMBER,
    CONFIDENCE_THRESHOLD, ATTENDANCE_COOLDOWN_SECONDS
)
from database import (
    add_member,
    deactivate_member,
    get_member_name,
    get_members,
    get_recent_attendance,
    get_training_status,
    init_db,
    log_attendance,
    mark_dataset_changed,
    reactivate_member,
)
from vision import (
    VisionSetupError,
    create_face_recognizer,
    ensure_valid_member_name,
    load_face_cascade,
    model_file_exists,
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

# ----------------------------------------------------------------------------
# THEME
# ----------------------------------------------------------------------------
C = {
    "bg":           "#0f1117",
    "sidebar":      "#141821",
    "sidebar_hi":   "#1c2130",
    "card":         "#1a1e29",
    "card_border":  "#262c3d",
    "accent":       "#7c5cff",
    "accent_hi":    "#9179ff",
    "accent_soft":  "#241f3d",
    "teal":         "#00d9c0",
    "text":         "#f2f3f7",
    "text_dim":     "#8b92a8",
    "text_faint":   "#565d72",
    "success":      "#22c55e",
    "success_soft": "#123321",
    "danger":       "#ef4444",
    "danger_soft":  "#3a1a1a",
    "warning":      "#f5a524",
}
F_TITLE  = ("Segoe UI Semibold", 20)
F_H2     = ("Segoe UI Semibold", 13)
F_BODY   = ("Segoe UI", 10)
F_SMALL  = ("Segoe UI", 9)
F_NAV    = ("Segoe UI", 11)
F_MONO   = ("Consolas", 9)


def round_rect(canvas, x1, y1, x2, y2, r, **kwargs):
    """Draw a rounded rectangle using arcs and rectangles."""
    points = [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]
    canvas.create_polygon(points, smooth=False, **kwargs)
    # Draw corner arcs
    canvas.create_arc(x1, y1, x1 + 2*r, y1 + 2*r, start=90, extent=90,
                      style=tk.ARC, outline=kwargs.get('fill', ''), width=0)
    canvas.create_arc(x2 - 2*r, y1, x2, y1 + 2*r, start=0, extent=90,
                      style=tk.ARC, outline=kwargs.get('fill', ''), width=0)
    canvas.create_arc(x1, y2 - 2*r, x1 + 2*r, y2, start=180, extent=90,
                      style=tk.ARC, outline=kwargs.get('fill', ''), width=0)
    canvas.create_arc(x2 - 2*r, y2 - 2*r, x2, y2, start=270, extent=90,
                      style=tk.ARC, outline=kwargs.get('fill', ''), width=0)


class Button(tk.Canvas):
    """Flat, rounded, hover-aware button drawn on a canvas."""

    def __init__(self, parent, text, command=None, bg=None, fg=None,
                 hover_bg=None, width=160, height=40, font=F_BODY, outline=False):
        super().__init__(parent, width=width, height=height,
                         bg=parent["bg"], highlightthickness=0, bd=0, cursor="hand2")
        self.command = command
        self.bg = bg or C["accent"]
        self.hover_bg = hover_bg or C["accent_hi"]
        self.fg = fg or "#ffffff"
        self.text = text
        self.font = font
        self.w = width
        self.h = height
        self.outline = outline
        self.disabled = False
        self._draw(self.bg)
        self.bind("<Enter>", lambda e: self._draw(self.hover_bg) if not self.disabled else None)
        self.bind("<Leave>", lambda e: self._draw(self.bg) if not self.disabled else None)
        self.bind("<Button-1>", self._on_click)

    def _draw(self, fill):
        self.delete("all")
        if self.outline:
            round_rect(self, 1, 1, self.w - 1, self.h - 1, 10,
                       fill=self["bg"], outline=fill, width=1.5)
        else:
            round_rect(self, 1, 1, self.w - 1, self.h - 1, 10, fill=fill, outline="")
        text_fill = self.fg if not self.disabled else C["text_faint"]
        self.create_text(self.w // 2, self.h // 2, text=self.text,
                          fill=text_fill, font=self.font)

    def _on_click(self, e):
        if not self.disabled and self.command:
            self.command()

    def set_state(self, enabled: bool):
        self.disabled = not enabled
        self.configure(cursor="hand2" if enabled else "arrow")
        self._draw(self.bg if enabled else C["card_border"])


class Badge(tk.Canvas):
    """Small status pill with a colored dot + text."""

    def __init__(self, parent, text="Ready", color=None, width=160, height=28):
        super().__init__(parent, width=width, height=height,
                          bg=parent["bg"], highlightthickness=0, bd=0)
        self.w, self.h = width, height
        self.set_status(text, color or C["text_dim"])

    def set_status(self, text, color):
        self.delete("all")
        round_rect(self, 0, 0, self.w, self.h, self.h // 2, fill=C["card"], outline=C["card_border"])
        self.create_oval(10, self.h // 2 - 4, 18, self.h // 2 + 4, fill=color, outline="")
        self.create_text(26, self.h // 2, text=text, fill=C["text"], font=F_SMALL, anchor="w")


class Card(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=C["card"], highlightbackground=C["card_border"],
                          highlightthickness=1, bd=0, **kwargs)


# ----------------------------------------------------------------------------
# APP
# ----------------------------------------------------------------------------
class TechVisionApp:
    NAV_ITEMS = [
        ("enroll",     "＋", "Enroll"),
        ("train",      "◆", "Train"),
        ("recognize",  "▣", "Recognize"),
        ("attendance", "≡", "Attendance"),
        ("members",    "◐", "Members"),
    ]

    def __init__(self, root):
        self.root = root
        self.root.title("TechVision")
        self.root.geometry("1080x720")
        self.root.configure(bg=C["bg"])
        self.root.minsize(920, 620)

        init_db()
        self._configure_ttk()
        self._build_layout()
        self.show_panel("enroll")

    # ---------------- ttk theming (Treeview / Scrollbar) ----------------
    def _configure_ttk(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Treeview",
                         background=C["card"], fieldbackground=C["card"],
                         foreground=C["text"], borderwidth=0, rowheight=30,
                         font=F_BODY)
        style.map("Treeview", background=[("selected", C["accent_soft"])],
                  foreground=[("selected", C["text"])])
        style.configure("Treeview.Heading",
                         background=C["sidebar"], foreground=C["text_dim"],
                         borderwidth=0, font=F_SMALL, relief="flat")
        style.map("Treeview.Heading", background=[("active", C["sidebar_hi"])])

        style.configure("Vertical.TScrollbar", background=C["card_border"],
                         troughcolor=C["bg"], borderwidth=0, arrowsize=12)
        style.map("Vertical.TScrollbar", background=[("active", C["text_dim"])])

        style.configure("Modern.TEntry", fieldbackground=C["card"],
                         foreground=C["text"], borderwidth=1,
                         insertcolor=C["text"])

    # ---------------- layout skeleton ----------------
    def _build_layout(self):
        # Sidebar
        self.sidebar = tk.Frame(self.root, bg=C["sidebar"], width=210)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        logo = tk.Frame(self.sidebar, bg=C["sidebar"])
        logo.pack(fill=tk.X, pady=(28, 8), padx=22)
        tk.Label(logo, text="TechVision", font=("Segoe UI Semibold", 16),
                 fg=C["text"], bg=C["sidebar"]).pack(anchor="w")
        tk.Label(logo, text="Gym entry system", font=F_SMALL,
                 fg=C["text_faint"], bg=C["sidebar"]).pack(anchor="w", pady=(2, 0))

        sep = tk.Frame(self.sidebar, bg=C["card_border"], height=1)
        sep.pack(fill=tk.X, padx=22, pady=(16, 16))

        self.nav_buttons = {}
        for key, icon, label in self.NAV_ITEMS:
            self._add_nav_item(key, icon, label)

        # Bottom status
        bottom = tk.Frame(self.sidebar, bg=C["sidebar"])
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=22, pady=22)
        tk.Frame(self.sidebar, bg=C["card_border"], height=1).pack(
            side=tk.BOTTOM, fill=tk.X, padx=22, pady=(0, 16))
        self.sidebar_status = Badge(bottom, "System ready", C["success"], width=166)
        self.sidebar_status.pack(anchor="w")

        # Main area
        main = tk.Frame(self.root, bg=C["bg"])
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.header = tk.Frame(main, bg=C["bg"])
        self.header.pack(fill=tk.X, padx=36, pady=(30, 10))
        self.header_title = tk.Label(self.header, text="", font=F_TITLE,
                                      fg=C["text"], bg=C["bg"])
        self.header_title.pack(anchor="w")
        self.header_subtitle = tk.Label(self.header, text="", font=F_BODY,
                                         fg=C["text_dim"], bg=C["bg"])
        self.header_subtitle.pack(anchor="w", pady=(2, 0))

        self.content = tk.Frame(main, bg=C["bg"])
        self.content.pack(fill=tk.BOTH, expand=True, padx=36, pady=(10, 30))

    def _add_nav_item(self, key, icon, label):
        row = tk.Frame(self.sidebar, bg=C["sidebar"], cursor="hand2")
        row.pack(fill=tk.X, padx=12, pady=3)

        indicator = tk.Frame(row, bg=C["sidebar"], width=3)
        indicator.pack(side=tk.LEFT, fill=tk.Y)

        inner = tk.Frame(row, bg=C["sidebar"])
        inner.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        icon_lbl = tk.Label(inner, text=icon, font=F_NAV, fg=C["text_dim"], bg=C["sidebar"])
        icon_lbl.pack(side=tk.LEFT, padx=(4, 12))
        text_lbl = tk.Label(inner, text=label, font=F_NAV, fg=C["text_dim"], bg=C["sidebar"])
        text_lbl.pack(side=tk.LEFT)

        # Bind click only to the row frame (prevents double firing)
        row.bind("<Button-1>", lambda e, k=key: self.show_panel(k))

        # Hover effect
        def on_enter(e):
            if not self._is_nav_active(key):
                row.configure(bg=C["sidebar_hi"])
                inner.configure(bg=C["sidebar_hi"])
                icon_lbl.configure(bg=C["sidebar_hi"])
                text_lbl.configure(bg=C["sidebar_hi"])

        def on_leave(e):
            if not self._is_nav_active(key):
                row.configure(bg=C["sidebar"])
                inner.configure(bg=C["sidebar"])
                icon_lbl.configure(bg=C["sidebar"])
                text_lbl.configure(bg=C["sidebar"])

        row.bind("<Enter>", on_enter)
        row.bind("<Leave>", on_leave)

        self.nav_buttons[key] = {
            'row': row, 'indicator': indicator, 'inner': inner,
            'icon_lbl': icon_lbl, 'text_lbl': text_lbl
        }

    def _is_nav_active(self, key):
        active_key = getattr(self, '_active_nav', None)
        return key == active_key

    def _highlight_nav(self, active_key):
        self._active_nav = active_key
        for key, widgets in self.nav_buttons.items():
            active = key == active_key
            bg = C["sidebar_hi"] if active else C["sidebar"]
            fg = C["text"] if active else C["text_dim"]
            widgets['row'].configure(bg=bg)
            widgets['inner'].configure(bg=bg)
            widgets['icon_lbl'].configure(bg=bg, fg=C["accent"] if active else fg)
            widgets['text_lbl'].configure(bg=bg, fg=fg)
            widgets['indicator'].configure(bg=C["accent"] if active else C["sidebar"])

    def clear_content(self):
        self.stop_camera()
        for widget in self.content.winfo_children():
            widget.destroy()

    def show_panel(self, key):
        self._highlight_nav(key)
        titles = {
            "enroll": ("Enroll a member", "Capture reference photos for a new member."),
            "train": ("Train the model", "Rebuild recognition from everyone currently enrolled."),
            "recognize": ("Live recognition", "Match faces at the entrance in real time."),
            "attendance": ("Attendance log", "Recent recognized entries."),
            "members": ("Manage members", "Activate, deactivate, and review enrolled members."),
        }
        title, subtitle = titles[key]
        self.header_title.config(text=title)
        self.header_subtitle.config(text=subtitle)
        getattr(self, f"show_{key}_panel")()

    # ---------------- Helper: display frame in label ----------------
    def _display_frame(self, frame):
        """Resize frame to fit the label while maintaining aspect ratio and display."""
        if not hasattr(self, 'video_label'):
            return
        label_w = self.video_label.winfo_width()
        label_h = self.video_label.winfo_height()
        if label_w <= 1 or label_h <= 1:
            label_w, label_h = FRAME_WIDTH, FRAME_HEIGHT

        frame_h, frame_w = frame.shape[:2]
        scale = min(label_w / frame_w, label_h / frame_h)
        new_w = int(frame_w * scale)
        new_h = int(frame_h * scale)
        if new_w > 0 and new_h > 0:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def _format_training_status(self):
        status = get_training_status()
        if status["ready"]:
            trained_at = status["model_trained_at"] or "unknown time"
            return f"Model ready. Last trained: {trained_at}", C["success"]

        return f"Model needs training. {status['reason']}", C["warning"]

    def _cleanup_incomplete_enrollment(self):
        if hasattr(self, 'member_id') and getattr(self, 'captured', 0) < PHOTOS_PER_MEMBER:
            deactivate_member(self.member_id)
            if hasattr(self, 'member_dir'):
                shutil.rmtree(self.member_dir, ignore_errors=True)
        self.member_id = None
        self.member_dir = None

    # ---------------- ENROLL ----------------
    def show_enroll_panel(self):
        self.clear_content()

        card = Card(self.content)
        card.pack(fill=tk.BOTH, expand=True)
        pad = tk.Frame(card, bg=C["card"])
        pad.pack(fill=tk.BOTH, expand=True, padx=28, pady=24)

        # Use grid to pin controls at bottom, video flexible
        pad.grid_rowconfigure(1, weight=1)   # video row expands
        pad.grid_columnconfigure(0, weight=1)

        # Member name row
        row = tk.Frame(pad, bg=C["card"])
        row.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        tk.Label(row, text="Member name", font=F_H2, fg=C["text"], bg=C["card"]).pack(anchor="w")
        self.name_entry = tk.Entry(row, font=F_BODY, bg=C["bg"], fg=C["text"],
                                    insertbackground=C["text"], relief="flat",
                                    highlightthickness=1, highlightbackground=C["card_border"],
                                    highlightcolor=C["accent"])
        self.name_entry.pack(anchor="w", fill=tk.X, pady=(8, 0), ipady=7)

        # Video container (flexible)
        video_wrap = tk.Frame(pad, bg=C["bg"], highlightbackground=C["card_border"],
                               highlightthickness=1)
        video_wrap.grid(row=1, column=0, sticky="nsew", pady=18)
        self.video_label = tk.Label(video_wrap, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Buttons and status at the bottom
        btn_row = tk.Frame(pad, bg=C["card"])
        btn_row.grid(row=2, column=0, sticky="w", pady=(4, 12))
        self.enroll_start_btn = Button(btn_row, "Start Enrollment", self.start_enrollment,
                                        bg=C["accent"], hover_bg=C["accent_hi"], width=190)
        self.enroll_start_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.enroll_cancel_btn = Button(btn_row, "Cancel", self.cancel_enrollment,
                                         bg=C["card"], fg=C["danger"], hover_bg=C["danger_soft"],
                                         width=120, outline=True)
        self.enroll_cancel_btn.set_state(False)
        self.enroll_cancel_btn.pack(side=tk.LEFT)

        self.enroll_status_label = tk.Label(pad, text="", font=F_BODY,
                                             fg=C["text_dim"], bg=C["card"])
        self.enroll_status_label.grid(row=3, column=0, sticky="w")
        status_text, status_color = self._format_training_status()
        self.enroll_status_label.config(text=status_text, fg=status_color)

    def start_enrollment(self):
        try:
            name = ensure_valid_member_name(self.name_entry.get())
        except ValueError as exc:
            messagebox.showerror("Error", str(exc))
            return

        try:
            self.face_cascade = load_face_cascade()
            self.cap = open_camera(CAMERA_INDEX)
        except VisionSetupError as exc:
            messagebox.showerror("Error", str(exc))
            self.cap = None
            return

        self.member_id = add_member(name)
        safe_name = sanitize_member_name(name)
        self.member_dir = os.path.join(DATASET_DIR, f"{self.member_id}_{safe_name}")
        os.makedirs(self.member_dir, exist_ok=True)

        self.captured = 0
        self.last_capture_time = 0
        self.capture_delay = 1.0

        self.enroll_start_btn.set_state(False)
        self.enroll_cancel_btn.set_state(True)
        self.enroll_status_label.config(text=f"Enrolling {name} — follow the prompts...")
        self.sidebar_status.set_status("Enrolling…", C["warning"])

        self.update_enrollment_frame()

    def update_enrollment_frame(self):
        if not hasattr(self, 'cap') or not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self._cleanup_incomplete_enrollment()
            self.enroll_status_label.config(text="Camera feed was interrupted. Enrollment was cancelled.", fg=C["danger"])
            self.stop_camera()
            self.enroll_start_btn.set_state(True)
            self.enroll_cancel_btn.set_state(False)
            self.sidebar_status.set_status("System ready", C["success"])
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (124, 92, 255), 2)

        if self.captured < PHOTOS_PER_MEMBER:
            prompt = POSE_PROMPTS[self.captured % len(POSE_PROMPTS)]
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (10, 10, 15), -1)
            cv2.putText(frame, f"{self.captured + 1}/{PHOTOS_PER_MEMBER}  {prompt}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 217, 192), 2)

            if len(faces) == 1 and (time.time() - self.last_capture_time) > self.capture_delay:
                (x, y, w, h) = faces[0]
                face_img = prepare_face_image(gray, x, y, w, h)
                filename = os.path.join(self.member_dir, f"img_{self.captured}.jpg")
                cv2.imwrite(filename, face_img)
                self.captured += 1
                self.last_capture_time = time.time()

                if self.captured >= PHOTOS_PER_MEMBER:
                    mark_dataset_changed()
                    self.enroll_status_label.config(text="Enrollment complete. Train the model before recognition.", fg=C["warning"])
                    self.sidebar_status.set_status("System ready", C["success"])
                    self.stop_camera()
                    self.enroll_start_btn.set_state(True)
                    self.enroll_cancel_btn.set_state(False)
                    self.member_id = None
                    self.member_dir = None
                    return
        else:
            self.stop_camera()
            self.enroll_start_btn.set_state(True)
            self.enroll_cancel_btn.set_state(False)
            return

        self._display_frame(frame)
        self.video_label.after(15, self.update_enrollment_frame)

    def cancel_enrollment(self):
        self._cleanup_incomplete_enrollment()
        self.stop_camera()
        status_text, status_color = self._format_training_status()
        self.enroll_status_label.config(text=f"Enrollment cancelled. {status_text}", fg=status_color)
        self.sidebar_status.set_status("System ready", C["success"])
        self.enroll_start_btn.set_state(True)
        self.enroll_cancel_btn.set_state(False)

    def stop_camera(self):
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        if hasattr(self, 'video_label'):
            try:
                self.video_label.config(image='')
            except tk.TclError:
                pass

    # ---------------- TRAIN ----------------
    def show_train_panel(self):
        self.clear_content()
        card = Card(self.content)
        card.pack(fill=tk.X)
        pad = tk.Frame(card, bg=C["card"])
        pad.pack(fill=tk.X, padx=28, pady=28)

        tk.Label(pad, text="Rebuild the recognition model from every photo currently in dataset/.",
                 font=F_BODY, fg=C["text_dim"], bg=C["card"]).pack(anchor="w", pady=(0, 18))

        self.train_btn = Button(pad, "Train Model", self.train_model,
                                 bg=C["teal"], fg="#04211c", hover_bg="#22e8cf", width=170)
        self.train_btn.pack(anchor="w")

        status_text, status_color = self._format_training_status()
        self.train_status = tk.Label(pad, text=status_text, font=F_BODY, fg=status_color, bg=C["card"])
        self.train_status.pack(anchor="w", pady=(16, 0))

    def train_model(self):
        self.train_btn.set_state(False)
        self.train_status.config(text="Training in progress…", fg=C["warning"])
        self.sidebar_status.set_status("Training…", C["warning"])
        threading.Thread(target=self._train_model_thread, daemon=True).start()

    def _train_model_thread(self):
        try:
            import train_model
            trained = train_model.main()
            if trained:
                self.root.after(0, self._train_complete)
            else:
                self.root.after(0, self._train_incomplete)
        except Exception as e:
            self.root.after(0, self._train_error, str(e))

    def _train_complete(self):
        self.train_btn.set_state(True)
        status_text, status_color = self._format_training_status()
        self.train_status.config(text=status_text, fg=status_color)
        self.sidebar_status.set_status("System ready", C["success"])

    def _train_incomplete(self):
        self.train_btn.set_state(True)
        self.train_status.config(
            text="Training could not run because no active member photos were available.",
            fg=C["warning"],
        )
        self.sidebar_status.set_status("Training needed", C["warning"])

    def _train_error(self, msg):
        self.train_btn.set_state(True)
        self.train_status.config(text=f"Training failed: {msg}", fg=C["danger"])
        self.sidebar_status.set_status("Training failed", C["danger"])
        messagebox.showerror("Error", f"Training failed:\n{msg}")

    # ---------------- RECOGNIZE ----------------
    def show_recognize_panel(self):
        self.clear_content()
        card = Card(self.content)
        card.pack(fill=tk.BOTH, expand=True)
        pad = tk.Frame(card, bg=C["card"])
        pad.pack(fill=tk.BOTH, expand=True, padx=28, pady=24)

        # Use grid to pin controls at bottom, video flexible
        pad.grid_rowconfigure(1, weight=1)
        pad.grid_columnconfigure(0, weight=1)

        # Video container
        video_wrap = tk.Frame(pad, bg=C["bg"], highlightbackground=C["card_border"],
                               highlightthickness=1)
        video_wrap.grid(row=0, column=0, sticky="nsew", pady=(0, 18))
        self.video_label = tk.Label(video_wrap, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Buttons and status at bottom
        btn_row = tk.Frame(pad, bg=C["card"])
        btn_row.grid(row=1, column=0, sticky="w")
        self.recog_start_btn = Button(btn_row, "Start Recognition", self.start_recognition,
                                       bg=C["accent"], hover_bg=C["accent_hi"], width=190)
        self.recog_start_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.recog_stop_btn = Button(btn_row, "Stop", self.stop_recognition,
                                      bg=C["card"], fg=C["danger"], hover_bg=C["danger_soft"],
                                      width=120, outline=True)
        self.recog_stop_btn.set_state(False)
        self.recog_stop_btn.pack(side=tk.LEFT)

        self.recog_status = tk.Label(pad, text="", font=F_BODY, fg=C["text_dim"], bg=C["card"])
        self.recog_status.grid(row=2, column=0, sticky="w", pady=(12, 0))

    def start_recognition(self):
        if not model_file_exists(MODEL_PATH):
            messagebox.showerror("Error", "No trained model found. Please train first.")
            return

        training_status = get_training_status()
        if not training_status["ready"]:
            messagebox.showerror("Error", f"Recognition is blocked until retraining.\n\n{training_status['reason']}")
            return

        try:
            self.recognizer = create_face_recognizer()
            self.face_cascade = load_face_cascade()
            self.cap = open_camera(CAMERA_INDEX)
        except VisionSetupError as exc:
            messagebox.showerror("Error", str(exc))
            self.cap = None
            return

        self.recognizer.read(MODEL_PATH)

        self.last_logged = {}
        self.frame_count = 0
        self.last_results = []

        self.recog_start_btn.set_state(False)
        self.recog_stop_btn.set_state(True)
        self.recog_status.config(text="Recognition running…", fg=C["text_dim"])
        self.sidebar_status.set_status("Recognizing…", C["accent"])

        self.update_recognition_frame()

    def update_recognition_frame(self):
        if not hasattr(self, 'cap') or not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.recog_status.config(text="Failed to read from camera.")
            self.stop_recognition()
            return

        self.frame_count += 1

        if self.frame_count % 3 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

            self.last_results = []
            for (x, y, w, h) in faces:
                face_img = prepare_face_image(gray, x, y, w, h)

                label_id, distance = self.recognizer.predict(face_img)
                name = get_member_name(label_id) if distance < CONFIDENCE_THRESHOLD else None

                if name:
                    color = (0, 217, 192)
                    text = f"{name} ({distance:.0f})"
                    now = time.time()
                    if now - self.last_logged.get(label_id, 0) > ATTENDANCE_COOLDOWN_SECONDS:
                        log_attendance(label_id, distance)
                        self.last_logged[label_id] = now
                        self.recog_status.config(text=f"Access granted: {name}", fg=C["success"])
                else:
                    color = (239, 68, 68)
                    text = f"Unknown ({distance:.0f})"

                self.last_results.append((x, y, w, h, text, color))

        for (x, y, w, h, text, color) in self.last_results:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        self._display_frame(frame)
        self.video_label.after(15, self.update_recognition_frame)

    def stop_recognition(self):
        self.stop_camera()
        self.recog_start_btn.set_state(True)
        self.recog_stop_btn.set_state(False)
        self.recog_status.config(text="Recognition stopped.", fg=C["text_dim"])
        self.sidebar_status.set_status("System ready", C["success"])

    # ---------------- ATTENDANCE ----------------
    def show_attendance_panel(self):
        self.clear_content()
        card = Card(self.content)
        card.pack(fill=tk.BOTH, expand=True)
        pad = tk.Frame(card, bg=C["card"])
        pad.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tree_wrap = tk.Frame(pad, bg=C["card"])
        tree_wrap.pack(fill=tk.BOTH, expand=True)

        columns = ("id", "name", "timestamp", "confidence")
        self.att_tree = ttk.Treeview(tree_wrap, columns=columns, show="headings", height=18)
        for col, label, w in [("id", "ID", 60), ("name", "Member", 220),
                               ("timestamp", "Timestamp", 220), ("confidence", "Confidence", 110)]:
            self.att_tree.heading(col, text=label)
            self.att_tree.column(col, width=w)
        self.att_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL, command=self.att_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.att_tree.configure(yscrollcommand=scrollbar.set)

        Button(pad, "Refresh", self.load_attendance, bg=C["card"], fg=C["text"],
               hover_bg=C["sidebar_hi"], width=120, outline=True).pack(anchor="w", pady=(14, 0))

        self.load_attendance()

    def load_attendance(self):
        if not hasattr(self, 'att_tree'):
            return
        for row in self.att_tree.get_children():
            self.att_tree.delete(row)

        for row in get_recent_attendance():
            conf = f"{row['confidence']:.1f}" if row["confidence"] is not None else ""
            self.att_tree.insert("", tk.END, values=(row["id"], row["name"], row["timestamp"], conf))

    # ---------------- MEMBERS ----------------
    def show_members_panel(self):
        self.clear_content()
        card = Card(self.content)
        card.pack(fill=tk.BOTH, expand=True)
        pad = tk.Frame(card, bg=C["card"])
        pad.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tree_wrap = tk.Frame(pad, bg=C["card"])
        tree_wrap.pack(fill=tk.BOTH, expand=True)

        columns = ("id", "name", "created_at", "active")
        self.members_tree = ttk.Treeview(tree_wrap, columns=columns, show="headings", height=15)
        for col, label, w in [("id", "ID", 60), ("name", "Name", 220),
                               ("created_at", "Created At", 200), ("active", "Active", 90)]:
            self.members_tree.heading(col, text=label)
            self.members_tree.column(col, width=w)
        self.members_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL, command=self.members_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.members_tree.configure(yscrollcommand=scrollbar.set)

        btn_row = tk.Frame(pad, bg=C["card"])
        btn_row.pack(anchor="w", pady=(14, 0))
        Button(btn_row, "Deactivate", self.deactivate_selected, bg=C["card"], fg=C["warning"],
               hover_bg="#2e2410", width=130, outline=True).pack(side=tk.LEFT, padx=(0, 8))
        Button(btn_row, "Reactivate", self.reactivate_selected, bg=C["card"], fg=C["success"],
               hover_bg=C["success_soft"], width=130, outline=True).pack(side=tk.LEFT, padx=(0, 8))
        Button(btn_row, "Refresh", self.load_members, bg=C["card"], fg=C["text"],
               hover_bg=C["sidebar_hi"], width=110, outline=True).pack(side=tk.LEFT)

        self.load_members()

    def load_members(self):
        if not hasattr(self, 'members_tree'):
            return
        for row in self.members_tree.get_children():
            self.members_tree.delete(row)

        for row in get_members():
            active_text = "Yes" if row["active"] else "No"
            self.members_tree.insert("", tk.END, values=(row["id"], row["name"], row["created_at"], active_text))

    def get_selected_member_id(self):
        selection = self.members_tree.selection()
        if not selection:
            messagebox.showerror("Error", "Please select a member.")
            return None
        item = self.members_tree.item(selection[0])
        return item["values"][0]

    def deactivate_selected(self):
        member_id = self.get_selected_member_id()
        if member_id:
            deactivate_member(member_id)
            mark_dataset_changed()
            messagebox.showinfo("Info", "Member deactivated.\nTrain the model again before using recognition.")
            self.load_members()

    def reactivate_selected(self):
        member_id = self.get_selected_member_id()
        if member_id:
            reactivate_member(member_id)
            mark_dataset_changed()
            messagebox.showinfo("Info", "Member reactivated.\nTrain the model again before using recognition.")
            self.load_members()

    def on_closing(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = TechVisionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
