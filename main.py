"""Smart Posture Monitor - Desktop Version

Uses MediaPipe Face Detection (preferred) with a Haar Cascade fallback,
so detection works reliably across lighting / skin-tone / angle.
"""

import time
import tkinter as tk
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


# ---------- Detection ----------

class PoseDetector:
    """Detect face position; expose a single centerY value for posture logic."""

    def __init__(self):
        self.calibrated_y = None
        self.smoothed_y = None
        self.smoothing = 0.3
        self.slouch_threshold = 35

        if HAS_MEDIAPIPE:
            self.mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
            self.backend = "mediapipe"
        else:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self.profile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_profileface.xml"
            )
            self.backend = "haar"

    def _detect_mediapipe(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.mp_face.process(rgb)
        if not results.detections:
            return None
        h, w = frame.shape[:2]
        # pick the largest detection
        best = max(
            results.detections,
            key=lambda d: d.location_data.relative_bounding_box.width
            * d.location_data.relative_bounding_box.height,
        )
        box = best.location_data.relative_bounding_box
        x, y = int(box.xmin * w), int(box.ymin * h)
        bw, bh = int(box.width * w), int(box.height * h)
        return x, y, bw, bh

    def _detect_haar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        if len(faces) == 0:
            faces = self.profile_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        if len(faces) == 0:
            return None
        # pick largest
        return max(faces, key=lambda f: f[2] * f[3])

    def detect(self, frame):
        if self.backend == "mediapipe":
            return self._detect_mediapipe(frame)
        return self._detect_haar(frame)

    def analyze(self, frame):
        """Returns (status, annotated_frame, info_dict)."""
        info = {"center_y": None, "drop": None}
        face = self.detect(frame)

        # baseline line
        if self.calibrated_y is not None:
            cv2.line(
                frame,
                (0, int(self.calibrated_y)),
                (frame.shape[1], int(self.calibrated_y)),
                (235, 102, 99),  # blue baseline
                2,
                cv2.LINE_AA,
            )

        if face is None:
            self.smoothed_y = None
            return "no_person", frame, info

        x, y, w, h = face
        center_y = y + h / 2

        # smooth to remove jitter
        if self.smoothed_y is None:
            self.smoothed_y = center_y
        else:
            self.smoothed_y = (
                self.smoothed_y * (1 - self.smoothing) + center_y * self.smoothing
            )
        center_y = self.smoothed_y
        info["center_y"] = center_y

        # decide colour based on posture
        if self.calibrated_y is not None:
            drop = center_y - self.calibrated_y
            info["drop"] = drop
            slouching = drop > self.slouch_threshold
            color = (68, 68, 239) if slouching else (129, 199, 16)  # red / green
        else:
            slouching = False
            color = (245, 158, 11)  # amber while uncalibrated

        # draw bounding box + corner accents
        self._draw_box(frame, x, y, w, h, color)
        # current center line
        cv2.line(frame, (x, int(center_y)), (x + w, int(center_y)), color, 2, cv2.LINE_AA)

        if self.calibrated_y is None:
            return "need_calibration", frame, info
        return ("slouching" if slouching else "normal"), frame, info

    @staticmethod
    def _draw_box(frame, x, y, w, h, color):
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
        # corner accents
        cl = 16
        thickness = 4
        # top-left
        cv2.line(frame, (x, y), (x + cl, y), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y + cl), color, thickness, cv2.LINE_AA)
        # top-right
        cv2.line(frame, (x + w, y), (x + w - cl, y), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x + w, y), (x + w, y + cl), color, thickness, cv2.LINE_AA)
        # bottom-left
        cv2.line(frame, (x, y + h), (x + cl, y + h), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y + h), (x, y + h - cl), color, thickness, cv2.LINE_AA)
        # bottom-right
        cv2.line(frame, (x + w, y + h), (x + w - cl, y + h), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - cl), color, thickness, cv2.LINE_AA)


# ---------- Data ----------

class DataManager:
    def __init__(self):
        self.normal_seconds = 0
        self.slouch_seconds = 0
        self.last_record_time = 0.0

    def record(self, status):
        now = time.time()
        if status in ("normal", "slouching") and (now - self.last_record_time >= 1.0):
            if status == "normal":
                self.normal_seconds += 1
            else:
                self.slouch_seconds += 1
            self.last_record_time = now

    def reset(self):
        self.normal_seconds = 0
        self.slouch_seconds = 0
        self.last_record_time = 0.0

    @property
    def total(self):
        return self.normal_seconds + self.slouch_seconds

    @property
    def accuracy(self):
        if self.total == 0:
            return 0.0
        return self.normal_seconds / self.total * 100


# ---------- UI ----------

BG = "#0f172a"
PANEL = "#1e293b"
CARD = "#334155"
TEXT = "#f1f5f9"
MUTED = "#94a3b8"
ACCENT = "#6366f1"
SUCCESS = "#10b981"
DANGER = "#ef4444"
WARN = "#f59e0b"


class PostureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("智慧坐姿監測 · Smart Posture Monitor")
        self.root.configure(bg=BG)
        self.root.minsize(1040, 560)

        self.detector = PoseDetector()
        self.data = DataManager()

        self.vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.vid.isOpened():
            messagebox.showerror("錯誤", "無法開啟攝影機，請檢查裝置與權限。")
            root.destroy()
            return

        self.slouch_start = None
        self.last_fps_time = time.time()
        self.frame_count = 0
        self.fps = 0

        self._build_ui()
        self.update()

    # --- UI ---

    def _build_ui(self):
        # left: video
        left = tk.Frame(self.root, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=16, pady=16)

        self.canvas = tk.Canvas(left, width=640, height=480, bg="#000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # right: control panel
        right = tk.Frame(self.root, bg=PANEL, width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 16), pady=16)
        right.pack_propagate(False)

        inner = tk.Frame(right, bg=PANEL)
        inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # header
        tk.Label(
            inner, text="智慧坐姿監測", bg=PANEL, fg=TEXT,
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")
        backend_label = "MediaPipe AI" if self.detector.backend == "mediapipe" else "Haar Cascade"
        tk.Label(
            inner, text=f"偵測引擎: {backend_label}", bg=PANEL, fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(0, 14))

        # status card
        self.status_frame = tk.Frame(inner, bg=CARD, height=80)
        self.status_frame.pack(fill=tk.X, pady=(0, 14))
        self.status_frame.pack_propagate(False)
        self.status_label = tk.Label(
            self.status_frame, text="準備中...", bg=CARD, fg=TEXT,
            font=("Segoe UI", 16, "bold"),
        )
        self.status_label.pack(expand=True)
        self.status_sub = tk.Label(
            self.status_frame, text="", bg=CARD, fg=MUTED, font=("Segoe UI", 9)
        )
        self.status_sub.pack()

        # calibration row
        calib_row = tk.Frame(inner, bg=CARD)
        calib_row.pack(fill=tk.X, pady=(0, 14))
        tk.Label(
            calib_row, text="校準狀態", bg=CARD, fg=MUTED, font=("Segoe UI", 10)
        ).pack(side=tk.LEFT, padx=12, pady=10)
        self.calib_label = tk.Label(
            calib_row, text="未校準", bg=CARD, fg=WARN, font=("Segoe UI", 10, "bold")
        )
        self.calib_label.pack(side=tk.RIGHT, padx=12, pady=10)

        # stats grid
        stats = tk.Frame(inner, bg=PANEL)
        stats.pack(fill=tk.X, pady=(0, 14))

        self.normal_value = self._stat_card(stats, "良好姿勢", "0 秒", SUCCESS, 0)
        self.slouch_value = self._stat_card(stats, "駝背時間", "0 秒", DANGER, 1)

        # accuracy
        acc_frame = tk.Frame(inner, bg=PANEL)
        acc_frame.pack(fill=tk.X, pady=(0, 14))
        tk.Label(
            acc_frame, text="姿勢分數", bg=PANEL, fg=MUTED, font=("Segoe UI", 10)
        ).pack()
        self.accuracy_label = tk.Label(
            acc_frame, text="--", bg=PANEL, fg=TEXT, font=("Segoe UI", 28, "bold")
        )
        self.accuracy_label.pack()

        # progress bar
        self.progress_bg = tk.Frame(inner, bg=CARD, height=8)
        self.progress_bg.pack(fill=tk.X, pady=(0, 18))
        self.progress_fill = tk.Frame(self.progress_bg, bg=SUCCESS, height=8)
        self.progress_fill.place(x=0, y=0, relwidth=0, relheight=1)

        # buttons
        self._button(inner, "🎯  開始校準", self.calibrate, ACCENT)
        self._button(inner, "📊  生成報告", self.show_report, CARD)
        self._button(inner, "🔄  重置數據", self.reset, CARD)

        # footer tip
        self.tip_label = tk.Label(
            inner,
            text="💡 請坐直後點擊「開始校準」",
            bg=PANEL, fg=MUTED, font=("Segoe UI", 9),
            wraplength=260, justify="center",
        )
        self.tip_label.pack(side=tk.BOTTOM, pady=(14, 0))

    def _stat_card(self, parent, label, value, color, col):
        frame = tk.Frame(parent, bg=CARD)
        frame.grid(row=0, column=col, sticky="nsew", padx=(0, 8) if col == 0 else (8, 0))
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)
        tk.Label(
            frame, text=label, bg=CARD, fg=MUTED, font=("Segoe UI", 9)
        ).pack(pady=(10, 2))
        val_lbl = tk.Label(
            frame, text=value, bg=CARD, fg=color, font=("Segoe UI", 16, "bold")
        )
        val_lbl.pack(pady=(0, 10))
        return val_lbl

    def _button(self, parent, text, command, bg):
        btn = tk.Button(
            parent, text=text, command=command,
            bg=bg, fg=TEXT, activebackground=bg, activeforeground=TEXT,
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT, bd=0, padx=12, pady=10, cursor="hand2",
        )
        btn.pack(fill=tk.X, pady=4)
        return btn

    # --- actions ---

    def calibrate(self):
        if self.detector.smoothed_y is None:
            self._set_status("未偵測到人臉", WARN, "請正對鏡頭並確保光線充足")
            return
        self.detector.calibrated_y = self.detector.smoothed_y
        self.calib_label.config(text="已校準 ✓", fg=SUCCESS)
        self.tip_label.config(text="✨ 校準完成！系統正在監測您的姿勢")

    def reset(self):
        if not messagebox.askyesno("重置", "確定要重置所有數據與校準嗎？"):
            return
        self.data.reset()
        self.detector.calibrated_y = None
        self.slouch_start = None
        self.calib_label.config(text="未校準", fg=WARN)
        self.tip_label.config(text="💡 請坐直後點擊「開始校準」")
        self._refresh_stats()

    def show_report(self):
        if self.data.total == 0:
            messagebox.showinfo("提示", "尚無數據，請先進行姿勢偵測。")
            return

        acc = self.data.accuracy
        if acc >= 90:
            advice = "🏆 完美！您的坐姿堪稱典範，請繼續保持！"
        elif acc >= 75:
            advice = "👍 不錯！偶爾的駝背要注意，整體表現良好。"
        elif acc >= 50:
            advice = "⚠️ 需要改善！請更注意挺直背部。"
        else:
            advice = "🚨 警告！長時間駝背會傷害脊椎，請立即調整姿勢並休息。"

        win = tk.Toplevel(self.root)
        win.title("📊 姿勢報告")
        win.configure(bg=PANEL)
        win.geometry("360x320")
        win.resizable(False, False)

        tk.Label(
            win, text="📊 姿勢健康報告", bg=PANEL, fg=TEXT,
            font=("Segoe UI", 14, "bold"),
        ).pack(pady=(20, 16))

        for txt, color in (
            (f"良好姿勢：{self.data.normal_seconds} 秒", SUCCESS),
            (f"駝背時間：{self.data.slouch_seconds} 秒", DANGER),
            (f"姿勢分數：{acc:.1f}%", ACCENT),
        ):
            tk.Label(win, text=txt, bg=PANEL, fg=color, font=("Segoe UI", 12, "bold")).pack(pady=4)

        tk.Label(
            win, text=advice, bg=PANEL, fg=TEXT, font=("Segoe UI", 10),
            wraplength=300, justify="center",
        ).pack(pady=20)

        tk.Button(
            win, text="關閉", command=win.destroy,
            bg=ACCENT, fg=TEXT, activebackground=ACCENT, activeforeground=TEXT,
            font=("Segoe UI", 10, "bold"), relief=tk.FLAT, bd=0, padx=20, pady=8, cursor="hand2",
        ).pack()

    # --- frame loop ---

    def update(self):
        ret, frame = self.vid.read()
        if not ret:
            self.root.after(30, self.update)
            return

        frame = cv2.flip(frame, 1)
        status, annotated, info = self.detector.analyze(frame)

        self.data.record(status)
        self._update_status(status, info)
        self._refresh_stats()
        self._update_fps()

        # render to canvas
        img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(img))

        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw > 1 and ch > 1:
            self.canvas.create_image(cw // 2, ch // 2, image=self.photo, anchor=tk.CENTER)
        else:
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.root.after(15, self.update)

    def _update_status(self, status, info):
        if status == "no_person":
            self._set_status("未偵測到人像", WARN, "請正對鏡頭")
            self.slouch_start = None
            return

        if status == "need_calibration":
            self._set_status("請進行校準", WARN, "點擊「開始校準」按鈕")
            return

        if status == "slouching":
            if self.slouch_start is None:
                self.slouch_start = time.time()
            duration = time.time() - self.slouch_start
            if duration > 2:
                self._set_status(f"⚠️ 駝背警告 ({duration:.0f}s)", DANGER, "請立刻坐直！")
            else:
                self._set_status("姿勢偏離", WARN, "正在偵測中...")
        else:
            self.slouch_start = None
            self._set_status("✅ 姿勢良好", SUCCESS, f"偏離: {info['drop']:.0f}px" if info["drop"] is not None else "")

    def _set_status(self, text, color, sub=""):
        self.status_label.config(text=text, fg=color)
        self.status_sub.config(text=sub)

    def _refresh_stats(self):
        self.normal_value.config(text=f"{self.data.normal_seconds} 秒")
        self.slouch_value.config(text=f"{self.data.slouch_seconds} 秒")

        if self.data.total == 0:
            self.accuracy_label.config(text="--", fg=TEXT)
            self.progress_fill.place_configure(relwidth=0)
            return

        acc = self.data.accuracy
        self.accuracy_label.config(text=f"{acc:.0f}%")
        self.progress_fill.place_configure(relwidth=acc / 100)
        if acc >= 80:
            self.progress_fill.config(bg=SUCCESS)
            self.accuracy_label.config(fg=SUCCESS)
        elif acc >= 50:
            self.progress_fill.config(bg=WARN)
            self.accuracy_label.config(fg=WARN)
        else:
            self.progress_fill.config(bg=DANGER)
            self.accuracy_label.config(fg=DANGER)

    def _update_fps(self):
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = now


if __name__ == "__main__":
    root = tk.Tk()
    PostureApp(root)
    root.mainloop()
