import cv2
import tkinter as tk
from PIL import Image, ImageTk
import time

class PoseDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.calibrated_y = None

    def check_posture(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 使用最黏、最不容易因為低頭而消失的參數
        faces = self.face_cascade.detectMultiScale(gray, 1.05, 2)
        
        # 只要校正過，就畫出藍色基準線
        if self.calibrated_y is not None:
            cv2.line(frame, (0, int(self.calibrated_y)), (640, int(self.calibrated_y)), (255, 0, 0), 2)

        if len(faces) == 0:
            return "No Person", frame

        (x, y, w, h) = faces[0]
        # 【核心修正】：改用「臉部中心點」來判斷高度，再也騙不倒電腦
        center_y = y + (h / 2)
        
        # 畫出當前臉部的黃色中心線與綠色外框
        cv2.line(frame, (x, int(center_y)), (x+w, int(center_y)), (0, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        if self.calibrated_y is None:
            return "Need Calibration", frame

        # 超嚴格閾值：中心點黃線只要掉到藍線下方超過 20 像素，立刻抓！
        slouch_threshold = 20
        if (center_y - self.calibrated_y) > slouch_threshold:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3) # 駝背時變紅框
            return "Slouching", frame
        
        return "Normal", frame

class DataManager:
    def __init__(self):
        self.records = []
        self.last_record_time = 0

    def record(self, status):
        current_time = time.time()
        # 現實時間每過 1 秒才記錄一次，時間不再暴增
        if status in ["Normal", "Slouching"] and (current_time - self.last_record_time >= 1.0):
            self.records.append({"Time": current_time, "Status": status})
            self.last_record_time = current_time

    def generate_report(self):
        if not self.records:
            return
        
        normal_count = sum(1 for r in self.records if r["Status"] == "Normal")
        slouch_count = sum(1 for r in self.records if r["Status"] == "Slouching")
        total = normal_count + slouch_count
        
        report_window = tk.Toplevel()
        report_window.title("Posture Report")
        report_window.geometry("300x220")
        
        normal_pct = (normal_count / total) * 100 if total > 0 else 0
        slouch_pct = (slouch_count / total) * 100 if total > 0 else 0
        
        tk.Label(report_window, text="--- Posture Report ---", font=("Arial", 14, "bold")).pack(pady=10)
        tk.Label(report_window, text=f"Good Posture: {normal_count} sec ({normal_pct:.1f}%)", fg="green", font=("Arial", 12)).pack(pady=5)
        tk.Label(report_window, text=f"Slouching: {slouch_count} sec ({slouch_pct:.1f}%)", fg="red", font=("Arial", 12)).pack(pady=5)
        
        advice = "Excellent! Keep it up." if normal_pct > 80 else "Please sit straight up!"
        tk.Label(report_window, text=f"Advice: {advice}", font=("Arial", 11, "italic"), fg="blue").pack(pady=10)

class PostureApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        self.detector = PoseDetector()
        self.data_manager = DataManager()
        
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.right_panel = tk.Frame(window)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        self.status_label = tk.Label(self.right_panel, text="Initializing...", font=("Arial", 20), fg="blue")
        self.status_label.pack(pady=20)
        
        self.btn_calibrate = tk.Button(self.right_panel, text="Calibrate Posture", font=("Arial", 12), command=self.calibrate)
        self.btn_calibrate.pack(pady=10)
        
        self.btn_report = tk.Button(self.right_panel, text="Generate Report", font=("Arial", 12), command=self.data_manager.generate_report)
        self.btn_report.pack(pady=10)
        
        self.slouch_start_time = None
        self.update()
        self.window.mainloop()

    def calibrate(self):
        ret, frame = self.vid.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.face_cascade.detectMultiScale(gray, 1.05, 2)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                # 校正時，儲存「中心點」作為藍線基準
                self.detector.calibrated_y = y + (h / 2)
                self.status_label.config(text="Calibrated!", fg="green")

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.flip(frame, 1)
            status, processed_frame = self.detector.check_posture(frame)
            
            self.data_manager.record(status)
            
            if status == "Slouching":
                if self.slouch_start_time is None:
                    self.slouch_start_time = time.time()
                elif time.time() - self.slouch_start_time > 2:
                    self.status_label.config(text="WARNING: Slouching!", fg="red")
                    self.window.config(bg="red")
            else:
                self.slouch_start_time = None
                self.window.config(bg="SystemButtonFace")
                if status == "Normal":
                    self.status_label.config(text="Good Posture", fg="green")
                elif status == "Need Calibration":
                    self.status_label.config(text="Please Calibrate", fg="orange")
                else:
                    self.status_label.config(text="No Person Detected", fg="gray")

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            
        self.window.after(15, self.update)

if __name__ == "__main__":
    PostureApp(tk.Tk(), "Smart Posture Monitor")