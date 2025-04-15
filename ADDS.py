from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QFileDialog, QTextEdit, QVBoxLayout,
    QHBoxLayout, QWidget
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl  
import cv2
import sys
from pipeline_image import process_image
from pipeline_mp4 import process_video
import os
import vlc 
from pipeline_live import process_live_feed 
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl



class LiveSignalPlot(QWidget):
    def __init__(self, max_points=30, parent=None):
        super().__init__(parent)
        self.max_points = max_points
        self.data = [0] * max_points  # Initial signal values

        self.figure = Figure(figsize=(5, 2))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # 📉 Set graph appearance like ECG
        self.ax.set_ylim(-2, 2)
        self.ax.set_xlim(0, self.max_points)
        self.ax.set_yticks([-1, 0, 1])
        self.ax.set_yticklabels(['No Drone', '', 'Drone'])
        self.ax.set_xticks([])  # Hide X-axis
        self.ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
        self.ax.set_title("Drone Detection Signal")

        self.line, = self.ax.plot(self.data, color='lime', linewidth=2)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def update_signal(self, value):
        self.data.append(value)
        self.data = self.data[-self.max_points:]  # Keep only last 30 values

        self.line.set_ydata(self.data)
        self.line.set_xdata(np.arange(len(self.data)))

        # Update x-axis range dynamically
        self.ax.set_xlim(0, self.max_points)
        self.canvas.draw()


class ConfidencePlot(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super(ConfidencePlot, self).__init__(self.fig)
        self.setParent(parent)

    def plot_confidence(self, scores):
        self.ax.clear()
        self.ax.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='-')
        self.ax.set_title("Detection Confidence")
        self.ax.set_xlabel("")
        self.ax.set_ylabel("Confidence")
        self.ax.set_ylim(0, 1.05)
        self.draw()



class LiveFeedThread(QThread):
    frame_signal = pyqtSignal(QImage)  # ✅ Emits processed frames
    log_signal = pyqtSignal(str)  # ✅ Emits logs
    latest_frame_signal = pyqtSignal(str)  # ✅ Emit latest detected frame path
    
    def __init__(self, live_signal_plot = None, beep_player = None):
        super().__init__()
        self.running = True  # ✅ Control flag for stopping
        self.live_signal_plot = live_signal_plot  # 📈 Store plot
        self.beep_player = None  # ✅ New

    def run(self):
        cap = cv2.VideoCapture(2)  # ✅ Open webcam (0 = default camera)
        if not cap.isOpened():
            self.log_signal.emit("❌ Error: Unable to access webcam.")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.log_signal.emit("❌ Error: Failed to capture frame.")
                break

            # ✅ Run detection on the frame
            processed_frame, log_message, latest_frame = process_live_feed(frame, 
                    signal_callback = self.live_signal_plot.update_signal,
                     beep_callback=self.beep_player.play if self.beep_player else None)

            # ✅ Convert OpenCV frame to QImage for PyQt
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            self.frame_signal.emit(q_image)  # ✅ Send frame to GUI
            self.log_signal.emit(log_message)  # ✅ Send log message

            if latest_frame:
                self.latest_frame_signal.emit(latest_frame)  # ✅ Send detected frame path

        cap.release()

    def stop(self):
        """✅ Stop the live feed gracefully."""
        self.running = False
        self.wait()


class VideoProcessingThread(QThread):
    progress_signal = pyqtSignal(str)  # ✅ For updating logs
    completed_signal = pyqtSignal(str)  # ✅ Emits processed video path when done

    def __init__(self, video_path, live_signal_plot=None, beep_player = None):
        super().__init__()
        self.video_path = video_path
        self._stop_flag = False
        self.live_signal_plot = live_signal_plot  # 📈 Store plot
        self.beep_player = None  # ✅ New

    def run(self):
        self.progress_signal.emit(f"⏳ Processing Video: {self.video_path}")

        # ✅ Pass stop flag check to process_video
        processed_video_path = process_video(
            self.video_path,
            log_callback=self.log_callback,
            stop_flag_check=self._check_stop,
            signal_callback=self.live_signal_plot.update_signal if self.live_signal_plot else None,
            beep_callback=self.beep_player.play if self.beep_player else None  # ✅ New
        )

        # ✅ Emit only if not manually stopped
        if not self._stop_flag:
            self.completed_signal.emit(processed_video_path)

    def log_callback(self, log_message):
        """✅ This function is called inside process_video() to emit logs live"""
        self.progress_signal.emit(log_message)
    
    def stop(self):
        self._stop_flag = True

    def process_video_with_stop(self, file_path):
        # This part depends on your actual process_video logic
        # Replace with a loop that checks self._stop_flag regularly

        from pipeline_mp4 import process_video  # Assuming your function can be modified

        return process_video(file_path, stop_flag_check=self._check_stop)

    def _check_stop(self):
        return self._stop_flag


class DroneDetectionGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.stop_processing = False  # ✅ Control flag for stopping processing
        self.beep_player = QMediaPlayer()
        self.beep_player.setMedia(QMediaContent(QUrl.fromLocalFile("assets/beep3.mp3")))



        # ✅ Apply Dark Mode Theme
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: #ffffff;
                font-family: Arial;
            }
            QLabel {
                border: 2px solid #444;
                background-color: #1E1E1E;
                font-size: 16px;
                padding: 10px;
            }
            QPushButton {
                background-color: #333;
                border: 2px solid #555;
                color: white;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #444;
            }
            QPushButton:pressed {
                background-color: #555;
            }
            QTextEdit {
                border: 2px solid #444;
                background-color: #1E1E1E;
                color: #ddd;
                font-size: 14px;
                
            }
        """)

        # ✅ Live Feed Section
        self.live_feed_label = QLabel("🔴 Live Feed / Uploaded Image / Uploaded Video")
        self.live_feed_label.setMinimumSize(400, 350)
        self.live_feed_label.setAlignment(Qt.AlignCenter)

        # ✅ Latest Detection Frame Section
        self.latest_frame_label = QLabel("🖼 Detection Results")
        self.latest_frame_label.setMinimumSize(400, 350)
        self.latest_frame_label.setAlignment(Qt.AlignCenter)

        # ✅ Logs Section
        self.logs_text = QTextEdit()
        self.logs_text.setReadOnly(True)
        self.logs_text.setMinimumHeight(150)
        self.logs_text.setMaximumHeight(300)

        # Graph
        self.confidence_plot = ConfidencePlot()
        # self.confidence_plot.setMinimumWidth(250)
        self.live_signal_plot = LiveSignalPlot()
        # self.confidence_plot.setMinimumWidth(250)
        self.confidence_plot.hide()
        self.live_signal_plot.hide()


        # ✅ Buttons
        self.image_test_btn = QPushButton("📸 Image-Based Testing")
        self.image_test_btn.clicked.connect(self.image_testing)

        self.video_test_btn = QPushButton("🎥 Video-Based Testing")
        self.video_test_btn.clicked.connect(self.video_testing)

        self.stop_button = QPushButton("🛑 Stop Video Processing")
        self.stop_button.clicked.connect(self.stop_video_processing)

        self.start_live_feed_btn = QPushButton("🚀 Start Live Feed")
        self.start_live_feed_btn.clicked.connect(self.start_live_feed)

        self.stop_live_feed_btn = QPushButton("⏹ Stop Live Feed")
        self.stop_live_feed_btn.clicked.connect(self.stop_live_feed)


         # ✅ Video Player Instances (VLC)
        self.instance = vlc.Instance()
        self.player_uploaded = self.instance.media_player_new()
        self.player_processed = self.instance.media_player_new()

        # ✅ Layouts
        main_layout = QVBoxLayout()

        # 🔹 Live Feed & Latest Frame Side-by-Side
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.live_feed_label)
        top_layout.addWidget(self.latest_frame_label)

        # 🔹 Logs and Graph side by side
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.logs_text, 1)
        bottom_layout.addWidget(self.confidence_plot, 1)
        bottom_layout.addWidget(self.live_signal_plot, 1)  # Add beside logs or feed


        # 🔹 Buttons Layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.image_test_btn)
        button_layout.addWidget(self.video_test_btn)
        button_layout.addWidget(self.stop_button)  # Wherever it fits in your layout
        self.stop_button.setEnabled(False)  # Only enable when video is running
        button_layout.addWidget(self.start_live_feed_btn)
        button_layout.addWidget(self.stop_live_feed_btn)

        # ✅ Add layouts to main layout
        main_layout.addLayout(top_layout, 5)
        main_layout.addLayout(bottom_layout, 4)
        main_layout.addLayout(button_layout, 1)

        self.setLayout(main_layout)
        self.setWindowTitle("Autonomous Drone Detection System")
        self.setGeometry(100, 100, 900, 600)

    # ✅ Live Feed Thread
        self.live_feed_thread = None
    
    def start_live_feed(self):
        """✅ Start live feed with detection."""
        if self.live_feed_thread is None or not self.live_feed_thread.isRunning():
            self.logs_text.append("🚀 Live Feed Started")
            self.live_feed_label.setText("🔴 Live Feed Running...")

            self.live_feed_thread = LiveFeedThread()
            self.confidence_plot.hide()
            self.live_signal_plot.show()
            self.live_signal_plot.data = [0] * self.live_signal_plot.max_points
            self.live_signal_plot.update_signal(0)
            self.live_feed_thread.live_signal_plot = self.live_signal_plot  # Pass plot reference
            self.live_feed_thread.beep_player = self.beep_player
            self.live_feed_thread.frame_signal.connect(self.update_live_feed)
            self.live_feed_thread.log_signal.connect(self.update_logs)
            self.live_feed_thread.latest_frame_signal.connect(self.update_latest_detection_frame)  # ✅ NEW SIGNAL
            self.live_feed_thread.start()

    def stop_live_feed(self):
        """✅ Stop live feed."""
        if self.live_feed_thread and self.live_feed_thread.isRunning():
            self.logs_text.append("⏹ Live Feed Stopped")
            self.live_feed_thread.stop()
            self.live_feed_thread = None

        self.live_feed_label.setText("🔴 Live Feed / Uploaded Image / Uploaded Video")

    def update_live_feed(self, q_image):
        """✅ Update live feed label with detected frame."""
        pixmap = QPixmap.fromImage(q_image)
        self.live_feed_label.setPixmap(pixmap)

    def update_latest_detection_frame(self, image_path):
        pixmap = QPixmap(image_path)
        self.latest_frame_label.setPixmap(pixmap)
        self.latest_frame_label.setScaledContents(True)


    def convert_cv_to_qimage(self, cv_img):
        """ Convert OpenCV image to QImage """
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    
    def image_testing(self):
     """ Function to test image using pipeline """
     file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")

     if not file_path:
         return  # ❌ Agar file select nahi hui to exit

     # ✅ Reset logs **BEFORE** appending any new logs
     self.logs_text.clear()
     QApplication.processEvents()  # 🔹 **Force update GUI immediately**

     self.logs_text.append(f"📢 Processing Image: {file_path}")

     # ✅ Show Uploaded Image in Live Feed Section
     pixmap = QPixmap(file_path).scaled(720, 720, Qt.KeepAspectRatio)
     self.live_feed_label.setPixmap(pixmap)

     # ✅ Run Detection
     final_image_path, log_message, confidence_scores = process_image(file_path)

     # ✅ Show Detection Results
     if final_image_path:
         processed_pixmap = QPixmap(final_image_path).scaled(720, 720, Qt.KeepAspectRatio)
         self.latest_frame_label.setPixmap(processed_pixmap)
         self.live_signal_plot.hide()
         self.confidence_plot.show()
         self.logs_text.append(log_message)
         if len(confidence_scores) > 0:
            self.beep_player.play()
         self.confidence_plot.plot_confidence(confidence_scores)

     else:
         self.logs_text.append("⚠ Error during detection.")

    def smooth_loop_video(self, player, video_path):
        """Ensure seamless looping of video without a pause."""
        media_length = player.get_length()  # Get total duration (in ms)
        current_time = player.get_time()  # Get current playback time (in ms)

        if media_length > 0 and current_time >= media_length - 500:  # 500ms before end
            print("🔄 Preloading for seamless looping...")
            player.set_time(0)  # Instantly reset to beginning
            player.play()

    def update_logs(self, log):
        self.logs_text.append(log)  # ✅ Append logs instantly
        self.logs_text.verticalScrollBar().setValue(self.logs_text.verticalScrollBar().maximum())  # Auto-scroll


    def video_testing(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Videos (*.mp4 *.avi *.mov)")
        if not file_path:
            return
        
        # ✅ Stop previous videos and reset players
        self.player_uploaded.stop()
        self.player_processed.stop()


        self.logs_text.clear()
        QApplication.processEvents()  # 🔹 **Force update GUI immediately**
        self.logs_text.append(f"📢 Video Selected: {file_path}")

         # ✅ Display Video Thumbnail
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            qimage = QPixmap.fromImage(self.convert_cv_to_qimage(frame))
            self.live_feed_label.setPixmap(qimage)

         # ✅ Show Processing Message
        self.latest_frame_label.setText("⏳ Processing Video, Please Wait...")


        # ✅ Run Detection in a Separate Thread
        self.stop_button.setEnabled(True)  # Enable stop button
        self.video_thread = VideoProcessingThread(file_path)
        self.confidence_plot.hide()
        self.live_signal_plot.show()
        self.live_signal_plot.data = [0] * self.live_signal_plot.max_points
        self.live_signal_plot.update_signal(0)
        self.video_thread.live_signal_plot = self.live_signal_plot  # Pass plot reference
        self.video_thread.beep_player = self.beep_player
        self.video_thread.progress_signal.connect(self.update_logs)
        self.video_thread.completed_signal.connect(self.display_processed_video)
        # Connect signal for thread finish to disable stop button
        self.video_thread.finished.connect(lambda: self.stop_button.setEnabled(False))
        self.video_thread.start()

    def stop_video_processing(self):
        if hasattr(self, 'video_thread') and self.video_thread.isRunning():
            self.video_thread.stop()
            self.logs_text.append("❌ Stopping video processing...")

    def display_processed_video(self, processed_video_path):
        self.logs_text.append("✅ Video Processing Complete!")
        self.logs_text.append(f"📂 Processed Video: {processed_video_path}")

        # ✅ Stop previous videos before loading new ones
        self.player_uploaded.stop()
        self.player_processed.stop()

        # ✅ Load & Play Uploaded Video
        self.play_video(self.player_uploaded, self.live_feed_label, self.video_thread.video_path)

        # ✅ Load & Play Processed Video
        self.play_video(self.player_processed, self.latest_frame_label, processed_video_path)

    def play_video(self, player, widget, video_path):
        self.video_thread = VideoProcessingThread(video_path)
        self.video_thread.progress_signal.emit("")

        media = self.instance.media_new(video_path)
        player.set_media(media)
        player.set_hwnd(int(widget.winId()))
        player.play()

        # ✅ Setup a QTimer to restart the video when it ends
        timer = QTimer(self)
        timer.timeout.connect(lambda: self.smooth_loop_video(player,video_path))
        timer.start(300)  # Check every second


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DroneDetectionGUI()
    window.showMaximized()
    sys.exit(app.exec_())
