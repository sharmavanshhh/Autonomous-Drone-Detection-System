import cv2
import os
import numpy as np
import joblib
from skimage.feature import hog
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Import modular detection functions
from yolo_testing import run_yolo_detection
from retinanet_testing import run_retinanet_detection_live

# ✅ Load trained SVM model
svm_model_path = "svm/one_class_svm_for_drone.pkl"
svm_model = joblib.load(svm_model_path)

# ✅ Create directories for saving detections
RESULTS_DIR = "results/live"
CROPPED_DIR = "cropped_objects/live"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CROPPED_DIR, exist_ok=True)

# ✅ OpenCV Optimizations
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# ✅ Function to extract HOG features
def extract_hog_features(image):
    image = cv2.resize(image, (60, 60))  # Resize to match training size
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)  
    return features

# ✅ Function to classify cropped object using SVM (Runs in Parallel)
def classify_with_svm(cropped_img):
    try:
        features = extract_hog_features(cropped_img).reshape(1, -1)  # Reshape for SVM
        prediction = svm_model.predict(features)  # Predict using One-Class SVM
        return prediction[0] == 1  # Return True if drone, False otherwise
    except Exception as e:
        print(f"⚠ Error in SVM classification: {e}")
        return False

# ✅ Open Webcam (DroidCam ke liye 2, Laptop Webcam ke liye 0)
cap = cv2.VideoCapture(2)  # Change to 0 if using laptop webcam

# ✅ Resize frame to improve FPS
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

# ✅ Multi-threading setup
executor = ThreadPoolExecutor(max_workers=2)


def process_live_feed(frame, signal_callback=None, beep_callback = None):
    """
    Process a single frame for drone detection.
    Returns:
        - processed_frame (numpy array): Frame with bounding boxes.
        - log_message (str): Log message with timestamp when a drone is detected.
    """
    if frame is None:
        return None, "❌ Error: Empty frame received."

    # ✅ Resize frame for better FPS
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # ✅ YOLO Detection
    yolo_detections = run_yolo_detection(frame)

    # ✅ RetinaNet Detection
    retinanet_detections = run_retinanet_detection_live(frame)

    all_detections = yolo_detections + retinanet_detections

    # Ensure both detections are lists (if None, replace with empty list)
    yolo_detections = yolo_detections if yolo_detections is not None else []
    retinanet_detections = retinanet_detections if retinanet_detections is not None else []
    
    final_detections = []
    log_entries = []  # ✅ Store log messages
    frame_filename = None # ✅ Store frame filename

    for detections in all_detections:
        x1, y1, x2, y2, conf = detections

        # ✅ Ensure valid bounding box
        if x2 <= x1 or y2 <= y1:
            print(f"⚠ Invalid bounding box: {x1, y1, x2, y2}")
            continue  

        # ✅ Crop detected object
        cropped_obj = frame[int(y1):int(y2), int(x1):int(x2)]
        if cropped_obj.size == 0:
            continue  # Ignore empty crops

        # ✅ Save cropped image
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cropped_image_path = f"{CROPPED_DIR}/{timestamp}.jpg"
        cv2.imwrite(cropped_image_path, cropped_obj)
        if os.path.exists(cropped_image_path):
            print(f"✅ Image successfully saved: {cropped_image_path}")
        else:
            print(f"❌ Image NOT saved: {cropped_image_path}")

        # ✅ Run SVM classification in parallel
        future = executor.submit(classify_with_svm, cropped_obj)
        is_drone = future.result()

        if is_drone:
            final_detections.append((x1, y1, x2, y2, conf))

            # ✅ Generate timestamped log message
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entries.append(f"{timestamp} - 🛰️ Drone detected!")

    # ✅ Draw bounding boxes on frame
    for (x1, y1, x2, y2, conf) in final_detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Drone ({conf:.2f})", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    # ✅ Frame-wise graphs update
    if signal_callback:
        if final_detections:
            # Use max confidence of detected drones
            max_conf = max([conf for (_, _, _, _, conf) in final_detections])
            signal_callback(max_conf)
        else:
            signal_callback(-1)

    if final_detections:  # Detections found in this frame
            if beep_callback:
                beep_callback()

    # ✅ Save Frame *ONLY IF* drones are detected
    if final_detections:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        frame_filename = f"{RESULTS_DIR}/{timestamp}.jpg"
        cv2.imwrite(frame_filename, frame)

    # ✅ Prepare log message
    # ✅ Ensure log_message is always defined
    log_message = "\n".join(log_entries) if log_entries else "❌ No drones detected."   

    return frame, log_message, frame_filename  # ✅ Now returns both values

# ✅ Cleanup
cap.release()
cv2.destroyAllWindows()