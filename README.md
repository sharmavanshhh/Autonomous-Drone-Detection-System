# Autonomous-Drone-Detection-System

A **Multimodal Camera-Based Drone Detection System** designed to identify unauthorized drone activities in restricted areas. The system leverages **YOLO**, **RetinaNet** (Detectron2), and **One-Class SVM** to detect drones in images, videos, and live feeds using a mobile phone camera (via DroidCam).

This repository provides the **pipeline for drone detection** and the **GUI** for visualizing detection results in real-time. It does not contain the dataset, trained models, or model training/testing codes.

## Project Overview

### Features:
- **Multimodal Detection**: The system detects drones using multiple models:
  - **YOLO**: For real-time drone detection.
  - **RetinaNet** (via Detectron2): For more accurate detection, complementing YOLO.
  - **One-Class SVM**: To filter false positives and further classify detected objects.
- **Real-time Detection**: Process images, videos, and live webcam feeds.
- **Graphical User Interface (GUI)**: Built with **PyQt5**, providing a user-friendly interface for drone detection and real-time data visualization.
- **Alarm System**: Plays an audio alarm whenever a drone is detected.
- **ECG-style Graph**: Displays real-time drone detection data over the last 30 frames, like an ECG waveform.

### Workflow:
1. **Image/Video Processing**:
   - Detection results are saved in the following directories:
     - `results/images`
     - `results/videos`
     - `results/live`
   - Cropped objects from the detections are saved for further classification:
     - `cropped_objects/images`
     - `cropped_objects/videos`
     - `cropped_objects/live`
   - Cropped objects are passed to **One-Class SVM** for further classification to reduce false positives.
   
2. **Live Feed**:
   - The system supports real-time drone detection using your mobile phone camera via **DroidCam**.

3. **Detection Alert**:
   - Whenever a drone is detected, an audio alarm from the `assets/beep3.wav` folder plays as an alert.

## Installation

### Prerequisites:
- **Conda** must be installed on the system.

### Steps to Set Up:
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/Autonomous-Drone-Detection.git
   cd Autonomous-Drone-Detection
Install Dependencies:

Run the setup_migrations.bat file to install all the required dependencies.

```bash
./setup_migrations.bat
```

Run the Project:

After setting up the dependencies, use the run.bat file to run the project.

```bash
./run.bat
```
Note:
This repository does not contain the dataset on which models are trained, the trained models, or the model training/testing codes.

For dataset, trained models, and training/testing codes, contact the developer at sharmavansh0512@gmail.com.

### Usage
**GUI Interface:**

The GUI provides three modes for testing:

*Image-based Testing:* Upload an image to detect drones.

*Video-based Testing:* Upload a video file for detection.

*Live Webcam Feed:* Use your mobile camera (via DroidCam) for live detection.

The real-time ECG-like graph will display the drone detection status.

Images, videos, and live feed detections will have bounding boxes around detected drones.

**Cropped Objects:**

The system saves cropped detections for further classification by SVM to reduce false positives.

**Alarm Sound:**

Whenever a drone is detected, an alarm sound will play to notify the user.

Folder Structure
```bash

Autonomous-Drone-Detection/
├── assets/                  # Alarm audio files for alert system
├── detectron2/              # Detectron2
├── cropped_objects/         # Folder to store cropped drone objects for SVM classification
│   ├── images/              # Cropped objects from image detections
│   ├── videos/              # Cropped objects from video detections
│   └── live/                # Cropped objects from live feed detections
├── results/                 # Folder to store detection results
│   ├── images/              # Image detections
│   ├── videos/              # Video detections
│   └── live/                # Live feed detections
├── pipeline_image.py        # Python file for image-based detection pipeline
├── pipeline_mp4.py          # Python file for video-based detection pipeline (MP4 files)
├── pipeline_live.py         # Python file for live feed detection pipeline
├── ADDS.py                  # Python file for the GUI (Autonomous Drone Detection System)
├── setup_migrations.bat     # Batch file to install dependencies
├── run.bat                  # Batch file to run the project
└── README.md                # This file

```
### Screenshots
**GUI Interface:**
![Screenshot 2025-04-15 151156](https://github.com/user-attachments/assets/ce9038cd-d2e3-4bba-9cad-45c893b4b14e)


**GUI Interface with Image - Based Detections:**
![Screenshot 2025-04-15 145736](https://github.com/user-attachments/assets/2d4e1bdb-c7a6-4220-8776-c66bc5fa1c08)
![Screenshot 2025-04-15 145804](https://github.com/user-attachments/assets/77ce832f-885c-492f-af06-0722622ea51d)



**GUI Interface with Video - Based Frame - Wise Detections:**
![Screenshot 2025-04-15 145914](https://github.com/user-attachments/assets/631db248-1902-4361-8ae1-d0d7259c5436)
![Screenshot 2025-04-15 150329](https://github.com/user-attachments/assets/44a7474c-9748-4db8-9038-8dff3c905804)


*Same way GUI works in Real - Time Detections*

### Detection Results:
![IMG-20250409-WA0041_final](https://github.com/user-attachments/assets/90a41ea3-c635-4a47-bddc-59afe824b90b)
![drbrd_final](https://github.com/user-attachments/assets/e0d6fb9a-6d69-469f-8c18-e7eb6f5303af)
![IMG-20250409-WA0026_final](https://github.com/user-attachments/assets/d05b0b9d-53dd-4fbd-a9a7-15d18dfce6b7)
![2025-03-19_16-42-48](https://github.com/user-attachments/assets/20ad988e-fc5e-446a-a5bd-904496338e9d)
![2025-03-19_16-42-55](https://github.com/user-attachments/assets/fccc68b8-83fa-4521-9409-19b3ae15c80e)

### Note : *This repository does not contain trained models, training dataset, training and testing codes for models*

### Contributing
Feel free to fork this repository and contribute by submitting pull requests or raising issues if you find any bugs or have suggestions for improvements.

### Contact
For any questions or requests regarding the dataset, trained models, or training/testing code, please contact the developer:

**Email: sharmavansh0512@gmail.com**

