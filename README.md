# 🚗 Driver Drowsiness Monitoring System

## 📌 Project Overview

The **Driver Drowsiness Monitoring System** is a real-time computer vision application designed to monitor a driver's facial features and identify signs of **drowsiness and yawning**.

The system uses **MediaPipe Face Landmarker** to detect facial landmarks and calculates:

- 👁️ **Eye Aspect Ratio (EAR)** – used to detect prolonged eye closure
- 👄 **Mouth Aspect Ratio (MAR)** – used to detect yawning
- 📷 **Real-time camera feed** – using WebRTC
- 🚨 **Drowsiness Alert** – when the driver's eyes remain closed for a specific duration

The application is developed using **Python** and deployed as an interactive **Streamlit application**.

---

## 🎯 Problem Statement

Driver fatigue and drowsiness are major causes of road accidents.

The objective of this project is to develop a real-time monitoring system that can:

1. Detect the driver's face.
2. Monitor eye movements.
3. Detect prolonged eye closure.
4. Detect yawning.
5. Display the driver's current alertness status.
6. Provide a simple browser-based interface.

---

## 🎯 Objectives

- Real-time driver face detection
- Eye landmark detection
- Mouth landmark detection
- Calculate EAR
- Calculate MAR
- Detect drowsiness
- Detect yawning
- Display live monitoring results
- Build an interactive Streamlit dashboard
- Deploy the application using Streamlit Community Cloud

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| 🐍 Python | Programming Language |
| 📷 OpenCV | Image and video processing |
| 🧠 MediaPipe | Facial landmark detection |
| 🔢 NumPy | Numerical calculations |
| 🌐 Streamlit | Web application |
| 🎥 Streamlit-WebRTC | Real-time webcam streaming |
| 🔊 WebRTC | Browser camera communication |

---

## 📂 Project Structure

```text
driver-drowsiness-monitoring/
│
├── app.py
├── main.py
├── requirements.txt
├── packages.txt
│
├── models/
│   └── face_landmarker(1).task
│
└── README.md
