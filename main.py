# ============================================
# DRIVER DROWSINESS MONITORING SYSTEM
# MacBook Compatible
# ============================================

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import subprocess

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================
# 1. CONFIGURATION
# ============================================

MODEL_PATH = "models/face_landmarker(1).task"

# Eye Aspect Ratio threshold
EAR_THRESHOLD = 0.21

# Eyes closed duration
DROWSY_TIME = 1.5

# Mouth Aspect Ratio threshold
MAR_THRESHOLD = 0.60

# Yawning duration
YAWN_TIME = 1.0

# Alert gap
ALERT_COOLDOWN = 5


# ============================================
# 2. CHECK MODEL FILE
# ============================================

if not os.path.exists(MODEL_PATH):

    print("❌ Model file not found!")
    print("Expected:")
    print(os.path.abspath(MODEL_PATH))

    exit()

print("✅ Model found:")
print(os.path.abspath(MODEL_PATH))


# ============================================
# 3. MEDIAPIPE SETUP
# ============================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

print("Loading MediaPipe...")

detector = vision.FaceLandmarker.create_from_options(
    options
)

print("✅ MediaPipe loaded successfully!")


# ============================================
# 4. DISTANCE FUNCTION
# ============================================

def distance(p1, p2):

    return np.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# ============================================
# 5. EAR FUNCTION
# ============================================

def calculate_ear(landmarks, points):

    p1 = landmarks[points[0]]
    p2 = landmarks[points[1]]
    p3 = landmarks[points[2]]
    p4 = landmarks[points[3]]
    p5 = landmarks[points[4]]
    p6 = landmarks[points[5]]

    # Vertical eye distances
    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)

    # Horizontal eye distance
    horizontal = distance(p1, p4)

    if horizontal == 0:
        return 0

    # Eye Aspect Ratio
    ear = (
        vertical_1 + vertical_2
    ) / (2 * horizontal)

    return ear
    ear = (
        vertical_1 + vertical_2
    ) / (2 * horizontal)

    return ear


# ============================================
# 6. MAR FUNCTION
# ============================================

def calculate_mar(landmarks):

    left = landmarks[61]

    right = landmarks[291]

    top = landmarks[13]

    bottom = landmarks[14]

    horizontal = distance(left, right)

    vertical = distance(top, bottom)

    if horizontal == 0:
        return 0

    mar = vertical / horizontal

    return mar


# ============================================
# 7. MAC ALERT
# ============================================

def alert():

    print("🚨 DROWSINESS ALERT!")

    try:

        subprocess.Popen(
            [
                "say",
                "Wake up. You look drowsy."
            ]
        )

    except Exception as e:

        print("Alert error:", e)


# ============================================
# 8. EYE LANDMARKS
# ============================================

LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


# ============================================
# 9. CAMERA
# ============================================

print("Opening camera...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("❌ Camera not available!")

    print(
        "Go to System Settings → "
        "Privacy & Security → Camera"
    )

    exit()

print("✅ Camera opened!")

print("Press Q to quit.")


# ============================================
# 10. VARIABLES
# ============================================

eyes_closed_start = None

yawn_start = None

last_alert_time = 0

frame_count = 0


# ============================================
# 11. MAIN LOOP
# ============================================

while True:

    # ----------------------------------------
    # Read camera
    # ----------------------------------------

    ret, frame = cap.read()

    if not ret:

        print("❌ Camera frame error")

        break


    # ----------------------------------------
    # Mirror image
    # ----------------------------------------

    frame = cv2.flip(frame, 1)


    # ----------------------------------------
    # Convert BGR → RGB
    # ----------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ----------------------------------------
    # MediaPipe Image
    # ----------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    # ----------------------------------------
    # Timestamp
    # ----------------------------------------

    timestamp = frame_count * 33

    frame_count += 1


    # ----------------------------------------
    # Face Detection
    # ----------------------------------------

    result = detector.detect_for_video(
        mp_image,
        timestamp
    )


    # ========================================
    # DEFAULT VALUES
    # ========================================

    status = "NO FACE"

    status_color = (0, 0, 255)

    ear = 0.0

    mar = 0.0


    # ========================================
    # FACE FOUND
    # ========================================

    if len(result.face_landmarks) > 0:

        # Get first face
        landmarks = result.face_landmarks[0]


        # ------------------------------------
        # Face detected
        # ------------------------------------

        status = "AWAKE"

        status_color = (0, 255, 0)


        # ------------------------------------
        # Calculate LEFT EAR
        # ------------------------------------

        left_ear = calculate_ear(
            landmarks,
            LEFT_EYE
        )


        # ------------------------------------
        # Calculate RIGHT EAR
        # ------------------------------------

        right_ear = calculate_ear(
            landmarks,
            RIGHT_EYE
        )


        # ------------------------------------
        # Average EAR
        # ------------------------------------

        ear = (
            left_ear +
            right_ear
        ) / 2


        # ------------------------------------
        # Calculate MAR
        # ------------------------------------

        mar = calculate_mar(
            landmarks
        )


        current_time = time.time()


        # ====================================
        # DROWSINESS DETECTION
        # ====================================

        if ear < EAR_THRESHOLD:

            if eyes_closed_start is None:

                eyes_closed_start = current_time


            closed_time = (
                current_time -
                eyes_closed_start
            )


            if closed_time >= DROWSY_TIME:

                status = "DROWSY!"

                status_color = (0, 0, 255)


                # Alert
                if (
                    current_time -
                    last_alert_time
                    >= ALERT_COOLDOWN
                ):

                    alert()

                    last_alert_time = (
                        current_time
                    )


            else:

                status = "EYES CLOSED"

                status_color = (
                    0,
                    165,
                    255
                )


        else:

            eyes_closed_start = None


        # ====================================
        # YAWN DETECTION
        # ====================================

        if mar > MAR_THRESHOLD:

            if yawn_start is None:

                yawn_start = current_time


            yawn_duration = (
                current_time -
                yawn_start
            )


            if yawn_duration >= YAWN_TIME:

                status = "YAWNING"

                status_color = (
                    255,
                    0,
                    255
                )

        else:

            yawn_start = None


        # ====================================
        # DRAW EYE POINTS
        # ====================================

        h, w, _ = frame.shape


        for index in LEFT_EYE:

            point = landmarks[index]

            x = int(point.x * w)

            y = int(point.y * h)

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )


        for index in RIGHT_EYE:

            point = landmarks[index]

            x = int(point.x * w)

            y = int(point.y * h)

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )


        # ====================================
        # DRAW MOUTH POINTS
        # ====================================

        MOUTH_POINTS = [
            61,
            291,
            13,
            14
        ]


        for index in MOUTH_POINTS:

            point = landmarks[index]

            x = int(point.x * w)

            y = int(point.y * h)

            cv2.circle(
                frame,
                (x, y),
                3,
                (255, 0, 255),
                -1
            )


    # ========================================
    # DISPLAY STATUS
    # ========================================

    cv2.putText(
        frame,
        f"Status: {status}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        2
    )


    # ========================================
    # DISPLAY EAR
    # ========================================

    cv2.putText(
        frame,
        f"EAR: {ear:.2f}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================
    # DISPLAY MAR
    # ========================================

    cv2.putText(
        frame,
        f"MAR: {mar:.2f}",
        (30, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================
    # INSTRUCTION
    # ========================================

    cv2.putText(
        frame,
        "Press Q to Quit",
        (30, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================
    # SHOW CAMERA
    # ========================================

    cv2.imshow(
        "Driver Drowsiness Monitoring System",
        frame
    )


    # ========================================
    # QUIT
    # ========================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================
# 12. CLEANUP
# ============================================

cap.release()

cv2.destroyAllWindows()

detector.close()

print("✅ Program stopped.")