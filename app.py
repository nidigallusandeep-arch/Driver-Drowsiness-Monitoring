import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av


# ============================================
# STREAMLIT PAGE
# ============================================

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)


st.title("🚗 Driver Drowsiness Monitoring System")

st.write(
    "Real-time driver drowsiness detection using "
    "MediaPipe and OpenCV."
)


# ============================================
# MODEL PATH
# ============================================

MODEL_PATH = "models/face_landmarker(1).task"


# ============================================
# CHECK MODEL
# ============================================

if not os.path.exists(MODEL_PATH):

    st.error(
        f"Model not found: {MODEL_PATH}"
    )

    st.stop()


# ============================================
# MEDIAPIPE
# ============================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)


options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)


# ============================================
# HELPER FUNCTIONS
# ============================================

def distance(p1, p2):

    return np.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def calculate_ear(landmarks, points):

    p1 = landmarks[points[0]]
    p2 = landmarks[points[1]]
    p3 = landmarks[points[2]]
    p4 = landmarks[points[3]]
    p5 = landmarks[points[4]]
    p6 = landmarks[points[5]]

    vertical_1 = distance(p2, p6)

    vertical_2 = distance(p3, p5)

    horizontal = distance(p1, p4)

    if horizontal == 0:
        return 0

    ear = (
        vertical_1 + vertical_2
    ) / (2 * horizontal)

    return ear


def calculate_mar(landmarks):

    left = landmarks[61]

    right = landmarks[291]

    top = landmarks[13]

    bottom = landmarks[14]

    horizontal = distance(
        left,
        right
    )

    vertical = distance(
        top,
        bottom
    )

    if horizontal == 0:
        return 0

    return vertical / horizontal


# ============================================
# EYE POINTS
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


MOUTH_POINTS = [
    61,
    291,
    13,
    14
]


# ============================================
# VIDEO PROCESSOR
# ============================================

class DrowsinessProcessor(VideoProcessorBase):

    def __init__(self):

        self.detector = (
            vision.FaceLandmarker.create_from_options(
                options
            )
        )

        self.eyes_closed_start = None

        self.yawn_start = None


    def recv(self, frame):

        # Convert video frame
        img = frame.to_ndarray(
            format="bgr24"
        )


        # Mirror camera
        img = cv2.flip(
            img,
            1
        )


        # Convert BGR → RGB
        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )


        # MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # Detect face
        result = self.detector.detect(
            mp_image
        )


        # Default
        status = "NO FACE"

        status_color = (
            0,
            0,
            255
        )

        ear = 0.0

        mar = 0.0


        # ========================================
        # FACE DETECTED
        # ========================================

        if len(result.face_landmarks) > 0:

            landmarks = (
                result.face_landmarks[0]
            )


            status = "AWAKE"

            status_color = (
                0,
                255,
                0
            )


            # ------------------------------------
            # EAR
            # ------------------------------------

            left_ear = calculate_ear(
                landmarks,
                LEFT_EYE
            )

            right_ear = calculate_ear(
                landmarks,
                RIGHT_EYE
            )

            ear = (
                left_ear +
                right_ear
            ) / 2


            # ------------------------------------
            # MAR
            # ------------------------------------

            mar = calculate_mar(
                landmarks
            )


            current_time = time.time()


            # ====================================
            # DROWSINESS
            # ====================================

            if ear < 0.21:

                if self.eyes_closed_start is None:

                    self.eyes_closed_start = (
                        current_time
                    )


                closed_time = (
                    current_time -
                    self.eyes_closed_start
                )


                if closed_time >= 1.5:

                    status = "DROWSY!"

                    status_color = (
                        0,
                        0,
                        255
                    )

                else:

                    status = "EYES CLOSED"

                    status_color = (
                        0,
                        165,
                        255
                    )


            else:

                self.eyes_closed_start = None


            # ====================================
            # YAWNING
            # ====================================

            if mar > 0.60:

                if self.yawn_start is None:

                    self.yawn_start = (
                        current_time
                    )


                yawn_time = (
                    current_time -
                    self.yawn_start
                )


                if yawn_time >= 1.0:

                    status = "YAWNING"

                    status_color = (
                        255,
                        0,
                        255
                    )

            else:

                self.yawn_start = None


            # ====================================
            # DRAW EYE LANDMARKS
            # ====================================

            h, w, _ = img.shape


            for index in LEFT_EYE:

                point = landmarks[index]

                x = int(point.x * w)

                y = int(point.y * h)

                cv2.circle(
                    img,
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
                    img,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1
                )


            # ====================================
            # DRAW MOUTH
            # ====================================

            for index in MOUTH_POINTS:

                point = landmarks[index]

                x = int(point.x * w)

                y = int(point.y * h)

                cv2.circle(
                    img,
                    (x, y),
                    3,
                    (255, 0, 255),
                    -1
                )


        # ========================================
        # DISPLAY STATUS
        # ========================================

        cv2.putText(
            img,
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
            img,
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
            img,
            f"MAR: {mar:.2f}",
            (30, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # Return frame
        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# ============================================
# STREAMLIT WEBRTC
# ============================================

st.subheader("📷 Live Camera")

webrtc_streamer(
    key="driver-drowsiness",
    video_processor_factory=DrowsinessProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)


# ============================================
# INFORMATION
# ============================================

st.markdown("---")

st.subheader("📊 Detection Information")

col1, col2, col3 = st.columns(3)

with col1:

    st.info(
        "**EAR**\n\n"
        "Eye Aspect Ratio is used "
        "to detect closed eyes."
    )


with col2:

    st.info(
        "**MAR**\n\n"
        "Mouth Aspect Ratio is used "
        "to detect yawning."
    )


with col3:

    st.info(
        "**Drowsiness**\n\n"
        "Eyes closed for more than "
        "1.5 seconds → DROWSY."
    )


st.markdown("---")

st.caption(
    "Built with Python • OpenCV • MediaPipe • Streamlit"
)