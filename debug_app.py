import streamlit as st
st.write("Step 1: streamlit ok")

import mediapipe as mp
st.write("Step 2: mediapipe ok, versi:", mp.__version__)

# Tes paling ringan dulu: mp.solutions.hands (bukan Holistic)
mp_hands = mp.solutions.hands
st.write("Step 3: mp_hands reference ok")

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
st.write("Step 4: Hands() berhasil diinstansiasi")

# Kalau lolos, baru coba Holistic dengan model_complexity paling ringan
mp_holistic = mp.solutions.holistic
st.write("Step 5: mp_holistic reference ok")

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=0,   # <-- diturunkan dari 1 ke 0 (paling ringan)
    smooth_landmarks=False,
    enable_segmentation=False,
    smooth_segmentation=False,
    refine_face_landmarks=False
)
st.write("Step 6: Holistic() berhasil diinstansiasi")
