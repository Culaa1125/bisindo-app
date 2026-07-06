import streamlit as st
st.write("Step 1: streamlit ok")

import numpy as np
st.write("Step 2: numpy ok")

import cv2
st.write("Step 3: cv2 ok")

import mediapipe as mp
st.write("Step 4: mediapipe ok, versi:", mp.__version__)

import tensorflow as tf
st.write("Step 5: tensorflow ok, versi:", tf.__version__)

model_cnn = tf.keras.models.load_model('models/model_cnn_abjad.keras', compile=False)
st.write("Step 6: model CNN berhasil di-load")

model_lstm = tf.keras.models.load_model('models/best_bisindo_lstm_4200dataset.keras', compile=False)
st.write("Step 7: model LSTM berhasil di-load")

import json
with open('labels/label_abjad.json', 'r') as f:
    la = json.load(f)
with open('labels/label_map.json', 'r') as f:
    lk = json.load(f)
st.write("Step 8: labels berhasil di-load")

mp_holistic = mp.solutions.holistic
st.write("Step 9: mp_holistic.Holistic reference ok")

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=True,
    smooth_segmentation=True
)
st.write("Step 10: Holistic() berhasil diinstansiasi")

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
st.write("Step 11: streamlit_webrtc import ok")

from av import VideoFrame
st.write("Step 12: av import ok")

st.write("SEMUA STEP LOLOS — tidak ada crash sampai di sini")
