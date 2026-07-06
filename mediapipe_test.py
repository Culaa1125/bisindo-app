import mediapipe as mp
import streamlit as st

st.write("MediaPipe:", mp.__version__)

try:
    h = mp.solutions.holistic.Holistic(
        model_complexity=1,
        enable_segmentation=False
    )
    st.success("Holistic berhasil dibuat")
except Exception as e:
    st.error(e)
