import streamlit as st
st.write("Step 1: streamlit ok")

import numpy as np
st.write("Step 2: numpy ok")

import cv2
st.write("Step 3: cv2 ok")

import mediapipe as mp
st.write("Step 4: mediapipe ok, versi:", mp.__version__)

st.write("Mencoba import tensorflow...")
import tensorflow as tf
st.write("Step 5: tensorflow ok, versi:", tf.__version__)

st.write("Mencoba load model CNN...")
model = tf.keras.models.load_model('models/model_cnn_abjad.keras', compile=False)
st.write("Step 6: model CNN berhasil di-load")
