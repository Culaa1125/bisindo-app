import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import time
import threading
from threading import Lock
from collections import deque
from streamlit_webrtc import (webrtc_streamer,
                               VideoProcessorBase,
                               RTCConfiguration)
from av import VideoFrame

# ─────────────── PAGE CONFIG ───────────────
st.set_page_config(
    page_title="BISINDO Translator",
    page_icon="🤟",
    layout="wide"
)

@st.cache_resource
def get_rtc_configuration():
    """
    Bangun konfigurasi ICE server, dengan urutan prioritas:
    1. Twilio (kalau credential ada) — paling stabil untuk produksi.
    2. Open Relay Project (Metered) — TURN gratis, TIDAK perlu daftar
       akun/verifikasi nomor HP, tapi kurang stabil untuk skala besar.
       Bagus untuk testing dulu sambil urus Twilio.
    3. STUN-only Google — fallback terakhir, bisa gagal di jaringan HP.
    """
    # ── 1. Coba Twilio dulu ──
    try:
        account_sid = st.secrets.get("TWILIO_ACCOUNT_SID")
        auth_token  = st.secrets.get("TWILIO_AUTH_TOKEN")
    except Exception:
        account_sid = None
        auth_token  = None

    account_sid = account_sid or os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token  = auth_token  or os.environ.get("TWILIO_AUTH_TOKEN")

    if account_sid and auth_token:
        try:
            from twilio.rest import Client
            client = Client(account_sid, auth_token)
            token  = client.tokens.create()
            return RTCConfiguration({"iceServers": token.ice_servers})
        except Exception as e:
            print(f"Gagal ambil TURN server dari Twilio: {e}")

    # ── 2. Fallback: Metered.ca TURN (free tier, cukup daftar pakai
    # email — TIDAK perlu verifikasi nomor HP seperti Twilio) ──
    # Daftar gratis di https://www.metered.ca/tools/openrelay/ lalu
    # ambil "API Key" dari dashboard, simpan sebagai METERED_API_KEY
    # di Secrets. Kode di bawah akan fetch kredensial TURN sementara
    # dari API mereka (kredensialnya expire & di-generate ulang tiap
    # request, jadi tidak bisa di-hardcode).
    metered_key = None
    try:
        metered_key = st.secrets.get("METERED_API_KEY")
    except Exception:
        pass
    metered_key = metered_key or os.environ.get("METERED_API_KEY")

    if metered_key:
        try:
            import requests
            # GANTI "YOUR-APP-NAME" sesuai subdomain yang diberikan
            # Metered di dashboard kamu (Settings → App name), atau
            # simpan sebagai secret METERED_APP_NAME.
            app_name = None
            try:
                app_name = st.secrets.get("METERED_APP_NAME")
            except Exception:
                pass
            app_name = (app_name
                        or os.environ.get("METERED_APP_NAME")
                        or "YOUR-APP-NAME")

            resp = requests.get(
                f"https://{app_name}.metered.live/api/v1/turn/credentials"
                f"?apiKey={metered_key}",
                timeout=5
            )
            if resp.status_code == 200:
                return RTCConfiguration({"iceServers": resp.json()})
            else:
                print(f"Metered API error: {resp.status_code} {resp.text}")
        except Exception as e:
            print(f"Gagal ambil TURN server dari Metered: {e}")

    # ── 3. Fallback terakhir: STUN-only ──
    return RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    })

RTC_CONFIGURATION = get_rtc_configuration()

# ─────────────── LOAD MODEL ───────────────
@st.cache_resource(show_spinner="Loading AI Models...")
def load_models():
    cnn = tf.keras.models.load_model(
        "models/model_cnn_abjad.keras",
        compile=False,
    )
    lstm = tf.keras.models.load_model(
        "models/best_bisindo_lstm_4200dataset.keras",
        compile=False,
    )
    return {
        "cnn": cnn,
        "lstm": lstm,
    }

@st.cache_data
def load_labels():
    with open('labels/label_abjad.json', 'r') as f:
        la = json.load(f)
    with open('labels/label_map.json', 'r') as f:
        lk = json.load(f)
    abjad    = [la[str(i)] for i in range(len(la))]
    kosakata = [k for k, v in sorted(lk.items(),
                                      key=lambda x: x[1])]
    return abjad, kosakata

def get_models():
    models = load_models()
    return models["cnn"], models["lstm"]

ABJAD, KOSAKATA       = load_labels()

mp_holistic = mp.solutions.holistic
mp_draw     = mp.solutions.drawing_utils

# ─────────────── LANDMARK EXTRACTION ───────────────
def extract_cnn(results):
    """126 fitur tangan untuk CNN."""
    lm = np.zeros(126)
    if results.right_hand_landmarks:
        for j, pt in enumerate(
                results.right_hand_landmarks.landmark):
            lm[j*3]   = pt.x
            lm[j*3+1] = pt.y
            lm[j*3+2] = pt.z
    if results.left_hand_landmarks:
        for j, pt in enumerate(
                results.left_hand_landmarks.landmark):
            lm[63+j*3]   = pt.x
            lm[63+j*3+1] = pt.y
            lm[63+j*3+2] = pt.z
    return lm

def extract_lstm(results):
    """
    258 fitur dengan urutan SAMA dengan saat training:
    pose (132) + left_hand (63) + right_hand (63)
    """
    pose = np.array(
        [[lm.x, lm.y, lm.z, lm.visibility]
         for lm in results.pose_landmarks.landmark]
        if results.pose_landmarks
        else np.zeros((33, 4))
    ).flatten()

    lh = np.array(
        [[lm.x, lm.y, lm.z]
         for lm in results.left_hand_landmarks.landmark]
        if results.left_hand_landmarks
        else np.zeros((21, 3))
    ).flatten()

    rh = np.array(
        [[lm.x, lm.y, lm.z]
         for lm in results.right_hand_landmarks.landmark]
        if results.right_hand_landmarks
        else np.zeros((21, 3))
    ).flatten()

    return np.concatenate([pose, lh, rh])

def normalize_hand_landmarks(lm_cnn):
    """Normalize tangan relatif ke pergelangan."""
    lm = lm_cnn.copy()
    wrist_r = lm[:3].copy()
    if np.any(wrist_r):
        scale = np.max(np.abs(lm[:63])) + 1e-6
        for j in range(21):
            lm[j*3]   = (lm[j*3]   - wrist_r[0]) / scale
            lm[j*3+1] = (lm[j*3+1] - wrist_r[1]) / scale
            lm[j*3+2] = (lm[j*3+2] - wrist_r[2]) / scale
    wrist_l = lm[63:66].copy()
    if np.any(wrist_l):
        scale = np.max(np.abs(lm[63:])) + 1e-6
        for j in range(21):
            base = 63 + j*3
            lm[base]   = (lm[base]   - wrist_l[0]) / scale
            lm[base+1] = (lm[base+1] - wrist_l[1]) / scale
            lm[base+2] = (lm[base+2] - wrist_l[2]) / scale
    return lm

def normalize_pose_landmarks(lm_lstm):
    """
    Normalize sesuai urutan: pose(132) + lh(63) + rh(63)
    """
    lm = lm_lstm.copy()

    left_shoulder  = lm[11*4 : 11*4+3]
    right_shoulder = lm[12*4 : 12*4+3]

    if not (np.any(left_shoulder) and np.any(right_shoulder)):
        return lm

    center = (left_shoulder + right_shoulder) / 2
    scale  = np.linalg.norm(left_shoulder - right_shoulder)
    if scale < 1e-6:
        return lm

    for j in range(33):
        lm[j*4]   = (lm[j*4]   - center[0]) / scale
        lm[j*4+1] = (lm[j*4+1] - center[1]) / scale
        lm[j*4+2] = (lm[j*4+2] - center[2]) / scale

    for j in range(21):
        base = 132 + j*3
        lm[base]   = (lm[base]   - center[0]) / scale
        lm[base+1] = (lm[base+1] - center[1]) / scale
        lm[base+2] = (lm[base+2] - center[2]) / scale

    for j in range(21):
        base = 195 + j*3
        lm[base]   = (lm[base]   - center[0]) / scale
        lm[base+1] = (lm[base+1] - center[1]) / scale
        lm[base+2] = (lm[base+2] - center[2]) / scale

    return lm

# ─────────────── MOTION DETECTOR ───────────────
class MotionDetector:
    def __init__(self, window=15):
        self.window = window
        self.votes  = deque(maxlen=window)

    def update(self, lm_buffer, low=0.005, high=0.015):
        if len(lm_buffer) < 10:
            return 'unknown'

        recent = list(lm_buffer)[-10:]
        diffs  = []
        for i in range(1, len(recent)):
            if np.any(recent[i]) and np.any(recent[i-1]):
                d = np.mean(np.abs(recent[i] - recent[i-1]))
                diffs.append(d)

        if not diffs:
            return 'unknown'

        score = np.mean(diffs)
        max_s = np.max(diffs)

        if score < low:
            vote = 'static'
        elif score > high or max_s > high * 2:
            vote = 'dynamic'
        else:
            vote = 'unknown'

        self.votes.append(vote)

        valid = [v for v in self.votes if v != 'unknown']
        if len(valid) < 5:
            return 'unknown'

        n_static  = valid.count('static')
        n_dynamic = valid.count('dynamic')
        total     = len(valid)

        if n_static  / total >= 0.7:
            return 'static'
        elif n_dynamic / total >= 0.7:
            return 'dynamic'
        else:
            return 'unknown'

# ─────────────── VIDEO PROCESSOR ───────────────
# PENTING: Processor ini TIDAK BOLEH mengakses st.session_state sama sekali,
# karena recv() dan _predict_loop() berjalan di thread lain (bukan main
# thread Streamlit) sehingga tidak punya ScriptRunContext.
# Semua "setting" disimpan sebagai atribut instance biasa, dan di-update
# dari main thread (di dalam main()) setelah widget sidebar dirender.
class BISINDOProcessor(VideoProcessorBase):
    def __init__(self):
        self.holistic = None
        self._buf_cnn    = deque(maxlen=30)
        self._buf_lstm   = deque(maxlen=30)
        self._lock       = threading.Lock()
        self._motion     = MotionDetector(window=15)

        # ── Hasil prediksi (dibaca oleh main thread) ──
        self.result_text     = ""
        self.result_mode     = ""
        self.confidence      = 0.0
        self.last_prediction = ""
        self.motion_status   = "unknown"

        # ── Setting (ditulis oleh main thread, dibaca oleh worker) ──
        # default values, akan di-override dari main() lewat sidebar
        self.ui_mode     = "🤖 Otomatis"
        self.conf_cnn    = 0.80
        self.conf_lstm   = 0.65
        self.motion_low  = 0.005
        self.motion_high = 0.015
        self.use_norm    = True
        self.blur_bg     = False
        self._settings_lock = threading.Lock()

        self._running     = True
        self._new_data    = threading.Event()
        self._pred_thread = threading.Thread(
            target=self._predict_loop, daemon=True)
        self._pred_thread.start()

    def get_holistic(self):
    """
    Lazy initialization MediaPipe Holistic.
    Dibuat hanya sekali ketika kamera mulai mengirim frame.
    """
    if self.holistic is None:
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

    return self.holistic
    
    def update_settings(self, **kwargs):
        """Dipanggil dari MAIN THREAD untuk mengoper nilai
        st.session_state ke processor dengan aman."""
        with self._settings_lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def _get_settings(self):
        with self._settings_lock:
            return (self.ui_mode, self.conf_cnn, self.conf_lstm,
                     self.motion_low, self.motion_high,
                     self.use_norm, self.blur_bg)

    def _predict_loop(self):
        COOLDOWN  = 1.2
        last_pred = 0

        while self._running:
            self._new_data.wait(timeout=0.1)
            self._new_data.clear()

            now = time.time()
            if (now - last_pred) < COOLDOWN:
                continue

            with self._lock:
                buf_cnn  = list(self._buf_cnn)
                buf_lstm = list(self._buf_lstm)

            if not buf_cnn:
                continue

            lm = buf_cnn[-1]
            if not np.any(lm):
                continue

            # Ambil setting dari atribut instance (BUKAN st.session_state)
            (ui_mode, conf_cnn, conf_lstm,
             motion_low, motion_high, use_norm, _blur_bg) = self._get_settings()

            # Tentukan mode
            if ui_mode == "🔤 Abjad (CNN)":
                mode = 'static'
            elif ui_mode == "💬 Kosakata (LSTM)":
                mode = 'dynamic'
            else:
                motion = self._motion.update(
                    buf_cnn, motion_low, motion_high)
                with self._lock:
                    self.motion_status = motion
                if motion == 'unknown':
                    continue
                mode = motion

            try:
                if mode == 'static':
                    # ── CNN ──
                    lm_input = (normalize_hand_landmarks(lm)
                                if use_norm else lm)
                    inp  = lm_input.reshape(1,-1).astype('float32')
                    pred = cnn_model.predict(inp, verbose=0)
                    idx  = int(np.argmax(pred))
                    conf = float(np.max(pred))
                    if conf > conf_cnn:
                        with self._lock:
                            self.result_text     = ABJAD[idx]
                            self.result_mode     = "ABJAD"
                            self.confidence      = conf
                            self.last_prediction = ABJAD[idx]
                        last_pred = now

                elif mode == 'dynamic' and len(buf_lstm) >= 20:
                    # ── LSTM ──
                    arr = np.array(buf_lstm, dtype='float32')
                    if len(arr) > 30:
                        arr = arr[-30:]
                    elif len(arr) < 30:
                        pad = np.zeros(
                            (30-len(arr), 258), dtype='float32')
                        arr = np.vstack([pad, arr])

                    if use_norm:
                        arr = np.array([
                            normalize_pose_landmarks(f)
                            for f in arr])

                    inp  = arr.reshape(1, 30, 258)
                    pred = lstm_model.predict(inp, verbose=0)
                    idx  = int(np.argmax(pred))
                    conf = float(np.max(pred))
                    if conf > conf_lstm:
                        with self._lock:
                            self.result_text     = KOSAKATA[idx]
                            self.result_mode     = "KOSAKATA"
                            self.confidence      = conf
                            self.last_prediction = KOSAKATA[idx]
                        last_pred = now

            except Exception as e:
                print(f"Predict error: {e}")

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        # img TIDAK di-resize — dipakai apa adanya untuk TAMPILAN,
        # mengikuti resolusi asli kamera.

        # Ambil setting dari atribut instance (BUKAN st.session_state)
        (ui_mode, _conf_cnn, _conf_lstm,
         _motion_low, _motion_high, _use_norm, blur_bg) = self._get_settings()

        # ── PENTING ──
        # MediaPipe Holistic (khususnya SegmentationSmoothingCalculator)
        # mensyaratkan ukuran frame yang KONSISTEN antar-panggilan. Kalau
        # ukuran frame kamera berubah-ubah (mis. saat negosiasi WebRTC di
        # awal koneksi), mediapipe akan crash dengan RET_CHECK failure
        # (current_mat->rows == previous_mat->rows).
        # Solusi: proses deteksi landmark di resolusi TETAP (proc_img),
        # lalu gambar hasilnya ke gambar tampilan (img) yang resolusinya
        # asli — landmark MediaPipe berupa koordinat ternormalisasi (0-1)
        # sehingga tetap presisi digambar di ukuran gambar berapa pun.
        PROC_W, PROC_H = 640, 480
        proc_img = cv2.resize(img, (PROC_W, PROC_H))

        blurred = None
        if blur_bg:
            blurred = cv2.GaussianBlur(img, (55, 55), 0)

        rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        holistic = self.get_holistic()
        try:
            with self.mp_lock:
                results = holistic.process(rgb)
        finally:
            rgb.flags.writeable = True
        rgb.flags.writeable = True

        # Terapkan blur background (mask perlu di-resize dulu ke ukuran
        # gambar tampilan, karena mask dihasilkan pada resolusi proc_img)
        if blur_bg and results.segmentation_mask is not None:
            h_disp, w_disp = img.shape[:2]
            mask = cv2.resize(results.segmentation_mask,
                               (w_disp, h_disp))
            mask = np.stack([mask]*3, axis=-1)
            img  = np.where(mask > 0.5, img,
                            blurred).astype(np.uint8)

        lm_cnn  = extract_cnn(results)
        lm_lstm = extract_lstm(results)

        with self._lock:
            self._buf_cnn.append(lm_cnn)
            self._buf_lstm.append(lm_lstm)

        self._new_data.set()

        # Gambar landmark
        spec_g = mp_draw.DrawingSpec((0,255,0), 2, 2)
        spec_w = mp_draw.DrawingSpec((255,255,255), 2)
        spec_b = mp_draw.DrawingSpec((0,200,255), 1, 1)

        if results.right_hand_landmarks:
            mp_draw.draw_landmarks(
                img, results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS, spec_g, spec_w)
        if results.left_hand_landmarks:
            mp_draw.draw_landmarks(
                img, results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS, spec_g, spec_w)
        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                img, results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS, spec_b, spec_w)

        # Overlay info
        with self._lock:
            result_text   = self.result_text
            result_mode   = self.result_mode
            confidence    = self.confidence
            motion_status = self.motion_status

        h, w = img.shape[:2]

        # Skala semua elemen overlay relatif terhadap resolusi asli
        # (referensi: lebar 1280px = skala 1.0), supaya teks/kotak
        # tetap proporsional di resolusi berapa pun (720p, 1080p, 4K, dll).
        scale     = w / 1280.0
        bar_h     = int(95 * scale)
        pad       = int(15 * scale)
        f_small   = 0.75 * scale
        f_tiny    = 0.6  * scale
        f_result  = 1.1  * scale
        f_conf    = 0.65 * scale
        th_small  = max(1, int(2 * scale))
        th_result = max(1, int(3 * scale))
        th_conf   = max(1, int(2 * scale))

        overlay = img.copy()
        cv2.rectangle(overlay, (0,0), (w,bar_h), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

        # Status motion
        motion_colors = {
            'static' : (0,255,100),
            'dynamic': (0,200,255),
            'unknown': (150,150,150)
        }
        motion_labels = {
            'static' : 'DIAM  → Abjad',
            'dynamic': 'GERAK → Kosakata',
            'unknown': 'Mendeteksi...'
        }
        mc = motion_colors.get(motion_status, (150,150,150))
        ml = motion_labels.get(motion_status, 'Mendeteksi...')
        cv2.putText(img, ml, (pad, int(28*scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, f_small, mc, th_small)

        # Mode UI
        cv2.putText(img,
                    f"Mode: {ui_mode}",
                    (pad, int(55*scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    f_tiny, (200,200,200), 1)

        # Hasil prediksi
        if result_text:
            mode_color = ((0,255,100)
                          if result_mode == 'ABJAD'
                          else (0,200,255))
            cv2.putText(img,
                        f"[{result_mode}]  {result_text}",
                        (pad, int(80*scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        f_result, mode_color, th_result)
            (conf_w, _), _ = cv2.getTextSize(
                f"Conf: {confidence*100:.1f}%",
                cv2.FONT_HERSHEY_SIMPLEX, f_conf, th_conf)
            cv2.putText(img,
                        f"Conf: {confidence*100:.1f}%",
                        (w - conf_w - pad, int(28*scale)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        f_conf, (200,200,200), th_conf)

        return VideoFrame.from_ndarray(img, format="bgr24")

    def on_ended(self):
        self._running = False
        self._new_data.set()

    def cleanup(self):
    if self.holistic is not None:
        self.holistic.close()
        self.holistic = None

# ─────────────── MAIN UI ───────────────
def main():
    cnn_model, lstm_model = get_models()
    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Pengaturan")

        st.subheader("🎮 Mode Deteksi")
        st.session_state.detection_mode = st.radio(
            "Pilih mode:",
            ["🤖 Otomatis", "🔤 Abjad (CNN)",
             "💬 Kosakata (LSTM)"],
            index=0,
            help=(
                "Otomatis: sistem deteksi sendiri\n"
                "Abjad: paksa deteksi huruf A-Z\n"
                "Kosakata: paksa deteksi kata"
            )
        )

        st.divider()
        st.subheader("🎯 Confidence Threshold")
        st.session_state.conf_cnn = st.slider(
            "Min. Confidence Abjad",
            min_value=0.50,
            max_value=1.00,
            value=0.80,
            step=0.05,
            help="Turunkan jika abjad susah terdeteksi"
        )
        st.session_state.conf_lstm = st.slider(
            "Min. Confidence Kosakata",
            min_value=0.50,
            max_value=1.00,
            value=0.65,
            step=0.05,
            help="Turunkan jika kosakata susah terdeteksi"
        )

        st.divider()
        st.subheader("🔍 Motion Sensitivity")
        st.session_state.motion_low = st.slider(
            "Batas Bawah (Diam)",
            min_value=0.001,
            max_value=0.020,
            value=0.005,
            step=0.001,
            help="Turunkan = lebih sensitif deteksi diam"
        )
        st.session_state.motion_high = st.slider(
            "Batas Atas (Gerak)",
            min_value=0.010,
            max_value=0.050,
            value=0.015,
            step=0.001,
            help="Turunkan = lebih sensitif deteksi gerak"
        )

        st.divider()
        st.subheader("🔧 Opsi Tambahan")
        st.session_state.use_norm = st.toggle(
            "Normalisasi Landmark",
            value=True,
            help="Normalize posisi tangan agar tidak "
                 "terpengaruh jarak/posisi kamera"
        )
        st.session_state.blur_bg = st.toggle(
            "Blur Background",
            value=False,
            help="Blur latar belakang untuk deteksi "
                 "lebih akurat di background ramai"
        )

        st.divider()
        st.subheader("📊 Info Model")
        st.caption(f"Abjad   : {len(ABJAD)} kelas (A-Z)")
        st.caption(f"Kosakata: {len(KOSAKATA)} kelas")
        st.caption("Input CNN  : 126 fitur")
        st.caption("Input LSTM : 258 fitur")

        st.divider()
        st.subheader("🌐 Status Koneksi")
        has_turn = any(
            "turn:" in str(s.get("urls", ""))
            for s in RTC_CONFIGURATION.get("iceServers", [])
        )
        if has_turn:
            st.success("TURN server aktif (Twilio) — "
                        "stabil untuk akses via HP/data seluler")
        else:
            st.warning("Hanya STUN — bisa gagal diakses dari "
                       "jaringan HP/data seluler. Set "
                       "TWILIO_ACCOUNT_SID & TWILIO_AUTH_TOKEN "
                       "di Secrets untuk mengaktifkan TURN.")

    # ── Main ──
    st.title("🤟 BISINDO Sign Language Translator")
    st.markdown(
        "Penerjemah **Bahasa Isyarat Indonesia (BISINDO)** "
        "— MediaPipe Holistic + CNN + LSTM"
    )
    st.divider()

    col_cam, col_info = st.columns([2, 1])

    with col_cam:
        st.subheader("📷 Kamera Real-time")

        mode_now = st.session_state.get(
            'detection_mode', '🤖 Otomatis')
        if mode_now == "🔤 Abjad (CNN)":
            st.success("Mode aktif: 🔤 Abjad — CNN")
        elif mode_now == "💬 Kosakata (LSTM)":
            st.info("Mode aktif: 💬 Kosakata — LSTM")
        else:
            st.warning("Mode aktif: 🤖 Otomatis")

        ctx = webrtc_streamer(
            key="bisindo",
            video_processor_factory=BISINDOProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    # "ideal" tinggi (bukan "min"/exact) supaya browser
                    # memilih resolusi setinggi mungkin yang didukung
                    # kamera, tapi tetap fallback otomatis ke resolusi
                    # native kamera kalau kameranya tidak sanggup 1080p.
                    "width":     {"ideal": 1920},
                    "height":    {"ideal": 1080},
                    "frameRate": {"ideal": 30}
                },
                "audio": False
            },
            async_processing=True
        )

        # ── KUNCI FIX ──
        # Oper nilai st.session_state ke processor DI SINI, di main thread,
        # setiap kali script Streamlit dijalankan ulang (setiap interaksi
        # widget / rerun). Processor tidak pernah menyentuh st.session_state
        # secara langsung, jadi tidak ada lagi warning ScriptRunContext.
        if ctx.video_processor:
            ctx.video_processor.update_settings(
                ui_mode=st.session_state.get('detection_mode', '🤖 Otomatis'),
                conf_cnn=st.session_state.get('conf_cnn', 0.80),
                conf_lstm=st.session_state.get('conf_lstm', 0.65),
                motion_low=st.session_state.get('motion_low', 0.005),
                motion_high=st.session_state.get('motion_high', 0.015),
                use_norm=st.session_state.get('use_norm', True),
                blur_bg=st.session_state.get('blur_bg', False),
            )

    with col_info:
        st.subheader("📋 Hasil Deteksi")

        # Init session state
        defaults = {
            'history'     : [],
            'current_word': '-',
            'current_mode': '-',
            'current_conf': 0.0,
            'kalimat'     : []
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

        # Update dari processor
        if ctx.video_processor:
            with ctx.video_processor._lock:
                pred = ctx.video_processor.last_prediction
                mode = ctx.video_processor.result_mode
                conf = ctx.video_processor.confidence

            if pred and pred != st.session_state.current_word:
                st.session_state.current_word = pred
                st.session_state.current_mode = mode
                st.session_state.current_conf = conf
                st.session_state.history.append({
                    'kata': pred,
                    'mode': mode,
                    'conf': conf,
                    'time': time.strftime('%H:%M:%S')
                })
                st.session_state.kalimat.append(pred)

        # Tampilkan hasil
        wcolor = ("green"
                  if st.session_state.current_mode == "ABJAD"
                  else "blue")
        st.markdown(
            f"<h1 style='color:{wcolor}; text-align:center;'>"
            f"{st.session_state.current_word}</h1>",
            unsafe_allow_html=True
        )
        if st.session_state.current_conf > 0:
            st.progress(st.session_state.current_conf)
            st.caption(
                f"Mode: {st.session_state.current_mode} | "
                f"Conf: {st.session_state.current_conf*100:.1f}%"
            )

        st.divider()

        # Kalimat
        st.markdown("**📝 Kalimat:**")
        kalimat_str = (" ".join(st.session_state.kalimat)
                       if st.session_state.kalimat else "-")
        st.info(kalimat_str)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Hapus Terakhir",
                          use_container_width=True):
                if st.session_state.kalimat:
                    st.session_state.kalimat.pop()
                    st.rerun()
        with c2:
            if st.button("🔄 Reset Semua",
                          use_container_width=True):
                st.session_state.kalimat      = []
                st.session_state.history      = []
                st.session_state.current_word = '-'
                st.session_state.current_mode = '-'
                st.session_state.current_conf = 0.0
                st.rerun()

        st.divider()

        # Riwayat
        st.markdown("**🕐 Riwayat:**")
        if st.session_state.history:
            for item in reversed(
                    st.session_state.history[-10:]):
                badge = ("🟢" if item['mode'] == 'ABJAD'
                         else "🔵")
                st.caption(
                    f"{badge} [{item['time']}] "
                    f"**{item['kata']}** "
                    f"({item['conf']*100:.1f}%)"
                )
        else:
            st.caption("Belum ada deteksi...")

    st.divider()
    with st.expander("ℹ️ Cara Penggunaan"):
        st.markdown(f"""
        **Mode Otomatis:**
        - Tahan tangan **diam** → deteksi Abjad 🟢
        - Lakukan **gerakan** → deteksi Kosakata 🔵

        **Mode Manual (Sidebar):**
        - Pilih **Abjad** → selalu deteksi huruf A-Z
        - Pilih **Kosakata** → selalu deteksi kata

        **Tips akurasi:**
        - Gunakan **mode manual** untuk hasil lebih akurat
        - Latar belakang **polos** lebih baik
        - Aktifkan **Blur Background** jika latar ramai
        - Aktifkan **Normalisasi** jika jarak kamera berbeda
        - Turunkan threshold jika sulit terdeteksi

        **Kosakata yang didukung ({len(KOSAKATA)} kata):**
        {', '.join(KOSAKATA)}
        """)

if __name__ == '__main__':
    self.mp_lock = Lock()
    main()
  
