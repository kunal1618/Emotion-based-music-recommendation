import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

# ---------------------------
# Load Model and Labels
# ---------------------------
model = load_model("model.h5")
label = np.load("labels.npy")

holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# ---------------------------
# Streamlit App Header
# ---------------------------
st.title("🎵 Emotion Based Music Recommender")

# ---------------------------
# Initialize session state
# ---------------------------
if "run" not in st.session_state:
    st.session_state["run"] = True  # True = webcam runs
if "emotion" not in st.session_state:
    st.session_state["emotion"] = ""

# Load last emotion if exists
try:
    st.session_state["emotion"] = np.load("emotion.npy")[0]
except:
    st.session_state["emotion"] = ""

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("🎧 Music Preferences")
lang = st.sidebar.text_input("Language")
singer = st.sidebar.text_input("Singer")
st.sidebar.markdown("**Instructions:** Capture your emotion using the webcam, then click 'Recommend me songs'.")

# ---------------------------
# Emotion Processor
# ---------------------------
class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            # Left hand
            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0]*42)

            # Right hand
            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0]*42)

            lst = np.array(lst).reshape(1, -1)
            pred = label[np.argmax(model.predict(lst))]

            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
            st.session_state["emotion"] = pred
            np.save("emotion.npy", np.array([pred]))

        # Draw landmarks
        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
                               connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# ---------------------------
# Display Webcam Feed
# ---------------------------
if lang and singer and st.session_state["run"]:
    st.subheader("📷 Live Emotion Capture")
    webrtc_streamer(
        key="key",
        desired_playing_state=True,
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )

# ---------------------------
# Display Detected Emotion & Emoji
# ---------------------------
emotion_colors = {
    "Happy": "#FFD700",
    "Sad": "#1E90FF",
    "Angry": "#FF4500",
    "Neutral": "#90EE90"
}

emotion_emoji = {
    "Happy": "😄",
    "Sad": "😢",
    "Angry": "😠",
    "Neutral": "😐"
}

if st.session_state["emotion"]:
    st.markdown(
        f"<div style='background-color: {emotion_colors.get(st.session_state['emotion'],'#FFFFFF')}; "
        f"padding:20px; border-radius:10px; text-align:center; font-size:24px; color:black;'>"
        f"Detected Emotion: {st.session_state['emotion']} {emotion_emoji.get(st.session_state['emotion'],'')}</div>",
        unsafe_allow_html=True
    )

# ---------------------------
# Recommend Button
# ---------------------------
btn = st.button("🎧 Recommend me songs", help="Click after your emotion is captured")

if btn:
    if not st.session_state["emotion"]:
        st.warning("⚠️ Please let me capture your emotion first")
        st.session_state["run"] = True
    else:
        query = f"{lang} {st.session_state['emotion']} song {singer}"
        st.success(f"🔎 Searching on YouTube for: **{query}**")
        webbrowser.open(f"https://www.youtube.com/results?search_query={query}")

        # Reset emotion and restart webcam automatically
        np.save("emotion.npy", np.array([""]))
        st.session_state["emotion"] = ""
        st.session_state["run"] = True
        # No rerun needed! Streamlit will refresh components automatically