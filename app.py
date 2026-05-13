import streamlit as st
from streamlit_webrtc import webrtc_streamer
from streamlit_autorefresh import st_autorefresh
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import spotipy
from spotipy.oauth2 import SpotifyOAuth


SPOTIFY_SCOPES = "user-modify-playback-state user-read-playback-state"


def get_spotify_oauth():
    return SpotifyOAuth(
        client_id=st.secrets["spotify"]["client_id"],
        client_secret=st.secrets["spotify"]["client_secret"],
        redirect_uri=st.secrets["spotify"]["redirect_uri"],
        scope=SPOTIFY_SCOPES,
        cache_path=".cache",
        open_browser=False,
    )


def get_spotify_client():
    try:
        oauth = get_spotify_oauth()
    except (KeyError, FileNotFoundError):
        return None, None

    try:
        token_info = oauth.get_cached_token()
    except Exception:
        return None, oauth

    if not token_info:
        code = st.query_params.get("code")
        if code:
            try:
                token_info = oauth.get_access_token(code, as_dict=True, check_cache=False)
                st.query_params.clear()
            except Exception:
                token_info = None

    if not token_info:
        return None, oauth

    return spotipy.Spotify(auth_manager=oauth), oauth


st.set_page_config(
    page_title="etunes",
    layout="centered",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_assets():
    return load_model("model.h5"), np.load("labels.npy")


@st.cache_resource
def load_holistic():
    return mp.solutions.holistic.Holistic()


model, label = load_assets()
holis = load_holistic()
holistic = mp.solutions.holistic
hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils


MOOD_COLORS = {
    "happy":     ("#fef3c7", "#92400e"),
    "sad":       ("#dbeafe", "#1e40af"),
    "angry":     ("#fee2e2", "#991b1b"),
    "neutral":   ("#e7e5e4", "#44403c"),
    "surprise":  ("#fce7f3", "#9d174d"),
    "rock":      ("#ede9fe", "#5b21b6"),
    "melancholy":("#dbeafe", "#1e40af"),
    "calm":      ("#d1fae5", "#065f46"),
    "energetic": ("#ffedd5", "#9a3412"),
    "romantic":  ("#fce7f3", "#9d174d"),
    "focused":   ("#ede9fe", "#5b21b6"),
}


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,400;0,500;0,600;1,400;1,500&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, system-ui, sans-serif;
    color: #2a2a2a;
}

.stApp {
    background:
        radial-gradient(at 8% 8%,  rgba(196, 181, 245, 0.55) 0%, transparent 35%),
        radial-gradient(at 92% 6%, rgba(255, 213, 184, 0.55) 0%, transparent 35%),
        radial-gradient(at 50% 45%, rgba(255, 230, 245, 0.35) 0%, transparent 55%),
        radial-gradient(at 12% 92%, rgba(255, 200, 220, 0.50) 0%, transparent 35%),
        radial-gradient(at 92% 92%, rgba(196, 230, 210, 0.50) 0%, transparent 35%),
        #fdfaf5;
    background-attachment: fixed;
}

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 5rem;
    max-width: 720px;
}

.wordmark {
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 500;
    font-size: 1.1rem;
    color: #2a2a2a;
    letter-spacing: -0.01em;
    margin-bottom: 2.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.wordmark .dot {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: linear-gradient(135deg, #a574f0, #c97cf2);
    display: inline-block;
}

.badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(0, 0, 0, 0.05);
    border-radius: 999px;
    padding: 0.4rem 0.95rem;
    font-size: 0.78rem;
    color: #5a5a5a;
    margin-bottom: 1.25rem;
    font-weight: 400;
}

.hero {
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 500;
    font-size: 3.2rem;
    letter-spacing: -0.03em;
    line-height: 1.05;
    color: #1f1f1f;
    margin: 0 0 1.25rem 0;
    text-align: center;
}

.hero em {
    font-style: italic;
    font-weight: 400;
    color: #9b6fe8;
}

.subtitle {
    font-size: 1rem;
    color: #6e6e6e;
    line-height: 1.55;
    text-align: center;
    max-width: 460px;
    margin: 0 auto 2.5rem auto;
}

.section-label {
    font-size: 0.7rem;
    color: #9b6fe8;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-weight: 600;
    text-align: center;
    margin: 2.5rem 0 0.75rem 0;
}

.section-title {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.8rem;
    font-weight: 500;
    text-align: center;
    color: #1f1f1f;
    margin: 0 0 2rem 0;
    letter-spacing: -0.02em;
}

[data-testid="stIFrame"] {
    display: flex !important;
    justify-content: center !important;
    margin: 1rem 0 0.5rem 0;
}

[data-testid="stIFrame"] iframe,
section.main iframe[title*="webrtc"],
section.main iframe[title*="streamlit"] {
    width: 480px !important;
    height: 460px !important;
    max-width: 90vw !important;
    box-shadow:
        0 25px 60px -15px rgba(155, 111, 232, 0.35),
        0 0 0 1px rgba(255, 255, 255, 0.6);
    border-radius: 16px !important;
    background: linear-gradient(135deg, #e9defc 0%, #fce4d6 100%);
}

.cam-hint {
    text-align: center;
    color: #8a8a8a;
    font-size: 0.82rem;
    margin: 0.75rem 0 2rem 0;
}

.stTextInput > div > div > input {
    border: 1px solid rgba(0, 0, 0, 0.08);
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(6px);
    border-radius: 10px;
    padding: 0.7rem 0.95rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: #1f1f1f;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}

.stTextInput > div > div > input:focus {
    border-color: #9b6fe8;
    box-shadow: 0 0 0 3px rgba(155, 111, 232, 0.15);
}

.stTextInput label {
    font-size: 0.72rem !important;
    color: #6e6e6e !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase;
}

.stButton > button {
    background: linear-gradient(135deg, #9b6fe8 0%, #b07ef0 100%);
    color: #ffffff;
    border: none;
    border-radius: 999px;
    padding: 0.85rem 1.6rem;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 0.95rem;
    letter-spacing: 0.01em;
    width: 100%;
    transition: all 0.2s ease;
    box-shadow: 0 8px 20px -6px rgba(155, 111, 232, 0.45);
    margin-top: 1rem;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 28px -6px rgba(155, 111, 232, 0.55);
    background: linear-gradient(135deg, #8a5ee0 0%, #a06ee8 100%);
    color: #ffffff;
}

.stButton > button:focus {
    box-shadow: 0 8px 20px -6px rgba(155, 111, 232, 0.45);
    outline: none;
}

.mood-card {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(0, 0, 0, 0.05);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    margin: 1.5rem 0 0.5rem 0;
    text-align: center;
}

.mood-card .label {
    font-size: 0.7rem;
    color: #9a9a9a;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 500;
    margin-bottom: 0.5rem;
}

.mood-pill {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.15rem;
    font-weight: 500;
    text-transform: lowercase;
    letter-spacing: -0.01em;
}

.mood-pill.pending {
    background: #f0ede7;
    color: #b0aca5;
    font-style: italic;
}

.result {
    margin-top: 1.25rem;
    padding: 1.25rem 1.5rem;
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(155, 111, 232, 0.2);
    border-radius: 16px;
    text-align: center;
}

.result .query {
    color: #5a5a5a;
    font-size: 0.9rem;
    margin-bottom: 0.6rem;
}

.result .query strong {
    color: #1f1f1f;
    font-weight: 500;
}

.result a {
    display: inline-block;
    color: #9b6fe8;
    text-decoration: none;
    font-weight: 500;
    font-size: 0.95rem;
    border-bottom: 1px solid #9b6fe8;
    padding-bottom: 1px;
}

.result a:hover {
    color: #7c4ed4;
    border-bottom-color: #7c4ed4;
}

.stAlert {
    background: rgba(255, 255, 255, 0.85) !important;
    border: 1px solid rgba(155, 111, 232, 0.2) !important;
    border-left: 3px solid #9b6fe8 !important;
    border-radius: 10px !important;
    color: #1f1f1f !important;
    font-size: 0.9rem !important;
}

.auth-status {
    text-align: center;
    color: #6e6e6e;
    font-size: 0.82rem;
    margin: -1.75rem 0 1.5rem 0;
}

.auth-status strong {
    color: #1f1f1f;
    font-weight: 500;
}

.auth-status .dotg {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #1db954;
    margin-right: 0.4rem;
    vertical-align: middle;
}

.spotify-now {
    display: flex;
    align-items: center;
    gap: 1rem;
    text-align: left;
}

.album-art {
    width: 72px;
    height: 72px;
    border-radius: 8px;
    object-fit: cover;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.now-meta {
    flex: 1;
    min-width: 0;
}

.now-label {
    font-size: 0.7rem;
    color: #9b6fe8;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
    margin-bottom: 0.25rem;
}

.now-track {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.15rem;
    font-weight: 500;
    color: #1f1f1f;
    line-height: 1.2;
    margin-bottom: 0.15rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.now-artist {
    font-size: 0.88rem;
    color: #6e6e6e;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.footer {
    text-align: center;
    color: #a0a0a0;
    font-size: 0.78rem;
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(0, 0, 0, 0.06);
}

.footer .brand {
    font-family: 'Fraunces', Georgia, serif;
    color: #6e6e6e;
}
</style>
""",
    unsafe_allow_html=True,
)


if "emotion" not in st.session_state:
    try:
        st.session_state["emotion"] = str(np.load("emotion.npy")[0])
    except (FileNotFoundError, IndexError):
        st.session_state["emotion"] = ""


st.markdown(
    '<div class="wordmark"><span class="dot"></span>etunes</div>',
    unsafe_allow_html=True,
)

sp, oauth = get_spotify_client()
if sp is not None and "spotify_display_name" not in st.session_state:
    try:
        me = sp.current_user()
        st.session_state["spotify_display_name"] = me.get("display_name") or me.get("id")
    except Exception:
        st.session_state["spotify_display_name"] = None

display_name = st.session_state.get("spotify_display_name")
if sp is not None and display_name:
    st.markdown(
        f'<div class="auth-status"><span class="dotg"></span>Connected to Spotify as <strong>{display_name}</strong></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div style="text-align:center;"><span class="badge">Music that meets you where you are</span></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<h1 class="hero">Songs for <em>every</em> shade of feeling.</h1>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="subtitle">etunes reads your expression in real time and hands you a song tuned to your mood — opened straight on YouTube.</p>',
    unsafe_allow_html=True,
)


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

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            lst = np.array(lst).reshape(1, -1)
            pred = str(label[np.argmax(model.predict(lst, verbose=0))])
            np.save("emotion.npy", np.array([pred]))

        drawing.draw_landmarks(
            frm,
            res.face_landmarks,
            holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing.DrawingSpec(color=(255, 255, 255), thickness=-1, circle_radius=1),
            connection_drawing_spec=drawing.DrawingSpec(color=(200, 180, 240), thickness=1),
        )
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


st.markdown('<div class="section-label">Live demo</div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Let your face choose the song</h2>', unsafe_allow_html=True)

webrtc_streamer(
    key="etunes-cam",
    desired_playing_state=True,
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    },
    async_processing=True,
)

st.markdown(
    '<p class="cam-hint">Your camera stays on your device. Nothing is uploaded.</p>',
    unsafe_allow_html=True,
)


if not st.session_state["emotion"]:
    st_autorefresh(interval=800, key="mood_poll", limit=150)

try:
    latest = str(np.load("emotion.npy")[0])
    if latest:
        st.session_state["emotion"] = latest
except (FileNotFoundError, IndexError):
    pass


current = st.session_state["emotion"].strip().lower()
if current:
    bg, fg = MOOD_COLORS.get(current, ("#ede9fe", "#5b21b6"))
    st.markdown(
        f'<div class="mood-card">'
        f'<div class="label">Detected mood</div>'
        f'<span class="mood-pill" style="background:{bg};color:{fg};">{current}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="mood-card">'
        '<div class="label">Detected mood</div>'
        '<span class="mood-pill pending">awaiting capture</span>'
        '</div>',
        unsafe_allow_html=True,
    )


col1, col2 = st.columns(2)
with col1:
    lang = st.text_input("Language", placeholder="English, Hindi, Tamil…")
with col2:
    singer = st.text_input("Artist", placeholder="Arijit Singh, Adele…")


selected_device = None
if sp is not None:
    try:
        device_list = sp.devices().get("devices", [])
    except Exception:
        device_list = []

    if device_list:
        default_idx = next(
            (i for i, d in enumerate(device_list)
             if d["type"].lower() == "smartphone" or "phone" in d["name"].lower()),
            0,
        )
        selected_idx = st.selectbox(
            "Play on",
            range(len(device_list)),
            format_func=lambda i: f"{device_list[i]['name']} · {device_list[i]['type'].lower()}",
            index=default_idx,
        )
        selected_device = device_list[selected_idx]
    else:
        st.caption("No active Spotify device. Open Spotify on your phone and play any track for 2 seconds, then refresh this page.")


if st.button("Read my mood and recommend"):
    if not st.session_state["emotion"]:
        st.warning("No mood captured yet. Hold the camera for a moment until your mood appears above.")
    elif not lang or not singer:
        st.warning("Please add both a language and an artist to tune the search.")
    elif sp is None:
        if oauth is None:
            st.error("Spotify credentials are missing. Open `.streamlit/secrets.toml` and paste your Client ID and Secret from the Spotify Developer Dashboard.")
        else:
            auth_url = oauth.get_authorize_url()
            st.markdown(
                f'<div class="result">'
                f'<div class="query">First, connect your Spotify account.</div>'
                f'<a href="{auth_url}" target="_self">Log in with Spotify</a>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        emotion = st.session_state["emotion"]
        query = f"{lang} {emotion} {singer}"
        try:
            results = sp.search(q=query, type="track", limit=5)
            tracks = results.get("tracks", {}).get("items", [])
            if not tracks:
                st.warning(f"No tracks found for {query}. Try a different artist or language.")
            elif selected_device is None:
                st.warning("No active Spotify device. Open Spotify on your phone and play any track for 2 seconds so it registers, then refresh this page.")
            else:
                track = tracks[0]
                sp.start_playback(device_id=selected_device["id"], uris=[track["uri"]])
                art = track["album"]["images"][0]["url"] if track["album"].get("images") else ""
                artists = ", ".join(a["name"] for a in track["artists"])
                st.markdown(
                    f'<div class="result spotify-now">'
                    f'<img src="{art}" alt="" class="album-art">'
                    f'<div class="now-meta">'
                    f'<div class="now-label">Now playing on {selected_device["name"]}</div>'
                    f'<div class="now-track">{track["name"]}</div>'
                    f'<div class="now-artist">{artists}</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                np.save("emotion.npy", np.array([""]))
                st.session_state["emotion"] = ""
        except spotipy.SpotifyException as e:
            if e.http_status == 403:
                st.error("Spotify Premium is required to control playback. Upgrade your account or use the listening account that has Premium.")
            elif e.http_status == 404:
                st.error("Couldn't find an active Spotify device. Open Spotify and play any track briefly so the device registers, then try again.")
            else:
                st.error(f"Spotify error: {e.msg}")


st.markdown(
    '<div class="footer">© 2026 <span class="brand">etunes</span> — Music tuned to feeling.</div>',
    unsafe_allow_html=True,
)
