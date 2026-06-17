import streamlit as st
import cv2
import torch
import numpy as np
import time
import os
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, VideoHTMLAttributes

from model import SignLSTM
from mediapipe_extractor import MediaPipeFeatureExtractor

# --- EMOTION MODULE INTEGRATION ---
try:
    import calculate_emotion
except ImportError:
    calculate_emotion = None

# --- AUDIO SPEECH VIA BROWSER JS SPEECH SYNTHESIS ---
def speak_web(text):
    # Injects client-side JS into the parent window to speak natively in user's speakers
    js_code = f"""
    <script>
        var synth = window.parent.speechSynthesis || window.speechSynthesis;
        if (synth) {{
            var utterance = new SpeechSynthesisUtterance("{text}");
            synth.speak(utterance);
        }}
    </script>
    """
    st.components.v1.html(js_code, height=0, width=0)

# --- MAKE SCREEN BIG & CUSTOM STYLING (CSS) ---
st.set_page_config(page_title="ISL Cloud Translator", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0b0c10;
        color: #c5c6c7;
    }
    .stMainBlockContainer {max-width: 100% !important; padding: 1rem !important;}
    
    /* Modern Dashboard Cards */
    .dashboard-card {
        background: linear-gradient(135deg, #1f2833 0%, #151a21 100%);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        margin-bottom: 12px;
        transition: border-color 0.3s ease;
    }
    .dashboard-card:hover {
        border-color: #66fcf1;
    }
    .card-title {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #8b9bb4;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .card-value {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
    }
    .val-active { color: #66fcf1; }
    .val-warning { color: #ffeb3b; }
    .val-error { color: #ff5252; }
    
    /* Input border customization */
    .stTextInput input {
        background-color: #1f2833 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid #45f3ff !important;
    }

    /* WebRTC Video Element Customizations */
    div[data-testid="stWebRTCStreamer"] video {
        pointer-events: none !important; /* Disables double click/tapping controls */
    }
    div[data-testid="stWebRTCStreamer"] select {
        display: none !important; /* Hides camera select dropdown */
    }
    div[data-testid="stWebRTCStreamer"] button {
        background-color: #1f2833 !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        font-family: 'Outfit', sans-serif !important;
    }
    div[data-testid="stWebRTCStreamer"] button:hover {
        border-color: #66fcf1 !important;
        color: #66fcf1 !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🤟 ISL Cloud Multimodal Translator")

# --- INITIALIZE STATE ---
if 'current_word' not in st.session_state:
    st.session_state.current_word = ""
if 'incoming_message' not in st.session_state:
    st.session_state.incoming_message = ""
if 'type_input_buffer' not in st.session_state:
    st.session_state.type_input_buffer = ""
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = "Neutral"

# Animation State Variables
if 'animation_queue' not in st.session_state:
    st.session_state.animation_queue = []
if 'animation_char' not in st.session_state:
    st.session_state.animation_char = ""
if 'animation_time' not in st.session_state:
    st.session_state.animation_time = 0.0

@st.cache_resource
def load_assets():
    m = SignLSTM(num_classes=25)
    m.load_state_dict(torch.load("isl_model.pth", map_location="cpu"))
    m.eval()
    return m, MediaPipeFeatureExtractor()

model, extractor = load_assets()

# Use same labels list as webcam_sign_to_sound.py
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
          'N', 'O', 'P', 'Q', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Video Processor for WebRTC Thread
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.extractor = extractor
        self.labels = labels
        
        self.current_word = ""
        self.last_added_letter = ""
        self.last_addition_time = 0.0
        self.stability_buffer = []
        self.sequence = []
        self.current_emotion = "Neutral"
        self.new_letters = []  # Queue of newly confirmed letters
        self.animation_char = ""  # Sync from main thread

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h_h, w_h, _ = img.shape
        
        # Create a black screen canvas matching input frame size
        black_screen = np.zeros((h_h, w_h, 3), dtype=np.uint8)
        
        # Calculate emotion
        if calculate_emotion is not None:
            try:
                self.current_emotion = calculate_emotion.calculate_emotion(img)
            except Exception:
                self.current_emotion = "Neutral"

        features = self.extractor.extract_features(img)
        
        # Draw hand landmarks on black canvas
        if self.extractor.hands_result and self.extractor.hands_result.hand_landmarks:
            for hand_landmarks in self.extractor.hands_result.hand_landmarks:
                points = []
                for lm in hand_landmarks:
                    cx = int(lm.x * w_h)
                    cy = int(lm.y * h_h)
                    points.append((cx, cy))
                    cv2.circle(black_screen, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.circle(black_screen, (cx, cy), 3, (255, 255, 255), -1)
                
                connections = [
                    (0,1), (1,2), (2,3), (3,4),       # Thumb
                    (0,5), (5,6), (6,7), (7,8),       # Index
                    (5,9), (9,10), (10,11), (11,12),  # Middle
                    (9,13), (13,14), (14,15), (15,16),# Ring
                    (13,17), (0,17), (17,18), (18,19), (19,20) # Pinky
                ]
                for start_idx, end_idx in connections:
                    if start_idx < len(points) and end_idx < len(points):
                        cv2.line(black_screen, points[start_idx], points[end_idx], (0, 255, 0), 2)
                        
            self.sequence.append(features)
            self.sequence = self.sequence[-30:]
            
            if len(self.sequence) == 30:
                with torch.no_grad():
                    res = self.model(torch.from_numpy(np.array([self.sequence])).float())
                    prob = torch.nn.functional.softmax(res, dim=1)
                    max_prob, idx = torch.max(prob, dim=1)
                    
                    prediction = self.labels[idx.item()]
                    confidence = max_prob.item()
                    self.stability_buffer.append(prediction)
                    self.stability_buffer = self.stability_buffer[-5:]
                    
                    if confidence >= 0.80 and self.stability_buffer.count(prediction) >= 3:
                        current_time = time.time()
                        if prediction != self.last_added_letter or (current_time - self.last_addition_time > 1.5):
                            self.current_word += prediction
                            self.last_added_letter = prediction
                            self.last_addition_time = current_time
                            self.new_letters.append(prediction)
        else:
            self.sequence = []
            self.stability_buffer = []
            self.last_added_letter = ""

        # Draw dynamic reference image on the black screen canvas
        target_letter = self.animation_char
        if target_letter and target_letter.isalpha():
            img_path = os.path.join(".", f"{target_letter}.png")
            if os.path.exists(img_path):
                ref_img = cv2.imread(img_path)
                if ref_img is not None:
                    # Resize to fit proportionally in the corner
                    ref_size = min(int(w_h * 0.35), int(h_h * 0.35))
                    ref_img_resized = cv2.resize(ref_img, (ref_size, ref_size))
                    # Safely place it in the top right corner
                    if h_h >= ref_size + 20 and w_h >= ref_size + 20:
                        black_screen[10:10+ref_size, w_h - ref_size - 10:w_h - 10] = ref_img_resized

        return av.VideoFrame.from_ndarray(black_screen, format="bgr24")

# Layout setup: 2 columns
col1, col2 = st.columns([2.5, 1.2])

with col1:
    # WebRTC Streamer element
    ctx = webrtc_streamer(
        key="isl-streamer",
        video_processor_factory=SignLanguageProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        video_html_attrs=VideoHTMLAttributes(
            autoPlay=True,
            controls=False,      # Disables HTML5 native play/pause/fullscreen controls overlay
            playsInline=True,
            muted=True,
            style={"width": "100%", "height": "auto", "border-radius": "12px", "pointer-events": "none"}
        )
    )
    
    # 3 Status cards side-by-side
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        status_card = st.empty()
    with col_s2:
        word_card = st.empty()
    with col_s3:
        mood_card = st.empty()
        
    # Conversation bubble panel
    chat_card = st.empty()

with col2:
    st.markdown('<div class="dashboard-card"><h3 style="margin:0 0 10px 0; color:#66fcf1; font-family:\'Outfit\';">Controls & Chat</h3></div>', unsafe_allow_html=True)
    
    # Text Input for typing reply
    def handle_submit():
        val = st.session_state.chat_input_val.strip().upper()
        if val:
            clean_val = "".join([c for c in val if c.isalpha() or c.isspace()])
            if clean_val:
                st.session_state.incoming_message = clean_val
                st.session_state.animation_queue = [c for c in clean_val if c.isalpha()]
                st.session_state.type_input_buffer = ""
                st.session_state.animation_char = ""
                st.session_state.animation_time = 0.0
            st.session_state.chat_input_val = ""

    st.text_input("Type Reply & press Enter:", key="chat_input_val", on_change=handle_submit)
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔊 Speak signs", use_container_width=True):
            if st.session_state.current_word:
                speak_web(st.session_state.current_word.lower())
    with col_btn2:
        if st.button("🧹 Clear word", use_container_width=True):
            st.session_state.current_word = ""
            if ctx.video_processor:
                ctx.video_processor.current_word = ""
                ctx.video_processor.last_added_letter = ""
    
    # --- SIDE-BY-SIDE GALLERY OF SPELLED LETTERS ---
    if st.session_state.incoming_message:
        st.write("---")
        st.write("**Reply Spelled in Signs:**")
        letters = [c for c in st.session_state.incoming_message if c.isalpha()]
        if letters:
            gallery_cols = st.columns(4)
            for idx, char in enumerate(letters):
                col_idx = idx % 4
                img_path = os.path.join(".", f"{char}.png")
                if os.path.exists(img_path):
                    gallery_cols[col_idx].image(img_path, caption=char, use_container_width=True)

# Main UI Update Loop running on Streamlit main thread while streaming
if ctx.state.playing:
    while ctx.state.playing:
        if ctx.video_processor:
            # Sync variables from background WebRTC thread
            st.session_state.current_word = ctx.video_processor.current_word
            st.session_state.current_emotion = ctx.video_processor.current_emotion
            # Sync animation state to background WebRTC thread
            ctx.video_processor.animation_char = st.session_state.animation_char
            
            # Read and speak any new confirmed letters
            if ctx.video_processor.new_letters:
                new_letter = ctx.video_processor.new_letters.pop(0)
                speak_web(new_letter)
            
            # Tracking status text
            if ctx.video_processor.extractor.hands_result and ctx.video_processor.extractor.hands_result.hand_landmarks:
                if len(ctx.video_processor.sequence) == 30:
                    status_text = f"Tracking Active"
                    status_class = "val-active"
                else:
                    status_text = "Detecting..."
                    status_class = "val-warning"
            else:
                status_text = "Searching for Hands..."
                status_class = "val-error"
        else:
            status_text = "Connecting Camera..."
            status_class = "val-warning"

        # --- ANIMATION CONTROLLER FOR TYPED REPLY ---
        if st.session_state.animation_queue or st.session_state.animation_char:
            now = time.time()
            if not st.session_state.animation_char or (now - st.session_state.animation_time > 1.0):
                if st.session_state.animation_queue:
                    next_char = st.session_state.animation_queue.pop(0)
                    st.session_state.animation_char = next_char
                    st.session_state.animation_time = now
                    st.session_state.type_input_buffer += next_char
                else:
                    if now - st.session_state.animation_time > 1.0:
                        st.session_state.animation_char = ""
                        st.session_state.type_input_buffer = st.session_state.incoming_message

        # --- UPDATE UI CARDS (HTML) ---
        status_card.markdown(f"""<div class="dashboard-card">
<div class="card-title">Tracking Status</div>
<div class="card-value {status_class}">{status_text}</div>
</div>""", unsafe_allow_html=True)
        
        word_card.markdown(f"""<div class="dashboard-card">
<div class="card-title">Your Signs Word</div>
<div class="card-value val-active">{st.session_state.current_word or "..."}</div>
</div>""", unsafe_allow_html=True)
        
        mood_emojis = {
            "Happy": "😄 Happy",
            "Sad": "😢 Sad",
            "Angry": "😠 Angry",
            "Surprise": "😲 Surprise",
            "Neutral": "😐 Neutral"
        }
        mood_display = mood_emojis.get(st.session_state.current_emotion, "😐 Neutral")
        mood_card.markdown(f"""<div class="dashboard-card">
<div class="card-title">Detected Mood</div>
<div class="card-value" style="color: #ff70a6;">{mood_display}</div>
</div>""", unsafe_allow_html=True)
        
        typing_indicator = ""
        if st.session_state.animation_char:
            typing_indicator = f"""<div style="align-self: flex-end; color: #8b9bb4; font-size: 13px; font-style: italic; margin-top: 5px;">
Spelling sign: <span style="color: #66fcf1; font-weight: bold; font-size: 15px;">{st.session_state.animation_char}</span> (spelled: "{st.session_state.type_input_buffer}...")
</div>"""
            
        chat_card.markdown(f"""<div class="dashboard-card">
<div class="card-title">Chat Conversation</div>
<div style="display: flex; flex-direction: column; gap: 12px; margin-top: 8px;">
<div style="display: flex; justify-content: space-between; align-items: center; background: rgba(102, 252, 241, 0.08); border-left: 4px solid #66fcf1; padding: 12px 16px; border-radius: 4px 12px 12px 4px; width: 100%;">
<span style="font-size: 9px; color: #8b9bb4; font-weight: bold; letter-spacing: 0.5px;">YOUR SIGNS (SENT)</span>
<span style="font-size: 16px; color: #ffffff; font-weight: 500; letter-spacing: 0.5px;">{st.session_state.current_word or "..."}</span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; background: rgba(255, 112, 166, 0.08); border-left: 4px solid #ff70a6; padding: 12px 16px; border-radius: 4px 12px 12px 4px; width: 100%;">
<span style="font-size: 9px; color: #8b9bb4; font-weight: bold; letter-spacing: 0.5px;">REPLY RECEIVED</span>
<span style="font-size: 16px; color: #ffffff; font-weight: 500; letter-spacing: 0.5px;">{st.session_state.incoming_message or "Waiting for reply..."}</span>
</div>
{typing_indicator}
</div>
</div>""", unsafe_allow_html=True)
        
        # Control refresh rate
        time.sleep(0.1)
else:
    # If not playing, show instructions / initial state
    status_card.markdown("""<div class="dashboard-card">
<div class="card-title">Tracking Status</div>
<div class="card-value val-error">Camera Offline</div>
</div>""", unsafe_allow_html=True)
    
    word_card.markdown("""<div class="dashboard-card">
<div class="card-title">Your Signs Word</div>
<div class="card-value">...</div>
</div>""", unsafe_allow_html=True)
    
    mood_card.markdown("""<div class="dashboard-card">
<div class="card-title">Detected Mood</div>
<div class="card-value">😐 Neutral</div>
</div>""", unsafe_allow_html=True)

    chat_card.markdown("""<div class="dashboard-card">
<div class="card-title">Chat Conversation</div>
<div style="display: flex; flex-direction: column; gap: 12px; margin-top: 8px; font-style: italic; color: #8b9bb4;">
Start the camera stream above to begin translating sign language.
</div>
</div>""", unsafe_allow_html=True)
