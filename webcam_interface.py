import streamlit as st
import cv2
import torch
import numpy as np
import pyttsx3
import threading
import time
import os

from model import SignLSTM
from mediapipe_extractor import MediaPipeFeatureExtractor

# --- EMOTION MODULE INTEGRATION ---
try:
    import calculate_emotion
except ImportError:
    calculate_emotion = None

# --- AUDIO THREADING ---
def speak(text):
    def run_speech():
        try:
            engine = pyttsx3.init('sapi5')
            engine.say(text)
            engine.runAndWait()
            del engine
        except Exception:
            try:
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
                del engine
            except Exception:
                pass
    threading.Thread(target=run_speech, daemon=True).start()

# --- MAKE SCREEN BIG & CUSTOM STYLING (CSS) ---
st.set_page_config(page_title="ISL Neural Translator", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0b0c10;
        color: #c5c6c7;
    }
    .stMainBlockContainer {max-width: 100% !important; padding: 1rem !important;}
    div[data-testid="stImage"] img { 
        width: 100% !important; 
        height: auto !important; 
        border-radius: 12px;
        border: 2px solid #1f2833;
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    }
    
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
    </style>
    """, unsafe_allow_html=True)

st.title("🤟 ISL Neural Multimodal Translator")

# --- INITIALIZE STATE ---
if 'current_word' not in st.session_state:
    st.session_state.current_word = ""
if 'last_added_letter' not in st.session_state:
    st.session_state.last_added_letter = ""
if 'last_addition_time' not in st.session_state:
    st.session_state.last_addition_time = 0.0
if 'stability_buffer' not in st.session_state:
    st.session_state.stability_buffer = []
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

# Layout setup: 2 columns
col1, col2 = st.columns([2.5, 1.2])

with col1:
    frame_placeholder = st.empty()
    
    # 3 Status cards side-by-side below the camera feed
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
                speak(st.session_state.current_word.lower())
    with col_btn2:
        if st.button("🧹 Clear word", use_container_width=True):
            st.session_state.current_word = ""
            st.session_state.last_added_letter = ""
    
    # --- SIDE-BY-SIDE GALLERY OF SPELLED LETTERS ---
    if st.session_state.incoming_message:
        st.write("---")
        st.write("**Reply Spelled in Signs:**")
        letters = [c for c in st.session_state.incoming_message if c.isalpha()]
        if letters:
            # Always create exactly 4 columns so that unused columns are cleared on new shorter inputs
            gallery_cols = st.columns(4)
            for idx, char in enumerate(letters):
                col_idx = idx % 4
                img_path = os.path.join(".", f"{char}.png")
                if os.path.exists(img_path):
                    gallery_cols[col_idx].image(img_path, caption=char, use_container_width=True)

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

sequence = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break
    
    frame = cv2.flip(frame, 1)
    black_screen = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    
    # Calculate facial expression
    if calculate_emotion is not None:
        try:
            st.session_state.current_emotion = calculate_emotion.calculate_emotion(frame)
        except Exception:
            st.session_state.current_emotion = "Neutral"
            
    features = extractor.extract_features(frame)
    status_text = "Searching for Hands..."
    status_class = "val-error"
    
    # Draw hand landmarks
    if extractor.hands_result and extractor.hands_result.hand_landmarks:
        for hand_landmarks in extractor.hands_result.hand_landmarks:
            points = []
            for lm in hand_landmarks:
                cx = int(lm.x * (WINDOW_WIDTH - 400)) + 50  
                cy = int(lm.y * (WINDOW_HEIGHT - 220)) + 40  
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
                    
        sequence.append(features)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            with torch.no_grad():
                res = model(torch.from_numpy(np.array([sequence])).float())
                prob = torch.nn.functional.softmax(res, dim=1)
                max_prob, idx = torch.max(prob, dim=1)
                
                prediction = labels[idx.item()]
                confidence = max_prob.item()
                st.session_state.stability_buffer.append(prediction)
                st.session_state.stability_buffer = st.session_state.stability_buffer[-5:]
                
                if confidence >= 0.80 and st.session_state.stability_buffer.count(prediction) >= 3:
                    status_text = f"Confirmed: {prediction}"
                    status_class = "val-active"
                    
                    current_time = time.time()
                    if prediction != st.session_state.last_added_letter or (current_time - st.session_state.last_addition_time > 1.5):
                        st.session_state.current_word += prediction
                        st.session_state.last_added_letter = prediction
                        st.session_state.last_addition_time = current_time
                        speak(prediction)
                else:
                    status_text = "Detecting..."
                    status_class = "val-warning"
    else:
        sequence = []
        st.session_state.stability_buffer = []
        st.session_state.last_added_letter = ""
        
    # --- ANIMATION CONTROLLER FOR TYPED REPLY ---
    if st.session_state.animation_queue or st.session_state.animation_char:
        now = time.time()
        # Progress to next letter every 1.0 second
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
                    
    target_letter = st.session_state.animation_char
    
    # --- DYNAMIC REFERENCE IMAGE RENDERING ON FRAME ---
    if target_letter and target_letter.isalpha():
        img_path = os.path.join(".", f"{target_letter}.png")
        if os.path.exists(img_path):
            ref_img = cv2.imread(img_path)
            if ref_img is not None:
                ref_img_resized = cv2.resize(ref_img, (250, 250))
                black_screen[80:330, WINDOW_WIDTH - 350:WINDOW_WIDTH - 100] = ref_img_resized
                cv2.putText(black_screen, f"Sign: {target_letter}", (WINDOW_WIDTH - 350, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            
    # --- DRAW STATUS CARDS (HTML) ---
    status_card.markdown(f"""<div class="dashboard-card">
<div class="card-title">Tracking Status</div>
<div class="card-value {status_class}">{status_text}</div>
</div>""", unsafe_allow_html=True)
    
    word_card.markdown(f"""<div class="dashboard-card">
<div class="card-title">Your Signs Word</div>
<div class="card-value val-active">{st.session_state.current_word or "..."}</div>
</div>""", unsafe_allow_html=True)
    
    # Mood formatting
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
    
    # --- DRAW TWO-WAY CHAT HUB (HTML) ---
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
                
    # Downscale for ultra-fast WebSocket transfer (browser GPU will scale it back up smoothly)
    display_frame = cv2.resize(black_screen, (640, 360))
    # Display the final frame in Streamlit
    frame_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

cap.release()
