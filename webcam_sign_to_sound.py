import os
import cv2
import torch
import numpy as np
import pyttsx3
import threading
import queue
import time

from model import SignLSTM
from mediapipe_extractor import MediaPipeFeatureExtractor

# --- EMOTION MODULE INTEGRATION ---
try:
    import calculate_emotion
except ImportError:
    calculate_emotion = None

# 1. Configuration & Weight Path Management
MODEL_PATH = r"isl_model.pth"  # Absolute fallback path configuration
REF_IMG_DIR = r"." 
device = torch.device("cpu") 

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
          'N', 'O', 'P', 'Q', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

model = SignLSTM(len(labels)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- SYSTEM BUFFER VARIABLES ---
current_word = ""          
last_added_letter = ""     
last_addition_time = 0     
DEBOUNCE_DELAY = 1.5       
current_emotion = "Neutral" 

# Two-Way Buffer Streams
type_input_buffer = ""      
incoming_message = ""       

# 2. Asynchronous Voice Engine Queue Loop
audio_queue = queue.Queue()

def say_text():
    while True:
        text = audio_queue.get()
        if text is None: break
        try:
            temp_engine = pyttsx3.init('sapi5')
            temp_engine.say(text)
            temp_engine.runAndWait()
            del temp_engine
        except Exception as e:
            pass
        finally:
            audio_queue.task_done()

threading.Thread(target=say_text, daemon=True).start()

# 3. Main Frame Processing Setup
extractor = MediaPipeFeatureExtractor()
cap = cv2.VideoCapture(0)

sequence = []
stability_buffer = []

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

cv2.namedWindow("Sign2Sound Multimodal", cv2.WINDOW_NORMAL)
print("\n=== TWO-WAY CHANNELS ACTIVATED: WIDESCREEN DESKTOP SYSTEM READY ===")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    black_frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    
    if calculate_emotion is not None:
        try:
            current_emotion = calculate_emotion.calculate_emotion(frame)
        except Exception as e:
            current_emotion = "Neutral"

    features = extractor.extract_features(frame)
    
    # Process and render hand structures using clean array iterations
    if extractor.hands_result and extractor.hands_result.hand_landmarks:
        for hand_landmarks in extractor.hands_result.hand_landmarks:
            points = []
            for lm in hand_landmarks:
                cx = int(lm.x * (WINDOW_WIDTH - 400)) + 50  
                cy = int(lm.y * (WINDOW_HEIGHT - 220)) + 40  
                points.append((cx, cy))
                cv2.circle(black_frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.circle(black_frame, (cx, cy), 3, (255, 255, 255), -1)
            
            # Independent physical link mapping lines
            connections = [
                (0,1), (1,2), (2,3), (3,4),       # Thumb
                (0,5), (5,6), (6,7), (7,8),       # Index
                (5,9), (9,10), (10,11), (11,12),  # Middle
                (9,13), (13,14), (14,15), (15,16),# Ring
                (13,17), (0,17), (17,18), (18,19), (19,20) # Pinky
            ]
            for start_idx, end_idx in connections:
                if start_idx < len(points) and end_idx < len(points):
                    cv2.line(black_frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
            
        sequence.append(features)
        sequence = sequence[-30:] 

        if len(sequence) == 30:
            with torch.no_grad():
                res = model(torch.from_numpy(np.array([sequence])).float().to(device))
                prob = torch.nn.functional.softmax(res, dim=1)
                max_prob, idx = torch.max(prob, dim=1)
                
                prediction = labels[idx.item()]
                confidence = max_prob.item()
                stability_buffer.append(prediction)
                stability_buffer = stability_buffer[-5:] 

                if confidence >= 0.80 and stability_buffer.count(prediction) >= 3:
                    status_text = f"Confirmed: {prediction}"
                    color = (0, 255, 0)
                    
                    current_time = time.time()
                    if prediction != last_added_letter or (current_time - last_addition_time > DEBOUNCE_DELAY):
                        current_word += prediction
                        last_added_letter = prediction
                        last_addition_time = current_time
                else:
                    status_text = f"Detecting..."
                    color = (0, 255, 255)

                cv2.putText(black_frame, status_text, (50, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
    else:
        sequence = []
        stability_buffer = []
        last_added_letter = "" 
        cv2.putText(black_frame, "Searching for Hands...", (50, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)

    # --- DYNAMIC REFERENCE IMAGE RENDERING COLUMN ---
    target_letter = type_input_buffer[-1].upper() if type_input_buffer else ""
    
    if target_letter and target_letter.isalpha():
        img_path = os.path.join(REF_IMG_DIR, f"{target_letter}.png")
        if os.path.exists(img_path):
            ref_img = cv2.imread(img_path)
            if ref_img is not None:
                ref_img_resized = cv2.resize(ref_img, (250, 250))
                black_frame[80:330, WINDOW_WIDTH - 350:WINDOW_WIDTH - 100] = ref_img_resized
                cv2.putText(black_frame, f"Sign: {target_letter}", (WINDOW_WIDTH - 350, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # --- UI DISPLAY HUD PANEL ---
    cv2.rectangle(black_frame, (0, WINDOW_HEIGHT - 180), (WINDOW_WIDTH, WINDOW_HEIGHT), (18, 18, 18), -1)
    
    cv2.putText(black_frame, f"Your Signs: {current_word}", (40, WINDOW_HEIGHT - 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
    
    cv2.putText(black_frame, f"Type Reply: {type_input_buffer}_", (40, WINDOW_HEIGHT - 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    cv2.putText(black_frame, f"Received: {incoming_message}", (40, WINDOW_HEIGHT - 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 150, 50), 2)
    
    cv2.putText(black_frame, f"Mood: {current_emotion}", (WINDOW_WIDTH - 280, WINDOW_HEIGHT - 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 2)

    cv2.imshow("Sign2Sound Multimodal", black_frame)
    
    # --- KEYBOARD CONTROLS ENGINE ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == 32:  # Spacebar clears sign string
        if current_word:
            audio_queue.put(current_word)
            current_word = ""  
            last_added_letter = ""
    elif key == 13:  # Enter locks typed text
        if type_input_buffer:
            incoming_message = type_input_buffer
            type_input_buffer = "" 
    elif key == 8:   # Backspace mechanics
        if current_word:
            current_word = current_word[:-1]
        elif type_input_buffer:
            type_input_buffer = type_input_buffer[:-1]
    elif 32 < key < 127:
        type_input_buffer += chr(key)

audio_queue.put(None)
cap.release()
cv2.destroyAllWindows()
