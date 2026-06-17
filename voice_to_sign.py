import os
import cv2
import numpy as np
import speech_recognition as sr

# Path to your downloaded internet reference images
IMAGE_FOLDER = r"."

def load_sign_image(letter_name):
    """Searches the folder for the spoken letter image file."""
    for ext in ['.jpg', '.jpeg', '.png', '.BMP', '.JPG', '.PNG']:
        img_path = os.path.join(IMAGE_FOLDER, f"{letter_name}{ext}")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                return img
    return None

def main():
    # Initialize the microphone listener
    recognizer = sr.Recognizer()
    
    # Strictly locked to your verified Airdopes mic index
    mic = sr.Microphone(device_index=2)
    
    # Create a persistent window layout to display the images
    window_name = "Voice to Sign Language Display"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 600, 600)
    
    # Start with a clean black screen
    display_frame = np.zeros((600, 600, 3), dtype=np.uint8)
    cv2.putText(display_frame, "Say an alphabet... (e.g., 'A')", (100, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    print("\n=== VOICE TO SIGN SYSTEM ACTIVE ===")
    print("Using Airdopes Microphone (Index 2). Press 'q' in the window to quit.\n")

    while True:
        cv2.imshow(window_name, display_frame)
        
        # Check if user pressed 'q' to close the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        with mic as source:
            print("Calibrating mic background noise...")
            # 1. Solid calibration pass to handle Bluetooth compression
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            
            # 2. Sensitivity tuning based on your working audio file
            recognizer.pause_threshold = 1.0  
            recognizer.energy_threshold = 120  # Keeps it highly sensitive to your earbuds
            
            try:
                print("Listening for an alphabet...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
                
                print("Processing audio...")
                # Convert speech to text via Google's engine
                spoken_text = recognizer.recognize_google(audio).strip().upper()
                print(f"Recognized Voice input: '{spoken_text}'")

                # --- ALPHABET CHECK LOGIC ---
                # Clean up any trailing periods or small noise characters
                clean_text = "".join([c for c in spoken_text if c.isalpha()])

                # Check if what you said corresponds to a single alphabet letter
                if len(clean_text) == 1:
                    letter = clean_text
                    
                    # Try to load the image from your reference folder
                    sign_img = load_sign_image(letter)
                    
                    if sign_img is not None:
                        # Resize the image to fit our 600x600 display window perfectly
                        display_frame = cv2.resize(sign_img, (600, 600))
                        
                        # Overlay a clean green text banner showing what letter it is displaying
                        cv2.rectangle(display_frame, (0, 0), (160, 50), (0, 0, 0), -1)
                        cv2.putText(display_frame, f"Sign: {letter}", (15, 35), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        print(f"Showing sign image for letter: {letter}\n")
                    else:
                        # If image is missing from folder, clear screen and show error text
                        display_frame = np.zeros((600, 600, 3), dtype=np.uint8)
                        cv2.putText(display_frame, f"Image for '{letter}' not found", (80, 300), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        print(f"Error: Missing image file '{letter}' inside reference_images.\n")
                else:
                    print(f"Ignored: '{spoken_text}' is not a single alphabet letter.\n")

            except sr.WaitTimeoutError:
                # Silently cycle back if you don't speak within the window timeout
                continue
            except sr.UnknownValueError:
                print("Could not understand the audio clearly, try speaking closer to your Airdopes.\n")
                continue
            except sr.RequestError as e:
                print(f"Speech service error: {e}\n")
                continue
            except Exception as e:
                continue

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
