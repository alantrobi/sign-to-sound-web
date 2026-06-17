import speech_recognition as sr

r = sr.Recognizer()
# Testing Index 2 (Airdopes)
mic = sr.Microphone(device_index=2)

print("\n--- RECORDING TEST ---")
print("Speak into your Airdopes NOW... say 'Testing 1 2 3'")

with mic as source:
    r.adjust_for_ambient_noise(source, duration=0.5)
    audio = r.listen(source, timeout=3, phrase_time_limit=3)

print("Recording stopped. Saving file...")

# Write the raw audio data straight to a wav file
with open("mic_output.wav", "wb") as f:
    f.write(audio.get_wav_data())

print("Done! Open 'D:\\twoway_sign\\mic_output.wav' and check if you can hear your voice.\n")
