import speech_recognition as sr
import pyaudio

p = pyaudio.PyAudio()
print("\n=== AVAILABLE AUDIO INPUT DEVICES ===")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev['maxInputChannels'] > 0:
        print(f"Index {i}: {dev['name']}")
print("=====================================\n")
