import pyttsx3
import time

# Example gestures
detected_gestures = ["Thank you", "Good morning", "How are you"]

# Build a full sentence for TTS
if detected_gestures:
    sentence = "So, The gestures detected were: " + ", ".join(detected_gestures) + "."

    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 170)
    engine.setProperty('volume', 1.0)

    # Give engine time to initialize
    time.sleep(0.8)

    print(f"Speaking: {sentence}")
    engine.say(sentence)
    engine.runAndWait()
    time.sleep(0.8)

print("✅ Done speaking.")
