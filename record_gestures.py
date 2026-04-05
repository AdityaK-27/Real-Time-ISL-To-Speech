import os
import cv2
import time

# Configuration
FPS = 30.0
RECORD_SECONDS = 2.5  # 2.5 seconds ensures we capture the 2.0s (60 frames) perfectly
NUM_FRAMES = int(FPS * RECORD_SECONDS)
BASE_DIR = "greetings_data_large"

# The clean gesture names our pipeline expects
GESTURES = [
    "alright",
    "good_afternoon",
    "good_evening",
    "good_morning",
    "good_night",
    "hello",
    "how_are_you",
    "pleased",
    "thank_you"
]

def ensure_folders():
    """Make sure all necessary directories exist."""
    os.makedirs(BASE_DIR, exist_ok=True)
    for gesture in GESTURES:
        os.makedirs(os.path.join(BASE_DIR, gesture), exist_ok=True)

def find_next_video_id(gesture_dir):
    """Find the next available video number to avoid overwriting."""
    files = [f for f in os.listdir(gesture_dir) if f.endswith('.mp4')]
    if not files:
        return 1
    
    # Try to extract numbers from filenames like 'vid_001.mp4'
    max_id = 0
    for f in files:
        try:
            # Assumes format "vid_XXX.mp4"
            num = int(f.replace('vid_', '').replace('.mp4', ''))
            if num > max_id:
                max_id = num
        except:
            continue
    return max_id + 1

def record_gesture(gesture_name, num_samples):
    """Record multiple samples for a specific gesture."""
    gesture_dir = os.path.join(BASE_DIR, gesture_name)
    start_id = find_next_video_id(gesture_dir)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    # Warm up camera
    print("Starting camera... please wait a second.")
    for _ in range(10):
        cap.read()
        time.sleep(0.01)

    # Get actual camera resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps == 0 or actual_fps > 60:
        actual_fps = 30.0  # Fallback

    print(f"\nCamera initialized at {width}x{height} @{actual_fps}fps")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for i in range(num_samples):
        vid_id = start_id + i
        filename = f"vid_{vid_id:03d}.mp4"
        filepath = os.path.join(gesture_dir, filename)

        # 3-Second Countdown before recording
        for countdown in [3, 2, 1]:
            timer_start = time.time()
            while time.time() - timer_start < 1.0: # Wait 1 second exactly
                ret, frame = cap.read()
                if not ret: break
                
                # Draw countdown
                display_frame = frame.copy()
                text = f"Get Ready: {countdown}"
                cv2.putText(display_frame, text, (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                cv2.putText(display_frame, f"Gesture: {gesture_name}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Sample {i+1} of {num_samples}", (50, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                cv2.imshow('Recording', display_frame)
                cv2.waitKey(1)
        
        # ACTUAL RECORDING
        out = cv2.VideoWriter(filepath, fourcc, FPS, (width, height))
        print(f"🎬 RECORDING: {filename} ({RECORD_SECONDS}s)")
        
        frames_recorded = 0
        while frames_recorded < NUM_FRAMES:
            ret, frame = cap.read()
            if not ret: break
            
            # Save raw frame to video
            out.write(frame)
            frames_recorded += 1
            
            # Show on screen
            display_frame = frame.copy()
            cv2.putText(display_frame, "[ RECORDING ]", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            progress = f"{frames_recorded}/{NUM_FRAMES} frames"
            cv2.putText(display_frame, progress, (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Recording', display_frame)
            cv2.waitKey(1)
            
        out.release()
        print(f"✓ Saved {filepath}")

        # Pause slightly between videos
        time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Successfully recorded {num_samples} samples for '{gesture_name}'!")


def main():
    ensure_folders()
    while True:
        print("\n" + "="*50)
        print("ISL GESTURE VIDEO RECORDER")
        print("="*50)
        for i, g in enumerate(GESTURES):
            print(f"  [{i}] {g}")
        print("  [q] Quit")
        
        choice = input("\nWhich gesture do you want to record? (enter number or 'q'): ").strip()
        if choice.lower() == 'q':
            break
            
        try:
            choice_idx = int(choice)
            if choice_idx < 0 or choice_idx >= len(GESTURES):
                print("Invalid choice.")
                continue
            gesture_name = GESTURES[choice_idx]
            
            samples = int(input(f"How many samples of '{gesture_name}' do you want to record? "))
            if samples <= 0:
                continue
                
            record_gesture(gesture_name, samples)
            
        except ValueError:
            print("Please enter a valid number.")

if __name__ == "__main__":
    main()
