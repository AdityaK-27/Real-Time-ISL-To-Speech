import os

# Path to your keypoint data
KEYPOINT_PATH = os.path.join('Real-Time-ISL-Translation/keypoint_data')

# Dictionary to store counts
gesture_counts = {}

# Iterate through each gesture folder
for gesture in os.listdir(KEYPOINT_PATH):
    gesture_folder = os.path.join(KEYPOINT_PATH, gesture)
    if os.path.isdir(gesture_folder):
        num_files = len([f for f in os.listdir(gesture_folder) if f.endswith('.npy')])
        gesture_counts[gesture] = num_files

# Print results neatly
print("\n🧩 Gesture Sample Counts:\n")
for gesture, count in sorted(gesture_counts.items()):
    print(f"{gesture:<20} : {count} samples")

# Check imbalance
max_class = max(gesture_counts, key=gesture_counts.get)
min_class = min(gesture_counts, key=gesture_counts.get)

print("\n📊 Summary:")
print(f"Total Gestures: {len(gesture_counts)}")
print(f"Most samples → {max_class} ({gesture_counts[max_class]} files)")
print(f"Fewest samples → {min_class} ({gesture_counts[min_class]} files)")
print(f"Difference: {gesture_counts[max_class] - gesture_counts[min_class]} files\n")