import os
import numpy as np
import librosa

DATASET_PATH = "dataset/"

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = librosa.util.fix_length(mel_db, size=128, axis=1)
    return mel_db

# Get class names but IGNORE .DS_Store or files
class_names = [
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
]

print("Detected classes:", class_names)

X = []
Y = []

for label_index, label in enumerate(class_names):
    folder = os.path.join(DATASET_PATH, label)
    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        # Skip hidden files
        if file.startswith('.') or not os.path.isfile(path):
            continue

        try:
            mel = extract_features(path)
            X.append(mel)
            Y.append(label_index)
        except Exception as e:
            print("Error:", e)

X = np.array(X)
Y = np.array(Y)

np.save("X.npy", X)
np.save("Y.npy", Y)
np.save("class_names.npy", np.array(class_names))

print("Saved X.npy, Y.npy and class_names.npy")
