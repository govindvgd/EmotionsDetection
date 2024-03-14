import librosa
import numpy as np

# Function to extract features from audio file
def extract_features(file_path):
    audio_data, _ = librosa.load(file_path)  # Load audio data without using 'with'
    # Extract features
    mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=librosa.get_samplerate(file_path), n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio_data, sr=librosa.get_samplerate(file_path)).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=librosa.get_samplerate(file_path)).T, axis=0)
    return np.hstack((mfccs, chroma, mel))  # Concatenate features into a single array

# Path to your input audio file
input_file_path = "/home/manav/downloads/DC_a01.wav"

# Extract features from the input audio file
audio_features = extract_features(input_file_path)

# Define thresholds for each emotion
# For example, if mean MFCC value is higher, classify as happy, if lower, classify as sad, etc.
thresholds = {
    "happy": 0.5,
    "sad": -0.5,
    "angry": 0.3,
    "surprised": 0.2,
    "disgust": -0.3,
    "fear": -0.2
}

# Predict emotions based on features and thresholds
predicted_emotions = []
for emotion, threshold in thresholds.items():
    mean_feature_value = np.mean(audio_features)  # Use mean of all features for simplicity
    if mean_feature_value > threshold:
        predicted_emotions.append(emotion)

# Print the predicted emotions
print("Predicted Emotions:", predicted_emotions)
