import numpy as np
import cv2
from tensorflow.keras.models import load_model

def detect_emotions(model, emotion_dict, frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Calculate square coordinates
        size = max(w, h)
        x1 = x + (w - size) // 2
        y1 = y + (h - size) // 2
        x2 = x1 + size
        y2 = y1 + size

        # Draw square around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Extract face region
        roi_gray = gray[y1:y2, x1:x2]

        # Resize and preprocess face image
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

        # Predict emotion
        prediction = model.predict(cropped_img)
        max_index = int(np.argmax(prediction))
        emotion_label = emotion_dict[max_index]

        # Calculate percentages
        scores = prediction[0] * 100 / np.sum(prediction[0])
        percentages = [f"{emotion_dict[i]}: {scores[i]:.2f}%" for i in range(len(scores))]

        # Display emotion label and percentages
        cv2.putText(frame, f"{emotion_label}: {scores[max_index]:.2f}%", (x1, y1-120), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)
        for i, text in enumerate(percentages):
            cv2.putText(frame, text, (x1-200, y1 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

    return frame

def start_webcam_emotion_detection(model, emotion_dict):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_with_emotions = detect_emotions(model, emotion_dict, frame)

        cv2.imshow('Video', cv2.resize(frame_with_emotions, (800, 600), interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    # Load pre-trained model
    model = load_model('model.h5')

    # Define emotion labels
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # Start webcam emotion detection
    start_webcam_emotion_detection(model, emotion_dict)

if __name__ == "__main__":
    main()
