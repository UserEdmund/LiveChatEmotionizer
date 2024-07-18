#LiveChatEmotionizer

This program detects emotions in real-time using a webcam and displays the results using OpenCV and Matplotlib. It captures video, performs face detection and emotion recognition, and allows user interaction through a tkinter window.

## Features

- **Real-time Emotion Detection:** Utilizes FER (Face Emotion Recognition) to detect emotions including anger, disgust, fear, happiness, sadness, surprise, and neutral.
- **Video Recording:** Captures video from the webcam and saves it as `emotion_video.avi`.
- **Interactive Chart:** Displays a live-updating bar chart of detected emotions.
- **User Interaction:** Allows users to input text via a tkinter window and receive responses based on emotion statistics.

## Requirements

- Python 3.x
- OpenCV
- Matplotlib
- NumPy
- Pandas
- Imageio
- FER (Face Emotion Recognition)
- tkinter
- ollama (for chatbot integration)
