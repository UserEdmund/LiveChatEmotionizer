# 💬 LiveChatEmotionizer

**LiveChatEmotionizer** is a real-time emotion recognition system that leverages computer vision and machine learning to detect facial emotions via webcam. It provides interactive feedback through charts and chatbot responses, all wrapped in a user-friendly GUI.

---

## ⚙️ Features

- **Real-time Emotion Detection**  
  Uses **FER (Face Emotion Recognition)** to identify emotions:  
  _Anger, Disgust, Fear, Happiness, Sadness, Surprise, Neutral_

- **🎥 Video Recording**  
  Captures webcam feed and saves it as `emotion_video.avi`.

- **📊 Live Emotion Chart**  
  Displays a continuously updating bar chart using Matplotlib to visualize emotional trends.

- **🗣️ Interactive Chatbot**  
  Users can input text via a Tkinter window and receive chatbot responses based on current emotion stats.

---

## 🧰 Requirements

Make sure the following packages are installed:

- Python 3.x
- OpenCV
- Matplotlib
- NumPy
- Pandas
- Imageio
- [FER (Face Emotion Recognition)](https://github.com/justinshenk/fer)
- Tkinter
- [Ollama](https://ollama.com) (for chatbot integration)

---

Ready to detect feelings in frames and give your webcam a sixth sense? Let me know if you want help writing a `README`, adding screenshots, or optimizing performance. I’m all in!
