import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio
import matplotlib
import time
from fer import FER
import threading
import tkinter as tk
from queue import Queue
from matplotlib.animation import FuncAnimation
import ollama

# Set matplotlib backend
matplotlib.use('TkAgg')

# Initialize the FER (Face Emotion Recognition) detector using MTCNN
detector = FER(mtcnn=True)

# Start capturing video from the webcam (device 0)
cap = cv2.VideoCapture(0)

# Set a frame rate for recording the video (adjust based on your webcam's capabilities)
frame_rate = 4.3

# Initialize OpenCV's VideoWriter to save the video with the specified frame rate
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('emotion_video.avi', fourcc, frame_rate, (640, 480))

# Set up a matplotlib figure for displaying live emotion detection results
fig, ax = plt.subplots()
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
bars = ax.bar(emotion_labels, [0]*7, color='lightblue')
plt.ylim(0, 1)
plt.ylabel('Confidence')
plt.title('Real-time Emotion Detection')
ax.set_xticklabels(emotion_labels, rotation=45)

# Initialize imageio writer to save live chart updates as a GIF
gif_writer = imageio.get_writer('emotion_chart.gif', mode='I', duration=0.1)

# List to store cumulative emotion statistics for each frame
emotion_statistics = []

# Shared variable to store user input text
input_text = ""

# Queue for passing data between threads
data_queue = Queue()

# Function to handle the emotion detection and video processing
def emotion_detection():
    global input_text, cap, out, gif_writer, emotion_statistics, data_queue
    webcam_start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = detector.detect_emotions(frame)
            largest_face = None
            max_area = 0

            for face in result:
                box = face["box"]
                x, y, w, h = box
                area = w * h
                if area > max_area:
                    max_area = area
                    largest_face = face

            if largest_face:
                box = largest_face["box"]
                current_emotions = largest_face["emotions"]
                emotion_statistics.append(current_emotions)

                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                emotion_type = max(current_emotions, key=current_emotions.get)
                emotion_score = current_emotions[emotion_type]
                emotion_text = f"{emotion_type}: {emotion_score:.2f}"
                cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                #cv2.putText(frame, input_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                data_queue.put((frame.copy(), current_emotions.copy()))

            cv2.imshow('Emotion Detection', frame)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        
    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        webcam_end_time = time.time()
        print(f"Webcam active time: {webcam_end_time - webcam_start_time:.2f} seconds")

        cap.release()
        cv2.destroyAllWindows()

        out.release()
        gif_writer.close()

        emotion_df = pd.DataFrame(emotion_statistics)
        #print("Emotion statistics:", emotion_statistics)
        plt.figure(figsize=(10, 10))
        for emotion in emotion_labels:
            plt.plot(emotion_df[emotion].cumsum(), label=emotion)
        plt.title('Cumulative Emotion Statistics Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Cumulative Confidence')
        plt.legend()
        plt.savefig('cumulative_emotions.jpg')
        plt.close()

# Function to update the Matplotlib chart from the main thread
def update_chart(frame, emotions):
    ax.clear()
    ax.bar(emotion_labels, [emotions.get(emotion, 0) for emotion in emotion_labels], color='lightblue')
    plt.ylim(0, 1)
    plt.ylabel('Confidence')
    plt.title('Real-time Emotion Detection')
    ax.set_xticklabels(emotion_labels, rotation=45)
    plt.draw()
    plt.pause(0.001)  # Needed to update the plot

# Function to handle user input in a tkinter window
def create_input_window():
    global input_text

    def on_text_change(event):
        global input_text
        input_text = entry.get()

        emotion_dff = pd.DataFrame(emotion_statistics)

        # Initialize sumstats as a dictionary to accumulate cumsum values
        sumstats = {emotion: 0.0 for emotion in emotion_labels}

        for emotion in emotion_labels:
            sumstats[emotion] = emotion_dff[emotion].cumsum().iloc[-1]

        # Normalize the cumulative sums to get percentages
        threshold = 20
        total_sum = sum(value for value in sumstats.values() if value >= threshold)
        sumstats = {emotion: 0 if value < threshold else value / total_sum * 100 for emotion, value in sumstats.items()}

        for emotion in emotion_labels:
            print(f"{emotion}: {sumstats[emotion]}")
        
       
        print("Message:"+input_text+"\n"+"Emotion Percentages:"+str(sumstats)+"\n")

        response = ollama.chat(model='chatemo', messages=[
        {
            'role': 'user',
            'content': "Message:"+input_text+"\n"+"Emotion Percentages:"+str(sumstats)+" "
        },
        ])
        print(response['message']['content'])


    root = tk.Tk()
    root.title("Text Input")

    entry = tk.Entry(root)
    entry.pack()

    entry.bind("<Return>", on_text_change)

    root.mainloop()

# Start the emotion detection thread
emotion_thread = threading.Thread(target=emotion_detection)
emotion_thread.daemon = True
emotion_thread.start()

# Update the Matplotlib chart from data_queue
def update_from_queue():
    while True:
        if not data_queue.empty():
            frame, emotions = data_queue.get()
            update_chart(frame, emotions)
        time.sleep(0.01)  # Adjust sleep time as needed

# Start the thread to update the Matplotlib chart
update_thread = threading.Thread(target=update_from_queue)
update_thread.daemon = True
update_thread.start()

# Run the tkinter mainloop for user input
create_input_window()
