# signtospeech
Sign Language to Speech Converter

Project Overview:
This project is a deep learning–based system that recognizes hand gestures from sign language and converts them into speech in real time. The system captures gesture frames from a webcam, processes them into motion silhouettes, and then uses a trained neural network model to predict the gesture class and generate speech output.

Objective:
The objective of this project is to assist communication by translating sign language gestures into audible speech using computer vision and machine learning techniques.

Dataset Collection:
The dataset for this project was collected manually using a webcam.
- Total Gestures: 10
- Samples per Gesture: 20
- Frames per Sample: 30
Each gesture sequence consists of multiple frames captured during the motion of the hand. These frames were later processed and used for training the model.

Project Pipeline:
The complete workflow of the project includes the following steps:
1. Folder Creation
   Organizing dataset folders for different gesture classes.
2. Data Collection
   Capturing gesture sequences using a webcam.
3. Preprocessing
   Converting frames into motion masks / silhouettes to highlight gesture movement.
4. Model Training
   Training a deep learning model using the processed dataset.
5. Prediction
   Real-time gesture recognition using the trained model.
6. Speech Output
   Converting predicted gesture labels into speech output.

Files in the Repository:
- datacollect.py → Script used to collect gesture data.
- foldercreation.py → Creates dataset directories for gestures.
- preprocess.py → Processes frames and generates motion masks.
- trainmodel.py → Trains the gesture recognition model.
- predict.py → Performs real-time gesture prediction.
- datacollection.md → Documentation of dataset collection process.

Model Performance:
Training Accuracy: ~92%
Validation Accuracy: ~96%
The model performs well in recognizing gestures under controlled conditions with proper lighting and background.

Tools Used:
- Python
- OpenCV
- TensorFlow / Keras
- NumPy
- pyttsx3 (Text-to-Speech)

How to Run the Project:
1. Collect Dataset
   Run datacollect.py
2. Preprocess Data
   Run:preprocess.py
3. Train Model
   Run:trainmodel.py
4. Prediction
   Run:predict.py

Thankyou!
