# Fatigue and Smoking Detection System

A real-time fatigue and smoking detection application for classifying drowsy and non-drowsy states, as well as smokers and non-smokers.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Files](#files)
- [Usage](#usage)
- [How it Works](#how-it-works)
- [Model Details](#model-details)
- [Datasets](#datasets)
- [Future Improvements](#future-improvements)

---

## Introduction

The **Fatigue and Smoking Detection System** is an AI-powered application designed to enhance safety and health by monitoring a user's state in real-time. It tracks fatigue through the **Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)** using a webcam. Additionally, it detects smoking behavior using a **Convolutional Neural Network (CNN)** model. The system provides timely alerts to help mitigate risks associated with drowsiness or smoking, particularly in environments like driving, workplaces, or monitoring stations.

---

## Features

- **Real-time Fatigue Detection**:
  - Monitors EAR to detect signs of drowsiness or fatigue.
  - Tracks MAR to identify yawning patterns.
- **Smoking Detection**:
  - Uses a trained CNN model to classify smoking behavior.
- **Alerts**:
  - Visual indicators for user status (Active, Fatigued, Smoking).
  - Audible alarms to alert the user when fatigue or smoking is detected.
- **User-Friendly Interface**:
  - Simple controls and visualizations via **Streamlit**.

---

## Requirements

To run this application, you need the following dependencies:

- **Python 3.8.10**
- **Libraries/Frameworks**:
  - OpenCV: For real-time video capture and facial landmark detection.
  - Mediapipe: To extract facial landmarks.
  - TensorFlow: For the CNN model for smoking detection.
  - Pygame: To generate audible alerts.
  - NumPy: For mathematical computations.
  - Matplotlib: For data visualization.
  - Scikit-learn (sklearn): For additional data preprocessing tasks.
  - Sphinx RTD Theme: To generate documentation.

---

## Files

detection/
  |___building-smoking_model.ipynb = Python notebook contains scripts to build the model
  |___README.md = Repository description
  |___notebook_mediapipe.ipynp =
  |___app.py =
  |___alert.mp3 =
  |___requirements =
  |___.readthedocks.yaml

---


## Usage

To start the application, use the following command:
```bash
streamlit run app.py
```

- **Before Starting**:
  - Ensure your webcam is functional.
  - Close any application using the webcam to avoid conflicts.
- **While Running**:
  - The interface will show real-time tracking of fatigue and smoking.
  - Alerts and indicators will activate based on user behavior.

---

## How it Works

The application uses advanced AI and computer vision techniques to detect fatigue and smoking behavior:

1. **Webcam Monitoring**:
   - Continuously streams video and detects the user's face.

2. **Facial Landmark Detection**:
   - Uses Mediapipe's pre-trained model to identify critical points around the eyes and mouth.

3. **Feature Extraction**:
   - **EAR (Eye Aspect Ratio)**: Measures the ratio of eye height to width to detect drowsiness or closed eyes.
   - **MAR (Mouth Aspect Ratio)**: Tracks mouth movements to identify yawning patterns.

4. **Smoking Detection**:
   - Captured video frames are passed through a trained **CNN model** to classify smoking behavior in real-time.

5. **State Classification**:
   - based on trained models rf,mlp,svm to classify the user's state as:
     - **Active**: Fully alert.
     - **Fatigued**: Signs of drowsiness.
   - Detects smoking based on CNN model predictions.

6. **Alerts**:
   - Visual alerts on the Streamlit interface.
   - Audible alarms generated using Pygame for immediate response.

---

## Model Details

1. **Fatigue Detection**:
   - Uses EAR and MAR metrics derived from facial landmarks detected by Mediapipe.
   - Input: Cropped video frames of the user's face.
   - Output: Binary classification (drowsy/Non-drowsy)
2. **Smoking Detection**:
   - A Convolutional Neural Network (CNN) trained on labeled smoking/non-smoking datasets.
   - Input: Cropped video frames of the user's face.
   - Output: Binary classification (Smoking/Non-Smoking).

3. **Alert System**:
   - Based on mlp,rf,svm outpouts, and also CNN outputs.
   - Customizable alarm sounds using Pygame.

---

## Datasets

### Fatigue Dataset
- **Source**: Kaggle
- **Categories**:
  - **Drowsy**: Images and video frames of individuals exhibiting signs of fatigue, such as closed eyes or yawning.
  - **Non-Drowsy**: Images and video frames of individuals fully alert with open eyes.
- **Usage**:
  - Used for training and testing the EAR and MAR threshold calculations.

### Smoking Dataset
- **Source**: Kaggle
- **Categories**:
  - **Smoker**: Images of individuals smoking, showing clear visual indicators like cigarette presence.
  - **Non-Smoker**: Images of individuals not engaging in smoking behavior.
- **Usage**:
  - Used to train the CNN model for binary classification of smoking behavior.

---

## Future Improvements

- **Enhanced Accuracy**:
  - Use larger datasets for better CNN training.
  - Integrate more robust pre-trained models for facial landmark detection.

- **Multimodal Detection**:
  - Combine additional metrics such as head position or blinking rate for fatigue detection.

- **Platform Expansion**:
  - Deploy as a mobile app for broader accessibility.
  - Add support for external cameras and sensors.

- **User Analytics**:
  - Provide detailed reports of user behavior over time.

- **Customizable Thresholds**:
  - Allow users to modify EAR and MAR thresholds based on their individual needs.
