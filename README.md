# Drowsy Driver Detection 

A real-time fatigue detection application for classifying drowsy and non-drowsy .

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How it Works](#how-it-works)

## Introduction

The Fatigue Detection System is designed to monitor the user's alertness by analyzing their eye aspect ratio (EAR)  and mouth aspect ratio (MAR) through a webcam. It provides visual and audible alerts when signs of fatigue are detected, thereby reducing the risk of accidents caused by drowsiness.

## Features

- Real-time eye tracking to detect fatigue
- Visual indicators showing the user's alertness status
- Audible alarms to alert the user

## Requirements

This application requires the following:

- Python 3.8.10
- OpenCV
- Mediapipe
- Pygame
- numpy                                     

## Usage
To run the application, execute the following command:
 streamlit run app.py
```
Make sure your webcam is enabled and properly set up before starting the application!

## How it Works
- The application uses a webcam to continuously monitor the user.
- Facial landmarks are detected using dlib and Mediapipe's pre-trained model.
- Eye aspect ratios and mouth aspect ratios are calculated to determine the user's level of alertness.
- The system classifies the user's state as active, fatigued, or asleep based on EAR and MAR values and provides corresponding alerts.
  
