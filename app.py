import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pygame
import pickle
import time

# Charger les modèles entraînés
with open("./feats/phot_mp_drowsy_feats.pkl", "rb") as fp:
    drowsy_feats = pickle.load(fp)
with open("./feats/phot_mp_not_drowsy_feats.pkl", "rb") as fp:
    non_drowsy_feats = pickle.load(fp)
with open("./models/svm_model.pkl", "rb") as svm_file:
    loaded_svm = pickle.load(svm_file)

# Initialisation des bibliothèques
pygame.init()
pygame.mixer.init()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.3, min_tracking_confidence=0.8)

# Spécifications pour les points
right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]]
left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]]
mouth = [[61, 291], [39, 181], [0, 17], [269, 405]]

# Fonction de calcul des distances
def distance(p1, p2):
    return np.sqrt(np.sum((p1[:2] - p2[:2])**2))

# Calcul EAR (Eye Aspect Ratio)
def eye_aspect_ratio(landmarks, eye):
    N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
    N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
    N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
    D = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
    return (N1 + N2 + N3) / (3 * D)

# Calcul MAR (Mouth Aspect Ratio)
def mouth_feature(landmarks):
    N1 = distance(landmarks[mouth[1][0]], landmarks[mouth[1][1]])
    N2 = distance(landmarks[mouth[2][0]], landmarks[mouth[2][1]])
    N3 = distance(landmarks[mouth[3][0]], landmarks[mouth[3][1]])
    D = distance(landmarks[mouth[0][0]], landmarks[mouth[0][1]])
    return (N1 + N2 + N3) / (3 * D)

# Charger l'alerte sonore
alert_sound = r"C:\Users\n\Desktop\projet ia\alert.mp3"
pygame.mixer.music.load(alert_sound)

# Définir l'application Streamlit
st.set_page_config(page_title="Détection de Fatigue", layout="wide", initial_sidebar_state="expanded")

st.title("🛌 Détection de Fatigue en Temps Réel")
st.write("""
Cette application utilise **MediaPipe** et un modèle SVM pré-entraîné pour détecter les signes de fatigue 
en temps réel. Les alertes sonores sont déclenchées lorsqu'une fatigue prolongée est détectée.
""")

run = st.checkbox("Activer la détection de fatigue")
fatigue_threshold = st.slider("Seuil d'alerte (secondes)", 1, 10, 3)

if run:
    # Capturer le flux vidéo
    cap = cv2.VideoCapture(0)
    fatigue_start_time = None

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Impossible d'accéder à la caméra.")
            break

        # Préparer l'image pour MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks_positions = []
                for data_point in face_landmarks.landmark:
                    landmarks_positions.append([data_point.x, data_point.y, data_point.z])
                landmarks_positions = np.array(landmarks_positions)
                landmarks_positions[:, 0] *= frame.shape[1]
                landmarks_positions[:, 1] *= frame.shape[0]

                # Calculer EAR et MAR
                ear = (eye_aspect_ratio(landmarks_positions, left_eye) +
                       eye_aspect_ratio(landmarks_positions, right_eye)) / 2
                mar = mouth_feature(landmarks_positions)
                features = np.array([[ear, mar]])

                # Prédiction avec le modèle SVM
                pred = loaded_svm.predict(features)[0]
                current_time = time.time()

                # Gestion du timer pour la fatigue
                if pred == 1:  # Fatigue détectée
                    if fatigue_start_time is None:
                        fatigue_start_time = current_time
                    elif current_time - fatigue_start_time >= fatigue_threshold:
                        if not pygame.mixer.music.get_busy():
                            pygame.mixer.music.play()
                        cv2.putText(image, "Fatigue détectée!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    fatigue_start_time = None

        # Convertir pour Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_column_width=True)

    cap.release()
