========================================
Détection de la Fatigue par la mediapipe
========================================

Bienvenue dans la documentation de notre projet de détection de fatigue basé sur MediaPipe. 
Ce projet a pour objectif de détecter les signes de fatigue en temps réel grâce à l'analyse des indices 
EAR (Eye Aspect Ratio) et MAR (Mouth Aspect Ratio).

.. contents:: Table des matières
   :local:
======================================
Fatigue and Smoking Detection Project
======================================

Ce projet est une application de détection en temps réel basée sur des techniques d’apprentissage automatique et des outils comme MediaPipe et TensorFlow. Il se concentre sur deux objectifs principaux :

1. **Détection de la fatigue** : Identifier les signes de somnolence à partir d’images ou de vidéos en analysant des caractéristiques faciales comme le clignement des yeux et l’ouverture de la bouche.
2. **Détection des comportements de fumée** : Reconnaître automatiquement si une personne est en train de fumer à partir de vidéos en utilisant un modèle CNN (Convolutional Neural Network).

Ce document fournit une description détaillée du projet, des étapes de développement et des résultats obtenus.

Table des matières
==================

   :maxdepth: 2
   :caption: Contenu

   installation
   fatigue_detection
   smoke_detection
   evaluation_results

Introduction
============

La détection de la fatigue et des comportements à risque tels que fumer est un sujet crucial pour la sécurité et le bien-être. Ce projet vise à :

- Fournir une solution automatisée pour la détection en temps réel.
- Utiliser des algorithmes robustes et des outils de pointe.
- Permettre une intégration dans des applications concrètes (véhicules intelligents, systèmes de surveillance, etc.).

Installation
============

Avant de démarrer, installez les bibliothèques nécessaires :

1. `os` : Manipulation des fichiers.
2. `pickle` : Sauvegarde et chargement des données.
3. `cv2` : Traitement des images avec OpenCV.
4. `numpy` : Calculs mathématiques.
5. `mediapipe` : Détection des landmarks faciaux.
6. `sklearn` : Modèles de machine learning.
7. `tensorflow` : Conception du modèle CNN.

Pour des instructions détaillées sur l'installation, consultez la section **installation.rst**.

Détection de la fatigue
=======================

**1. Collecte des données**  
- Deux ensembles d'images : *drowsy* (somnolentes) et *non drowsy* (éveillées).
- Les données sont organisées en deux catégories pour faciliter l’entraînement du modèle.

**2. Détection des landmarks faciaux avec MediaPipe**  
- Utilisation de MediaPipe FaceMesh pour extraire les points clés du visage, notamment :
  - Les yeux (pour analyser le clignement).
  - La bouche (pour évaluer l’ouverture).

**3. Calcul des caractéristiques**  
- Deux caractéristiques principales sont extraites :
  - **EAR (Eye Aspect Ratio)** : Ratio basé sur les distances entre les landmarks des yeux pour détecter la fermeture.
  - **MAR (Mouth Aspect Ratio)** : Ratio pour mesurer l’ouverture de la bouche.
def distance(p1, p2):
    return (((p1[:2] - p2[:2])**2).sum())**0.5


**4. Extraction des caractéristiques**  
- Les valeurs EAR et MAR sont calculées pour chaque image et stockées pour l’entraînement des modèles.

**5. Modélisation**  
- Trois modèles de machine learning sont utilisés :
  - **MLP (Multi-layer Perceptron)**.
  - **SVM (Support Vector Machine)**.
  - **Random Forest**.
- Les données sont divisées en ensembles d’entraînement et de test.


.. code-block:: python

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
# Charger le modèle SVM
with open("./models/svm_model.pkl", "rb") as svm_file:
    loaded_svm = pickle.load(svm_file)

print("Modèle chargé avec succès.")

# Initialisation des bibliothèques
pygame.init()
pygame.mixer.init()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.3, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Spécifications pour les points
right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]]  # right eye
left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]]  # left eye
mouth = [[61, 291], [39, 181], [0, 17], [269, 405]]  # mouth

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

# Capturer le flux vidéo
cap = cv2.VideoCapture(0)

# Variables pour le timer
fatigue_start_time = None  # Temps où la fatigue commence à être détectée
fatigue_threshold = 3  # Temps en secondes avant déclenchement de l'alarme

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Préparer l'image pour MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)

    # Dessiner les résultats
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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

            # Gestion du timer pour la fatigue
            current_time = time.time()
            if pred == 1:  # Fatigue détectée
                if fatigue_start_time is None:
                    fatigue_start_time = current_time  # Démarrer le timer
                elif current_time - fatigue_start_time >= fatigue_threshold:
                    cv2.putText(image, "Fatigue detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play()
            else:
                fatigue_start_time = None  # Réinitialiser si la fatigue n'est plus détectée

            # Affichage du statut
            if fatigue_start_time is None:
                cv2.putText(image, "Normal", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Afficher l'image
    cv2.imshow("Fatigue Detection", image)

    # Quitter avec la touche 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

mlp = MLPClassifier(hidden_layer_sizes=(5, 3), random_state=1, max_iter=1000)
mlp.fit(X_train, y_train)
mlp_preds = mlp.predict(X_test)
mlp_probas = mlp.predict_proba(X_test)

svm = SVC(probability=True)
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_test)
svm_probas = svm.predict_proba(X_test)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_probas = rf.predict_proba(X_test)


Détection des comportements de fumée
====================================

**1. Collecte des données**  
- Les datasets sont téléchargés depuis Kaggle à l’aide d’une clé API (*kaggle.json*).
- Organisation des images en deux catégories : *smoking* (fumeur) et *not smoking* (non-fumeur).

**2. Construction du modèle CNN**  
- Utilisation de TensorFlow pour développer un modèle CNN capable de détecter automatiquement les comportements de fumée.
- Les scripts sont contenus dans `building_model.ipynb`.

**3. Résultats obtenus**  
- Précision du modèle : **60 %**.
- Le modèle est capable de distinguer les comportements avec une précision modérée, qualifiée de "bonne adéquation".

Résultats et évaluation
=======================

**1. Évaluation des performances**  
Les performances des modèles sont mesurées à l’aide de différentes métriques :
- **Accuracy** : Pourcentage de prédictions correctes.
- **Precision** : Précision des prédictions positives.
- **Recall** : Capacité du modèle à détecter les vrais positifs.
- **F1-score** : Moyenne harmonique entre précision et rappel.

**2. Visualisation des résultats**  
- Les courbes ROC et Precision-Recall sont tracées pour comparer les modèles.
- Ces visualisations montrent les points forts et les limites des différentes approches.

**3. Résumé des performances**
- Les modèles de détection de fatigue affichent une précision élevée grâce à l’utilisation des caractéristiques EAR et MAR.
- Le modèle de détection de fumée atteint une précision moyenne de 60 %, montrant qu'il peut être amélioré avec davantage de données.

Liens utiles
============

- `README.md <README.md>`_: Guide principal du projet.
- `building_model.ipynb <building_model.ipynb>`_: Script pour construire le modèle CNN.
- `app.py <app.py>`_: Application principale pour la détection.

---

### Étapes pour insérer ce fichier dans Read the Docs

1. Remplacez ou mettez à jour votre fichier **index.rst** avec ce contenu.
2. Ajoutez des fichiers supplémentaires (par exemple, `installation.rst`, `fatigue_detection.rst`, etc.) si nécessaire.
3. Poussez les modifications vers votre dépôt GitHub.
4. Synchronisez votre projet avec Read the Docs en vérifiant que le fichier `index.rst` est bien configuré comme point d'entrée principal.
