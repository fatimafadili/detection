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
=========================================
Documentation : Détection de Fatigue
=========================================

Introduction
============

Ce projet utilise divers algorithmes de machine learning (MLP, SVM, Random Forest) pour détecter la fatigue et surveiller les comportements en temps réel. Voici un extrait du code montrant l'importation des bibliothèques nécessaires pour la modélisation machine learning.

Code Python
===========

Voici le code des bibliothèques importées utilisées pour le projet :

.. code-block:: python
   :linenos:

   import sklearn
   from sklearn.model_selection import train_test_split
   from sklearn.pipeline import make_pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.neural_network import MLPClassifier
   from sklearn.svm import SVC
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve

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

    import cv2====================================
Détection de la Fatigue - Documentation
====================================

Description
===========
Ce projet utilise MediaPipe et des techniques de machine learning pour détecter la fatigue à partir de flux vidéo en temps réel. Il repose sur les concepts suivants :
- **EAR (Eye Aspect Ratio)** : Mesure la fermeture des yeux.
- **MAR (Mouth Aspect Ratio)** : Mesure l'ouverture de la bouche.
- Modélisation avec un modèle **SVM** pour détecter la fatigue.

Exemple de Code
===============
Voici un exemple de code Python utilisé pour capturer le flux vidéo, extraire les caractéristiques faciales et prédire la fatigue en temps réel :

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

---

### Points clés :
1. Utilisez `.. code-block:: python` pour formater le code Python.
2. Assurez-vous d'indenter correctement le code après `.. code-block::`.
3. Placez le fichier `index.rst` dans votre projet Read the Docs et vérifiez qu'il est bien configuré dans le fichier `conf.py`.
=====================================
Documentation : Détection de Fatigue
=====================================

   cap.release()
   cv2.destroyAllWindows()
