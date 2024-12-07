====================================================================
Détection de la Fatigue par la Mediapipe et du Comportement de Fumer
====================================================================

Bienvenue dans la documentation du projet **Détection de la Fatigue et du Comportement de Fumer**. Ce document détaille les méthodologies, les outils utilisés, et les résultats obtenus pour détecter la fatigue à l'aide de MediaPipe et le comportement de fumer grâce à un modèle CNN.

**Table des matières**

   introduction
   installation
   fatigue_detection
   smoke_detection
   evaluation_results
   future_work
   conclusion


Introduction
============

La détection de la fatigue et des comportements dangereux comme fumer est un enjeu majeur pour la sécurité et la santé. Ce projet vise à fournir une solution robuste en temps réel à l'aide d'outils modernes de vision par ordinateur et de machine learning.

- **Fatigue** : Détection basée sur l'analyse des mouvements des yeux et de la bouche via **MediaPipe**.
- **Fumer** : Classification à l'aide d'un modèle **CNN** entraîné sur des images annotées.

Objectifs du projet :
1. Fournir un système automatisé pour la surveillance.
2. Démontrer l'utilisation combinée de MediaPipe et TensorFlow.

Installation
============

Les bibliothèques suivantes sont nécessaires pour le projet :
1. os : Manipulation des fichiers.
2. pickle : Sauvegarde et chargement des données.
3. cv2  : Traitement d'images avec OpenCV.
4. numpy : Calculs mathématiques.
5. mediapipe : Détection des landmarks faciaux.
6. sklearn: Modélisation en machine learning.

.. code-block:: python

   import os
   import pickle
   import cv2
   import numpy as np
   import mediapipe as mp
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
   from sklearn.svm import SVC
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.neural_network import MLPClassifier

Détection de la Fatigue
=======================

1. **Collecte des données** :
   - Organisation en deux classes :
     - **Drowsy** : Images de personnes somnolentes.
     - **Non Drowsy** : Images de personnes éveillées.

2. **Analyse des landmarks faciaux avec MediaPipe** :
   - Utilisation de **MediaPipe FaceMesh** pour extraire les points clés.

.. code-block:: python

   mp_face_mesh = mp.solutions.face_mesh
   face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

3. **Calcul des caractéristiques** :
   - **EAR** : Eye Aspect Ratio.
   - **MAR** : Mouth Aspect Ratio.

.. code-block:: python

   def distance(p1, p2):
       return np.sqrt(np.sum((p1 - p2)**2))

   def eye_aspect_ratio(landmarks, eye):
       N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
       return (N1 + N2) / (3 * D)

4. **Extraction et sauvegarde** :
   - Calcul des ratios et stockage des données dans des fichiers pour réutilisation.

.. code-block:: python

   with open('drowsy_feats.pkl', 'wb') as file:
       pickle.dump(drowsy_features, file)

5. **Modélisation et entraînement** :
Trois algorithmes de machine learning sont comparés :
  1. SVM (Support Vector Machine).
  2. MLP (Multi-Layer Perceptron).
  3. Random Forest.

Chaque modèle est évalué à l'aide de métriques standard.

Détection du Comportement de Fumer
==================================

1. **Collecte et préparation des données** :
   - Données téléchargées depuis Kaggle.
   - Organisation en deux classes :
     - **Smoking**
     - **Not Smoking**

2. **Conception du modèle CNN** :
   - Architecture avec **Conv2D**, **MaxPooling2D**, et couches denses.

.. code-block:: python

   from tensorflow.keras import Sequential
   from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

   model = Sequential([
       Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
       MaxPooling2D(pool_size=(2, 2)),
       Flatten(),
       Dense(128, activation='relu'),
       Dense(1, activation='sigmoid')
   ])

   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

3. **Résultats** :
   - Précision atteinte : **70%**.

Évaluation des Performances
===========================

1. **Évaluation des Performances** :
Pour évaluer les performances des modèles, les métriques suivantes sont calculées :
   - Accuracy : Mesure globale des prédictions correctes.
   - Precision : Précision des prédictions positives.
   - Recall : Capacité à détecter les exemples positifs.
   - F1-score : Moyenne harmonique entre précision et rappel.

2. **Visualisation des Résultats** :

Les visualisations incluent :
   - Courbes ROC : Représentent le compromis entre le rappel et le taux de faux positifs.
   - Courbes Precision-Recall : Mettent en évidence les performances globales.

.. code-block:: python

   from sklearn.metrics import roc_curve, precision_recall_curve
   import matplotlib.pyplot as plt

   fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
   plt.plot(fpr, tpr, label='Courbe ROC')
   plt.xlabel('Taux de Faux Positifs')
   plt.ylabel('Taux de Vrais Positifs')
   plt.legend()
   plt.show()

Travaux Futurs
==============

1. Améliorer les modèles en utilisant plus de données.
2. Étendre la classification pour inclure d'autres comportements (vapoter, boire, etc.).
3. Intégrer les résultats à des systèmes IoT pour des alertes en temps réel.

Conclusion
==========

Ce projet démontre la puissance de **MediaPipe** et **TensorFlow** pour résoudre des problèmes critiques liés à la sécurité et au bien-être. L'intégration de ces outils offre une solution robuste et extensible.

