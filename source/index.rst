Détection de la Fatigue et du Comportement de Fumer
===================================================

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
- Fournir un système automatisé pour la surveillance.
- Démontrer l'utilisation combinée de MediaPipe et TensorFlow.

Installation
============

Pour installer les dépendances nécessaires, exécutez la commande suivante :

.. code-block:: bash

   pip install opencv-python mediapipe tensorflow numpy scikit-learn

Voici les bibliothèques utilisées :
- **os** : Gestion des fichiers.

- **pickle** : Sauvegarde et chargement des données.

- **OpenCV** : Capture et traitement d'images.

- **MediaPipe** : Détection des landmarks faciaux.

- **TensorFlow** : Entraînement du modèle CNN.

- **scikit-learn** : Évaluation des modèles.

Détection de la Fatigue
=======================

### Collecte des données

Les données sont organisées en deux catégories :
- **Somnolent** : Correspond aux états de fatigue.

- **Éveillé** : Correspond aux états de vigilance.

Les images sont collectées et annotées pour entraîner le modèle.

### Analyse des landmarks faciaux avec MediaPipe

Les landmarks faciaux (points clés) sont extraits à l'aide de **MediaPipe FaceMesh** pour analyser les yeux et la bouche.

.. code-block:: python

   import mediapipe as mp

   mp_face_mesh = mp.solutions.face_mesh
   face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

### Calcul des caractéristiques

Les ratios suivants sont calculés :
- **EAR (Eye Aspect Ratio)** : Pour détecter la fermeture des yeux.
- **MAR (Mouth Aspect Ratio)** : Pour analyser les bâillements.

.. code-block:: python

   def calculate_ear(eye_landmarks):
       # Calcul basé sur les distances entre les points de l'œil
       pass

   def calculate_mar(mouth_landmarks):
       # Calcul basé sur les distances des points de la bouche
       pass

Les caractéristiques calculées sont enregistrées pour un entraînement ultérieur.

### Modélisation et entraînement

Trois algorithmes de machine learning sont comparés :
1. **SVM (Support Vector Machine)**.
2. **MLP (Multi-Layer Perceptron)**.
3. **Random Forest**.

Chaque modèle est évalué à l'aide de métriques standard.

Évaluation des Performances
===========================

Pour évaluer les performances des modèles, les métriques suivantes sont calculées :
- **Accuracy** : Mesure globale des prédictions correctes.
- **Precision** : Précision des prédictions positives.
- **Recall** : Capacité à détecter les exemples positifs.
- **F1-score** : Moyenne harmonique entre précision et rappel.

### Visualisation des Résultats

Les visualisations incluent :
- **Courbes ROC** : Représentent le compromis entre le rappel et le taux de faux positifs.
- **Courbes Precision-Recall** : Mettent en évidence les performances globales.

.. code-block:: python

   from sklearn.metrics import roc_curve, precision_recall_curve
   import matplotlib.pyplot as plt

   fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
   plt.plot(fpr, tpr, label='Courbe ROC')
   plt.xlabel('Taux de Faux Positifs')
   plt.ylabel('Taux de Vrais Positifs')
   plt.legend()
   plt.show()

Détection du Comportement de Fumer
==================================

### Prétraitement des données

Les images sont redimensionnées à \( 224 \times 224 \) et normalisées.

### Modèle CNN

L'architecture du modèle CNN est composée de :
- **Couches de convolution** : Pour extraire les caractéristiques visuelles.
- **Couches de pooling** : Pour réduire les dimensions.
- **Couches entièrement connectées** : Pour la classification finale.

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

Travaux Futurs
==============

1. Améliorer les modèles en utilisant plus de données.
2. Étendre la classification pour inclure d'autres comportements (vapoter, boire, etc.).
3. Intégrer les résultats à des systèmes IoT pour des alertes en temps réel.

Conclusion
==========

Ce projet démontre la puissance de **MediaPipe** et **TensorFlow** pour résoudre des problèmes critiques liés à la sécurité et au bien-être. L'intégration de ces outils offre une solution robuste et extensible.


Détection de la Fatigue et du Comportement de Fumer
===================================================

**Ce projet implémente des solutions basées sur MediaPipe et des modèles d'apprentissage automatique pour :**
1. Détecter la fatigue en analysant les mouvements des yeux et de la bouche.
2. Identifier automatiquement le comportement de fumer à partir de vidéos en temps réel.

Table des matières
------------------
.. toctree::
   :maxdepth: 2
   :caption: Contenu:

   installation_et_importation
   collecte_de_donnees
   detection_landmarks
   calcul_des_caracteristiques
   extraction_caracteristiques
   sauvegarde_donnees
   modelisation
   evaluation_performances
   visualisation_resultats
   prediction_fumer

Installation et Importation
---------------------------

Les bibliothèques suivantes sont nécessaires pour le projet :
- **os** : Manipulation des fichiers.
- **pickle** : Sauvegarde et chargement des données.
- **cv2** : Traitement d'images avec OpenCV.
- **numpy** : Calculs mathématiques.
- **mediapipe** : Détection des landmarks faciaux.
- **sklearn** : Modélisation en machine learning.

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

Collecte de Données
-------------------
Les images sont organisées en deux ensembles principaux :
- **Drowsy** : Images de personnes somnolentes.
- **Non Drowsy** : Images de personnes éveillées.

Chemins des fichiers et chargement des images :
.. code-block:: python

   drowsy_path = "path/to/drowsy_images"
   non_drowsy_path = "path/to/non_drowsy_images"

   drowsy_images = [cv2.imread(os.path.join(drowsy_path, img)) for img in os.listdir(drowsy_path)]
   non_drowsy_images = [cv2.imread(os.path.join(non_drowsy_path, img)) for img in os.listdir(non_drowsy_path)]

Détection des Landmarks Faciaux
-------------------------------
MediaPipe FaceMesh est utilisé pour extraire les points clés (landmarks) du visage.

.. code-block:: python

   mp_face_mesh = mp.solutions.face_mesh
   face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

Calcul des Caractéristiques
---------------------------
Deux caractéristiques principales sont calculées :
- **EAR (Eye Aspect Ratio)**
- **MAR (Mouth Aspect Ratio)**

.. code-block:: python

   def distance(p1, p2):
       return np.sqrt(np.sum((p1 - p2)**2))

   def eye_aspect_ratio(landmarks, eye):
       N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
       N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
       N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
       D = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
       return (N1 + N2 + N3) / (3 * D)

Extraction des Caractéristiques
-------------------------------
Pour chaque image, les valeurs EAR et MAR sont calculées et stockées dans des listes pour les deux catégories.

Sauvegarde des Données
----------------------
Les données extraites sont sauvegardées dans des fichiers pour une réutilisation ultérieure.

.. code-block:: python

   with open('drowsy_feats.pkl', 'wb') as file:
       pickle.dump(drowsy_features, file)

Modélisation
------------
Trois modèles sont utilisés :
- **MLP (Multi-layer Perceptron)**
- **SVM (Support Vector Machine)**
- **Random Forest**

Les caractéristiques EAR et MAR servent à l'entraînement.

Évaluation des Performances
---------------------------
Les métriques suivantes sont calculées :
- Accuracy
- Precision
- Recall
- F1-score

Visualisation des Résultats
---------------------------
Des courbes ROC et Precision-Recall sont tracées pour comparer les modèles.

Détection de Fumer
------------------
Un modèle CNN est développé pour identifier le comportement de fumer.

**Données** : Les images sont téléchargées depuis Kaggle et organisées en deux catégories :
- **Smoking**
- **Not Smoking**

**Résultats** : Le modèle atteint une précision de 70 %, indiquant une bonne performance.

**Code Exemple** :

.. code-block:: python

   from tensorflow.keras.models import load_model

   model = load_model("CNN_smoking_model.h5")
   pred = model.predict(video_frame)
   print("Fumer" if pred > 0.5 else "Non Fumer")
