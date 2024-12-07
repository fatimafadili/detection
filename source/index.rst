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



.. code-block:: python

   # This is a Python example
   def greet(name):
       return f"Hello, {name}!"


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

Introduction
============

Ce projet repose sur un modèle de détection de fatigue basé sur MediaPipe et des algorithmes d'apprentissage automatique. Il surveille les mouvements des yeux et de la bouche en temps réel, en utilisant des ratios spécifiques comme **EAR** (Eye Aspect Ratio) et **MAR** (Mouth Aspect Ratio).

Exemple de Code
===============

Le code suivant implémente la détection de fatigue en utilisant OpenCV, MediaPipe et un modèle SVM :

.. code-block:: python
   :linenos:
   :emphasize-lines: 6,23

   import cv2
   import mediapipe as mp
   import numpy as np
   import pickle
   import pygame
   import time

   # Charger les modèles entraînés
   with open("./models/svm_model.pkl", "rb") as svm_file:
       loaded_svm = pickle.load(svm_file)
   print("Modèle chargé avec succès.")

   # Initialisation
   pygame.init()
   mp_face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

   # Fonction pour calculer EAR (Eye Aspect Ratio)
   def eye_aspect_ratio(landmarks, eye_indices):
       def distance(p1, p2):
           return np.linalg.norm(p1 - p2)
       N = distance(landmarks[eye_indices[1]], landmarks[eye_indices[2]])
       D = distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
       return N / D

   # Démarrer la capture vidéo
   cap = cv2.VideoCapture(0)
   while cap.isOpened():
       ret, frame = cap.read()
       if not ret:
           break

       # Conversion des couleurs et détection
       image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       results = mp_face_mesh.process(image)

       # Affichage des résultats
       if results.multi_face_landmarks:
           for face_landmarks in results.multi_face_landmarks:
               print(face_landmarks)  # Debug: Affiche les landmarks détectés

       cv2.imshow("Détection de fatigue", frame)

       if cv2.waitKey(5) & 0xFF == ord('q'):
           break

   cap.release()
   cv2.destroyAllWindows()



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


   Détection de Fatigue et Comportement de Fumer
=============================================

Bienvenue dans la documentation du projet. Ce document décrit en détail les étapes, les méthodologies, et les résultats pour la détection de la fatigue via MediaPipe et la détection du comportement de fumer à l'aide d'un modèle CNN. 

Sommaire
--------

.. toctree::
   :maxdepth: 2
   :caption: Contenu Principal

   introduction
   technologies_utilisees
   fatigue_detection
   fatigue_model_training
   smoking_behavior_detection
   cnn_model_training
   real_time_application
   challenges_and_future_work
   conclusion


Introduction
============

Ce projet aborde deux problématiques essentielles :
1. Détection de la fatigue en temps réel pour prévenir les accidents, en particulier pour les conducteurs.
2. Identification automatique du comportement de fumer, qui peut être utilisé pour des études de santé publique ou des systèmes de surveillance.

Objectifs principaux :
- Exploiter les capacités de **MediaPipe** pour extraire des landmarks faciaux en temps réel.
- Développer un modèle **CNN** pour détecter le comportement de fumer avec une précision élevée.
- Créer une interface utilisateur interactive pour démontrer ces capacités.

---

Technologies Utilisées
======================

1. **MediaPipe** :
   - Pour l'extraction des landmarks faciaux (points clés du visage).
   - Modules utilisés : **FaceMesh** et **Hands**.

2. **TensorFlow/Keras** :
   - Création et entraînement du modèle CNN.
   - Sauvegarde et exportation du modèle pour une utilisation en production.

3. **OpenCV** :
   - Manipulation des vidéos et des images pour le traitement en temps réel.

4. **Streamlit** :
   - Développement d'une interface utilisateur simple et intuitive.

5. **Langages** : Python (principalement).

---

Détection de Fatigue
====================

### Méthodologie

1. **Extraction des Landmarks** :
   - Utilisation de **MediaPipe FaceMesh** pour capturer des points clés tels que les yeux, la bouche, et le visage entier.

2. **Calculs Spécifiques** :
   - **Eye Aspect Ratio (EAR)** :
     - \( EAR = \frac{{\text{Distance verticale des points des yeux}}}{{\text{Distance horizontale}}} \)
     - Indicateur de clignement ou d'endormissement.
   - **Mouth Aspect Ratio (MAR)** :
     - Mesure l'ouverture de la bouche pour détecter les bâillements.

3. **Extraction des Caractéristiques** :
   - Les données brutes des landmarks sont transformées en caractéristiques exploitables (EAR, MAR).

4. **Collecte et Prétraitement des Données** :
   - Base de données vidéo annotée : états somnolents (fatigue) et éveillés.
   - Division en ensembles d'entraînement, de validation et de test.

### Implémentation du Modèle
- Comparaison des algorithmes : **SVM**, **MLP**, et **Random Forest**.
- Métriques d'évaluation :
  - Précision
  - Rappel
  - F1-Score
  - Courbe ROC

### Détection en Temps Réel
- Intégration avec OpenCV pour la capture vidéo.
- Alarme sonore lorsque des signes de fatigue sont détectés.

---

Entraînement des Modèles de Fatigue
===================================

1. **Préparation des Données** :
   - Normalisation des valeurs EAR et MAR.
   - Augmentation des données pour équilibrer les classes.

2. **Comparaison des Modèles** :
   - SVM : performant mais sensible au bruit.
   - MLP : meilleure généralisation.
   - Random Forest : rapide, mais moins précis.

3. **Optimisation des Hyperparamètres** :
   - Recherche par grille pour optimiser les paramètres comme le nombre de neurones et les couches cachées.

---

Détection du Comportement de Fumer
==================================

### Collecte et Prétraitement des Données

1. **Données Utilisées** :
   - Ensemble d'images annotées de fumeurs et de non-fumeurs.
   - Provenance : Kaggle ou base de données interne.

2. **Prétraitement** :
   - Redimensionnement des images à \( 224 \times 224 \).
   - Normalisation des pixels entre 0 et 1.

### Modèle CNN pour la Détection

1. **Architecture** :
   - Convolution : extraction des caractéristiques.
   - Pooling : réduction dimensionnelle.
   - Fully Connected Layers : classification finale.

2. **Formation du Modèle** :
   - Fonction de perte : cross-entropie.
   - Optimiseur : Adam.
   - Taux d'apprentissage : \( 10^{-3} \).

3. **Résultats** :
   - Précision : 70%.
   - Courbe ROC et matrice de confusion.

---

Application en Temps Réel
=========================

1. **Détection de la Fatigue** :
   - Interface avec caméra.
   - Notification sonore et visuelle en cas de détection.

2. **Détection du Comportement de Fumer** :
   - Chargement du modèle CNN via TensorFlow.
   - Prédictions en direct sur des vidéos.

---

Défis et Travaux Futurs
=======================

1. **Défis Actuels** :
   - Variabilité des visages dans des conditions d'éclairage différentes.
   - Faux positifs dans la détection de fatigue.

2. **Travaux Futurs** :
   - Amélioration des modèles CNN avec des données supplémentaires.
   - Détection multiclasses : fumer, vapoter, boire, etc.
   - Intégration avec des dispositifs IoT pour des alertes instantanées.

---

Conclusion
==========

Ce projet montre comment des outils modernes de vision par ordinateur et d'apprentissage automatique peuvent être utilisés pour répondre à des problèmes critiques. L'intégration de technologies comme MediaPipe et TensorFlow offre des solutions robustes et extensibles.

---

Indices
=======

Pour approfondir chaque module, consultez les sections dédiées dans ce guide.
