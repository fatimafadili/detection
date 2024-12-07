====================================================================
Détection de la Fatigue par la Mediapipe et du Comportement de Fumer
====================================================================

Bienvenue dans la documentation du projet **Détection de la Fatigue et du Comportement de Fumer**. Ce document détaille les méthodologies, les outils utilisés, et les résultats obtenus pour détecter la fatigue à l'aide de MediaPipe et le comportement de fumer grâce à un modèle CNN.

**Table des matières**

  - introduction
  - installation
  - Détection de la Fatigue
  - Détection du Comportement de Fumer
  - Évaluation des Performances
  - Travaux Futurs
  - conclusion


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

Les bibliothèques suivantes sont nécessaires pour le projet :
  1. os : Manipulation des fichiers.
  2. pickle : Sauvegarde et chargement des données.
  3. cv2 : Traitement d'images avec OpenCV.
  4. numpy : Calculs mathématiques.
  5. mediapipe : Détection des landmarks faciaux.
  6. sklearn : Modélisation en machine learning.
  7. matplotlip : Créer des graphiques et visualiser des données
  8. tesorflow : Construire et entraîner des modèles de deep learning, comme les réseaux neuronaux.
  9. streamlit : Développer rapidement des applications web interactives pour partager des modèles et des analyses.

.. code-block:: python

   import os
   import pickle
   import cv2
   import numpy as np
   import mediapipe as mp
   import sklearn
   from sklearn.model_selection import train_test_split
   from sklearn.pipeline import make_pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.neural_network import MLPClassifier
   from sklearn.svm import SVC
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve
   import matplotlib
   import tensorflow as tf
   import streamlit as st

Détection de la Fatigue
=======================

1. **Collecte des données** :
- Télécharger et collecter le dataset depuis Kaggle en utilisant le site suivant : https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd    

- Organisation en deux dossiers :
     - **Drowsy** : Images de personnes somnolentes.
     - **Non Drowsy** : Images de personnes éveillées.

.. code-block:: python

    path = r"C:\Users\n\Desktop\projet ia\data1\FATIGUE"
    suffix ="phot"

exemple de data :

.. list-table::
   :widths: 50 50
   :align: center

   * - .. image:: image/A0100.png
         :alt: Image 1
         :width: 300px
     - .. image:: image/a0103.png
         :alt: Image 2
         :width: 300px

2. **Analyse des landmarks faciaux avec MediaPipe** :
   - Utilisation de **MediaPipe FaceMesh** pour extraire les points clés.

.. code-block:: python

   mp_face_mesh = mp.solutions.face_mesh
   face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.3, min_tracking_confidence=0.8)
   mp_drawing = mp.solutions.drawing_utils 
   drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

3. **Calcul des caractéristiques** :
   - EAR : Eye Aspect Ratio.
   - MAR : Mouth Aspect Ratio.
   
.. code-block:: python

  right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]] # right eye landmark positions
  left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]] # left eye landmark positions
  mouth = [[61, 291], [39, 181], [0, 17], [269, 405]] # mouth landmark coordinates

.. code-block:: python

  def distance(p1, p2):
      return (((p1[:2] - p2[:2])**2).sum())**0.5

  def eye_aspect_ratio(landmarks, eye):
      N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
      N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
      N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
      D = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
      return (N1 + N2 + N3) / (3 * D)

  def eye_feature(landmarks):
      return (eye_aspect_ratio(landmarks, left_eye) + eye_aspect_ratio(landmarks, right_eye)) / 2

  def mouth_feature(landmarks):
      N1 = distance(landmarks[mouth[1][0]], landmarks[mouth[1][1]])
      N2 = distance(landmarks[mouth[2][0]], landmarks[mouth[2][1]])
      N3 = distance(landmarks[mouth[3][0]], landmarks[mouth[3][1]])
      D = distance(landmarks[mouth[0][0]], landmarks[mouth[0][1]])
      return (N1 + N2 + N3) / (3 * D)

4. **Extraction et sauvegarde** :

pour les images somnolentes
===========================

Étape 1: extraction de caractéristiques
--------------------------------------
Le code suivant extrait les caractéristiques (ear et mar) des images somnolentes dans le jeu de données et les enregistre dans un fichier pickle :

.. code-block:: python

    drowsy_feats = [] 
    drowsy_path = os.path.join(path, "drowsy")

    # Check if directory exists
    if not os.path.exists(drowsy_path):
        print(f"Directory {drowsy_path} does not exist.")
    else:
        drowsy_list = os.listdir(drowsy_path)
        print(f"Total images in drowsy directory: {len(drowsy_list)}")

        for name in drowsy_list:
            image_path = os.path.join(drowsy_path, name)
            image = cv2.imread(image_path)
            
            # Check if image was loaded successfully
            if image is None:
                print(f"Could not read image {image_path}. Skipping.")
                continue

            # Flip and convert the image to RGB
            image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            
            # Process the image with face mesh
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                landmarks_positions = []
                # assume that only face is present in the image
                for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
                    landmarks_positions.append([data_point.x, data_point.y, data_point.z]) # saving normalized landmark positions
                landmarks_positions = np.array(landmarks_positions)
                landmarks_positions[:, 0] *= image.shape[1]
                landmarks_positions[:, 1] *= image.shape[0]

                ear = eye_feature(landmarks_positions)
                mar = mouth_feature(landmarks_positions)
                drowsy_feats.append((ear, mar))
            else:
                continue

        # Convert features list to numpy array and save to a file
        drowsy_feats = np.array(drowsy_feats)
        output_path = os.path.join("./feats", f"{suffix}_mp_drowsy_feats.pkl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "wb") as fp:
            pickle.dump(drowsy_feats, fp)

        print(f"Feature extraction complete. Saved to {output_path}")

Étape 2: Charger les caractéristiques extraites
----------------------------------------------

.. code-block:: python

    with open("./feats/phot_mp_drowsy_feats.pkl", "rb") as fp:
        drowsy_feats = pickle.load(fp)

pour les images non somnolentes
===============================     

Étape 1 : Extraction de caractéristiques
----------------------------------------

Le code suivant extrait les caractéristiques (`ear` et `mar`) des images non somnolentes dans le jeu de données et les enregistre dans un fichier pickle :

.. code-block:: python

    not_drowsy_feats = [] 
    not_drowsy_path = os.path.join(path, "notdrowsy")

    # Vérifier si le répertoire existe
    if not os.path.exists(not_drowsy_path):
        print(f"Le répertoire {not_drowsy_path} n'existe pas.")
    else:
        not_drowsy_list = os.listdir(not_drowsy_path)
        print(f"Total d'images dans le répertoire notdrowsy : {len(not_drowsy_list)}")

        for name in not_drowsy_list:
            image_path = os.path.join(not_drowsy_path, name)
            image = cv2.imread(image_path)
            
            # Vérifier si l'image a été chargée correctement
            if image is None:
                print(f"Impossible de lire l'image {image_path}. Passage à l'image suivante.")
                continue

            # Retourner et convertir l'image en RGB
            image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            
            # Traiter l'image avec le mesh du visage
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                landmarks_positions = []
                # Supposer qu'il n'y a qu'un seul visage dans l'image
                for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
                    landmarks_positions.append([data_point.x, data_point.y, data_point.z]) # Sauvegarder les positions des landmarks normalisées
                landmarks_positions = np.array(landmarks_positions)
                landmarks_positions[:, 0] *= image.shape[1]  # Mise à l'échelle des coordonnées x
                landmarks_positions[:, 1] *= image.shape[0]  # Mise à l'échelle des coordonnées y

                # Extraire les caractéristiques
                ear = eye_feature(landmarks_positions)
                mar = mouth_feature(landmarks_positions)
                not_drowsy_feats.append((ear, mar))
            else:
                continue

        # Convertir la liste de caractéristiques en un tableau numpy et l'enregistrer dans un fichier
        not_drowsy_feats = np.array(not_drowsy_feats)
        output_path = os.path.join("./feats", f"{suffix}_mp_not_drowsy_feats.pkl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "wb") as fp:
            pickle.dump(not_drowsy_feats, fp)

        print(f"L'extraction des caractéristiques est terminée. Sauvegardé dans {output_path}")

Étape 2 : Charger les caractéristiques extraites
------------------------------------------------

.. code-block:: python

    with open("./feats/phot_mp_not_drowsy_feats.pkl", "rb") as fp:
        non_drowsy_feats = pickle.load(fp)

5. **statistique de data** :

.. code-block:: python

   print(f"Drowsy Images: {drowsy_feats.shape[0]}")
   drowsy_ear = drowsy_feats[:, 0]
   print(f"EAR | Min, Median, Mean, Max, SD: [{drowsy_ear.min()}, {np.median(drowsy_ear)}, {drowsy_ear.mean()}, {drowsy_ear.max()}, {drowsy_ear.std()}]")
   drowsy_mar = drowsy_feats[:, 1]
   print(f"MAR | Min, Median, Mean, Max, SD: [{drowsy_mar.min()}, {np.median(drowsy_mar)}, {drowsy_mar.mean()}, {drowsy_mar.max()}, {drowsy_mar.std()}]")

Drowsy Images: 22348
EAR | Min, Median, Mean, Max, SD: [0.05643663213581103, 0.23440516640901327, 0.23769841002149675, 0.4788618089840052, 0.06175599084484693]
MAR | Min, Median, Mean, Max, SD: [0.1579104064072938, 0.27007593084743897, 0.29444085404221526, 0.852751604533097, 0.07479365878783618]

.. code-block:: python

   print(f"Non Drowsy Images: {non_drowsy_feats.shape[0]}")
   non_drowsy_ear = non_drowsy_feats[:, 0]
   print(f"EAR | Min, Median, Mean, Max, SD: [{non_drowsy_ear.min()}, {np.median(non_drowsy_ear)}, {non_drowsy_ear.mean()}, {non_drowsy_ear.max()}, {non_drowsy_ear.std()}]")
   non_drowsy_mar = non_drowsy_feats[:, 1]
   print(f"MAR | Min, Median, Mean, Max, SD: [{non_drowsy_mar.min()}, {np.median(non_drowsy_mar)}, {non_drowsy_mar.mean()}, {non_drowsy_mar.max()}, {non_drowsy_mar.std()}]")

Non Drowsy Images: 19445
EAR | Min, Median, Mean, Max, SD: [0.0960194509125116, 0.26370564454608236, 0.2704957278714779, 0.4394997191869294, 0.047188973064084226]
MAR | Min, Median, Mean, Max, SD: [0.139104718407629, 0.2955462164966127, 0.30543910382658035, 0.5770066727463391, 0.06818546886870354]

6. **Modélisation et entraînement** :

.. code-block:: python

    s = 192
    np.random.seed(s)
    random.seed(s)

    drowsy_labs = np.ones(drowsy_feats.shape[0])
    non_drowsy_labs = np.zeros(non_drowsy_feats.shape[0])

    X = np.vstack((drowsy_feats, non_drowsy_feats))
    y = np.concatenate((drowsy_labs, non_drowsy_labs))

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25, random_state=42)


Trois algorithmes de machine learning sont comparés :

1. SVM (Support Vector Machine).
.. code-block:: python

    svm = SVC(probability=True)
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    svm_probas = svm.predict_proba(X_test)

2. MLP (Multi-Layer Perceptron).
.. code-block:: python

    mlp = MLPClassifier(hidden_layer_sizes=(5, 3), random_state=1, max_iter=1000)
    mlp.fit(X_train, y_train)
    mlp_preds = mlp.predict(X_test)
    mlp_probas = mlp.predict_proba(X_test)

3. Random Forest.
 .. code-block:: python

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_probas = rf.predict_proba(X_test)

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


Évaluation et visualisation des Performances
============================================

pour fatigue 
------------

1. **Évaluation des Performances** :
Pour évaluer les performances des modèles de fatigue , les métriques suivantes sont calculées :
   - Accuracy : Mesure globale des prédictions correctes.
   - Precision : Précision des prédictions positives.
   - Recall : Capacité à détecter les exemples positifs.
   - F1-score : Moyenne harmonique entre précision et rappel.

.. code-block:: python

   print("Classifier: RF")
   preds = rf_preds
   print(f"Accuracy: {accuracy_score(y_test, preds)}")
   print(f"Precision: {precision_score(y_test, preds)}")
   print(f"Macro Precision: {precision_score(y_test, preds, average='macro')}")
   print(f"Recall: {recall_score(y_test, preds)}")
   print(f"Macro F1 score: {f1_score(y_test, preds, average='macro')}")

Classifier: RF
Accuracy: 0.6812135132548569
Precision: 0.7006515231554851
Macro Precision: 0.6793614009907405
Recall: 0.7092691622103386
Macro F1 score: 0.6791399140903065
 
.. code-block:: python

    print("Classifier: MLP")
    preds = mlp_preds
    print(f"Accuracy: {accuracy_score(y_test, preds)}")
    print(f"Precision: {precision_score(y_test, preds)}")
    print(f"Macro Precision: {precision_score(y_test, preds, average='macro')}")
    print(f"Recall: {recall_score(y_test, preds)}")
    print(f"Macro F1 score: {f1_score(y_test, preds, average='macro')}")

Classifier: MLP
Accuracy: 0.6342233706574791
Precision: 0.7178362573099415
Macro Precision: 0.6489890506407863
Recall: 0.5251336898395722
Macro F1 score: 0.632404526982427

.. code-block:: python

    print("Classifier: SVM")
    preds = svm_preds
    print(f"Accuracy: {accuracy_score(y_test, preds)}")
    print(f"Precision: {precision_score(y_test, preds)}")
    print(f"Macro Precision: {precision_score(y_test, preds, average='macro')}")
    print(f"Recall: {recall_score(y_test, preds)}")
    print(f"Macro F1 score: {f1_score(y_test, preds, average='macro')}")

print("Classifier: SVM")
preds = svm_preds
print(f"Accuracy: {accuracy_score(y_test, preds)}")
print(f"Precision: {precision_score(y_test, preds)}")
print(f"Macro Precision: {precision_score(y_test, preds, average='macro')}")
print(f"Recall: {recall_score(y_test, preds)}")
print(f"Macro F1 score: {f1_score(y_test, preds, average='macro')}")


2. **Visualisation des Résultats** :

Les visualisations incluent :
   - Courbes ROC : Représentent le compromis entre le rappel et le taux de faux positifs.
   - Courbes Precision-Recall : Mettent en évidence les performances globales.

.. code-block:: python

    plt.figure(figsize=(8, 6))
    plt.title("ROC Curve for the models")
    # mlp
    fpr, tpr, _ = roc_curve(y_test, mlp_probas[:, 1])
    auc = round(roc_auc_score(y_test, mlp_probas[:, 1]), 4)
    plt.plot(fpr, tpr, label="MLP, AUC="+str(auc))

    # svm
    fpr, tpr, _ = roc_curve(y_test, svm_probas[:, 1])
    auc = round(roc_auc_score(y_test, svm_probas[:, 1]), 4)
    plt.plot(fpr, tpr, label="SVM, AUC="+str(auc))

    # RF
    fpr, tpr, _ = roc_curve(y_test, rf_probas[:, 1])
    auc = round(roc_auc_score(y_test, rf_probas[:, 1]), 4)
    plt.plot(fpr, tpr, label="RF, AUC="+str(auc))

    plt.plot(fpr, fpr, '--', label="No skill")
    plt.legend()
    plt.xlabel('True Positive Rate (TPR)')
    plt.ylabel('False Positive Rate (FPR)')
    plt.show()

.. image:: /image/1.png
   :alt: Texte alternatif pour l'image
   :width: 400px
   :align: center

.. code-block:: python

    plt.figure(figsize=(8, 6))
    plt.title("Precision-Recall Curve for the models")

    # mlp
    y, x, _ = precision_recall_curve(y_test, mlp_probas[:, 1])
    plt.plot(x, y, label="MLP")

    # svm
    y, x, _ = precision_recall_curve(y_test, svm_probas[:, 1])
    plt.plot(x, y, label="SVM")

    # RF
    y, x, _ = precision_recall_curve(y_test, rf_probas[:, 1])
    plt.plot(x, y, label="RF")

    plt.legend()
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.show()

.. image:: /image/2.png
   :alt: Texte alternatif pour l'image
   :width: 400px
   :align: center


.. code-block:: python

    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    import numpy as np

    def main():
        # Simuler des données fictives pour y_test et les probabilités des modèles
        np.random.seed(42)
        y_test = np.random.randint(0, 2, 100)  # Labels binaires
        mlp_probas = np.random.rand(100, 2)    # Probabilités du modèle MLP
        svm_probas = np.random.rand(100, 2)    # Probabilités du modèle SVM
        rf_probas = np.random.rand(100, 2)     # Probabilités du modèle RF

        # Tracer la courbe Precision-Recall
        plt.figure(figsize=(8, 6))
        plt.title("Precision-Recall Curve for the models")

        # MLP
        y, x, _ = precision_recall_curve(y_test, mlp_probas[:, 1])
        plt.plot(x, y, label="MLP")

        # SVM
        y, x, _ = precision_recall_curve(y_test, svm_probas[:, 1])
        plt.plot(x, y, label="SVM")

        # RF
        y, x, _ = precision_recall_curve(y_test, rf_probas[:, 1])
        plt.plot(x, y, label="RF")

        # Ajout des légendes et labels
        plt.legend()
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()

    if __name__ == "__main__":
        main()

.. image:: /image/3.png
   :alt: Texte alternatif pour l'image
   :width: 400px
   :align: center


Travaux Futurs
==============

1. Améliorer les modèles en utilisant plus de données.
2. Étendre la classification pour inclure d'autres comportements (vapoter, boire, etc.).
3. Intégrer les résultats à des systèmes IoT pour des alertes en temps réel.

Conclusion
==========

Ce projet démontre la puissance de **MediaPipe** et **TensorFlow** pour résoudre des problèmes critiques liés à la sécurité et au bien-être. L'intégration de ces outils offre une solution robuste et extensible.

