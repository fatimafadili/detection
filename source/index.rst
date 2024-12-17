============================================================================
Détection de la fatigue par Mediapipe et analyse du comportement de la fumée
============================================================================

Bienvenue dans la documentation du projet **Détection de la fatigue par Mediapipe et analyse du comportement de la fumée**. Ce document détaille les méthodologies, les outils utilisés, et les résultats obtenus pour détecter la fatigue à l'aide de MediaPipe et le comportement de fumee grâce à un modèle CNN.

**Table des matières**

  - introduction
  - installation
  - Détection de la Fatigue
  - Détection du Comportement de Fumer
  - Évaluation et visualisation des Performances
  - test des models  
  - creation de l'application streamlit  
  - Travaux Futurs
  - conclusion


Introduction
============

La détection de la fatigue et des comportements dangereux comme fumer est un enjeu majeur pour la sécurité et la santé. Ce projet vise à fournir une solution robuste en temps réel à l'aide d'outils modernes de vision par ordinateur et de machine learning.

- **Fatigue** : Détection basée sur l'analyse des mouvements des yeux et de la bouche via **MediaPipe**.
- **Fumee** : Classification à l'aide d'un modèle **CNN** entraîné sur des images annotées.

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
  10. pygame :pour generer les audios alertes.

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
==========================

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
         :width: 500px
     - .. image:: image/a0103.png
         :alt: Image 2
         :width: 500px
__________________________somnolent_____________________________________________________________non somnolent______________________________
                          =========                                                             =============
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

4-1 pour les images somnolentes:

Étape 1: extraction de caractéristiques:

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

.. code-block:: python

    with open("./feats/phot_mp_drowsy_feats.pkl", "rb") as fp:
        drowsy_feats = pickle.load(fp)

4-2 pour les images non somnolentes :    

Étape 1 : Extraction de caractéristiques

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

Détection du Comportement de Fumee
==================================
1. **Collecte des données** :
- Télécharger et collecter le dataset depuis Kaggle par la combinaison de plusieurs datasets.  

- Organisation en deux dossiers :
     - **smoking** : Images de personnes qui fument .
     - **Nonsmoking** : Images de personnes qui ne fument pas.
.. code-block:: python

    import tensorflow
    import os

    # Chemin vers le répertoire dans lequel vous voulez organiser les données
    datasets_dir = r"C:\Users\n\Desktop\projet ia\data2"

exemple de data :

.. list-table::
   :widths: 50 50
   :align: center

   * - .. image:: image/notsmoking_0941.jpg
         :alt: Image 1
         :width: 700px
     - .. image:: image/smok64.jpg
         :alt: Image 2
         :width: 500px
__________________________Non-smoking_____________________________________________________________smoking______________________________
                          ===========                                                             =======

2. **Repartition de donnees** :
-on repartie datasets entre les ensembles d'entraînement et de validation:

.. code-block:: python

    import os
    import shutil
    import random

    # Chemin source où les images sont décompressées
    source_dir = r"C:\Users\n\Desktop\projet ia\data2"

    # Chemins pour les ensembles d'entraînement et de validation
    train_dir = r'C:\Users\n\Desktop\projet ia\data2\train'
    val_dir = r'C:\Users\n\Desktop\projet ia\data2\val'

    # Créer les répertoires s'ils n'existent pas déjà
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Liste des classes
    classes = ['notsmoking', 'smoking']

    # Fonction pour répartir les images en ensembles d'entraînement et de validation
    def split_data(source_dir, train_dir, val_dir, split_ratio=0.8):
        for class_name in classes:
            # Créer des sous-dossiers pour chaque classe dans train et val
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
            # Liste des images dans chaque classe
            class_dir = os.path.join(source_dir, class_name)
            images = os.listdir(class_dir)
            random.shuffle(images)  # Mélanger les images

            # Calcul du nombre d'images pour l'entraînement
            train_size = int(len(images) * split_ratio)
            train_images = images[:train_size]
            val_images = images[train_size:]

            # Déplacer les images dans les dossiers train et val correspondants
            for img in train_images:
                 shutil.move(os.path.join(class_dir, img), os.path.join(train_dir, class_name, img))

            for img in val_images:
                shutil.move(os.path.join(class_dir, img), os.path.join(val_dir, class_name, img))

    # Appel de la fonction pour organiser les images
    split_data(source_dir, train_dir, val_dir)
    print("Images réparties entre les ensembles d'entraînement et de validation.")


3. **Normalisation de donnees** :

.. code-block:: python

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Création des générateurs d'images pour l'entraînement et la validation
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Normalisation des pixels
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    # Chargement des images depuis les dossiers train et val
    train_generator = train_datagen.flow_from_directory(
        r'C:\Users\n\Desktop\projet ia\data2\train',
        target_size=(150, 150),  # Taille de redimensionnement des images
        batch_size=32,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        r'C:\Users\n\Desktop\projet ia\data2\val',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

4. **Constuction du model CNN** :

.. code-block:: python

    from tensorflow.keras import layers, models

    # Définir le modèle CNN
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))  

    # Compiler le modèle
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

5. **Entrainement du modele** :

.. code-block:: python

    # Entraînement du modèle
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=30,
        validation_data=val_generator,
        validation_steps=val_generator.samples // val_generator.batch_size
    )

6. **Sauvegarde du modele** :

.. code-block:: python

    model_save_path = r'C:\Users\n\Desktop\projet ia\data2\cnn_model_SMOKING.h5'
    model.save(model_save_path)
    print(f"Modèle sauvegardé à : {model_save_path}")


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

Classifier: SVM
Accuracy: 0.690879510000957
Precision: 0.7048898071625345
Macro Precision: 0.6891180343720451
Recall: 0.7297682709447415
Macro F1 score: 0.688198015126017


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

- .. image:: image/1.png
         :alt: Image 1
         :width: 500px

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

- .. image:: image/2.png
         :alt: Image 1
         :width: 500px


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

- .. image:: image/3.png
         :alt: Image 1
         :width: 500px

pour la fumee:
--------------
1. **Évaluation des Performances** :

.. code-block:: python

    # Évaluation des performances sur l'ensemble de validation
    val_loss, val_accuracy = model.evaluate(val_generator)
    print(f"Perte de validation : {val_loss}")
    print(f"Précision de validation : {val_accuracy}")

43/43 [==============================] - 120s 4s/step - loss: 0.4912 - accuracy: 0.770

Perte de validation : 0.4912235140800476

Précision de validation : 0.7709565010070801

.. code-block:: python

    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Prédire les classes pour l'ensemble de validation
    val_generator.reset()  # Réinitialiser le générateur
    predictions = model.predict(val_generator, steps=val_generator.samples // val_generator.batch_size + 1)
    predicted_classes = np.argmax(predictions, axis=1)

    # Obtenir les vraies classes
    true_classes = val_generator.classes
    class_labels = list(val_generator.class_indices.keys())  # Labels de classes

    # Générer le rapport de classification
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)

    # Matrice de confusion
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Matrice de Confusion")
    plt.ylabel('Vraies classes')
    plt.xlabel('Classes prédites')
    plt.show()


- .. image:: image/61.png
         :alt: Image 1
         :width: 500px

- .. image:: image/6.png
         :alt: Image 1
         :width: 500px

2. **Visualisation des Résultats** :

.. code-block:: python

    import matplotlib.pyplot as plt

    # Visualiser la précision d'entraînement et de validation
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # Précision
    plt.plot(epochs, acc, 'bo', label='Précision Entraînement')
    plt.plot(epochs, val_acc, 'b', label='Précision Validation')
    plt.title('Précision Entraînement et Validation')
    plt.legend()
    plt.figure()

    # Perte
    plt.plot(epochs, loss, 'bo', label='Perte Entraînement')
    plt.plot(epochs, val_loss, 'b', label='Perte Validation')
    plt.title('Perte Entraînement et Validation')
    plt.legend()
    plt.show()


- .. image:: image/4.png
         :alt: Image 1
         :width: 500px

- .. image:: image/5.png
         :alt: Image 1
         :width: 500px



test des models 
===============

pour les models SVM,MLP,RF :

1. **Créer un répertoire pour sauvegarder les modèles**:

.. code-block:: python

    import os
    os.makedirs("./models", exist_ok=True)

    # Sauvegarder le modèle Random Forest
    with open("./models/rf_model.pkl", "wb") as rf_file:
    pickle.dump(rf, rf_file)

    # Sauvegarder le modèle SVM
    with open("./models/svm_model.pkl", "wb") as svm_file:
    pickle.dump(svm, svm_file)

    # Sauvegarder le modèle MLP
    with open("./models/mlp_model.pkl", "wb") as mlp_file:
    pickle.dump(mlp, mlp_file)

    print("Modèles sauvegardés avec succès dans le dossier './models'.")


2. **test des modeles  de Fatigue (rf , svm, mlp)**:

Le code ci-dessous utilise OpenCV, MediaPipe et un modèle SVM pour détecter la fatigue en surveillant les expressions faciales, telles que les mouvements des yeux et de la bouche, dans un flux vidéo en temps réel. Si la fatigue est détectée, une alerte sonore est  apres 3 seconde de detection succesive de fatigue .
pour changer le modele il faut juste remplacer svm par rf ou mlp , on peut regler aussi Temps en secondes avant déclenchement de l'alarme.

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


voici quelque exemple d'affichage par OPENCV et mediapipe:

- .. image:: image/7.png
         :alt: Image 1
         :width: 500px

- .. image:: image/8.png
         :alt: Image 1
         :width: 500px

- .. image:: image/9.png
         :alt: Image 1
         :width: 500px

pour le model CNN de fumee :

on teste ce model par interface streamlit , on le teste maintenant par des images ne trouvent pas en datasets  mais apres en application finale , on teste tous les modules par un video reel grace a open cv

.. code-block:: python

    import streamlit as st
    import tensorflow as tf
    from PIL import Image, ImageOps
    import numpy as np

    # Charger le modèle avec mise en cache pour éviter de le recharger à chaque fois
    @st.cache(allow_output_mutation=True)
    def load_model():
        model = tf.keras.models.load_model(r"C:\Users\n\Desktop\projet ia\data2\cnn_model_SMOKING.h5")
        return model

    model = load_model()

    # Message de bienvenue au début
    st.markdown("## Bienvenue dans le test de la presence de fumee !")
    st.write("Grâce à ce test, vous pouvez télécharger une image et nous vous indiquerons de quelle catégorie elle fait partie en utilisant un modèle d'intelligence artificielle.")

    # Titre de test
    st.title("Classification d'images - smokers or no")

    # Option pour choisir une fonctionnalité via la barre latérale
    option = st.sidebar.selectbox(
        "Qu'est-ce que vous voulez faire?",
        ("Classification d'images")
    )

    # Affichage de l'option sélectionnée
    if option is not None:
        st.sidebar.write("Vous avez sélectionné:", option)

    # Si l'option Classification d'images est sélectionnée
    if option == "Classification d'images":
        # Instructions
        st.markdown("### Veuillez télécharger une image (formats acceptés : .jpg, .png)")

        # Uploader pour sélectionner une image
        file = st.file_uploader("Téléchargez une image", type=["jpg", "png"])

        # Fonction pour traiter et prédire la classe de l'image
        def import_and_predict(image_data, model):
            try:
                size = (150, 150)
                image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)  # Redimensionner avec anti-aliasing
                img = np.asarray(image) / 255.0  # Normaliser les données de l'image
                img_reshape = img[np.newaxis, ...]  # Ajouter une dimension batch pour le modèle
                prediction = model.predict(img_reshape)
                return prediction
            except Exception as e:
                st.error(f"Erreur lors du traitement de l'image : {e}")
                return None

        # Noms des classes
        class_names = ['notsmoking', 'smoking']

        # Vérification si un fichier a été téléchargé
        if file is None:
            st.text("Veuillez télécharger une image pour continuer.")
        else:
            # Afficher l'image téléchargée
            image = Image.open(file)
            st.image(image, caption="Image téléchargée", use_column_width=True)

            # Ajouter un bouton pour déclencher la classification
            if st.button("Classifier l'image"):
                with st.spinner('Classification en cours...'):
                    predictions = import_and_predict(image, model)

                    if predictions is not None:
                        # Obtenir la classe prédite et le score de confiance
                        predicted_class = class_names[np.argmax(predictions)]
                        confidence = np.max(predictions) * 100

                        # Afficher le résultat avec le score de confiance
                        st.success(f"*L'image est probablement de la classe : {predicted_class}*")
                        st.write(f"Confiance de la prédiction : {confidence:.2f}%")
                    else:
                        st.error("La classification a échoué.")
        
            # Remerciements à la fin
            st.markdown("---")
            st.markdown("### Merci d'avoir utilisé notre test de classification d'images !")
            st.write("Nous espérons que cela vous a été utile. À bientôt !")



pour l'execution de ce test de smoking il faut taper en terminal streamlit run testsmoking.py


-voici exemples dans l'interface streamlit de quelques images pour tester la presence de fumee:

.. list-table::
   :widths: 150 150
   :align: center

   * - .. image:: image/10.png
         :alt: Image 1
         :width: 600px
     - .. image:: image/11.png
         :alt: Image 2
         :width: 600px

.. list-table::
   :widths: 150 150
   :align: center

   * - .. image:: image/12.png
         :alt: Image 1
         :width: 600px
     - .. image:: image/13.png
         :alt: Image 2
         :width: 600px


creation de l'application streamlit  
===================================

La génération d'une application Streamlit (par un fichier python app.py ) qui effectue la détection de la fatigue par MAR, EAR et la fumée en temps réel. Lorsqu'un de ces signes est détecté, l'application émet des alertes sonores

.. code-block:: python

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

pour l'execution de cette application il faut taper en terminal streamlit run app.py



Travaux Futurs
==============

1. Améliorer les modèles en utilisant plus de données.
2. Étendre la classification pour inclure d'autres comportements (vapoter, boire, etc.).

Conclusion
==========

Ce projet démontre la puissance de **MediaPipe** et **TensorFlow** pour résoudre des problèmes critiques liés à la sécurité et au bien-être. L'intégration de ces outils offre une solution robuste et extensible.

