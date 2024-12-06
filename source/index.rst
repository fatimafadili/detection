=========================
Détection de la Fatigue
=========================

Bienvenue dans la documentation de notre projet de détection de fatigue basé sur MediaPipe. 
Ce projet a pour objectif de détecter les signes de fatigue en temps réel grâce à l'analyse des indices 
EAR (Eye Aspect Ratio) et MAR (Mouth Aspect Ratio).

.. contents:: Table des matières
   :local:

Introduction
============
La détection de la fatigue est un défi critique dans plusieurs domaines, tels que la sécurité routière et 
la prévention des accidents au travail. Nous utilisons **MediaPipe** pour extraire les caractéristiques faciales 
et un modèle SVM (Support Vector Machine) pour classifier les signes de fatigue en fonction des ratios EAR et MAR.

Structure du projet
===================
Le projet est organisé comme suit :

- **feats/** : Contient les fichiers de caractéristiques (EAR et MAR) extraites des données d'entraînement.
- **models/** : Stocke les modèles d'apprentissage automatique pré-entraînés (comme SVM).
- **source/** : Inclut le code source pour l'extraction des caractéristiques et l'entraînement des modèles.
- **CNN_smoking_model.h5** : Modèle CNN pour d'autres tâches (par exemple, détection de la cigarette).
- **app.py** : Application principale Streamlit pour la détection en temps réel.
- **alert.mp3** : Alerte sonore déclenchée en cas de fatigue détectée.
- **building_model.ipynb** : Notebook Jupyter contenant le pipeline d'entraînement des modèles.
- **df_train_images_file.csv** : Fichier CSV avec les données d'entraînement (images et annotations).

Étapes du projet
================

1. **Collecte des données**
   Nous avons collecté des données faciales, notamment les coordonnées des landmarks pour les yeux et la bouche, 
   à l'aide de MediaPipe. Ces données ont été utilisées pour calculer les indices suivants :
   
   - **EAR (Eye Aspect Ratio)** : Mesure basée sur les distances entre les points des yeux.
   - **MAR (Mouth Aspect Ratio)** : Mesure basée sur les distances entre les points de la bouche.

2. **Prétraitement**
   Les caractéristiques EAR et MAR ont été extraites et sauvegardées dans des fichiers sous le dossier `feats/`.

3. **Entraînement du modèle**
   Un modèle SVM a été entraîné à l'aide des caractéristiques pour classifier lesgit états de fatigue (fatigué ou non-fatigué). 
   Le modèle entraîné est sauvegardé dans le dossier `models/`.

4. **Détection en temps réel**
   L'application principale (`app.py`) utilise une webcam pour détecter les visages, extraire les ratios EAR et MAR 
   en temps réel, et déclencher une alerte sonore si un état de fatigue est détecté.

Comment utiliser le projet
==========================

1. **Installation des dépendances**
   Installez les bibliothèques nécessaires :
   ```bash
   pip install streamlit opencv-python mediapipe pygame numpy



