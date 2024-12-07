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

.. toctree::
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

**4. Extraction des caractéristiques**  
- Les valeurs EAR et MAR sont calculées pour chaque image et stockées pour l’entraînement des modèles.

**5. Modélisation**  
- Trois modèles de machine learning sont utilisés :
  - **MLP (Multi-layer Perceptron)**.
  - **SVM (Support Vector Machine)**.
  - **Random Forest**.
- Les données sont divisées en ensembles d’entraînement et de test.

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
