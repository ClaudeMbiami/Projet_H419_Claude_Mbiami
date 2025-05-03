# Projet_H419_Claude_Mbiami
Classification Thorax Pulmonaire par CNN
1) Contexte et Objectifs
Ce projet développe un outil de diagnostic automatisé à partir de radiographies thoraciques, permettant de classer les images selon :

COVID-19
Infection virale non-COVID
Poumons normaux

L’objectif est d’assister les professionnels de santé, en particulier dans les contextes de pénurie de tests ou zones sous-équipées, grâce à l’intelligence artificielle et aux réseaux de neurones convolutifs (CNN).

Précision atteinte sur test : 88,5%

 2) Architecture du Modèle
L’architecture du CNN est la suivante :
Input (224x224x1)
   ↓
Conv2D (32 filtres, 3x3) + ReLU
   ↓
MaxPooling2D (2x2)
   ↓
Conv2D (64 filtres, 3x3) + ReLU
   ↓
MaxPooling2D (2x2)
   ↓
Conv2D (128 filtres, 3x3) + ReLU
   ↓
MaxPooling2D (2x2)
   ↓
Flatten
   ↓
Dense (256) + ReLU
   ↓
Dropout (0.4)
   ↓
Dense (3) + Softmax
Fonction de perte : categorical_crossentropy
Optimiseur : Adam
EarlyStopping basé sur val_loss

3) Jeu de Données
Le dataset provient de Kaggle - COVID-19 Chest X-ray :

Ensemble	COVID-19	Virus non-COVID	Normal	Total
Train	236	346	374	956
Validation	29	43	46	118
Test	30	44	48	122

Images prétraitées : redimension 224x224, niveaux de gris, normalisation [0,1]

4) Augmentation des Données
Utilisation de ImageDataGenerator (Keras) :

rotation_range=30

zoom_range=0.2

horizontal_flip=True

brightness_range=[0.8,1.2]

shear_range=0.2

width_shift_range=0.2

height_shift_range=0.2

5) Résultats
Précision globale sur test : 88,5%
Perte sur test : 0.4242

Classe	Précision	Rappel	F1-score
COVID-19	1.00	0.87	0.93
Normal	0.83	0.94	0.88
Virus	0.88	0.84	0.86

6) Environnement et Librairies
Python 3.x

TensorFlow / Keras

Scikit-learn

Matplotlib

 Installation
pip install tensorflow scikit-learn matplotlib
Utilisation

Cloner le dépôt :
git clone https://github.com/ClaudeMbiami/Projet_H419_Claude_Mbiami/blob/main/projet-h-419-claude-mbiami.ipynb
cd REPO_NAME

Placer les données :
data/
 ├── train/
 ├── validation/
 └── test/
 
Entraîner le modèle :
python train_model.py

Prédire une image :
python predict.py --image path/to/image.png

7) Limites et Améliorations Futures
Jeu de test limité (122 images) → risque de biais

Confusion entre Virus non-COVID et Normal

Perspectives :

Transfer Learning (VGG16, ResNet, EfficientNet…)

Validation croisée

Plus de données

Interface web (Flask, Django, Streamlit)

- Références
Goodfellow, I., Bengio, Y., Courville, A. (2016). Deep Learning.

TensorFlow/Keras, Scikit-learn, Matplotlib.

Dataset Kaggle

Contenu du dépôt
train_model.py : Script d’entraînement

predict.py : Script de prédiction

best_model.keras : Modèle sauvegardé

README.md : Ce fichier

requirements.txt : Dépendances

notebooks/demo.ipynb : Notebook interactif

 Projet réalisé par MBIAMI NJANANG Claude — Avril 2025
