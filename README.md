# Spotify Audio Analysis

Projet académique portant sur l'analyse et la classification de morceaux musicaux à partir de leurs caractéristiques audio issues de Spotify. Il couvre deux axes principaux : la classification de genres et la prédiction de popularité, ainsi qu'un système d'identification audio par empreinte numérique.

---

## Partie 1 — Machine Learning sur features audio

### Exercice 1 : Classification du genre musical

Prédiction du genre d'un morceau (pop, rock, country…) à partir de ses features audio numériques (danceability, energy, tempo, loudness, etc.).

Cinq modèles comparés via F1-score micro :

| Modèle | Remarque |
|---|---|
| K-Nearest Neighbors | Baseline simple |
| Logistic Regression | Modèle linéaire multiclasse |
| SVM (kernel RBF) | Bon sur espaces de grande dimension |
| Random Forest | **Meilleur F1-score → modèle retenu** |
| XGBoost | Gradient boosting avec encodage des labels |

### Exercice 2 : Prédiction de la popularité

Problème de régression sur la variable `popularity` (0–100). Modèles testés : régression linéaire et Random Forest Regressor, évalués par MSE et R².

---

## Partie 2 — Système d'identification audio (Shazam-like)

Implémentation d'un moteur de reconnaissance musicale basé sur le fingerprinting audio.

**Pipeline :**
1. Calcul du spectrogramme via STFT (librosa, sr=3000 Hz)
2. Extraction des maxima locaux (points d'intérêt)
3. Génération de hashes par paires ancre / target zone
4. Construction d'une base de données de signatures
5. Identification d'un extrait de 10 secondes par corrélation temporelle des hashes

---

## Partie 3 — Prédiction conforme (Conformal Prediction)

Quantification de l'incertitude du classifieur KNN via la prédiction conforme. Génération d'intervalles de confiance garantissant que le vrai genre appartient à l'ensemble prédit avec une probabilité fixée (ex : 95 %).

---

## Stack

Python · scikit-learn · XGBoost · librosa · pandas · matplotlib · seaborn · pickle

---

## Structure

```
├── GHAZI__TAREK.ipynb   # Notebook principal (toutes les parties)
└── GHAZI_TAREK.csv      # Prédictions finales (soumission exercice 1)
```
