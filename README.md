# Spotify Audio Analysis

Academic project on the analysis and classification of music tracks from their Spotify audio features. It covers two main areas: genre classification and popularity prediction, plus an audio identification system based on digital fingerprinting.

---

## Part 1: Machine learning on audio features

### Exercise 1: Music genre classification

Predicting a track's genre (pop, rock, country...) from its numeric audio features (danceability, energy, tempo, loudness, etc.).

Five models compared via micro F1-score:

| Model | Note |
|---|---|
| K-Nearest Neighbors | Simple baseline |
| Logistic Regression | Multiclass linear model |
| SVM (RBF kernel) | Strong on high-dimensional spaces |
| Random Forest | **Best F1-score, selected model** |
| XGBoost | Gradient boosting with label encoding |

### Exercise 2: Popularity prediction

Regression problem on the `popularity` variable (0 to 100). Models tested: linear regression and Random Forest Regressor, evaluated with MSE and R².

---

## Part 2: Audio identification system (Shazam-like)

Implementation of a music recognition engine based on audio fingerprinting.

**Pipeline:**
1. Spectrogram computation via STFT (librosa, sr=3000 Hz)
2. Local maxima extraction (points of interest)
3. Hash generation from anchor / target zone pairs
4. Building a signature database
5. Identifying a 10-second excerpt through temporal correlation of hashes

---

## Part 3: Conformal prediction

Quantifying the uncertainty of the KNN classifier through conformal prediction. Generating confidence sets that guarantee the true genre belongs to the predicted set with a fixed probability (e.g. 95%).

---

## Stack

Python · scikit-learn · XGBoost · librosa · pandas · matplotlib · seaborn · pickle

---

## Structure

```
├── GHAZI__TAREK.ipynb   # Main notebook (all parts)
└── GHAZI_TAREK.csv      # Final predictions (exercise 1 submission)
```
