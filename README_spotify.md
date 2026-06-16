# Predicting Spotify Song Popularity from Audio Features

> Can a song's audio characteristics predict how popular it becomes? This project
> trains and compares three tree-based regressors on ~114k Spotify tracks to find
> out, and to identify which features carry the most signal.

**Tech:** Python · pandas · scikit-learn · XGBoost · seaborn
**Type:** End-to-end data science project · regression

---

## Research question

> *To what extent can audio features predict a song's popularity on Spotify, and
> which features are most informative?*

## Data

The [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
from Kaggle, with **114,000 tracks** combining acoustic features (danceability,
energy, loudness, tempo, etc.) with metadata (genre, artist, duration). After
removing 450 duplicate rows and a handful of missing values, **113,549 tracks**
remain.

## Approach

1. **Data quality:** duplicate/missing checks, outlier inspection (kept, since
   they reflect real track types rather than noise).
2. **EDA:** distributions, feature–popularity scatter plots, category boxplots,
   and a correlation matrix. The takeaway: popularity has only **weak linear
   relationships** with individual audio features, motivating non-linear models.
3. **Preprocessing:** ordinal encoding of categoricals (fit on train only to
   avoid leakage), 80/20 train/test split.
4. **Modeling:** three tree-based ensembles: Random Forest (bagging),
   Gradient Boosting, and XGBoost.
5. **Evaluation:** R² and MAE on the hold-out set, plus feature importance.

## Results

| Model | R² | MAE |
|---|---|---|
| **Random Forest** | **0.52** | **10.9** |
| XGBoost | 0.41 | 12.8 |
| Gradient Boosting | 0.33 | 14.1 |

Random Forest performed best. An R² of ~0.52 means audio and contextual features
explain **roughly half** the variation in popularity, which is meaningful but far
from complete. Across all three models, **`track_genre`** was consistently the most
important feature, while individual acoustic features (danceability, energy,
valence, loudness) carried more moderate weight.

**Conclusion:** audio features alone provide limited predictive power. Popularity
is also driven by external factors the dataset doesn't capture: marketing,
playlist placement, social-media exposure, artist reputation.

## How to run

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

1. Download the dataset from Kaggle and place it at `Data/dataset.csv`.
2. Open `spotify-popularity-analysis.ipynb` and run top to bottom.

## Future work

Systematic hyperparameter optimisation and cross-validation to push performance
further, and incorporating external contextual data for more realistic models.

## Author

Robin Dogan ([@RobinDogan](https://github.com/RobinDogan))
