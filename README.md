# JC3509 Machine Learning Assessment

**Student ID**: JC3509  
**Task**: Bike Sharing Demand Prediction - Comparing Linear Regression and Random Forest Regressor

---

## Project Structure

```
JC3509_Assessment_23/
├── README.md                          # Project overview and usage guide
├── requirements.txt                   # Python dependencies
├── data/
│   ├── raw/
│   │   └── day.csv                    # Original dataset (731×16)
│   └── processed/
│       ├── day_processed.csv          # Preprocessed data (731×33)
│       └── split_indices.json         # Train/val/test split indices
├── src/
│   ├── preprocess.py                  # Data preprocessing script
│   ├── train_evaluate.py              # Model training & evaluation script
│   └── visualize_eda.py               # EDA visualization
└── results/
    ├── metrics.csv                    # Model performance metrics
    ├── lr_coefficients.csv            # Linear Regression coefficients
    ├── rf_importances.csv             # Random Forest feature importances
    ├── rf_best_params.json            # Random Forest best hyperparameters
    └── figures/                       # Visualization outputs
        ├── prediction_scatter.png     # Predicted vs actual values
        ├── feature_importance.png     # Feature importance comparison
        └── residuals.png              # Residual plots
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Data Preprocessing

```bash
cd src
python preprocess.py
```

**What it does:**
- Loads `data/raw/day.csv` (731 days × 16 features)
- Removes columns: `instant`, `dteday`, `casual`, `registered`, `temp`
- One-hot encodes: `season`, `mnth`, `weekday`, `weathersit` (keeps all dummies)
- Splits data: 70% train (511) / 15% val (110) / 15% test (110)
- Validates: no NaN, all features in [0, 1], shape (731, 33)

**Output:**
- `data/processed/day_processed.csv` - 731 rows × 33 columns
- `data/processed/split_indices.json` - train/val/test indices

### 3. Model Training & Evaluation

```bash
cd src
python train_evaluate.py
```

**What it does:**
- Trains Linear Regression (default parameters)
- Trains Random Forest with hyperparameter tuning:
  - Grid search: n_estimators [100, 200, 500] × max_depth [None, 10, 20]
  - Selects best params on validation set
  - Retrains on train+val combined data
- Evaluates both models on train/test sets (RF's validation metrics labeled as `train_seen_val` since val data was used in final retraining)
- Generates visualizations and saves results

**Output:**
- `results/metrics.csv` - Performance metrics (MAE, RMSE, R²)
- `results/lr_coefficients.csv` - Top-15 LR coefficients
- `results/rf_importances.csv` - Top-15 RF feature importances
- `results/rf_best_params.json` - RF best hyperparameters
- `results/figures/*.png` - 3 visualization plots

**Expected runtime:** 2-5 minutes

---

## Dataset

### Source
Capital Bikeshare (Washington D.C., 2011-2012)  
Original data: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset

### Data Description
- **Original**: 731 days × 16 features
- **Processed**: 731 days × 33 features (after one-hot encoding)
- **Target variable**: `cnt` (daily total bike rental count)

### Features (after preprocessing)
- **Numerical** (6): `yr`, `holiday`, `workingday`, `atemp`, `hum`, `windspeed`
- **Categorical (one-hot)** (26): 
  - `season_1` to `season_4` (4 columns)
  - `mnth_1` to `mnth_12` (12 columns)
  - `weekday_0` to `weekday_6` (7 columns)
  - `weathersit_1` to `weathersit_3` (3 columns)
- **Target** (1): `cnt`

### Data Split
- **Training**: 70% (511 samples) - for model fitting
- **Validation**: 15% (110 samples) - for hyperparameter tuning
- **Test**: 15% (110 samples) - for final evaluation
- **Split method**: Random split with `random_state=42`

---

## Models

### Linear Regression
- **Implementation**: `sklearn.linear_model.LinearRegression()`
- **Parameters**: Default (OLS)
- **No hyperparameter tuning**

### Random Forest Regressor
- **Implementation**: `sklearn.ensemble.RandomForestRegressor(random_state=42)`
- **Hyperparameter tuning**:
  - `n_estimators`: [100, 200, 500]
  - `max_depth`: [None, 10, 20]
  - Tuning method: Grid search on validation set
  - **Best parameters**: `n_estimators=500`, `max_depth=None`
- **Final training**: Retrained on train+val combined (621 samples)

---

## Evaluation Metrics

| Metric | Formula | Unit | Interpretation |
|---|---|---|---|
| **MAE** | (1/n) Σ\|y - ŷ\| | bikes/day | Average prediction error |
| **RMSE** | √[(1/n) Σ(y - ŷ)²] | bikes/day | Penalizes large errors more |
| **R²** | 1 - (SS_res / SS_tot) | - | Proportion of variance explained |

---

## Results Summary

### Model Performance (Test Set)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 595.98 | 747.70 | 0.85 |
| **Random Forest** | **494.02** | **630.70** | **0.90** |

**Key findings:**
- Random Forest outperforms Linear Regression by **15.6% in RMSE**
- RF achieves **R² = 0.90**, explaining 90% of variance
- RF reduces MAE by **17.1%** (102 bikes/day improvement)

### Feature Importance (Random Forest Top-5)

| Feature | Importance | Interpretation |
|---|---|---|
| `atemp` | 47.9% | Apparent temperature (most important) |
| `yr` | 27.6% | Year (2012 vs 2011, system growth) |
| `hum` | 6.4% | Humidity |
| `season_1` | 4.2% | Spring season |
| `season_4` | 3.3% | Winter season |

---

## Key Insights

### Why Random Forest Performs Better

1. **Non-linear relationships**: Captures complex patterns between weather and demand
2. **Feature interactions**: Automatically learns interactions (e.g., temperature × workday)
3. **Robust to multicollinearity**: Tree-based splitting unaffected by one-hot encoding issues

### Linear Regression Limitations

1. **Multicollinearity**: One-hot encoding without `drop_first` causes coefficient instability (10^16 magnitude)
2. **Linear assumption**: Cannot capture non-linear temperature-demand relationships
3. **No automatic interactions**: Requires manual feature engineering

### Overfitting Analysis

- **LR**: Train RMSE (759.60) ≈ Test RMSE (747.70) → Good generalization
- **RF**: Train RMSE (257.35) << Test RMSE (630.70) → Some overfitting, but still outperforms LR

---

## Preprocessing Details

### Removed Columns

| Column | Reason |
|---|---|
| `instant` | Row index, no predictive value |
| `dteday` | Date already encoded in yr/mnth/weekday |
| `casual` | Data leakage: cnt = casual + registered |
| `registered` | Data leakage: cnt = casual + registered |
| `temp` | High correlation with atemp (r=0.9917) |

### One-Hot Encoding Strategy

- **Method**: `pd.get_dummies(drop_first=False)`
- **Rationale**: Keeps all categories for interpretability in paper
- **Trade-off**: Introduces multicollinearity in LR (but RF is robust to this)

### No Additional Scaling

- Original dataset already min-max normalized to [0, 1]
- One-hot encoded features are naturally 0/1
- No further scaling needed

---

## Files and Outputs

### Input Files
- `data/raw/day.csv` - Original dataset

### Intermediate Files
- `data/processed/day_processed.csv` - Preprocessed features
- `data/processed/split_indices.json` - Train/val/test indices

### Result Files
- `results/metrics.csv` - Performance metrics table
- `results/lr_coefficients.csv` - LR coefficients (top-15)
- `results/rf_importances.csv` - RF feature importances (top-15)
- `results/rf_best_params.json` - RF best hyperparameters

### Visualization Files
- `results/figures/prediction_scatter.png` - Predicted vs actual scatter plots
- `results/figures/feature_importance.png` - Feature importance comparison
- `results/figures/residuals.png` - Residual distribution plots

### Documentation
- `README.md` - This file

---

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib

See `requirements.txt` for specific versions.

---

## Reproducibility

All experiments use fixed random seeds:
- Data split: `random_state=42`
- Random Forest: `random_state=42`

Running the scripts will produce identical results.

---

## Future Work

1. **Feature engineering**: Interaction terms (temp × workday), lagged features
2. **Separate modeling**: Model casual and registered users independently
3. **Advanced models**: XGBoost, LightGBM, Neural Networks
4. **More data**: Incorporate additional years or cities
5. **Hyperparameter optimization**: Bayesian optimization for RF

---

## Citation

If using this dataset, please cite:

Fanaee-T, Hadi, and Gama, Joao. "Event labeling combining ensemble detectors and background knowledge." Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg.

---

## Author

**JC3509** - Machine Learning Group Assessment 2026

For questions or issues, please contact the project team.
