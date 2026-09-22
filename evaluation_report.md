# Task 1.4 - Model Evaluation Report
**Skill Set Go EduTech - AI/ML Internship, Week 1**
Author: _<your full name>_ | Offer Letter ID: _<your ID>_ | Date: _<submission date>_

All numbers below are produced by `ml_models.py` / `notebook.ipynb` with
`random_state = 42` and are written verbatim to `results/metrics.json`.

---

## 1. Workflow A - Regression

**Problem.** Predict diabetes disease progression one year after baseline from 10
physiological measurements (UCI Diabetes, 442 patients, 10 features). Continuous
target, so regression metrics apply.

**Setup.** 80/20 train-test split (354 / 88), `StandardScaler` fitted inside a
`Pipeline` on training data only, 5-fold KFold cross-validation on the training set.

### Results

| Model | MAE | RMSE | R² (test) | R² (5-fold CV) |
|---|---|---|---|---|
| Baseline - predict the mean | 64.006 | 73.222 | -0.012 | - |
| Linear Regression | 42.794 | 53.853 | 0.453 | 0.480 ± 0.041 |
| **Ridge (α = 1.0)** | **42.812** | **53.778** | **0.454** | **0.481 ± 0.040** |
| Random Forest (300 trees) | 44.752 | 54.921 | 0.431 | 0.414 ± 0.072 |

**Selected model: Ridge regression.**

### Why these metrics
- **MAE (42.8)** - the typical error in the target's own units, robust to a few large misses.
- **RMSE (53.8)** - squares errors first, so large misses dominate it. The **11-point gap
  between RMSE and MAE** is itself a finding: the error distribution has a heavy tail, so
  a handful of badly-missed patients are responsible for a disproportionate share of the loss.
- **R² (0.454)** - the model explains about 45% of the variance in progression. The
  baseline's R² of ≈0 is what makes that number meaningful.

Reporting any one of these alone would hide something: R² alone hides the size of a typical
error, MAE alone hides the tail.

### Error inspection
- Mean residual ≈ 0 overall, so the model is unbiased **on average**.
- Split by target level, it is not: mean residual **+35.6** on the high-progression half
  and **−27.0** on the low half. The model under-predicts severe cases and over-predicts
  mild ones - textbook regression toward the mean.
- Every one of the 8 worst individual errors sits at the high end of the target range.
- The random forest, despite far more capacity, performs *worse* and with triple the CV
  variance (±0.072 vs ±0.040). With 354 training rows it overfits; the linear models'
  built-in simplicity is an advantage here, not a limitation.

**Conclusion.** The model beats the baseline decisively (MAE 64 → 43) but explains under
half the variance and systematically misses the patients who matter most. It is a
population-level trend model, not something to use for an individual clinical decision.

---

## 2. Workflow B - Classification

**Problem.** Predict the cultivar (A / B / C) of a wine sample from 13 chemical
measurements, using the dataset **cleaned in Task 1.2** (178 samples after duplicate
removal).

**Setup.** Stratified 80/20 split (142 / 36), `StandardScaler` inside a `Pipeline`,
5-fold StratifiedKFold cross-validation on the training set.

### Results

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (5-fold CV) |
|---|---|---|---|---|---|
| Baseline - always predict the largest class | 0.389 | - | - | - | - |
| Logistic Regression | 1.000 | 1.000 | 1.000 | 1.000 | 0.978 ± 0.029 |
| **Random Forest (300 trees)** | 1.000 | 1.000 | 1.000 | 1.000 | **0.986 ± 0.027** |

### Why these metrics
Classes are imbalanced (A 59 / B 71 / C 48, i.e. 33% / 40% / 27%), so the always-predict-B
baseline already reaches **38.9% accuracy** without learning anything. Macro averaging
weights all three classes equally, so the smallest class cannot be ignored by the score.
The confusion matrix is reported because no single number shows *which* classes get
confused.

### Error inspection
Both models classify all 36 test samples correctly - the confusion matrix is purely
diagonal, so there are no errors to analyse. **This perfect score is not the headline
number.** On 36 samples a single mistake would move accuracy by 2.8 points, so the
cross-validated macro F1 (**0.978** and **0.986** over 5 folds of the training data) is the
honest estimate: the cultivars are *nearly* separable and a few borderline samples flip
when the split changes.

### Feature importance - the EDA prediction held
The random forest's top three features are `proline_mg_l` (0.174), `flavanoids` (0.156)
and `color_intensity` (0.155) - **exactly** the three columns the Task 1.2 boxplots and
scatter plot identified as separating the cultivars. Together they carry roughly half the
total importance. The exploratory analysis did genuine predictive work.

---

## 3. Assumptions
- Test rows are independent of training rows. This is only true **because** Task 1.2
  removed the 6 duplicate rows first - had they survived, the same wine could have landed
  in both sets and the classification score would have been inflated by leakage.
- Scaling statistics come from the training fold only; no test information leaks in.
- The wine measurements are comparable across batches, which the batch-date trend check in
  Task 1.2 supported.

## 4. Limitations and next steps
| Limitation | Next step |
|---|---|
| Regression explains <50% of variance and misses severe cases | more informative features; try a model with an explicit non-linear term and compare fairly |
| Wine test set is only 36 samples | report cross-validated scores, or repeat over several random splits |
| No hyperparameter tuning (defaults only) | deliberate - controlled experimentation is Week 2, Task 2.3; tuning before a baseline is the mistake the execution guide warns about |
| Imputed cells from Task 1.2 slightly reduce true variance | compare against a complete-case model to quantify the effect |

## 5. Reproducibility
```bash
pip install -r ../../requirements.txt
cd Week-1/Task-1.2-Data-Cleaning-EDA && python make_raw_dataset.py && python eda_pipeline.py
cd ../Task-1.4-Scikit-Learn-Models && python ml_models.py
```
Outputs: `results/metrics.json`, `results/*.csv`, `results/*.png`, `models/*.joblib`.
