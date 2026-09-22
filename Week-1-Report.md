# Week 1 Report – Python, Mathematics & ML Foundations
**Skill Set Go EduTech · AI/ML Internship · 4-Week Practical Learning Track**

Intern: _<your full name>_ · Offer Letter ID: _<your ID>_ · Domain: AI & Machine Learning
Submission deadline: **22 September 2026** · Repository: _<your repo URL>_

> Export this file to PDF before uploading if the Google Form requests a PDF.

---

## 1. What Week 1 covered
Four tasks, completed as one connected piece of work rather than four separate exercises.
The thread running through the week: **Task 1.2's exploratory analysis made a prediction,
and Task 1.4 tested it.**

| Task | Deliverable | Status |
|---|---|---|
| 1.1 Python Fundamentals | GitHub repository of topic-wise exercises | Completed |
| 1.2 Data Cleaning & EDA | Notebook + cleaned dataset + 6 visualisations | Completed |
| 1.3 Math for ML | Worked solutions notebook (exportable to PDF) | Completed |
| 1.4 Scikit-learn Models | Notebook with both models + evaluation report | Completed |

## 2. Task 1.1 – Python Fundamentals
Eight files, one per topic, plus `test_all.py` with 16 assertions that include invalid
input (marks of 140, division by zero, a missing file, a wrong object type). Every file
runs standalone.

**Main lesson:** a broad `except:` hides genuine bugs. Catching each error type separately
turned silent failures into readable messages.

## 3. Task 1.2 – Data Cleaning & EDA
**Dataset:** UCI Wine Recognition (178 real chemical analyses, 3 cultivars), re-exported as
a realistic messy CSV by a documented, reproducible generator script. The formatting damage
is simulated and disclosed; the measurements are real.

**Cleaning performed:** 6 duplicate rows removed · `proline` converted from `"1,065 mg/L"`
text to numeric · 12 category spellings collapsed to 3 · two date formats parsed ·
5 chemically impossible values nulled at cell level · missing values imputed with the
**cultivar-wise median**, with a written reason per column · outliers flagged by IQR but
kept as plausible chemistry.

**Findings:** `flavanoids`, `proline_mg_l` and `color_intensity` separate the cultivars
almost completely; `total_phenols`/`flavanoids`/`od280_od315` are strongly correlated
(r ≈ 0.86); three columns are right-skewed (which is why median imputation was used); and
no trend exists across batch dates.

## 4. Task 1.3 – Math for ML
Statistics, probability, linear algebra and calculus, solved by hand with the steps shown
and verified in NumPy. Selected results: variance 127.75 / 146.0 (population / sample) ·
r ≈ 0.996 · precision 0.846 vs recall 0.917 · **P(sick | positive test) = 0.167** ·
break-even accuracy 0.714 · gradient descent converging at η = 0.1 and diverging at η = 0.3.

**Main lesson:** every ML metric in this week is a conditional probability or a distance,
and the learning-rate divergence seen at η = 0.3 is the same failure that produces a NaN loss.

## 5. Task 1.4 – Two Scikit-learn Models
Both workflows split **before** preprocessing, scaled inside a `Pipeline`, and measured
against a dummy baseline first.

| | Regression (Diabetes) | Classification (cleaned Wine) |
|---|---|---|
| baseline | MAE 64.0, R² ≈ 0 | accuracy 0.389 |
| best model | Ridge (α=1.0) | Random Forest |
| result | MAE **42.8**, RMSE 53.8, **R² 0.454** | test accuracy 1.000, **CV macro F1 0.986** |

**Error analysis:** residuals average ≈ 0 but split by target level are **+35.6** on the
high-progression half and **−27.0** on the low half – the model regresses toward the mean
and misses the severe cases. The classifier makes no test errors, so the cross-validated F1
is reported as the honest figure.

**The week's payoff:** the random forest's top three features – `proline_mg_l`,
`flavanoids`, `color_intensity` – are exactly the three that Task 1.2's charts identified.
The exploratory work did real predictive work.

## 6. Limitations
- Both datasets are small (178 and 442 rows); the wine test set is only 36 samples.
- Imputation slightly reduces true variance in four columns; affected rows are flagged.
- No hyperparameter tuning – deliberately deferred to Week 2, Task 2.3.
- The raw-data quality problems in Task 1.2 are a documented simulation, not errors found
  in the wild.

## 7. Evidence checklist for the Week 1 Google Form
- ☐ GitHub repository link (public, opened in an incognito window to confirm access)
- ☐ Task 1.1 – repository folder link + terminal screenshots
- ☐ Task 1.2 – notebook link + cleaned dataset + visualisations
- ☐ Task 1.3 – worked solutions notebook / PDF export
- ☐ Task 1.4 – notebook link + `results/evaluation_report.md` (or its PDF export)
- ☐ Full name and offer letter ID exactly as instructed in the form

## 8. Week 2 plan
Start PyTorch tensors and the training loop on Mon–Tue, the CNN on Wed–Thu, the
hyperparameter experiment table Fri–Sat, and the 1–2 page technical report before the
**29 September** deadline. Continue the external credentials in parallel (Forage and IBM
SkillsBuild first) so Week 4 is not overloaded.
