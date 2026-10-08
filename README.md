# Explainable Machine Learning for Elevated hs-CRP in Indonesian Adults (IFLS-5)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23239973.svg)](https://doi.org/10.5281/zenodo.23239973)

Code for the study *"Explainable machine learning for predicting elevated high-sensitivity C-reactive protein in Indonesian adults: Evidence from the Indonesia Family Life Survey"*.

The study predicts elevated hs-CRP (≥3 mg/L) in 5,980 adults from IFLS-5 (2014–15). It compares logistic regression, random forest, XGBoost, LightGBM, two neural networks (a scikit-learn multilayer perceptron and a PyTorch deep neural network), a soft-voting ensemble, and two parsimonious logistic models (BMI only; BMI, sex, age, self-rated health and residence). Models are explained with SHAP.

## Data availability

IFLS data are distributed by RAND Corporation to registered users only, and the IFLS terms of use do not allow redistribution. **This repository contains code and aggregated results only, no IFLS data.** To reproduce the analysis:

1. Register and download from RAND (https://www.rand.org/well-being/social-and-behavioral-policy/data/FLS/IFLS.html):
   - `hh14_all_dta.zip` (IFLS-5 household survey)
   - `hh14_dbs_dta.zip` (IFLS-5 dried blood spot data)
2. Extract both archives into `data/raw/`.
3. Run, from the repository root:

```bash
pip install -r requirements.txt
python src/pce14.py data/raw data/pce14.csv                                   # per capita expenditure
python src/build14.py data/raw data/pce14.csv data/ifls5_analytic.csv          # individual analytic file
python src/make_model_dataset.py data/ifls5_analytic.csv data/crp_ifls5_model.csv
```

Then run the notebooks in this order (Run All in each; edit the `ROOT` path in the first cell):

1. `notebooks/model_crp_ifls5.ipynb`
2. `notebooks/analisis_tambahan.ipynb`
3. `notebooks/analisis_model_sederhana.ipynb`

## Repository structure

| Path | Content |
| --- | --- |
| `src/pce14.py` | Household per capita expenditure from Book 1 (KS) and Book 2 (KR), following the official IFLS East 2012 PCE do-file |
| `src/build14.py` | Individual-level analytic file (demographics, lifestyle, anthropometry, chronic disease, CES-D, DBS biomarkers) |
| `src/make_model_dataset.py` | Analytic sample: DBS subsample, age ≥15, not pregnant, hs-CRP ≤10 mg/L, complete core predictors (n = 5,980) |
| `notebooks/model_crp_ifls5.ipynb` | Descriptive statistics, logistic regression, tuning with 5-fold CV, test-set evaluation with bootstrap CIs, sensitivity at fixed specificity, SHAP, sex-stratified and sensitivity analyses |
| `notebooks/analisis_tambahan.ipynb` | Repeated cross-validation (individual and household-grouped folds), cluster-robust logistic regression, recalibration, further sensitivity analyses |
| `notebooks/analisis_model_sederhana.ipynb` | BMI-only and five-item logistic models compared with the ensemble and XGBoost |
| `results/` | Aggregated result tables (`.xlsx`) and figures (`fig1`, `fig2`, `fig3`, `figS1`) |
| `Supplementary_material.docx` | Supplementary Tables S1–S5 and Figs. S1–S2 |

Software used for the reported results: Python 3, scikit-learn 1.9.1, XGBoost 2.1.4, LightGBM 4.7.0, SHAP 0.46.0, PyTorch 2.14.1 (CPU). The notebooks print `MODE: FINAL` when all packages are installed.

## Citation

See `CITATION.cff` or cite the archived code: https://doi.org/10.5281/zenodo.23239973. Please also cite the IFLS-5 data: Strauss, J., Witoelar, F., & Sikoki, B. (2016). *The fifth wave of the Indonesia Family Life Survey: Overview and field report* (WR-1143/1-NIA/NICHD). RAND.

## License

Code: MIT License (see `LICENSE`). IFLS data remain subject to RAND's terms of use.
