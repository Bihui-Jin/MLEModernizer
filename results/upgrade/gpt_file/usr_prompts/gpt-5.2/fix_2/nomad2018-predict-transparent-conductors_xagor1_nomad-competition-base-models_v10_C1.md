# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Predict the formation energy and bandgap energy of a material.

## Metric
Column-wise root mean squared logarithmic error.

## Submission Format
For each id in the test set, you must predict a value for both formation_energy_ev_natom and bandgap_energy_ev. The file should contain a header and have the following format:
```
id,formation_energy_ev_natom,bandgap_energy_ev
1,0.1779,1.8892
2,0.1779,1.8892
3,0.1779,1.8892
...
```

## Dataset
The following information has been included:

- Spacegroup (a label identifying the symmetry of the material)
- Total number of Al, Ga, In and O atoms in the unit cell ($\N_{total}$)
- Relative compositions of Al, Ga, and In (x, y, z)
- Lattice vectors and angles: lv1, lv2, lv3 (which are lengths given in units of angstroms ($10^{-10}$ meters) and $\alpha, \beta, \gamma$ (which are angles in degrees between 0° and 360°)

Note: For each line of the CSV file, the corresponding spatial positions of all of the atoms in the unit cell (expressed in Cartesian coordinates) are provided as a separate file.

train.csv - contains a set of materials for which the bandgap and formation energies are provided

test.csv - contains the set of materials for which you must predict the bandgap and formation energies

/{train|test}/{id}/geometry.xyz - files with spatial information about the material. The file name corresponds to the id in the respective csv files.

# 2. Python version

3.6

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (89 lines)
            sample_submission.csv (241 lines)
            sample_submission.csv.zip (765 Bytes)
            test.csv (241 lines)
            test.csv.zip (6.0 kB)
            test.zip (505.0 kB)
            train.csv (2161 lines)
            train.csv.zip (56.7 kB)
            train.zip (4.5 MB)
            nomad2018-predict-transparent-conductors/
                description.md (89 lines)
                sample_submission.csv (241 lines)
                ... and 7 other files
                nomad2018-predict-transparent-conductors/
                test/
                    1/
                        geometry.xyz (3.0 kB)
                    10/
                        geometry.xyz (3.0 kB)
                    ... and 239 other folders
                train/
                    1/
                        geometry.xyz (5.5 kB)
                    10/
                        geometry.xyz (2.3 kB)
                    ... and 2159 other folders
            test/
                1/
                    geometry.xyz (3.0 kB)
                10/
                    geometry.xyz (3.0 kB)
                ... and 239 other folders
            train/
                1/
                    geometry.xyz (5.5 kB)
                10/
                    geometry.xyz (2.3 kB)
                ... and 2159 other folders
        input/
            description.md (89 lines)
            sample_submission.csv (241 lines)
            sample_submission.csv.zip (765 Bytes)
            test.csv (241 lines)
            test.csv.zip (6.0 kB)
            test.zip (505.0 kB)
            train.csv (2161 lines)
            train.csv.zip (56.7 kB)
            train.zip (4.5 MB)
            nomad2018-predict-transparent-conductors/
                description.md (89 lines)
                sample_submission.csv (241 lines)
                ... and 7 other files
                nomad2018-predict-transparent-conductors/
                test/
                    1/
                        geometry.xyz (3.0 kB)
                    10/
                        geometry.xyz (3.0 kB)
                    ... and 239 other folders
                train/
                    1/
                        geometry.xyz (5.5 kB)
                    10/
                        geometry.xyz (2.3 kB)
                    ... and 2159 other folders
            test/
                1/
                    geometry.xyz (3.0 kB)
                10/
                    geometry.xyz (3.0 kB)
                ... and 239 other folders
            train/
                1/
                    geometry.xyz (5.5 kB)
                10/
                    geometry.xyz (2.3 kB)
                ... and 2159 other folders
        working/
            nomad2018-predict-transparent-conductors/
                description.md (89 lines)
                sample_submission.csv (241 lines)
                ... and 7 other files
                nomad2018-predict-transparent-conductors/
                test/
                    1/
                        geometry.xyz (3.0 kB)
                    10/
                        geometry.xyz (3.0 kB)
                    ... and 239 other folders
                train/
                    1/
                        geometry.xyz (5.5 kB)
                    10/
                        geometry.xyz (2.3 kB)
                    ... and 2159 other folders
```

-> data/nomad2018-predict-transparent-conductors/sample_submission.csv has 240 rows and 3 columns.
The columns are: id, formation_energy_ev_natom, bandgap_energy_ev

-> data/nomad2018-predict-transparent-conductors/test.csv has 240 rows and 12 columns.
The columns are: id, spacegroup, number_of_total_atoms, percent_atom_al, percent_atom_ga, percent_atom_in, lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang, lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree

-> data/nomad2018-predict-transparent-conductors/train.csv has 2160 rows and 14 columns.
The columns are: id, spacegroup, number_of_total_atoms, percent_atom_al, percent_atom_ga, percent_atom_in, lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang, lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree, formation_energy_ev_natom, bandgap_energy_ev

-> data/sample_submission.csv has 240 rows and 3 columns.
The columns are: id, formation_energy_ev_natom, bandgap_energy_ev

-> data/test.csv has 240 rows and 12 columns.
The columns are: id, spacegroup, number_of_total_atoms, percent_atom_al, percent_atom_ga, percent_atom_in, lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang, lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree

-> data/train.csv has 2160 rows and 14 columns.
The columns are: id, spacegroup, number_of_total_atoms, percent_atom_al, percent_atom_ga, percent_atom_in, lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang, lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree, formation_energy_ev_natom, bandgap_energy_ev

-> (stopped after 10 files for performance)

# 5. Target score

0.07034

# 6. Current score

0.09557

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.09557) has done: 'I update deprecated/removed scikit-learn imports (e.g., `sklearn.cross_validation`, `sklearn.grid_search`) to their modern equivalents so the notebook runs in your current environment. I also fix the train/test slicing bug (it hard-coded 2400/3000 instead of using the actual train/test sizes), which currently breaks training/inference alignment and prevents a valid submission. Since the metric is RMSLE, I add a minimal, score-improving but semantics-preserving safety step by clipping predictions to be non-negative before writing the submission (RMSLE is undefined for negatives). Finally, I keep the original Ridge/Lasso/XGBoost core approach and ensure a valid `XGB_Nomad.csv` is always produced with the required columns.'

# 9. Code solution

## === cell 0
import os
import gc
import time
import numpy as np
import pandas as pd
import glob
import io
import math
import matplotlib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt

from sklearn.linear_model import (
    Ridge,
    RidgeCV,
    ElasticNet,
    LassoCV,
    LassoLarsCV,
    LinearRegression,
)
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

CANDIDATE_INPUTS = [
    "/kaggle/input/nomad2018-predict-transparent-conductors",
    "/kaggle/input",
    "../input",
    "../input/nomad2018-predict-transparent-conductors",
]
INPUT_DIR = None
for p in CANDIDATE_INPUTS:
    if os.path.exists(p):
        if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
            os.path.join(p, "test.csv")
        ):
            INPUT_DIR = p
            break
        if os.path.exists(
            os.path.join(p, "nomad2018-predict-transparent-conductors", "train.csv")
        ):
            INPUT_DIR = os.path.join(p, "nomad2018-predict-transparent-conductors")
            break

if INPUT_DIR is None:
    raise FileNotFoundError(
        "Could not locate Kaggle input directory with train.csv/test.csv."
    )

print("Using INPUT_DIR:", INPUT_DIR)
print("Top-level files:", sorted(os.listdir(INPUT_DIR))[:20])



## === cell 1
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))



## === cell 2
print("Training data shape")
print(train_df.shape)
print("Testing data shape")
print(test_df.shape)



## === cell 3
print("Training columns")
print(train_df.columns)
print("Testing columns")
print(test_df.columns)



## === cell 4
print(train_df.dtypes)
print(test_df.dtypes)



## === cell 5
Targets_df = pd.DataFrame()
Targets_df["bandgap_energy_ev"] = train_df["bandgap_energy_ev"].copy()
Targets_df["formation_energy_ev_natom"] = train_df["formation_energy_ev_natom"].copy()
train_df = train_df.drop(["formation_energy_ev_natom", "bandgap_energy_ev"], axis=1)



## === cell 6
train_id_df = pd.DataFrame()
train_id_df["id"] = train_df["id"].copy()
train_df = train_df.drop(["id"], axis=1)

test_id_df = pd.DataFrame()
test_id_df["id"] = test_df["id"].copy()
test_df = test_df.drop(["id"], axis=1)



## === cell 7
combined_df = pd.concat([train_df, test_df], ignore_index=True)



## === cell 8
numerical_df = combined_df[
    [
        "number_of_total_atoms",
        "percent_atom_al",
        "percent_atom_ga",
        "percent_atom_in",
        "lattice_vector_1_ang",
        "lattice_vector_2_ang",
        "lattice_vector_3_ang",
        "lattice_angle_alpha_degree",
        "lattice_angle_beta_degree",
        "lattice_angle_gamma_degree",
    ]
].copy()

one_hot_df = combined_df[["spacegroup"]].copy()
one_hot_df = pd.get_dummies(one_hot_df, prefix=["spacegroup"], columns=["spacegroup"])

features_df = pd.concat([numerical_df, one_hot_df], axis=1)



## === cell 9
print("Total number of null values in the df")
print(features_df.isna().sum().sum())



## === cell 10
n_train = len(Targets_df)
training_examples = features_df.iloc[:n_train].copy()
test_examples = features_df.iloc[n_train:].copy()

print("training_examples shape:", training_examples.shape)
print("test_examples shape:", test_examples.shape)




## === cell 11
def rmsle_cv(model):
    rmsle = np.sqrt(
        -cross_val_score(
            model,
            training_examples,
            training_targets,
            scoring="neg_mean_squared_log_error",
            cv=5,
        )
    )
    return rmsle




## === cell 12
training_targets = Targets_df["bandgap_energy_ev"].copy()
model_ridge = RidgeCV(
    alphas=[0, 0.001, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75], cv=5
).fit(training_examples, training_targets)

print("Ridge best alpha (BG):", model_ridge.alpha_)
BG_rmsle = rmsle_cv(model_ridge).mean()
print("BG CV RMSLE:", BG_rmsle)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2530171648.py in <cell line: 0>()
      2 model_ridge = RidgeCV(
      3     alphas=[0, 0.001, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75], cv=5
----> 4 ).fit(training_examples, training_targets)
      5 
      6 print("Ridge best alpha (BG):", model_ridge.alpha_)

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in fit(self, X, y, sample_weight)
   2358         self._validate_params()
   2359 
-> 2360         super().fit(X, y, sample_weight=sample_weight)
   2361         return self
   2362 

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in fit(self, X, y, sample_weight)
   2144             if n_alphas != 1:
   2145                 for index, alpha in enumerate(self.alphas):
-> 2146                     alpha = check_scalar_alpha(alpha, f"alphas[{index}]")
   2147             else:
   2148                 self.alphas[0] = check_scalar_alpha(self.alphas[0], "alphas")

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_scalar(x, name, target_type, min_val, max_val, include_boundaries)
   1524     )
   1525     if min_val is not None and comparison_operator(x, min_val):
-> 1526         raise ValueError(
   1527             f"{name} == {x}, must be"
   1528             f" {'>=' if include_boundaries in ('left', 'both') else '>'} {min_val}."

ValueError: alphas[0] == 0, must be > 0.0.

## === cell 13
ridge_BG_preds = model_ridge.predict(test_examples)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1486320188.py in <cell line: 0>()
----> 1 ridge_BG_preds = model_ridge.predict(test_examples)
      2 

NameError: name 'model_ridge' is not defined

## === cell 14
training_targets = Targets_df["formation_energy_ev_natom"].copy()
model_ridge = RidgeCV(
    alphas=[0, 0.001, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75], cv=5
).fit(training_examples, training_targets)

print("Ridge best alpha (EF):", model_ridge.alpha_)
EF_rmsle = rmsle_cv(model_ridge).mean()
print("EF CV RMSLE:", EF_rmsle)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2843649858.py in <cell line: 0>()
      2 model_ridge = RidgeCV(
      3     alphas=[0, 0.001, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75], cv=5
----> 4 ).fit(training_examples, training_targets)
      5 
      6 print("Ridge best alpha (EF):", model_ridge.alpha_)

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in fit(self, X, y, sample_weight)
   2358         self._validate_params()
   2359 
-> 2360         super().fit(X, y, sample_weight=sample_weight)
   2361         return self
   2362 

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in fit(self, X, y, sample_weight)
   2144             if n_alphas != 1:
   2145                 for index, alpha in enumerate(self.alphas):
-> 2146                     alpha = check_scalar_alpha(alpha, f"alphas[{index}]")
   2147             else:
   2148                 self.alphas[0] = check_scalar_alpha(self.alphas[0], "alphas")

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_scalar(x, name, target_type, min_val, max_val, include_boundaries)
   1524     )
   1525     if min_val is not None and comparison_operator(x, min_val):
-> 1526         raise ValueError(
   1527             f"{name} == {x}, must be"
   1528             f" {'>=' if include_boundaries in ('left', 'both') else '>'} {min_val}."

ValueError: alphas[0] == 0, must be > 0.0.

## === cell 15
print("Expected combined error (Ridge CV)")
combined_rmsle = (EF_rmsle + BG_rmsle) / 2
print(combined_rmsle)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3481342223.py in <cell line: 0>()
      1 print("Expected combined error (Ridge CV)")
----> 2 combined_rmsle = (EF_rmsle + BG_rmsle) / 2
      3 print(combined_rmsle)
      4 

NameError: name 'EF_rmsle' is not defined

## === cell 16
ridge_EF_preds = model_ridge.predict(test_examples)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3824510852.py in <cell line: 0>()
----> 1 ridge_EF_preds = model_ridge.predict(test_examples)
      2 

NameError: name 'model_ridge' is not defined

## === cell 17
Predictions_df = pd.DataFrame()
Predictions_df["id"] = test_id_df["id"].copy()
Predictions_df["formation_energy_ev_natom"] = np.clip(ridge_EF_preds, 0, None)
Predictions_df["bandgap_energy_ev"] = np.clip(ridge_BG_preds, 0, None)
Predictions_df.to_csv("Ridge_Nomad.csv", index=False)
print("Wrote Ridge_Nomad.csv with shape:", Predictions_df.shape)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3012659889.py in <cell line: 0>()
      3 Predictions_df["id"] = test_id_df["id"].copy()
      4 # FIX: RMSLE requires non-negative predictions; clip to 0 for safety (minimal, metric-aligned).
----> 5 Predictions_df["formation_energy_ev_natom"] = np.clip(ridge_EF_preds, 0, None)
      6 Predictions_df["bandgap_energy_ev"] = np.clip(ridge_BG_preds, 0, None)
      7 Predictions_df.to_csv("Ridge_Nomad.csv", index=False)

NameError: name 'ridge_EF_preds' is not defined

## === cell 18
pass



## === cell 19
training_targets = Targets_df["bandgap_energy_ev"].copy()
model_lasso = LassoCV(
    alphas=[1, 0.1, 0.001, 0.0005, 0.0001, 0.00005, 1e-5, 1e-6, 0],
    cv=5,
    max_iter=100000,
).fit(training_examples, np.ravel(training_targets))

print("Lasso best alpha (BG):", model_lasso.alpha_)
BG_rmsle = rmsle_cv(model_lasso).mean()
print("BG CV RMSLE:", BG_rmsle)
lasso_coef = pd.Series(model_lasso.coef_, index=training_examples.columns)
print(
    "Lasso picked "
    + str(sum(lasso_coef != 0))
    + " variables and eliminated the other "
    + str(sum(lasso_coef == 0))
    + " variables"
)



## === cell 20
lasso_BG_preds = model_lasso.predict(test_examples)



## === cell 21
training_targets = Targets_df["formation_energy_ev_natom"].copy()
model_lasso = LassoCV(
    alphas=[1, 0.1, 0.001, 0.0005, 1e-5, 0], cv=5, max_iter=100000
).fit(training_examples, np.ravel(training_targets))

print("Lasso best alpha (EF):", model_lasso.alpha_)
EF_rmsle = rmsle_cv(model_lasso).mean()
print("EF CV RMSLE:", EF_rmsle)
lasso_coef = pd.Series(model_lasso.coef_, index=training_examples.columns)
print(
    "Lasso picked "
    + str(sum(lasso_coef != 0))
    + " variables and eliminated the other "
    + str(sum(lasso_coef == 0))
    + " variables"
)



## === cell 22
lasso_EF_preds = model_lasso.predict(test_examples)



## === cell 23
print("Expected combined error (Lasso CV)")
combined_rmsle = (EF_rmsle + BG_rmsle) / 2
print(combined_rmsle)



## === cell 24
Predictions_df = pd.DataFrame()
Predictions_df["id"] = test_id_df["id"].copy()
Predictions_df["formation_energy_ev_natom"] = np.clip(lasso_EF_preds, 0, None)
Predictions_df["bandgap_energy_ev"] = np.clip(lasso_BG_preds, 0, None)
Predictions_df.to_csv("Lasso_Nomad.csv", index=False)
print("Wrote Lasso_Nomad.csv with shape:", Predictions_df.shape)



## === cell 25
pass



## === cell 26
pass



## === cell 27
if "model_xgb" in globals() and hasattr(model_xgb, "loc"):
    last = len(model_xgb.loc[:]) - 1
    print(model_xgb.loc[last])
else:
    print("model_xgb cv results not available yet; skipping.")



## === cell 28
training_targets = np.log1p(Targets_df["bandgap_energy_ev"].copy())
dtrain = xgb.DMatrix(training_examples, label=training_targets)
dtest = xgb.DMatrix(test_examples)

params = {
    "max_depth": 2,
    "eta": 0.1,
    "gamma": 0,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "min_child_weight": 7,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "seed": 42,
}
model_xgb = xgb.cv(
    params,
    dtrain,
    num_boost_round=500,
    early_stopping_rounds=100,
    nfold=5,
    stratified=False,
    verbose_eval=False,
)

try:
    model_xgb.loc[30:, ["test-rmse-mean", "train-rmse-mean"]].plot()
    plt.show()
except Exception:
    pass

last = len(model_xgb.loc[:]) - 1
print(model_xgb.loc[last:, ["test-rmse-mean"]])



## === cell 29
best_n_estimators_bg = int(len(model_xgb))
model_xgb_bg = xgb.XGBRegressor(
    n_estimators=best_n_estimators_bg,
    max_depth=2,
    learning_rate=0.1,
    gamma=0,
    subsample=0.6,
    colsample_bytree=0.6,
    min_child_weight=7,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=4,
)
model_xgb_bg.fit(training_examples, training_targets)

try:
    xgb.plot_importance(model_xgb_bg, max_num_features=20)
    plt.show()
except Exception:
    pass

xgb_BG_preds = np.expm1(model_xgb_bg.predict(test_examples))



## === cell 30
training_targets = Targets_df["formation_energy_ev_natom"].copy()
dtrain = xgb.DMatrix(training_examples, label=training_targets)
dtest = xgb.DMatrix(test_examples)

params = {
    "max_depth": 3,
    "eta": 0.1,
    "gamma": 0,
    "subsample": 1,
    "colsample_bytree": 1,
    "min_child_weight": 1,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "seed": 42,
}
model_xgb = xgb.cv(
    params,
    dtrain,
    num_boost_round=500,
    early_stopping_rounds=100,
    nfold=5,
    stratified=False,
    verbose_eval=False,
)

try:
    model_xgb.loc[30:, ["test-rmse-mean", "train-rmse-mean"]].plot()
    plt.show()
except Exception:
    pass



## === cell 31
best_n_estimators_ef = int(len(model_xgb))
model_xgb_ef = xgb.XGBRegressor(
    n_estimators=best_n_estimators_ef,
    max_depth=2,
    learning_rate=0.1,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=4,
)
model_xgb_ef.fit(training_examples, training_targets)

try:
    xgb.plot_importance(model_xgb_ef, max_num_features=20)
    plt.show()
except Exception:
    pass

xgb_EF_preds = model_xgb_ef.predict(test_examples)



## === cell 32
Predictions_df = pd.DataFrame()
Predictions_df["id"] = test_id_df["id"].copy()

Predictions_df["formation_energy_ev_natom"] = np.clip(xgb_EF_preds, 0, None)
Predictions_df["bandgap_energy_ev"] = np.clip(xgb_BG_preds, 0, None)

Predictions_df = Predictions_df[
    ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
]
Predictions_df.to_csv("XGB_Nomad.csv", index=False)
print("Wrote XGB_Nomad.csv with shape:", Predictions_df.shape)
print(Predictions_df.head())
