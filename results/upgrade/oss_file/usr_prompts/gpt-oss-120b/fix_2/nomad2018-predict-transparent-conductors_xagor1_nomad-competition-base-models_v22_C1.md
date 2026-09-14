# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.06924

# 6. Current score

0.05901

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.05901) has done: 'I fix the import errors, correct the data slicing, ensure the same feature transformation is used for both targets, and guarantee that a properly‑named CSV submission is written. These changes remove the runtime crashes and produce a valid *.csv file while preserving the original modelling approach.'

# 9. Code solution

## === cell 0
import os
import gc
import time
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error

import xgboost as xgb
import matplotlib.pyplot as plt

print(os.listdir("../input"))



## === cell 1
path = "../input/"
train_df = pd.read_csv(path + "/train.csv")
test_df = pd.read_csv(path + "/test.csv")



## === cell 2
print("Training data shape\n", train_df.shape)
print("Testing data shape\n", test_df.shape)

print("Training columns\n", train_df.columns)
print("Testing columns\n", test_df.columns)

print("Train data types\n", train_df.dtypes)
print("Test data types\n", test_df.dtypes)



## === cell 3
Targets_df = train_df[["bandgap_energy_ev", "formation_energy_ev_natom"]].copy()
train_ids = train_df["id"].copy()
test_ids = test_df["id"].copy()

train_features = train_df.drop(
    columns=["id", "bandgap_energy_ev", "formation_energy_ev_natom"]
)
test_features = test_df.drop(columns=["id"])

combined_df = pd.concat([train_features, test_features], ignore_index=True)
print("Total number of null values in the combined df:", combined_df.isna().sum().sum())



## === cell 4
num_cols = [
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

numerical_df = combined_df[num_cols].copy()
one_hot_df = pd.get_dummies(combined_df[["spacegroup"]], prefix="spacegroup")

features_df = pd.concat([numerical_df, one_hot_df], axis=1)

skewed_feats = numerical_df.skew()
skewed = skewed_feats[skewed_feats > 0.1].index
unskewed = skewed_feats[skewed_feats <= 0.1].index

transform_df = pd.DataFrame()
transform_df[unskewed] = (numerical_df[unskewed] - numerical_df[unskewed].mean()) / (
    numerical_df[unskewed].max() - numerical_df[unskewed].min()
)
transform_df[skewed] = np.log1p(numerical_df[skewed])

features_transform_df = pd.concat([transform_df, one_hot_df], axis=1)



## === cell 5
train_len = train_df.shape[0]  # 2160
training_examples = features_df.iloc[:train_len].reset_index(drop=True)
test_examples = features_df.iloc[train_len:].reset_index(drop=True)
training_examples_transform = features_transform_df.iloc[:train_len].reset_index(
    drop=True
)
test_examples_transform = features_transform_df.iloc[train_len:].reset_index(drop=True)




## === cell 6
def rmsle_cv(model, X, y):
    """Cross‑validated RMSLE using negative MSLE score."""
    rmsle = np.sqrt(
        -cross_val_score(model, X, y, scoring="neg_mean_squared_log_error", cv=5)
    )
    return rmsle.mean()




## === cell 7
bg_target = Targets_df["bandgap_energy_ev"]
model_bg_lin = LinearRegression().fit(training_examples_transform, bg_target)
bg_pred_lin = model_bg_lin.predict(test_examples_transform)
bg_rmsle = rmsle_cv(model_bg_lin, training_examples_transform, bg_target)
print("Bandgap Linear RMSLE:", bg_rmsle)

ef_target = Targets_df["formation_energy_ev_natom"]
model_ef_lin = LinearRegression().fit(training_examples_transform, ef_target)
ef_pred_lin = model_ef_lin.predict(test_examples_transform)  # use transformed features
ef_rmsle = rmsle_cv(model_ef_lin, training_examples_transform, ef_target)
print("Formation Energy Linear RMSLE:", ef_rmsle)

combined_rmsle = (bg_rmsle + ef_rmsle) / 2
print("Combined RMSLE (linear):", combined_rmsle)



## === cell 8
bg_target_log = np.log1p(bg_target)
dtrain_bg = xgb.DMatrix(training_examples, label=bg_target_log)
params_bg = {
    "max_depth": 2,
    "eta": 0.1,
    "gamma": 0,
    "subsample": 0.8,
    "colsample_bytree": 1,
    "min_child_weight": 10,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
}
bg_cv = xgb.cv(
    params_bg,
    dtrain_bg,
    num_boost_round=500,
    early_stopping_rounds=100,
    verbose_eval=False,
)
bg_best_nrounds = len(bg_cv)
print("Best BG rounds:", bg_best_nrounds)

model_bg_xgb = xgb.XGBRegressor(
    n_estimators=bg_best_nrounds,
    max_depth=2,
    learning_rate=0.1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=1,
    min_child_weight=10,
    objective="reg:squarederror",
)
model_bg_xgb.fit(training_examples, bg_target_log)
xgb.plot_importance(
    model_bg_xgb, max_num_features=10, height=0.5, title="BG Feature Importance"
)
bg_pred_xgb = np.expm1(model_bg_xgb.predict(test_examples))



## === cell 9
ef_target_log = np.log1p(ef_target)
dtrain_ef = xgb.DMatrix(training_examples, label=ef_target_log)
params_ef = {
    "max_depth": 4,
    "eta": 0.08,
    "gamma": 0,
    "subsample": 1,
    "colsample_bytree": 0.4,
    "min_child_weight": 3,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
}
ef_cv = xgb.cv(
    params_ef,
    dtrain_ef,
    num_boost_round=500,
    early_stopping_rounds=100,
    verbose_eval=False,
)
ef_best_nrounds = len(ef_cv)
print("Best EF rounds:", ef_best_nrounds)

model_ef_xgb = xgb.XGBRegressor(
    n_estimators=ef_best_nrounds,
    max_depth=4,
    learning_rate=0.08,
    gamma=0,
    subsample=1,
    colsample_bytree=0.4,
    min_child_weight=3,
    objective="reg:squarederror",
)
model_ef_xgb.fit(training_examples, ef_target_log)
xgb.plot_importance(
    model_ef_xgb, max_num_features=10, height=0.5, title="EF Feature Importance"
)
ef_pred_xgb = np.expm1(model_ef_xgb.predict(test_examples))



## === cell 10
stacked_bg = 0.95 * bg_pred_xgb + 0.05 * bg_pred_lin

submission = pd.DataFrame(
    {
        "id": test_ids,
        "formation_energy_ev_natom": ef_pred_xgb,
        "bandgap_energy_ev": stacked_bg,
    }
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
