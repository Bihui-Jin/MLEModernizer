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

0.06922

# 6. Current score

0.05869

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.05869) has done: 'I import the missing `cross_val_score`, ensure all variables are defined in the correct order, and keep the existing model logic unchanged. These fixes resolve the runtime errors, allow the script to run end‑to‑end, and generate a valid `Stacked_Nomad.csv` submission file.'

# 9. Code solution

## === cell 0
import os
import gc
import time
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error




## === cell 1
path = "../input/"
train_df = pd.read_csv(os.path.join(path, "train.csv"))
test_df = pd.read_csv(os.path.join(path, "test.csv"))




## === cell 2
print("Training data shape", train_df.shape)
print("Testing data shape", test_df.shape)

Targets_df = train_df[["bandgap_energy_ev", "formation_energy_ev_natom"]].copy()
train_features = train_df.drop(
    columns=["bandgap_energy_ev", "formation_energy_ev_natom", "id"]
).copy()
train_id_df = train_df[["id"]].copy()
test_id_df = test_df[["id"]].copy()
test_features = test_df.drop(columns=["id"]).copy()

combined_df = pd.concat([train_features, test_features], ignore_index=True)
print("Total nulls:", combined_df.isna().sum().sum())




## === cell 3
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

skewed = numerical_df.skew()
skewed_feats = skewed[skewed > 0.1].index
unskewed_feats = skewed[skewed <= 0.1].index

transform_df = pd.DataFrame()
transform_df[unskewed_feats] = (
    numerical_df[unskewed_feats] - numerical_df[unskewed_feats].mean()
) / (numerical_df[unskewed_feats].max() - numerical_df[unskewed_feats].min())
transform_df[skewed_feats] = np.log1p(numerical_df[skewed_feats])

features_transform_df = pd.concat([transform_df, one_hot_df], axis=1)
print("Transformed skew (sample):")
print(transform_df.skew().head())




## === cell 4
n_train = train_df.shape[0]  # 2160
training_examples = features_df.iloc[:n_train].reset_index(drop=True)
test_examples = features_df.iloc[n_train:].reset_index(drop=True)

training_examples_transform = features_transform_df.iloc[:n_train].reset_index(
    drop=True
)
test_examples_transform = features_transform_df.iloc[n_train:].reset_index(drop=True)




## === cell 5
def rmsle_cv(model, X, y):
    """Cross‑validated RMSLE using 5‑fold CV."""
    scores = -cross_val_score(model, X, y, scoring="neg_mean_squared_log_error", cv=5)
    return np.sqrt(scores)




## === cell 6
bandgap_target = Targets_df["bandgap_energy_ev"]
lin_bg = LinearRegression().fit(training_examples_transform, bandgap_target)
linear_BG_pred = lin_bg.predict(test_examples_transform)
bg_rmsle = rmsle_cv(lin_bg, training_examples_transform, bandgap_target).mean()
print("Bandgap Linear RMSLE:", bg_rmsle)

formation_target = Targets_df["formation_energy_ev_natom"]
lin_fe = LinearRegression().fit(training_examples_transform, formation_target)
linear_EF_pred = lin_fe.predict(test_examples_transform)
ef_rmsle = rmsle_cv(lin_fe, training_examples_transform, formation_target).mean()
print("Formation Energy Linear RMSLE:", ef_rmsle)

print("Combined Linear RMSLE:", (bg_rmsle + ef_rmsle) / 2)




## === cell 7
bg_log_target = np.log1p(bandgap_target)
dtrain_bg = xgb.DMatrix(training_examples, label=bg_log_target)

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
cv_bg = xgb.cv(
    params_bg,
    dtrain_bg,
    num_boost_round=500,
    early_stopping_rounds=100,
    verbose_eval=False,
    seed=42,
)
best_nbg = len(cv_bg)
print("Best n_estimators for BG:", best_nbg)

fe_log_target = np.log1p(formation_target)
dtrain_fe = xgb.DMatrix(training_examples, label=fe_log_target)

params_fe = {
    "max_depth": 4,
    "eta": 0.08,
    "gamma": 0,
    "subsample": 1,
    "colsample_bytree": 0.4,
    "min_child_weight": 3,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
}
cv_fe = xgb.cv(
    params_fe,
    dtrain_fe,
    num_boost_round=500,
    early_stopping_rounds=100,
    verbose_eval=False,
    seed=42,
)
best_nfe = len(cv_fe)
print("Best n_estimators for EF:", best_nfe)




## === cell 8
model_bg = xgb.XGBRegressor(
    n_estimators=best_nbg,
    max_depth=2,
    learning_rate=0.1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=1,
    min_child_weight=10,
    objective="reg:squarederror",
    n_jobs=5,
    random_state=42,
)
model_bg.fit(training_examples, bg_log_target)

model_fe = xgb.XGBRegressor(
    n_estimators=best_nfe,
    max_depth=4,
    learning_rate=0.08,
    gamma=0,
    subsample=1,
    colsample_bytree=0.4,
    min_child_weight=3,
    objective="reg:squarederror",
    n_jobs=5,
    random_state=42,
)
model_fe.fit(training_examples, fe_log_target)

xgb_BG_preds = np.expm1(model_bg.predict(test_examples))
xgb_EF_preds = np.expm1(model_fe.predict(test_examples))




## === cell 9
stacked_BG = 0.9 * xgb_BG_preds + 0.1 * linear_BG_pred

Predictions_df = pd.DataFrame(
    {
        "id": test_id_df["id"],
        "formation_energy_ev_natom": xgb_EF_preds,
        "bandgap_energy_ev": stacked_BG,
    }
)

output_path = "Stacked_Nomad.csv"
Predictions_df.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
