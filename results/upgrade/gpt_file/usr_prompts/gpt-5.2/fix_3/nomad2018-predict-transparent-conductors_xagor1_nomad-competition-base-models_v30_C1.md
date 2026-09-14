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

0.50313

# 6. Current score

0.07484

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.08337) has done: 'I fix the import errors caused by deprecated scikit-learn modules and missing symbols so the notebook runs end-to-end. I also correct the train/test row slicing bug (your code assumes 2400/3000 rows, but the dataset is 2160/240) so models train on the full training set and predict for the full test set. Finally, because the metric is RMSLE, I ensure predictions are strictly non-negative before saving the submission to avoid invalid/penalized log errors, while keeping the same overall modeling approach (linear + XGB + simple stacking). The script write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.07484) has done: 'Your current score (0.08337, lower-is-better) is already much better than the target (0.50313), so to move *toward* the target we should intentionally and minimally degrade performance without breaking validity. The smallest safe lever in your existing core logic is the final stacking weights: shifting weight away from the stronger XGB predictions toward the weaker linear predictions increase RMSLE while keeping the same models, features, and training loops. I also keep the non-negativity clamp (required for RMSLE stability) and ensure the submission schema stays identical. The patch only changes the blend weights in the final cell and writes the same `submission.csv`.'

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

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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

for p in [
    "../input",
    "/kaggle/input",
    "/kaggle/data/input",
    "/kaggle/data/nomad2018-predict-transparent-conductors",
]:
    if os.path.exists(p):
        try:
            print("Listing:", p)
            print(os.listdir(p)[:20])
        except Exception as e:
            print("Could not list", p, "due to", repr(e))



## === cell 1
path = "../input"
train_path = os.path.join(path, "train.csv")
test_path = os.path.join(path, "test.csv")

if not os.path.exists(train_path) or not os.path.exists(test_path):
    alt = os.path.join(path, "nomad2018-predict-transparent-conductors")
    if os.path.exists(os.path.join(alt, "train.csv")):
        path = alt
        train_path = os.path.join(path, "train.csv")
        test_path = os.path.join(path, "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)



## === cell 2
print("Training data shape", "\n")
print(train_df.shape, "\n")
print("Testing data shape", "\n")
print(test_df.shape, "\n")

print("Training columns", "\n")
print(train_df.columns, "\n")
print("Testing columns", "\n")
print(test_df.columns, "\n")



## === cell 3
Targets_df = pd.DataFrame()
Targets_df["bandgap_energy_ev"] = train_df["bandgap_energy_ev"].copy()
Targets_df["formation_energy_ev_natom"] = train_df["formation_energy_ev_natom"].copy()
train_df = train_df.drop(["formation_energy_ev_natom", "bandgap_energy_ev"], axis=1)

train_id_df = pd.DataFrame()
train_id_df["id"] = train_df["id"].copy()
train_df = train_df.drop(["id"], axis=1)

test_id_df = pd.DataFrame()
test_id_df["id"] = test_df["id"].copy()
test_df = test_df.drop(["id"], axis=1)

combined_df = pd.concat([train_df, test_df], ignore_index=True)
print("Total number of null values in the df", "\n")
print(combined_df.isna().sum().sum())



## === cell 4
numerical_df = pd.DataFrame.copy(
    combined_df[
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
    ]
)

one_hot_df = pd.DataFrame.copy(combined_df[["spacegroup"]])

one_hot_df = pd.get_dummies(one_hot_df, prefix=["spacegroup"], columns=["spacegroup"])

features_df = pd.concat([numerical_df, one_hot_df], axis=1)

skewed_feats = numerical_df.skew()
skewed_feats = skewed_feats[skewed_feats > 0.1]
skewed_feats = skewed_feats.index

unskewed_feats = numerical_df.skew()
unskewed_feats = unskewed_feats[unskewed_feats < 0.1]
unskewed_feats = unskewed_feats.index

transform_df = pd.DataFrame()
denom = (
    numerical_df[unskewed_feats].max() - numerical_df[unskewed_feats].min()
).replace(0, 1.0)
transform_df[unskewed_feats] = (
    numerical_df[unskewed_feats] - numerical_df[unskewed_feats].mean()
) / denom
transform_df[skewed_feats] = np.log1p(numerical_df[skewed_feats])

features_transform_df = pd.concat([transform_df, one_hot_df], axis=1)
features_df = pd.concat([numerical_df, one_hot_df], axis=1)



## === cell 5
n_train = Targets_df.shape[0]
n_total = combined_df.shape[0]
n_test = n_total - n_train
print("n_train:", n_train, "n_test:", n_test, "n_total:", n_total)

training_examples = features_df.iloc[:n_train].copy()
test_examples = features_df.iloc[n_train:].copy()

training_examples_transform = features_transform_df.iloc[:n_train].copy()
test_examples_transform = features_transform_df.iloc[n_train:].copy()

assert training_examples.shape[0] == n_train
assert test_examples.shape[0] == n_test
assert test_id_df.shape[0] == n_test




## === cell 6
def rmsle_cv(model, X, y):
    rmsle = np.sqrt(
        -cross_val_score(model, X, y, scoring="neg_mean_squared_log_error", cv=5)
    )
    return rmsle




## === cell 7
training_targets = Targets_df["bandgap_energy_ev"].copy()
model_linear_bg = LinearRegression().fit(training_examples_transform, training_targets)
linear_BG_pred = model_linear_bg.predict(test_examples_transform)
BG_rmsle = rmsle_cv(
    model_linear_bg, training_examples_transform, training_targets
).mean()
print("Band gap RMSLE:")
print(BG_rmsle, "\n")

training_targets = Targets_df["formation_energy_ev_natom"].copy()
model_linear_fe = LinearRegression().fit(training_examples_transform, training_targets)
linear_EF_pred = model_linear_fe.predict(test_examples_transform)
EF_rmsle = rmsle_cv(
    model_linear_fe, training_examples_transform, training_targets
).mean()
print("Formation Energy RMSLE:")
print(EF_rmsle, "\n")

print("Expected combined RMSLE")
combined_rmsle = (EF_rmsle + BG_rmsle) / 2
print(combined_rmsle)

linear_EF_pred = np.maximum(linear_EF_pred, 0.0)
linear_BG_pred = np.maximum(linear_BG_pred, 0.0)

Predictions_df = pd.DataFrame()
Predictions_df["id"] = test_id_df["id"].copy()
Predictions_df["formation_energy_ev_natom"] = linear_EF_pred
Predictions_df["bandgap_energy_ev"] = linear_BG_pred
Predictions_df.to_csv("Linear_Nomad.csv", index=False)

print("Wrote Linear_Nomad.csv with shape:", Predictions_df.shape)



## === cell 8
training_targets = Targets_df["bandgap_energy_ev"].copy()

model_xgb_bg = xgb.XGBRegressor(
    max_depth=2,
    learning_rate=0.1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=1,
    min_child_weight=10,
    n_estimators=500,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=4,
)
model_xgb_bg.fit(training_examples, training_targets)

xgb_BG_preds = model_xgb_bg.predict(test_examples)

xgb_BG_preds = np.maximum(xgb_BG_preds, 0.0)



## === cell 9
training_targets = Targets_df["formation_energy_ev_natom"].copy()

model_xgb_fe = xgb.XGBRegressor(
    max_depth=4,
    learning_rate=0.08,
    gamma=0,
    colsample_bytree=0.4,
    min_child_weight=3,
    subsample=0.8,
    n_estimators=800,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=4,
)
model_xgb_fe.fit(training_examples, training_targets)

xgb_EF_preds = model_xgb_fe.predict(test_examples)
xgb_EF_preds = np.maximum(xgb_EF_preds, 0.0)



## === cell 10
Predictions_df = pd.DataFrame()
Predictions_df["id"] = test_id_df["id"].copy()
Predictions_df["formation_energy_ev_natom"] = xgb_EF_preds
Predictions_df["bandgap_energy_ev"] = xgb_BG_preds
Predictions_df.to_csv("XGB_Nomad.csv", index=False)

print("Wrote XGB_Nomad.csv with shape:", Predictions_df.shape)



## === cell 11
Predictions_df = pd.DataFrame()
Predictions_df["id"] = test_id_df["id"].copy()

stack_w_xgb = 0.20
stack_w_lin = 0.80

stacked_EF_preds = stack_w_xgb * xgb_EF_preds + stack_w_lin * linear_EF_pred
stacked_BG_preds = stack_w_xgb * xgb_BG_preds + stack_w_lin * linear_BG_pred

stacked_EF_preds = np.maximum(stacked_EF_preds, 0.0)
stacked_BG_preds = np.maximum(stacked_BG_preds, 0.0)

Predictions_df["formation_energy_ev_natom"] = stacked_EF_preds
Predictions_df["bandgap_energy_ev"] = stacked_BG_preds

Predictions_df.to_csv("submission.csv", index=False)
Predictions_df.to_csv("Stacked_Nomad.csv", index=False)

print("Wrote submission.csv with shape:", Predictions_df.shape)
print("Stack weights -> XGB:", stack_w_xgb, "Linear:", stack_w_lin)
print(Predictions_df.head())
