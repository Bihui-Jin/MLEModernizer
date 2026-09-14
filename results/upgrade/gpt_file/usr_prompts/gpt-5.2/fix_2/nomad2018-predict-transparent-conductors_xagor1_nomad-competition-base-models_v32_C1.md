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

0.06949

# 6. Current score

0.08337

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.08337) has done: 'I fix the import/runtime errors caused by deprecated scikit-learn modules (`sklearn.cross_validation`, `sklearn.grid_search`) and missing definitions after the first cell fails (which cascades into `LinearRegression`/`xgb` not being defined). I also correct the train/test row slicing bug (the code hard-codes 2400/3000 rows while the dataset is 2160/240), which currently would mis-split data even if it ran. Finally, I make the submission generation robust by clipping predictions to be non-negative (needed for RMSLE) and writing a valid `.csv` with the required column names.'

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

CANDIDATE_INPUT_DIRS = [
    "../input/nomad2018-predict-transparent-conductors",
    "../input",
    "/kaggle/input/nomad2018-predict-transparent-conductors",
    "/kaggle/input",
    "/kaggle/data/nomad2018-predict-transparent-conductors",
    "/kaggle/data",
]


def _pick_input_dir(cands):
    for d in cands:
        try:
            if os.path.exists(d) and any(
                os.path.exists(os.path.join(d, f)) for f in ["train.csv", "test.csv"]
            ):
                return d
        except Exception:
            pass
    return "."


INPUT_DIR = _pick_input_dir(CANDIDATE_INPUT_DIRS)

print("Using INPUT_DIR:", INPUT_DIR)
try:
    print("INPUT_DIR listing (first 20):", sorted(os.listdir(INPUT_DIR))[:20])
except Exception as e:
    print("Could not list INPUT_DIR:", e)



## === cell 1
path = INPUT_DIR
train_df = pd.read_csv(os.path.join(path, "train.csv"))
test_df = pd.read_csv(os.path.join(path, "test.csv"))



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
transform_df[unskewed_feats] = (
    numerical_df[unskewed_feats] - numerical_df[unskewed_feats].mean()
) / (numerical_df[unskewed_feats].max() - numerical_df[unskewed_feats].min())
transform_df[skewed_feats] = np.log1p(numerical_df[skewed_feats])

features_transform_df = pd.concat([transform_df, one_hot_df], axis=1)
features_df = pd.concat([numerical_df, one_hot_df], axis=1)



## === cell 5
n_train = len(Targets_df)
training_examples = features_df.iloc[:n_train].copy()
test_examples = features_df.iloc[n_train:].copy()
training_examples_transform = features_transform_df.iloc[:n_train].copy()
test_examples_transform = features_transform_df.iloc[n_train:].copy()

print("n_train:", n_train, "n_test:", len(test_examples))




## === cell 6
def rmsle_cv(model):
    rmsle = np.sqrt(
        -cross_val_score(
            model,
            training_examples_transform,
            training_targets,
            scoring="neg_mean_squared_log_error",
            cv=5,
        )
    )
    return rmsle




## === cell 7
training_targets = Targets_df["bandgap_energy_ev"].copy()
model_linear = LinearRegression().fit(training_examples_transform, training_targets)
linear_BG_pred = model_linear.predict(test_examples_transform)
BG_rmsle = rmsle_cv(model_linear).mean()
print("Band gap RMSLE:")
print(BG_rmsle, "\n")

training_targets = Targets_df["formation_energy_ev_natom"].copy()
model_linear = LinearRegression().fit(training_examples_transform, training_targets)
linear_EF_pred = model_linear.predict(test_examples_transform)
EF_rmsle = rmsle_cv(model_linear).mean()
print("Formation Energy RMSLE:")
print(EF_rmsle, "\n")

print("Expected combined RMSLE")
combined_rmsle = (EF_rmsle + BG_rmsle) / 2
print(combined_rmsle)

Predictions_df = pd.DataFrame()
Predictions_df["id"] = test_id_df["id"].copy()
Predictions_df["formation_energy_ev_natom"] = np.clip(linear_EF_pred, 0, None)
Predictions_df["bandgap_energy_ev"] = np.clip(linear_BG_pred, 0, None)
Predictions_df.to_csv("Linear_Nomad.csv", index=False)

print("Wrote:", "Linear_Nomad.csv", "shape:", Predictions_df.shape)



## === cell 8
training_targets = np.log1p(Targets_df["bandgap_energy_ev"].copy())
dtrain = xgb.DMatrix(training_examples, label=training_targets)
dtest = xgb.DMatrix(test_examples)
params = {
    "max_depth": 2,
    "eta": 0.1,
    "gamma": 0,
    "subsample": 0.8,
    "colsample_bytree": 1,
    "min_child_weight": 10,
}
model_xgb = xgb.cv(params, dtrain, num_boost_round=500, early_stopping_rounds=100)
try:
    model_xgb.loc[30:, ["test-rmse-mean", "train-rmse-mean"]].plot()
    plt.show()
except Exception:
    pass
last = len(model_xgb.loc[:]) - 1
print(model_xgb.loc[last:, ["test-rmse-mean"]])



## === cell 9
model_xgb = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=2,
    learning_rate=0.1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=1,
    min_child_weight=10,
    objective="reg:squarederror",
    random_state=0,
)
model_xgb.fit(training_examples, training_targets)
try:
    xgb.plot_importance(model_xgb)
    plt.show()
except Exception:
    pass

xgb_BG_preds = np.expm1(model_xgb.predict(test_examples))



## === cell 10
training_targets = np.log1p(Targets_df["formation_energy_ev_natom"].copy())
dtrain = xgb.DMatrix(training_examples, label=training_targets)
dtest = xgb.DMatrix(test_examples)
params = {
    "max_depth": 4,
    "eta": 0.08,
    "gamma": 0,
    "subsample": 1,
    "colsample_bytree": 0.4,
    "min_child_weight": 3,
}
model_xgb = xgb.cv(params, dtrain, num_boost_round=500, early_stopping_rounds=100)
try:
    model_xgb.loc[30:, ["test-rmse-mean", "train-rmse-mean"]].plot()
    plt.show()
except Exception:
    pass
last = len(model_xgb.loc[:]) - 1
print(model_xgb.loc[last:, ["test-rmse-mean"]])



## === cell 11
model_xgb = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.08,
    gamma=0,
    colsample_bytree=0.4,
    min_child_weight=3,
    subsample=1.0,
    objective="reg:squarederror",
    random_state=0,
)
model_xgb.fit(training_examples, training_targets)
try:
    xgb.plot_importance(model_xgb)
    plt.show()
except Exception:
    pass

xgb_EF_preds = np.expm1(model_xgb.predict(test_examples))



## === cell 12
Predictions_df = pd.DataFrame()
Predictions_df["id"] = test_id_df["id"].copy()
Predictions_df["formation_energy_ev_natom"] = np.clip(xgb_EF_preds, 0, None)
Predictions_df["bandgap_energy_ev"] = np.clip(xgb_BG_preds, 0, None)
Predictions_df.to_csv("XGB_Nomad.csv", index=False)

print("Wrote:", "XGB_Nomad.csv", "shape:", Predictions_df.shape)



## === cell 13
Predictions_df = pd.DataFrame()
Predictions_df["id"] = test_id_df["id"].copy()
stacked_EF_preds = 0.9 * xgb_EF_preds + 0.1 * linear_EF_pred
stacked_BG_preds = 0.9 * xgb_BG_preds + 0.1 * linear_BG_pred

Predictions_df["formation_energy_ev_natom"] = np.clip(stacked_EF_preds, 0, None)
Predictions_df["bandgap_energy_ev"] = np.clip(stacked_BG_preds, 0, None)

Predictions_df = Predictions_df[
    ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
]
Predictions_df.to_csv("Stacked_Nomad.csv", index=False)

print("Wrote:", "Stacked_Nomad.csv", "shape:", Predictions_df.shape)
print(Predictions_df.head())
