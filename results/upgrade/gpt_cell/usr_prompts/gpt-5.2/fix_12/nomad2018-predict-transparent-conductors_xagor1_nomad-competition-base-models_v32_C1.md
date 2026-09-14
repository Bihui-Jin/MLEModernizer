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

0.05855

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.08665) has done: 'The crash happens because `test_examples_transform` becomes empty in cell 7: earlier cell 5 slices by fixed indices (2400:3000), but your actual `train.csv` has 2160 rows and `test.csv` has 240 rows. In cell 7 you correctly compute `n_train`/`n_test`, but `test_examples_transform` is already empty and stays empty after slicing. The minimal fix is to rebuild `training_examples_transform` and `test_examples_transform` from `features_transform_df` using `n_train`/`n_test` so they always have the correct number of rows. This preserves the same model/training logic and keeps outputs (`linear_BG_pred`, `linear_EF_pred`, `Predictions_df`) compatible for later cells.'
- What this solution (achieved 0.05873) has done: 'The crash happens because `training_examples` and `test_examples` were created earlier using hard-coded row slices (0:2400, 2400:3000), but the actual dataset has 2160 train rows and 240 test rows. As a result, `training_examples` has 2400 rows while the label vector has 2160 elements, so `xgb.DMatrix` raises an “Incorrect size for labels” error. The minimal fix is to re-derive `training_examples` and `test_examples` inside cell 8 using `n_train`/`n_test` (already computed in cell 7) so feature rows align with labels. This preserves the same model logic and parameters while making the data shapes consistent.'
- What this solution (achieved 0.06208) has done: 'Your current score (0.05873) is better than the target (0.06949) on a lower-is-better metric, so the goal is to slightly worsen performance into the target tolerance band with minimal risk. The smallest legitimate knob that preserves core logic is to adjust the final stacking weights (cell 13) to lean more on the weaker linear model, which should increase RMSLE in a controlled way while keeping the same models and training intact. I also add a tiny safety clip to keep predictions non-negative (RMSLE requires non-negative), which prevents accidental invalid submissions without meaningfully changing behavior. Everything else (features, models, training) stays identical, and the script still writes the three submission CSVs.'
- What this solution (achieved 0.08337) has done: 'Your current score (0.06208) is better than the target (0.06949) on a lower-is-better metric, so we should *slightly worsen* performance into the target band with the smallest, safest change. The minimal legitimate knob is the final stacking weights: leaning a bit more on the weaker linear model should increase RMSLE in a controlled way while keeping the same models, features, and training intact. I only adjust the blend coefficients (and keep the existing non-negativity clipping to ensure RMSLE-valid submissions). Everything else remains unchanged and it still write the three submission CSVs.'
- What this solution (achieved 0.08337) has done: 'Your current score (0.08337) is worse than the target (0.06949) for a lower-is-better metric, so we should make a minimal change that legitimately improves RMSLE without changing the core modeling approach. The smallest, safest knob is the final stacking weights: shift the blend to rely more on the stronger XGBoost predictions and less on the weaker linear model. I keep the same models, features, training, and non-negativity clipping; only the blend coefficients change. This should move the score downward (improve) toward the target band while preserving the pipeline and producing the same submission files.'
- What this solution (achieved 0.05868) has done: 'Your current score (0.08337) is worse than the target (0.06949) for a lower-is-better metric, so we should make the smallest legitimate change that tends to improve RMSLE without altering the core models or training. The safest minimal knob here is the final blending in cell 13: rely a bit more on the stronger XGBoost predictions and less on the weaker linear model. This keeps the exact same feature engineering, linear regressions, XGBoost parameters, and training procedure; only the ensemble weights change. I also keep the existing non-negativity clipping since RMSLE requires non-negative predictions and this avoids invalid submissions.'
- What this solution (achieved 0.08337) has done: 'Your current score (0.05868) is better than the target (0.06949) on a lower-is-better metric, so we should make a small, legitimate change that slightly worsens performance into the target tolerance band without altering the models, features, or training. The safest minimal knob is the final blend in the stacked submission: shift weight away from the stronger XGBoost predictions and toward the weaker linear predictions. I only adjust the stacking coefficients and keep the existing non-negativity clipping (RMSLE-valid) and submission-writing intact. Everything else remains identical to preserve core logic and runtime behavior.'
- What this solution (achieved 0.08337) has done: 'Your current score (0.08337, lower-is-better) is worse than the target (0.06949), so we should make the smallest reliable improvement without changing the models or training procedure. The most direct minimal knob is the final blend: increase reliance on the stronger XGBoost predictions and reduce the linear contribution. I only adjust the stacking weights in cell 13 (keeping the same features, models, and clipping), which should move RMSLE down toward the target band. The script still run end-to-end and write the same three submission CSVs.'
- What this solution (achieved 0.05855) has done: 'Your current score (0.08337, lower-is-better) is worse than the target (0.06949), so we should make a minimal, low-risk change that tends to improve RMSLE without changing the feature engineering, models, or training loops. The simplest legitimate knob is the final blend in cell 13: rely slightly more on the stronger XGBoost predictions and slightly less on the weaker linear predictions. I keep the non-negativity clipping (RMSLE requires non-negative) and keep writing the same three submission CSVs. Everything else remains identical to preserve core logic and runtime behavior.'

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

from sklearn.model_selection import train_test_split
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
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import seaborn as sns

print(os.listdir("../input"))
from sklearn import tree

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectFromModel
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings("ignore")



## === cell 1
path = "../input/"
train_df = pd.read_csv(path + "/train.csv")
test_df = pd.read_csv(path + "/test.csv")



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
training_examples = features_df.iloc[0:2400].copy()
test_examples = features_df.iloc[2400:3000].copy()
training_examples_transform = features_transform_df.iloc[0:2400].copy()
test_examples_transform = features_transform_df.iloc[2400:3000].copy()




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
n_train = len(Targets_df)
n_test = len(test_id_df)

training_examples_transform = features_transform_df.iloc[:n_train].copy()
test_examples_transform = features_transform_df.iloc[n_train : n_train + n_test].copy()

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

linear_EF_pred = np.clip(linear_EF_pred, 0, None)
linear_BG_pred = np.clip(linear_BG_pred, 0, None)

Predictions_df = pd.DataFrame()
Predictions_df["id"] = test_id_df["id"].copy()
Predictions_df["formation_energy_ev_natom"] = linear_EF_pred
Predictions_df["bandgap_energy_ev"] = linear_BG_pred
Predictions_df.to_csv("Linear_Nomad.csv", index=False)



## === cell 8
n_train = len(Targets_df)
n_test = len(test_id_df)

training_examples = features_df.iloc[:n_train].copy()
test_examples = features_df.iloc[n_train : n_train + n_test].copy()

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
model_xgb.loc[30:, ["test-rmse-mean", "train-rmse-mean"]].plot()
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
)
model_xgb.fit(training_examples, training_targets)
xgb.plot_importance(model_xgb)

xgb_BG_preds = np.expm1(model_xgb.predict(test_examples))

xgb_BG_preds = np.clip(xgb_BG_preds, 0, None)



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
model_xgb.loc[30:, ["test-rmse-mean", "train-rmse-mean"]].plot()
last = len(model_xgb.loc[:]) - 1
print(model_xgb.loc[last:, ["test-rmse-mean"]])



## === cell 11
model_xgb = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.08,
    gamma=0,
    colsample=1,
    colsample_bytree=0.4,
    min_child_weight=3,
)
model_xgb.fit(training_examples, training_targets)
xgb.plot_importance(model_xgb)

xgb_EF_preds = np.expm1(model_xgb.predict(test_examples))

xgb_EF_preds = np.clip(xgb_EF_preds, 0, None)



## === cell 12
Predictions_df = pd.DataFrame()
Predictions_df["id"] = test_id_df["id"].copy()
Predictions_df["formation_energy_ev_natom"] = xgb_EF_preds
Predictions_df["bandgap_energy_ev"] = xgb_BG_preds
Predictions_df.to_csv("XGB_Nomad.csv", index=False)



## === cell 13
Predictions_df = pd.DataFrame()
Predictions_df["id"] = test_id_df["id"].copy()

stacked_EF_preds = 0.92 * xgb_EF_preds + 0.08 * linear_EF_pred
stacked_BG_preds = 0.92 * xgb_BG_preds + 0.08 * linear_BG_pred

stacked_EF_preds = np.clip(stacked_EF_preds, 0, None)
stacked_BG_preds = np.clip(stacked_BG_preds, 0, None)

Predictions_df["formation_energy_ev_natom"] = stacked_EF_preds
Predictions_df["bandgap_energy_ev"] = stacked_BG_preds
Predictions_df.to_csv("Stacked_Nomad.csv", index=False)
