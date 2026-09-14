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

3.7

# 3. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

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

0.0688157060447855

# 6. Current score

0.05833

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.05833) has done: 'I fix the broken input paths by switching to the dataset that actually exists in this environment (`../input/nomad2018-predict-transparent-conductors/train.csv` and `test.csv`) and remove the missing external cluster-label dependency while keeping the same LightGBM CV training approach. I also fix a syntax bug in the `categorical` list and make KFold deterministic and valid under scikit-learn by enabling `shuffle=True` when providing `random_state`. To better match the competition’s RMSLE metric without changing the model type/loop, I train on `log1p(target)` and apply `expm1` at inference, plus clip predictions to be non-negative (RMSLE requirement). Finally, I ensure the submission has the exact required columns and is written to `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse

import os
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 2319



## === cell 1
BASE_PATH = "../input/nomad2018-predict-transparent-conductors"

train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

if "cluster" in train.columns:
    train = train.drop(columns=["cluster"])
if "cluster" in test.columns:
    test = test.drop(columns=["cluster"])



## === cell 2
categorical = ["spacegroup"]

for c in train.columns:
    if c in categorical:
        train[c] = train[c].astype("category")
    elif c not in ["formation_energy_ev_natom", "bandgap_energy_ev"]:
        train[c] = pd.to_numeric(train[c], errors="coerce")

for c in test.columns:
    if c in categorical:
        test[c] = test[c].astype("category")
    else:
        test[c] = pd.to_numeric(test[c], errors="coerce")



## === cell 3
formation = train["formation_energy_ev_natom"].astype(float)
bandgap = train["bandgap_energy_ev"].astype(float)

y1 = np.log1p(np.clip(formation.values, 0, None))
y2 = np.log1p(np.clip(bandgap.values, 0, None))



## === cell 4
param = {
    "num_leaves": 7,
    "objective": "regression",
    "min_data_in_leaf": 18,
    "learning_rate": 0.04,
    "feature_fraction": 0.93,
    "bagging_fraction": 0.93,
    "bagging_freq": 1,
    "metric": "l2",
    "num_threads": 1,
    "verbosity": -1,
    "seed": RANDOM_STATE,
}



## === cell 5
num_folds = 20
features = [
    c
    for c in train.columns
    if c not in ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
]

folds = KFold(n_splits=num_folds, shuffle=True, random_state=RANDOM_STATE)

getVal1 = np.zeros(len(train), dtype=float)
predictions1 = np.zeros(len(test), dtype=float)

print("LightGBM Model - formation_energy_ev_natom (log1p target)")
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train[features], y1)):
    X_tr, y_tr = train.iloc[trn_idx][features], y1[trn_idx]
    X_valid, y_valid = train.iloc[val_idx][features], y1[val_idx]

    print("Fold idx:{}".format(fold_ + 1))
    trn_data = lgb.Dataset(
        X_tr,
        label=y_tr,
        categorical_feature=categorical if categorical else "auto",
        free_raw_data=False,
    )
    val_data = lgb.Dataset(
        X_valid,
        label=y_valid,
        categorical_feature=categorical if categorical else "auto",
        free_raw_data=False,
    )

    clf = lgb.train(
        param,
        trn_data,
        num_boost_round=1000000,
        valid_sets=[trn_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    getVal1[val_idx] += clf.predict(X_valid, num_iteration=clf.best_iteration)
    predictions1 += (
        clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits
    )

print("CV RMSE on log1p(formation): {:<8.5f}".format(np.sqrt(mse(y1, getVal1))))



## === cell 6
folds = KFold(n_splits=num_folds, shuffle=True, random_state=RANDOM_STATE)

getVal2 = np.zeros(len(train), dtype=float)
predictions2 = np.zeros(len(test), dtype=float)

print("LightGBM Model - bandgap_energy_ev (log1p target)")
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train[features], y2)):
    X_tr, y_tr = train.iloc[trn_idx][features], y2[trn_idx]
    X_valid, y_valid = train.iloc[val_idx][features], y2[val_idx]

    print("Fold idx:{}".format(fold_ + 1))
    trn_data = lgb.Dataset(
        X_tr,
        label=y_tr,
        categorical_feature=categorical if categorical else "auto",
        free_raw_data=False,
    )
    val_data = lgb.Dataset(
        X_valid,
        label=y_valid,
        categorical_feature=categorical if categorical else "auto",
        free_raw_data=False,
    )

    clf = lgb.train(
        param,
        trn_data,
        num_boost_round=1000000,
        valid_sets=[trn_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    getVal2[val_idx] += clf.predict(X_valid, num_iteration=clf.best_iteration)
    predictions2 += (
        clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits
    )

print("CV RMSE on log1p(bandgap): {:<8.5f}".format(np.sqrt(mse(y2, getVal2))))



## === cell 7
sub = sample_sub.copy()

sub["formation_energy_ev_natom"] = np.expm1(predictions1)
sub["bandgap_energy_ev"] = np.expm1(predictions2)
sub["formation_energy_ev_natom"] = sub["formation_energy_ev_natom"].clip(lower=0)
sub["bandgap_energy_ev"] = sub["bandgap_energy_ev"].clip(lower=0)

if not sub["id"].equals(test["id"]):
    pred_df = pd.DataFrame(
        {
            "id": test["id"].values,
            "formation_energy_ev_natom": sub["formation_energy_ev_natom"].values,
            "bandgap_energy_ev": sub["bandgap_energy_ev"].values,
        }
    )
    sub = sample_sub[["id"]].merge(pred_df, on="id", how="left")

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
