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

0.2217017047010093

# 6. Current score

0.05836

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.05836) has done: 'I replace the missing external feature files with the competition-provided `train.csv`/`test.csv` so the notebook can run in this environment. To preserve your core modeling logic (two separate LightGBM regressors with the same params and KFold CV), I keep the same training loops and only adjust the data-loading and categorical handling so LightGBM receives valid dtypes. I also fix `KFold` to use `shuffle=True` (required when setting `random_state`) and update the LightGBM early-stopping arguments for LightGBM 4.6.0 compatibility. Finally, I ensure predictions are clipped to be non-negative (required for RMSLE/log-based metrics) and write a valid `submission.csv` with the exact required columns.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error as msle

import os
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 2319



## === cell 1
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/nomad2018-predict-transparent-conductors",
    "/kaggle/data/nomad2018-predict-transparent-conductors",
    "/kaggle/input",
    "/kaggle/data",
]


def _find_file(filename):
    for root in DATA_ROOT_CANDIDATES:
        p = os.path.join(root, filename)
        if os.path.exists(p):
            return p
    if os.path.exists(filename):
        return filename
    raise FileNotFoundError(f"Could not find {filename} under {DATA_ROOT_CANDIDATES}")


train_path = _find_file("train.csv")
test_path = _find_file("test.csv")
sub_path = _find_file("sample_submission.csv")

train_df = pd.read_csv(train_path)
test = pd.read_csv(test_path)

train1 = train_df.copy()
train2 = train_df.copy()

train1["cluster"] = 0
train2["cluster"] = 0
test["cluster"] = 0



## === cell 2
categorical = [
    "spacegroup",
    "number_of_total_atoms",
    "cluster",
    "a1",
    "a2",
    "a3",
    "a4",
    "a4",
    "a6",
    "a7",
    "a8",
    "a9",
    "a10",
    "a11",
    "a12",
    "a13",
    "a14",
    "a15",
    "SG_12",
    "SG_33",
    "SG_167",
    "SG_194",
    "SG_206",
    "SG_227",
]


def cast_types(df, categorical_list):
    for c in df.columns:
        if c in categorical_list:
            df[c] = df[c].astype("category")
        else:
            if c == "id":
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


train1 = cast_types(train1, categorical)
train2 = cast_types(train2, categorical)
test = cast_types(test, categorical)



## === cell 3
formation = train1["formation_energy_ev_natom"]
bandgap = train2["bandgap_energy_ev"]



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
    "seed": RANDOM_STATE,
}



## === cell 5
num_folds = 11
train = train1
features = [
    c
    for c in train.columns
    if c not in ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
]

folds = KFold(n_splits=num_folds, shuffle=True, random_state=RANDOM_STATE)
getVal1 = np.zeros(len(train), dtype=float)
predictions1 = np.zeros(len(test), dtype=float)

print("Light GBM Model - formation_energy_ev_natom")
for fold_, (trn_idx, val_idx) in enumerate(
    folds.split(train[features], formation.values)
):
    X_tr, y_tr = train.iloc[trn_idx][features], formation.iloc[trn_idx]
    X_valid, y_valid = train.iloc[val_idx][features], formation.iloc[val_idx]

    print("Fold idx:{}".format(fold_ + 1))
    trn_data = lgb.Dataset(
        X_tr, label=y_tr, categorical_feature="auto", free_raw_data=False
    )
    val_data = lgb.Dataset(
        X_valid, label=y_valid, categorical_feature="auto", free_raw_data=False
    )

    clf = lgb.train(
        param,
        trn_data,
        num_boost_round=1000000,
        valid_sets=[trn_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=200, first_metric_only=False, verbose=True
            ),
            lgb.log_evaluation(period=100),
        ],
    )

    getVal1[val_idx] = clf.predict(X_valid, num_iteration=clf.best_iteration)
    predictions1 += (
        clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits
    )

getVal1_clip = np.clip(getVal1, 0, None)
formation_clip = np.clip(formation.values, 0, None)
print("CV RMSLE: {:<8.5f}".format(np.sqrt(msle(formation_clip, getVal1_clip))))



## === cell 6
train = train2

folds = KFold(n_splits=num_folds, shuffle=True, random_state=RANDOM_STATE)
getVal2 = np.zeros(len(train), dtype=float)
predictions2 = np.zeros(len(test), dtype=float)

print("Light GBM Model - bandgap_energy_ev")
for fold_, (trn_idx, val_idx) in enumerate(
    folds.split(train[features], bandgap.values)
):
    X_tr, y_tr = train.iloc[trn_idx][features], bandgap.iloc[trn_idx]
    X_valid, y_valid = train.iloc[val_idx][features], bandgap.iloc[val_idx]

    print("Fold idx:{}".format(fold_ + 1))
    trn_data = lgb.Dataset(
        X_tr, label=y_tr, categorical_feature="auto", free_raw_data=False
    )
    val_data = lgb.Dataset(
        X_valid, label=y_valid, categorical_feature="auto", free_raw_data=False
    )

    clf = lgb.train(
        param,
        trn_data,
        num_boost_round=1000000,
        valid_sets=[trn_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=200, first_metric_only=False, verbose=True
            ),
            lgb.log_evaluation(period=100),
        ],
    )

    getVal2[val_idx] = clf.predict(X_valid, num_iteration=clf.best_iteration)
    predictions2 += (
        clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits
    )

getVal2_clip = np.clip(getVal2, 0, None)
bandgap_clip = np.clip(bandgap.values, 0, None)
print("CV RMSLE: {:<8.5f}".format(np.sqrt(msle(bandgap_clip, getVal2_clip))))



## === cell 7
sub = pd.read_csv(sub_path)

pred1 = np.clip(predictions1, 0, None)
pred2 = np.clip(predictions2, 0, None)

sub["formation_energy_ev_natom"] = pred1
sub["bandgap_energy_ev"] = pred2
sub.to_csv("submission.csv", index=False)

train_oof = train1.copy()
train_oof["formation_energy_ev_natom"] = getVal1_clip
train_oof["bandgap_energy_ev"] = getVal2_clip
train_oof.to_csv("new_train1.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
