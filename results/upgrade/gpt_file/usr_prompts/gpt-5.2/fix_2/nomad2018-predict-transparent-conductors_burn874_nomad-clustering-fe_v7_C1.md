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

0.05826

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.05826) has done: 'I remove the dependency on missing external “feature-engineering” and “cluster” datasets by switching the data loading to the provided competition `train.csv`/`test.csv` paths, while keeping the same LightGBM CV training approach and parameters. I fix the KFold usage (enable shuffling for a deterministic split with `random_state`) and update the LightGBM training call to be compatible with LightGBM 4.6.0 using callbacks for early stopping and logging. To preserve the original “two separate models” core logic, I train one model for formation energy and one for bandgap on the same base features, and I clip predictions to be positive to avoid RMSLE issues. Finally, I write a valid `submission.csv` with exactly the required columns and correct `id` alignment from `test.csv`.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error as msle

RANDOM_STATE = 2319
np.random.seed(RANDOM_STATE)



## === cell 1

BASE_INPUT = "/kaggle/input/nomad2018-predict-transparent-conductors"
train_path = os.path.join(BASE_INPUT, "train.csv")
test_path = os.path.join(BASE_INPUT, "test.csv")
sample_sub_path = os.path.join(BASE_INPUT, "sample_submission.csv")

if not (
    os.path.exists(train_path)
    and os.path.exists(test_path)
    and os.path.exists(sample_sub_path)
):
    BASE_INPUT = "/kaggle/data/nomad2018-predict-transparent-conductors"
    train_path = os.path.join(BASE_INPUT, "train.csv")
    test_path = os.path.join(BASE_INPUT, "test.csv")
    sample_sub_path = os.path.join(BASE_INPUT, "sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

train1 = train_df.copy()
train2 = train_df.copy()
test = test_df.copy()



## === cell 2
categorical = ["spacegroup", "number_of_total_atoms"]

for c in train1.columns:
    if c in categorical:
        train1[c] = train1[c].astype("category")
    else:
        train1[c] = pd.to_numeric(train1[c], errors="coerce")

for c in train2.columns:
    if c in categorical:
        train2[c] = train2[c].astype("category")
    else:
        train2[c] = pd.to_numeric(train2[c], errors="coerce")

for c in test.columns:
    if c in categorical:
        test[c] = test[c].astype("category")
    else:
        test[c] = pd.to_numeric(test[c], errors="coerce")

assert train1["id"].notna().all()
assert test["id"].notna().all()



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
    "feature_fraction_seed": RANDOM_STATE,
    "bagging_seed": RANDOM_STATE,
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

getVal1 = np.zeros(len(train))
predictions1 = np.zeros(len(test))

print("LightGBM Model - formation_energy_ev_natom")
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train[features], formation)):
    X_tr, y_tr = train.iloc[trn_idx][features], formation.iloc[trn_idx]
    X_valid, y_valid = train.iloc[val_idx][features], formation.iloc[val_idx]

    print(f"Fold idx:{fold_ + 1}")
    trn_data = lgb.Dataset(
        X_tr,
        label=y_tr,
        categorical_feature=[c for c in categorical if c in features],
        free_raw_data=False,
    )
    val_data = lgb.Dataset(
        X_valid,
        label=y_valid,
        categorical_feature=[c for c in categorical if c in features],
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

    oof_pred = clf.predict(X_valid, num_iteration=clf.best_iteration)
    test_pred = clf.predict(test[features], num_iteration=clf.best_iteration)

    getVal1[val_idx] += oof_pred
    predictions1 += test_pred / folds.n_splits

oof1_clip = np.clip(getVal1, 1e-9, None)
y1_clip = np.clip(formation.values, 1e-9, None)
print("CV RMSLE: {:<8.5f}".format(np.sqrt(msle(y1_clip, oof1_clip))))



## === cell 6
train = train2
folds = KFold(n_splits=num_folds, shuffle=True, random_state=RANDOM_STATE)

getVal2 = np.zeros(len(train))
predictions2 = np.zeros(len(test))

print("LightGBM Model - bandgap_energy_ev")
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train[features], bandgap)):
    X_tr, y_tr = train.iloc[trn_idx][features], bandgap.iloc[trn_idx]
    X_valid, y_valid = train.iloc[val_idx][features], bandgap.iloc[val_idx]

    print(f"Fold idx:{fold_ + 1}")
    trn_data = lgb.Dataset(
        X_tr,
        label=y_tr,
        categorical_feature=[c for c in categorical if c in features],
        free_raw_data=False,
    )
    val_data = lgb.Dataset(
        X_valid,
        label=y_valid,
        categorical_feature=[c for c in categorical if c in features],
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

    oof_pred = clf.predict(X_valid, num_iteration=clf.best_iteration)
    test_pred = clf.predict(test[features], num_iteration=clf.best_iteration)

    getVal2[val_idx] += oof_pred
    predictions2 += test_pred / folds.n_splits

oof2_clip = np.clip(getVal2, 1e-9, None)
y2_clip = np.clip(bandgap.values, 1e-9, None)
print("CV RMSLE: {:<8.5f}".format(np.sqrt(msle(y2_clip, oof2_clip))))



## === cell 7

sub = sample_sub.copy()

sub["id"] = test["id"].values

sub["formation_energy_ev_natom"] = np.clip(predictions1, 1e-9, None)
sub["bandgap_energy_ev"] = np.clip(predictions2, 1e-9, None)

sub = sub[["id", "formation_energy_ev_natom", "bandgap_energy_ev"]]
sub.to_csv("submission.csv", index=False)

train_oof = train_df.copy()
train_oof["formation_energy_ev_natom"] = getVal1
train_oof["bandgap_energy_ev"] = getVal2
train_oof.to_csv("new_train1.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
