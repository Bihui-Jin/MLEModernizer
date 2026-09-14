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

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.65708) has done: 'I fix the KFold initialization by enabling shuffling (so the provided random_state is valid) in both training loops, and renumber the notebook cells to start from 1 while preserving all original logic. This eliminates the ValueError, allows the models to train, and ensures the prediction arrays are defined for creating a proper `submission.csv` file.'
- What this solution (achieved 0.65708) has done: 'I remove the unsupported `verbose_eval` argument from the LightGBM `train` calls and replace it with the `log_evaluation` callback (which provides the same periodic logging). This fixes the TypeError that stopped training, allowing both models to run and produce predictions, and consequently a valid `submission.csv`. No other logic is changed, preserving the original training strategy and keeping the score close to the current level.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error as msle

import warnings

warnings.filterwarnings("ignore")



## === cell 1
base_dir = "/kaggle/input/nomad2018-predict-transparent-conductors"

train_path = os.path.join(base_dir, "train.csv")
test_path = os.path.join(base_dir, "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)



## === cell 2
categorical = ["spacegroup"]

for df in [train, test]:
    for c in df.columns:
        if c in categorical:
            df[c] = df[c].astype("category")
        else:
            df[c] = df[c].astype("float64")

train["atom_fraction_sum"] = (
    train["percent_atom_al"] + train["percent_atom_ga"] + train["percent_atom_in"]
)
test["atom_fraction_sum"] = (
    test["percent_atom_al"] + test["percent_atom_ga"] + test["percent_atom_in"]
)

train["lattice_vector_sum"] = (
    train["lattice_vector_1_ang"]
    + train["lattice_vector_2_ang"]
    + train["lattice_vector_3_ang"]
)
test["lattice_vector_sum"] = (
    test["lattice_vector_1_ang"]
    + test["lattice_vector_2_ang"]
    + test["lattice_vector_3_ang"]
)

train["lattice_angle_mean"] = (
    train["lattice_angle_alpha_degree"]
    + train["lattice_angle_beta_degree"]
    + train["lattice_angle_gamma_degree"]
) / 3.0
test["lattice_angle_mean"] = (
    test["lattice_angle_alpha_degree"]
    + test["lattice_angle_beta_degree"]
    + test["lattice_angle_gamma_degree"]
) / 3.0

train["lattice_vector_mean"] = train["lattice_vector_sum"] / 3.0
test["lattice_vector_mean"] = test["lattice_vector_sum"] / 3.0

train["lattice_angle_std"] = np.sqrt(
    (
        (train["lattice_angle_alpha_degree"] - train["lattice_angle_mean"]) ** 2
        + (train["lattice_angle_beta_degree"] - train["lattice_angle_mean"]) ** 2
        + (train["lattice_angle_gamma_degree"] - train["lattice_angle_mean"]) ** 2
    )
    / 3.0
)
test["lattice_angle_std"] = np.sqrt(
    (
        (test["lattice_angle_alpha_degree"] - test["lattice_angle_mean"]) ** 2
        + (test["lattice_angle_beta_degree"] - test["lattice_angle_mean"]) ** 2
        + (test["lattice_angle_gamma_degree"] - test["lattice_angle_mean"]) ** 2
    )
    / 3.0
)

train["atom_fraction_mean"] = train["atom_fraction_sum"] / 3.0
test["atom_fraction_mean"] = test["atom_fraction_sum"] / 3.0



## === cell 3
param = {
    "num_leaves": 1024,  # increased capacity
    "objective": "regression",
    "min_data_in_leaf": 10,  # allow finer leaves
    "learning_rate": 0.005,  # slower, more precise learning
    "feature_fraction": 0.8,  # a bit stronger column subsampling
    "bagging_fraction": 0.7,  # stronger row subsampling
    "bagging_freq": 1,
    "metric": "l2",
    "num_threads": 1,
    "verbosity": -1,
    "lambda_l2": 0.1,
}



## === cell 4
formation = train["formation_energy_ev_natom"]
bandgap = train["bandgap_energy_ev"]

formation_log = np.log1p(formation)
bandgap_log = np.log1p(bandgap)



## === cell 5
num_folds = 11
features = [
    c
    for c in train.columns
    if c not in ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
]

folds = KFold(n_splits=num_folds, random_state=2319, shuffle=True)

getVal1 = np.zeros(len(train))
predictions1 = np.zeros(len(test))

print("Training LightGBM for formation_energy_ev_natom")
for fold_, (trn_idx, val_idx) in enumerate(
    folds.split(train.values, formation_log.values)
):
    X_tr, y_tr = train.iloc[trn_idx][features], formation_log.iloc[trn_idx]
    X_valid, y_valid = train.iloc[val_idx][features], formation_log.iloc[val_idx]

    trn_data = lgb.Dataset(
        X_tr, label=y_tr, categorical_feature=categorical, free_raw_data=False
    )
    val_data = lgb.Dataset(
        X_valid, label=y_valid, categorical_feature=categorical, free_raw_data=False
    )

    clf = lgb.train(
        param,
        trn_data,
        num_boost_round=1_000_000,
        valid_sets=[trn_data, val_data],
        callbacks=[
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=200, verbose=False),
        ],
    )

    val_pred = np.expm1(
        clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    )
    test_pred = np.expm1(clf.predict(test[features], num_iteration=clf.best_iteration))

    getVal1[val_idx] += val_pred
    predictions1 += test_pred / folds.n_splits

predictions1 = np.maximum(predictions1, 0)

pred_min, pred_max = (
    train["formation_energy_ev_natom"].min(),
    train["formation_energy_ev_natom"].max(),
)
predictions1 = np.clip(predictions1, pred_min, pred_max)

print(
    "CV RMSLE (formation): {:<8.5f}".format(
        np.sqrt(
            msle(
                np.maximum(formation.values, 1e-6),
                np.maximum(getVal1, 1e-6),
            )
        )
    )
)



## === cell 6
getVal2 = np.zeros(len(train))
predictions2 = np.zeros(len(test))

print("Training LightGBM for bandgap_energy_ev")
for fold_, (trn_idx, val_idx) in enumerate(
    folds.split(train.values, bandgap_log.values)
):
    X_tr, y_tr = train.iloc[trn_idx][features], bandgap_log.iloc[trn_idx]
    X_valid, y_valid = train.iloc[val_idx][features], bandgap_log.iloc[val_idx]

    trn_data = lgb.Dataset(
        X_tr, label=y_tr, categorical_feature=categorical, free_raw_data=False
    )
    val_data = lgb.Dataset(
        X_valid, label=y_valid, categorical_feature=categorical, free_raw_data=False
    )

    clf = lgb.train(
        param,
        trn_data,
        num_boost_round=1_000_000,
        valid_sets=[trn_data, val_data],
        callbacks=[
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=200, verbose=False),
        ],
    )

    val_pred = np.expm1(
        clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    )
    test_pred = np.expm1(clf.predict(test[features], num_iteration=clf.best_iteration))

    getVal2[val_idx] += val_pred
    predictions2 += test_pred / folds.n_splits

predictions2 = np.maximum(predictions2, 0)

pred_min2, pred_max2 = (
    train["bandgap_energy_ev"].min(),
    train["bandgap_energy_ev"].max(),
)
predictions2 = np.clip(predictions2, pred_min2, pred_max2)

print(
    "CV RMSLE (bandgap): {:<8.5f}".format(
        np.sqrt(
            msle(
                np.maximum(bandgap.values, 1e-6),
                np.maximum(getVal2, 1e-6),
            )
        )
    )
)



## === cell 7
submission = pd.DataFrame(
    {
        "id": test["id"].astype(int),
        "formation_energy_ev_natom": predictions1,
        "bandgap_energy_ev": predictions2,
    }
)

submission.to_csv("submission.csv", index=False)

train_out = train.copy()
train_out["formation_energy_ev_natom"] = getVal1
train_out["bandgap_energy_ev"] = getVal2
train_out.to_csv("new_train1.csv", index=False)
