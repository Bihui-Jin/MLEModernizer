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
lightgbm==4.6.0
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
seaborn==0.12.2
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

0.0702

# 6. Current score

0.05814

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.05814) has done: 'I fix the LightGBM CV metric key mismatch that causes the `KeyError` by detecting the correct key returned by `lgb.cv` (it can be `l2-mean` or `valid l2-mean` depending on LightGBM version) and using it consistently. I also make the input path resolution robust to your provided directory structure so the code always finds `train.csv/test.csv/sample_submission.csv` in Kaggle. Finally, I keep the modeling/training logic the same (still MSE on log1p targets with CV-picked boosting rounds) and ensure a valid `submission.csv` is always written with the required columns.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
import multiprocessing

CANDIDATE_INPUT_DIRS = [
    "../input/nomad2018-predict-transparent-conductors",
    "/kaggle/input/nomad2018-predict-transparent-conductors",
    "/kaggle/data/nomad2018-predict-transparent-conductors",
    "../input",
    "/kaggle/input",
    "/kaggle/data",
]

BASE_INPUT = None
for d in CANDIDATE_INPUT_DIRS:
    if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
        os.path.join(d, "test.csv")
    ):
        BASE_INPUT = d
        break

if BASE_INPUT is None:
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv in any of the candidate input dirs: "
        + ", ".join(CANDIDATE_INPUT_DIRS)
    )

train = pd.read_csv(os.path.join(BASE_INPUT, "train.csv"))
test = pd.read_csv(os.path.join(BASE_INPUT, "test.csv"))

print("Using BASE_INPUT:", BASE_INPUT)
print(train.shape)
train.head(10)



## === cell 1
print(test.shape)
test.head(10)



## === cell 2
target_fe = np.log1p(train["formation_energy_ev_natom"].astype(float))
target_be = np.log1p(train["bandgap_energy_ev"].astype(float))

train_id = train["id"].copy()
test_id = test["id"].copy()

train = train.drop(["formation_energy_ev_natom", "bandgap_energy_ev", "id"], axis=1)
test = test.drop(["id"], axis=1)



## === cell 3
sorted(train["spacegroup"].unique())



## === cell 4
sorted(test["spacegroup"].unique())



## === cell 5
train = pd.concat(
    [
        train.drop(["spacegroup"], axis=1),
        pd.get_dummies(train["spacegroup"], prefix="SG"),
    ],
    axis=1,
)
test = pd.concat(
    [
        test.drop(["spacegroup"], axis=1),
        pd.get_dummies(test["spacegroup"], prefix="SG"),
    ],
    axis=1,
)

train, test = train.align(test, join="left", axis=1, fill_value=0)




## === cell 6
def _pick_cv_metric_key(cv_result, preferred_substr=("l2-mean", "mse-mean")):
    """
    LightGBM's lgb.cv key naming differs across versions, e.g.:
      - 'l2-mean' / 'l2-stdv'
      - 'valid l2-mean' / 'valid l2-stdv'
    This selects the right mean key robustly without changing training semantics.
    """
    keys = list(cv_result.keys())
    for substr in preferred_substr:
        if substr in cv_result:
            return substr
        for k in keys:
            if substr in k and k.endswith("mean"):
                return k
    for k in keys:
        if k.endswith("mean"):
            return k
    raise KeyError(
        f"Could not find a '*mean' metric key in lgb.cv output. Keys: {keys}"
    )


def cv_train_model(
    X,
    y,
    verbose_eval=None,
    early_stopping_rounds=None,
    params=None,
):
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = np.asarray(y).ravel()

    dstrain = lgb.Dataset(X, label=y)

    max_boost_round = 3000
    if params is None:
        lgb_params = {
            "objective": "regression_l2",
            "learning_rate": 0.008,
            "num_threads": 4,  # keep as original intent for stability
            "max_depth": 4,
            "feature_fraction": 0.93,
            "bagging_fraction": 0.93,
            "bagging_freq": 1,
            "lambda_l2": 1e2,
            "metric": ["mse"],  # keep same intent; LightGBM may report as l2 internally
            "verbosity": -1,
        }
    else:
        lgb_params = params

    print("lgb cv and training...")

    if verbose_eval is None:
        verbose_eval = int(max_boost_round / 30)
    if early_stopping_rounds is None:
        early_stopping_rounds = int(max_boost_round / 10)

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=verbose_eval),
    ]

    cv_lgb = lgb.cv(
        lgb_params,
        dstrain,
        num_boost_round=max_boost_round,
        nfold=10,
        stratified=False,
        callbacks=callbacks,
        seed=42,
    )

    mean_key = _pick_cv_metric_key(cv_lgb, preferred_substr=("l2-mean", "mse-mean"))
    best_round = int(np.argmin(cv_lgb[mean_key]) + 1)  # +1 because rounds are 1-indexed
    best_cv_mean = float(np.min(cv_lgb[mean_key]))

    print("cv metric key:", mean_key)
    print("best round", best_round)
    print("best mse/l2-mean", best_cv_mean)

    model_lgb = lgb.train(
        lgb_params,
        dstrain,
        num_boost_round=best_round,
        valid_sets=[dstrain],
        callbacks=[lgb.log_evaluation(period=verbose_eval)],
    )

    print("lgb cv and training finished...")
    return model_lgb, best_cv_mean


def get_feat_weight(model_lgb, feat_names, plot=True):
    feat_weight = pd.DataFrame(
        model_lgb.feature_importance(),
        columns=["feature_importance"],
        index=feat_names,
    )
    if plot:
        indices = np.argsort(feat_weight["feature_importance"])[::-1]
        plt.figure(figsize=(12, 6))
        plt.title("feature importance (lightgbm)")
        plt.bar(range(len(feat_weight)), list(feat_weight.iloc[indices, 0]))
        plt.xticks(
            range(len(feat_weight)),
            feat_weight.iloc[indices].index,
            rotation="vertical",
        )
        plt.xlim([-1, len(feat_weight)])
        plt.show()
    return feat_weight


def get_model_cv(df, y, plot=True, verbose_eval=False):
    model_lgb, best_cv_mean = cv_train_model(df, y, verbose_eval=verbose_eval)
    feat_weight = get_feat_weight(model_lgb, feat_names=df.columns, plot=plot)
    print("best cv mean", best_cv_mean)
    return best_cv_mean, feat_weight, model_lgb


best_cv_mean_fe, feat_weight_fe, model_lgb_fe = get_model_cv(
    train, target_fe, plot=False, verbose_eval=100
)
pred_fe = np.expm1(model_lgb_fe.predict(test))

best_cv_mean_be, feat_weight_be, model_lgb_be = get_model_cv(
    train, target_be, plot=False, verbose_eval=100
)
pred_be = np.expm1(model_lgb_be.predict(test))

scr_total = np.mean([np.sqrt(best_cv_mean_fe), np.sqrt(best_cv_mean_be)])
print(f"total cv score: {scr_total}")

sub_path = os.path.join(BASE_INPUT, "sample_submission.csv")
if not os.path.exists(sub_path):
    alt = os.path.join(os.path.dirname(BASE_INPUT), "sample_submission.csv")
    if os.path.exists(alt):
        sub_path = alt
    else:
        raise FileNotFoundError(
            f"Could not find sample_submission.csv at {sub_path} or {alt}"
        )

sub = pd.read_csv(sub_path)
sub["formation_energy_ev_natom"] = pred_fe
sub["bandgap_energy_ev"] = pred_be

out_name = "submission.csv"
sub.to_csv(out_name, index=False)
print(f"Wrote {out_name} with shape {sub.shape} and columns {list(sub.columns)}")
print(sub.head())
