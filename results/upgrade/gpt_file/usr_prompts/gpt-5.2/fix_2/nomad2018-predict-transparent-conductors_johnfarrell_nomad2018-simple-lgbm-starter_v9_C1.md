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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.07016

# 6. Current score

0.05796

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.05796) has done: 'I fix the LightGBM API breakage causing the crash by replacing deprecated `verbose_eval`/`early_stopping_rounds` arguments in `lgb.cv` (LightGBM v4+) with callback-based equivalents, keeping the same training/cv logic. I also make the one-hot encoding robust by aligning train/test dummy columns (some spacegroups can appear only in one split), preventing shape mismatch at predict time. Finally, I ensure the submission is always written with a valid `.csv` filename (no problematic characters) and that predictions are clipped to be non-negative before `expm1` inversion to avoid invalid log-RMSLE behavior. These are execution/stability fixes and should keep the intended score behavior while producing a valid submission file end-to-end.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _resolve_path(rel_like):
    candidates = [
        rel_like,
        rel_like.replace("../input", "/kaggle/input"),
        rel_like.replace("../input", "/kaggle/data"),
        "/kaggle/data/nomad2018-predict-transparent-conductors/"
        + os.path.basename(rel_like),
        "/kaggle/input/nomad2018-predict-transparent-conductors/"
        + os.path.basename(rel_like),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return rel_like


train_path = _resolve_path("../input/train.csv")
test_path = _resolve_path("../input/test.csv")
sample_sub_path = _resolve_path("../input/sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
print(train.shape)
train.head(10)



## === cell 1
print(test.shape)
test.head(10)



## === cell 2
target_fe = np.log1p(train.formation_energy_ev_natom)
target_be = np.log1p(train.bandgap_energy_ev)

train_ids = train["id"].copy()
test_ids = test["id"].copy()

del (
    train["formation_energy_ev_natom"],
    train["bandgap_energy_ev"],
    train["id"],
    test["id"],
)



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

train, test = train.align(test, join="outer", axis=1, fill_value=0)



## === cell 6
import lightgbm as lgb
import multiprocessing


def cv_train_model(
    X,
    y,
    verbose_eval=None,
    early_stopping_rounds=None,
    params=None,
):
    if isinstance(y, pd.core.frame.DataFrame) or isinstance(y, pd.Series):
        y = np.asarray(y).ravel()

    dstrain = lgb.Dataset(X, label=y)

    max_boost_round = 4000
    if params is None:
        lgb_params = {
            "objective": "regression_l2",
            "learning_rate": 0.008,
            "num_threads": 4,  # keep as original intent
            "max_depth": 4,
            "min_data_in_leaf": 23,
            "feature_fraction": 0.93,
            "bagging_fraction": 0.93,
            "bagging_freq": 1,
            "lambda_l2": 1e2,
            "metric": ["mse"],
            "verbosity": -1,
        }
    else:
        lgb_params = params

    print("lgb cv and training...")
    if verbose_eval is None:
        verbose_eval = int(max_boost_round / 30)
    if early_stopping_rounds is None:
        early_stopping_rounds = int(max_boost_round / 10)

    callbacks = []
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        callbacks.append(
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
        )
    if verbose_eval is not None and verbose_eval != 0:
        callbacks.append(lgb.log_evaluation(period=verbose_eval))

    cv_lgb = lgb.cv(
        lgb_params,
        dstrain,
        num_boost_round=max_boost_round,
        nfold=10,
        stratified=False,
        callbacks=callbacks,
        seed=233,
        return_cvbooster=False,
    )

    if "l2-mean" in cv_lgb:
        key = "l2-mean"
    else:
        mean_keys = [k for k in cv_lgb.keys() if k.endswith("-mean")]
        if not mean_keys:
            raise RuntimeError(f"Unexpected cv result keys: {list(cv_lgb.keys())}")
        key = mean_keys[0]

    best_round = int(
        np.argmin(cv_lgb[key]) + 1
    )  # +1 because num_boost_round is 1-indexed in practice
    best_cv_mean = float(np.min(cv_lgb[key]))
    print("best round", best_round)
    print("best mse-mean", best_cv_mean)

    model_lgb = lgb.train(
        lgb_params,
        dstrain,
        num_boost_round=best_round,
        valid_sets=[dstrain],
        callbacks=(
            [lgb.log_evaluation(period=verbose_eval)]
            if (verbose_eval is not None and verbose_eval != 0)
            else None
        ),
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




## === cell 7
best_cv_mean_fe, feat_weight_fe, model_lgb_fe = get_model_cv(train, target_fe)

pred_fe_log = model_lgb_fe.predict(test)
pred_fe = np.expm1(np.clip(pred_fe_log, a_min=-20, a_max=20))
pred_fe = np.maximum(pred_fe, 0.0)

best_cv_mean_be, feat_weight_be, model_lgb_be = get_model_cv(train, target_be)
pred_be_log = model_lgb_be.predict(test)
pred_be = np.expm1(np.clip(pred_be_log, a_min=-20, a_max=20))
pred_be = np.maximum(pred_be, 0.0)

scr_total = float(np.mean([np.sqrt(best_cv_mean_fe), np.sqrt(best_cv_mean_be)]))
print(f"total cv score: {scr_total}")

sub = pd.read_csv(sample_sub_path)
sub["formation_energy_ev_natom"] = pred_fe
sub["bandgap_energy_ev"] = pred_be

out_path = f"submission_{scr_total:.6f}.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", sub.shape)
print(sub.head())



## === cell 8
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

rg = RidgeCV(alphas=[0.003, 0.01, 0.3, 3, 10], cv=5)


def kfold_cv(
    X,
    y,
    test,
    n_splits=10,
    train_lgb=True,
    lgb_ratio=0.8,
    cv_pred_test=False,
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=233)
    cv_pred = np.zeros((len(test)))
    scr_total = 0.0
    for fold_id, (tr_idx, te_idx) in enumerate(kf.split(y)):
        print(f"Starting No.{fold_id} fold CV out of {n_splits} ...")
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        rg.fit(X_tr_sc, y_tr)
        pred_rg = rg.predict(X_te_sc)
        mse = mean_squared_error(y_te, pred_rg)
        print("=======rg mse:", mse)

        if train_lgb is True:
            _, _, model_lgb = get_model_cv(X_tr, y_tr, False, 0)
            print("=======lgb mse:", mean_squared_error(y_te, model_lgb.predict(X_te)))
            avg_mse = mean_squared_error(
                y_te,
                pred_rg * (1 - lgb_ratio) + model_lgb.predict(X_te) * lgb_ratio,
            )
            print(f"=======avg mse: {avg_mse}")
            scr_total += avg_mse / n_splits
        else:
            scr_total += mse / n_splits

        if cv_pred_test is True:
            if train_lgb is True:
                test_sc = scaler.transform(test)
                cv_pred += (
                    rg.predict(test_sc) * (1 - lgb_ratio)
                    + model_lgb.predict(test) * lgb_ratio
                ) / n_splits
            else:
                test_sc = scaler.transform(test)
                cv_pred += rg.predict(test_sc) / n_splits

    print(f"score total: {scr_total}")
    if not cv_pred_test:
        return scr_total
    else:
        return scr_total, np.expm1(np.clip(cv_pred, a_min=-20, a_max=20))
