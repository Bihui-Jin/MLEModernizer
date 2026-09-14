# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Predict the `scalar_coupling_constant` between atom pairs in molecules, given the two atom types (e.g., C and H), the coupling type (e.g., `2JHC`), and any features you are able to create from the molecule structure (`xyz`) files.

## Metric
Log of the Mean Absolute Error, calculated for each scalar coupling type, and then averaged across types.

## Submission Format
```
id,scalar_coupling_constant
2324604,0.0
2324605,0.0
2324606,0.0
etc.
```

## Dataset
The training and test splits are by *molecule*, so that no molecule in the training data is found in the test data.

- **train.csv** - the training set, where the first column (`molecule_name`) is the name of the molecule where the coupling constant originates (the corresponding XYZ file is located at ./structures/.xyz), the second (`atom_index_0`) and third column (`atom_index_1`) is the atom indices of the atom-pair creating the coupling and the fourth column (`scalar_coupling_constant`) is the scalar coupling constant that we want to be able to predict
- **test.csv** - the test set; same info as train, without the target variable
- **sample_submission.csv** - a sample submission file in the correct format
- **structures.zip** - folder containing molecular structure (xyz) files, where the first line is the number of atoms in the molecule, followed by a blank line, and then a line for every atom, where the first column contains the atomic element (H for hydrogen, C for carbon etc.) and the remaining columns contain the X, Y and Z cartesian coordinates (a standard format for chemists and molecular visualization programs)
- **structures.csv** - this file contains the **same** information as the individual xyz structure files, but in a single file
- **dipole_moments.csv** - contains the molecular electric dipole moments. These are three dimensional vectors that indicate the charge distribution in the molecule. The first column (`molecule_name`) are the names of the molecule, the second to fourth column are the `X`, `Y` and `Z` components respectively of the dipole moment.
- **magnetic_shielding_tensors.csv** - contains the magnetic shielding tensors for all atoms in the molecules. The first column (`molecule_name`) contains the molecule name, the second column (`atom_index`) contains the index of the atom in the molecule, the third to eleventh columns contain the `XX`, `YX`, `ZX`, `XY`, `YY`, `ZY`, `XZ`, `YZ` and `ZZ` elements of the tensor/matrix respectively.
- **mulliken_charges.csv** - contains the mulliken charges for all atoms in the molecules. The first column (`molecule_name`) contains the name of the molecule, the second column (`atom_index`) contains the index of the atom in the molecule, the third column (`mulliken_charge`) contains the mulliken charge of the atom.
- **potential_energy.csv** - contains the potential energy of the molecules. The first column (`molecule_name`) contains the name of the molecule, the second column (`potential_energy`) contains the potential energy of the molecule.
- **scalar_coupling_contributions.csv** - The scalar coupling constants in `train.csv` (or corresponding files) are a sum of four terms. `scalar_coupling_contributions.csv` contain all these terms. The first column (`molecule_name`) are the name of the molecule, the second (`atom_index_0`) and third column (`atom_index_1`) are the atom indices of the atom-pair, the fourth column indicates the type of coupling, the fifth column (`fc`) is the Fermi Contact contribution, the sixth column (`sd`) is the Spin-dipolar contribution, the seventh column (`pso`) is the Paramagnetic spin-orbit contribution and the eighth column (`dso`) is the Diamagnetic spin-orbit contribution.

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (111 lines)
            dipole_moments.csv (76511 lines)
            dipole_moments.csv.zip (892.4 kB)
            magnetic_shielding_tensors.csv (1379965 lines)
            magnetic_shielding_tensors.csv.zip (47.9 MB)
            mulliken_charges.csv (1379965 lines)
            mulliken_charges.csv.zip (9.5 MB)
            potential_energy.csv (76511 lines)
            potential_energy.csv.zip (641.9 kB)
            sample_submission.csv (467814 lines)
            sample_submission.csv.zip (846.9 kB)
            scalar_coupling_contributions.csv (4191264 lines)
            scalar_coupling_contributions.csv.zip (90.0 MB)
            structures.csv (1379965 lines)
            structures.csv.zip (33.0 MB)
            structures.zip (44.3 MB)
            test.csv (467814 lines)
            test.csv.zip (2.6 MB)
            train.csv (4191264 lines)
            train.csv.zip (43.6 MB)
            champs-scalar-coupling/
                description.md (111 lines)
                dipole_moments.csv (76511 lines)
                ... and 18 other files
                champs-scalar-coupling/
                structures/
                    dsgdb9nsd_000001.xyz (212 Bytes)
                    dsgdb9nsd_000002.xyz (171 Bytes)
                    ... and 76508 other files
            structures/
                dsgdb9nsd_000001.xyz (212 Bytes)
                dsgdb9nsd_000002.xyz (171 Bytes)
                ... and 76508 other files
        input/
            description.md (111 lines)
            dipole_moments.csv (76511 lines)
            dipole_moments.csv.zip (892.4 kB)
            magnetic_shielding_tensors.csv (1379965 lines)
            magnetic_shielding_tensors.csv.zip (47.9 MB)
            mulliken_charges.csv (1379965 lines)
            mulliken_charges.csv.zip (9.5 MB)
            potential_energy.csv (76511 lines)
            potential_energy.csv.zip (641.9 kB)
            sample_submission.csv (467814 lines)
            sample_submission.csv.zip (846.9 kB)
            scalar_coupling_contributions.csv (4191264 lines)
            scalar_coupling_contributions.csv.zip (90.0 MB)
            structures.csv (1379965 lines)
            structures.csv.zip (33.0 MB)
            structures.zip (44.3 MB)
            test.csv (467814 lines)
            test.csv.zip (2.6 MB)
            train.csv (4191264 lines)
            train.csv.zip (43.6 MB)
            champs-scalar-coupling/
                description.md (111 lines)
                dipole_moments.csv (76511 lines)
                ... and 18 other files
                champs-scalar-coupling/
                structures/
                    dsgdb9nsd_000001.xyz (212 Bytes)
                    dsgdb9nsd_000002.xyz (171 Bytes)
                    ... and 76508 other files
            structures/
                dsgdb9nsd_000001.xyz (212 Bytes)
                dsgdb9nsd_000002.xyz (171 Bytes)
                ... and 76508 other files
        working/
            champs-scalar-coupling/
                description.md (111 lines)
                dipole_moments.csv (76511 lines)
                ... and 18 other files
                champs-scalar-coupling/
                structures/
                    dsgdb9nsd_000001.xyz (212 Bytes)
                    dsgdb9nsd_000002.xyz (171 Bytes)
                    ... and 76508 other files
```

-> data/champs-scalar-coupling/dipole_moments.csv has 76510 rows and 4 columns.
The columns are: molecule_name, X, Y, Z

-> data/champs-scalar-coupling/magnetic_shielding_tensors.csv has 1379964 rows and 11 columns.
The columns are: molecule_name, atom_index, XX, YX, ZX, XY, YY, ZY, XZ, YZ, ZZ

-> data/champs-scalar-coupling/mulliken_charges.csv has 1379964 rows and 3 columns.
The columns are: molecule_name, atom_index, mulliken_charge

-> data/champs-scalar-coupling/potential_energy.csv has 76510 rows and 2 columns.
The columns are: molecule_name, potential_energy

-> data/champs-scalar-coupling/sample_submission.csv has 467813 rows and 2 columns.
The columns are: id, scalar_coupling_constant

-> data/champs-scalar-coupling/scalar_coupling_contributions.csv has 4191263 rows and 8 columns.
The columns are: molecule_name, atom_index_0, atom_index_1, type, fc, sd, pso, dso

-> data/champs-scalar-coupling/structures.csv has 1379964 rows and 6 columns.
The columns are: molecule_name, atom_index, atom, x, y, z

-> data/champs-scalar-coupling/test.csv has 467813 rows and 5 columns.
The columns are: id, molecule_name, atom_index_0, atom_index_1, type

-> data/champs-scalar-coupling/train.csv has 4191263 rows and 6 columns.
The columns are: id, molecule_name, atom_index_0, atom_index_1, type, scalar_coupling_constant

-> (stopped after 10 files for performance)

# 5. Target score

-2.108374155517372

# 6. Current score

1.99777

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.99777) has done: 'Implemented fixes to resolve import errors, TensorFlow session initialization, missing tqdm, and undefined variables. Added proper TensorFlow‑v2 compatibility imports, corrected session configuration, ensured required libraries are loaded, stored test IDs before scaling, and created a final submission CSV (`final_sub.csv`) from averaged NN predictions. Also safeguarded optional plotting code and maintained original workflow structure.'
- What this solution (achieved 1.99777) has done: 'The fix replaces the deprecated `K.set_session` call (which caused an AttributeError) with the TF‑v2 compatible `tf.compat.v1.keras.backend.set_session`. This resolves the import error, allows the rest of the pipeline to run, and ensures a `final_sub.csv` is generated.'
- What this solution (achieved 1.99777) has done: 'Fix import issues, bypass TensorFlow when unavailable, safely handle missing LightGBM, and replace the neural‑network predictions with an ensemble average of the pre‑computed OOF/model predictions. This removes the session‑setting error, ensures the script runs end‑to‑end, and produces a valid `final_sub.csv` whose predictions are based on the available model outputs, moving the score toward the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
import os
import copy
import gc
import random
from tqdm import tqdm

warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Input, BatchNormalization
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import callbacks
    from tensorflow.keras import backend as K

    TF_AVAILABLE = True
except Exception:
    tf = None
    TF_AVAILABLE = False

try:
    import lightgbm as lgb

    LGB_AVAILABLE = True
except Exception:
    lgb = None
    LGB_AVAILABLE = False




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def calc_logmae(y_val, y_pred):
    return np.log(np.sum(np.abs(y_val - y_pred) / len(y_pred)))


def permutation_importance(
    model, X_val, y_val, calc_logmae, threshold=0.01, minimize=True, verbose=True
):
    results = {}
    y_pred = model.predict(X_val)
    results["base_score"] = calc_logmae(y_val, y_pred)
    if verbose:
        print(f'Base score {results["base_score"]:.5}')
    for col in tqdm(X_val.columns):
        freezed_col = X_val[col].copy()
        X_val[col] = np.random.permutation(X_val[col])
        preds = model.predict(X_val)
        results[col] = calc_logmae(y_val, preds)
        X_val[col] = freezed_col
        if verbose:
            print(f"column: {col} - {results[col]:.5}")
    if minimize:
        bad_features = [
            k for k in results if results[k] < results["base_score"] + threshold
        ]
    else:
        bad_features = [
            k for k in results if results[k] > results["base_score"] + threshold
        ]
    bad_features.remove("base_score")
    return results, bad_features


def load_lgb_params(mol_type, n_estimators=2000):
    seed = 2319
    param_1J = {
        "num_leaves": int(0.7 * (25**2)),
        "learning_rate": 0.1,
        "feature_fraction": 1,
        "save_binary": True,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "drop_seed": seed,
        "data_random_seed": seed,
        "objective": "regression_l2",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "metric": "mae",
        "is_unbalance": True,
        "boost_from_average": "false",
        "bagging_fraction": 1,
        "bagging_freq": 0,
        "lambda_l1": 0.5,
        "lambda_l2": 1.7244553699717466,
        "max_bin": 238,
        "max_depth": 25,
        "n_estimators": n_estimators,
        "sparse_threshold": 1.0,
        "n_jobs": 6,
    }
    param_2J = {
        "num_leaves": int(1 * (20**2)),
        "learning_rate": 0.1,
        "feature_fraction": 1,
        "save_binary": True,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "drop_seed": seed,
        "data_random_seed": seed,
        "objective": "regression_l2",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "metric": "mae",
        "is_unbalance": True,
        "boost_from_average": "false",
        "bagging_fraction": 1,
        "bagging_freq": int(0),
        "lambda_l1": 1,
        "lambda_l2": 1.89,
        "max_bin": 255,
        "max_depth": 20,
        "min_data_in_leaf": int(10),
        "min_gain_to_split": 0,
        "min_sum_hessian_in_leaf": 1 / 869,
        "n_estimators": n_estimators,
        "sparse_threshold": 1.0,
        "n_jobs": 6,
    }
    param_3J = {
        "num_leaves": int(1 * (30**2)),
        "learning_rate": 0.1,
        "feature_fraction": 1,
        "save_binary": True,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "drop_seed": seed,
        "data_random_seed": seed,
        "objective": "regression_l2",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "metric": "mae",
        "is_unbalance": True,
        "boost_from_average": "false",
        "bagging_fraction": 1,
        "bagging_freq": int(0),
        "lambda_l1": 0.5,
        "lambda_l2": 1,
        "max_bin": 50,
        "max_depth": 20,
        "min_data_in_leaf": int(10),
        "min_gain_to_split": 0,
        "min_sum_hessian_in_leaf": 1 / 202,
        "n_estimators": n_estimators,
        "sparse_threshold": 1.0,
        "n_jobs": 6,
    }
    if mol_type[0] == "1":
        return param_1J
    if mol_type[0] == "2":
        return param_2J
    if mol_type[0] == "3":
        return param_3J
    return {}


def create_nn_model(input_shape):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available")
    inp = Input(shape=(input_shape,))
    x = Dense(2048, activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation="relu")(x)
    x = BatchNormalization()(x)
    out = Dense(5, activation="linear")(x)
    model = Model(inputs=inp, outputs=out)
    return model


def select_best_features(file_folder, train, scalar_coupling_contributions):
    if not LGB_AVAILABLE:
        print("LightGBM not available – using all features.")
        return [
            c
            for c in train.columns
            if c
            not in [
                "scalar_coupling_constant",
                "id",
                "molecule_name",
                "type",
                "aromaticity_vec_1",
                "aromaticity_vec_0",
                "fc",
                "sd",
                "pso",
                "dso",
            ]
        ]

    train, index_folds = preprocess_train_data(
        mol_type,
        train,
        scalar_coupling_contributions,
        k_folds=1,
        val_data_ratio_if_no_kfolds=0.05,
        verbose=True,
    )
    trn_idx, val_idx = index_folds[0]
    mol_features = [
        c
        for c in train.columns
        if c
        not in [
            "scalar_coupling_constant",
            "id",
            "molecule_name",
            "type",
            "aromaticity_vec_1",
            "aromaticity_vec_0",
            "fc",
            "sd",
            "pso",
            "dso",
        ]
    ]
    X_t, X_v, y_t, y_v = split_train_and_val_data(train, trn_idx, val_idx, mol_features)
    lgb_params = load_lgb_params(mol_type, n_estimators=1000)
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_t,
        y_t["scalar_coupling_constant"],
        eval_set=[
            (X_t, y_t["scalar_coupling_constant"]),
            (X_v, y_v["scalar_coupling_constant"]),
        ],
        eval_metric="mae",
        verbose=100,
        early_stopping_rounds=250,
    )
    results, bad_features = permutation_importance(
        model=lgb_model,
        X_val=pd.DataFrame(X_v, columns=mol_features),
        y_val=y_v["scalar_coupling_constant"],
        calc_logmae=calc_logmae,
        threshold=0.01,
        minimize=True,
        verbose=True,
    )
    mol_features = [feat for feat in mol_features if feat not in bad_features]
    return mol_features


def create_or_load_selected_features(
    preds_and_oofs_folder, train, scalar_coupling_contributions
):
    try:
        selected_features = np.load(
            preds_and_oofs_folder + f"/nn_mol_features_{mol_type}.npy",
            allow_pickle=True,
        )
        print(
            f"-------There are {len(selected_features)} features in your model-------"
        )
    except Exception:
        selected_features = select_best_features(
            preds_and_oofs_folder, train, scalar_coupling_contributions
        )
        np.save(
            preds_and_oofs_folder + f"/nn_mol_features_{mol_type}", selected_features
        )
    return list(selected_features)


def get_oofs_and_preds(df_type, mol_type):
    if df_type == "oof":
        df = pd.read_csv(f"{original_data_folder}/train.csv", usecols=["id", "type"])
        file_names = [
            "OOF_FELIPE_LGB_1944.csv",
            "OOF_FELIPE_NN_1917.csv",
            "harsh_oof_1.688.csv",
            "oof_lolstart_lgb_1720.csv",
            "oof_yassine_lgb_5_folds_-1.295.csv",
            "harsh_10fold_oof_1.670.csv",
        ]
    else:
        df = pd.read_csv(f"{original_data_folder}/test.csv", usecols=["id", "type"])
        file_names = [
            "PRED_FELIPE_LGB_1944.csv",
            "PRED_FELIPE_NN_1917.csv",
            "harsh_pred_1.688.csv",
            "pred_lolstart_lgb_1720.csv",
            "pred_yassine_lgb_5_folds-1.295.csv",
            "harsh_10fold_pred_1.670.csv",
        ]
    sc_columns = []
    for i, file_name in enumerate(file_names):
        path = f"{preds_and_oofs_folder}/{file_name}"
        if not os.path.exists(path):
            continue
        data = pd.read_csv(path)
        for col in ["oof", "pred", "prediction", "ind"]:
            if col in data.columns:
                rename_map = {
                    col: (
                        "scalar_coupling_constant"
                        if col in ["oof", "pred", "prediction"]
                        else "id"
                    )
                }
                data.rename(columns=rename_map, inplace=True)
        data.sort_values("id", inplace=True)
        df[f"sc_{i}"] = data["scalar_coupling_constant"].values
        sc_columns.append(f"sc_{i}")
    df = df[df["type"] == mol_type]
    df.drop(columns=["type"], inplace=True)
    df.loc[:, sc_columns] -= df.loc[:, sc_columns].mean().mean()
    df.loc[:, sc_columns] /= df.loc[:, sc_columns].stack().std()
    return df, sc_columns




## === cell 2
original_data_folder = "../input/champs-scalar-coupling"
preds_and_oofs_folder = "../input/preds-on-oof-and-test"
train_and_test_with_feats_folder = "../input/features-for-top-5-lb-with-nn-or-lgb"
scalar_coupling_contributions = pd.read_csv(
    original_data_folder + "/scalar_coupling_contributions.csv"
)

if TF_AVAILABLE:
    config = tf.compat.v1.ConfigProto(device_count={"GPU": 1, "CPU": 4})
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    sess = tf.compat.v1.Session(config=config)

epoch_n = 500
verbose = 1
batch_size = 2048

mol_types = ["1JHN"]
run_number = 0
k_folds = 1

scores_nn = dict()
for mol_type_index, mol_type in enumerate(mol_types):
    print(mol_type, f"- run number {run_number}")

    try:
        scores_nn = np.load(f"run_{run_number}_scores_nn.npy", allow_pickle=True).item()
    except Exception:
        scores_nn = dict()

    train_path = train_and_test_with_feats_folder + f"/train_{mol_type}.csv"
    test_path = train_and_test_with_feats_folder + f"/test_{mol_type}.csv"
    if os.path.exists(train_path):
        train = pd.read_csv(train_path).fillna(0)
    else:
        train = pd.read_csv(original_data_folder + "/train.csv")
        train = train[train["type"] == mol_type].reset_index(drop=True)

    if os.path.exists(test_path):
        test_full = pd.read_csv(test_path).fillna(0)
    else:
        test_full = pd.read_csv(original_data_folder + "/test.csv")
        test_full = test_full[test_full["type"] == mol_type].reset_index(drop=True)

    print(f" Working with data of type {mol_type} and shape {train.shape}")

    df_oofs, sc_columns = get_oofs_and_preds("oof", mol_type)
    df_pred, _ = get_oofs_and_preds("pred", mol_type)

    molecules_names = train["molecule_name"].unique()
    type_scalar_contributions = scalar_coupling_contributions[
        scalar_coupling_contributions["molecule_name"].isin(molecules_names)
    ]
    type_scalar_contributions = type_scalar_contributions[
        type_scalar_contributions["type"] == mol_type
    ]
    for col in ["fc", "sd", "pso", "dso"]:
        train[col] = type_scalar_contributions[col].values
    del type_scalar_contributions, molecules_names

    if not TF_AVAILABLE:
        test_full = pd.merge(test_full, df_pred, on=["id"])
        test_ids = test_full["id"].copy()
        pred_mean = test_full[sc_columns].mean(axis=1)
        submission = pd.DataFrame(
            {"id": test_ids, "scalar_coupling_constant": pred_mean}
        )
        submission.to_csv("final_sub.csv", index=False)
        print("TensorFlow not available – submission created from OOF averages.")
        continue  # move to next mol_type (if any)

    nn_mol_features = create_or_load_selected_features(
        preds_and_oofs_folder, train, scalar_coupling_contributions
    )
    for col in sc_columns:
        nn_mol_features.append(col)

    train, index_folds = preprocess_train_data(
        mol_type,
        train,
        scalar_coupling_contributions,
        k_folds=k_folds,
        val_data_ratio_if_no_kfolds=0.2,
        verbose=True,
    )
    train = pd.merge(train, df_oofs, on=["id"])
    test_full = pd.merge(test_full, df_pred, on=["id"])
    test_ids = test_full["id"].copy()
    del df_oofs, df_pred

    std_scaler = StandardScaler().fit(train[nn_mol_features])
    test_selected = std_scaler.transform(test_full[nn_mol_features].values)
    del test_full

    oof_nn = pd.DataFrame(np.zeros((len(train), 1)))
    pred_nn = pd.DataFrame(
        np.zeros((len(test_selected), k_folds)), columns=list(range(k_folds))
    )
    n_features = 32
    cols_nn_features_for_lgb = np.array(
        ["sc", "fc", "sd", "pso", "dso"]
        + [
            f"nn_feat_{k_fold}_{i}"
            for k_fold in range(k_folds)
            for i in range(1, n_features + 1)
        ]
    ).flatten()
    nn_features_for_lgb_train = pd.DataFrame(
        np.zeros((train.shape[0], n_features + 5)),
        columns=cols_nn_features_for_lgb[: n_features + 5],
    )
    nn_features_for_lgb_test = pd.DataFrame(
        np.zeros((test_selected.shape[0], n_features * k_folds + 5)),
        columns=cols_nn_features_for_lgb,
    )
    if mol_type not in scores_nn:
        scores_nn[mol_type] = dict()
    gc.collect()
    for k_fold, (trn_idx, val_idx) in enumerate(index_folds):
        (
            nn_features_for_lgb_test,
            nn_features_for_lgb_train,
            pred_nn,
            oof_nn,
            scores_nn,
        ) = run_nn(
            load_existing_model=True,
            k_fold=k_fold,
            trn_idx=trn_idx,
            val_idx=val_idx,
            train=train,
            nn_mol_features=nn_mol_features,
            k_folds=k_folds,
            file_folder=preds_and_oofs_folder,
            nn_features_for_lgb_test=nn_features_for_lgb_test,
            nn_features_for_lgb_train=nn_features_for_lgb_train,
            scores_nn=scores_nn,
            test_selected=test_selected,
            mol_type=mol_type,
            oof_nn=oof_nn,
            pred_nn=pred_nn,
            run_number=run_number,
        )
        gc.collect()

    nn_features_for_lgb_test.loc[:, ["sc", "fc", "sd", "pso", "dso"]] /= k_folds
    np.save(f"run_{run_number}_scores_nn", scores_nn)
    np.save(f"run_{run_number}_{mol_type}_oof_nn", oof_nn)
    pred_nn.to_pickle(f"run_{run_number}_{mol_type}_pred_nn")
    nn_features_for_lgb_train.to_pickle(
        f"run_{run_number}_{mol_type}_nn_features_for_lgb_train"
    )
    nn_features_for_lgb_test.to_pickle(
        f"run_{run_number}_{mol_type}_nn_features_for_lgb_test"
    )
    pred_mean = pred_nn.mean(axis=1)
    submission = pd.DataFrame({"id": test_ids, "scalar_coupling_constant": pred_mean})
    submission.to_csv("final_sub.csv", index=False)
    gc.collect()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_56/3055312090.py in create_or_load_selected_features(preds_and_oofs_folder, train, scalar_coupling_contributions)
    223     try:
--> 224         selected_features = np.load(
    225             preds_and_oofs_folder + f"/nn_mol_features_{mol_type}.npy",

/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py in load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size)
    426         else:
--> 427             fid = stack.enter_context(open(os_fspath(file), "rb"))
    428             own_fid = True

FileNotFoundError: [Errno 2] No such file or directory: '../input/preds-on-oof-and-test/nn_mol_features_1JHN.npy'

During handling of the above exception, another exception occurred:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/4245242652.py in <cell line: 0>()
     87     # When TensorFlow is available we follow the original NN pipeline.
     88     # ----------------------------------------------------------------------
---> 89     nn_mol_features = create_or_load_selected_features(
     90         preds_and_oofs_folder, train, scalar_coupling_contributions
     91     )

/tmp/ipykernel_56/3055312090.py in create_or_load_selected_features(preds_and_oofs_folder, train, scalar_coupling_contributions)
    230         )
    231     except Exception:
--> 232         selected_features = select_best_features(
    233             preds_and_oofs_folder, train, scalar_coupling_contributions
    234         )

/tmp/ipykernel_56/3055312090.py in select_best_features(file_folder, train, scalar_coupling_contributions)
    165         ]
    166 
--> 167     train, index_folds = preprocess_train_data(
    168         mol_type,
    169         train,

NameError: name 'preprocess_train_data' is not defined

## === cell 3
try:
    for i in range(32):
        nn_feat = nn_features_for_lgb_train.iloc[val_idx, i + 5]
        target = train.loc[val_idx, "scalar_coupling_constant"]
        plt.scatter(target, nn_feat, s=0.2)
        plt.xlabel("scalar coupling constant")
        plt.ylabel(f"nn_feat_{i}")
        plt.show()
except Exception:
    pass




## === cell 4
if not os.path.exists("final_sub.csv"):
    placeholder = pd.read_csv(f"{original_data_folder}/sample_submission.csv")
    placeholder["scalar_coupling_constant"] = 0.0
    placeholder.to_csv("final_sub.csv", index=False)
