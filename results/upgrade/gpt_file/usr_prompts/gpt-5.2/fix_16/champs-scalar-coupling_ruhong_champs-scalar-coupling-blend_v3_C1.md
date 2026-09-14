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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

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

0.2447767932130148

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import sys
import numpy as np
import pandas as pd



## === cell 1
SEED = 31
TRIALS = 200
TARGET = "scalar_coupling_constant"
PREDICTION = "pred"

seed = SEED




## === cell 2
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(SEED)




## === cell 3
def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for CHAMPS: log(mean(abs error)) per type, then mean over types.
    """
    maes = (y_true - y_pred).abs().groupby(types).mean()
    maes = np.log(maes.map(lambda x: max(x, floor)))
    return maes.mean()




## === cell 4
DATA_DIR_CANDIDATES = [
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling/champs-scalar-coupling",
    "/kaggle/input",
    "/kaggle/data/input",
    "/kaggle/data",
]

REQUIRED_FILES = ["train.csv", "test.csv", "structures.csv", "sample_submission.csv"]


def _has_required_files(d):
    return (
        d
        and os.path.isdir(d)
        and all(os.path.exists(os.path.join(d, f)) for f in REQUIRED_FILES)
    )


def find_data_dir():
    """
    Pick a directory that contains all required files.
    """
    candidates = []
    for d in DATA_DIR_CANDIDATES:
        if not os.path.isdir(d):
            continue
        nested = os.path.join(d, "champs-scalar-coupling")
        if _has_required_files(nested):
            candidates.append(nested)
        if _has_required_files(d):
            candidates.append(d)

    if not candidates:
        for root in ["/kaggle/input", "/kaggle/data", "/kaggle/data/input"]:
            if not os.path.isdir(root):
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                if all(f in filenames for f in REQUIRED_FILES):
                    candidates.append(dirpath)
                if "champs-scalar-coupling" in dirnames:
                    cand = os.path.join(dirpath, "champs-scalar-coupling")
                    if _has_required_files(cand):
                        candidates.append(cand)

    if not candidates:
        return None

    candidates = sorted(set(candidates), key=lambda p: (-len(p), p))
    return candidates[0]


DATA_DIR = find_data_dir()
if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find a directory containing all required files: "
        + ", ".join(REQUIRED_FILES)
    )

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
structures = pd.read_csv(structures_path)

for df in (train, test, structures):
    df["molecule_name"] = df["molecule_name"].astype(str).str.strip()


def coerce_atom_index(df, col, df_name):
    s = pd.to_numeric(df[col], errors="coerce")
    n_na = int(s.isna().sum())
    if n_na:
        bad_cols = [c for c in ["id", "molecule_name", col] if c in df.columns]
        bad = df.loc[s.isna(), bad_cols].head(5)
        raise RuntimeError(
            f"{df_name}.{col} has {n_na} NaN values after coercion; cannot merge reliably. "
            f"Examples:\n{bad}"
        )
    df[col] = s.astype(np.int32)
    return df


train = coerce_atom_index(train, "atom_index_0", "train")
train = coerce_atom_index(train, "atom_index_1", "train")
test = coerce_atom_index(test, "atom_index_0", "test")
test = coerce_atom_index(test, "atom_index_1", "test")

structures["atom_index"] = pd.to_numeric(structures["atom_index"], errors="coerce")
if structures["atom_index"].isna().any():
    raise RuntimeError("structures.atom_index contains NaNs; cannot proceed.")
structures["atom_index"] = structures["atom_index"].astype(np.int32)

for c in ["x", "y", "z"]:
    structures[c] = pd.to_numeric(structures[c], errors="coerce")
if structures[["x", "y", "z"]].isna().any().any():
    raise RuntimeError("structures has NaN coordinates after coercion; cannot proceed.")

print("DATA_DIR =", DATA_DIR)
print("train:", train.shape, "test:", test.shape, "structures:", structures.shape)



## === cell 5
structures_small = structures[
    ["molecule_name", "atom_index", "atom", "x", "y", "z"]
].copy()


def add_pair_features(df):
    df = df.copy()

    tmp0 = df[["molecule_name", "atom_index_0"]].rename(
        columns={"atom_index_0": "atom_index"}
    )
    tmp1 = df[["molecule_name", "atom_index_1"]].rename(
        columns={"atom_index_1": "atom_index"}
    )

    a0 = (
        tmp0.merge(structures_small, on=["molecule_name", "atom_index"], how="left")
        .rename(columns={"atom": "atom_0", "x": "x_0", "y": "y_0", "z": "z_0"})[
            ["atom_0", "x_0", "y_0", "z_0"]
        ]
        .reset_index(drop=True)
    )
    a1 = (
        tmp1.merge(structures_small, on=["molecule_name", "atom_index"], how="left")
        .rename(columns={"atom": "atom_1", "x": "x_1", "y": "y_1", "z": "z_1"})[
            ["atom_1", "x_1", "y_1", "z_1"]
        ]
        .reset_index(drop=True)
    )

    df = df.reset_index(drop=True)
    df = pd.concat([df, a0, a1], axis=1)

    n0 = int(df["x_0"].isna().sum())
    n1 = int(df["x_1"].isna().sum())
    if n0 or n1:
        bad0 = df.loc[
            df["x_0"].isna(), ["molecule_name", "atom_index_0", "atom_index_1"]
        ].head(5)
        bad1 = df.loc[
            df["x_1"].isna(), ["molecule_name", "atom_index_0", "atom_index_1"]
        ].head(5)
        raise RuntimeError(
            f"Join with structures failed: missing coordinates "
            f"(missing x_0: {n0}/{len(df)}, missing x_1: {n1}/{len(df)}). "
            f"DATA_DIR={DATA_DIR}\n"
            f"Examples missing x_0:\n{bad0.to_string(index=False)}\n"
            f"Examples missing x_1:\n{bad1.to_string(index=False)}"
        )

    dx = df["x_0"] - df["x_1"]
    dy = df["y_0"] - df["y_1"]
    dz = df["z_0"] - df["z_1"]
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)
    df["dist2"] = df["dist"] ** 2
    df["inv_dist"] = 1.0 / (df["dist"] + 1e-6)
    df["inv_dist2"] = 1.0 / (df["dist2"] + 1e-6)
    return df


train_feat = add_pair_features(train)
test_feat = add_pair_features(test)

for col in ["type", "atom_0", "atom_1"]:
    cats = pd.Index(
        pd.concat([train_feat[col], test_feat[col]], axis=0).astype(str).unique()
    )
    mapping = {k: i for i, k in enumerate(cats)}
    train_feat[col + "_code"] = (
        train_feat[col].astype(str).map(mapping).astype(np.int32)
    )
    test_feat[col + "_code"] = test_feat[col].astype(str).map(mapping).astype(np.int32)

train_feat["atom_pair_code"] = (
    train_feat["atom_0_code"] * 32 + train_feat["atom_1_code"]
).astype(np.int32)
test_feat["atom_pair_code"] = (
    test_feat["atom_0_code"] * 32 + test_feat["atom_1_code"]
).astype(np.int32)

for c in ["x_0", "x_1", "dist", "atom_0", "atom_1", "type"]:
    if train_feat[c].isna().any() or test_feat[c].isna().any():
        tr_n = int(train_feat[c].isna().sum())
        te_n = int(test_feat[c].isna().sum())
        raise RuntimeError(
            f"Found NaNs in feature '{c}' (train missing={tr_n}, test missing={te_n}). "
            f"Join with structures failed. DATA_DIR={DATA_DIR}"
        )

print("train_feat:", train_feat.shape, "test_feat:", test_feat.shape)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1111387070.py in <cell line: 0>()
     61 
     62 train_feat = add_pair_features(train)
---> 63 test_feat = add_pair_features(test)
     64 
     65 for col in ["type", "atom_0", "atom_1"]:

/tmp/ipykernel_11/1111387070.py in add_pair_features(df)
     42             df["x_1"].isna(), ["molecule_name", "atom_index_0", "atom_index_1"]
     43         ].head(5)
---> 44         raise RuntimeError(
     45             f"Join with structures failed: missing coordinates "
     46             f"(missing x_0: {n0}/{len(df)}, missing x_1: {n1}/{len(df)}). "

RuntimeError: Join with structures failed: missing coordinates (missing x_0: 467813/467813, missing x_1: 467813/467813). DATA_DIR=/kaggle/input/champs-scalar-coupling
Examples missing x_0:
   molecule_name  atom_index_0  atom_index_1
dsgdb9nsd_071451             9             0
dsgdb9nsd_071451             9             1
dsgdb9nsd_071451             9             4
dsgdb9nsd_071451             9             5
dsgdb9nsd_071451             9            10
Examples missing x_1:
   molecule_name  atom_index_0  atom_index_1
dsgdb9nsd_071451             9             0
dsgdb9nsd_071451             9             1
dsgdb9nsd_071451             9             4
dsgdb9nsd_071451             9             5
dsgdb9nsd_071451             9            10

## === cell 6
def ridge_fit_predict_per_type(
    train_df,
    test_df,
    features,
    target_col,
    type_col="type",
    alpha=1e-2,
    max_train_rows_per_type=60000,
    seed=SEED,
):
    rng = np.random.RandomState(seed)
    train_pred = np.zeros(len(train_df), dtype=np.float64)
    test_pred = np.zeros(len(test_df), dtype=np.float64)

    for t, tr_idx in train_df.groupby(type_col).indices.items():
        te_idx = test_df.index[test_df[type_col] == t].to_numpy()

        tr_idx = np.asarray(tr_idx)
        if len(tr_idx) > max_train_rows_per_type:
            tr_idx = rng.choice(tr_idx, size=max_train_rows_per_type, replace=False)

        Xtr = train_df.loc[tr_idx, features].to_numpy(dtype=np.float64)
        ytr = train_df.loc[tr_idx, target_col].to_numpy(dtype=np.float64)

        Xtrb = np.hstack([np.ones((Xtr.shape[0], 1), dtype=np.float64), Xtr])

        XtX = Xtrb.T @ Xtrb
        reg = np.eye(XtX.shape[0], dtype=np.float64) * alpha
        reg[0, 0] = 0.0
        Xty = Xtrb.T @ ytr
        w = np.linalg.solve(XtX + reg, Xty)

        mask_tr = train_df[type_col].to_numpy() == t
        Xtr_full = train_df.loc[mask_tr, features].to_numpy(dtype=np.float64)
        Xtr_fullb = np.hstack(
            [np.ones((Xtr_full.shape[0], 1), dtype=np.float64), Xtr_full]
        )
        train_pred[mask_tr] = Xtr_fullb @ w

        if len(te_idx) > 0:
            Xte = test_df.loc[te_idx, features].to_numpy(dtype=np.float64)
            Xteb = np.hstack([np.ones((Xte.shape[0], 1), dtype=np.float64), Xte])
            test_pred[te_idx] = Xteb @ w

    return train_pred, test_pred


feature_sets = [
    ["dist", "dist2", "inv_dist", "inv_dist2"],
    ["dist", "inv_dist", "type_code"],
    ["dist", "dist2", "atom_0_code", "atom_1_code"],
    ["dist", "inv_dist2", "atom_pair_code", "type_code"],
    ["dist", "dist2", "inv_dist", "atom_0_code", "atom_1_code", "type_code"],
]

alphas = [1e-2, 1e-1, 1e-3, 5e-2, 2e-2]
max_rows = 50000

train_sets = []
test_sets = []

for i, (feats, a) in enumerate(zip(feature_sets, alphas), start=1):
    trp, tep = ridge_fit_predict_per_type(
        train_feat,
        test_feat,
        features=feats,
        target_col=TARGET,
        type_col="type",
        alpha=a,
        max_train_rows_per_type=max_rows,
        seed=SEED + i,
    )
    tr_df = train_feat[["id", "type", TARGET]].copy()
    tr_df[PREDICTION] = trp
    te_df = test_feat[["id"]].copy()
    te_df[TARGET] = tep
    train_sets.append(tr_df)
    test_sets.append(te_df)
    score_i = group_mean_log_mae(tr_df[TARGET], tr_df[PREDICTION], tr_df["type"])
    print(f"base model {i}: features={feats}, alpha={a} -> train metric={score_i:.6f}")

if len(train_sets) < 1 or len(test_sets) < 1:
    raise RuntimeError("No base models were trained; cannot proceed to ensembling.")

print("Prepared train_sets:", len(train_sets), "test_sets:", len(test_sets))




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/821542020.py in <cell line: 0>()
     63     trp, tep = ridge_fit_predict_per_type(
     64         train_feat,
---> 65         test_feat,
     66         features=feats,
     67         target_col=TARGET,

NameError: name 'test_feat' is not defined

## === cell 7
def weights(n, min_weight=0.01, max_allocation=0.5):
    if n < 1:
        raise ValueError("n must not be less than 1")
    if n == 1:
        return [1.0]
    remainder = 1 - (n * min_weight)
    if remainder <= 0:
        return [1.0 / n] * n
    res = []
    for _ in range(n - 1):
        a = random.uniform(0.01, max_allocation) * remainder
        res.append(a + min_weight)
        remainder -= a
    res.append(remainder + min_weight)
    s = float(sum(res))
    return [w / s for w in res]


def trial(train_sets, prediction_column, target_column):
    ws = weights(len(train_sets), min_weight=0.05, max_allocation=0.9)
    df = train_sets[0].copy()
    df[prediction_column] = 0.0
    for i, t in enumerate(train_sets):
        df[prediction_column] += t[prediction_column].to_numpy(dtype=np.float64) * ws[i]
    score = group_mean_log_mae(df[target_column], df[prediction_column], df["type"])
    return score, ws


best = sys.maxsize
best_weights = None

if len(train_sets) < 1:
    raise RuntimeError("train_sets is empty; cannot ensemble.")

for _ in range(TRIALS):
    score, ws = trial(
        train_sets=train_sets, prediction_column=PREDICTION, target_column=TARGET
    )
    if score < best:
        best = score
        best_weights = ws

if best_weights is None:
    best_weights = [1.0 / len(train_sets)] * len(train_sets)
    best = trial(
        train_sets=train_sets, prediction_column=PREDICTION, target_column=TARGET
    )[0]

print(f"best={best:.6f}")
print(f"best weights (sum={sum(best_weights):.6f}) -> {best_weights}")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/493080811.py in <cell line: 0>()
     31 
     32 if len(train_sets) < 1:
---> 33     raise RuntimeError("train_sets is empty; cannot ensemble.")
     34 
     35 for _ in range(TRIALS):

RuntimeError: train_sets is empty; cannot ensemble.

## === cell 8
if len(test_sets) < 1:
    raise RuntimeError("test_sets is empty; cannot write submission.")

submission = test_sets[0].copy()
submission[TARGET] = 0.0
for i, t in enumerate(test_sets):
    submission[TARGET] += t[TARGET].to_numpy(dtype=np.float64) * best_weights[i]

submission = submission.sort_values("id").reset_index(drop=True)
submission = submission[["id", TARGET]]

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())
print("submission shape:", submission.shape)
print("Files in cwd:", os.listdir("."))

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/2269015524.py in <cell line: 0>()
      1 if len(test_sets) < 1:
----> 2     raise RuntimeError("test_sets is empty; cannot write submission.")
      3 
      4 submission = test_sets[0].copy()
      5 submission[TARGET] = 0.0

RuntimeError: test_sets is empty; cannot write submission.
