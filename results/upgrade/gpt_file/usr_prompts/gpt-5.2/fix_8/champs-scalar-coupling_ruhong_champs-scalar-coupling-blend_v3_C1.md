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

None

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


def _load_basic(d):
    tr = pd.read_csv(
        os.path.join(d, "train.csv"),
        usecols=[
            "molecule_name",
            "atom_index_0",
            "atom_index_1",
            "type",
            "scalar_coupling_constant",
            "id",
        ],
    )
    te = pd.read_csv(
        os.path.join(d, "test.csv"),
        usecols=["molecule_name", "atom_index_0", "atom_index_1", "type", "id"],
    )
    st = pd.read_csv(
        os.path.join(d, "structures.csv"),
        usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    )
    return tr, te, st


def _clean_keys(train_df, test_df, structures_df):
    for df in (train_df, test_df, structures_df):
        df["molecule_name"] = df["molecule_name"].astype(str).str.strip()
    return train_df, test_df, structures_df


def _coverage_ok(train_df, test_df, structures_df):
    mol_struct = set(structures_df["molecule_name"].unique())
    mol_train = set(train_df["molecule_name"].unique())
    mol_test = set(test_df["molecule_name"].unique())
    missing_train = len(mol_train - mol_struct)
    missing_test = len(mol_test - mol_struct)
    return missing_train, missing_test


def find_data_dir_with_coverage():
    valid = []
    for d in DATA_DIR_CANDIDATES:
        if not _has_required_files(d):
            continue
        try:
            tr, te, st = _load_basic(d)
            tr, te, st = _clean_keys(tr, te, st)
            missing_train, missing_test = _coverage_ok(tr, te, st)
            valid.append((missing_test + missing_train, missing_train, missing_test, d))
        except Exception:
            continue

    if not valid:
        for root in ["/kaggle/input", "/kaggle/data", "/kaggle/data/input"]:
            if not os.path.isdir(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                fn = set(filenames)
                if not all(f in fn for f in REQUIRED_FILES):
                    continue
                try:
                    tr, te, st = _load_basic(dirpath)
                    tr, te, st = _clean_keys(tr, te, st)
                    missing_train, missing_test = _coverage_ok(tr, te, st)
                    valid.append(
                        (
                            missing_test + missing_train,
                            missing_train,
                            missing_test,
                            dirpath,
                        )
                    )
                except Exception:
                    continue

    if not valid:
        return None, None

    valid.sort(key=lambda x: (x[0], x[2], x[1]))
    best = valid[0]
    return best[3], best


DATA_DIR, cov_info = find_data_dir_with_coverage()
if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find a directory containing all required files with valid coverage: "
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

mol_struct = set(structures["molecule_name"].unique())
mol_train = set(train["molecule_name"].unique())
mol_test = set(test["molecule_name"].unique())
missing_train = len(mol_train - mol_struct)
missing_test = len(mol_test - mol_struct)

print("DATA_DIR =", DATA_DIR)
if cov_info is not None:
    _, mi_tr, mi_te, _ = cov_info
    print(f"coverage check (selected dir): missing_train={mi_tr}, missing_test={mi_te}")
print("train:", train.shape, "test:", test.shape, "structures:", structures.shape)
print(
    "dtypes:",
    {
        "train.molecule_name": train["molecule_name"].dtype,
        "test.atom_index_0": test["atom_index_0"].dtype,
        "structures.molecule_name": structures["molecule_name"].dtype,
        "structures.atom_index": structures["atom_index"].dtype,
    },
)

if missing_train or missing_test:
    ex_test = list((mol_test - mol_struct))[:5]
    ex_train = list((mol_train - mol_struct))[:5]
    raise RuntimeError(
        f"structures.csv missing molecules even in selected DATA_DIR={DATA_DIR}: "
        f"train_missing={missing_train}, test_missing={missing_test}. "
        f"Examples missing_test={ex_test}, missing_train={ex_train}"
    )



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3400318375.py in <cell line: 0>()
    180     ex_test = list((mol_test - mol_struct))[:5]
    181     ex_train = list((mol_train - mol_struct))[:5]
--> 182     raise RuntimeError(
    183         f"structures.csv missing molecules even in selected DATA_DIR={DATA_DIR}: "
    184         f"train_missing={missing_train}, test_missing={missing_test}. "

RuntimeError: structures.csv missing molecules even in selected DATA_DIR=/kaggle/input/champs-scalar-coupling: train_missing=0, test_missing=8502. Examples missing_test=['dsgdb9nsd_087078', 'dsgdb9nsd_019233', 'dsgdb9nsd_125235', 'dsgdb9nsd_086723', 'dsgdb9nsd_081688'], missing_train=[]

## === cell 5
s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x_0",
        "y": "y_0",
        "z": "z_0",
    }
)
s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x_1",
        "y": "y_1",
        "z": "z_1",
    }
)


def add_pair_features(df):
    df = df.merge(
        s0[["molecule_name", "atom_index_0", "atom_0", "x_0", "y_0", "z_0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
        sort=False,
        copy=False,
    )
    df = df.merge(
        s1[["molecule_name", "atom_index_1", "atom_1", "x_1", "y_1", "z_1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
        sort=False,
        copy=False,
    )

    n0 = int(df["x_0"].isna().sum())
    n1 = int(df["x_1"].isna().sum())
    if n0 or n1:
        print(
            f"[merge diagnostic] missing x_0: {n0}/{len(df)} missing x_1: {n1}/{len(df)}"
        )

    dx = df["x_0"] - df["x_1"]
    dy = df["y_0"] - df["y_1"]
    dz = df["z_0"] - df["z_1"]
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)
    df["dist2"] = df["dist"] ** 2
    df["inv_dist"] = 1.0 / (df["dist"] + 1e-6)
    df["inv_dist2"] = 1.0 / (df["dist2"] + 1e-6)
    return df


train_feat = add_pair_features(train.copy())
test_feat = add_pair_features(test.copy())

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
            f"Merge with structures failed. DATA_DIR={DATA_DIR}"
        )

print("train_feat:", train_feat.shape, "test_feat:", test_feat.shape)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3382409440.py in <cell line: 0>()
     77         tr_n = int(train_feat[c].isna().sum())
     78         te_n = int(test_feat[c].isna().sum())
---> 79         raise RuntimeError(
     80             f"Found NaNs in feature '{c}' (train missing={tr_n}, test missing={te_n}). "
     81             f"Merge with structures failed. DATA_DIR={DATA_DIR}"

RuntimeError: Found NaNs in feature 'x_0' (train missing=0, test missing=467813). Merge with structures failed. DATA_DIR=/kaggle/input/champs-scalar-coupling

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

        Xtr_full = train_df.loc[train_df[type_col] == t, features].to_numpy(
            dtype=np.float64
        )
        Xtr_fullb = np.hstack(
            [np.ones((Xtr_full.shape[0], 1), dtype=np.float64), Xtr_full]
        )
        train_pred[train_df[type_col].to_numpy() == t] = Xtr_fullb @ w

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

alphas = [1e-2, 1e-1, 1e-3, 5e-2, 2e-2]  # small differences to diversify
max_rows = 50000  # keep runtime controlled

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




## === cell 7
def weights(n, min_weight=0.01, max_allocation=0.5):
    if n < 1:
        raise ValueError("n must not be less than 1")
    remainder = 1 - (n * min_weight)
    if remainder <= 0:
        raise ValueError("min weight exceeds budget of 1")
    res = []
    for _ in range(n - 1):
        a = random.uniform(0.01, max_allocation) * remainder
        res.append(a + min_weight)
        remainder -= a
    res.append(remainder + min_weight)
    return res


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



## === cell 8
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
