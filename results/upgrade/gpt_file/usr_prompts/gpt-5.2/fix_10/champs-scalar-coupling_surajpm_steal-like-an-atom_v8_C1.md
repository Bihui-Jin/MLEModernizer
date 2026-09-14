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

-1.3547912745601447

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'The current notebook fails because it tries to read four external submission files from `../input/...` folders that don’t exist in your environment, so `sub1..sub4` are never created and everything downstream crashes. To make it run end-to-end and generate a valid `submission.csv`, I replace that broken ensemble-with-missing-files with a minimal, self-contained baseline that uses only the provided competition data. Specifically, it compute the mean `scalar_coupling_constant` per `type` from `train.csv` and use that as predictions for `test.csv` (a standard “type-mean” baseline), then write `id,scalar_coupling_constant` to `submission.csv`. This preserves the overall “simple blending/aggregation” spirit while ensuring correct I/O, correct submission format, and a nontrivial score (better than all-zeros) without introducing new modeling complexity.'
- What this solution (achieved 1.21344) has done: 'You’re hitting NaNs in the design matrix because some atom-coordinate merges fail (missing structure rows or unexpected indices), which propagates into `dist/dist2` and makes `Ridge` crash; fixing that unblocks everything downstream. I add a minimal, score-neutral imputation: fill missing `dist/dist2` with the per-`type` median distance computed from train (fallback to global median), and also guard against any remaining NaNs in dummy columns. I also make `cell 3`/`cell 4` robust by ensuring `sub` is always defined once predictions are created and by importing matplotlib for plotting. Core approach (per-type Ridge on distance + atom one-hot) is preserved.'
- What this solution (achieved 1.28381) has done: 'Your current Ridge-per-type baseline is far from the target (lower is better), so we should improve it modestly without changing the overall approach. The biggest safe gain here is to add a few physically meaningful, cheap-to-compute geometry features (absolute deltas and inverse-distance terms) while keeping the same per-type Ridge training loop and loss semantics. I also standardize features within each coupling type (using train statistics only) to make the single global `alpha` behave more consistently across types, which typically reduces MAE without changing the model family. All changes keep the same I/O paths and still produce a valid `submission.csv`.'
- What this solution (achieved 4.18986) has done: 'The timeout is dominated by building huge one-hot DataFrames twice (train/test), repeated `.loc/.copy()` slicing per type, and fitting Ridge with `sag` on dense pandas frames. I keep the exact same features, per-type standardization, and per-type alpha selection logic, but switch the design matrix to a single NumPy dense array built via integer encoding (equivalent to one-hot), and avoid repeated DataFrame operations inside the type loop. I also replace expensive pandas groupby/index handling with integer type codes and precomputed index arrays, and use `Ridge(solver="lsqr")` which is deterministic and much faster for dense problems while preserving the same objective/solution. All merges are kept the same, but made lighter via `sort=False` and early conversion to compact dtypes.'
- What this solution (achieved 4.19294) has done: 'Your current score (4.18986, lower is better) is far worse than the target (-1.3548), so we should improve accuracy while keeping the same per-type Ridge approach and the same core geometry+atom one-hot feature set. The biggest issue is that `build_design_numpy()` uses a global atom mapping, but then each per-type model is trained on only a subset; columns corresponding to atoms not present in that type become all-zero and can destabilize/slow the solve and harm generalization. I keep the exact same features and per-type Ridge loop, but for each type I (1) drop zero-variance columns based on that type’s training subset and (2) standardize *all* remaining columns (numeric + one-hot) using train statistics for that type (one-hot columns stay centered properly), which typically reduces MAE for ridge without changing the model family. I also remove the CV feature truncation (it currently can select an alpha based on a mismatched feature set), and instead choose alpha on the same per-type filtered column set to better match the final fit while remaining very lightweight.'
- What this solution (achieved 5.09504) has done: 'Your current score is much worse than the target (lower is better), so the smallest safe way to move toward the target is to improve feature quality while keeping the same per-type Ridge setup. The main accuracy gap here is that the model only sees pairwise geometry and atom identities; adding a few standard, cheap structural context features (per-atom neighbor counts within each molecule) typically gives a large lift for this competition without changing the model family or training loop. I compute per-(molecule, atom_index) neighbor counts at a couple of radii using only `structures.csv`, merge those counts onto `atom_index_0/1`, and include the resulting count features (and simple differences/products) in the same Ridge design matrix. Everything else (per-type Ridge, alpha selection, standardization, submission writing) stays the same.'

# 9. Code solution

## === cell 0
import os

BASE_INPUT = "/kaggle/data/champs-scalar-coupling"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/input/champs-scalar-coupling"

assert os.path.exists(BASE_INPUT), f"Could not find dataset directory at {BASE_INPUT}"

TRAIN_PATH = os.path.join(BASE_INPUT, "train.csv")
TEST_PATH = os.path.join(BASE_INPUT, "test.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")
STRUCTURES_PATH = os.path.join(BASE_INPUT, "structures.csv")



## === cell 1
import numpy as np  # linear algebra
import pandas as pd  # data processing
import matplotlib.pyplot as plt

print("BASE_INPUT =", BASE_INPUT)
print("Files in BASE_INPUT (first 20):", sorted(os.listdir(BASE_INPUT))[:20])



## === cell 2
from sklearn.linear_model import Ridge

train = pd.read_csv(
    TRAIN_PATH,
    usecols=[
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
        "scalar_coupling_constant": np.float32,
    },
)
test = pd.read_csv(
    TEST_PATH,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
sample_sub = pd.read_csv(
    SAMPLE_SUB_PATH, usecols=["id", "scalar_coupling_constant"], dtype={"id": np.int32}
)

structures = pd.read_csv(
    STRUCTURES_PATH,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
)

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


def add_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(s0, on=["molecule_name", "atom_index_0"], how="left", sort=False)
    df = df.merge(s1, on=["molecule_name", "atom_index_1"], how="left", sort=False)

    dx = df["x_0"].to_numpy(dtype=np.float32, copy=False) - df["x_1"].to_numpy(
        dtype=np.float32, copy=False
    )
    dy = df["y_0"].to_numpy(dtype=np.float32, copy=False) - df["y_1"].to_numpy(
        dtype=np.float32, copy=False
    )
    dz = df["z_0"].to_numpy(dtype=np.float32, copy=False) - df["z_1"].to_numpy(
        dtype=np.float32, copy=False
    )

    df["dx"] = dx
    df["dy"] = dy
    df["dz"] = dz
    df["adx"] = np.abs(dx)
    df["ady"] = np.abs(dy)
    df["adz"] = np.abs(dz)

    dist = np.sqrt(dx * dx + dy * dy + dz * dz, dtype=np.float32).astype(
        np.float32, copy=False
    )
    df["dist"] = dist
    df["dist2"] = (dist * dist).astype(np.float32, copy=False)

    eps = np.float32(1e-3)
    inv = (np.float32(1.0) / np.maximum(dist, eps)).astype(np.float32, copy=False)
    df["inv_dist"] = inv
    df["inv_dist2"] = (inv * inv).astype(np.float32, copy=False)

    df["log_dist"] = np.log1p(np.maximum(dist, np.float32(0.0))).astype(
        np.float32, copy=False
    )
    df["log_dist2"] = np.log1p(
        np.maximum(df["dist2"].to_numpy(dtype=np.float32, copy=False), np.float32(0.0))
    ).astype(np.float32, copy=False)
    df["log_inv_dist"] = np.log1p(np.maximum(inv, np.float32(0.0))).astype(
        np.float32, copy=False
    )
    df["log_inv_dist2"] = np.log1p(
        np.maximum(
            df["inv_dist2"].to_numpy(dtype=np.float32, copy=False), np.float32(0.0)
        )
    ).astype(np.float32, copy=False)

    df.drop(columns=["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"], inplace=True)
    return df


train_f = add_pair_features(train)
test_f = add_pair_features(test)

global_dist_median = float(np.nanmedian(train_f["dist"].to_numpy()))
type_dist_median = train_f.groupby("type", observed=True)["dist"].median()


def impute_dist_features(df: pd.DataFrame) -> pd.DataFrame:
    dist = df["dist"]
    if dist.isna().any():
        fill_vals = df["type"].map(type_dist_median).astype(np.float32)
        fill_vals = fill_vals.fillna(global_dist_median).astype(np.float32)
        mask = dist.isna().to_numpy()
        df.loc[mask, "dist"] = fill_vals.to_numpy()[mask]
        mask2 = df["dist2"].isna().to_numpy()
        if mask2.any():
            d = df.loc[mask2, "dist"].astype(np.float32).to_numpy()
            df.loc[mask2, "dist2"] = d * d

    if df["dist2"].isna().any():
        mask2 = df["dist2"].isna().to_numpy()
        d = df.loc[mask2, "dist"].astype(np.float32).to_numpy()
        df.loc[mask2, "dist2"] = d * d

    eps = np.float32(1e-3)
    if df["inv_dist"].isna().any() or df["inv_dist2"].isna().any():
        inv = (
            np.float32(1.0) / np.maximum(df["dist"].astype(np.float32).to_numpy(), eps)
        ).astype(np.float32)
        df["inv_dist"] = inv
        df["inv_dist2"] = (inv * inv).astype(np.float32)

    for c in ["dx", "dy", "dz", "adx", "ady", "adz"]:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(np.float32(0.0))

    for c in ["log_dist", "log_dist2", "log_inv_dist", "log_inv_dist2"]:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(np.float32(0.0))

    return df


train_f = impute_dist_features(train_f)
test_f = impute_dist_features(test_f)

type_mean = train_f.groupby("type", observed=True)["scalar_coupling_constant"].mean()
global_mean = float(train_f["scalar_coupling_constant"].mean())
train_f["type_mean"] = train_f["type"].map(type_mean).astype(np.float32)
test_f["type_mean"] = (
    test_f["type"].map(type_mean).fillna(global_mean).astype(np.float32)
)

numeric_cols = [
    "dist",
    "dist2",
    "inv_dist",
    "inv_dist2",
    "log_dist",
    "log_dist2",
    "log_inv_dist",
    "log_inv_dist2",
    "dx",
    "dy",
    "dz",
    "adx",
    "ady",
    "adz",
]


def build_design_numpy(train_df: pd.DataFrame, test_df: pd.DataFrame):
    a0_all = (
        pd.concat([train_df["atom_0"], test_df["atom_0"]], ignore_index=True)
        .astype("object")
        .fillna("UNK")
    )
    a1_all = (
        pd.concat([train_df["atom_1"], test_df["atom_1"]], ignore_index=True)
        .astype("object")
        .fillna("UNK")
    )

    atom0_levels = pd.Index(a0_all.unique())
    atom1_levels = pd.Index(a1_all.unique())

    atom0_map = {k: i for i, k in enumerate(atom0_levels)}
    atom1_map = {k: i for i, k in enumerate(atom1_levels)}

    n0 = len(atom0_levels)
    n1 = len(atom1_levels)
    n_num = len(numeric_cols)

    def _make(df: pd.DataFrame):
        n = len(df)
        X = np.zeros((n, n_num + n0 + n1), dtype=np.float32)

        X[:, :n_num] = df[numeric_cols].to_numpy(dtype=np.float32, copy=False)

        a0 = (
            df["atom_0"]
            .astype("object")
            .fillna("UNK")
            .map(atom0_map)
            .to_numpy(dtype=np.int32, copy=False)
        )
        a1 = (
            df["atom_1"]
            .astype("object")
            .fillna("UNK")
            .map(atom1_map)
            .to_numpy(dtype=np.int32, copy=False)
        )

        rows = np.arange(n, dtype=np.int32)
        X[rows, n_num + a0] = 1.0
        X[rows, n_num + n0 + a1] = 1.0
        return X

    return _make(train_df), _make(test_df)


X_train_all, X_test_all = build_design_numpy(train_f, test_f)

pred_test = np.empty(len(test_f), dtype=np.float64)

rng = np.random.RandomState(0)
alpha_grid = np.array([0.1, 1.0, 10.0], dtype=np.float64)
max_cv_n = 120000  # cap per type


def choose_alpha_for_type(Xtr: np.ndarray, ytr: np.ndarray) -> float:
    n = Xtr.shape[0]
    if n < 2000:
        return 1.0

    m = min(n, max_cv_n)
    idx = rng.choice(n, size=m, replace=False)

    split = int(m * 0.9)
    tr_i = idx[:split]
    va_i = idx[split:]

    Xtr_cv = Xtr[tr_i]
    ytr_cv = ytr[tr_i]
    Xva_cv = Xtr[va_i]
    yva_cv = ytr[va_i]

    best_a = 1.0
    best_mae = np.inf

    for a in alpha_grid:
        model = Ridge(alpha=float(a), random_state=0, solver="lsqr")
        model.fit(Xtr_cv, ytr_cv)
        pred = model.predict(Xva_cv)
        mae = float(np.mean(np.abs(pred - yva_cv)))
        if mae < best_mae:
            best_mae = mae
            best_a = float(a)
    return best_a


train_type_codes, type_levels = pd.factorize(train_f["type"], sort=False)
test_type_codes = pd.Categorical(test_f["type"], categories=type_levels).codes

n_types = len(type_levels)
train_idx_by_type = [None] * n_types
test_idx_by_type = [None] * n_types
for tc in range(n_types):
    train_idx_by_type[tc] = np.flatnonzero(train_type_codes == tc)
    test_idx_by_type[tc] = np.flatnonzero(test_type_codes == tc)

y_all = train_f["scalar_coupling_constant"].to_numpy(dtype=np.float64, copy=False)
tm_all = train_f["type_mean"].to_numpy(dtype=np.float64, copy=False)

y_resid_all = (y_all - tm_all).astype(np.float64, copy=False)

for tc, t in enumerate(type_levels):
    te_idx = test_idx_by_type[tc]
    if te_idx.size == 0:
        continue
    tr_idx = train_idx_by_type[tc]

    if tr_idx.size < 50:
        pred_test[te_idx] = float(type_mean.get(t, global_mean))
        continue

    Xtr = X_train_all[tr_idx]
    Xte = X_test_all[te_idx]
    y = y_resid_all[tr_idx]

    col_std = Xtr.std(axis=0)
    keep = col_std > 0.0
    Xtr_k = Xtr[:, keep]
    Xte_k = Xte[:, keep]

    mu = Xtr_k.mean(axis=0)
    sig = Xtr_k.std(axis=0)
    sig = np.where(sig == 0.0, 1.0, sig).astype(np.float32, copy=False)

    Xtr_s = (Xtr_k - mu) / sig
    Xte_s = (Xte_k - mu) / sig

    a_t = choose_alpha_for_type(Xtr_s, y)

    model = Ridge(alpha=a_t, random_state=0, solver="lsqr")
    model.fit(Xtr_s, y)
    resid_pred = model.predict(Xte_s)

    pred_test[te_idx] = resid_pred + test_f.loc[te_idx, "type_mean"].to_numpy(
        dtype=np.float64, copy=False
    )

sub = pd.DataFrame(
    {"id": test_f["id"].to_numpy(copy=False), "scalar_coupling_constant": pred_test}
)

if sample_sub.shape[0] == sub.shape[0] and sample_sub["id"].is_unique:
    sub = sample_sub[["id"]].merge(sub, on="id", how="left", sort=False)
    sub["scalar_coupling_constant"] = sub["scalar_coupling_constant"].fillna(
        global_mean
    )

print(sub.head())
print(sub["scalar_coupling_constant"].describe())
print("Any NaN predictions?", sub["scalar_coupling_constant"].isna().any())



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/772795425.py in <cell line: 0>()
    174 train_f["type_mean"] = train_f["type"].map(type_mean).astype(np.float32)
    175 test_f["type_mean"] = (
--> 176     test_f["type"].map(type_mean).fillna(global_mean).astype(np.float32)
    177 )
    178 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in fillna(self, value, method, axis, inplace, limit, downcast)
   7347                     )
   7348 
-> 7349                 new_data = self._mgr.fillna(
   7350                     value=value, limit=limit, inplace=inplace, downcast=downcast
   7351                 )

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/base.py in fillna(self, value, limit, inplace, downcast)
    184             limit = libalgos.validate_limit(None, limit=limit)
    185 
--> 186         return self.apply_with_block(
    187             "fillna",
    188             value=value,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in fillna(self, value, limit, inplace, downcast, using_cow, already_warned)
   2332                 # 3rd party EA that has not implemented copy keyword yet
   2333                 refs = None
-> 2334                 new_values = self.values.fillna(value=value, method=None, limit=limit)
   2335                 # issue the warning *after* retrying, in case the TypeError
   2336                 #  was caused by an invalid fill_value

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/_mixins.py in fillna(self, value, method, limit, copy)
    374             # We validate the fill_value even if there is nothing to fill
    375             if value is not None:
--> 376                 self._validate_setitem_value(value)
    377 
    378             if not copy:

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/categorical.py in _validate_setitem_value(self, value)
   1587             return self._validate_listlike(value)
   1588         else:
-> 1589             return self._validate_scalar(value)
   1590 
   1591     def _validate_scalar(self, fill_value):

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/categorical.py in _validate_scalar(self, fill_value)
   1612             fill_value = self._unbox_scalar(fill_value)
   1613         else:
-> 1614             raise TypeError(
   1615                 "Cannot setitem on a Categorical with a new "
   1616                 f"category ({fill_value}), set the categories first"

TypeError: Cannot setitem on a Categorical with a new category (15.916068077087402), set the categories first

## === cell 3
SUB_PATH = "submission.csv"
sub.to_csv(SUB_PATH, index=False)
print("Wrote:", SUB_PATH, "rows:", len(sub), "cols:", list(sub.columns))



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1912214558.py in <cell line: 0>()
      1 SUB_PATH = "submission.csv"
----> 2 sub.to_csv(SUB_PATH, index=False)
      3 print("Wrote:", SUB_PATH, "rows:", len(sub), "cols:", list(sub.columns))
      4 

NameError: name 'sub' is not defined

## === cell 4
ax = sub["scalar_coupling_constant"].plot(
    kind="hist", bins=100, title="Predicted scalar_coupling_constant distribution"
)
ax.set_xlabel("scalar_coupling_constant")
plt.show()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2999400004.py in <cell line: 0>()
----> 1 ax = sub["scalar_coupling_constant"].plot(
      2     kind="hist", bins=100, title="Predicted scalar_coupling_constant distribution"
      3 )
      4 ax.set_xlabel("scalar_coupling_constant")
      5 plt.show()

NameError: name 'sub' is not defined
