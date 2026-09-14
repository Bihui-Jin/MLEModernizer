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

-1.348743156545858

# 6. Current score

1.18497

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.18497) has done: 'I remove the dependency on missing `../input/blender*` submission files (the current root cause of the crash) and instead generate a self-contained baseline model that trains on the provided `train.csv` and predicts `test.csv`. To keep the core approach simple and stable within the installed packages, I use a per-`type` median target (robust, fast, and metric-aligned) and write predictions in the exact `id,scalar_coupling_constant` format. I also add strict checks for column presence and ID alignment to guarantee a valid `.csv` submission is always produced. The plotting cell be kept but made optional so it won’t break headless execution.'
- What this solution (achieved 1.18497) has done: 'Your current baseline is a per-`type` median, which is robust but too coarse and leaves a large gap to the target (lower is better). To move the score toward the target with minimal core-logic change, I keep the same “predict by group statistics” approach but replace the median with a per-`type` linear correction using only one very strong physical feature: the inter-atomic distance computed from `structures.csv`. This stays fast (single merge + vectorized distance) and metric-aligned (still MAE-based), while remaining self-contained and avoiding any new modeling libraries. I also add safe fallbacks for missing structure rows and unseen `type`s, and still write a valid `submission.csv` with the required columns and ID alignment.'
- What this solution (achieved 1.18497) has done: 'Your current model is still a simple per-type linear fit on distance, but the score gap to the target is large (lower is better), so we need a small, metric-aligned improvement without changing the overall approach. I keep the same core “per-type regression using distance” logic, but make it more robust by fitting a ridge-regularized linear model per type and by adding one extra physically strong feature (inverse distance) while still staying within the same group-statistics/regression family. I also reduce compute by fitting via closed-form normal equations per type (fast) and vectorize prediction (remove the Python row loop) to improve stability and avoid any accidental slowdowns/timeouts. Submission format, paths, and all checks remain the same, and it still always write `submission.csv`.'
- What this solution (achieved 1.18497) has done: 'Your current gap to the target is very large (lower is better), so the smallest meaningful improvement is to keep the same per-`type` ridge regression approach but add a couple of very cheap, high-signal geometric features derived from the same coordinates: squared distance and absolute coordinate deltas. This preserves the same “per-type closed-form ridge fit on distance-derived features” core logic, but gives the linear model enough flexibility to better approximate the non-linear distance dependence without changing the training loop style or adding new libraries. I also add a per-type centering/scaling computed on train (and applied to test) to stabilize the solve across types and make the regularization behave more consistently (still ridge, same semantics). Submission writing, ID alignment checks, and fallbacks remain unchanged to guarantee a valid `submission.csv`.'
- What this solution (achieved 1.18497) has done: 'Your score is much worse than the target (lower is better), so we should make a small, metric-aligned improvement without changing the overall “per-type ridge regression on geometry features” approach. The most direct fix is to train/predict in log1p-space of the absolute target per coupling `type`, then restore sign; this reduces the influence of large-magnitude couplings and tends to improve MAE-then-log metrics without changing the model family. I also clip extreme restored predictions per type to a robust train-based range to reduce outlier MAE spikes (a common driver of the log-MAE metric), while keeping all features, training loops, and ridge solve identical. Submission writing and ID alignment checks remain unchanged to guarantee a valid `submission.csv`.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower-is-better) is far worse than the target (-1.3487), so we should improve performance without changing the overall “per-type ridge regression on geometry features” core logic. The biggest score killer for this competition is that the metric is computed per `type`, so we keep your per-type setup but make the objective metric-aligned by fitting in a robust MAE-like way: use Huber-style iterative reweighted least squares (IRLS) per type while keeping the same closed-form ridge solves and the same features. This remains the same regression family (still linear ridge per type, same features, same prediction pipeline), but it typically reduces outlier-driven MAE for log-MAE scoring. We also make the clipping bounds slightly less aggressive (1%–99%) to avoid over-clipping signal while still controlling extreme errors, which should move the score downward toward the target.'
- What this solution (achieved 1.18497) has done: 'I fix the crash in `make_pair_label` by avoiding NumPy string ufunc addition and instead building the pair label using pandas string operations, which are stable across pandas/numpy versions. This unblocks feature creation, model fitting, and ensures `submission` is defined so the CSV-writing cell runs. I also make the pair construction robust to missing atoms by filling with empty strings before combining, keeping the rest of the per-type ridge+IRLS logic unchanged. Finally, I keep the same output path and add a small sanity check that the submission columns and row count match expectations.'
- What this solution (achieved 1.18497) has done: 'Your current approach is already a per-type robust linear model on distance-derived geometry, but it likely underfits because it (a) ignores systematic differences by atom pair within each type except for a few one-hot pairs and (b) uses a signed-log target transform that can distort optimization relative to MAE on the original scale. To move the score downward toward the target with minimal semantic change, I keep the same per-type ridge+IRLS core, same features, and same runtime profile, but switch the target to the original `scalar_coupling_constant` scale (better aligned to MAE) while keeping the same robust IRLS weighting to control outliers. I also add a tiny “pair mean residual correction” per (type, pair) learned from train (with smoothing/fallback), which is still just group-statistics on top of the same regression and is very cheap. Submission formatting, ID alignment, and all safety fallbacks remain intact to always produce a valid `submission.csv`.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower-is-better) is far from the target (-1.3487), so we should improve accuracy with the smallest change that keeps your same per-type robust ridge+IRLS regression core. The biggest missing signal that fits your existing pipeline is the per-coupling-type baseline offset (intercept): instead of relying purely on the regression intercept (which is learned after global standardization and can be biased), we add a simple per-`type` mean correction as an explicit baseline feature (a constant “type_bias” term) and keep the rest identical. We also fix a subtle leakage/instability in scaling by ensuring the added constant feature is not standardized (so the intercept remains well-conditioned) and we slightly expand the pair correction to use both (type,pair) and (type,atom_0,atom_1 ordered) with the same smoothing style—still just post-hoc residual group statistics, very cheap. These are minimal, metric-aligned adjustments that usually reduce per-type MAE without changing your model family or training loop, and they still produce a valid `submission.csv` with correct IDs.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/data/champs-scalar-coupling"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "/kaggle/input/champs-scalar-coupling"
print("Using DATA_DIR:", DATA_DIR)
print("Files (first 20):", sorted(os.listdir(DATA_DIR))[:20])



## === cell 1
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

required_train_cols = {
    "id",
    "molecule_name",
    "atom_index_0",
    "atom_index_1",
    "type",
    "scalar_coupling_constant",
}
required_test_cols = {"id", "molecule_name", "atom_index_0", "atom_index_1", "type"}
required_sub_cols = {"id", "scalar_coupling_constant"}

missing_train = required_train_cols - set(train.columns)
missing_test = required_test_cols - set(test.columns)
missing_sub = required_sub_cols - set(sample_sub.columns)
if missing_train:
    raise ValueError(f"train.csv missing columns: {missing_train}")
if missing_test:
    raise ValueError(f"test.csv missing columns: {missing_test}")
if missing_sub:
    raise ValueError(f"sample_submission.csv missing columns: {missing_sub}")

print("train shape:", train.shape)
print("test shape:", test.shape)
print("sample_sub shape:", sample_sub.shape)
print("train types:", train["type"].nunique(), " test types:", test["type"].nunique())

structures = pd.read_csv(
    structures_path, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
)
structures["atom_index"] = structures["atom_index"].astype(np.int32)
structures["atom"] = structures["atom"].astype("category")
print("structures shape:", structures.shape)



## === cell 2
s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)
s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)


def add_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
    df2 = df2.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

    dx = (df2["x0"] - df2["x1"]).astype(np.float64)
    dy = (df2["y0"] - df2["y1"]).astype(np.float64)
    dz = (df2["z0"] - df2["z1"]).astype(np.float64)

    df2["dx_abs"] = np.abs(dx)
    df2["dy_abs"] = np.abs(dy)
    df2["dz_abs"] = np.abs(dz)

    dist2 = dx * dx + dy * dy + dz * dz
    df2["dist2"] = dist2.astype(np.float64)
    df2["dist"] = np.sqrt(dist2).astype(np.float64)
    return df2


train_f = add_distance_feature(train)
test_f = add_distance_feature(test)

type_median = train.groupby("type")["scalar_coupling_constant"].median()
global_median = float(train["scalar_coupling_constant"].median())

EPS = 1e-6
train_f["inv_dist"] = 1.0 / (train_f["dist"].astype(np.float64) + EPS)
test_f["inv_dist"] = 1.0 / (test_f["dist"].astype(np.float64) + EPS)

train_f["inv_dist2"] = train_f["inv_dist"] * train_f["inv_dist"]
test_f["inv_dist2"] = test_f["inv_dist"] * test_f["inv_dist"]


def make_pair_label(a0: pd.Series, a1: pd.Series) -> pd.Series:
    a0s = a0.astype("string").fillna("")
    a1s = a1.astype("string").fillna("")
    lo = a0s.where(a0s <= a1s, a1s)
    hi = a1s.where(a0s <= a1s, a0s)
    return (lo + "_" + hi).astype("string")


type_mean = train.groupby("type")["scalar_coupling_constant"].mean()
train_f["type_bias"] = train_f["type"].map(type_mean).astype(np.float64)
test_f["type_bias"] = (
    test_f["type"].map(type_mean).fillna(global_median).astype(np.float64)
)

train_f["pair"] = make_pair_label(train_f["atom_0"], train_f["atom_1"]).astype(
    "category"
)
test_f["pair"] = make_pair_label(test_f["atom_0"], test_f["atom_1"]).astype("category")

train_f["pair_ord"] = (
    train_f["atom_0"].astype("string").fillna("")
    + ">"
    + train_f["atom_1"].astype("string").fillna("")
).astype("string")
test_f["pair_ord"] = (
    test_f["atom_0"].astype("string").fillna("")
    + ">"
    + test_f["atom_1"].astype("string").fillna("")
).astype("string")

TOPK_PAIRS = 8
top_pairs_by_type = {}
for t, g in train_f.groupby("type", sort=False):
    vc = g["pair"].value_counts(dropna=True)
    top_pairs_by_type[t] = vc.head(TOPK_PAIRS).index.tolist()

FEATURE_COLS_BASE = [
    "dist",
    "inv_dist",
    "inv_dist2",
    "dist2",
    "dx_abs",
    "dy_abs",
    "dz_abs",
    "type_bias",
]

LAMBDA = 3e-3
min_rows = 200

train_f["_y"] = train_f["scalar_coupling_constant"].astype(np.float64)

clip_bounds = {}
for t, g in train_f.groupby("type", sort=False):
    yy = g["_y"].to_numpy(np.float64)
    if yy.size == 0:
        continue
    lo = float(np.nanpercentile(yy, 1.0))
    hi = float(np.nanpercentile(yy, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        continue
    clip_bounds[t] = (lo, hi)


def ridge_solve(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    XtX = X.T @ X
    Xty = X.T @ y
    R = np.diag([0.0] + [lam] * (X.shape[1] - 1)).astype(np.float64)
    A = XtX + R
    return np.linalg.solve(A, Xty).astype(np.float64)


IRLS_ITERS = 4
HUBER_K = 1.5

params = {}
scalers = {}
pair_feature_names_by_type = {}

test_types = test_f["type"].to_numpy()
mask_has_feat = test_f["dist"].notna().to_numpy()
Xtest_base_all = test_f[FEATURE_COLS_BASE].to_numpy(np.float64)
test_pair = test_f["pair"].astype("string").fillna("").to_numpy()

for t, g in train_f.groupby("type", sort=False):
    cols = FEATURE_COLS_BASE + ["pair", "_y"]
    gg = g[cols].dropna(subset=FEATURE_COLS_BASE + ["_y"])
    if gg.shape[0] < min_rows:
        continue

    X_base = gg[FEATURE_COLS_BASE].to_numpy(np.float64)

    top_pairs = top_pairs_by_type.get(t, [])
    pair_feature_names_by_type[t] = top_pairs
    if len(top_pairs) > 0:
        pair_str = gg["pair"].astype("string").fillna("").to_numpy()
        X_pair = np.zeros((gg.shape[0], len(top_pairs)), dtype=np.float64)
        for j, p in enumerate(top_pairs):
            X_pair[:, j] = (pair_str == str(p)).astype(np.float64)
        X_raw = np.concatenate([X_base, X_pair], axis=1)
    else:
        X_raw = X_base

    y = gg["_y"].to_numpy(np.float64)

    mu = X_raw.mean(axis=0)
    sigma = X_raw.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)

    type_bias_idx = FEATURE_COLS_BASE.index("type_bias")
    mu[type_bias_idx] = 0.0
    sigma[type_bias_idx] = 1.0

    Xs = (X_raw - mu) / sigma
    X = np.concatenate([np.ones((Xs.shape[0], 1), dtype=np.float64), Xs], axis=1)

    try:
        w = ridge_solve(X, y, LAMBDA)
    except np.linalg.LinAlgError:
        continue

    for _ in range(IRLS_ITERS):
        r = y - (X @ w)
        abs_r = np.abs(r)

        med = np.median(r)
        mad = np.median(np.abs(r - med))
        s = 1.4826 * mad + 1e-12

        a = abs_r / (HUBER_K * s)
        weights = np.where(a <= 1.0, 1.0, 1.0 / a).astype(np.float64)

        sw = np.sqrt(weights)
        Xw = X * sw[:, None]
        yw = y * sw

        try:
            w = ridge_solve(Xw, yw, LAMBDA)
        except np.linalg.LinAlgError:
            break

    params[t] = w.astype(np.float64)
    scalers[t] = (mu.astype(np.float64), sigma.astype(np.float64))

print("Fitted per-type ridge params:", len(params), "out of", train["type"].nunique())

pred = np.full(test_f.shape[0], np.nan, dtype=np.float64)

for t, w in params.items():
    m = (test_types == t) & mask_has_feat
    if not np.any(m):
        continue

    X_base = Xtest_base_all[m]
    top_pairs = pair_feature_names_by_type.get(t, [])
    if len(top_pairs) > 0:
        pair_str = test_pair[m]
        X_pair = np.zeros((X_base.shape[0], len(top_pairs)), dtype=np.float64)
        for j, p in enumerate(top_pairs):
            X_pair[:, j] = (pair_str == str(p)).astype(np.float64)
        Xraw = np.concatenate([X_base, X_pair], axis=1)
    else:
        Xraw = X_base

    mu, sigma = scalers[t]
    Xs = (Xraw - mu) / sigma
    X = np.concatenate([np.ones((Xs.shape[0], 1), dtype=np.float64), Xs], axis=1)
    pred[m] = X @ w

pred_series = pd.Series(pred, index=test_f.index).astype(np.float64)

train_pred = np.full(train_f.shape[0], np.nan, dtype=np.float64)
train_types = train_f["type"].to_numpy()
mask_train_has_feat = train_f["dist"].notna().to_numpy()
Xtrain_base_all = train_f[FEATURE_COLS_BASE].to_numpy(np.float64)
train_pair = train_f["pair"].astype("string").fillna("").to_numpy()

for t, w in params.items():
    m = (train_types == t) & mask_train_has_feat
    if not np.any(m):
        continue
    X_base = Xtrain_base_all[m]
    top_pairs = pair_feature_names_by_type.get(t, [])
    if len(top_pairs) > 0:
        pair_str = train_pair[m]
        X_pair = np.zeros((X_base.shape[0], len(top_pairs)), dtype=np.float64)
        for j, p in enumerate(top_pairs):
            X_pair[:, j] = (pair_str == str(p)).astype(np.float64)
        Xraw = np.concatenate([X_base, X_pair], axis=1)
    else:
        Xraw = X_base
    mu, sigma = scalers[t]
    Xs = (Xraw - mu) / sigma
    X = np.concatenate([np.ones((Xs.shape[0], 1), dtype=np.float64), Xs], axis=1)
    train_pred[m] = X @ w

train_f["_pred"] = train_pred
train_f["_resid"] = (train_f["_y"] - train_f["_pred"]).astype(np.float64)

resid_df = train_f.loc[
    np.isfinite(train_f["_pred"]), ["type", "pair", "pair_ord", "_resid"]
].copy()

PAIR_RESID_ALPHA = 50.0
grp_u = resid_df.groupby(["type", "pair"])["_resid"].agg(["sum", "count"]).reset_index()
grp_u["corr_u"] = grp_u["sum"] / (grp_u["count"] + PAIR_RESID_ALPHA)
corr_u_map = grp_u.set_index(["type", "pair"])["corr_u"]

PAIR_ORD_ALPHA = 80.0
grp_o = (
    resid_df.groupby(["type", "pair_ord"])["_resid"].agg(["sum", "count"]).reset_index()
)
grp_o["corr_o"] = grp_o["sum"] / (grp_o["count"] + PAIR_ORD_ALPHA)
corr_o_map = grp_o.set_index(["type", "pair_ord"])["corr_o"]

test_key_u = pd.MultiIndex.from_arrays([test_f["type"], test_f["pair"]])
corr_u = corr_u_map.reindex(test_key_u).to_numpy(dtype=np.float64)
corr_u = np.where(np.isfinite(corr_u), corr_u, 0.0)

test_key_o = pd.MultiIndex.from_arrays([test_f["type"], test_f["pair_ord"]])
corr_o = corr_o_map.reindex(test_key_o).to_numpy(dtype=np.float64)
corr_o = np.where(np.isfinite(corr_o), corr_o, 0.0)

pred_series = (pred_series.fillna(0.0) + corr_u + corr_o).astype(np.float64)

pred_series = (
    pred_series.where(
        ~pd.isna(pd.Series(pred, index=test_f.index)),
        other=test_f["type"].map(type_median),
    )
    .fillna(global_median)
    .astype(np.float64)
)

pred_arr = pred_series.to_numpy(np.float64)
for t, (lo, hi) in clip_bounds.items():
    m = test_types == t
    if np.any(m):
        pred_arr[m] = np.clip(pred_arr[m], lo, hi)
pred_series = pd.Series(pred_arr, index=test_f.index)

submission = pd.DataFrame(
    {"id": test["id"].values, "scalar_coupling_constant": pred_series.values}
)

if list(submission.columns) != ["id", "scalar_coupling_constant"]:
    raise ValueError("Submission columns are incorrect.")
if submission["id"].isna().any():
    raise ValueError("Submission contains NaN ids.")
if submission["scalar_coupling_constant"].isna().any():
    raise ValueError("Submission contains NaN predictions.")
if submission.shape[0] != test.shape[0]:
    raise ValueError("Submission row count does not match test row count.")
if submission["id"].nunique() != submission.shape[0]:
    raise ValueError("Duplicate ids detected in submission.")

print(submission["scalar_coupling_constant"].describe())
print("Missing dist in test:", int(test_f["dist"].isna().sum()), " / ", test_f.shape[0])
print(
    "Missing atom_0 in test:",
    int(test_f["atom_0"].isna().sum()),
    " / ",
    test_f.shape[0],
)
print(
    "Missing atom_1 in test:",
    int(test_f["atom_1"].isna().sum()),
    " / ",
    test_f.shape[0],
)



## === cell 3
out_path = "submission.csv"
submission.to_csv(out_path, index=False)

sample_ids = set(sample_sub["id"].astype(np.int64).tolist())
sub_ids = set(submission["id"].astype(np.int64).tolist())
if sample_ids != sub_ids:
    print("WARNING: submission ids set differs from sample_submission ids set.")
else:
    print("Submission ids set matches sample_submission.")

print("Wrote:", out_path, " rows:", submission.shape[0])



## === cell 4
try:
    ax = submission["scalar_coupling_constant"].plot(
        kind="hist", bins=100, title="Predicted scalar_coupling_constant (hist)"
    )
    fig = ax.get_figure()
    fig.tight_layout()
except Exception as e:
    print("Plot skipped:", repr(e))
