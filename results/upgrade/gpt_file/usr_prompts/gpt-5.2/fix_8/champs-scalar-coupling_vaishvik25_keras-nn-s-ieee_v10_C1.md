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

-1.572352650827347

# 6. Current score

1.23982

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'The notebook is trying to ensemble (“stack”) predictions from external datasets that aren’t present in your environment (`../input/top-mol`, `../input/another-one`, etc.), so it fails before producing any submission. I fix this by switching the input discovery to the actual provided competition path (`/kaggle/data/champs-scalar-coupling/`) and by adding a safe fallback that generates a baseline submission when no external prediction files exist. I also remove IPython-only magic (`%matplotlib inline`) so it runs as a plain Python script, and fix deprecated `np.bool`. This run end-to-end and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is far worse than the target (-1.5723), so we should improve performance while keeping your “groupby-mean fallback” core logic intact. The biggest issue is leakage/overfitting in the fallback: you’re using `molecule_name` at inference, but train/test are split by molecule so those lookups almost always miss and add noise; switching to a type-level baseline (plus a tiny amount of atom-pair geometry from `structures.csv`) is a minimal, legitimate improvement. We keep the same overall approach (compute simple aggregated statistics from train, merge into test, fill NAs, write submission), but add one robust feature: inter-atomic distance and type-specific distance→target calibration via per-type linear regression (closed-form, no iterative training loop). If any external submission-like CSVs are found, we keep your ensemble path unchanged.'
- What this solution (achieved 1.23566) has done: 'We keep your existing “type-wise distance→target linear calibration” fallback intact, but make two minimal, score-relevant fixes: (1) compute distances in float64 (then store as float32) to reduce numerical noise in the per-type regression coefficients, and (2) make the regression more robust by fitting on `1/dist` (physically more correlated with coupling strength than raw distance) while preserving the same closed-form, per-type linear model structure (still just `a + b*x` per type). This typically improves MAE/logMAE substantially versus raw distance while staying within your same feature-extraction/training semantics (no iterative training, no new models). Everything else (external-sub ensembling path, I/O paths, submission writing) remains unchanged and still always produce a valid `submission.csv`.'
- What this solution (achieved 1.24003) has done: 'Your current score (1.23566, lower-is-better) is far from the target (-1.5723), so we should legitimately improve predictions while keeping your “per-type closed-form linear calibration on 1/dist” core logic unchanged. The biggest minimal win is to fit the linear calibration per coupling `type` on a **log1p-transformed target** (still just `a + b*x` per type, still closed-form) and then invert back with `expm1`; this aligns better with the competition’s log-MAE behavior and reduces the effect of large outliers without changing the modeling approach. I also add a tiny, safe clamp on distance to avoid infinities/unstable extreme `1/dist`, and keep the external-submission ensembling path exactly as-is. The script still runs end-to-end and always writes a valid `submission.csv` with the required columns and row alignment.'
- What this solution (achieved 1.23982) has done: 'Your current score (1.24003, lower-is-better) is far worse than the target (-1.57235), so we should improve accuracy while keeping your same fallback core logic (per-type closed-form linear fit on `inv_dist` with a log1p-signed target transform). The largest minimal win is to make the linear fit less sensitive to outliers by winsorizing the transformed target **within each coupling type** before fitting (still the same model form `a + b*x`, still closed-form, just a more robust estimate). I also compute the regression coefficients in float64 (then store as float32) to reduce numerical noise, without changing the model structure. Everything else (external submission ensembling path, distance feature, submission format) remains the same and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 1.23982) has done: 'We keep your exact fallback modeling approach (per-type closed-form linear fit on `inv_dist` with signed-log1p target transform and per-type winsorization), but make one score-relevant improvement: add a second physically meaningful feature (`inv_dist2 = 1/dist^2`) and fit the same closed-form linear model on both features per coupling type. This preserves the “no iterative training loop” core logic while typically improving MAE because coupling strength often follows a steeper distance decay than 1/r. To avoid overfitting/noise, we also add a minimal ridge term in the 2×2 normal equations (still closed-form) and keep the existing small-sample fallback behavior. Everything else (external submission ensembling path, I/O paths, submission format) stays unchanged and it still always writes a valid `submission.csv`.'
- What this solution (achieved 1.23982) has done: 'We keep your exact fallback approach (type-wise closed-form linear regression on `inv_dist` and `inv_dist2` with signed-log1p target and per-type winsorization), but fix two score-relevant issues that can silently hurt MAE: (1) the per-type quantile clipping currently uses float32 and a slow/unstable `transform(lambda quantile)` path; we compute per-type clip thresholds once in float64 and merge them back (same semantics, more stable and faster), and (2) we make the regression numerically safer by solving the 2×2 system per type using sums (not means) and a slightly larger ridge to avoid near-singular types (still closed-form, same model). These are minimal changes that generally reduce noise/outliers sensitivity and should improve the score (lower is better), without changing the overall modeling logic or adding iterative training. Output format/paths remain identical and it still always writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

BASE_DIR = "/kaggle/data/champs-scalar-coupling"
TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")
STRUCTURES_PATH = os.path.join(BASE_DIR, "structures.csv")

print("Listing base dir:", BASE_DIR)
print("Exists:", os.path.exists(BASE_DIR))
print("Some files:", sorted(os.listdir(BASE_DIR))[:20])



## === cell 1
train = pd.read_csv(
    TRAIN_PATH,
    usecols=[
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
)
test = pd.read_csv(
    TEST_PATH,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print("train shape:", train.shape)
print("test shape:", test.shape)
print("sample_sub shape:", sample_sub.shape)
print("train columns:", train.columns.tolist())
print("test columns:", test.columns.tolist())



## === cell 2
SEARCH_ROOTS = [
    "/kaggle/data",
    "/kaggle/input",
    "/kaggle/working",
]


def find_candidate_prediction_csvs(roots, max_files=50):
    candidates = []
    for root in roots:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                if not fn.lower().endswith(".csv"):
                    continue
                fpath = os.path.join(dirpath, fn)
                if os.path.basename(fpath) in {
                    "train.csv",
                    "test.csv",
                    "structures.csv",
                    "scalar_coupling_contributions.csv",
                    "magnetic_shielding_tensors.csv",
                    "mulliken_charges.csv",
                    "dipole_moments.csv",
                    "potential_energy.csv",
                    "sample_submission.csv",
                }:
                    continue
                candidates.append(fpath)
                if len(candidates) >= max_files:
                    return candidates
    return candidates


candidate_csvs = find_candidate_prediction_csvs(SEARCH_ROOTS, max_files=200)
print("Found candidate external CSVs (up to 200):", len(candidate_csvs))
print("\n".join(candidate_csvs[:20]))




## === cell 3
def load_submission_like_csv(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    cols = set(df.columns)
    if "id" in cols and "scalar_coupling_constant" in cols:
        df = df[["id", "scalar_coupling_constant"]].copy()
        return df
    return None


external_subs = []
for p in candidate_csvs:
    df = load_submission_like_csv(p)
    if df is None:
        continue
    external_subs.append((p, df))

print("External submission-like CSVs loaded:", len(external_subs))
for p, df in external_subs[:10]:
    print(p, df.shape)




## === cell 4
def add_distance_feature(df_pairs, structures_df):
    s = structures_df[["molecule_name", "atom_index", "x", "y", "z"]].copy()

    s0 = s.rename(
        columns={
            "atom_index": "atom_index_0",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
    )
    s1 = s.rename(
        columns={
            "atom_index": "atom_index_1",
            "x": "x1",
            "y": "y1",
            "z": "z1",
        }
    )

    out = df_pairs.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
    out = out.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

    dx = out["x0"].astype(np.float64) - out["x1"].astype(np.float64)
    dy = out["y0"].astype(np.float64) - out["y1"].astype(np.float64)
    dz = out["z0"].astype(np.float64) - out["z1"].astype(np.float64)
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)

    out["dist"] = dist.astype(np.float32)
    out = out.drop(columns=["x0", "y0", "z0", "x1", "y1", "z1"])
    return out


def fit_typewise_linear_2feat(train_feat, x1_col, x2_col, y_col, ridge=1e-5):
    tmp = train_feat[["type", x1_col, x2_col, y_col]].copy()

    tmp["x1"] = tmp[x1_col].astype(np.float64)
    tmp["x2"] = tmp[x2_col].astype(np.float64)
    tmp["y"] = tmp[y_col].astype(np.float64)

    tmp["x1x1"] = tmp["x1"] * tmp["x1"]
    tmp["x2x2"] = tmp["x2"] * tmp["x2"]
    tmp["x1x2"] = tmp["x1"] * tmp["x2"]
    tmp["x1y"] = tmp["x1"] * tmp["y"]
    tmp["x2y"] = tmp["x2"] * tmp["y"]

    stats = (
        tmp.groupby("type", sort=False)
        .agg(
            n=("y", "size"),
            sum_y=("y", "sum"),
            sum_x1=("x1", "sum"),
            sum_x2=("x2", "sum"),
            sum_x1x1=("x1x1", "sum"),
            sum_x2x2=("x2x2", "sum"),
            sum_x1x2=("x1x2", "sum"),
            sum_x1y=("x1y", "sum"),
            sum_x2y=("x2y", "sum"),
            mean_y=("y", "mean"),
        )
        .reset_index()
    )

    n = stats["n"].astype(np.float64).values
    sum_y = stats["sum_y"].astype(np.float64).values
    sum_x1 = stats["sum_x1"].astype(np.float64).values
    sum_x2 = stats["sum_x2"].astype(np.float64).values

    sum_x1x1 = stats["sum_x1x1"].astype(np.float64).values
    sum_x2x2 = stats["sum_x2x2"].astype(np.float64).values
    sum_x1x2 = stats["sum_x1x2"].astype(np.float64).values
    sum_x1y = stats["sum_x1y"].astype(np.float64).values
    sum_x2y = stats["sum_x2y"].astype(np.float64).values

    eps_n = 1e-30
    inv_n = 1.0 / (n + eps_n)

    a11 = sum_x1x1 - (sum_x1 * sum_x1) * inv_n
    a22 = sum_x2x2 - (sum_x2 * sum_x2) * inv_n
    a12 = sum_x1x2 - (sum_x1 * sum_x2) * inv_n

    c1 = sum_x1y - (sum_x1 * sum_y) * inv_n
    c2 = sum_x2y - (sum_x2 * sum_y) * inv_n

    r = float(ridge)
    a11r = a11 + r
    a22r = a22 + r

    det = a11r * a22r - a12 * a12
    inv_det = 1.0 / (det + 1e-18)

    b1 = (a22r * c1 - a12 * c2) * inv_det
    b2 = (a11r * c2 - a12 * c1) * inv_det

    mean_x1 = sum_x1 * inv_n
    mean_x2 = sum_x2 * inv_n
    mean_y = sum_y * inv_n
    a0 = mean_y - b1 * mean_x1 - b2 * mean_x2

    stats["a"] = a0.astype(np.float32)
    stats["b1"] = b1.astype(np.float32)
    stats["b2"] = b2.astype(np.float32)

    stats["type_mean"] = stats["mean_y"].astype(np.float32)
    small = stats["n"] < 50
    stats.loc[small, "b1"] = 0.0
    stats.loc[small, "b2"] = 0.0
    stats.loc[small, "a"] = stats.loc[small, "type_mean"].astype(np.float32)

    return stats[["type", "a", "b1", "b2", "type_mean", "n"]]


def build_fallback_submission(train_df, test_df, sample_sub_df):
    structures = pd.read_csv(
        STRUCTURES_PATH,
        usecols=["molecule_name", "atom_index", "x", "y", "z"],
        dtype={
            "molecule_name": "category",
            "atom_index": np.int16,
            "x": np.float32,
            "y": np.float32,
            "z": np.float32,
        },
    )

    train_feat = add_distance_feature(train_df, structures)
    test_feat = add_distance_feature(test_df, structures)

    dist_floor = np.float32(1e-3)
    train_dist = np.maximum(train_feat["dist"].values.astype(np.float32), dist_floor)
    test_dist = np.maximum(test_feat["dist"].values.astype(np.float32), dist_floor)

    train_feat["inv_dist"] = (1.0 / train_dist).astype(np.float32)
    test_feat["inv_dist"] = (1.0 / test_dist).astype(np.float32)

    train_feat["inv_dist2"] = (
        train_feat["inv_dist"].values.astype(np.float32) ** 2
    ).astype(np.float32)
    test_feat["inv_dist2"] = (
        test_feat["inv_dist"].values.astype(np.float32) ** 2
    ).astype(np.float32)

    y = train_feat["scalar_coupling_constant"].astype(np.float64).values
    y_sign = np.sign(y)
    y_trans = y_sign * np.log1p(np.abs(y))
    train_feat["y_trans"] = y_trans.astype(np.float32)

    q = (
        train_feat[["type", "y_trans"]]
        .assign(y_trans=lambda d: d["y_trans"].astype(np.float64))
        .groupby("type", sort=False)["y_trans"]
        .quantile([0.01, 0.99])
        .unstack()
        .reset_index()
        .rename(columns={0.01: "q_low", 0.99: "q_high"})
    )
    train_feat = train_feat.merge(q, on="type", how="left")
    y_trans_f64 = train_feat["y_trans"].astype(np.float64)
    train_feat["y_trans_clip"] = y_trans_f64.clip(
        lower=train_feat["q_low"].astype(np.float64),
        upper=train_feat["q_high"].astype(np.float64),
    ).astype(np.float32)
    train_feat = train_feat.drop(columns=["q_low", "q_high"])

    type_lr = fit_typewise_linear_2feat(
        train_feat,
        x1_col="inv_dist",
        x2_col="inv_dist2",
        y_col="y_trans_clip",
        ridge=1e-5,
    )
    test_feat = test_feat.merge(type_lr, on="type", how="left")

    pred_trans = (
        test_feat["a"].astype(np.float64)
        + test_feat["b1"].astype(np.float64) * test_feat["inv_dist"].astype(np.float64)
        + test_feat["b2"].astype(np.float64) * test_feat["inv_dist2"].astype(np.float64)
    )
    pred = np.sign(pred_trans) * np.expm1(np.abs(pred_trans))
    test_feat["scalar_coupling_constant"] = pred.astype(np.float32)

    missing = test_feat["scalar_coupling_constant"].isna()
    if missing.any():
        tm = test_feat.loc[missing, "type_mean"].astype(np.float64).values
        tm_inv = np.sign(tm) * np.expm1(np.abs(tm))
        test_feat.loc[missing, "scalar_coupling_constant"] = tm_inv.astype(np.float32)

    merged = test_feat[["id", "scalar_coupling_constant"]].copy()
    out = sample_sub_df[["id"]].merge(merged, on="id", how="left")

    if out["scalar_coupling_constant"].isna().any():
        overall_mean = float(train_df["scalar_coupling_constant"].mean())
        out["scalar_coupling_constant"] = out["scalar_coupling_constant"].fillna(
            overall_mean
        )

    return out


if len(external_subs) >= 1:
    base = sample_sub[["id"]].copy()
    pred_cols = []
    for i, (p, df) in enumerate(external_subs):
        col = f"mol{i}"
        tmp = df.rename(columns={"scalar_coupling_constant": col})
        base = base.merge(tmp, on="id", how="left")
        pred_cols.append(col)

    non_null_cols = [c for c in pred_cols if base[c].notna().any()]
    base = base[["id"] + non_null_cols].copy()
    concat_sub = base

    if len(non_null_cols) == 0:
        print(
            "External files found but none had usable predictions for our ids; using fallback baseline."
        )
        submission = build_fallback_submission(train, test, sample_sub)
    else:
        concat_sub["m_mean"] = concat_sub.loc[:, non_null_cols].mean(axis=1)

        submission = concat_sub[["id"]].copy()
        submission["scalar_coupling_constant"] = concat_sub["m_mean"].astype(float)

        if submission["scalar_coupling_constant"].isna().any():
            fallback = build_fallback_submission(train, test, sample_sub)
            submission = submission.merge(
                fallback, on="id", how="left", suffixes=("", "_fb")
            )
            submission["scalar_coupling_constant"] = submission[
                "scalar_coupling_constant"
            ].fillna(submission["scalar_coupling_constant_fb"])
            submission = submission[["id", "scalar_coupling_constant"]]
else:
    print("No external prediction files available; using fallback baseline.")
    submission = build_fallback_submission(train, test, sample_sub)

print("submission shape:", submission.shape)
print(submission.head())



## === cell 5
submission.to_csv("submission.csv", index=False, float_format="%.6f")

chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["id", "scalar_coupling_constant"]
assert chk.shape[0] == sample_sub.shape[0]
assert chk["id"].isna().sum() == 0
print("Wrote submission.csv with rows:", chk.shape[0])
print(chk.head())
