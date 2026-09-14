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

-1.6777209112242684

# 6. Current score

4.34123

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.18497) has done: 'I remove the dependency on missing external Kaggle datasets (`keras-neural-net-and-distance-features` and `keras-nn-with-multi-output`) and instead build a self-contained baseline that always produces predictions for every test `id`. To keep core logic minimal and stable, I use a simple per-`type` median target computed from `train.csv` and apply it to `test.csv`, with a global fallback for any unseen types. I also ensure the submission contains all required ids exactly once, sorted by `id`, matching the sample submission format. This fix the runtime errors and the “Missing required ids” submission error while yielding a reasonable non-null score.'
- What this solution (achieved 1.18523) has done: 'Your current score (1.18497, lower-is-better) is far worse than the target (-1.6777), so we should improve legitimately while keeping the same “per-type constant prediction” core logic. The smallest strong gain is to use the known decomposition of the target: predict per-type medians for the four contribution terms (`fc`,`sd`,`pso`,`dso`) from `scalar_coupling_contributions.csv` and sum them to form the final prediction; for test, merge contributions by `(molecule_name, atom_index_0, atom_index_1, type)` and fall back to per-type median constants when missing. This preserves the “median-per-type baseline” approach but aligns it with a much more predictive signal that is available for both train and test. We also keep all the submission integrity checks (ids, sorting, no duplicates) unchanged to ensure a valid submission CSV.'
- What this solution (achieved 1.18523) has done: 'Your current approach already uses the best “available-for-test” signal (the contributions file) and then falls back to per-type medians, so the main realistic gain with minimal logic change is to (1) make the contributions join robust to reversed atom order, and (2) align training/test aggregates by symmetrizing the contributions table the same way. This keeps the same “merge contributions if present, else per-type median of terms, else per-type target median” semantics, but should reduce missing merges and improve MAE without changing the model concept. I also avoid the slow per-row `map(lambda ...)` and replace it with a vectorized reindex+fill, which is deterministic and reduces runtime risk. Output path, submission schema, id alignment checks, and the overall baseline logic remain unchanged.'
- What this solution (achieved 1.23566) has done: 'You’re far from the (lower-is-better) target, so the most direct way to move the score toward it—without changing the “use contributions when available, else per-type medians” core logic—is to stop using **medians** and use **means** for the contribution aggregation and per-type term estimates, because the competition metric is based on MAE and the contributions are already a strong signal (we want the most accurate expectation, not a robust constant). I also make the merge against the symmetrized contributions deterministic by pre-building a unique index on the key (after mean-aggregation) and joining via `set_index().join(...)`, which avoids any subtle merge duplication behavior while keeping exactly the same semantics. No model, loops, features, or loss are introduced—this remains a constant-per-key/per-type baseline, just with a better (MAE-aligned) aggregator. Submission format checks and output path remain unchanged.'
- What this solution (achieved 4.34123) has done: 'Your current score (1.23566, lower-is-better) is far worse than the target (-1.6777), so we should improve with the smallest change that keeps your “use contributions when available, else per-type constant” core logic. The main issue is that you’re using the `scalar_coupling_contributions.csv` values, which are known only for the training set and therefore cannot match test keys—so almost all test rows fall back to weak per-type constants. The minimal legitimate fix is to instead build contribution-like signals from files that exist for both train and test (structures, mulliken charges, shielding tensors, dipole, potential energy) and use those as per-pair features to predict each term (`fc/sd/pso/dso`) with a simple per-type linear regression, then sum terms (same semantic “sum of terms” logic). We keep deterministic splits by molecule, fit separate models per type (no architecture change beyond swapping the constant aggregator for a tiny sklearn linear model), and preserve submission integrity checks and output path.'
- What this solution (achieved 4.34123) has done: 'I fix the crash by ensuring the calibration merge doesn’t lose/rename the `scalar_coupling_constant` column: right now `train_terms` already contains the target (from `train_feat`), so merging it again by `id` creates suffixes and triggers the KeyError. I make calibration use the existing target column directly (and add a safe fallback if it’s missing), keeping the model training/prediction logic unchanged. I also keep the submission integrity checks and ensure `sub` is always created so cell 5 can write `/kaggle/working/submission.csv`. These changes are score-neutral to mildly positive (better calibration correctness) and mainly unblock end-to-end execution.'
- What this solution (achieved 4.34123) has done: 'Your current score (4.34123, lower-is-better) is much worse than the target (-1.6777), and the biggest driver is that the “term models” are trained on `fc/sd/pso/dso` contributions that are **not available for test**, so the pipeline effectively learns a proxy that cannot generalize well. To move the score toward the target with minimal semantic change, I keep the same “predict 4 terms then sum + per-type linear calibration” structure, but I switch the term targets to **legitimate, always-available residual terms** derived from `scalar_coupling_constant - per_type_mean`, split evenly into 4 pseudo-terms (so their sum equals the residual). This preserves your existing multi-term Ridge training and calibration logic while making the supervision signal consistent between train and test. I also remove the dependency on `scalar_coupling_contributions.csv` to avoid leaking train-only information and stabilize the training set size.'

# 9. Code solution

## === cell 0
import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

BASE_PATH = "../input/champs-scalar-coupling"
print("Listing ../input:", os.listdir("../input")[:10])
print("Using BASE_PATH:", BASE_PATH)

np.random.seed(0)



## === cell 1
train = pd.read_csv(
    f"{BASE_PATH}/train.csv",
    usecols=[
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
)
test = pd.read_csv(
    f"{BASE_PATH}/test.csv",
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
)

sample_sub = pd.read_csv(
    f"{BASE_PATH}/sample_submission.csv", usecols=["id", "scalar_coupling_constant"]
)

print("train:", train.shape, "test:", test.shape, "sample_sub:", sample_sub.shape)
print(
    "Unique types in train:",
    train["type"].nunique(),
    "in test:",
    test["type"].nunique(),
)



## === cell 2
structures = pd.read_csv(
    f"{BASE_PATH}/structures.csv",
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
)

mulliken = pd.read_csv(
    f"{BASE_PATH}/mulliken_charges.csv",
    usecols=["molecule_name", "atom_index", "mulliken_charge"],
)

shield = pd.read_csv(
    f"{BASE_PATH}/magnetic_shielding_tensors.csv",
    usecols=["molecule_name", "atom_index", "XX", "YY", "ZZ"],
)

dipole = pd.read_csv(
    f"{BASE_PATH}/dipole_moments.csv",
    usecols=["molecule_name", "X", "Y", "Z"],
).rename(columns={"X": "dipole_X", "Y": "dipole_Y", "Z": "dipole_Z"})

pot = pd.read_csv(
    f"{BASE_PATH}/potential_energy.csv",
    usecols=["molecule_name", "potential_energy"],
)

atom_map = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
structures["atom_num"] = structures["atom"].map(atom_map).fillna(0).astype(np.int16)

atom_df = structures.merge(mulliken, on=["molecule_name", "atom_index"], how="left")
atom_df = atom_df.merge(shield, on=["molecule_name", "atom_index"], how="left")

for c in ["mulliken_charge", "XX", "YY", "ZZ"]:
    atom_df[c] = atom_df[c].astype(np.float32)

for c in ["mulliken_charge", "XX", "YY", "ZZ"]:
    atom_df[c] = atom_df[c].fillna(atom_df[c].mean())

atom_df.head()



## === cell 3
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge

pair_key_cols = ["molecule_name", "atom_index_0", "atom_index_1", "type"]


def add_pair_features(df: pd.DataFrame, atom_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    a0 = atom_df.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom_num": "atom0_num",
            "x": "x0",
            "y": "y0",
            "z": "z0",
            "mulliken_charge": "q0",
            "XX": "xx0",
            "YY": "yy0",
            "ZZ": "zz0",
        }
    )[
        [
            "molecule_name",
            "atom_index_0",
            "atom0_num",
            "x0",
            "y0",
            "z0",
            "q0",
            "xx0",
            "yy0",
            "zz0",
        ]
    ]

    a1 = atom_df.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom_num": "atom1_num",
            "x": "x1",
            "y": "y1",
            "z": "z1",
            "mulliken_charge": "q1",
            "XX": "xx1",
            "YY": "yy1",
            "ZZ": "zz1",
        }
    )[
        [
            "molecule_name",
            "atom_index_1",
            "atom1_num",
            "x1",
            "y1",
            "z1",
            "q1",
            "xx1",
            "yy1",
            "zz1",
        ]
    ]

    df = df.merge(a0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(a1, on=["molecule_name", "atom_index_1"], how="left")

    dx = (df["x0"] - df["x1"]).astype(np.float32)
    dy = (df["y0"] - df["y1"]).astype(np.float32)
    dz = (df["z0"] - df["z1"]).astype(np.float32)
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)
    df["dist2"] = (df["dist"] ** 2).astype(np.float32)
    df["inv_dist"] = (1.0 / (df["dist"] + 1e-6)).astype(np.float32)

    df["atom_sum"] = (df["atom0_num"] + df["atom1_num"]).astype(np.float32)
    df["atom_prod"] = (df["atom0_num"] * df["atom1_num"]).astype(np.float32)
    df["q_sum"] = (df["q0"] + df["q1"]).astype(np.float32)
    df["q_diff"] = (df["q0"] - df["q1"]).astype(np.float32)
    df["q_absdiff"] = np.abs(df["q_diff"]).astype(np.float32)

    df["shield0"] = (df["xx0"] + df["yy0"] + df["zz0"]).astype(np.float32)
    df["shield1"] = (df["xx1"] + df["yy1"] + df["zz1"]).astype(np.float32)
    df["shield_sum"] = (df["shield0"] + df["shield1"]).astype(np.float32)
    df["shield_absdiff"] = np.abs(df["shield0"] - df["shield1"]).astype(np.float32)

    df = df.merge(dipole, on="molecule_name", how="left")
    df = df.merge(pot, on="molecule_name", how="left")
    for c in ["dipole_X", "dipole_Y", "dipole_Z", "potential_energy"]:
        df[c] = df[c].astype(np.float32).fillna(0.0)

    df["dipole_norm"] = np.sqrt(
        df["dipole_X"] ** 2 + df["dipole_Y"] ** 2 + df["dipole_Z"] ** 2
    ).astype(np.float32)

    feat_cols = [
        "dist",
        "dist2",
        "inv_dist",
        "atom0_num",
        "atom1_num",
        "atom_sum",
        "atom_prod",
        "q0",
        "q1",
        "q_sum",
        "q_diff",
        "q_absdiff",
        "shield0",
        "shield1",
        "shield_sum",
        "shield_absdiff",
        "dipole_X",
        "dipole_Y",
        "dipole_Z",
        "dipole_norm",
        "potential_energy",
    ]
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].astype(np.float32).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return df, feat_cols


train_feat, feat_cols = add_pair_features(train, atom_df)
test_feat, _ = add_pair_features(test, atom_df)

print("Feature columns:", len(feat_cols))
train_feat[feat_cols].describe().T.head()



## === cell 4
term_cols = ["fc", "sd", "pso", "dso"]

type_mean_target = train.groupby("type")["scalar_coupling_constant"].mean()
global_mean_target = float(train["scalar_coupling_constant"].mean())

train_terms = train_feat.copy()
train_terms["type_mean_target"] = (
    train_terms["type"].map(type_mean_target).astype(np.float32)
)
train_terms["residual"] = (
    train_terms["scalar_coupling_constant"].astype(np.float32)
    - train_terms["type_mean_target"]
).astype(np.float32)

for c in term_cols:
    train_terms[c] = (train_terms["residual"] * 0.25).astype(np.float32)

print("Train rows used for term fitting:", train_terms.shape, "(no term rows dropped)")

type_term_mean = train_terms.groupby("type")[term_cols].mean()
global_term_mean = train_terms[term_cols].mean()

gkf = GroupKFold(n_splits=3)

alpha = 1.0
models = {}  # (type, term) -> fitted model
type_feat_means = {}  # type -> feature mean for safe fill

X_all = train_terms[feat_cols].to_numpy(dtype=np.float32)
groups_all = train_terms["molecule_name"].values
types_all = train_terms["type"].values

for t in sorted(train_terms["type"].unique()):
    mask = types_all == t
    X_t = X_all[mask]
    g_t = groups_all[mask]
    if X_t.shape[0] < 500:
        continue
    if pd.Series(g_t).nunique() < 3:
        continue

    type_feat_means[t] = np.nanmean(X_t, axis=0).astype(np.float32)

    tr_idx, _ = next(gkf.split(X_t, groups=g_t))

    X_tr = X_t[tr_idx]
    for term in term_cols:
        y_t = train_terms.loc[mask, term].to_numpy(dtype=np.float32)
        y_tr = y_t[tr_idx]
        m = Ridge(alpha=alpha)
        m.fit(X_tr, y_tr)
        models[(t, term)] = m

print("Fitted term models:", len(models))

X_test = test_feat[feat_cols].to_numpy(dtype=np.float32)
test_types = test_feat["type"].values

pred_terms = np.zeros((test_feat.shape[0], len(term_cols)), dtype=np.float32)
default_terms = (
    type_term_mean.reindex(test_feat["type"])
    .fillna(global_term_mean)
    .to_numpy(dtype=np.float32)
)
pred_terms[:] = default_terms

for i, term in enumerate(term_cols):
    for t in np.unique(test_types):
        idx = np.where(test_types == t)[0]
        key = (t, term)
        if key in models:
            X_part = X_test[idx].copy()
            if np.isnan(X_part).any():
                fill = type_feat_means.get(
                    t, np.zeros(X_part.shape[1], dtype=np.float32)
                )
                nanmask = np.isnan(X_part)
                X_part[nanmask] = np.take(fill, np.where(nanmask)[1])
            pred_terms[idx, i] = models[key].predict(X_part).astype(np.float32)

pred_from_terms_sum = pred_terms.sum(axis=1).astype(np.float32)

train_terms_sum = train_terms[term_cols].to_numpy(dtype=np.float32).sum(axis=1)
y_target = train_terms["scalar_coupling_constant"].to_numpy(dtype=np.float32)
types_train_terms = train_terms["type"].values

calib_a = {}
calib_b = {}
for t in sorted(pd.unique(types_train_terms)):
    m = types_train_terms == t
    if m.sum() < 1000:
        continue
    x = train_terms_sum[m].astype(np.float64)  # predicted residual proxy
    y = (y_target[m] - type_mean_target.loc[t]).astype(np.float64)  # true residual
    x_mean = x.mean()
    y_mean = y.mean()
    var = ((x - x_mean) ** 2).mean()
    if var < 1e-12:
        continue
    cov = ((x - x_mean) * (y - y_mean)).mean()
    a = cov / var
    b = y_mean - a * x_mean
    calib_a[t] = float(a)
    calib_b[t] = float(b)

pred_resid_cal = pred_from_terms_sum.astype(np.float64)
for t in np.unique(test_types):
    idx = np.where(test_types == t)[0]
    if t in calib_a:
        pred_resid_cal[idx] = calib_a[t] * pred_resid_cal[idx] + calib_b[t]

pred_final = pred_resid_cal.astype(np.float64) + test_feat["type"].map(
    type_mean_target
).to_numpy(dtype=np.float64)

pred = pd.Series(pred_final).astype(float)
pred = (
    pred.fillna(test_feat["type"].map(type_mean_target))
    .fillna(global_mean_target)
    .astype(float)
)

sub = (
    pd.DataFrame({"id": test["id"].astype(np.int64), "scalar_coupling_constant": pred})
    .sort_values("id")
    .reset_index(drop=True)
)

missing = set(sample_sub["id"]) - set(sub["id"])
extra = set(sub["id"]) - set(sample_sub["id"])
dupes = sub["id"].duplicated().sum()

print("Missing ids vs sample:", len(missing))
print("Extra ids vs sample:", len(extra))
print("Duplicate ids:", dupes)

assert len(missing) == 0, "Submission is missing required ids."
assert len(extra) == 0, "Submission has extra ids not in sample submission."
assert dupes == 0, "Submission contains duplicate ids."
assert sub.shape[0] == sample_sub.shape[0], "Row count mismatch with sample submission."

sub.head()



## === cell 5
out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub.describe(include="all"))
