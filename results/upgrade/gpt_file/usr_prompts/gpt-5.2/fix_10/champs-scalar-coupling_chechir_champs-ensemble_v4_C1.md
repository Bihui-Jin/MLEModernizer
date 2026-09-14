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

-2.087064111212748

# 6. Current score

2.07752

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'The current notebook fails because it tries to read out-of-environment input files (`../input/champ-preds` and `../input/1-mpnn`) that do not exist in your provided filesystem, so no submission is ever written. I replace that dependency with an in-notebook baseline that uses only the provided competition data files, while keeping the overall approach “simple and stable” (a per-coupling-type mean predictor) so it runs fast and always outputs a valid `submission.csv`. This yield a valid score (not necessarily competitive), unblocking end-to-end execution and allowing further score-calibration iterations later. I also make the output strictly match the required columns and row order by merging onto `sample_submission.csv`.'
- What this solution (achieved 1.23566) has done: 'Your current predictor is only using the coupling `type` mean, which leaves a lot of easy signal unused and results in a much worse (higher) log-MAE than your target. To move the score downward toward the target with minimal core-logic change, I keep the same “groupby mean fallback baseline” approach, but condition the mean on `(type, atom_0, atom_1)` by joining atom symbols from `structures.csv`. This adds strong chemistry signal while staying within the same simple aggregation/prediction semantics and should materially reduce error without changing modeling/training loops (there are none). I also keep safe backoffs: if a fine-grained group is unseen, fall back to type mean, then global mean, and ensure the submission aligns exactly to `sample_submission.csv`.'
- What this solution (achieved 1.23566) has done: 'The crash comes from missing structure merges (some rows have NaN coordinates/atoms), which makes `dist` NaN and then `.astype("int32")` fails when creating `dist_bin`. I keep the same core “groupby mean with backoff” logic, but make `dist_bin` nullable (`Int32`) and explicitly handle non-finite distances so the pipeline never errors. I also add a final safety fill for any remaining missing `atom_0/atom_1` keys so the mapping works predictably, and ensure the submission is written as `submission.csv` with the required columns and row order.'
- What this solution (achieved 1.23566) has done: 'Your current baseline is still far above the target (lower-is-better), so we should add a bit more signal without changing the core “groupby-mean with backoff” logic. The smallest high-impact change is to use a continuous distance feature more effectively by (1) clipping extreme distances, (2) using finer bins, and (3) adding a second “dist-only within type” fallback before dropping to `(type, atom_0, atom_1)` and then `type`. This keeps the same prediction semantics (pure aggregations from train) but reduces variance and improves matching for unseen atom-pairs/dist bins. I also enforce a canonical atom-pair ordering (swap atoms/indices so `(atom_0,atom_1)` is order-invariant), which increases group counts and stabilizes the means.'
- What this solution (achieved 1.23566) has done: 'Your current baseline is still far above the target (lower is better), so we should reduce error by adding one more strong, “still-just-aggregation” signal while keeping the same groupby-mean-with-backoff core logic. The smallest high-impact addition is to merge in `mulliken_charges.csv` (per-atom) and `magnetic_shielding_tensors.csv` (per-atom) and then bin simple pairwise aggregates (sum/diff) similarly to `dist_bin`, letting the model learn different means for chemically different environments. We keep your exact backoff chain semantics (fine → type+dist → type+atompair → type → global) but make the “fine” key slightly richer; if any new fields are missing, the existing fallbacks handle it safely. This should move the score downward toward the target without changing the fundamental approach or introducing training loops/models.'
- What this solution (achieved 1.51795) has done: 'Your current score (1.23566, lower-is-better) is still far above the target (-2.087), so we should materially reduce MAE while keeping the same “pure groupby mean with backoff” core logic. The biggest remaining weakness is that we’re using very fine keys that can overfit noisy bins, and we aren’t leveraging the strongest single physical signal in this competition: inverse-distance scaling, which is still compatible with aggregation semantics. I keep your exact merge/feature pipeline, but (1) add a stable per-`type` linear calibration of the mean predictions against `1/(dist+eps)` computed on train (closed-form least squares, no training loop), and (2) add a safer backoff level using `(type, atom_0, atom_1, dist_bin)` between “fine” and “type+dist”. This should reduce error substantially without changing the approach beyond deterministic post-calibration.'
- What this solution (achieved 1.44823) has done: 'Your current score is far worse than the target (lower-is-better), so the smallest “same-core-logic” improvement is to make the calibration and aggregation less noisy rather than adding new modeling. I keep your exact groupby-mean-with-backoff predictor, but (1) compute train-time `_base_pred` using the same fallback chain as test (so the per-type linear calibration is fit on the true deployed baseline, not on fine-only keys), and (2) fit the per-type linear calibration on a per-molecule aggregated dataset to better match the competition’s molecule-split evaluation (reduces leakage-like overfitting from having many rows per molecule). Finally, I make the calibration robust by clipping extreme `inv_dist` values and adding a tiny ridge term in the closed-form solve (still deterministic, no training loop) to stabilize coefficients and typically reduce MAE. These are minimal changes that preserve your approach and should move the score downward toward the target.'
- What this solution (achieved 2.07752) has done: 'Your current score is much worse than the target (lower-is-better), so we should improve it with the smallest change that reduces noise/overfitting while keeping the exact same “groupby means + backoff + per-type linear calibration” core logic. The biggest issue is that the “fine” key uses four binned auxiliary features with very small bin widths, creating an enormous number of sparse groups; this makes the mapped means unstable and hurts generalization on the molecule-split test. I keep all features and the same fallback chain, but (1) coarsen the bin widths for Mulliken/shield features (fewer groups, more samples per group) and (2) switch `dist_bin` to use flooring instead of rounding (more stable, monotone binning), which typically reduces MAE without changing the approach. Everything else (merges, backoffs, calibration, submission alignment) stays the same and it still write `submission.csv`.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

DATA_DIR = Path("/kaggle/data/champs-scalar-coupling")

print("DATA_DIR exists:", DATA_DIR.exists())
print("Files (first 20):", sorted([p.name for p in DATA_DIR.iterdir()])[:20])



## === cell 1
import pandas as pd
import numpy as np

train_path = DATA_DIR / "train.csv"
test_path = DATA_DIR / "test.csv"
sample_path = DATA_DIR / "sample_submission.csv"
structures_path = DATA_DIR / "structures.csv"
mulliken_path = DATA_DIR / "mulliken_charges.csv"
shield_path = DATA_DIR / "magnetic_shielding_tensors.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

print("train:", train.shape, "test:", test.shape, "sample:", sample.shape)
print("train cols:", train.columns.tolist())
print("test cols:", test.columns.tolist())
print("sample cols:", sample.columns.tolist())

structures = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "object",
        "atom_index": "int32",
        "atom": "object",
        "x": "float32",
        "y": "float32",
        "z": "float32",
    },
)
print("structures:", structures.shape)

mulliken = pd.read_csv(
    mulliken_path,
    usecols=["molecule_name", "atom_index", "mulliken_charge"],
    dtype={
        "molecule_name": "object",
        "atom_index": "int32",
        "mulliken_charge": "float32",
    },
)
print("mulliken:", mulliken.shape)

shield = pd.read_csv(
    shield_path,
    usecols=["molecule_name", "atom_index", "XX", "YY", "ZZ"],
    dtype={
        "molecule_name": "object",
        "atom_index": "int32",
        "XX": "float32",
        "YY": "float32",
        "ZZ": "float32",
    },
)
shield["shield_trace"] = (shield["XX"] + shield["YY"] + shield["ZZ"]).astype("float32")
shield = shield[["molecule_name", "atom_index", "shield_trace"]]
print("shield (reduced):", shield.shape)



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

m0 = mulliken.rename(columns={"atom_index": "atom_index_0", "mulliken_charge": "q0"})
m1 = mulliken.rename(columns={"atom_index": "atom_index_1", "mulliken_charge": "q1"})

sh0 = shield.rename(columns={"atom_index": "atom_index_0", "shield_trace": "sh0"})
sh1 = shield.rename(columns={"atom_index": "atom_index_1", "shield_trace": "sh1"})

train_f = (
    train.merge(
        s0, on=["molecule_name", "atom_index_0"], how="left", validate="many_to_one"
    )
    .merge(s1, on=["molecule_name", "atom_index_1"], how="left", validate="many_to_one")
    .merge(m0, on=["molecule_name", "atom_index_0"], how="left", validate="many_to_one")
    .merge(m1, on=["molecule_name", "atom_index_1"], how="left", validate="many_to_one")
    .merge(
        sh0, on=["molecule_name", "atom_index_0"], how="left", validate="many_to_one"
    )
    .merge(
        sh1, on=["molecule_name", "atom_index_1"], how="left", validate="many_to_one"
    )
)

test_f = (
    test.merge(
        s0, on=["molecule_name", "atom_index_0"], how="left", validate="many_to_one"
    )
    .merge(s1, on=["molecule_name", "atom_index_1"], how="left", validate="many_to_one")
    .merge(m0, on=["molecule_name", "atom_index_0"], how="left", validate="many_to_one")
    .merge(m1, on=["molecule_name", "atom_index_1"], how="left", validate="many_to_one")
    .merge(
        sh0, on=["molecule_name", "atom_index_0"], how="left", validate="many_to_one"
    )
    .merge(
        sh1, on=["molecule_name", "atom_index_1"], how="left", validate="many_to_one"
    )
)

for df in (train_f, test_f):
    df["atom_0"] = df["atom_0"].astype("object")
    df["atom_1"] = df["atom_1"].astype("object")

for df in (train_f, test_f):
    a0 = df["atom_0"].astype("string")
    a1 = df["atom_1"].astype("string")
    swap = (a0 > a1) | ((a0 == a1) & (df["atom_index_0"] > df["atom_index_1"]))
    if swap.any():
        df.loc[swap, ["atom_index_0", "atom_index_1"]] = df.loc[
            swap, ["atom_index_1", "atom_index_0"]
        ].to_numpy()
        df.loc[swap, ["atom_0", "atom_1"]] = df.loc[
            swap, ["atom_1", "atom_0"]
        ].to_numpy()
        df.loc[swap, ["x0", "y0", "z0", "x1", "y1", "z1"]] = df.loc[
            swap, ["x1", "y1", "z1", "x0", "y0", "z0"]
        ].to_numpy()
        df.loc[swap, ["q0", "q1"]] = df.loc[swap, ["q1", "q0"]].to_numpy()
        df.loc[swap, ["sh0", "sh1"]] = df.loc[swap, ["sh1", "sh0"]].to_numpy()

for df in (train_f, test_f):
    dx = df["x0"] - df["x1"]
    dy = df["y0"] - df["y1"]
    dz = df["z0"] - df["z1"]
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype("float64")

BIN_WIDTH = 0.02
MAX_DIST = 5.0
for df in (train_f, test_f):
    dist = df["dist"].where(np.isfinite(df["dist"]), np.nan)
    dist = dist.clip(lower=0.0, upper=MAX_DIST)
    df["dist"] = dist
    dist_scaled = dist / BIN_WIDTH
    df["dist_bin"] = np.floor(dist_scaled).astype("Int32")

Q_BIN = 0.05
SH_BIN = 2.0

for df in (train_f, test_f):
    q0 = df["q0"].astype("float64")
    q1 = df["q1"].astype("float64")
    sh0v = df["sh0"].astype("float64")
    sh1v = df["sh1"].astype("float64")

    q_sum = q0 + q1
    q_absdiff = (q0 - q1).abs()
    sh_sum = sh0v + sh1v
    sh_absdiff = (sh0v - sh1v).abs()

    df["q_sum_bin"] = np.floor(q_sum / Q_BIN).astype("Int32")
    df["q_absdiff_bin"] = np.floor(q_absdiff / Q_BIN).astype("Int32")
    df["sh_sum_bin"] = np.floor(sh_sum / SH_BIN).astype("Int32")
    df["sh_absdiff_bin"] = np.floor(sh_absdiff / SH_BIN).astype("Int32")

g_cols_fine = [
    "type",
    "atom_0",
    "atom_1",
    "dist_bin",
    "q_sum_bin",
    "q_absdiff_bin",
    "sh_sum_bin",
    "sh_absdiff_bin",
]
fine_mean = train_f.groupby(g_cols_fine, dropna=False)[
    "scalar_coupling_constant"
].mean()

g_cols_atompair_dist = ["type", "atom_0", "atom_1", "dist_bin"]
atompair_dist_mean = train_f.groupby(g_cols_atompair_dist, dropna=False)[
    "scalar_coupling_constant"
].mean()

g_cols_type_dist = ["type", "dist_bin"]
type_dist_mean = train_f.groupby(g_cols_type_dist, dropna=False)[
    "scalar_coupling_constant"
].mean()

g_cols_no_dist = ["type", "atom_0", "atom_1"]
atompair_mean = train_f.groupby(g_cols_no_dist, dropna=False)[
    "scalar_coupling_constant"
].mean()

type_mean = train_f.groupby("type")["scalar_coupling_constant"].mean()
global_mean = float(train_f["scalar_coupling_constant"].mean())

test_key_fine = list(
    zip(
        test_f["type"],
        test_f["atom_0"],
        test_f["atom_1"],
        test_f["dist_bin"],
        test_f["q_sum_bin"],
        test_f["q_absdiff_bin"],
        test_f["sh_sum_bin"],
        test_f["sh_absdiff_bin"],
    )
)
pred_fine = pd.Series(test_key_fine, index=test_f.index).map(fine_mean)

test_key_atompair_dist = list(
    zip(test_f["type"], test_f["atom_0"], test_f["atom_1"], test_f["dist_bin"])
)
pred_atompair_dist = pd.Series(test_key_atompair_dist, index=test_f.index).map(
    atompair_dist_mean
)

test_key_type_dist = list(zip(test_f["type"], test_f["dist_bin"]))
pred_type_dist = pd.Series(test_key_type_dist, index=test_f.index).map(type_dist_mean)

test_key_atompair = list(zip(test_f["type"], test_f["atom_0"], test_f["atom_1"]))
pred_atompair = pd.Series(test_key_atompair, index=test_f.index).map(atompair_mean)

test_pred = pred_fine
test_pred = test_pred.fillna(pred_atompair_dist)
test_pred = test_pred.fillna(pred_type_dist)
test_pred = test_pred.fillna(pred_atompair)
test_pred = test_pred.fillna(test_f["type"].map(type_mean))
test_pred = test_pred.fillna(global_mean).astype("float64")

train_key_fine = list(
    zip(
        train_f["type"],
        train_f["atom_0"],
        train_f["atom_1"],
        train_f["dist_bin"],
        train_f["q_sum_bin"],
        train_f["q_absdiff_bin"],
        train_f["sh_sum_bin"],
        train_f["sh_absdiff_bin"],
    )
)
train_base = pd.Series(train_key_fine, index=train_f.index).map(fine_mean)

train_key_atompair_dist = list(
    zip(train_f["type"], train_f["atom_0"], train_f["atom_1"], train_f["dist_bin"])
)
train_base = train_base.fillna(
    pd.Series(train_key_atompair_dist, index=train_f.index).map(atompair_dist_mean)
)

train_key_type_dist = list(zip(train_f["type"], train_f["dist_bin"]))
train_base = train_base.fillna(
    pd.Series(train_key_type_dist, index=train_f.index).map(type_dist_mean)
)

train_key_atompair = list(zip(train_f["type"], train_f["atom_0"], train_f["atom_1"]))
train_base = train_base.fillna(
    pd.Series(train_key_atompair, index=train_f.index).map(atompair_mean)
)

train_base = train_base.fillna(train_f["type"].map(type_mean))
train_base = train_base.fillna(global_mean).astype("float64")
train_f["_base_pred"] = train_base

EPS = 1e-3
for df in (train_f, test_f):
    inv = (1.0 / (df["dist"].astype("float64") + EPS)).replace(
        [np.inf, -np.inf], np.nan
    )
    df["_inv_dist"] = inv.clip(lower=0.0, upper=5.0)

agg = (
    train_f[
        ["type", "molecule_name", "scalar_coupling_constant", "_base_pred", "_inv_dist"]
    ]
    .groupby(["type", "molecule_name"], sort=False, dropna=False)
    .mean(numeric_only=True)
    .reset_index()
)

RIDGE = 1e-3

coef = {}
for t, g in agg.groupby("type", sort=False):
    y = g["scalar_coupling_constant"].to_numpy(dtype=np.float64)
    x1 = g["_base_pred"].to_numpy(dtype=np.float64)
    x2 = g["_inv_dist"].to_numpy(dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
    if m.sum() < 100:
        coef[t] = (1.0, 0.0, 0.0)
        continue
    X = np.vstack([x1[m], x2[m], np.ones(m.sum(), dtype=np.float64)]).T
    XtX = X.T @ X
    XtX += RIDGE * np.eye(3, dtype=np.float64)
    Xty = X.T @ y[m]
    beta = np.linalg.solve(XtX, Xty)
    coef[t] = (float(beta[0]), float(beta[1]), float(beta[2]))

a = test_f["type"].map({k: v[0] for k, v in coef.items()}).astype("float64")
b = test_f["type"].map({k: v[1] for k, v in coef.items()}).astype("float64")
c = test_f["type"].map({k: v[2] for k, v in coef.items()}).astype("float64")

inv_d = test_f["_inv_dist"].astype("float64").fillna(0.0)

test_pred_cal = (a * test_pred + b * inv_d + c).astype("float64")

sub_pred = pd.DataFrame(
    {"id": test_f["id"].values, "scalar_coupling_constant": test_pred_cal.values}
)

sub = sample[["id"]].merge(sub_pred, on="id", how="left", validate="one_to_one")
sub["scalar_coupling_constant"] = (
    sub["scalar_coupling_constant"].fillna(global_mean).astype("float64")
)

print(sub.head())
print("submission shape:", sub.shape)
print("missing preds:", sub["scalar_coupling_constant"].isna().sum())

out_path = Path("submission.csv")
sub.to_csv(out_path, index=False)
print("Wrote:", out_path.resolve())



## === cell 3
sub.head(20)
