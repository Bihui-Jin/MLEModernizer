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

-1.6838160788283738

# 6. Current score

2.28021

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.18497) has done: 'Your notebook fails because it tries to read out-of-environment blend files (e.g., `../input/champs-blending-tutorial/1.csv`) that do not exist in the provided filesystem. To make it run end-to-end and generate a valid `.csv` submission, I replace that external blending step with a minimal, self-contained baseline model that uses only the provided competition data (`train.csv`/`test.csv`) and produces predictions per coupling `type` (median target per type, with a global fallback). This preserves the overall “simple aggregation-based predictor” spirit while removing missing dependencies. The output be written as `my_blend_1.csv` with the required columns and row alignment by `id`.'
- What this solution (achieved 4.11578) has done: 'I fix the Ridge crash by ensuring the numeric distance features never contain NaN/inf (these can appear after merges if any structure rows are missing, or if a distance is zero leading to inf in `inv_dist`). Concretely, I coerce the coordinate columns to numeric, compute `dist/dist2/inv_dist` robustly, then replace any remaining NaN/inf in the numeric feature matrix with safe finite values (medians, with 0 fallback). This keeps the same model/feature logic and training loops, but makes the pipeline run end-to-end and produce a valid `my_blend_1.csv` submission.'
- What this solution (achieved 7.87216) has done: 'Your current score (4.11578, lower-is-better) is far worse than the target (-1.6838), so we should improve while keeping the same overall “Ridge per type on simple structure-distance + atom pair one-hot” core. The biggest likely issue is that the numeric feature scales are extremely ill-conditioned (dist, dist2, inv_dist), which makes Ridge behave poorly without standardization; adding a `StandardScaler` inside a `Pipeline` preserves the same linear model logic but typically yields a large MAE reduction on this competition. I also make the train/test one-hot alignment lighter and safer by using `OneHotEncoder(handle_unknown="ignore")` in a `ColumnTransformer`, which avoids giant dense matrices and keeps consistent columns without depending on concatenation order. Everything else (per-type training loop, fallbacks, submission alignment) stays the same.'
- What this solution (achieved 7.36455) has done: 'Your current score (7.87216, lower-is-better) is far worse than the target (-1.6838), so we should improve while keeping the same per-type Ridge-on-distance+atom-onehot core. The biggest issue is that the model is missing the most informative “competition-provided” atom/molecule features (mulliken charges, magnetic shielding tensors, dipole moments, potential energy), which can be merged in without changing the training loop or model family. I minimally extend the feature set by joining these files for atom_index_0/1 and molecule_name, add a few simple derived diffs/sums (still linear-friendly), and keep the same preprocessing (StandardScaler + OneHot) and per-type Ridge fitting. This should move the score substantially toward the target while preserving evaluation semantics and producing the same submission format.'
- What this solution (achieved 2.21329) has done: 'We keep your per-type Ridge-on-engineered-features core unchanged, but fix two issues that commonly inflate this metric: (1) the categorical space is missing the coupling `type` itself (even though you train per-type, the shared preprocessing can still benefit from type in the fallback cases and keeps semantics consistent), and (2) Ridge `alpha=1.0` is likely over-regularizing given the expanded auxiliary features; we tune `alpha` minimally via a tiny in-train molecule-split validation per type and pick from a small fixed grid. This preserves the same model family, same loop structure, and same feature sources, but should move the score substantially downward toward the target without changing evaluation semantics. We also add a small, safe clip of extreme predictions per type to the training target’s 0.5–99.5 percentile band, which tends to reduce log-MAE sensitivity to outliers while remaining a purely post-processing calibration step.'
- What this solution (achieved 2.33989) has done: 'Your current score (2.21329, lower-is-better) is still far from the target (-1.6838), so we should improve while keeping the same per-type Ridge + same feature set and training loop. The biggest likely score blocker in your current code is that per-type training is iterating over `test` types and uses a shared `ColumnTransformer` whose one-hot space is fit separately per type; we can make it more stable and accurate by (a) training for every type present in train, then predicting for matching test rows, and (b) adding a very small, metric-aligned target transform (fit Ridge on `log1p(|y|)` with sign restored) which reduces the competition’s log-MAE sensitivity without changing the model family or loops. These are minimal changes: same Ridge, same features, same per-type loop, same preprocessing; just more stable iteration coverage and a monotonic transform/undo around the existing regression. We keep your clipping and fallbacks, and still write `my_blend_1.csv` in the required format.'
- What this solution (achieved 2.27999) has done: 'We keep your per-type Ridge + same feature sources and target transform, but fix two stability issues that can hurt this metric: (1) avoid leaking molecule-specific scaling by fitting the `StandardScaler` only on the train fold (not including the per-type validation fold) during alpha selection via a proper per-type Pipeline fit on the training fold, and (2) make the log-target transform better aligned with MAE by using a robust centering per type (subtract the per-type median before transform and add it back after inverse), which reduces the impact of large offsets without changing the model family. We also include a tiny, safe feature addition that preserves the same “distance + aux + one-hot” logic: add `dist3 = dist^3` and `inv_dist2 = inv_dist^2` (linear-friendly) to improve fit toward the target. All I/O, per-type loop, Ridge model, and submission writing remain the same.'
- What this solution (achieved 2.21915) has done: 'We keep the exact per-type Ridge + same feature set and centered log1p target transform, but fix one metric-critical bug: the inverse transform currently uses `expm1(abs(yt))` which is mathematically inconsistent with `log1p(abs(y))` (it should be `expm1(abs(yt))` but applied to the *magnitude* that came from `log1p(abs(y))`; the current code is correct there—however the real issue is that you’re transforming `y-center` but validating MAE in original space without any per-type scale stabilization, causing alpha selection to be noisy). To move the score downward toward the target with minimal change, we (1) select `alpha` using the competition’s log-MAE style proxy by evaluating MAE in transformed space (still monotonic, preserves semantics) and (2) tighten clipping to 1%–99% per type to reduce extreme outlier impact (a small calibration change). Everything else (data sources, merges, model family, per-type loop, submission writing) stays the same and it still produce `my_blend_1.csv`.'
- What this solution (achieved 2.21928) has done: 'Your score is much worse than the target (lower-is-better), so we should improve it with minimal, metric-aligned changes while keeping the same per-type Ridge + same feature sources and preprocessing. The largest likely error source is the current validation/alpha selection using MAE in transformed space, which is only a proxy and can pick suboptimal alphas for the actual log-MAE-by-type metric. I switch alpha selection to use a per-type, original-space MAE (after inverse transform) on the validation fold, and also compute per-type clipping bounds from the same training fold used for fitting (avoids a tiny bit of fold contamination and makes clipping more stable). Everything else (data merges, features, Ridge, per-type loop, output file/path/format) stays the same and still writes `my_blend_1.csv`.'
- What this solution (achieved 2.28021) has done: 'Your current score (2.21928, lower-is-better) is far worse than the target (-1.6838), so we should improve with minimal, metric-aligned changes while keeping the exact per-type Ridge + preprocessing + feature-merging core. The biggest practical issue is that alpha selection is using plain MAE, while the competition scores log(MAE) averaged by type; selecting alpha by a log-MAE proxy per type (still evaluated in original target space after inverse-transform) should pick better regularization without changing the model family or training loop. Second, we should clip predictions using the training-fold distribution (as you already do) but make it slightly less aggressive (0.005–0.995) to reduce bias while still protecting against outliers that hurt log-MAE. Everything else (features, target transform, per-type fitting, submission writing/path) remains unchanged and it still write `my_blend_1.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/data/champs-scalar-coupling"

print("Listing /kaggle/data:")
print(os.listdir("/kaggle/data")[:20])
print("\nListing DATA_DIR:")
print(os.listdir(DATA_DIR)[:20])



## === cell 1
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _hash_split_mask(values: pd.Series, frac: float = 0.1, seed: int = 0) -> np.ndarray:
    v = values.astype("string").fillna("NA").to_numpy()
    codes, _ = pd.factorize(v, sort=False)
    x = (codes.astype(np.uint64) + np.uint64(seed)) * np.uint64(1103515245) + np.uint64(
        12345
    )
    u = (x % np.uint64(10_000_000)).astype(np.float64) / 10_000_000.0
    return u < float(frac)


def _y_transform_centered(y: np.ndarray, center: float) -> np.ndarray:
    y = y.astype(np.float64) - float(center)
    return np.sign(y) * np.log1p(np.abs(y))


def _y_inverse_centered(yt: np.ndarray, center: float) -> np.ndarray:
    yt = yt.astype(np.float64)
    y = np.sign(yt) * (np.expm1(np.abs(yt)))
    return y + float(center)


def _log_mae(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return float(np.log(max(mae, eps)))


train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")
mulliken_path = os.path.join(DATA_DIR, "mulliken_charges.csv")
shield_path = os.path.join(DATA_DIR, "magnetic_shielding_tensors.csv")
dipole_path = os.path.join(DATA_DIR, "dipole_moments.csv")
pe_path = os.path.join(DATA_DIR, "potential_energy.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

structures = pd.read_csv(structures_path)
mulliken = pd.read_csv(mulliken_path)
shield = pd.read_csv(shield_path)
dipole = pd.read_csv(dipole_path)
pe = pd.read_csv(pe_path)

for c in ["x", "y", "z"]:
    structures[c] = pd.to_numeric(structures[c], errors="coerce").astype("float32")
structures["atom_index"] = pd.to_numeric(
    structures["atom_index"], errors="coerce"
).astype("int32")

mulliken["atom_index"] = pd.to_numeric(mulliken["atom_index"], errors="coerce").astype(
    "int32"
)
mulliken["mulliken_charge"] = pd.to_numeric(
    mulliken["mulliken_charge"], errors="coerce"
).astype("float32")

shield["atom_index"] = pd.to_numeric(shield["atom_index"], errors="coerce").astype(
    "int32"
)
for c in ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]:
    shield[c] = pd.to_numeric(shield[c], errors="coerce").astype("float32")

for c in ["X", "Y", "Z"]:
    dipole[c] = pd.to_numeric(dipole[c], errors="coerce").astype("float32")
pe["potential_energy"] = pd.to_numeric(pe["potential_energy"], errors="coerce").astype(
    "float32"
)

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

sh0 = shield.rename(
    columns={
        "atom_index": "atom_index_0",
        **{c: f"{c}0" for c in ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]},
    }
)
sh1 = shield.rename(
    columns={
        "atom_index": "atom_index_1",
        **{c: f"{c}1" for c in ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]},
    }
)


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(
        s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    df = df.merge(
        s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )

    x0 = df["x0"].astype("float32")
    y0 = df["y0"].astype("float32")
    z0 = df["z0"].astype("float32")
    x1 = df["x1"].astype("float32")
    y1 = df["y1"].astype("float32")
    z1 = df["z1"].astype("float32")

    dx = (x0 - x1).astype("float32")
    dy = (y0 - y1).astype("float32")
    dz = (z0 - z1).astype("float32")

    dist2 = (dx * dx + dy * dy + dz * dz).astype("float32")
    dist = np.sqrt(dist2.astype("float64")).astype("float32")
    inv_dist = (1.0 / (dist.astype("float64") + 1e-6)).astype("float32")

    df["dist"] = dist
    df["dist2"] = dist2
    df["inv_dist"] = inv_dist

    df["dist3"] = (df["dist"].astype("float64") ** 3).astype("float32")
    df["inv_dist2"] = (df["inv_dist"].astype("float64") ** 2).astype("float32")

    for c in ["dist", "dist2", "inv_dist", "dist3", "inv_dist2"]:
        v = df[c].to_numpy()
        bad = ~np.isfinite(v)
        if bad.any():
            finite = v[np.isfinite(v)]
            fill = float(np.median(finite)) if finite.size else 0.0
            v[bad] = fill
            df[c] = v.astype("float32")

    return df


def add_aux_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(
        m0[["molecule_name", "atom_index_0", "q0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    df = df.merge(
        m1[["molecule_name", "atom_index_1", "q1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )

    df = df.merge(
        sh0[
            ["molecule_name", "atom_index_0"]
            + [f"{c}0" for c in ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]]
        ],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    df = df.merge(
        sh1[
            ["molecule_name", "atom_index_1"]
            + [f"{c}1" for c in ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]]
        ],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )

    df = df.merge(dipole, on="molecule_name", how="left")
    df = df.merge(pe, on="molecule_name", how="left")

    df["q_sum"] = (df["q0"].astype("float32") + df["q1"].astype("float32")).astype(
        "float32"
    )
    df["q_diff"] = (df["q0"].astype("float32") - df["q1"].astype("float32")).astype(
        "float32"
    )

    dip = df[["X", "Y", "Z"]].astype("float32")
    df["dipole_mag"] = np.sqrt((dip * dip).sum(axis=1).astype("float64")).astype(
        "float32"
    )

    df["trace0"] = (df["XX0"] + df["YY0"] + df["ZZ0"]).astype("float32")
    df["trace1"] = (df["XX1"] + df["YY1"] + df["ZZ1"]).astype("float32")
    df["trace_sum"] = (df["trace0"] + df["trace1"]).astype("float32")
    df["trace_diff"] = (df["trace0"] - df["trace1"]).astype("float32")

    return df


train_f = add_aux_features(add_structure_features(train))
test_f = add_aux_features(add_structure_features(test))

type_median = train.groupby("type")["scalar_coupling_constant"].median()
global_median = float(train["scalar_coupling_constant"].median())

cat_cols = ["atom_0", "atom_1", "type"]
for c in cat_cols:
    train_f[c] = train_f[c].fillna("UNK").astype("string")
    test_f[c] = test_f[c].fillna("UNK").astype("string")

num_cols = [
    "dist",
    "dist2",
    "dist3",
    "inv_dist",
    "inv_dist2",
    "q0",
    "q1",
    "q_sum",
    "q_diff",
    "XX0",
    "YX0",
    "ZX0",
    "XY0",
    "YY0",
    "ZY0",
    "XZ0",
    "YZ0",
    "ZZ0",
    "XX1",
    "YX1",
    "ZX1",
    "XY1",
    "YY1",
    "ZY1",
    "XZ1",
    "YZ1",
    "ZZ1",
    "trace0",
    "trace1",
    "trace_sum",
    "trace_diff",
    "X",
    "Y",
    "Z",
    "dipole_mag",
    "potential_energy",
]

for c in num_cols:
    train_f[c] = pd.to_numeric(train_f[c], errors="coerce").astype("float32")
    test_f[c] = pd.to_numeric(test_f[c], errors="coerce").astype("float32")

for df_ in (train_f, test_f):
    for c in num_cols:
        v = df_[c].to_numpy()
        bad = ~np.isfinite(v)
        if bad.any():
            finite = v[np.isfinite(v)]
            fill = float(np.median(finite)) if finite.size else 0.0
            v[bad] = fill
            df_[c] = v.astype("float32")

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(steps=[("scaler", StandardScaler(with_mean=True, with_std=True))]),
            num_cols,
        ),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
    ],
    remainder="drop",
    sparse_threshold=0.3,
)

pred_test = np.empty(len(test_f), dtype=np.float32)
pred_test[:] = np.nan

train_f = train_f.reset_index(drop=True)
test_f = test_f.reset_index(drop=True)
y_train_full = (
    train_f["scalar_coupling_constant"].astype("float32").reset_index(drop=True)
)

alpha_grid = [0.1, 0.3, 1.0, 3.0, 10.0]
clip_bounds = {}

all_types = pd.Index(train_f["type"].unique())
test_types = pd.Index(test_f["type"].unique())
types_to_process = all_types.intersection(test_types)

for t in types_to_process:
    test_idx = test_f.index[test_f["type"] == t].to_numpy(dtype=np.int64)
    trn_mask = (train_f["type"] == t).to_numpy()

    n_trn = int(trn_mask.sum())
    if n_trn < 1000:
        pred_test[test_idx] = float(type_median.get(t, global_median))
        continue

    X_trn_df = train_f.loc[trn_mask, cat_cols + num_cols]
    y_trn = y_train_full.loc[trn_mask]
    X_tst_df = test_f.loc[test_idx, cat_cols + num_cols]

    center_t = float(np.median(y_trn.to_numpy(dtype=np.float64)))
    y_trn_t = _y_transform_centered(y_trn.to_numpy(dtype=np.float64), center=center_t)

    mols = train_f.loc[trn_mask, "molecule_name"]
    is_val = _hash_split_mask(mols, frac=0.08, seed=17)

    if is_val.sum() < 200 or (~is_val).sum() < 200:
        best_alpha = 1.0
        y_clip_source = y_trn.to_numpy(dtype=np.float64)
    else:
        X_trn_in = X_trn_df.loc[~is_val]
        y_trn_in_t = y_trn_t[~is_val]
        X_val = X_trn_df.loc[is_val]
        y_val = y_trn.to_numpy(dtype=np.float64)[is_val]

        best_alpha = 1.0
        best_score = np.inf
        for a in alpha_grid:
            model = Pipeline(
                steps=[
                    ("prep", preprocess),
                    ("ridge", Ridge(alpha=float(a), random_state=0)),
                ]
            )
            model.fit(X_trn_in, y_trn_in_t)

            p_val_t = model.predict(X_val).astype(np.float64)
            p_val = _y_inverse_centered(p_val_t, center=center_t)
            score = _log_mae(y_val, p_val)
            if score < best_score:
                best_score = score
                best_alpha = float(a)

        y_clip_source = y_trn.to_numpy(dtype=np.float64)[~is_val]

    model = Pipeline(
        steps=[
            ("prep", preprocess),
            ("ridge", Ridge(alpha=float(best_alpha), random_state=0)),
        ]
    )
    model.fit(X_trn_df, y_trn_t)
    p_test_t = model.predict(X_tst_df).astype(np.float64)
    p_test = _y_inverse_centered(p_test_t, center=center_t).astype(np.float32)
    pred_test[test_idx] = p_test

    lo = float(np.quantile(y_clip_source, 0.005))
    hi = float(np.quantile(y_clip_source, 0.995))
    clip_bounds[t] = (lo, hi)

pred_series = pd.Series(pred_test, index=test_f.index)

clip_lo = np.full(len(test_f), -np.inf, dtype=np.float64)
clip_hi = np.full(len(test_f), np.inf, dtype=np.float64)
for t, (lo, hi) in clip_bounds.items():
    m = (test_f["type"] == t).to_numpy()
    clip_lo[m] = lo
    clip_hi[m] = hi

pred_series = pred_series.astype("float64")
pred_series = pred_series.clip(lower=pd.Series(clip_lo), upper=pd.Series(clip_hi))

pred_series = (
    pred_series.fillna(test["type"].map(type_median))
    .fillna(global_median)
    .astype("float64")
)

submission = pd.DataFrame(
    {"id": test["id"].values, "scalar_coupling_constant": pred_series.values}
)

if not submission["id"].equals(sample["id"]):
    submission = sample[["id"]].merge(submission, on="id", how="left")
    submission["scalar_coupling_constant"] = submission[
        "scalar_coupling_constant"
    ].fillna(global_median)

out_path = "my_blend_1.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Rows:", len(submission), "Columns:", submission.columns.tolist())
assert out_path.endswith(".csv")
assert submission.shape[0] == sample.shape[0]
assert list(submission.columns) == ["id", "scalar_coupling_constant"]
assert np.isfinite(submission["scalar_coupling_constant"].to_numpy()).all()
