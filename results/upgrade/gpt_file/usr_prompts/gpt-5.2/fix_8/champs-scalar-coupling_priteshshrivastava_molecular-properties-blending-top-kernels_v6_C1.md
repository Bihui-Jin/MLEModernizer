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

-1.624688611396132

# 6. Current score

1.18497

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.18497) has done: 'The original notebook fails because it depends on external Kaggle datasets (`champs-blending-tutorial`, `otherkernelsadded`) that are not present in your environment. I replace that broken blending input with an in-environment, minimal baseline that uses only the provided competition files and still writes a valid `submission.csv` with the correct columns. To keep core logic simple and stable, the fix builds a lightweight, leakage-free prediction using per-`type` statistics learned from `train.csv` and applies them to `test.csv`. This run end-to-end within the time limit and yield a valid (non-empty, correctly aligned) submission.'
- What this solution (achieved 1.34025) has done: 'The crash happens because some `test["dist"]` values are still non-finite (NaN/inf) after the merges, so converting the binned distances to `int32` fails. I make the distance computation robust by forcing `dist` to be finite (replace inf with NaN) and then filling NaNs using the *training* median for both train/test (score-neutral and avoids distribution shift). I also use pandas’ nullable integer (`Int32`) for `dist_bin` and fill any remaining NaNs before building the MultiIndex, ensuring the pipeline always completes and writes a valid `submission.csv`. Core logic (type+distance-bin medians with fallbacks) stays the same.'
- What this solution (achieved 1.3458) has done: 'Your current model is a type+distance-bin median baseline; the easiest legitimate way to move the score downward (better, since lower is better) without changing the core approach is to make the distance feature slightly more informative while keeping the same aggregation logic. I keep the same merges and “median by (type, dist_bin) with fallbacks” pipeline, but I (1) compute **both** Euclidean distance and squared distance and (2) use a **combined bin key** (still just a binning trick) that reduces collisions where different geometries land in the same bin. This typically improves MAE for this competition while preserving identical semantics (groupby-median lookup + fallback). The output submission format, row alignment, and id sorting stay unchanged.'
- What this solution (achieved 1.3458) has done: 'Your current score (1.3458; lower is better) is still far from the target (-1.6247), so we should legitimately improve accuracy while keeping the same “median lookup by binned geometry with fallbacks” core logic. The smallest high-impact change is to make the geometric key slightly richer without changing the modeling approach: add a second physically-relevant distance feature (`1/dist`) and fold its bin into the existing combined bin key, so the same groupby-median can separate near/far pairs better. To keep runtime under control, we use a coarse bin width for `inv_dist` and keep your existing `dist`/`dist2` bins and fallbacks unchanged. This should reduce MAE (and thus log-MAE) versus the current setup while preserving the same pipeline structure and producing a valid `submission.csv`.'
- What this solution (achieved 1.18497) has done: 'Your current pipeline is a “median lookup by (type, binned-geometry) with fallbacks”, so the smallest legitimate way to improve (lower) log-MAE is to make the geometry key slightly more informative while keeping the exact same aggregation logic. I add one extra geometric signal derived from the same coordinates—`log1p(dist)`—and fold its bin into your existing combined bin key, so different regimes of near/far pairs separate better without changing the model family. I keep your existing `dist/dist2/inv_dist` features, binning approach, and fallback chain untouched, only extending the combo key. This should move the score downward (better) while staying stable and still producing a valid `submission.csv`.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower-is-better) is still far from the target (-1.6247), so we should improve accuracy while keeping your exact “median lookup by (type, binned-geometry) with fallbacks” core logic. The smallest high-impact improvement that preserves the same semantics is to (1) add atom-type information (the `atom` labels) for both atoms from `structures.csv`, and (2) include a compact atom-pair key in the groupby key; this typically reduces MAE a lot because coupling depends strongly on the involved elements. To avoid destabilizing the binning scheme, we keep all your existing distance-derived bins unchanged and simply extend the grouping key to `["type","atom_pair","dist_bin_combo"]`, with the same fallback chain (`type` median then global median). This remains leakage-free, runs fast (only two extra merges/columns), and still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

BASE_PATH = "/kaggle/data/champs-scalar-coupling"

print("Listing base path:", BASE_PATH)
print(os.listdir(BASE_PATH))



## === cell 1
train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
sample_path = os.path.join(BASE_PATH, "sample_submission.csv")
structures_path = os.path.join(BASE_PATH, "structures.csv")

train = pd.read_csv(
    train_path,
    usecols=[
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
)
test = pd.read_csv(
    test_path, usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]
)
sample = pd.read_csv(sample_path, usecols=["id", "scalar_coupling_constant"])

structures = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
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

train = train.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
train = train.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

test = test.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
test = test.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

eps = 1e-12

for df in (train, test):
    dx = df["x0"] - df["x1"]
    dy = df["y0"] - df["y1"]
    dz = df["z0"] - df["z1"]

    dist2 = (dx * dx + dy * dy + dz * dz).astype("float64")
    dist = np.sqrt(dist2).astype("float64")

    df["dist2"] = pd.Series(dist2, index=df.index).replace([np.inf, -np.inf], np.nan)
    df["dist"] = pd.Series(dist, index=df.index).replace([np.inf, -np.inf], np.nan)

    inv_dist = 1.0 / (dist + eps)
    df["inv_dist"] = pd.Series(inv_dist, index=df.index).replace(
        [np.inf, -np.inf], np.nan
    )

    log_dist = np.log1p(dist).astype("float64")
    df["log_dist"] = pd.Series(log_dist, index=df.index).replace(
        [np.inf, -np.inf], np.nan
    )

train_dist_median = float(train["dist"].median(skipna=True))
train_dist2_median = float(train["dist2"].median(skipna=True))
train_inv_dist_median = float(train["inv_dist"].median(skipna=True))
train_log_dist_median = float(train["log_dist"].median(skipna=True))

train["dist"] = train["dist"].fillna(train_dist_median)
test["dist"] = test["dist"].fillna(train_dist_median)

train["dist2"] = train["dist2"].fillna(train_dist2_median)
test["dist2"] = test["dist2"].fillna(train_dist2_median)

train["inv_dist"] = train["inv_dist"].fillna(train_inv_dist_median)
test["inv_dist"] = test["inv_dist"].fillna(train_inv_dist_median)

train["log_dist"] = train["log_dist"].fillna(train_log_dist_median)
test["log_dist"] = test["log_dist"].fillna(train_log_dist_median)

bin_width = 0.05
bin_width2 = 0.10  # keep as-is
inv_bin_width = 0.5
log_bin_width = 0.03

train["dist_bin"] = np.floor(train["dist"] / bin_width).astype("Int32")
test["dist_bin"] = np.floor(test["dist"] / bin_width).astype("Int32")

train["dist2_bin"] = np.floor(train["dist2"] / bin_width2).astype("Int32")
test["dist2_bin"] = np.floor(test["dist2"] / bin_width2).astype("Int32")

train["invdist_bin"] = np.floor(train["inv_dist"] / inv_bin_width).astype("Int32")
test["invdist_bin"] = np.floor(test["inv_dist"] / inv_bin_width).astype("Int32")

train["logdist_bin"] = np.floor(train["log_dist"] / log_bin_width).astype("Int32")
test["logdist_bin"] = np.floor(test["log_dist"] / log_bin_width).astype("Int32")

train["dist_bin"] = train["dist_bin"].fillna(0).astype("int32")
test["dist_bin"] = test["dist_bin"].fillna(0).astype("int32")
train["dist2_bin"] = train["dist2_bin"].fillna(0).astype("int32")
test["dist2_bin"] = test["dist2_bin"].fillna(0).astype("int32")
train["invdist_bin"] = train["invdist_bin"].fillna(0).astype("int32")
test["invdist_bin"] = test["invdist_bin"].fillna(0).astype("int32")
train["logdist_bin"] = train["logdist_bin"].fillna(0).astype("int32")
test["logdist_bin"] = test["logdist_bin"].fillna(0).astype("int32")

BASE_DIST2 = 20000
BASE_INVD = 2000
BASE_LOGD = 2000

train["dist_bin_combo"] = (
    (
        (
            (train["dist_bin"] * BASE_DIST2 + train["dist2_bin"]) * BASE_INVD
            + train["invdist_bin"]
        )
        * BASE_LOGD
    )
    + train["logdist_bin"]
).astype("int64")

test["dist_bin_combo"] = (
    (
        (
            (test["dist_bin"] * BASE_DIST2 + test["dist2_bin"]) * BASE_INVD
            + test["invdist_bin"]
        )
        * BASE_LOGD
    )
    + test["logdist_bin"]
).astype("int64")

for df in (train, test):
    df["atom_0"] = df["atom_0"].fillna("X").astype("string")
    df["atom_1"] = df["atom_1"].fillna("X").astype("string")
    df["atom_pair"] = (df["atom_0"] + "_" + df["atom_1"]).astype("string")

type_pair_dist_median = train.groupby(["type", "atom_pair", "dist_bin_combo"])[
    "scalar_coupling_constant"
].median()
type_median = train.groupby("type")["scalar_coupling_constant"].median()
global_median = float(train["scalar_coupling_constant"].median())

test_key = pd.MultiIndex.from_frame(test[["type", "atom_pair", "dist_bin_combo"]])
pred = pd.Series(
    type_pair_dist_median.reindex(test_key).to_numpy(),
    index=test.index,
    dtype="float64",
)

mask_na = pred.isna()
if mask_na.any():
    pred.loc[mask_na] = (
        test.loc[mask_na, "type"].map(type_median).astype("float64").to_numpy()
    )

pred = pred.fillna(global_median).astype("float64")

submission = pd.DataFrame(
    {"id": test["id"].values, "scalar_coupling_constant": pred.values}
)

assert (
    submission.shape[0] == sample.shape[0]
), f"Row count mismatch: {submission.shape[0]} vs {sample.shape[0]}"
assert submission["id"].is_unique, "Submission ids are not unique"
assert set(submission.columns) == {"id", "scalar_coupling_constant"}

submission = submission.sort_values("id").reset_index(drop=True)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print(submission.describe(include="all"))
