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

-1.5209593019916507

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 2.14717) has done: 'The crash comes from trying to read two external kernel submission files that don’t exist in your environment, so `sub1/sub2` never get created and all later cells fail. To keep the core “blend two submissions” logic intact while making it runnable end-to-end, I load the competition’s `sample_submission.csv` and build two baseline prediction vectors from the provided `train.csv` (a global mean and a per-`type` mean), then blend them with the same 0.6/0.4 weights. This produces a valid `stackers_blend.csv` with the required columns and should score meaningfully better than all-zeros without changing the overall approach (still a simple blend). I also make the paths robust to the given `/kaggle/data/...` layout and replace the deprecated `sns.distplot` call so the final diagnostic plot doesn’t error.'
- What this solution (achieved 1.23566) has done: 'I fix the crash in `add_dist_bin` by handling missing/invalid distances before converting to integer bins (the current NaNs come from unmatched structure merges). I also add lightweight checks after merging structures to quantify missing coordinates and fill missing predictions safely using the existing fallback chain (type mean → global mean), preserving your “blend two predictors” core logic. Finally, I ensure the blend cell always produces a correctly shaped submission aligned to `sample_submission.csv` and writes `stackers_blend.csv`. These changes are score-neutral to mildly positive (fewer NaN-induced failures; better defined bins), but primarily ensure an end-to-end runnable pipeline.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is far worse than the target (-1.52096), so we should improve while keeping the same “blend two simple predictors” core logic. The biggest gain with minimal risk is to compute the distance-bin mean on the correct directed pair key: the dataset’s `atom_index_0/1` ordering matters, but your `type_dist_mean` currently ignores direction, mixing different physical pairs and hurting MAE. I add an `atom_order` flag (whether `atom_index_0 < atom_index_1`) into both train/test features and into the groupby key for `type_dist_mean`, keeping everything else (features, binning, fallback chain, blending weights, submission writing) the same. This is a small, semantically consistent change that should move the score substantially toward the target without changing the overall approach.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import seaborn as sns

BASE_INPUT_CANDIDATES = [
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "../input/champs-scalar-coupling",
]

BASE_INPUT = None
for p in BASE_INPUT_CANDIDATES:
    if os.path.exists(p):
        BASE_INPUT = p
        break

if BASE_INPUT is None:
    for root in ["/kaggle/input", "/kaggle/data", "../input", "."]:
        if os.path.isdir(root):
            for dirpath, dirnames, filenames in os.walk(root):
                if "train.csv" in filenames and "test.csv" in filenames:
                    BASE_INPUT = dirpath
                    break
            if BASE_INPUT is not None:
                break

if BASE_INPUT is None:
    raise FileNotFoundError("Could not locate champs-scalar-coupling dataset folder.")

print("Using BASE_INPUT:", BASE_INPUT)
print("Files in BASE_INPUT (first 20):", sorted(os.listdir(BASE_INPUT))[:20])



## === cell 1
print("Listing parent directory of BASE_INPUT:", os.path.dirname(BASE_INPUT))
print(sorted(os.listdir(os.path.dirname(BASE_INPUT)))[:50])



## === cell 2
sample = pd.read_csv(os.path.join(BASE_INPUT, "sample_submission.csv"))
test = pd.read_csv(os.path.join(BASE_INPUT, "test.csv"))
train = pd.read_csv(os.path.join(BASE_INPUT, "train.csv"))

structures = pd.read_csv(
    os.path.join(BASE_INPUT, "structures.csv"),
    usecols=["molecule_name", "atom_index", "x", "y", "z"],
)

s0 = structures.rename(
    columns={"atom_index": "atom_index_0", "x": "x0", "y": "y0", "z": "z0"}
)
s1 = structures.rename(
    columns={"atom_index": "atom_index_1", "x": "x1", "y": "y1", "z": "z1"}
)

train_feat = train.merge(s0, on=["molecule_name", "atom_index_0"], how="left").merge(
    s1, on=["molecule_name", "atom_index_1"], how="left"
)
test_feat = test.merge(s0, on=["molecule_name", "atom_index_0"], how="left").merge(
    s1, on=["molecule_name", "atom_index_1"], how="left"
)


def add_dist(df: pd.DataFrame) -> pd.DataFrame:
    dx = (df["x0"] - df["x1"]).astype(np.float64)
    dy = (df["y0"] - df["y1"]).astype(np.float64)
    dz = (df["z0"] - df["z1"]).astype(np.float64)
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)
    return df


train_feat = add_dist(train_feat)
test_feat = add_dist(test_feat)

global_mean = float(train_feat["scalar_coupling_constant"].mean())
type_mean = train_feat.groupby("type")["scalar_coupling_constant"].mean()

BIN_WIDTH = 0.05
MAX_DIST = 6.0
MISSING_BIN = -1  # sentinel for missing distances


def add_dist_bin(df: pd.DataFrame) -> pd.DataFrame:
    d = df["dist"].astype(np.float64)
    is_finite = np.isfinite(d.to_numpy())
    bins = np.full(shape=len(df), fill_value=MISSING_BIN, dtype=np.int32)
    if is_finite.any():
        d_clip = np.clip(d.to_numpy()[is_finite], 0.0, MAX_DIST)
        bins[is_finite] = np.floor(d_clip / BIN_WIDTH).astype(np.int32)
    df["dist_bin"] = bins
    return df


train_feat = add_dist_bin(train_feat)
test_feat = add_dist_bin(test_feat)

train_feat["atom_order"] = (
    train_feat["atom_index_0"] < train_feat["atom_index_1"]
).astype(np.int8)
test_feat["atom_order"] = (
    test_feat["atom_index_0"] < test_feat["atom_index_1"]
).astype(np.int8)

missing_coords_test = int(
    test_feat[["x0", "y0", "z0", "x1", "y1", "z1"]].isna().any(axis=1).sum()
)
missing_dist_test = int(test_feat["dist"].isna().sum())
print("test rows:", len(test_feat))
print("test rows with any missing coords:", missing_coords_test)
print("test dist missing:", missing_dist_test)

type_dist_mean = train_feat.groupby(["type", "atom_order", "dist_bin"])[
    "scalar_coupling_constant"
].mean()

sub2 = test_feat[["id", "type", "atom_order", "dist_bin"]].copy()
sub2["scalar_coupling_constant"] = (
    sub2.set_index(["type", "atom_order", "dist_bin"])
    .index.map(type_dist_mean)
    .astype(np.float64)
)
sub2["scalar_coupling_constant"] = (
    sub2["scalar_coupling_constant"]
    .fillna(sub2["type"].map(type_mean))
    .fillna(global_mean)
)

sub2 = sample[["id"]].merge(
    sub2[["id", "scalar_coupling_constant"]], on="id", how="left"
)
sub2["scalar_coupling_constant"] = (
    sub2["scalar_coupling_constant"].fillna(global_mean).astype(np.float64)
)

sub1 = test[["id", "type"]].copy()
sub1["scalar_coupling_constant"] = sub1["type"].map(type_mean).astype(np.float64)
sub1["scalar_coupling_constant"] = sub1["scalar_coupling_constant"].fillna(global_mean)
sub1 = sample[["id"]].merge(
    sub1[["id", "scalar_coupling_constant"]], on="id", how="left"
)
sub1["scalar_coupling_constant"] = (
    sub1["scalar_coupling_constant"].fillna(global_mean).astype(np.float64)
)

print(
    "sub1 shape:", sub1.shape, "sub2 shape:", sub2.shape, "sample shape:", sample.shape
)
print("global_mean:", global_mean)
print("type_mean entries:", len(type_mean))
print("type_dist_mean entries:", len(type_dist_mean))



## === cell 3
print(sub1["scalar_coupling_constant"].describe())



## === cell 4
print(sub2["scalar_coupling_constant"].describe())



## === cell 5
out = sample.merge(sub2, on="id", how="left", suffixes=("", "_sub2"))
out = out.merge(sub1, on="id", how="left", suffixes=("", "_sub1"))

out["scalar_coupling_constant"] = 0.85 * out["scalar_coupling_constant_sub2"].astype(
    np.float64
) + 0.15 * out["scalar_coupling_constant_sub1"].astype(np.float64)

out["scalar_coupling_constant"] = (
    out["scalar_coupling_constant"].fillna(global_mean).astype(np.float64)
)

out = out[["id", "scalar_coupling_constant"]]
out.to_csv("stackers_blend.csv", index=False)

print("Wrote submission:", "stackers_blend.csv")
print(out.head())
print("Submission shape:", out.shape)
print("Any NaNs in submission:", int(out["scalar_coupling_constant"].isna().sum()))



## === cell 6
ax = sns.histplot(out["scalar_coupling_constant"], bins=60, kde=True)
ax.set_title("Blended scalar_coupling_constant distribution")
