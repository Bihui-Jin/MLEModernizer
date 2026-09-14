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

-1.31958

# 6. Current score

1.25632

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.32398) has done: 'I remove the dependency on external Kaggle dataset submissions (the missing `../input/...` CSVs causing the `FileNotFoundError`) and instead generate a valid prediction directly from the provided competition data. To keep the core “blending” logic intact, I create three simple baseline predictors from the training set: (1) mean by coupling `type`, (2) mean by `(type, atom0, atom1)` element pair, and (3) a light smoothing of (1) toward the global mean. Then I blend these three predictions with the same weights and write a correctly formatted `submission.csv` with `id,scalar_coupling_constant`. This run end-to-end in the given environment and should yield a reasonable baseline score instead of failing before submission generation.'
- What this solution (achieved 1.28121) has done: 'Your current score (1.32398, lower-is-better) is far from the target (-1.31958), so we should legitimately improve the model output while keeping the same “blend of simple aggregate predictors” core logic. The biggest gap in your current approach is that the “pair_mean” uses only element symbols and ignores the actual bond distance/geometry, which is crucial for this competition; we add a minimal distance feature from `structures.csv` and compute an additional grouped mean by `(type, distance_bin)` to capture geometry without changing the overall training approach. Then we blend this new predictor into the existing weighted average with small weight rebalancing (still a linear blend of baselines), and ensure test-time merges are aligned and efficient. This should move the score materially downward toward the target while preserving the same basic pipeline and producing a valid `submission.csv`.'
- What this solution (achieved 1.25632) has done: 'We keep the same “blend of aggregated baselines” core logic, but make the distance-conditioned baseline more informative by conditioning on both `type` and the element pair (`atom_0`,`atom_1`) in addition to a distance bin. This is a minimal extension of your existing geometry feature and usually reduces error substantially because coupling depends strongly on both chemistry and geometry. We also add light smoothing to these group means (shrinking toward the `type` mean) to reduce noise in sparse bins, which should improve generalization without changing the overall approach. Finally, we rebalance the blend weights slightly toward the improved distance+pair predictor to move the score downward toward your target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR_CANDIDATES = [
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/data/input/champs-scalar-coupling",
    "../input/champs-scalar-coupling",
]

DATA_DIR = None
for d in DATA_DIR_CANDIDATES:
    if os.path.isdir(d) and os.path.exists(os.path.join(d, "train.csv")):
        DATA_DIR = d
        break

if DATA_DIR is None:
    for root, _, files in os.walk("/kaggle"):
        if "train.csv" in files and "test.csv" in files:
            DATA_DIR = root
            break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not locate champs-scalar-coupling dataset directory containing train.csv/test.csv."
    )

print("Using DATA_DIR:", DATA_DIR)
print("Files:", sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])[:20])



## === cell 1
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

assert {
    "id",
    "molecule_name",
    "atom_index_0",
    "atom_index_1",
    "type",
    "scalar_coupling_constant",
}.issubset(train.columns)
assert {"id", "molecule_name", "atom_index_0", "atom_index_1", "type"}.issubset(
    test.columns
)
assert {"id", "scalar_coupling_constant"}.issubset(sample_sub.columns)

global_mean = float(train["scalar_coupling_constant"].mean())
type_mean = train.groupby("type")["scalar_coupling_constant"].mean()

structures = pd.read_csv(os.path.join(DATA_DIR, "structures.csv"))
structures_key = structures[["molecule_name", "atom_index", "atom"]]
structures_xyz = structures[["molecule_name", "atom_index", "x", "y", "z"]]


def add_atom_symbols(df, suffix0="_0", suffix1="_1"):
    df = df.merge(
        structures_key.rename(
            columns={"atom_index": "atom_index_0", "atom": f"atom{suffix0}"}
        ),
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    df = df.merge(
        structures_key.rename(
            columns={"atom_index": "atom_index_1", "atom": f"atom{suffix1}"}
        ),
        on=["molecule_name", "atom_index_1"],
        how="left",
    )
    return df


def add_pair_distance(df):
    df = df.merge(
        structures_xyz.rename(
            columns={
                "atom_index": "atom_index_0",
                "x": "x0",
                "y": "y0",
                "z": "z0",
            }
        ),
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    df = df.merge(
        structures_xyz.rename(
            columns={
                "atom_index": "atom_index_1",
                "x": "x1",
                "y": "y1",
                "z": "z1",
            }
        ),
        on=["molecule_name", "atom_index_1"],
        how="left",
    )
    dx = df["x0"] - df["x1"]
    dy = df["y0"] - df["y1"]
    dz = df["z0"] - df["z1"]
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float64)
    return df


train_atoms = add_atom_symbols(train)
test_atoms = add_atom_symbols(test)

pair_mean = train_atoms.groupby(["type", "atom_0", "atom_1"])[
    "scalar_coupling_constant"
].mean()

shrink = 0.15
type_mean_smoothed = (1.0 - shrink) * type_mean + shrink * global_mean

train_dist = add_pair_distance(
    train[
        [
            "molecule_name",
            "atom_index_0",
            "atom_index_1",
            "type",
            "scalar_coupling_constant",
        ]
    ].copy()
)
test_dist = add_pair_distance(
    test[["molecule_name", "atom_index_0", "atom_index_1", "type"]].copy()
)

n_bins = 120
try:
    train_dist["dist_bin"] = pd.qcut(train_dist["dist"], q=n_bins, duplicates="drop")
    edges = np.unique(
        np.concatenate(
            (
                [train_dist["dist"].min() - 1e-9],
                train_dist["dist_bin"].cat.categories.left.values,
                train_dist["dist_bin"].cat.categories.right.values,
                [train_dist["dist"].max() + 1e-9],
            )
        )
    )
    edges = np.unique(edges)
    test_dist["dist_bin"] = pd.cut(test_dist["dist"], bins=edges, include_lowest=True)
except Exception:
    lo, hi = float(train_dist["dist"].min()), float(train_dist["dist"].max())
    edges = np.linspace(lo - 1e-9, hi + 1e-9, num=n_bins + 1)
    train_dist["dist_bin"] = pd.cut(train_dist["dist"], bins=edges, include_lowest=True)
    test_dist["dist_bin"] = pd.cut(test_dist["dist"], bins=edges, include_lowest=True)

type_dist_mean = train_dist.groupby(["type", "dist_bin"])[
    "scalar_coupling_constant"
].mean()

train_dist_atoms = add_atom_symbols(
    train_dist.rename(columns={"scalar_coupling_constant": "scc"}),
    suffix0="_0",
    suffix1="_1",
)
test_dist_atoms = add_atom_symbols(test_dist.copy(), suffix0="_0", suffix1="_1")

grp_cols = ["type", "atom_0", "atom_1", "dist_bin"]
g = train_dist_atoms.groupby(grp_cols)["scc"].agg(["mean", "count"]).reset_index()

pair_mean_df = (
    train_atoms.groupby(["type", "atom_0", "atom_1"])["scalar_coupling_constant"]
    .mean()
    .rename("pair_mean")
    .reset_index()
)
g = g.merge(pair_mean_df, on=["type", "atom_0", "atom_1"], how="left")
g["pair_mean"] = g["pair_mean"].fillna(g["mean"])

k = 30.0
g["mean_shrunk"] = (g["count"] * g["mean"] + k * g["pair_mean"]) / (g["count"] + k)

type_atom_dist_mean = g.set_index(grp_cols)["mean_shrunk"]

sub1 = test[["id"]].copy()
sub2 = test[["id"]].copy()
sub3 = test[["id"]].copy()
sub4 = test[["id"]].copy()
sub5 = test[["id"]].copy()

sub1["scalar_coupling_constant"] = (
    test["type"].map(type_mean).fillna(global_mean).astype(np.float64)
)

sub2_pred = test_atoms.set_index(["type", "atom_0", "atom_1"]).index.map(pair_mean)
sub2["scalar_coupling_constant"] = (
    pd.Series(sub2_pred, index=test.index)
    .fillna(sub1["scalar_coupling_constant"])
    .astype(np.float64)
)

sub3["scalar_coupling_constant"] = (
    test["type"].map(type_mean_smoothed).fillna(global_mean).astype(np.float64)
)

sub4_pred = test_dist.set_index(["type", "dist_bin"]).index.map(type_dist_mean)
sub4["scalar_coupling_constant"] = (
    pd.Series(sub4_pred, index=test.index)
    .fillna(sub1["scalar_coupling_constant"])
    .astype(np.float64)
)

sub5_pred = test_dist_atoms.set_index(grp_cols).index.map(type_atom_dist_mean)
sub5["scalar_coupling_constant"] = (
    pd.Series(sub5_pred, index=test.index)
    .fillna(sub2["scalar_coupling_constant"])  # back off to chemistry-only
    .fillna(sub4["scalar_coupling_constant"])  # then geometry-only
    .fillna(sub1["scalar_coupling_constant"])  # then type mean
    .astype(np.float64)
)

print(sub1["scalar_coupling_constant"].describe())
print(sub2["scalar_coupling_constant"].describe())
print(sub3["scalar_coupling_constant"].describe())
print(sub4["scalar_coupling_constant"].describe())
print(sub5["scalar_coupling_constant"].describe())



## === cell 2
mad_12 = (
    (sub1["scalar_coupling_constant"] - sub2["scalar_coupling_constant"]).abs().mean()
)
mad_13 = (
    (sub1["scalar_coupling_constant"] - sub3["scalar_coupling_constant"]).abs().mean()
)
mad_23 = (
    (sub2["scalar_coupling_constant"] - sub3["scalar_coupling_constant"]).abs().mean()
)
mad_14 = (
    (sub1["scalar_coupling_constant"] - sub4["scalar_coupling_constant"]).abs().mean()
)
mad_24 = (
    (sub2["scalar_coupling_constant"] - sub4["scalar_coupling_constant"]).abs().mean()
)
mad_34 = (
    (sub3["scalar_coupling_constant"] - sub4["scalar_coupling_constant"]).abs().mean()
)
mad_15 = (
    (sub1["scalar_coupling_constant"] - sub5["scalar_coupling_constant"]).abs().mean()
)
mad_25 = (
    (sub2["scalar_coupling_constant"] - sub5["scalar_coupling_constant"]).abs().mean()
)
mad_45 = (
    (sub4["scalar_coupling_constant"] - sub5["scalar_coupling_constant"]).abs().mean()
)

print("MAD(sub1, sub2):", mad_12)
print("MAD(sub1, sub3):", mad_13)
print("MAD(sub2, sub3):", mad_23)
print("MAD(sub1, sub4):", mad_14)
print("MAD(sub2, sub4):", mad_24)
print("MAD(sub3, sub4):", mad_34)
print("MAD(sub1, sub5):", mad_15)
print("MAD(sub2, sub5):", mad_25)
print("MAD(sub4, sub5):", mad_45)



## === cell 3
w1, w2, w3, w4, w5 = 0.12, 0.18, 0.10, 0.15, 0.45  # sums to 1.0
sub1["scalar_coupling_constant"] = (
    w1 * sub1["scalar_coupling_constant"]
    + w2 * sub2["scalar_coupling_constant"]
    + w3 * sub3["scalar_coupling_constant"]
    + w4 * sub4["scalar_coupling_constant"]
    + w5 * sub5["scalar_coupling_constant"]
).astype(np.float64)

submission = sample_sub[["id"]].merge(sub1, on="id", how="left")
submission["scalar_coupling_constant"] = (
    submission["scalar_coupling_constant"].fillna(global_mean).astype(np.float64)
)

assert submission.shape[0] == sample_sub.shape[0]
assert submission["id"].isna().sum() == 0
assert submission["scalar_coupling_constant"].isna().sum() == 0

submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print(submission["scalar_coupling_constant"].describe())
