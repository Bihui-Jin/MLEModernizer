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

-1.3684302901167014

# 6. Current score

1.18497

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.18497) has done: 'I remove the dependency on missing external blend files (`../input/blend/...` and `../input/statstack/...`) that cause the `FileNotFoundError`, and instead generate a valid baseline prediction directly from the provided competition data. To keep the core approach “simple blending/statistical” and score-improving versus all-zeros, I compute per-`type` median `scalar_coupling_constant` from `train.csv` and use that to predict `test.csv`, with a global median fallback for any unseen types. This run end-to-end in the given environment and write a correctly formatted `.csv` submission file.'
- What this solution (achieved 1.18497) has done: 'Your current baseline uses only per-`type` medians, which ignores key signal that varies strongly by atom pair geometry and chemistry; to move the score down toward the target, we add a small set of well-known CHAMPS features while keeping the same “simple statistical model” core logic. Specifically, we merge `structures.csv` to get both atoms’ coordinates and element types, compute the inter-atomic distance and a few lightweight derived features, then predict using per-(`type`, `atom_0`, `atom_1`) medians with sensible fallbacks to per-(`type`, binned distance) medians and finally to your existing per-`type` median. This preserves evaluation semantics (still a deterministic median-based estimator) but typically improves MAE a lot versus type-only. The script still runs end-to-end and writes a valid `submission.csv` with the required columns and row alignment.'
- What this solution (achieved 1.18497) has done: 'Your current median-based estimator is already valid but it “leaks” some avoidable error because it doesn’t canonicalize atom order consistently and uses a very fine distance bin (0.05Å) that sparsifies medians and pushes many test rows to weaker fallbacks. I keep the same core logic (deterministic medians with fallbacks) but (1) compute only canonicalized (`atom_a`,`atom_b`) medians (so train/test keying matches regardless of atom order), (2) use slightly wider distance bins (0.10Å) to reduce sparsity and improve the mid-level fallback, and (3) add an additional, still-statistical fallback using per-(`type`,`atom_a`,`atom_b`,`dist_bin`) medians before backing off to per-(`type`,`atom_a`,`atom_b`) and then per-`type`. These are minimal changes that usually reduce MAE materially (lower score is better) without changing the overall approach or adding any learning model. The script still runs end-to-end and writes `submission.csv` in the correct format.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

INPUT_ROOT = "/kaggle/data/champs-scalar-coupling"

print("Listing /kaggle/data:", os.listdir("/kaggle/data")[:20])
print("Using INPUT_ROOT:", INPUT_ROOT)
print("Files in INPUT_ROOT:", sorted(os.listdir(INPUT_ROOT))[:20])




## === cell 1
train_path = os.path.join(INPUT_ROOT, "train.csv")
test_path = os.path.join(INPUT_ROOT, "test.csv")
sample_path = os.path.join(INPUT_ROOT, "sample_submission.csv")
structures_path = os.path.join(INPUT_ROOT, "structures.csv")

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
    test_path,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
)
submission = pd.read_csv(sample_path)

assert "id" in submission.columns and "scalar_coupling_constant" in submission.columns
assert len(submission) == len(
    test
), "sample_submission and test must have the same number of rows"

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


def add_struct_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

    dx = (df["x0"] - df["x1"]).astype("float64")
    dy = (df["y0"] - df["y1"]).astype("float64")
    dz = (df["z0"] - df["z1"]).astype("float64")
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)

    df["inv_dist"] = 1.0 / (df["dist"] + 1e-12)
    df["dist2"] = df["dist"] * df["dist"]

    df["dist_bin"] = np.floor((df["dist"] / 0.10)).astype("Int64")  # 0.10 Å bins

    return df


train_f = add_struct_features(train)
test_f = add_struct_features(test)

type_median = train_f.groupby("type")["scalar_coupling_constant"].median()
global_median = float(train_f["scalar_coupling_constant"].median())




## === cell 2
train_f["atom_a"] = train_f[["atom_0", "atom_1"]].min(axis=1)
train_f["atom_b"] = train_f[["atom_0", "atom_1"]].max(axis=1)
test_f["atom_a"] = test_f[["atom_0", "atom_1"]].min(axis=1)
test_f["atom_b"] = test_f[["atom_0", "atom_1"]].max(axis=1)

med_type_atoms_distbin = (
    train_f.groupby(["type", "atom_a", "atom_b", "dist_bin"])[
        "scalar_coupling_constant"
    ]
    .median()
    .rename("m")
)

med_type_atoms_sorted = (
    train_f.groupby(["type", "atom_a", "atom_b"])["scalar_coupling_constant"]
    .median()
    .rename("m")
)

med_type_distbin = (
    train_f.groupby(["type", "dist_bin"])["scalar_coupling_constant"]
    .median()
    .rename("m")
)

pred = pd.Series(np.nan, index=test_f.index, dtype="float64")

k0 = pd.MultiIndex.from_frame(test_f[["type", "atom_a", "atom_b", "dist_bin"]])
pred0 = pd.Series(
    med_type_atoms_distbin.reindex(k0).to_numpy(), index=test_f.index, dtype="float64"
)
pred = pred.fillna(pred0)

k2 = pd.MultiIndex.from_frame(test_f[["type", "atom_a", "atom_b"]])
pred2 = pd.Series(
    med_type_atoms_sorted.reindex(k2).to_numpy(), index=test_f.index, dtype="float64"
)
pred = pred.fillna(pred2)

k3 = pd.MultiIndex.from_frame(test_f[["type", "dist_bin"]])
pred3 = pd.Series(
    med_type_distbin.reindex(k3).to_numpy(), index=test_f.index, dtype="float64"
)
pred = pred.fillna(pred3)

pred4 = test_f["type"].map(type_median).astype("float64")
pred = pred.fillna(pred4)

pred = pred.fillna(global_median)

assert pred.isna().sum() == 0




## === cell 3
submission = submission.copy()
submission["scalar_coupling_constant"] = pred.to_numpy()

assert len(submission) == len(test)
assert submission["scalar_coupling_constant"].isna().sum() == 0
assert submission["id"].nunique() == len(submission)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())




## === cell 4
submission["scalar_coupling_constant"].describe()
