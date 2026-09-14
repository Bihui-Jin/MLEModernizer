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

-1.5209593019916507

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I remove the dependency on missing external Kaggle datasets (the two `../input/.../submission.csv` files) that currently cause the `FileNotFoundError`. To keep the pipeline end-to-end and score-reasonable with minimal logic, I generate predictions directly from the provided `train.csv` by computing the mean `scalar_coupling_constant` per coupling `type`, then apply those means to `test.csv` (falling back to the global mean if needed). This produces a valid `stackers_blend.csv` with the exact required columns and id alignment. The rest of the code structure is kept minimal and consistent with the original intent of creating a submission file.'
- What this solution (achieved 1.23566) has done: 'Your current baseline uses only per-`type` means, which ignores huge variation by atom pair and molecule geometry, so the score is far from the target (lower is better). With minimal change to the core “mean-encoding” approach, we can add a higher-resolution backoff hierarchy: mean by (`type`, `atom_0`, `atom_1`) using the atom symbols from `structures.csv`, then back off to (`type`, `atom_0`), (`type`, `atom_1`), then `type`, then global mean. This stays deterministic, fast, and uses only provided files, and it should reduce MAE substantially (thus lowering log-MAE toward the target) without changing the overall modeling paradigm. The submission writing and id alignment via `sample_submission.csv` are preserved exactly.'
- What this solution (achieved 1.23566) has done: 'Your current hierarchy is still too coarse for this competition because it ignores geometry; we can keep the same “groupby mean-encoding with backoff” core logic but add one strong, minimal feature: the inter-atomic distance computed from `structures.csv` coordinates. Then we add distance-binned means at the highest resolution (`type, atom_0, atom_1, dist_bin`) with smooth backoff to your existing keys, which should materially reduce MAE (and thus lower the log-MAE score toward the target). To avoid overfitting noise from sparse bins without changing the paradigm, we also add simple count-based shrinkage toward the parent mean for the distance-binned level. The submission writing, id alignment via `sample_submission.csv`, and overall deterministic, fast pipeline remain unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

BASE_PATH = "/kaggle/data/champs-scalar-coupling"
train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
sample_path = os.path.join(BASE_PATH, "sample_submission.csv")
structures_path = os.path.join(BASE_PATH, "structures.csv")

print("BASE_PATH exists:", os.path.exists(BASE_PATH))
print("Files in BASE_PATH (first 15):", sorted(os.listdir(BASE_PATH))[:15])



## === cell 1

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
sample = pd.read_csv(sample_path, usecols=["id", "scalar_coupling_constant"])

structures = pd.read_csv(
    structures_path, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
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

train = train.merge(
    s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
train = train.merge(
    s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)

test = test.merge(
    s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
test = test.merge(
    s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)

train["atom_0"] = train["atom_0"].fillna("UNK")
train["atom_1"] = train["atom_1"].fillna("UNK")
test["atom_0"] = test["atom_0"].fillna("UNK")
test["atom_1"] = test["atom_1"].fillna("UNK")

for df in (train, test):
    dx = df["x0"] - df["x1"]
    dy = df["y0"] - df["y1"]
    dz = df["z0"] - df["z1"]
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)

global_mean = float(train["scalar_coupling_constant"].mean())

BIN_WIDTH = 0.05
train["dist_bin"] = np.floor(train["dist"] / BIN_WIDTH).astype("Int32")
test["dist_bin"] = np.floor(test["dist"] / BIN_WIDTH).astype("Int32")

m_type_a0a1 = (
    train.groupby(["type", "atom_0", "atom_1"], as_index=False)[
        "scalar_coupling_constant"
    ]
    .mean()
    .rename(columns={"scalar_coupling_constant": "m_type_a0a1"})
)

m_type_a0 = (
    train.groupby(["type", "atom_0"], as_index=False)["scalar_coupling_constant"]
    .mean()
    .rename(columns={"scalar_coupling_constant": "m_type_a0"})
)
m_type_a1 = (
    train.groupby(["type", "atom_1"], as_index=False)["scalar_coupling_constant"]
    .mean()
    .rename(columns={"scalar_coupling_constant": "m_type_a1"})
)
m_type = (
    train.groupby(["type"], as_index=False)["scalar_coupling_constant"]
    .mean()
    .rename(columns={"scalar_coupling_constant": "m_type"})
)

m_type_a0a1_db = (
    train.dropna(subset=["dist_bin"])
    .groupby(["type", "atom_0", "atom_1", "dist_bin"])["scalar_coupling_constant"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "m_type_a0a1_db", "count": "c_type_a0a1_db"})
)

pred_df = test[["id", "type", "atom_0", "atom_1", "dist_bin"]].copy()

pred_df = pred_df.merge(m_type_a0a1, on=["type", "atom_0", "atom_1"], how="left")
pred_df = pred_df.merge(m_type_a0, on=["type", "atom_0"], how="left")
pred_df = pred_df.merge(m_type_a1, on=["type", "atom_1"], how="left")
pred_df = pred_df.merge(m_type, on=["type"], how="left")
pred_df = pred_df.merge(
    m_type_a0a1_db, on=["type", "atom_0", "atom_1", "dist_bin"], how="left"
)

ALPHA = 20.0
parent = pred_df["m_type_a0a1"]
num = pred_df["c_type_a0a1_db"].astype(np.float32)
bin_mean = pred_df["m_type_a0a1_db"]

shrunk_bin = (num * bin_mean + ALPHA * parent) / (num + ALPHA)
shrunk_bin = shrunk_bin.where(~parent.isna(), bin_mean)

pred = (
    shrunk_bin.fillna(pred_df["m_type_a0a1"])
    .fillna(pred_df["m_type_a0"])
    .fillna(pred_df["m_type_a1"])
    .fillna(pred_df["m_type"])
    .fillna(global_mean)
    .astype(np.float32)
)

submission = pd.DataFrame(
    {"id": pred_df["id"].values, "scalar_coupling_constant": pred.values}
)

submission = sample[["id"]].merge(submission, on="id", how="left")

missing = int(submission["scalar_coupling_constant"].isna().sum())
if missing:
    submission["scalar_coupling_constant"] = submission[
        "scalar_coupling_constant"
    ].fillna(global_mean)

out_path = "stackers_blend.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Rows:", len(submission), "Missing preds:", missing)
print("Global mean:", global_mean)
print(
    "Test atom_0/atom_1 UNK counts:",
    int((test["atom_0"] == "UNK").sum()),
    int((test["atom_1"] == "UNK").sum()),
)
print(
    "Missing dist (test):",
    int(test["dist"].isna().sum()),
    "Missing dist_bin (test):",
    int(test["dist_bin"].isna().sum()),
)
print(
    "Merged distance-bin availability (test):",
    int(pred_df["m_type_a0a1_db"].notna().sum()),
    "/",
    len(pred_df),
)
