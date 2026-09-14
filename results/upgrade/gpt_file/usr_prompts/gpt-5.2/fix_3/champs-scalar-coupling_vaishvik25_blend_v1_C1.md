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

-1.32544

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I remove the dependency on missing `../input/blender/*.csv` files (the root cause of the FileNotFoundError) and instead build a valid submission from the provided competition data. To keep changes minimal while ensuring an end-to-end run, I generate predictions using a simple, leakage-safe baseline: predict the per-`type` mean `scalar_coupling_constant` from `train.csv`, and fall back to the global mean if a type is unseen. This produces a correctly formatted `submission.csv` with the required columns and should yield a reasonable (not top) score rather than failing to submit. I also keep the original “blend” intent by leaving a small blending helper that would work if extra submissions existed, but it default to the baseline in this environment.'
- What this solution (achieved 1.23566) has done: 'You’re currently far from the (much better) target (gap ≈ 2.56 with lower-is-better), so we need a legitimate performance lift while keeping the same “type-based mean baseline” core logic. The smallest high-impact upgrade is to compute the mean at a finer granularity: use the mean by (`type`, `atom_0`, `atom_1`) derived by joining `train/test` with `structures.csv` to get element symbols for each atom index. We then back off smoothly to the existing per-`type` mean and finally the global mean for any missing combinations, preserving the same prediction approach (group means) but with richer keys. This remains leakage-safe (only uses train targets for aggregations) and should move the score substantially toward the target while still producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

BASE_DIR_CANDIDATES = [
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/data/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling/champs-scalar-coupling",
    "/kaggle/data/input/champs-scalar-coupling/champs-scalar-coupling",
]


def find_competition_dir(candidates):
    for d in candidates:
        if os.path.exists(d) and os.path.isfile(os.path.join(d, "train.csv")):
            return d
    for root in ["/kaggle/input", "/kaggle/data", "/kaggle/data/input"]:
        if os.path.exists(root):
            for dirpath, dirnames, filenames in os.walk(root):
                if (
                    "train.csv" in filenames
                    and "test.csv" in filenames
                    and "sample_submission.csv" in filenames
                ):
                    return dirpath
    raise FileNotFoundError(
        "Could not locate competition directory containing train.csv/test.csv/sample_submission.csv"
    )


COMP_DIR = find_competition_dir(BASE_DIR_CANDIDATES)
print("Using competition directory:", COMP_DIR)
print(
    "Files:",
    sorted([f for f in os.listdir(COMP_DIR) if f.endswith(".csv")])[:10],
    "...",
)

train_path = os.path.join(COMP_DIR, "train.csv")
test_path = os.path.join(COMP_DIR, "test.csv")
sample_path = os.path.join(COMP_DIR, "sample_submission.csv")
structures_path = os.path.join(COMP_DIR, "structures.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

print("train:", train.shape, "test:", test.shape, "sample:", sample.shape)
print("train columns:", train.columns.tolist())
print("test columns:", test.columns.tolist())
print("sample columns:", sample.columns.tolist())



## === cell 1
required_train = {
    "type",
    "scalar_coupling_constant",
    "molecule_name",
    "atom_index_0",
    "atom_index_1",
}
required_test = {"id", "type", "molecule_name", "atom_index_0", "atom_index_1"}
if not required_train.issubset(train.columns):
    raise ValueError(
        f"train.csv missing required columns: {required_train - set(train.columns)}"
    )
if not required_test.issubset(test.columns):
    raise ValueError(
        f"test.csv missing required columns: {required_test - set(test.columns)}"
    )

if not os.path.isfile(structures_path):
    raise FileNotFoundError(f"structures.csv not found at: {structures_path}")

structures = pd.read_csv(
    structures_path, usecols=["molecule_name", "atom_index", "atom"]
)
structures["atom_index"] = structures["atom_index"].astype(np.int32)


def add_atom_symbols(df, structures_df):
    df = df.copy()
    df["atom_index_0"] = df["atom_index_0"].astype(np.int32)
    df["atom_index_1"] = df["atom_index_1"].astype(np.int32)

    s0 = structures_df.rename(columns={"atom_index": "atom_index_0", "atom": "atom_0"})
    s1 = structures_df.rename(columns={"atom_index": "atom_index_1", "atom": "atom_1"})

    df = df.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(s1, on=["molecule_name", "atom_index_1"], how="left")
    return df


train_e = add_atom_symbols(
    train[
        [
            "type",
            "scalar_coupling_constant",
            "molecule_name",
            "atom_index_0",
            "atom_index_1",
        ]
    ],
    structures,
)
test_e = add_atom_symbols(
    test[["id", "type", "molecule_name", "atom_index_0", "atom_index_1"]], structures
)

train_e["atom_0"] = train_e["atom_0"].fillna("UNK")
train_e["atom_1"] = train_e["atom_1"].fillna("UNK")
test_e["atom_0"] = test_e["atom_0"].fillna("UNK")
test_e["atom_1"] = test_e["atom_1"].fillna("UNK")

type_mean = train.groupby("type")["scalar_coupling_constant"].mean()
global_mean = float(train["scalar_coupling_constant"].mean())

type_atom_mean = train_e.groupby(["type", "atom_0", "atom_1"])[
    "scalar_coupling_constant"
].mean()

key = list(zip(test_e["type"].values, test_e["atom_0"].values, test_e["atom_1"].values))
pred = pd.Series(key, index=test_e.index).map(type_atom_mean)

pred = pred.fillna(test_e["type"].map(type_mean)).fillna(global_mean).astype(np.float64)

submission = pd.DataFrame(
    {"id": test_e["id"].astype(np.int64), "scalar_coupling_constant": pred.values}
)

if "id" in sample.columns and len(sample) == len(submission):
    submission = sample[["id"]].merge(submission, on="id", how="left")
    submission["scalar_coupling_constant"] = (
        submission["scalar_coupling_constant"].fillna(global_mean).astype(np.float64)
    )

assert submission.columns.tolist() == ["id", "scalar_coupling_constant"]
assert submission["id"].isna().sum() == 0
assert submission["scalar_coupling_constant"].isna().sum() == 0
assert len(submission) == len(test)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())
print(submission["scalar_coupling_constant"].describe())



## === cell 2
try:
    ax = submission["scalar_coupling_constant"].plot(
        kind="hist", bins=100, title="Predicted scalar_coupling_constant"
    )
    fig = ax.get_figure()
    fig.tight_layout()
except Exception as e:
    print("Plot skipped:", repr(e))
