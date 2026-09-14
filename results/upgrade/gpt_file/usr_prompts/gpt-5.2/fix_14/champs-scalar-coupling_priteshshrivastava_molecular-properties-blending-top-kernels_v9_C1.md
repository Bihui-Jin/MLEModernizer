# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

-1.67498

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

CANDIDATE_BASES = [
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling/champs-scalar-coupling",
]

REQUIRED_FILES = ["train.csv", "test.csv", "structures.csv", "sample_submission.csv"]


def _has_required_files(p: str) -> bool:
    return all(os.path.exists(os.path.join(p, fn)) for fn in REQUIRED_FILES)


def _norm_molecule_name(s: pd.Series) -> pd.Series:
    """
    Keep molecule_name unchanged except for safe whitespace stripping.
    (Do NOT strip prefixes; that breaks joins in the official CHAMPS files.)
    """
    return s.astype(str).str.strip()


def _join_miss_rate_for_path(p: str, sample_n: int = 2000) -> float:
    """
    Bug fix:
    Some mirrored directories can contain mismatched train/test vs structures.
    Validate the path by checking that test keys can join to structures for both atoms.
    """
    try:
        test_small = pd.read_csv(
            os.path.join(p, "test.csv"),
            nrows=sample_n,
            usecols=["molecule_name", "atom_index_0", "atom_index_1"],
            dtype={
                "molecule_name": "object",
                "atom_index_0": "int32",
                "atom_index_1": "int32",
            },
        )
        structures_small = pd.read_csv(
            os.path.join(p, "structures.csv"),
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
        test_small["molecule_name"] = _norm_molecule_name(test_small["molecule_name"])
        structures_small["molecule_name"] = _norm_molecule_name(
            structures_small["molecule_name"]
        )

        s0_small = structures_small.rename(
            columns={
                "atom_index": "atom_index_0",
                "atom": "atom_0",
                "x": "x_0",
                "y": "y_0",
                "z": "z_0",
            }
        )[["molecule_name", "atom_index_0", "atom_0", "x_0", "y_0", "z_0"]]
        s1_small = structures_small.rename(
            columns={
                "atom_index": "atom_index_1",
                "atom": "atom_1",
                "x": "x_1",
                "y": "y_1",
                "z": "z_1",
            }
        )[["molecule_name", "atom_index_1", "atom_1", "x_1", "y_1", "z_1"]]

        tmp = test_small.merge(
            s0_small, on=["molecule_name", "atom_index_0"], how="left"
        )
        tmp = tmp.merge(s1_small, on=["molecule_name", "atom_index_1"], how="left")
        miss = tmp[["atom_0", "atom_1", "x_0", "x_1"]].isna().any(axis=1).mean()
        return float(miss)
    except Exception:
        return 1.0


def _pick_data_path(candidates):
    candidates = [p for p in candidates if os.path.exists(p)]
    candidates = [p for p in candidates if _has_required_files(p)]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find any valid data directory. Tried: {CANDIDATE_BASES}"
        )

    scored = []
    for p in candidates:
        miss = _join_miss_rate_for_path(p, sample_n=2000)
        scored.append((miss, p))
    scored.sort(key=lambda x: x[0])

    best_miss, best_path = scored[0]
    print("Candidate DATA_PATH join-miss rates (lower is better):")
    for miss, p in scored:
        print(f"  miss={miss:.6f}  path={p}")
    if best_miss > 0.01:
        raise FileNotFoundError(
            f"No candidate data directory had an acceptable structures join miss-rate. Best miss={best_miss:.6f} at {best_path}"
        )
    return best_path


DATA_PATH = _pick_data_path(CANDIDATE_BASES)
print("Using DATA_PATH:", DATA_PATH)
print("Files sample:", sorted(os.listdir(DATA_PATH))[:10])



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3692336637.py in <cell line: 0>()
    114 
    115 
--> 116 DATA_PATH = _pick_data_path(CANDIDATE_BASES)
    117 print("Using DATA_PATH:", DATA_PATH)
    118 print("Files sample:", sorted(os.listdir(DATA_PATH))[:10])

/tmp/ipykernel_11/3692336637.py in _pick_data_path(candidates)
    108         print(f"  miss={miss:.6f}  path={p}")
    109     if best_miss > 0.01:
--> 110         raise FileNotFoundError(
    111             f"No candidate data directory had an acceptable structures join miss-rate. Best miss={best_miss:.6f} at {best_path}"
    112         )

FileNotFoundError: No candidate data directory had an acceptable structures join miss-rate. Best miss=1.000000 at /kaggle/data/champs-scalar-coupling

## === cell 1
train = pd.read_csv(
    os.path.join(DATA_PATH, "train.csv"),
    dtype={
        "id": "int64",
        "molecule_name": "object",
        "atom_index_0": "int32",
        "atom_index_1": "int32",
        "type": "object",
        "scalar_coupling_constant": "float32",
    },
)
test = pd.read_csv(
    os.path.join(DATA_PATH, "test.csv"),
    dtype={
        "id": "int64",
        "molecule_name": "object",
        "atom_index_0": "int32",
        "atom_index_1": "int32",
        "type": "object",
    },
)
structures = pd.read_csv(
    os.path.join(DATA_PATH, "structures.csv"),
    dtype={
        "molecule_name": "object",
        "atom_index": "int32",
        "atom": "object",
        "x": "float32",
        "y": "float32",
        "z": "float32",
    },
)

train["molecule_name"] = _norm_molecule_name(train["molecule_name"])
test["molecule_name"] = _norm_molecule_name(test["molecule_name"])
structures["molecule_name"] = _norm_molecule_name(structures["molecule_name"])

print(train.shape, test.shape, structures.shape)
print(train.columns)


def _compute_join_miss_rate(
    df_keys: pd.DataFrame, s0_small: pd.DataFrame, s1_small: pd.DataFrame
) -> float:
    tmp = df_keys.merge(s0_small, on=["molecule_name", "atom_index_0"], how="left")
    tmp = tmp.merge(s1_small, on=["molecule_name", "atom_index_1"], how="left")
    miss = tmp[["atom_0", "atom_1", "x_0", "x_1"]].isna().any(axis=1).mean()
    return float(miss)


s0_small = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x_0",
        "y": "y_0",
        "z": "z_0",
    }
)[["molecule_name", "atom_index_0", "atom_0", "x_0", "y_0", "z_0"]]
s1_small = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x_1",
        "y": "y_1",
        "z": "z_1",
    }
)[["molecule_name", "atom_index_1", "atom_1", "x_1", "y_1", "z_1"]]

sample_n = min(20000, len(test))
sample_keys = test.loc[
    : sample_n - 1, ["molecule_name", "atom_index_0", "atom_index_1"]
].copy()

candidates = [0, -1, 1]
miss_rates = {}
for shift in candidates:
    keys_shifted = sample_keys.copy()
    if shift != 0:
        keys_shifted["atom_index_0"] = (keys_shifted["atom_index_0"] + shift).astype(
            np.int32
        )
        keys_shifted["atom_index_1"] = (keys_shifted["atom_index_1"] + shift).astype(
            np.int32
        )
    miss_rates[shift] = _compute_join_miss_rate(keys_shifted, s0_small, s1_small)

best_shift = min(miss_rates, key=miss_rates.get)
base_miss = miss_rates[0]
best_miss = miss_rates[best_shift]
print("Join-miss rates by atom_index shift:", miss_rates)

apply_shift = best_shift if (best_shift != 0 and best_miss + 1e-12 < base_miss) else 0
print(
    "Base miss:", base_miss, "Best miss:", best_miss, "=> applying shift:", apply_shift
)

if apply_shift != 0:
    train["atom_index_0"] = (train["atom_index_0"] + apply_shift).astype(np.int32)
    train["atom_index_1"] = (train["atom_index_1"] + apply_shift).astype(np.int32)
    test["atom_index_0"] = (test["atom_index_0"] + apply_shift).astype(np.int32)
    test["atom_index_1"] = (test["atom_index_1"] + apply_shift).astype(np.int32)

sample_keys2 = test.loc[
    : sample_n - 1, ["molecule_name", "atom_index_0", "atom_index_1"]
].copy()
miss2 = _compute_join_miss_rate(sample_keys2, s0_small, s1_small)
print(
    f"Join-miss rate on first {sample_n} test rows after shift {apply_shift}:",
    float(miss2),
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/854521039.py in <cell line: 0>()
      1 train = pd.read_csv(
----> 2     os.path.join(DATA_PATH, "train.csv"),
      3     dtype={
      4         "id": "int64",
      5         "molecule_name": "object",

NameError: name 'DATA_PATH' is not defined

## === cell 2
struct_cols = ["molecule_name", "atom_index", "atom", "x", "y", "z"]
structures = structures[struct_cols].copy()

s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x_0",
        "y": "y_0",
        "z": "z_0",
    }
)
s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x_1",
        "y": "y_1",
        "z": "z_1",
    }
)


def add_structure_features(df):
    df = df.copy()
    df["molecule_name"] = _norm_molecule_name(df["molecule_name"])

    df = df.merge(
        s0,
        on=["molecule_name", "atom_index_0"],
        how="left",
        validate="many_to_many",
    )
    df = df.merge(
        s1,
        on=["molecule_name", "atom_index_1"],
        how="left",
        validate="many_to_many",
    )

    dx = df["x_0"] - df["x_1"]
    dy = df["y_0"] - df["y_1"]
    dz = df["z_0"] - df["z_1"]
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)

    df["dx"] = dx
    df["dy"] = dy
    df["dz"] = dz

    bad_mask = df[["atom_0", "atom_1", "dist"]].isna().any(axis=1)
    if bad_mask.any():
        n_bad = int(bad_mask.sum())
        example = df.loc[
            bad_mask, ["molecule_name", "atom_index_0", "atom_index_1"]
        ].head(10)
        raise ValueError(
            f"Missing structure join for {n_bad} rows; example keys:\n{example.to_string(index=False)}"
        )
    return df


train_feat = add_structure_features(train)
test_feat = add_structure_features(test)

print(train_feat.shape, test_feat.shape)
print(train_feat[["atom_0", "atom_1", "type", "dist"]].head())



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1653471570.py in <cell line: 0>()
      1 struct_cols = ["molecule_name", "atom_index", "atom", "x", "y", "z"]
----> 2 structures = structures[struct_cols].copy()
      3 
      4 s0 = structures.rename(
      5     columns={

NameError: name 'structures' is not defined

## === cell 3
from sklearn.preprocessing import LabelEncoder

cat_cols = ["type", "atom_0", "atom_1"]
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train_feat[c], test_feat[c]], axis=0).astype(str))
    train_feat[c] = le.transform(train_feat[c].astype(str))
    test_feat[c] = le.transform(test_feat[c].astype(str))
    encoders[c] = le

feature_cols = ["type", "atom_0", "atom_1", "dist", "dx", "dy", "dz"]
X_train = train_feat[feature_cols].values
y_train = train_feat["scalar_coupling_constant"].values
X_test = test_feat[feature_cols].values

print("X_train:", X_train.shape, "X_test:", X_test.shape)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4154990526.py in <cell line: 0>()
      5 for c in cat_cols:
      6     le = LabelEncoder()
----> 7     le.fit(pd.concat([train_feat[c], test_feat[c]], axis=0).astype(str))
      8     train_feat[c] = le.transform(train_feat[c].astype(str))
      9     test_feat[c] = le.transform(test_feat[c].astype(str))

NameError: name 'train_feat' is not defined

## === cell 4
from sklearn.ensemble import HistGradientBoostingRegressor

model = HistGradientBoostingRegressor(
    loss="absolute_error",  # aligns with MAE-style metric; stable baseline
    max_depth=10,
    learning_rate=0.08,
    max_iter=250,
    random_state=42,
)

model.fit(X_train, y_train)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3290020312.py in <cell line: 0>()
      9 )
     10 
---> 11 model.fit(X_train, y_train)
     12 

NameError: name 'X_train' is not defined

## === cell 5
test_pred = model.predict(X_test)

submission = pd.DataFrame(
    {"id": test_feat["id"].values, "scalar_coupling_constant": test_pred}
)

sample = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"), usecols=["id"])
submission = sample.merge(submission, on="id", how="left", validate="one_to_one")

if submission["scalar_coupling_constant"].isna().any():
    raise ValueError(
        "Submission has missing predictions after aligning to sample_submission ids."
    )

submission["id"] = submission["id"].astype(np.int64)
submission = submission.sort_values("id").reset_index(drop=True)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path, "shape:", submission.shape)
print(submission.head())
print(submission.tail())
print("Submission columns:", submission.columns.tolist())
print("id unique:", submission["id"].is_unique)
print("id min/max:", submission["id"].min(), submission["id"].max())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1393470841.py in <cell line: 0>()
----> 1 test_pred = model.predict(X_test)
      2 
      3 submission = pd.DataFrame(
      4     {"id": test_feat["id"].values, "scalar_coupling_constant": test_pred}
      5 )

NameError: name 'X_test' is not defined
