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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

-1.5192329259486128

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.18497) has done: 'I fix the immediate runtime blockers by removing notebook-only magics, replacing missing `../input/top-mol` and `../input/another-one` dependencies with a self-contained baseline that uses only the provided competition files, and ensuring all variables used downstream are defined. Because you currently have no valid submission (“Not yielded”), the priority is to reliably write a correctly formatted `submission.csv` with `id,scalar_coupling_constant`. To keep changes minimal while improving score versus a trivial constant, I use a simple type-wise median target computed from `train.csv` and mapped onto `test.csv` by `type` (a common strong baseline for this metric). The script also include safe fallbacks and alignment checks so it runs end-to-end in the Kaggle environment.'
- What this solution (achieved 1.23596) has done: 'Your current score (1.18497, lower-is-better) is far worse than the target (-1.5192), so we should improve substantially while keeping the same “type-wise statistic mapping” core approach. The smallest meaningful upgrade is to predict per-`type` **mean** instead of median (often better for MAE-like objectives here) and to add a tiny amount of robust shrinkage toward the global mean to stabilize rare types. We also compute these statistics from the full `train.csv` target (same as before), keep the same submission format and ID alignment checks, and still write `submission.csv` end-to-end.'
- What this solution (achieved 1.18495) has done: 'Your current approach is a per-`type` constant predictor; to move the score substantially toward the target while preserving that core logic, the most direct improvement is to predict the **per-`type` median** (optimal for MAE) instead of a shrunk mean. To keep stability for smaller `type` groups (and avoid extreme medians from tiny counts), I apply a minimal “shrinkage toward global median” using the same pseudo-count idea you already use. This stays within the same semantics (type-wise statistic mapping) and only changes which robust statistic is mapped and how it’s smoothed. The submission writing, ID alignment, and paths remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR_CANDIDATES = [
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "../input/champs-scalar-coupling",
    "../input",  # sometimes files are directly here
]


def find_data_dir():
    for d in DATA_DIR_CANDIDATES:
        if os.path.isdir(d):
            if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
                os.path.join(d, "test.csv")
            ):
                return d
    for d in DATA_DIR_CANDIDATES:
        if os.path.isdir(d):
            for root, _, files in os.walk(d):
                if "train.csv" in files and "test.csv" in files:
                    return root
    raise FileNotFoundError(
        "Could not locate champs-scalar-coupling data directory with train.csv/test.csv"
    )


DATA_DIR = find_data_dir()
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")

print("Using DATA_DIR:", DATA_DIR)
print(
    "Train exists:",
    os.path.exists(train_path),
    "Test exists:",
    os.path.exists(test_path),
    "Sample exists:",
    os.path.exists(sample_path),
    "Structures exists:",
    os.path.exists(structures_path),
)



## === cell 1

from sklearn.linear_model import Ridge

train = pd.read_csv(
    train_path,
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
    test_path,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
)

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


def add_pair_features(df):
    df = df.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

    dx = df["x0"] - df["x1"]
    dy = df["y0"] - df["y1"]
    dz = df["z0"] - df["z1"]
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype("float64")

    a0 = df["atom_0"].astype("string")
    a1 = df["atom_1"].astype("string")
    df["atom_pair"] = np.where(a0 <= a1, a0 + "_" + a1, a1 + "_" + a0).astype("string")

    df["dist"] = df["dist"].fillna(df["dist"].median())
    df["atom_pair"] = df["atom_pair"].fillna("UNK_UNK")
    return df


train_f = add_pair_features(train)
test_f = add_pair_features(test)

global_median = float(train_f["scalar_coupling_constant"].median())

print("Train features head:")
print(train_f[["type", "atom_pair", "dist", "scalar_coupling_constant"]].head())



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2442307913.py in <cell line: 0>()
     67 
     68 
---> 69 train_f = add_pair_features(train)
     70 test_f = add_pair_features(test)
     71 

/tmp/ipykernel_11/2442307913.py in add_pair_features(df)
     59     a0 = df["atom_0"].astype("string")
     60     a1 = df["atom_1"].astype("string")
---> 61     df["atom_pair"] = np.where(a0 <= a1, a0 + "_" + a1, a1 + "_" + a0).astype("string")
     62 
     63     # fallbacks if merge missing for any row (shouldn't happen)

TypeError: data type 'string' not understood

## === cell 2

type_stats = train_f.groupby("type")["scalar_coupling_constant"].agg(
    ["median", "count"]
)
alpha = 50.0
type_shrunk_median = (
    type_stats["median"] * type_stats["count"] + global_median * alpha
) / (type_stats["count"] + alpha)

pred = pd.Series(index=test_f.index, dtype="float64")

MIN_ROWS_PER_TYPE = 200  # ensure stable fits; small types fall back to shrunk median
RIDGE_ALPHA = 1.0  # light regularization for stability

for t, test_idx in test_f.groupby("type").groups.items():
    base = float(type_shrunk_median.get(t, global_median))

    tr_t = train_f[train_f["type"] == t]
    te_t = test_f.loc[test_idx]

    if len(tr_t) < MIN_ROWS_PER_TYPE:
        pred.loc[test_idx] = base
        continue

    X_tr = pd.get_dummies(tr_t[["atom_pair"]], prefix="ap", dummy_na=False)
    X_te = pd.get_dummies(te_t[["atom_pair"]], prefix="ap", dummy_na=False)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)

    X_tr = pd.concat(
        [tr_t[["dist"]].reset_index(drop=True), X_tr.reset_index(drop=True)], axis=1
    )
    X_te = pd.concat(
        [te_t[["dist"]].reset_index(drop=True), X_te.reset_index(drop=True)], axis=1
    )

    y_tr = tr_t["scalar_coupling_constant"].astype("float64").values

    if not np.isfinite(X_tr.values).all() or not np.isfinite(y_tr).all():
        pred.loc[test_idx] = base
        continue

    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, random_state=0)
    model.fit(X_tr.values, y_tr)
    y_hat = model.predict(X_te.values).astype("float64")

    blend = 0.10
    y_hat = (1.0 - blend) * y_hat + blend * base

    pred.loc[test_idx] = y_hat

pred = pred.fillna(global_median)

submission = pd.DataFrame(
    {
        "id": test_f["id"].astype(np.int64),
        "scalar_coupling_constant": pred.astype("float64"),
    }
)
submission.sort_values("id", inplace=True)

print(submission.head())
print("Submission shape:", submission.shape)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/854737844.py in <cell line: 0>()
      3 
      4 # Prior: shrunk per-type median (your existing logic), used as a safe fallback/prediction baseline
----> 5 type_stats = train_f.groupby("type")["scalar_coupling_constant"].agg(
      6     ["median", "count"]
      7 )

NameError: name 'train_f' is not defined

## === cell 3
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    if "id" in sample.columns and len(sample) == len(submission):
        same_ids = sample["id"].astype(np.int64).values
        sub_ids = submission["id"].astype(np.int64).values
        if not np.array_equal(np.sort(same_ids), np.sort(sub_ids)):
            print("Warning: submission ids do not match sample_submission ids set.")
        else:
            print("ID set matches sample_submission.")
    else:
        print(
            "sample_submission.csv present but shape/columns unexpected; skipping strict alignment check."
        )
else:
    print("sample_submission.csv not found; skipping alignment check.")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1791673003.py in <cell line: 0>()
      1 if os.path.exists(sample_path):
      2     sample = pd.read_csv(sample_path)
----> 3     if "id" in sample.columns and len(sample) == len(submission):
      4         same_ids = sample["id"].astype(np.int64).values
      5         sub_ids = submission["id"].astype(np.int64).values

NameError: name 'submission' is not defined

## === cell 4
out_path = "submission.csv"
submission.to_csv(out_path, index=False, float_format="%.6f")
print("Wrote:", out_path, "size(bytes):", os.path.getsize(out_path))
print(pd.read_csv(out_path).head())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1805010764.py in <cell line: 0>()
      1 out_path = "submission.csv"
----> 2 submission.to_csv(out_path, index=False, float_format="%.6f")
      3 print("Wrote:", out_path, "size(bytes):", os.path.getsize(out_path))
      4 print(pd.read_csv(out_path).head())

NameError: name 'submission' is not defined
