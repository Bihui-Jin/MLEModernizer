# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_PATH = "/kaggle/data/champs-scalar-coupling"
train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
structures_path = os.path.join(BASE_PATH, "structures.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
structures = pd.read_csv(structures_path)

mulliken_path = os.path.join(BASE_PATH, "mulliken_charges.csv")
mst_path = os.path.join(BASE_PATH, "magnetic_shielding_tensors.csv")
dipole_path = os.path.join(BASE_PATH, "dipole_moments.csv")
pe_path = os.path.join(BASE_PATH, "potential_energy.csv")

mulliken = pd.read_csv(mulliken_path) if os.path.exists(mulliken_path) else None
mst = pd.read_csv(mst_path) if os.path.exists(mst_path) else None
dipole = pd.read_csv(dipole_path) if os.path.exists(dipole_path) else None
potential_energy = pd.read_csv(pe_path) if os.path.exists(pe_path) else None



## === cell 1
structures_0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x_0",
        "y": "y_0",
        "z": "z_0",
    }
)
structures_1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x_1",
        "y": "y_1",
        "z": "z_1",
    }
)

train_2 = train.merge(structures_0, on=["molecule_name", "atom_index_0"], how="left")
train_2 = train_2.merge(structures_1, on=["molecule_name", "atom_index_1"], how="left")

test_2 = test.merge(structures_0, on=["molecule_name", "atom_index_0"], how="left")
test_2 = test_2.merge(structures_1, on=["molecule_name", "atom_index_1"], how="left")

if mulliken is not None:
    mull_0 = mulliken.rename(
        columns={"atom_index": "atom_index_0", "mulliken_charge": "mcharge_0"}
    )
    mull_1 = mulliken.rename(
        columns={"atom_index": "atom_index_1", "mulliken_charge": "mcharge_1"}
    )
    train_2 = train_2.merge(
        mull_0[["molecule_name", "atom_index_0", "mcharge_0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    train_2 = train_2.merge(
        mull_1[["molecule_name", "atom_index_1", "mcharge_1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )
    test_2 = test_2.merge(
        mull_0[["molecule_name", "atom_index_0", "mcharge_0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    test_2 = test_2.merge(
        mull_1[["molecule_name", "atom_index_1", "mcharge_1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )
else:
    train_2["mcharge_0"] = np.nan
    train_2["mcharge_1"] = np.nan
    test_2["mcharge_0"] = np.nan
    test_2["mcharge_1"] = np.nan

if mst is not None:
    tensor_cols = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
    mst_0 = mst.rename(columns={"atom_index": "atom_index_0"})
    mst_1 = mst.rename(columns={"atom_index": "atom_index_1"})
    mst_0 = mst_0[["molecule_name", "atom_index_0"] + tensor_cols].copy()
    mst_1 = mst_1[["molecule_name", "atom_index_1"] + tensor_cols].copy()

    for df, suf in [(mst_0, "_0"), (mst_1, "_1")]:
        vals = df[tensor_cols].astype(np.float64)
        df["mst_mean" + suf] = vals.mean(axis=1)
        df["mst_std" + suf] = vals.std(axis=1)
        df["mst_absmean" + suf] = vals.abs().mean(axis=1)
        df["mst_trace" + suf] = (df["XX"] + df["YY"] + df["ZZ"]).astype(np.float64)
        df.drop(columns=tensor_cols, inplace=True)

    train_2 = train_2.merge(mst_0, on=["molecule_name", "atom_index_0"], how="left")
    train_2 = train_2.merge(mst_1, on=["molecule_name", "atom_index_1"], how="left")
    test_2 = test_2.merge(mst_0, on=["molecule_name", "atom_index_0"], how="left")
    test_2 = test_2.merge(mst_1, on=["molecule_name", "atom_index_1"], how="left")
else:
    for col in [
        "mst_mean_0",
        "mst_std_0",
        "mst_absmean_0",
        "mst_trace_0",
        "mst_mean_1",
        "mst_std_1",
        "mst_absmean_1",
        "mst_trace_1",
    ]:
        train_2[col] = np.nan
        test_2[col] = np.nan

if dipole is not None:
    train_2 = train_2.merge(dipole, on="molecule_name", how="left")
    test_2 = test_2.merge(dipole, on="molecule_name", how="left")
else:
    train_2["X"] = np.nan
    train_2["Y"] = np.nan
    train_2["Z"] = np.nan
    test_2["X"] = np.nan
    test_2["Y"] = np.nan
    test_2["Z"] = np.nan

if potential_energy is not None:
    train_2 = train_2.merge(potential_energy, on="molecule_name", how="left")
    test_2 = test_2.merge(potential_energy, on="molecule_name", how="left")
else:
    train_2["potential_energy"] = np.nan
    test_2["potential_energy"] = np.nan

for c in ["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"]:
    train_2[c] = train_2[c].fillna(0.0)
    test_2[c] = test_2[c].fillna(0.0)

num_fill_zero = [
    "mcharge_0",
    "mcharge_1",
    "mst_mean_0",
    "mst_std_0",
    "mst_absmean_0",
    "mst_trace_0",
    "mst_mean_1",
    "mst_std_1",
    "mst_absmean_1",
    "mst_trace_1",
    "X",
    "Y",
    "Z",
    "potential_energy",
]
for c in num_fill_zero:
    if c in train_2.columns:
        train_2[c] = train_2[c].fillna(0.0).astype(np.float64)
    if c in test_2.columns:
        test_2[c] = test_2[c].fillna(0.0).astype(np.float64)

dx_tr = (train_2.x_0 - train_2.x_1).astype(np.float64)
dy_tr = (train_2.y_0 - train_2.y_1).astype(np.float64)
dz_tr = (train_2.z_0 - train_2.z_1).astype(np.float64)
dist_tr = np.sqrt(dx_tr * dx_tr + dy_tr * dy_tr + dz_tr * dz_tr)

dx_te = (test_2.x_0 - test_2.x_1).astype(np.float64)
dy_te = (test_2.y_0 - test_2.y_1).astype(np.float64)
dz_te = (test_2.z_0 - test_2.z_1).astype(np.float64)
dist_te = np.sqrt(dx_te * dx_te + dy_te * dy_te + dz_te * dz_te)

train_2["distance"] = dist_tr
test_2["distance"] = dist_te

eps = 1e-6
train_2["distance2"] = train_2["distance"] ** 2
test_2["distance2"] = test_2["distance"] ** 2
train_2["inv_distance"] = 1.0 / (train_2["distance"] + eps)
test_2["inv_distance"] = 1.0 / (test_2["distance"] + eps)

train_2["atom_pair"] = (
    train_2["atom_0"].astype(str) + "_" + train_2["atom_1"].astype(str)
)
test_2["atom_pair"] = test_2["atom_0"].astype(str) + "_" + test_2["atom_1"].astype(str)

train_2["mcharge_sum"] = train_2["mcharge_0"] + train_2["mcharge_1"]
test_2["mcharge_sum"] = test_2["mcharge_0"] + test_2["mcharge_1"]
train_2["mcharge_diff"] = train_2["mcharge_0"] - train_2["mcharge_1"]
test_2["mcharge_diff"] = test_2["mcharge_0"] - test_2["mcharge_1"]
train_2["mcharge_prod"] = train_2["mcharge_0"] * train_2["mcharge_1"]
test_2["mcharge_prod"] = test_2["mcharge_0"] * test_2["mcharge_1"]

train_2["mst_mean_diff"] = train_2["mst_mean_0"] - train_2["mst_mean_1"]
test_2["mst_mean_diff"] = test_2["mst_mean_0"] - test_2["mst_mean_1"]
train_2["mst_trace_diff"] = train_2["mst_trace_0"] - train_2["mst_trace_1"]
test_2["mst_trace_diff"] = test_2["mst_trace_0"] - test_2["mst_trace_1"]

train_2["dipole_mag"] = np.sqrt(
    train_2["X"] ** 2 + train_2["Y"] ** 2 + train_2["Z"] ** 2
)
test_2["dipole_mag"] = np.sqrt(test_2["X"] ** 2 + test_2["Y"] ** 2 + test_2["Z"] ** 2)

train_2["ux"] = dx_tr / (train_2["distance"] + eps)
train_2["uy"] = dy_tr / (train_2["distance"] + eps)
train_2["uz"] = dz_tr / (train_2["distance"] + eps)
test_2["ux"] = dx_te / (test_2["distance"] + eps)
test_2["uy"] = dy_te / (test_2["distance"] + eps)
test_2["uz"] = dz_te / (test_2["distance"] + eps)

train_2["dipole_proj"] = (
    train_2["X"] * train_2["ux"]
    + train_2["Y"] * train_2["uy"]
    + train_2["Z"] * train_2["uz"]
)
test_2["dipole_proj"] = (
    test_2["X"] * test_2["ux"] + test_2["Y"] * test_2["uy"] + test_2["Z"] * test_2["uz"]
)



## === cell 2
numeric_features = [
    "distance",
    "distance2",
    "inv_distance",
    "mcharge_0",
    "mcharge_1",
    "mcharge_sum",
    "mcharge_diff",
    "mcharge_prod",
    "mst_mean_0",
    "mst_std_0",
    "mst_absmean_0",
    "mst_trace_0",
    "mst_mean_1",
    "mst_std_1",
    "mst_absmean_1",
    "mst_trace_1",
    "mst_mean_diff",
    "mst_trace_diff",
    "dipole_mag",
    "dipole_proj",
    "potential_energy",
]
cat_features = ["type", "atom_0", "atom_1", "atom_pair"]

for c in numeric_features:
    train_2[c] = (
        train_2[c].astype(np.float64).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    )
    test_2[c] = test_2[c].astype(np.float64).replace([np.inf, -np.inf], 0.0).fillna(0.0)

for c in cat_features:
    train_2[c] = train_2[c].astype(str).fillna("NA")
    test_2[c] = test_2[c].astype(str).fillna("NA")

y_train = train_2["scalar_coupling_constant"].astype(np.float64).values
global_y_mean = float(np.mean(y_train))

type_pair_counts = train_2.groupby(["type", "atom_pair"]).size()
inv_type_pair_weight = (
    train_2.set_index(["type", "atom_pair"])
    .index.map(lambda k: 1.0 / float(type_pair_counts.loc[k]))
    .astype(np.float64)
    .values
)


def make_pipe():
    preprocess = ColumnTransformer(
        transformers=[
            (
                "cats",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    min_frequency=10,
                ),
                cat_features,
            ),
            ("nums", StandardScaler(with_mean=False), numeric_features),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    model = Ridge(alpha=10.0, fit_intercept=True, solver="sag")
    return Pipeline(steps=[("prep", preprocess), ("ridge", model)])


test_pred = np.zeros(len(test_2), dtype=np.float64)

types_train = train_2["type"].values
types_test = test_2["type"].values

for t in pd.unique(types_train):
    tr_idx = types_train == t
    te_idx = types_test == t
    if not np.any(te_idx):
        continue
    if not np.any(tr_idx):
        test_pred[te_idx] = global_y_mean
        continue

    y_t = y_train[tr_idx]
    mu = float(np.mean(y_t))
    sd = float(np.std(y_t))
    if not np.isfinite(sd) or sd < 1e-12:
        test_pred[te_idx] = mu
        continue

    Xtr_df = train_2.loc[tr_idx, cat_features + numeric_features]
    Xte_df = test_2.loc[te_idx, cat_features + numeric_features]

    w_t = inv_type_pair_weight[tr_idx]
    w_t = w_t / np.mean(w_t)  # keep weight scale stable

    pipe = make_pipe()
    pipe.fit(Xtr_df, (y_t - mu) / sd, ridge__sample_weight=w_t)
    yhat_std = pipe.predict(Xte_df)
    test_pred[te_idx] = yhat_std * sd + mu

test_2["scalar_coupling_constant"] = test_pred

submission = test_2[["id", "scalar_coupling_constant"]].copy()
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Unique types in test:", test_2["type"].nunique())
