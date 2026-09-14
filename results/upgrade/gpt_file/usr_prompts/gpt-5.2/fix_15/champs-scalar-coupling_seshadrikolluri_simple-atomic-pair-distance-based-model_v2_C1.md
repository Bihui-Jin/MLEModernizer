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

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_PATH = "/kaggle/data/champs-scalar-coupling"
train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
structures_path = os.path.join(BASE_PATH, "structures.csv")

train_dtypes = {
    "id": np.int32,
    "molecule_name": "category",
    "atom_index_0": np.int16,
    "atom_index_1": np.int16,
    "type": "category",
    "scalar_coupling_constant": np.float64,
}
test_dtypes = {
    "id": np.int32,
    "molecule_name": "category",
    "atom_index_0": np.int16,
    "atom_index_1": np.int16,
    "type": "category",
}
structures_dtypes = {
    "molecule_name": "category",
    "atom_index": np.int16,
    "atom": "category",
    "x": np.float32,
    "y": np.float32,
    "z": np.float32,
}

train = pd.read_csv(train_path, dtype=train_dtypes, usecols=list(train_dtypes.keys()))
test = pd.read_csv(test_path, dtype=test_dtypes, usecols=list(test_dtypes.keys()))
structures = pd.read_csv(
    structures_path, dtype=structures_dtypes, usecols=list(structures_dtypes.keys())
)

mulliken_path = os.path.join(BASE_PATH, "mulliken_charges.csv")
mst_path = os.path.join(BASE_PATH, "magnetic_shielding_tensors.csv")
dipole_path = os.path.join(BASE_PATH, "dipole_moments.csv")
pe_path = os.path.join(BASE_PATH, "potential_energy.csv")

mulliken = (
    pd.read_csv(
        mulliken_path,
        usecols=["molecule_name", "atom_index", "mulliken_charge"],
        dtype={
            "molecule_name": "category",
            "atom_index": np.int16,
            "mulliken_charge": np.float32,
        },
    )
    if os.path.exists(mulliken_path)
    else None
)

mst = (
    pd.read_csv(
        mst_path,
        usecols=[
            "molecule_name",
            "atom_index",
            "XX",
            "YX",
            "ZX",
            "XY",
            "YY",
            "ZY",
            "XZ",
            "YZ",
            "ZZ",
        ],
        dtype={
            "molecule_name": "category",
            "atom_index": np.int16,
            "XX": np.float32,
            "YX": np.float32,
            "ZX": np.float32,
            "XY": np.float32,
            "YY": np.float32,
            "ZY": np.float32,
            "XZ": np.float32,
            "YZ": np.float32,
            "ZZ": np.float32,
        },
    )
    if os.path.exists(mst_path)
    else None
)

dipole = (
    pd.read_csv(
        dipole_path,
        usecols=["molecule_name", "X", "Y", "Z"],
        dtype={
            "molecule_name": "category",
            "X": np.float32,
            "Y": np.float32,
            "Z": np.float32,
        },
    )
    if os.path.exists(dipole_path)
    else None
)

potential_energy = (
    pd.read_csv(
        pe_path,
        usecols=["molecule_name", "potential_energy"],
        dtype={"molecule_name": "category", "potential_energy": np.float32},
    )
    if os.path.exists(pe_path)
    else None
)



## === cell 1
s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x_0",
        "y": "y_0",
        "z": "z_0",
    }
)[["molecule_name", "atom_index_0", "atom_0", "x_0", "y_0", "z_0"]].set_index(
    ["molecule_name", "atom_index_0"]
)

s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x_1",
        "y": "y_1",
        "z": "z_1",
    }
)[["molecule_name", "atom_index_1", "atom_1", "x_1", "y_1", "z_1"]].set_index(
    ["molecule_name", "atom_index_1"]
)

train_2 = train.copy()
test_2 = test.copy()

train_2 = train_2.join(s0, on=["molecule_name", "atom_index_0"])
train_2 = train_2.join(s1, on=["molecule_name", "atom_index_1"])
test_2 = test_2.join(s0, on=["molecule_name", "atom_index_0"])
test_2 = test_2.join(s1, on=["molecule_name", "atom_index_1"])

if mulliken is not None:
    m0 = mulliken.rename(
        columns={"atom_index": "atom_index_0", "mulliken_charge": "mcharge_0"}
    )[["molecule_name", "atom_index_0", "mcharge_0"]].set_index(
        ["molecule_name", "atom_index_0"]
    )
    m1 = mulliken.rename(
        columns={"atom_index": "atom_index_1", "mulliken_charge": "mcharge_1"}
    )[["molecule_name", "atom_index_1", "mcharge_1"]].set_index(
        ["molecule_name", "atom_index_1"]
    )

    train_2 = train_2.join(m0, on=["molecule_name", "atom_index_0"])
    train_2 = train_2.join(m1, on=["molecule_name", "atom_index_1"])
    test_2 = test_2.join(m0, on=["molecule_name", "atom_index_0"])
    test_2 = test_2.join(m1, on=["molecule_name", "atom_index_1"])
else:
    train_2["mcharge_0"] = np.nan
    train_2["mcharge_1"] = np.nan
    test_2["mcharge_0"] = np.nan
    test_2["mcharge_1"] = np.nan

if mst is not None:
    tensor_cols = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
    vals = mst[tensor_cols].to_numpy(dtype=np.float64, copy=False)
    mst_ag = pd.DataFrame(
        {
            "molecule_name": mst["molecule_name"].to_numpy(copy=False),
            "atom_index": mst["atom_index"].to_numpy(copy=False),
            "mst_mean": vals.mean(axis=1),
            "mst_std": vals.std(axis=1),
            "mst_absmean": np.abs(vals).mean(axis=1),
            "mst_trace": mst["XX"].astype(np.float64).to_numpy(copy=False)
            + mst["YY"].astype(np.float64).to_numpy(copy=False)
            + mst["ZZ"].astype(np.float64).to_numpy(copy=False),
        }
    )
    mst0 = mst_ag.rename(columns={"atom_index": "atom_index_0"}).set_index(
        ["molecule_name", "atom_index_0"]
    )
    mst0.columns = [c + "_0" for c in mst0.columns]
    mst1 = mst_ag.rename(columns={"atom_index": "atom_index_1"}).set_index(
        ["molecule_name", "atom_index_1"]
    )
    mst1.columns = [c + "_1" for c in mst1.columns]

    train_2 = train_2.join(mst0, on=["molecule_name", "atom_index_0"])
    train_2 = train_2.join(mst1, on=["molecule_name", "atom_index_1"])
    test_2 = test_2.join(mst0, on=["molecule_name", "atom_index_0"])
    test_2 = test_2.join(mst1, on=["molecule_name", "atom_index_1"])
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
    dip = dipole.set_index("molecule_name")
    train_2 = train_2.join(dip, on="molecule_name")
    test_2 = test_2.join(dip, on="molecule_name")
else:
    train_2["X"] = np.nan
    train_2["Y"] = np.nan
    train_2["Z"] = np.nan
    test_2["X"] = np.nan
    test_2["Y"] = np.nan
    test_2["Z"] = np.nan

if potential_energy is not None:
    pe = potential_energy.set_index("molecule_name")
    train_2 = train_2.join(pe, on="molecule_name")
    test_2 = test_2.join(pe, on="molecule_name")
else:
    train_2["potential_energy"] = np.nan
    test_2["potential_energy"] = np.nan

mol_centroid = (
    structures.groupby("molecule_name", sort=False)[["x", "y", "z"]]
    .mean()
    .rename(columns={"x": "cx", "y": "cy", "z": "cz"})
)
train_2 = train_2.join(mol_centroid, on="molecule_name")
test_2 = test_2.join(mol_centroid, on="molecule_name")

for c in ["x_0", "y_0", "z_0", "x_1", "y_1", "z_1", "cx", "cy", "cz"]:
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
        train_2[c] = train_2[c].fillna(0.0)
    if c in test_2.columns:
        test_2[c] = test_2[c].fillna(0.0)

x0_tr = train_2["x_0"].to_numpy(dtype=np.float64, copy=False)
y0_tr = train_2["y_0"].to_numpy(dtype=np.float64, copy=False)
z0_tr = train_2["z_0"].to_numpy(dtype=np.float64, copy=False)
x1_tr = train_2["x_1"].to_numpy(dtype=np.float64, copy=False)
y1_tr = train_2["y_1"].to_numpy(dtype=np.float64, copy=False)
z1_tr = train_2["z_1"].to_numpy(dtype=np.float64, copy=False)

dx_tr = x0_tr - x1_tr
dy_tr = y0_tr - y1_tr
dz_tr = z0_tr - z1_tr
dist_tr = np.sqrt(dx_tr * dx_tr + dy_tr * dy_tr + dz_tr * dz_tr)

x0_te = test_2["x_0"].to_numpy(dtype=np.float64, copy=False)
y0_te = test_2["y_0"].to_numpy(dtype=np.float64, copy=False)
z0_te = test_2["z_0"].to_numpy(dtype=np.float64, copy=False)
x1_te = test_2["x_1"].to_numpy(dtype=np.float64, copy=False)
y1_te = test_2["y_1"].to_numpy(dtype=np.float64, copy=False)
z1_te = test_2["z_1"].to_numpy(dtype=np.float64, copy=False)

dx_te = x0_te - x1_te
dy_te = y0_te - y1_te
dz_te = z0_te - z1_te
dist_te = np.sqrt(dx_te * dx_te + dy_te * dy_te + dz_te * dz_te)

train_2["distance"] = dist_tr
test_2["distance"] = dist_te

eps = 1e-6
train_2["distance2"] = train_2["distance"] ** 2
test_2["distance2"] = test_2["distance"] ** 2
train_2["inv_distance"] = 1.0 / (train_2["distance"] + eps)
test_2["inv_distance"] = 1.0 / (test_2["distance"] + eps)

train_2["atom_pair"] = (
    train_2["atom_0"].astype("string") + "_" + train_2["atom_1"].astype("string")
).astype("string")
test_2["atom_pair"] = (
    test_2["atom_0"].astype("string") + "_" + test_2["atom_1"].astype("string")
).astype("string")

train_2["mcharge_sum"] = train_2["mcharge_0"].astype(np.float64) + train_2[
    "mcharge_1"
].astype(np.float64)
test_2["mcharge_sum"] = test_2["mcharge_0"].astype(np.float64) + test_2[
    "mcharge_1"
].astype(np.float64)
train_2["mcharge_diff"] = train_2["mcharge_0"].astype(np.float64) - train_2[
    "mcharge_1"
].astype(np.float64)
test_2["mcharge_diff"] = test_2["mcharge_0"].astype(np.float64) - test_2[
    "mcharge_1"
].astype(np.float64)
train_2["mcharge_prod"] = train_2["mcharge_0"].astype(np.float64) * train_2[
    "mcharge_1"
].astype(np.float64)
test_2["mcharge_prod"] = test_2["mcharge_0"].astype(np.float64) * test_2[
    "mcharge_1"
].astype(np.float64)

train_2["mst_mean_diff"] = train_2["mst_mean_0"].astype(np.float64) - train_2[
    "mst_mean_1"
].astype(np.float64)
test_2["mst_mean_diff"] = test_2["mst_mean_0"].astype(np.float64) - test_2[
    "mst_mean_1"
].astype(np.float64)
train_2["mst_trace_diff"] = train_2["mst_trace_0"].astype(np.float64) - train_2[
    "mst_trace_1"
].astype(np.float64)
test_2["mst_trace_diff"] = test_2["mst_trace_0"].astype(np.float64) - test_2[
    "mst_trace_1"
].astype(np.float64)

train_2["dipole_mag"] = np.sqrt(
    train_2["X"].astype(np.float64) ** 2
    + train_2["Y"].astype(np.float64) ** 2
    + train_2["Z"].astype(np.float64) ** 2
)
test_2["dipole_mag"] = np.sqrt(
    test_2["X"].astype(np.float64) ** 2
    + test_2["Y"].astype(np.float64) ** 2
    + test_2["Z"].astype(np.float64) ** 2
)

train_2["ux"] = dx_tr / (train_2["distance"] + eps)
train_2["uy"] = dy_tr / (train_2["distance"] + eps)
train_2["uz"] = dz_tr / (train_2["distance"] + eps)
test_2["ux"] = dx_te / (test_2["distance"] + eps)
test_2["uy"] = dy_te / (test_2["distance"] + eps)
test_2["uz"] = dz_te / (test_2["distance"] + eps)

train_2["dipole_proj"] = (
    train_2["X"].astype(np.float64) * train_2["ux"]
    + train_2["Y"].astype(np.float64) * train_2["uy"]
    + train_2["Z"].astype(np.float64) * train_2["uz"]
)
test_2["dipole_proj"] = (
    test_2["X"].astype(np.float64) * test_2["ux"]
    + test_2["Y"].astype(np.float64) * test_2["uy"]
    + test_2["Z"].astype(np.float64) * test_2["uz"]
)

cx_tr = train_2["cx"].to_numpy(dtype=np.float64, copy=False)
cy_tr = train_2["cy"].to_numpy(dtype=np.float64, copy=False)
cz_tr = train_2["cz"].to_numpy(dtype=np.float64, copy=False)
cx_te = test_2["cx"].to_numpy(dtype=np.float64, copy=False)
cy_te = test_2["cy"].to_numpy(dtype=np.float64, copy=False)
cz_te = test_2["cz"].to_numpy(dtype=np.float64, copy=False)

d0_tr = np.sqrt((x0_tr - cx_tr) ** 2 + (y0_tr - cy_tr) ** 2 + (z0_tr - cz_tr) ** 2)
d1_tr = np.sqrt((x1_tr - cx_tr) ** 2 + (y1_tr - cy_tr) ** 2 + (z1_tr - cz_tr) ** 2)
d0_te = np.sqrt((x0_te - cx_te) ** 2 + (y0_te - cy_te) ** 2 + (z0_te - cz_te) ** 2)
d1_te = np.sqrt((x1_te - cx_te) ** 2 + (y1_te - cy_te) ** 2 + (z1_te - cz_te) ** 2)

train_2["dcentroid_0"] = d0_tr
train_2["dcentroid_1"] = d1_tr
test_2["dcentroid_0"] = d0_te
test_2["dcentroid_1"] = d1_te
train_2["dcentroid_sum"] = train_2["dcentroid_0"] + train_2["dcentroid_1"]
test_2["dcentroid_sum"] = test_2["dcentroid_0"] + test_2["dcentroid_1"]
train_2["dcentroid_diff"] = train_2["dcentroid_0"] - train_2["dcentroid_1"]
test_2["dcentroid_diff"] = test_2["dcentroid_0"] - test_2["dcentroid_1"]



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
    "dcentroid_0",
    "dcentroid_1",
    "dcentroid_sum",
    "dcentroid_diff",
]
cat_features = ["type", "atom_0", "atom_1", "atom_pair"]

for c in numeric_features:
    tr = train_2[c].to_numpy(dtype=np.float64, copy=False)
    te = test_2[c].to_numpy(dtype=np.float64, copy=False)
    tr = np.nan_to_num(tr, nan=0.0, posinf=0.0, neginf=0.0)
    te = np.nan_to_num(te, nan=0.0, posinf=0.0, neginf=0.0)
    train_2[c] = tr
    test_2[c] = te

for c in cat_features:
    train_2[c] = train_2[c].astype("string").fillna("NA")
    test_2[c] = test_2[c].astype("string").fillna("NA")

y_train = train_2["scalar_coupling_constant"].astype(np.float64).to_numpy(copy=False)
global_y_mean = float(np.mean(y_train))

type_pair_counts = (
    train_2.groupby(["type", "atom_pair"], sort=False)
    .size()
    .rename("tp_count")
    .reset_index()
)
train_key = train_2[["type", "atom_pair"]].copy()
inv_type_pair_weight = train_key.merge(
    type_pair_counts, on=["type", "atom_pair"], how="left", sort=False
)["tp_count"].to_numpy(dtype=np.float64, copy=False)
inv_type_pair_weight = 1.0 / inv_type_pair_weight


def make_pipe(min_freq=None):
    ohe_kwargs = dict(handle_unknown="ignore", sparse_output=True)
    if min_freq is not None and min_freq >= 2:
        ohe_kwargs["min_frequency"] = int(min_freq)

    preprocess = ColumnTransformer(
        transformers=[
            ("cats", OneHotEncoder(**ohe_kwargs), cat_features),
            ("nums", StandardScaler(with_mean=False), numeric_features),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    model = Ridge(alpha=10.0, fit_intercept=True, solver="sag")
    return Pipeline(steps=[("prep", preprocess), ("ridge", model)])


test_pred = np.zeros(len(test_2), dtype=np.float64)

train_type_to_idx = train_2.groupby("type", sort=False).indices
test_type_to_idx = test_2.groupby("type", sort=False).indices

types_unique_train = list(train_type_to_idx.keys())

feature_cols = cat_features + numeric_features

for t in types_unique_train:
    te_idx = test_type_to_idx.get(t)
    if te_idx is None or len(te_idx) == 0:
        continue
    tr_idx = train_type_to_idx.get(t)
    te_idx = np.asarray(te_idx, dtype=np.int64)

    if tr_idx is None or len(tr_idx) == 0:
        test_pred[te_idx] = global_y_mean
        continue

    tr_idx = np.asarray(tr_idx, dtype=np.int64)
    y_t = y_train[tr_idx]
    mu = float(np.mean(y_t))
    sd = float(np.std(y_t))
    if not np.isfinite(sd) or sd < 1e-12:
        test_pred[te_idx] = mu
        continue

    Xtr_df = train_2[feature_cols].take(tr_idx)
    Xte_df = test_2[feature_cols].take(te_idx)

    w_t = inv_type_pair_weight[tr_idx]
    w_t = w_t / np.mean(w_t)  # keep weight scale stable

    n_tr = len(tr_idx)
    if n_tr < 20000:
        min_freq = None
    else:
        min_freq = 10

    pipe = make_pipe(min_freq=min_freq)
    pipe.fit(Xtr_df, (y_t - mu) / sd, ridge__sample_weight=w_t)
    yhat_std = pipe.predict(Xte_df)
    yhat = yhat_std * sd + mu

    lo, hi = np.quantile(y_t, [0.0001, 0.9999])
    yhat = np.clip(yhat, lo, hi)

    test_pred[te_idx] = yhat

test_2["scalar_coupling_constant"] = test_pred

submission = test_2[["id", "scalar_coupling_constant"]].copy()
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Unique types in test:", test_2["type"].nunique())
