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
from sklearn import preprocessing, ensemble, metrics

DATA_DIR = "/kaggle/input/champs-scalar-coupling"

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

train_dtypes = {
    "id": np.int32,
    "molecule_name": "category",
    "atom_index_0": np.int16,
    "atom_index_1": np.int16,
    "type": "category",
    "scalar_coupling_constant": np.float32,
}
test_dtypes = {
    "id": np.int32,
    "molecule_name": "category",
    "atom_index_0": np.int16,
    "atom_index_1": np.int16,
    "type": "category",
}
sub_dtypes = {"id": np.int32, "scalar_coupling_constant": np.float32}

train = pd.read_csv(f"{DATA_DIR}/train.csv", dtype=train_dtypes)
test = pd.read_csv(f"{DATA_DIR}/test.csv", dtype=test_dtypes)
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv", dtype=sub_dtypes)
print(train.shape, test.shape, sub.shape)

train_type_str = train["type"].astype(str)
test_type_str = test["type"].astype(str)

train["atom1"] = train_type_str.str[2].astype("category")
train["atom2"] = train_type_str.str[3].astype("category")
test["atom1"] = test_type_str.str[2].astype("category")
test["atom2"] = test_type_str.str[3].astype("category")

for i in range(4):
    lbl = preprocessing.LabelEncoder()
    all_chars = pd.concat(
        [train_type_str.str[i], test_type_str.str[i]], axis=0, ignore_index=True
    )
    lbl.fit(all_chars.values)
    train[f"type{i}"] = lbl.transform(train_type_str.str[i].values).astype(np.int16)
    test[f"type{i}"] = lbl.transform(test_type_str.str[i].values).astype(np.int16)

structures = pd.read_csv(
    f"{DATA_DIR}/structures.csv",
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
)

structures0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
        "atom": "atom1",
    }
)
train = pd.merge(
    train, structures0, how="left", on=["molecule_name", "atom_index_0", "atom1"]
)
test = pd.merge(
    test, structures0, how="left", on=["molecule_name", "atom_index_0", "atom1"]
)

structures1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
        "atom": "atom2",
    }
)
train = pd.merge(
    train, structures1, how="left", on=["molecule_name", "atom_index_1", "atom2"]
)
test = pd.merge(
    test, structures1, how="left", on=["molecule_name", "atom_index_1", "atom2"]
)
del structures0, structures1

mull = pd.read_csv(
    f"{DATA_DIR}/mulliken_charges.csv",
    usecols=["molecule_name", "atom_index", "mulliken_charge"],
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "mulliken_charge": np.float32,
    },
)
mull0 = mull.rename(columns={"atom_index": "atom_index_0", "mulliken_charge": "mull0"})
mull1 = mull.rename(columns={"atom_index": "atom_index_1", "mulliken_charge": "mull1"})

train = train.merge(
    mull0[["molecule_name", "atom_index_0", "mull0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
test = test.merge(
    mull0[["molecule_name", "atom_index_0", "mull0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
train = train.merge(
    mull1[["molecule_name", "atom_index_1", "mull1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)
test = test.merge(
    mull1[["molecule_name", "atom_index_1", "mull1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)
del mull, mull0, mull1

mst_cols = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
mst = pd.read_csv(
    f"{DATA_DIR}/magnetic_shielding_tensors.csv",
    usecols=["molecule_name", "atom_index"] + mst_cols,
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        **{c: np.float32 for c in mst_cols},
    },
)
mst0 = mst.rename(columns={"atom_index": "atom_index_0"})
mst1 = mst.rename(columns={"atom_index": "atom_index_1"})
mst0 = mst0[["molecule_name", "atom_index_0"] + mst_cols].rename(
    columns={c: f"mst0_{c}" for c in mst_cols}
)
mst1 = mst1[["molecule_name", "atom_index_1"] + mst_cols].rename(
    columns={c: f"mst1_{c}" for c in mst_cols}
)

train = train.merge(mst0, on=["molecule_name", "atom_index_0"], how="left")
test = test.merge(mst0, on=["molecule_name", "atom_index_0"], how="left")
train = train.merge(mst1, on=["molecule_name", "atom_index_1"], how="left")
test = test.merge(mst1, on=["molecule_name", "atom_index_1"], how="left")
del mst, mst0, mst1

dip = pd.read_csv(
    f"{DATA_DIR}/dipole_moments.csv",
    usecols=["molecule_name", "X", "Y", "Z"],
    dtype={
        "molecule_name": "category",
        "X": np.float32,
        "Y": np.float32,
        "Z": np.float32,
    },
).rename(columns={"X": "dip_X", "Y": "dip_Y", "Z": "dip_Z"})

pe = pd.read_csv(
    f"{DATA_DIR}/potential_energy.csv",
    usecols=["molecule_name", "potential_energy"],
    dtype={"molecule_name": "category", "potential_energy": np.float32},
)

train = train.merge(dip, on="molecule_name", how="left").merge(
    pe, on="molecule_name", how="left"
)
test = test.merge(dip, on="molecule_name", how="left").merge(
    pe, on="molecule_name", how="left"
)
del dip, pe

train_mols = train["molecule_name"].unique()
train_mol_set = set(train_mols.tolist())

scc = pd.read_csv(
    f"{DATA_DIR}/scalar_coupling_contributions.csv",
    usecols=["molecule_name", "type", "fc", "sd", "pso", "dso"],
    dtype={
        "molecule_name": "category",
        "type": "category",
        "fc": np.float32,
        "sd": np.float32,
        "pso": np.float32,
        "dso": np.float32,
    },
)
scc = scc[scc["molecule_name"].isin(train_mol_set)].copy()
scc["contrib_sum"] = (scc["fc"] + scc["sd"] + scc["pso"] + scc["dso"]).astype(
    np.float32
)

moltype_contrib_agg = scc.groupby(["molecule_name", "type"], sort=False)[
    ["fc", "sd", "pso", "dso", "contrib_sum"]
].agg(["mean", "std"])
moltype_contrib_agg.columns = [
    f"moltype_{a}_{b}" for a, b in moltype_contrib_agg.columns.to_flat_index()
]
moltype_contrib_agg = moltype_contrib_agg.reset_index()

train = train.merge(moltype_contrib_agg, on=["molecule_name", "type"], how="left")
test = test.merge(moltype_contrib_agg, on=["molecule_name", "type"], how="left")
del scc, moltype_contrib_agg, train_mols, train_mol_set

print(train.shape, test.shape, sub.shape)



## === cell 1

train_p0 = train[["x0", "y0", "z0"]].to_numpy(dtype=np.float32, copy=False)
train_p1 = train[["x1", "y1", "z1"]].to_numpy(dtype=np.float32, copy=False)
test_p0 = test[["x0", "y0", "z0"]].to_numpy(dtype=np.float32, copy=False)
test_p1 = test[["x1", "y1", "z1"]].to_numpy(dtype=np.float32, copy=False)

train_dx = train_p0[:, 0] - train_p1[:, 0]
train_dy = train_p0[:, 1] - train_p1[:, 1]
train_dz = train_p0[:, 2] - train_p1[:, 2]
test_dx = test_p0[:, 0] - test_p1[:, 0]
test_dy = test_p0[:, 1] - test_p1[:, 1]
test_dz = test_p0[:, 2] - test_p1[:, 2]

train["dx"] = train_dx
train["dy"] = train_dy
train["dz"] = train_dz
test["dx"] = test_dx
test["dy"] = test_dy
test["dz"] = test_dz

train_dist = np.sqrt(
    train_dx * train_dx + train_dy * train_dy + train_dz * train_dz, dtype=np.float32
)
test_dist = np.sqrt(
    test_dx * test_dx + test_dy * test_dy + test_dz * test_dz, dtype=np.float32
)
train["dist"] = train_dist
test["dist"] = test_dist

eps = 1e-12
train["inv_dist"] = 1.0 / (train_dist.astype(np.float64) + eps)
test["inv_dist"] = 1.0 / (test_dist.astype(np.float64) + eps)

train["log1p_dist"] = np.log1p(train_dist.astype(np.float64))
test["log1p_dist"] = np.log1p(test_dist.astype(np.float64))

type_mean_dist = train.groupby("type")["dist"].mean()
global_mean_dist = float(train["dist"].mean())

train_type_mean = (
    train["type"].map(type_mean_dist).astype("float32").fillna(global_mean_dist)
)
test_type_mean = (
    test["type"].map(type_mean_dist).astype("float32").fillna(global_mean_dist)
)

train["dist_to_type_mean"] = train["dist"] / train_type_mean
test["dist_to_type_mean"] = test["dist"] / test_type_mean

train_mid = 0.5 * (
    train[["x0", "y0", "z0"]].to_numpy(np.float32, copy=False)
    + train[["x1", "y1", "z1"]].to_numpy(np.float32, copy=False)
)
test_mid = 0.5 * (
    test[["x0", "y0", "z0"]].to_numpy(np.float32, copy=False)
    + test[["x1", "y1", "z1"]].to_numpy(np.float32, copy=False)
)

train_mid_df = pd.DataFrame(train_mid, columns=["mx", "my", "mz"], index=train.index)
test_mid_df = pd.DataFrame(test_mid, columns=["mx", "my", "mz"], index=test.index)

train_center = train_mid_df.groupby(train["molecule_name"], sort=False)[
    ["mx", "my", "mz"]
].transform("mean")
test_center = test_mid_df.groupby(test["molecule_name"], sort=False)[
    ["mx", "my", "mz"]
].transform("mean")

for df, center in ((train, train_center), (test, test_center)):
    cx = center["mx"].to_numpy(dtype=np.float32, copy=False)
    cy = center["my"].to_numpy(dtype=np.float32, copy=False)
    cz = center["mz"].to_numpy(dtype=np.float32, copy=False)

    x0 = df["x0"].to_numpy(dtype=np.float32, copy=False)
    y0 = df["y0"].to_numpy(dtype=np.float32, copy=False)
    z0 = df["z0"].to_numpy(dtype=np.float32, copy=False)
    x1 = df["x1"].to_numpy(dtype=np.float32, copy=False)
    y1 = df["y1"].to_numpy(dtype=np.float32, copy=False)
    z1 = df["z1"].to_numpy(dtype=np.float32, copy=False)

    r0 = np.sqrt((x0 - cx) ** 2 + (y0 - cy) ** 2 + (z0 - cz) ** 2, dtype=np.float32)
    r1 = np.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2 + (z1 - cz) ** 2, dtype=np.float32)

    df["r0"] = r0
    df["r1"] = r1
    df["r0_plus_r1"] = r0 + r1
    df["r0_minus_r1"] = r0 - r1

train_r0_vec = train_p0 - train_center.to_numpy(dtype=np.float32, copy=False)
train_r1_vec = train_p1 - train_center.to_numpy(dtype=np.float32, copy=False)
test_r0_vec = test_p0 - test_center.to_numpy(dtype=np.float32, copy=False)
test_r1_vec = test_p1 - test_center.to_numpy(dtype=np.float32, copy=False)

train_bond = np.vstack([train_dx, train_dy, train_dz]).T.astype(np.float32, copy=False)
test_bond = np.vstack([test_dx, test_dy, test_dz]).T.astype(np.float32, copy=False)


def safe_cos(a, b, eps=1e-12):
    a64 = a.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    an = np.sqrt((a64 * a64).sum(axis=1)) + eps
    bn = np.sqrt((b64 * b64).sum(axis=1)) + eps
    return (a64 * b64).sum(axis=1) / (an * bn)


train["cos_bond_r0"] = safe_cos(train_bond, train_r0_vec, eps=eps)
train["cos_bond_r1"] = safe_cos(-train_bond, train_r1_vec, eps=eps)
test["cos_bond_r0"] = safe_cos(test_bond, test_r0_vec, eps=eps)
test["cos_bond_r1"] = safe_cos(-test_bond, test_r1_vec, eps=eps)

train["mull_diff"] = train["mull0"] - train["mull1"]
test["mull_diff"] = test["mull0"] - test["mull1"]
train["mull_absdiff"] = np.abs(train["mull_diff"].to_numpy(copy=False))
test["mull_absdiff"] = np.abs(test["mull_diff"].to_numpy(copy=False))
train["mull_prod"] = train["mull0"] * train["mull1"]
test["mull_prod"] = test["mull0"] * test["mull1"]

train["mull_sum"] = train["mull0"] + train["mull1"]
test["mull_sum"] = test["mull0"] + test["mull1"]
train["mull_diff2"] = train["mull_diff"].to_numpy(copy=False) ** 2
test["mull_diff2"] = test["mull_diff"].to_numpy(copy=False) ** 2

for df in (train, test):
    df["mull_min"] = np.minimum(
        df["mull0"].to_numpy(copy=False), df["mull1"].to_numpy(copy=False)
    )
    df["mull_max"] = np.maximum(
        df["mull0"].to_numpy(copy=False), df["mull1"].to_numpy(copy=False)
    )

for c in ["dip_X", "dip_Y", "dip_Z"]:
    train[f"{c}_dist"] = train[c] * train["dist"]
    test[f"{c}_dist"] = test[c] * test["dist"]
    train[f"{c}_inv_dist"] = train[c] * train["inv_dist"]
    test[f"{c}_inv_dist"] = test[c] * test["inv_dist"]

for c in ["XX", "YY", "ZZ"]:
    a0 = f"mst0_{c}"
    a1 = f"mst1_{c}"
    train[f"{c}_diff"] = train[a0] - train[a1]
    test[f"{c}_diff"] = test[a0] - test[a1]
    train[f"{c}_absdiff"] = np.abs(train[f"{c}_diff"].to_numpy(copy=False))
    test[f"{c}_absdiff"] = np.abs(test[f"{c}_diff"].to_numpy(copy=False))
    train[f"{c}_prod"] = train[a0] * train[a1]
    test[f"{c}_prod"] = test[a0] * test[a1]
    train[f"{c}_sum"] = train[a0] + train[a1]
    test[f"{c}_sum"] = test[a0] + test[a1]
    train[f"{c}_diff2"] = train[f"{c}_diff"].to_numpy(copy=False) ** 2
    test[f"{c}_diff2"] = test[f"{c}_diff"].to_numpy(copy=False) ** 2
    train[f"{c}_min"] = np.minimum(
        train[a0].to_numpy(copy=False), train[a1].to_numpy(copy=False)
    )
    train[f"{c}_max"] = np.maximum(
        train[a0].to_numpy(copy=False), train[a1].to_numpy(copy=False)
    )
    test[f"{c}_min"] = np.minimum(
        test[a0].to_numpy(copy=False), test[a1].to_numpy(copy=False)
    )
    test[f"{c}_max"] = np.maximum(
        test[a0].to_numpy(copy=False), test[a1].to_numpy(copy=False)
    )

dist_agg = (
    train.groupby(["molecule_name", "type"], sort=False)["dist"]
    .agg(["mean", "std", "count"])
    .rename(
        columns={
            "mean": "moltype_dist_mean",
            "std": "moltype_dist_std",
            "count": "moltype_dist_count",
        }
    )
)
train = train.join(dist_agg, on=["molecule_name", "type"])
test = test.join(dist_agg, on=["molecule_name", "type"])
del dist_agg

mol_atom_cnt = structures.groupby("molecule_name", sort=False)["atom_index"].max() + 1
mol_atom_cnt = mol_atom_cnt.rename("mol_n_atoms")
train = train.join(mol_atom_cnt, on="molecule_name")
test = test.join(mol_atom_cnt, on="molecule_name")

num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
medians = train[num_cols].median(numeric_only=True)

train[num_cols] = train[num_cols].replace([np.inf, -np.inf], np.nan)
test_num_cols = [c for c in num_cols if c in test.columns]
test[test_num_cols] = test[test_num_cols].replace([np.inf, -np.inf], np.nan)

train[num_cols] = train[num_cols].fillna(medians)
test[test_num_cols] = test[test_num_cols].fillna(medians.reindex(test_num_cols))

assert (
    not train[num_cols].isna().any().any()
), "NaNs remain in train numeric columns after imputation."
assert (
    not test[test_num_cols].isna().any().any()
), "NaNs remain in test numeric columns after imputation."

del (
    mol_atom_cnt,
    train_center,
    test_center,
    train_mid,
    test_mid,
    train_mid_df,
    test_mid_df,
)



## === cell 2
col = [
    c
    for c in train.columns
    if c
    not in [
        "id",
        "molecule_name",
        "scalar_coupling_constant",
        "type",
        "atom1",
        "atom2",
        "atom_index_0",
        "atom_index_1",
    ]
]

base_reg_params = dict(n_jobs=-1, n_estimators=60, random_state=4)

PER_TYPE_TRAIN = 300_000  # keep same cap to preserve runtime/approach
SEED = 99

rng = np.random.RandomState(SEED)
test_pred = np.zeros(len(test), dtype=np.float64)

val_scores = []
types = sorted(train["type"].astype(str).unique().tolist())
print("Training per-type models for", len(types), "types:", types)

for t in types:
    tr_full = train[train["type"].astype(str) == t]

    if len(tr_full) > PER_TYPE_TRAIN:
        tr_t = tr_full.sample(n=PER_TYPE_TRAIN, random_state=SEED).reset_index(
            drop=True
        )
    else:
        tr_t = tr_full.reset_index(drop=True)

    te_mask = test["type"].astype(str).values == t

    mols = pd.unique(tr_t["molecule_name"].values)
    rng.shuffle(mols)
    n_val_mols = max(1, int(0.2 * len(mols)))
    val_mols = set(mols[:n_val_mols])

    is_val = tr_t["molecule_name"].isin(val_mols).values
    x_tr = tr_t.loc[~is_val, col]
    y_tr = tr_t.loc[~is_val, "scalar_coupling_constant"].astype(np.float64)

    x_va = tr_t.loc[is_val, col]
    y_va = tr_t.loc[is_val, "scalar_coupling_constant"].astype(np.float64)

    if x_tr.isna().any().any() or x_va.isna().any().any():
        fill_vals = x_tr.median(numeric_only=True)
        x_tr = x_tr.fillna(fill_vals)
        x_va = x_va.fillna(fill_vals.reindex(x_va.columns))

    reg = ensemble.ExtraTreesRegressor(**base_reg_params)
    reg.fit(x_tr, y_tr)

    va_pred = reg.predict(x_va)
    mae = metrics.mean_absolute_error(y_va, va_pred)
    val_scores.append((t, float(np.log(mae))))
    print(
        f"type={t:>4s}  n_tr={x_tr.shape[0]:>7d}  n_va={x_va.shape[0]:>7d}  log(MAE)={np.log(mae):.5f}"
    )

    if np.any(te_mask):
        x_all = tr_t[col]
        if x_all.isna().any().any():
            fill_vals = x_all.median(numeric_only=True)
            x_all = x_all.fillna(fill_vals)
        reg.fit(x_all, tr_t["scalar_coupling_constant"].astype(np.float64))

        x_te = test.loc[te_mask, col]
        if x_te.isna().any().any():
            x_te = x_te.fillna(fill_vals.reindex(x_te.columns))
        test_pred[te_mask] = reg.predict(x_te)

mean_log_mae_over_types = float(np.mean([s for _, s in val_scores]))
print("Validation mean log(MAE) over types:", mean_log_mae_over_types)

test["scalar_coupling_constant"] = test_pred

submission = test[["id", "scalar_coupling_constant"]].copy()
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
assert (
    submission.shape[0] == sub.shape[0]
), "Submission row count must match sample_submission."
assert submission.columns.tolist() == [
    "id",
    "scalar_coupling_constant",
], "Submission columns must match required format."
