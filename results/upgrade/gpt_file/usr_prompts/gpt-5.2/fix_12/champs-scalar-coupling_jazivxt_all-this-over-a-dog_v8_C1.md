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
    from sklearnex import patch_sklearn  # type: ignore

    patch_sklearn()
except Exception:
    pass

from sklearn import preprocessing, ensemble

np.random.seed(4)


def resolve_input_dir():
    candidates = [
        "../input/champs-scalar-coupling",
        "../input",
        "/kaggle/input/champs-scalar-coupling",
        "/kaggle/input",
        "/kaggle/data/champs-scalar-coupling",
        "/kaggle/data/input/champs-scalar-coupling",
        "/kaggle/data/input",
    ]
    needed = {"train.csv", "test.csv", "structures.csv", "sample_submission.csv"}
    for d in candidates:
        if os.path.isdir(d) and needed.issubset(set(os.listdir(d))):
            return d
    if needed.issubset(set(os.listdir("."))):
        return "."
    raise FileNotFoundError(
        "Could not locate input directory containing train/test/structures/sample_submission CSV files."
    )


INPUT_DIR = resolve_input_dir()

train_dtypes = {
    "id": "int32",
    "molecule_name": "category",
    "atom_index_0": "int16",
    "atom_index_1": "int16",
    "type": "category",
    "scalar_coupling_constant": "float32",
}
test_dtypes = {
    "id": "int32",
    "molecule_name": "category",
    "atom_index_0": "int16",
    "atom_index_1": "int16",
    "type": "category",
}

train = pd.read_csv(
    os.path.join(INPUT_DIR, "train.csv"),
    usecols=list(train_dtypes.keys()),
    dtype=train_dtypes,
)
test = pd.read_csv(
    os.path.join(INPUT_DIR, "test.csv"),
    usecols=list(test_dtypes.keys()),
    dtype=test_dtypes,
)
sub = pd.read_csv(
    os.path.join(INPUT_DIR, "sample_submission.csv"),
    usecols=["id", "scalar_coupling_constant"],
)
print(train.shape, test.shape, sub.shape)

train_type_str = train["type"].astype(str)
test_type_str = test["type"].astype(str)

train["atom1"] = train_type_str.str[2]
train["atom2"] = train_type_str.str[3]
test["atom1"] = test_type_str.str[2]
test["atom2"] = test_type_str.str[3]

lbl = preprocessing.LabelEncoder()
all_types = pd.concat([train_type_str, test_type_str], axis=0)
for i in range(4):
    s_all = all_types.str[i]
    lbl.fit(s_all)
    train["type" + str(i)] = lbl.transform(train_type_str.str[i]).astype("int8")
    test["type" + str(i)] = lbl.transform(test_type_str.str[i]).astype("int8")

structures = pd.read_csv(
    os.path.join(INPUT_DIR, "structures.csv"),
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "category",
        "atom_index": "int16",
        "atom": "category",
        "x": "float32",
        "y": "float32",
        "z": "float32",
    },
)
struct_idx = pd.MultiIndex.from_frame(structures[["molecule_name", "atom_index"]])
structures = structures.set_index(struct_idx, drop=True)
structures = structures.drop(columns=["molecule_name", "atom_index"])

tr_idx0 = pd.MultiIndex.from_frame(
    train[["molecule_name", "atom_index_0"]].rename(
        columns={"atom_index_0": "atom_index"}
    )
)
tr_idx1 = pd.MultiIndex.from_frame(
    train[["molecule_name", "atom_index_1"]].rename(
        columns={"atom_index_1": "atom_index"}
    )
)
te_idx0 = pd.MultiIndex.from_frame(
    test[["molecule_name", "atom_index_0"]].rename(
        columns={"atom_index_0": "atom_index"}
    )
)
te_idx1 = pd.MultiIndex.from_frame(
    test[["molecule_name", "atom_index_1"]].rename(
        columns={"atom_index_1": "atom_index"}
    )
)

tr_s0 = structures.reindex(tr_idx0)
tr_s1 = structures.reindex(tr_idx1)
te_s0 = structures.reindex(te_idx0)
te_s1 = structures.reindex(te_idx1)

tr_atom0 = tr_s0["atom"].astype(str).to_numpy()
tr_atom1 = tr_s1["atom"].astype(str).to_numpy()
te_atom0 = te_s0["atom"].astype(str).to_numpy()
te_atom1 = te_s1["atom"].astype(str).to_numpy()

mask_tr0 = tr_atom0 == train["atom1"].astype(str).to_numpy()
mask_tr1 = tr_atom1 == train["atom2"].astype(str).to_numpy()
mask_te0 = te_atom0 == test["atom1"].astype(str).to_numpy()
mask_te1 = te_atom1 == test["atom2"].astype(str).to_numpy()

for prefix, s0, s1, m0, m1, df in [
    ("", tr_s0, tr_s1, mask_tr0, mask_tr1, train),
    ("", te_s0, te_s1, mask_te0, mask_te1, test),
]:
    x0 = s0["x"].to_numpy(np.float32, copy=True)
    y0 = s0["y"].to_numpy(np.float32, copy=True)
    z0 = s0["z"].to_numpy(np.float32, copy=True)
    x1 = s1["x"].to_numpy(np.float32, copy=True)
    y1 = s1["y"].to_numpy(np.float32, copy=True)
    z1 = s1["z"].to_numpy(np.float32, copy=True)

    x0[~m0] = np.nan
    y0[~m0] = np.nan
    z0[~m0] = np.nan
    x1[~m1] = np.nan
    y1[~m1] = np.nan
    z1[~m1] = np.nan

    df["x0"], df["y0"], df["z0"] = x0, y0, z0
    df["x1"], df["y1"], df["z1"] = x1, y1, z1

del (
    tr_s0,
    tr_s1,
    te_s0,
    te_s1,
    tr_idx0,
    tr_idx1,
    te_idx0,
    te_idx1,
    tr_atom0,
    tr_atom1,
    te_atom0,
    te_atom1,
)
del mask_tr0, mask_tr1, mask_te0, mask_te1

mulliken = pd.read_csv(
    os.path.join(INPUT_DIR, "mulliken_charges.csv"),
    usecols=["molecule_name", "atom_index", "mulliken_charge"],
    dtype={
        "molecule_name": "category",
        "atom_index": "int16",
        "mulliken_charge": "float32",
    },
)
mull_idx = pd.MultiIndex.from_frame(mulliken[["molecule_name", "atom_index"]])
mull_series = pd.Series(
    mulliken["mulliken_charge"].to_numpy(np.float32, copy=False), index=mull_idx
)
train["mulliken_charge_0"] = mull_series.reindex(
    pd.MultiIndex.from_frame(
        train[["molecule_name", "atom_index_0"]].rename(
            columns={"atom_index_0": "atom_index"}
        )
    )
).to_numpy(np.float32, copy=False)
train["mulliken_charge_1"] = mull_series.reindex(
    pd.MultiIndex.from_frame(
        train[["molecule_name", "atom_index_1"]].rename(
            columns={"atom_index_1": "atom_index"}
        )
    )
).to_numpy(np.float32, copy=False)
test["mulliken_charge_0"] = mull_series.reindex(
    pd.MultiIndex.from_frame(
        test[["molecule_name", "atom_index_0"]].rename(
            columns={"atom_index_0": "atom_index"}
        )
    )
).to_numpy(np.float32, copy=False)
test["mulliken_charge_1"] = mull_series.reindex(
    pd.MultiIndex.from_frame(
        test[["molecule_name", "atom_index_1"]].rename(
            columns={"atom_index_1": "atom_index"}
        )
    )
).to_numpy(np.float32, copy=False)
del mulliken, mull_idx, mull_series

mst = pd.read_csv(
    os.path.join(INPUT_DIR, "magnetic_shielding_tensors.csv"),
    dtype={"molecule_name": "category", "atom_index": "int16"},
)
mst_cols = [c for c in mst.columns if c not in ["molecule_name", "atom_index"]]
for c in mst_cols:
    if mst[c].dtype != np.float32:
        mst[c] = mst[c].astype("float32", copy=False)
mst_idx = pd.MultiIndex.from_frame(mst[["molecule_name", "atom_index"]])
mst_df = mst.set_index(mst_idx, drop=True)[mst_cols]
del mst

tr_mst0 = mst_df.reindex(
    pd.MultiIndex.from_frame(
        train[["molecule_name", "atom_index_0"]].rename(
            columns={"atom_index_0": "atom_index"}
        )
    )
)
tr_mst1 = mst_df.reindex(
    pd.MultiIndex.from_frame(
        train[["molecule_name", "atom_index_1"]].rename(
            columns={"atom_index_1": "atom_index"}
        )
    )
)
te_mst0 = mst_df.reindex(
    pd.MultiIndex.from_frame(
        test[["molecule_name", "atom_index_0"]].rename(
            columns={"atom_index_0": "atom_index"}
        )
    )
)
te_mst1 = mst_df.reindex(
    pd.MultiIndex.from_frame(
        test[["molecule_name", "atom_index_1"]].rename(
            columns={"atom_index_1": "atom_index"}
        )
    )
)

train_mst0_cols = {c: f"mst0_{c}" for c in mst_cols}
train_mst1_cols = {c: f"mst1_{c}" for c in mst_cols}
test_mst0_cols = {c: f"mst0_{c}" for c in mst_cols}
test_mst1_cols = {c: f"mst1_{c}" for c in mst_cols}

train[list(train_mst0_cols.values())] = tr_mst0.rename(
    columns=train_mst0_cols
).to_numpy(dtype=np.float32, copy=False)
train[list(train_mst1_cols.values())] = tr_mst1.rename(
    columns=train_mst1_cols
).to_numpy(dtype=np.float32, copy=False)
test[list(test_mst0_cols.values())] = te_mst0.rename(columns=test_mst0_cols).to_numpy(
    dtype=np.float32, copy=False
)
test[list(test_mst1_cols.values())] = te_mst1.rename(columns=test_mst1_cols).to_numpy(
    dtype=np.float32, copy=False
)

del mst_df, tr_mst0, tr_mst1, te_mst0, te_mst1, mst_cols
del train_mst0_cols, train_mst1_cols, test_mst0_cols, test_mst1_cols

dip = pd.read_csv(
    os.path.join(INPUT_DIR, "dipole_moments.csv"),
    usecols=["molecule_name", "X", "Y", "Z"],
    dtype={"molecule_name": "category", "X": "float32", "Y": "float32", "Z": "float32"},
).rename(columns={"X": "dipole_X", "Y": "dipole_Y", "Z": "dipole_Z"})
pe = pd.read_csv(
    os.path.join(INPUT_DIR, "potential_energy.csv"),
    usecols=["molecule_name", "potential_energy"],
    dtype={"molecule_name": "category", "potential_energy": "float32"},
)

dip = dip.set_index("molecule_name", drop=True)
pe = pe.set_index("molecule_name", drop=True)

train[["dipole_X", "dipole_Y", "dipole_Z"]] = dip.reindex(train["molecule_name"])[
    ["dipole_X", "dipole_Y", "dipole_Z"]
].to_numpy(dtype=np.float32, copy=False)
test[["dipole_X", "dipole_Y", "dipole_Z"]] = dip.reindex(test["molecule_name"])[
    ["dipole_X", "dipole_Y", "dipole_Z"]
].to_numpy(dtype=np.float32, copy=False)

train["potential_energy"] = pe.reindex(train["molecule_name"])[
    "potential_energy"
].to_numpy(dtype=np.float32, copy=False)
test["potential_energy"] = pe.reindex(test["molecule_name"])[
    "potential_energy"
].to_numpy(dtype=np.float32, copy=False)

del dip, pe, structures, struct_idx

test = test.sort_values("id").reset_index(drop=True)
print("After physics merges:", train.shape, test.shape)



## === cell 1
train_p0 = train[["x0", "y0", "z0"]].to_numpy(dtype=np.float32, copy=False)
train_p1 = train[["x1", "y1", "z1"]].to_numpy(dtype=np.float32, copy=False)
test_p0 = test[["x0", "y0", "z0"]].to_numpy(dtype=np.float32, copy=False)
test_p1 = test[["x1", "y1", "z1"]].to_numpy(dtype=np.float32, copy=False)

dtrain = train_p0 - train_p1
dtest = test_p0 - test_p1

train["dx"] = dtrain[:, 0]
train["dy"] = dtrain[:, 1]
train["dz"] = dtrain[:, 2]
test["dx"] = dtest[:, 0]
test["dy"] = dtest[:, 1]
test["dz"] = dtest[:, 2]

train_dist = np.linalg.norm(dtrain, axis=1).astype(np.float32, copy=False)
test_dist = np.linalg.norm(dtest, axis=1).astype(np.float32, copy=False)

train["dist"] = train_dist
test["dist"] = test_dist
train["dist2"] = train_dist * train_dist
test["dist2"] = test_dist * test_dist

type_mean = train.groupby("type")["dist"].mean()
global_mean = float(train_dist.mean())

train["dist_to_type_mean"] = train["dist"] / train["type"].map(type_mean)
test_denom = test["type"].map(type_mean).fillna(global_mean)
test["dist_to_type_mean"] = test["dist"] / test_denom



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

train_X_df = train[col]
test_X_df = test[col]

med = train_X_df.median(numeric_only=True)
med_vec = med.reindex(col).to_numpy(dtype=np.float32, copy=False)

y = train["scalar_coupling_constant"].to_numpy(dtype=np.float32, copy=False)

train_type = train["type"].to_numpy()
test_type = test["type"].to_numpy()

test_pred = np.zeros(len(test), dtype=np.float64)
unique_types = pd.unique(train["type"])

X_train_all = np.ascontiguousarray(train_X_df.to_numpy(dtype=np.float32, copy=True))
X_test_all = np.ascontiguousarray(test_X_df.to_numpy(dtype=np.float32, copy=True))

nan_tr = np.isnan(X_train_all)
if nan_tr.any():
    cols = np.nonzero(nan_tr.any(axis=0))[0]
    for j in cols:
        m = nan_tr[:, j]
        if m.any():
            X_train_all[m, j] = med_vec[j]

nan_te = np.isnan(X_test_all)
if nan_te.any():
    cols = np.nonzero(nan_te.any(axis=0))[0]
    for j in cols:
        m = nan_te[:, j]
        if m.any():
            X_test_all[m, j] = med_vec[j]

np.nan_to_num(X_test_all, nan=0.0, copy=False)

for t in unique_types:
    tr_idx = train_type == t
    te_idx = test_type == t

    if not np.any(te_idx):
        continue

    reg = ensemble.ExtraTreesRegressor(n_jobs=-1, n_estimators=80, random_state=4)

    Xtr = X_train_all[tr_idx]
    Xte = X_test_all[te_idx]
    ytr = y[tr_idx]

    reg.fit(Xtr, ytr)
    test_pred[te_idx] = reg.predict(Xte)

test["scalar_coupling_constant"] = test_pred

submission = test[["id", "scalar_coupling_constant"]].sort_values("id")
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
