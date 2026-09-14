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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

_CANDIDATE_DIRS = [
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_DIR = None
for d in _CANDIDATE_DIRS:
    if os.path.isdir(d):
        if os.path.basename(d) in ("input", "data"):
            comp = os.path.join(d, "champs-scalar-coupling")
            if os.path.isdir(comp):
                DATA_DIR = comp
                break
        else:
            if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
                os.path.join(d, "structures.csv")
            ):
                DATA_DIR = d
                break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not locate champs-scalar-coupling dataset directory. Tried: "
        + ", ".join(_CANDIDATE_DIRS)
    )

print("Using DATA_DIR:", DATA_DIR)
print("Listing DATA_DIR (head):", sorted(os.listdir(DATA_DIR))[:20])



## === cell 1
from sklearn.ensemble import RandomForestRegressor

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")


def _normalize_molecule_name(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


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
    dtype={
        "id": np.int32,
        "molecule_name": "object",
        "atom_index_0": np.int32,
        "atom_index_1": np.int32,
        "type": "object",
        "scalar_coupling_constant": np.float32,  # target can be float32; model will cast to float64 below as before
    },
)
test = pd.read_csv(
    test_path,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    dtype={
        "id": np.int32,
        "molecule_name": "object",
        "atom_index_0": np.int32,
        "atom_index_1": np.int32,
        "type": "object",
    },
)
sample_sub = pd.read_csv(sample_sub_path, usecols=["id"], dtype={"id": np.int32})

for df in (train, test):
    df["molecule_name"] = _normalize_molecule_name(df["molecule_name"])
    df["type"] = df["type"].astype(str).str.strip()
    df["atom_index_0"] = df["atom_index_0"].astype(np.int32, copy=False)
    df["atom_index_1"] = df["atom_index_1"].astype(np.int32, copy=False)

needed_molecules = pd.Index(
    pd.concat([train["molecule_name"], test["molecule_name"]], axis=0).unique()
)

structures_iter = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "object",
        "atom_index": np.int32,
        "atom": "object",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
    chunksize=500_000,
)

structures_parts = []
needed_set = set(needed_molecules.tolist())  # faster membership during chunk filtering
for chunk in structures_iter:
    chunk["molecule_name"] = _normalize_molecule_name(chunk["molecule_name"])
    chunk["atom"] = chunk["atom"].astype(str).str.strip()
    chunk = chunk[chunk["molecule_name"].isin(needed_set)]
    if len(chunk):
        structures_parts.append(chunk)

if not structures_parts:
    raise RuntimeError(
        "No rows loaded from structures.csv after filtering to needed molecules."
    )

structures = pd.concat(structures_parts, axis=0, ignore_index=True)
structures["atom"] = structures["atom"].astype("category")

required_struct_cols = {"molecule_name", "atom_index", "atom", "x", "y", "z"}
missing = required_struct_cols - set(structures.columns)
if missing:
    raise ValueError(
        f"structures.csv is missing required columns: {missing}. "
        f"Available columns: {structures.columns.tolist()}"
    )

structures_mi = structures.set_index(["molecule_name", "atom_index"], drop=True)[
    ["atom", "x", "y", "z"]
]
del structures, structures_parts


def _attach_atom_features(df: pd.DataFrame, atom_col: str, suffix: str) -> pd.DataFrame:
    """Vectorized equivalent of merging structures onto df by (molecule_name, atom_index_{suffix})."""
    key = pd.MultiIndex.from_arrays([df["molecule_name"].values, df[atom_col].values])
    got = structures_mi.reindex(
        key
    )  # left-join semantics; missing -> NaN, as merge(how='left')
    out = df.copy()
    out[f"atom_{suffix}"] = got["atom"].to_numpy()
    out[f"x_{suffix}"] = got["x"].to_numpy(np.float32, copy=False)
    out[f"y_{suffix}"] = got["y"].to_numpy(np.float32, copy=False)
    out[f"z_{suffix}"] = got["z"].to_numpy(np.float32, copy=False)
    return out


train_feat = _attach_atom_features(train, "atom_index_0", "0")
test_feat = _attach_atom_features(test, "atom_index_0", "0")
train_feat = _attach_atom_features(train_feat, "atom_index_1", "1")
test_feat = _attach_atom_features(test_feat, "atom_index_1", "1")

for name, df in (("train_feat", train_feat), ("test_feat", test_feat)):
    missing0 = int(pd.isna(df["x_0"]).sum())
    missing1 = int(pd.isna(df["x_1"]).sum())
    if missing0 or missing1:
        bad = df[pd.isna(df["x_0"]) | pd.isna(df["x_1"])][
            ["molecule_name", "atom_index_0", "atom_index_1", "type"]
        ].head(5)
        print(
            f"WARNING: {name}: missing merged structure rows. "
            f"atom0 coord NaNs={missing0}, atom1 coord NaNs={missing1}. "
            f"Filling missing coords with 0.0 and atom with 'X'. Sample unmatched keys:\n{bad}"
        )
        for c in ["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"]:
            df[c] = df[c].fillna(0.0).astype(np.float32, copy=False)
        for c in ["atom_0", "atom_1"]:
            df[c] = df[c].astype("object").fillna("X").astype(str).str.strip()

for df in (train_feat, test_feat):
    x0 = df["x_0"].to_numpy(np.float32, copy=False)
    y0 = df["y_0"].to_numpy(np.float32, copy=False)
    z0 = df["z_0"].to_numpy(np.float32, copy=False)
    x1 = df["x_1"].to_numpy(np.float32, copy=False)
    y1 = df["y_1"].to_numpy(np.float32, copy=False)
    z1 = df["z_1"].to_numpy(np.float32, copy=False)

    dx = (x0 - x1).astype(np.float32, copy=False)
    dy = (y0 - y1).astype(np.float32, copy=False)
    dz = (z0 - z1).astype(np.float32, copy=False)
    dist2 = (dx * dx + dy * dy + dz * dz).astype(np.float32, copy=False)
    dist = np.sqrt(dist2.astype(np.float64)).astype(np.float32, copy=False)

    df["dx"] = dx
    df["dy"] = dy
    df["dz"] = dz
    df["dist2"] = dist2
    df["dist"] = dist

all_atoms0 = pd.concat([train_feat["atom_0"], test_feat["atom_0"]], axis=0).astype(
    "category"
)
all_atoms1 = pd.concat([train_feat["atom_1"], test_feat["atom_1"]], axis=0).astype(
    "category"
)
atom0_cats = all_atoms0.cat.categories
atom1_cats = all_atoms1.cat.categories

train_feat["atom_0_i"] = pd.Categorical(
    train_feat["atom_0"], categories=atom0_cats
).codes.astype(np.int16)
train_feat["atom_1_i"] = pd.Categorical(
    train_feat["atom_1"], categories=atom1_cats
).codes.astype(np.int16)
test_feat["atom_0_i"] = pd.Categorical(
    test_feat["atom_0"], categories=atom0_cats
).codes.astype(np.int16)
test_feat["atom_1_i"] = pd.Categorical(
    test_feat["atom_1"], categories=atom1_cats
).codes.astype(np.int16)

for col in ("atom_0_i", "atom_1_i"):
    if (train_feat[col] < 0).any() or (test_feat[col] < 0).any():
        raise ValueError(
            f"Found negative categorical codes in {col}; unexpected missing/unseen atom values."
        )

feature_cols = [
    "atom_index_0",
    "atom_index_1",
    "atom_0_i",
    "atom_1_i",
    "dx",
    "dy",
    "dz",
    "dist",
    "dist2",
]

preds = np.zeros(len(test_feat), dtype=np.float64)
GLOBAL_SEED = 42

train_feat["type"] = train_feat["type"].astype("category")
test_feat["type"] = test_feat["type"].astype("category")

types = sorted(train_feat["type"].astype(str).unique().tolist())
test_type_groups = test_feat.groupby("type", sort=False).indices
train_type_groups = train_feat.groupby("type", sort=False).indices

type_median = train_feat.groupby("type")["scalar_coupling_constant"].median().to_dict()
global_median = float(train_feat["scalar_coupling_constant"].median())

X_train_all = train_feat[feature_cols].to_numpy(copy=False)
y_train_all = train_feat["scalar_coupling_constant"].to_numpy(np.float64, copy=False)
X_test_all = test_feat[feature_cols].to_numpy(copy=False)

for t in types:
    tr_idx = train_type_groups.get(t, None)
    te_idx = test_type_groups.get(t, None)
    if te_idx is None or len(te_idx) == 0:
        continue
    if tr_idx is None or len(tr_idx) < 50:
        preds[te_idx] = type_median.get(t, global_median)
        continue

    X_tr = X_train_all[tr_idx]
    y_tr = y_train_all[tr_idx]
    X_te = X_test_all[te_idx]

    model = RandomForestRegressor(
        n_estimators=80,
        random_state=GLOBAL_SEED,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
    )
    model.fit(X_tr, y_tr)
    preds[te_idx] = model.predict(X_te)

submission = pd.DataFrame(
    {"id": test_feat["id"].values, "scalar_coupling_constant": preds}
)
submission = submission.sort_values("id").reset_index(drop=True)

submission = sample_sub[["id"]].merge(
    submission, on="id", how="left", validate="one_to_one"
)

if submission["scalar_coupling_constant"].isna().any():
    raise ValueError(
        "Submission contains NaN predictions after alignment; cannot write valid submission."
    )

out_path = "my_blend_1.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Rows:", len(submission), "Columns:", submission.columns.tolist())
print("NaNs in preds:", int(submission["scalar_coupling_constant"].isna().sum()))
