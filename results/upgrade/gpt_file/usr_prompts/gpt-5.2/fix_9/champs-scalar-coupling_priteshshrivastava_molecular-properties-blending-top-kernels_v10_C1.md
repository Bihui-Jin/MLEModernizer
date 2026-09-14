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

-1.6712010456498954

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

INPUT_DIR = "/kaggle/input/champs-scalar-coupling"
WORKING_DIR = "/kaggle/working"

print("Input dir exists:", os.path.exists(INPUT_DIR))
print("Files (sample):", sorted(os.listdir(INPUT_DIR))[:20])

np.random.seed(42)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone

train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")
structures_path = os.path.join(INPUT_DIR, "structures.csv")
sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

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
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
        "scalar_coupling_constant": np.float32,
    },
    engine="c",
    low_memory=False,
)
test = pd.read_csv(
    test_path,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
    engine="c",
    low_memory=False,
)
structures = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
    engine="c",
    low_memory=False,
)
sample_sub = pd.read_csv(
    sample_sub_path,
    usecols=["id"],
    dtype={"id": np.int32},
    engine="c",
    low_memory=False,
)

all_types = pd.Index(train["type"].cat.categories).union(test["type"].cat.categories)
train["type"] = train["type"].cat.set_categories(all_types)
test["type"] = test["type"].cat.set_categories(all_types)

all_mols = pd.Index(train["molecule_name"].cat.categories).union(
    test["molecule_name"].cat.categories
)
train["molecule_name"] = train["molecule_name"].cat.set_categories(all_mols)
test["molecule_name"] = test["molecule_name"].cat.set_categories(all_mols)
structures["molecule_name"] = structures["molecule_name"].cat.set_categories(all_mols)

feature_cols_num = ["distance", "distance_sq", "dx", "dy", "dz"]
feature_cols_cat = ["type", "atom_0", "atom_1"]

numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

categorical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, feature_cols_num),
        ("cat", categorical_transformer, feature_cols_cat),
    ],
    remainder="drop",
)

base_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_depth=None,
    min_samples_leaf=1,
    warm_start=False,
)

structures_small = structures[
    ["molecule_name", "atom_index", "atom", "x", "y", "z"]
].copy()
structures_small = structures_small.sort_values(
    ["molecule_name", "atom_index"], kind="mergesort", ignore_index=True
)




## === cell 1
def _build_structure_maps(struct_df: pd.DataFrame):
    mol_codes = struct_df["molecule_name"].cat.codes.to_numpy(copy=False)
    atom_idx = struct_df["atom_index"].to_numpy(copy=False)
    atom_codes = (
        struct_df["atom"].cat.codes.to_numpy(copy=False).astype(np.int16, copy=False)
    )
    x = struct_df["x"].to_numpy(copy=False)
    y = struct_df["y"].to_numpy(copy=False)
    z = struct_df["z"].to_numpy(copy=False)

    change = np.flatnonzero(np.diff(mol_codes)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(mol_codes)]
    uniq = mol_codes[starts]

    maps = {}
    for mc, s, e in zip(uniq.tolist(), starts.tolist(), ends.tolist()):
        ai = atom_idx[s:e].astype(np.int32, copy=False)
        max_ai = int(ai.max()) if len(ai) else -1
        atom_map = np.full(max_ai + 1, -1, dtype=np.int16)
        x_map = np.full(max_ai + 1, np.nan, dtype=np.float32)
        y_map = np.full(max_ai + 1, np.nan, dtype=np.float32)
        z_map = np.full(max_ai + 1, np.nan, dtype=np.float32)

        atom_map[ai] = atom_codes[s:e]
        x_map[ai] = x[s:e]
        y_map[ai] = y[s:e]
        z_map[ai] = z[s:e]
        maps[int(mc)] = (atom_map, x_map, y_map, z_map)
    return maps, struct_df["atom"].cat.categories


STRUCT_MAPS, ATOM_CATEGORIES = _build_structure_maps(structures_small)


def add_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]].copy()
    if "scalar_coupling_constant" in df.columns:
        out["scalar_coupling_constant"] = df["scalar_coupling_constant"].to_numpy(
            copy=False
        )

    mol_codes = (
        out["molecule_name"].cat.codes.to_numpy(copy=False).astype(np.int32, copy=False)
    )
    a0i = out["atom_index_0"].to_numpy(copy=False).astype(np.int32, copy=False)
    a1i = out["atom_index_1"].to_numpy(copy=False).astype(np.int32, copy=False)

    n = len(out)
    atom0_codes = np.empty(n, dtype=np.int16)
    atom1_codes = np.empty(n, dtype=np.int16)
    x0 = np.empty(n, dtype=np.float32)
    y0 = np.empty(n, dtype=np.float32)
    z0 = np.empty(n, dtype=np.float32)
    x1 = np.empty(n, dtype=np.float32)
    y1 = np.empty(n, dtype=np.float32)
    z1 = np.empty(n, dtype=np.float32)

    order = np.argsort(mol_codes, kind="mergesort")
    mol_sorted = mol_codes[order]
    change = np.flatnonzero(np.diff(mol_sorted)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, n]

    for s, e in zip(starts.tolist(), ends.tolist()):
        idx = order[s:e]
        mc = int(mol_sorted[s])
        atom_map, x_map, y_map, z_map = STRUCT_MAPS[mc]

        a0 = a0i[idx]
        a1 = a1i[idx]

        atom0_codes[idx] = atom_map[a0]
        atom1_codes[idx] = atom_map[a1]

        x0[idx] = x_map[a0]
        y0[idx] = y_map[a0]
        z0[idx] = z_map[a0]
        x1[idx] = x_map[a1]
        y1[idx] = y_map[a1]
        z1[idx] = z_map[a1]

    dx = (x0 - x1).astype(np.float32, copy=False)
    dy = (y0 - y1).astype(np.float32, copy=False)
    dz = (z0 - z1).astype(np.float32, copy=False)
    dist_sq = (dx * dx + dy * dy + dz * dz).astype(np.float32, copy=False)
    dist = np.sqrt(dist_sq).astype(np.float32, copy=False)

    out["distance"] = dist
    out["dx"] = dx
    out["dy"] = dy
    out["dz"] = dz
    out["distance_sq"] = dist_sq

    out["atom_0"] = pd.Categorical.from_codes(
        atom0_codes.astype(np.int32), categories=ATOM_CATEGORIES
    )
    out["atom_1"] = pd.Categorical.from_codes(
        atom1_codes.astype(np.int32), categories=ATOM_CATEGORIES
    )
    return out


train_fe = add_pair_features(train)
test_fe = add_pair_features(test)

y_all = train_fe["scalar_coupling_constant"].to_numpy(dtype=np.float32, copy=False)
pred_test = np.empty(len(test_fe), dtype=np.float32)

train_type_codes = train_fe["type"].cat.codes.to_numpy(copy=False)
test_type_codes = test_fe["type"].cat.codes.to_numpy(copy=False)

types = train_fe["type"].cat.categories
type_code_values = np.arange(len(types), dtype=np.int16)


def _group_slices_from_codes(codes: np.ndarray):
    order = np.argsort(codes, kind="mergesort")  # stable, deterministic
    sorted_codes = codes[order]
    change = np.flatnonzero(np.diff(sorted_codes)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(sorted_codes)]
    uniq = sorted_codes[starts]
    return order, uniq, starts, ends


train_order, train_uniq, train_starts, train_ends = _group_slices_from_codes(
    train_type_codes
)
test_order, test_uniq, test_starts, test_ends = _group_slices_from_codes(
    test_type_codes
)

train_slice = {
    int(c): (int(s), int(e)) for c, s, e in zip(train_uniq, train_starts, train_ends)
}
test_slice = {
    int(c): (int(s), int(e)) for c, s, e in zip(test_uniq, test_starts, test_ends)
}

train_X_all = train_fe[feature_cols_num + feature_cols_cat]
test_X_all = test_fe[feature_cols_num + feature_cols_cat]

for code in type_code_values:
    ts = test_slice.get(int(code))
    tr = train_slice.get(int(code))
    if ts is None or tr is None:
        continue

    t_start, t_end = ts
    r_start, r_end = tr

    test_idx = test_order[t_start:t_end]
    train_idx = train_order[r_start:r_end]

    X_train_t = train_X_all.iloc[train_idx]
    y_train_t = y_all[train_idx]
    X_test_t = test_X_all.iloc[test_idx]

    Xt_train = preprocess.fit_transform(X_train_t)
    Xt_test = preprocess.transform(X_test_t)

    rf = clone(base_model)
    rf.fit(Xt_train, y_train_t)

    pred_t = rf.predict(Xt_test).astype(np.float32, copy=False)
    pred_test[test_idx] = pred_t

if np.isnan(pred_test).any():
    pred_test = np.where(
        np.isnan(pred_test), np.float32(np.mean(y_all)), pred_test
    ).astype(np.float32, copy=False)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1213606674.py in <cell line: 0>()
    115 
    116 train_fe = add_pair_features(train)
--> 117 test_fe = add_pair_features(test)
    118 
    119 y_all = train_fe["scalar_coupling_constant"].to_numpy(dtype=np.float32, copy=False)

/tmp/ipykernel_11/1213606674.py in add_pair_features(df)
     77         idx = order[s:e]
     78         mc = int(mol_sorted[s])
---> 79         atom_map, x_map, y_map, z_map = STRUCT_MAPS[mc]
     80 
     81         a0 = a0i[idx]

KeyError: 18

## === cell 2
submission = pd.DataFrame(
    {"id": test["id"].values, "scalar_coupling_constant": pred_test}
)
submission = sample_sub.merge(submission, on="id", how="left", sort=False)

if submission["scalar_coupling_constant"].isna().any():
    submission["scalar_coupling_constant"] = submission[
        "scalar_coupling_constant"
    ].fillna(
        np.float32(
            np.mean(
                train_fe["scalar_coupling_constant"].to_numpy(
                    dtype=np.float32, copy=False
                )
            )
        )
    )

submission["scalar_coupling_constant"] = submission["scalar_coupling_constant"].astype(
    np.float32, copy=False
)

out_path = os.path.join(WORKING_DIR, "submission.csv")
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Rows:", len(submission), " Expected:", len(sample_sub))
print("Any NA preds:", submission["scalar_coupling_constant"].isna().any())

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/714446391.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"id": test["id"].values, "scalar_coupling_constant": pred_test}
      3 )
      4 submission = sample_sub.merge(submission, on="id", how="left", sort=False)
      5 

NameError: name 'pred_test' is not defined
