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



## === cell 1
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

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
)
sample_sub = pd.read_csv(sample_sub_path, usecols=["id"], dtype={"id": np.int32})

structures_idx = structures.set_index(["molecule_name", "atom_index"])[
    ["atom", "x", "y", "z"]
]


def add_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    base = df[["molecule_name", "atom_index_0", "atom_index_1"]].copy()

    key0 = pd.MultiIndex.from_frame(
        pd.DataFrame(
            {"molecule_name": base["molecule_name"], "atom_index": base["atom_index_0"]}
        )
    )
    key1 = pd.MultiIndex.from_frame(
        pd.DataFrame(
            {"molecule_name": base["molecule_name"], "atom_index": base["atom_index_1"]}
        )
    )

    a0 = structures_idx.loc[key0].reset_index(drop=True)
    a1 = structures_idx.loc[key1].reset_index(drop=True)

    out = df.copy()
    out["atom_0"] = a0["atom"].to_numpy()
    out["x0"] = a0["x"].to_numpy(dtype=np.float32, copy=False)
    out["y0"] = a0["y"].to_numpy(dtype=np.float32, copy=False)
    out["z0"] = a0["z"].to_numpy(dtype=np.float32, copy=False)

    out["atom_1"] = a1["atom"].to_numpy()
    out["x1"] = a1["x"].to_numpy(dtype=np.float32, copy=False)
    out["y1"] = a1["y"].to_numpy(dtype=np.float32, copy=False)
    out["z1"] = a1["z"].to_numpy(dtype=np.float32, copy=False)

    x0 = out["x0"].to_numpy(dtype=np.float32, copy=False)
    y0 = out["y0"].to_numpy(dtype=np.float32, copy=False)
    z0 = out["z0"].to_numpy(dtype=np.float32, copy=False)
    x1 = out["x1"].to_numpy(dtype=np.float32, copy=False)
    y1 = out["y1"].to_numpy(dtype=np.float32, copy=False)
    z1 = out["z1"].to_numpy(dtype=np.float32, copy=False)

    dx = x0 - x1
    dy = y0 - y1
    dz = z0 - z1
    dist_sq = dx * dx + dy * dy + dz * dz
    dist = np.sqrt(dist_sq, dtype=np.float32)

    out["distance"] = dist
    out["dx"] = dx
    out["dy"] = dy
    out["dz"] = dz
    out["distance_sq"] = dist_sq
    return out


feature_cols_num = ["distance", "distance_sq", "dx", "dy", "dz"]
feature_cols_cat = ["type", "atom_0", "atom_1"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ]
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
)



## === cell 2
types = pd.Index(
    train["type"].cat.categories
    if hasattr(train["type"], "cat")
    else train["type"].unique()
)

train_fe = add_pair_features(train)
test_fe = add_pair_features(test)

y_all = train_fe["scalar_coupling_constant"].to_numpy(dtype=np.float32, copy=False)

pred_test = np.empty(len(test_fe), dtype=np.float32)

for t in types:
    train_mask = (train_fe["type"] == t).to_numpy()
    test_mask = (test_fe["type"] == t).to_numpy()

    X_train_t = train_fe.loc[train_mask, feature_cols_num + feature_cols_cat]
    y_train_t = y_all[train_mask]
    X_test_t = test_fe.loc[test_mask, feature_cols_num + feature_cols_cat]

    clf_t = Pipeline(steps=[("preprocess", preprocess), ("model", base_model)])
    clf_t.fit(X_train_t, y_train_t)

    pred_t = clf_t.predict(X_test_t).astype(np.float32, copy=False)
    pred_test[test_mask] = pred_t



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3471828277.py in <cell line: 0>()
     11 # (Slicing before feature build would require repeated structure lookups per type, usually slower overall.)
     12 train_fe = add_pair_features(train)
---> 13 test_fe = add_pair_features(test)
     14 
     15 y_all = train_fe["scalar_coupling_constant"].to_numpy(dtype=np.float32, copy=False)

/tmp/ipykernel_11/2557561986.py in add_pair_features(df)
     79     )
     80 
---> 81     a0 = structures_idx.loc[key0].reset_index(drop=True)
     82     a1 = structures_idx.loc[key1].reset_index(drop=True)
     83 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1189             maybe_callable = com.apply_if_callable(key, self.obj)
   1190             maybe_callable = self._check_deprecated_callable_usage(key, maybe_callable)
-> 1191             return self._getitem_axis(maybe_callable, axis=axis)
   1192 
   1193     def _is_scalar_access(self, key: tuple):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1418                     raise ValueError("Cannot index with multidimensional key")
   1419 
-> 1420                 return self._getitem_iterable(key, axis=axis)
   1421 
   1422             # nested tuple slicing

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_iterable(self, key, axis)
   1358 
   1359         # A collection of keys
-> 1360         keyarr, indexer = self._get_listlike_indexer(key, axis)
   1361         return self.obj._reindex_with_indexers(
   1362             {axis: [keyarr, indexer]}, copy=True, allow_dups=True

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_listlike_indexer(self, key, axis)
   1556         axis_name = self.obj._get_axis_name(axis)
   1557 
-> 1558         keyarr, indexer = ax._get_indexer_strict(key, axis_name)
   1559 
   1560         return keyarr, indexer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/multi.py in _get_indexer_strict(self, key, axis_name)
   2764             return self[indexer], indexer
   2765 
-> 2766         return super()._get_indexer_strict(key, axis_name)
   2767 
   2768     def _raise_if_missing(self, key, indexer, axis_name: str) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/multi.py in _raise_if_missing(self, key, indexer, axis_name)
   2784                 raise KeyError(f"{keyarr} not in index")
   2785         else:
-> 2786             return super()._raise_if_missing(key, indexer, axis_name)
   2787 
   2788     def _get_indexer_level_0(self, target) -> npt.NDArray[np.intp]:

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [MultiIndex([('dsgdb9nsd_071451',  9),\n            ('dsgdb9nsd_071451',  9),\n            ('dsgdb9nsd_071451',  9),\n            ('dsgdb9nsd_071451',  9),\n            ('dsgdb9nsd_071451',  9),\n            ('dsgdb9nsd_071451',  9),\n            ('dsgdb9nsd_071451', 10),\n            ('dsgdb9nsd_071451', 10),\n            ('dsgdb9nsd_071451', 10),\n            ('dsgdb9nsd_071451', 10),\n            ...\n            ('dsgdb9nsd_118777', 23),\n            ('dsgdb9nsd_118777', 23),\n            ('dsgdb9nsd_118777', 23),\n            ('dsgdb9nsd_118777', 23),\n            ('dsgdb9nsd_118777', 23),\n            ('dsgdb9nsd_118777', 24),\n            ('dsgdb9nsd_118777', 24),\n            ('dsgdb9nsd_118777', 24),\n            ('dsgdb9nsd_118777', 24),\n            ('dsgdb9nsd_118777', 24)],\n           names=['molecule_name', 'atom_index'], length=467813)] are in the [index]"

## === cell 3
submission = pd.DataFrame(
    {"id": test["id"].values, "scalar_coupling_constant": pred_test}
)
submission = submission.merge(sample_sub[["id"]], on="id", how="right")
submission["scalar_coupling_constant"] = submission["scalar_coupling_constant"].astype(
    np.float32, copy=False
)

out_path = os.path.join(WORKING_DIR, "submission.csv")
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Rows:", len(submission), " Expected:", len(sample_sub))
print("Any NA preds:", submission["scalar_coupling_constant"].isna().any())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/244858690.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"id": test["id"].values, "scalar_coupling_constant": pred_test}
      3 )
      4 submission = submission.merge(sample_sub[["id"]], on="id", how="right")
      5 submission["scalar_coupling_constant"] = submission["scalar_coupling_constant"].astype(

NameError: name 'pred_test' is not defined
