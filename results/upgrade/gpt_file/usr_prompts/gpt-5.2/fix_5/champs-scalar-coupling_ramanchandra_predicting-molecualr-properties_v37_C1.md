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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

3.19931

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import warnings

warnings.filterwarnings("ignore")
import random

random.seed(42)
np.random.seed(42)

import os

BASE_INPUT = "../input"
if not os.path.exists(BASE_INPUT):
    if os.path.exists("/kaggle/data/champs-scalar-coupling"):
        BASE_INPUT = "/kaggle/data/champs-scalar-coupling"
    elif os.path.exists("/kaggle/input/champs-scalar-coupling"):
        BASE_INPUT = "/kaggle/input/champs-scalar-coupling"
    elif os.path.exists("/kaggle/data/input"):
        BASE_INPUT = "/kaggle/data/input"
    else:
        BASE_INPUT = "/kaggle/data/champs-scalar-coupling"

print("Using BASE_INPUT:", BASE_INPUT)
if os.path.exists("../input"):
    print(os.listdir("../input")[:20])



## === cell 1
usecols_train = [
    "id",
    "molecule_name",
    "atom_index_0",
    "atom_index_1",
    "type",
    "scalar_coupling_constant",
]
usecols_test = ["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]
usecols_struct = ["molecule_name", "atom_index", "atom", "x", "y", "z"]

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
struct_dtypes = {
    "molecule_name": "category",
    "atom_index": "int16",
    "atom": "category",
    "x": "float32",
    "y": "float32",
    "z": "float32",
}

train_df = pd.read_csv(
    f"{BASE_INPUT}/train.csv",
    usecols=usecols_train,
    dtype=train_dtypes,
    low_memory=False,
)
test_df = pd.read_csv(
    f"{BASE_INPUT}/test.csv", usecols=usecols_test, dtype=test_dtypes, low_memory=False
)
structures = pd.read_csv(
    f"{BASE_INPUT}/structures.csv",
    usecols=usecols_struct,
    dtype=struct_dtypes,
    low_memory=False,
)

SCC_sample_path = f"{BASE_INPUT}/sample_submission.csv"



## === cell 2
print("Shape of train dataset:", train_df.shape)
print("Shape of test dataset:", test_df.shape)
print("Shape of structures dataset:", structures.shape)



## === cell 3
print("Train dtypes:\n", train_df.dtypes)
print("Test dtypes:\n", test_df.dtypes)
print("Structures dtypes:\n", structures.dtypes)



## === cell 4
structures_small = structures[
    ["molecule_name", "atom_index", "atom", "x", "y", "z"]
].copy()
structures_small = structures_small.set_index(
    ["molecule_name", "atom_index"], drop=False
)


def map_atom_data_fast(df: pd.DataFrame, atom_idx: int) -> pd.DataFrame:
    keys = pd.MultiIndex.from_frame(df[["molecule_name", f"atom_index_{atom_idx}"]])
    picked = structures_small.loc[keys, ["atom", "x", "y", "z"]].reset_index(drop=True)
    picked.columns = [
        f"atom_{atom_idx}",
        f"x_{atom_idx}",
        f"y_{atom_idx}",
        f"z_{atom_idx}",
    ]
    return pd.concat([df.reset_index(drop=True), picked], axis=1)


train_df = map_atom_data_fast(train_df, 0)
train_df = map_atom_data_fast(train_df, 1)
test_df = map_atom_data_fast(test_df, 0)
test_df = map_atom_data_fast(test_df, 1)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1419264493.py in <cell line: 0>()
     26 train_df = map_atom_data_fast(train_df, 0)
     27 train_df = map_atom_data_fast(train_df, 1)
---> 28 test_df = map_atom_data_fast(test_df, 0)
     29 test_df = map_atom_data_fast(test_df, 1)
     30 

/tmp/ipykernel_11/1419264493.py in map_atom_data_fast(df, atom_idx)
     13 def map_atom_data_fast(df: pd.DataFrame, atom_idx: int) -> pd.DataFrame:
     14     keys = pd.MultiIndex.from_frame(df[["molecule_name", f"atom_index_{atom_idx}"]])
---> 15     picked = structures_small.loc[keys, ["atom", "x", "y", "z"]].reset_index(drop=True)
     16     picked.columns = [
     17         f"atom_{atom_idx}",

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1182             if self._is_scalar_access(key):
   1183                 return self.obj._get_value(*key, takeable=self._takeable)
-> 1184             return self._getitem_tuple(key)
   1185         else:
   1186             # we by definition only have the 0th axis

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple(self, tup)
   1366         with suppress(IndexingError):
   1367             tup = self._expand_ellipsis(tup)
-> 1368             return self._getitem_lowerdim(tup)
   1369 
   1370         # no multi-index, so validate all of the indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_lowerdim(self, tup)
   1039         # we may have a nested tuples indexer here
   1040         if self._is_nested_tuple_indexer(tup):
-> 1041             return self._getitem_nested_tuple(tup)
   1042 
   1043         # we maybe be using a tuple to represent multiple dimensions here

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_nested_tuple(self, tup)
   1151                 continue
   1152 
-> 1153             obj = getattr(obj, self.name)._getitem_axis(key, axis=axis)
   1154             axis -= 1
   1155 

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

KeyError: "None of [MultiIndex([('dsgdb9nsd_071451',  9),\n            ('dsgdb9nsd_071451',  9),\n            ('dsgdb9nsd_071451',  9),\n            ('dsgdb9nsd_071451',  9),\n            ('dsgdb9nsd_071451',  9),\n            ('dsgdb9nsd_071451',  9),\n            ('dsgdb9nsd_071451', 10),\n            ('dsgdb9nsd_071451', 10),\n            ('dsgdb9nsd_071451', 10),\n            ('dsgdb9nsd_071451', 10),\n            ...\n            ('dsgdb9nsd_118777', 23),\n            ('dsgdb9nsd_118777', 23),\n            ('dsgdb9nsd_118777', 23),\n            ('dsgdb9nsd_118777', 23),\n            ('dsgdb9nsd_118777', 23),\n            ('dsgdb9nsd_118777', 24),\n            ('dsgdb9nsd_118777', 24),\n            ('dsgdb9nsd_118777', 24),\n            ('dsgdb9nsd_118777', 24),\n            ('dsgdb9nsd_118777', 24)],\n           names=['molecule_name', 'atom_index_0'], length=467813)] are in the [index]"

## === cell 5
train_m_0 = train_df[["x_0", "y_0", "z_0"]].to_numpy(dtype=np.float32, copy=False)
train_m_1 = train_df[["x_1", "y_1", "z_1"]].to_numpy(dtype=np.float32, copy=False)
test_m_0 = test_df[["x_0", "y_0", "z_0"]].to_numpy(dtype=np.float32, copy=False)
test_m_1 = test_df[["x_1", "y_1", "z_1"]].to_numpy(dtype=np.float32, copy=False)

d_train = train_m_0 - train_m_1
d_test = test_m_0 - test_m_1

train_df["dist_vector"] = np.linalg.norm(d_train, axis=1)
train_df["dist_X"] = d_train[:, 0] ** 2
train_df["dist_Y"] = d_train[:, 1] ** 2
train_df["dist_Z"] = d_train[:, 2] ** 2

test_df["dist_vector"] = np.linalg.norm(d_test, axis=1)
test_df["dist_X"] = d_test[:, 0] ** 2
test_df["dist_Y"] = d_test[:, 1] ** 2
test_df["dist_Z"] = d_test[:, 2] ** 2



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3030641901.py in <cell line: 0>()
      2 train_m_0 = train_df[["x_0", "y_0", "z_0"]].to_numpy(dtype=np.float32, copy=False)
      3 train_m_1 = train_df[["x_1", "y_1", "z_1"]].to_numpy(dtype=np.float32, copy=False)
----> 4 test_m_0 = test_df[["x_0", "y_0", "z_0"]].to_numpy(dtype=np.float32, copy=False)
      5 test_m_1 = test_df[["x_1", "y_1", "z_1"]].to_numpy(dtype=np.float32, copy=False)
      6 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index(['x_0', 'y_0', 'z_0'], dtype='object')] are in the [columns]"

## === cell 6
train_df = train_df.drop(columns=["molecule_name"], axis=1)
train_df.head(6)



## === cell 7
test_df = test_df.drop(columns=["molecule_name"], axis=1)
test_df.head(10)



## === cell 8
train_df["type"] = train_df.type.astype("category")
train_df["atom_0"] = train_df.atom_0.astype("category")
train_df["atom_1"] = train_df.atom_1.astype("category")

test_df["type"] = test_df.type.astype("category")
test_df["atom_0"] = test_df.atom_0.astype("category")
test_df["atom_1"] = test_df.atom_1.astype("category")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2447593397.py in <cell line: 0>()
      4 
      5 test_df["type"] = test_df.type.astype("category")
----> 6 test_df["atom_0"] = test_df.atom_0.astype("category")
      7 test_df["atom_1"] = test_df.atom_1.astype("category")
      8 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'atom_0'

## === cell 9
threshold = 0.95

candidate_numeric = [
    "atom_index_0",
    "atom_index_1",
    "x_0",
    "y_0",
    "z_0",
    "x_1",
    "y_1",
    "z_1",
    "dist_vector",
    "dist_X",
    "dist_Y",
    "dist_Z",
]
candidate_numeric = [c for c in candidate_numeric if c in train_df.columns]

corr_matrix = train_df[candidate_numeric].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))



## === cell 10
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
print("There are are %d columns to remove." % (len(to_drop)))



## === cell 11
if len(to_drop) > 0:
    train_df = train_df.drop(columns=to_drop)
    test_df = test_df.drop(columns=[c for c in to_drop if c in test_df.columns])



## === cell 12
Attributes = [
    "id",
    "atom_index_0",
    "atom_index_1",
    "type",
    "x_0",
    "y_0",
    "z_0",
    "atom_0",
    "atom_1",
    "x_1",
    "y_1",
    "z_1",
    "dist_vector",
    "dist_X",
    "dist_Y",
    "dist_Z",
]
cat_attributes = ["type", "atom_0", "atom_1"]
target_label = ["scalar_coupling_constant"]

Attributes = [
    c
    for c in Attributes
    if c in train_df.columns or c in test_df.columns or c in target_label
]

X_train = train_df[Attributes]
X_test = test_df[Attributes]
y_target = train_df[target_label]



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1776634243.py in <cell line: 0>()
     27 
     28 X_train = train_df[Attributes]
---> 29 X_test = test_df[Attributes]
     30 y_target = train_df[target_label]
     31 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['x_0', 'y_0', 'z_0', 'atom_0', 'atom_1', 'x_1', 'y_1', 'z_1'] not in index"

## === cell 13
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

feature_cols = [
    c for c in Attributes if c in X_train.columns
]  # exclude target if present
cat_cols = [c for c in cat_attributes if c in feature_cols]
num_cols = [c for c in feature_cols if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop",
    sparse_threshold=0.0,  # force sparse output when any sparse component exists
)

X_train_mat = preprocess.fit_transform(X_train[feature_cols])
X_test_mat = preprocess.transform(X_test[feature_cols])



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1183125181.py in <cell line: 0>()
     22 
     23 X_train_mat = preprocess.fit_transform(X_train[feature_cols])
---> 24 X_test_mat = preprocess.transform(X_test[feature_cols])
     25 

NameError: name 'X_test' is not defined

## === cell 14
print(X_train_mat.shape, X_test_mat.shape)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/338885190.py in <cell line: 0>()
----> 1 print(X_train_mat.shape, X_test_mat.shape)
      2 

NameError: name 'X_test_mat' is not defined

## === cell 15
print(y_target.shape)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1862122102.py in <cell line: 0>()
----> 1 print(y_target.shape)
      2 

NameError: name 'y_target' is not defined

## === cell 16
print("Example numeric columns:", num_cols[:10], "...")
print("Example categorical columns:", cat_cols)



## === cell 17
from sklearn import linear_model

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

linear_reg = linear_model.Lasso(alpha=0.0001, random_state=42, max_iter=10000)
lasso_model = linear_reg.fit(X_train_mat, y_target.values.ravel())
score = np.round(lasso_model.score(X_train_mat, y_target.values.ravel()), 3)
print("Accuracy of trained model:", score)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3801306504.py in <cell line: 0>()
     10 
     11 linear_reg = linear_model.Lasso(alpha=0.0001, random_state=42, max_iter=10000)
---> 12 lasso_model = linear_reg.fit(X_train_mat, y_target.values.ravel())
     13 score = np.round(lasso_model.score(X_train_mat, y_target.values.ravel()), 3)
     14 print("Accuracy of trained model:", score)

NameError: name 'y_target' is not defined

## === cell 18
y_pred = lasso_model.predict(X_test_mat)

SCC = pd.read_csv(SCC_sample_path)
SCC["scalar_coupling_constant"] = np.asarray(y_pred).reshape(-1)

out_path = "Lasso_Regression_model.csv"
SCC.to_csv(out_path, index=False)
print("Wrote submission:", out_path, "shape:", SCC.shape)
print(SCC.head())

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3630284463.py in <cell line: 0>()
----> 1 y_pred = lasso_model.predict(X_test_mat)
      2 
      3 SCC = pd.read_csv(SCC_sample_path)
      4 SCC["scalar_coupling_constant"] = np.asarray(y_pred).reshape(-1)
      5 

NameError: name 'lasso_model' is not defined
