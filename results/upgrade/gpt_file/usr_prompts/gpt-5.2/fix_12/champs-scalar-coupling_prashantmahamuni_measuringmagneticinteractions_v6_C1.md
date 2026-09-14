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
lightgbm==4.6.0
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

0.66661

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 3.96173) has done: 'We fix the empty-test merge that makes `test` have 0 rows by using `how="left"` (like the train merge) and by retaining `atom_index_0/1` so we can safely join even when indices are missing. We also correct a small categorical-encoding bug: `atom_nm_0` is a categorical feature but the code encodes `atom_nm_1` instead; swapping to `atom_nm_0` is score-improving while keeping the same model and features. Finally, we keep the pipeline end-to-end stable and always write a valid `submission.csv` with the correct columns and row count.'
- What this solution (achieved 3.97889) has done: 'Your current split is row-wise random, which leaks molecule information across train/valid and makes training/early-stopping select a model that doesn’t generalize to the (molecule-disjoint) Kaggle test set, inflating the leaderboard error. I switch the validation split to be by `molecule_name` (still the same LightGBM regressor, same features, same loss/metric) so early stopping chooses iterations that better match the competition’s evaluation setting. I also avoid dropping `molecule_name` from the split inputs (but it still not be used as a feature), and keep the same submission writing logic/format so you get a valid `submission.csv`. These are minimal changes aimed specifically at reducing your large gap from 3.96 toward 0.666.'
- What this solution (achieved 1.81121) has done: 'Your current LightGBM is trained on all coupling types together while the Kaggle metric is an average over types, so a single global model tends to underfit type-specific offsets and hurts the per-type MAE. With minimal change to the core approach (still LightGBM regression on the same features and early stopping), I train one model per `type` and predict the test rows for that type, then concatenate predictions back in `id` order. I also align early stopping’s metric to MAE and use a molecule-disjoint split *within each type* so validation better matches the test setup. This should reduce the leaderboard error substantially (move down from ~3.98 toward your 0.6666 target) without changing feature engineering or the model family.'
- What this solution (achieved 1.9121) has done: 'You’re far from the target (1.81121 vs 0.66661; lower is better), so we need a small but meaningful generalization improvement without changing the core approach (LightGBM per coupling type on the same features). The biggest win with minimal logic change is to train per-type on a molecule-disjoint **train/valid split closer to typical practice (80/20 instead of 60/40)** and to make early stopping less twitchy by increasing patience slightly; both help select a more robust number of trees and usually reduce leaderboard error. I also use `metric='l1'`/`objective='regression_l1'` (still MAE) to align optimization more directly with the competition metric while keeping the same model family and features. The submission writing and feature pipeline stay identical, and it still produces `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import gc
import os

import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn import metrics

np.random.seed(420)



## === cell 1
print(os.listdir("../input"))



## === cell 2
BASE = "../input/champs-scalar-coupling"
if not os.path.exists(BASE):
    BASE = "../input"

train_original = pd.read_csv(
    os.path.join(BASE, "train.csv"),
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
test_original = pd.read_csv(
    os.path.join(BASE, "test.csv"),
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
structures_original = pd.read_csv(
    os.path.join(BASE, "structures.csv"),
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



## === cell 3
train_original.head()



## === cell 4
structures_original.head()



## === cell 5
test_original.head()



## === cell 6
structures_original[structures_original["molecule_name"] == "dsgdb9nsd_000015"]



## === cell 7
moleculeCount = (
    structures_original.groupby(["molecule_name", "atom"], observed=True)["atom"]
    .count()
    .rename("count")
    .unstack(fill_value=0)
    .reset_index()
)

moleculeCount.head()



## === cell 8
moleculeCount[moleculeCount["molecule_name"] == "dsgdb9nsd_000015"]



## === cell 9
structures = structures_original.merge(moleculeCount, on="molecule_name", how="inner")

structures.head()



## === cell 10

all_mols = (
    pd.Index(train_original["molecule_name"].cat.categories)
    .union(pd.Index(test_original["molecule_name"].cat.categories))
    .union(pd.Index(structures_original["molecule_name"].cat.categories))
)
train_original["molecule_name"] = train_original["molecule_name"].cat.set_categories(
    all_mols
)
test_original["molecule_name"] = test_original["molecule_name"].cat.set_categories(
    all_mols
)
structures_original["molecule_name"] = structures_original[
    "molecule_name"
].cat.set_categories(all_mols)

structures = structures_original.merge(
    moleculeCount, on="molecule_name", how="inner", sort=False, copy=False
)

mol_code_struct = structures["molecule_name"].cat.codes.to_numpy(np.int32, copy=False)
atom_index_struct = structures["atom_index"].to_numpy(np.int16, copy=False)

order = np.lexsort((atom_index_struct, mol_code_struct))
mol_sorted = mol_code_struct[order]
atom_sorted = atom_index_struct[order]

x_arr = structures["x"].to_numpy(np.float32, copy=False)
y_arr = structures["y"].to_numpy(np.float32, copy=False)
z_arr = structures["z"].to_numpy(np.float32, copy=False)
atom_cat = structures["atom"]  # category
C_arr = structures["C"].to_numpy(np.int16, copy=False)
F_arr = structures["F"].to_numpy(np.int16, copy=False)
H_arr = structures["H"].to_numpy(np.int16, copy=False)
N_arr = structures["N"].to_numpy(np.int16, copy=False)
O_arr = structures["O"].to_numpy(np.int16, copy=False)

atom_codes = atom_cat.cat.codes.to_numpy(np.int16, copy=False)
atom_categories = atom_cat.cat.categories.to_numpy()


def _gather_pair_features(df_in: pd.DataFrame) -> pd.DataFrame:
    mol_code = df_in["molecule_name"].cat.codes.to_numpy(np.int32, copy=False)
    a0 = df_in["atom_index_0"].to_numpy(np.int16, copy=False)
    a1 = df_in["atom_index_1"].to_numpy(np.int16, copy=False)

    SHIFT = 128
    key_sorted = mol_sorted.astype(np.int64) * SHIFT + atom_sorted.astype(np.int64)

    key0 = mol_code.astype(np.int64) * SHIFT + a0.astype(np.int64)
    key1 = mol_code.astype(np.int64) * SHIFT + a1.astype(np.int64)

    pos0 = np.searchsorted(key_sorted, key0)
    pos1 = np.searchsorted(key_sorted, key1)

    ok0 = (pos0 < key_sorted.size) & (key_sorted[pos0] == key0)
    ok1 = (pos1 < key_sorted.size) & (key_sorted[pos1] == key1)

    idx0 = np.where(ok0, order[pos0], -1)
    idx1 = np.where(ok1, order[pos1], -1)

    out = df_in.copy()

    out = out.rename(columns={"atom_index_0": "atom_0", "atom_index_1": "atom_1"})

    atom_nm_0 = np.empty(out.shape[0], dtype=object)
    atom_nm_1 = np.empty(out.shape[0], dtype=object)
    atom_nm_0[:] = None
    atom_nm_1[:] = None

    atom_nm_0[ok0] = atom_categories[atom_codes[idx0[ok0]]]
    atom_nm_1[ok1] = atom_categories[atom_codes[idx1[ok1]]]

    out["atom_nm_0"] = pd.Series(atom_nm_0, index=out.index, dtype="category")
    out["atom_nm_1"] = pd.Series(atom_nm_1, index=out.index, dtype="category")

    x0 = np.zeros(out.shape[0], dtype=np.float32)
    y0 = np.zeros(out.shape[0], dtype=np.float32)
    z0 = np.zeros(out.shape[0], dtype=np.float32)
    x1 = np.zeros(out.shape[0], dtype=np.float32)
    y1 = np.zeros(out.shape[0], dtype=np.float32)
    z1 = np.zeros(out.shape[0], dtype=np.float32)

    x0[ok0] = x_arr[idx0[ok0]]
    y0[ok0] = y_arr[idx0[ok0]]
    z0[ok0] = z_arr[idx0[ok0]]
    x1[ok1] = x_arr[idx1[ok1]]
    y1[ok1] = y_arr[idx1[ok1]]
    z1[ok1] = z_arr[idx1[ok1]]

    out["x_0"] = x0
    out["y_0"] = y0
    out["z_0"] = z0
    out["x_1"] = x1
    out["y_1"] = y1
    out["z_1"] = z1

    C = np.zeros(out.shape[0], dtype=np.int16)
    F = np.zeros(out.shape[0], dtype=np.int16)
    H = np.zeros(out.shape[0], dtype=np.int16)
    N = np.zeros(out.shape[0], dtype=np.int16)
    O = np.zeros(out.shape[0], dtype=np.int16)
    C[ok1] = C_arr[idx1[ok1]]
    F[ok1] = F_arr[idx1[ok1]]
    H[ok1] = H_arr[idx1[ok1]]
    N[ok1] = N_arr[idx1[ok1]]
    O[ok1] = O_arr[idx1[ok1]]
    out["C"] = C
    out["F"] = F
    out["H"] = H
    out["N"] = N
    out["O"] = O

    out.reset_index(inplace=True, drop=True)
    return out


train = _gather_pair_features(train_original)
test = _gather_pair_features(test_original)

train.head()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3831055262.py in <cell line: 0>()
     34 
     35 # Prepare keys for fast lookup
---> 36 mol_code_struct = structures["molecule_name"].cat.codes.to_numpy(np.int32, copy=False)
     37 atom_index_struct = structures["atom_index"].to_numpy(np.int16, copy=False)
     38 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/accessor.py in __get__(self, obj, cls)
    222             # we're accessing the attribute of the class, i.e., Dataset.geo
    223             return self._accessor
--> 224         accessor_obj = self._accessor(obj)
    225         # Replace the property with the accessor object. Inspired by:
    226         # https://www.pydanny.com/cached-property.html

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/categorical.py in __init__(self, data)
   2896 
   2897     def __init__(self, data) -> None:
-> 2898         self._validate(data)
   2899         self._parent = data.values
   2900         self._index = data.index

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/categorical.py in _validate(data)
   2905     def _validate(data):
   2906         if not isinstance(data.dtype, CategoricalDtype):
-> 2907             raise AttributeError("Can only use .cat accessor with a 'category' dtype")
   2908 
   2909     def _delegate_property_get(self, name: str):

AttributeError: Can only use .cat accessor with a 'category' dtype

## === cell 11
test.head()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3214727096.py in <cell line: 0>()
----> 1 test.head()
      2 

NameError: name 'test' is not defined

## === cell 12
train_original = None
del train_original
structures_original = None
del structures_original
test_original = None
del test_original
moleculeCount = None
del moleculeCount
structures = None
del structures
gc.collect()



## === cell 13
for df in (train, test):
    for c in ["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"]:
        if df[c].isna().any():
            df[c] = df[c].fillna(0.0)

a0 = train[["x_0", "y_0", "z_0"]].to_numpy(dtype=np.float32, copy=False)
a1 = train[["x_1", "y_1", "z_1"]].to_numpy(dtype=np.float32, copy=False)
train["dist"] = np.sqrt(((a0 - a1) ** 2).sum(axis=1)).astype(np.float32, copy=False)

b0 = test[["x_0", "y_0", "z_0"]].to_numpy(dtype=np.float32, copy=False)
b1 = test[["x_1", "y_1", "z_1"]].to_numpy(dtype=np.float32, copy=False)
test["dist"] = np.sqrt(((b0 - b1) ** 2).sum(axis=1)).astype(np.float32, copy=False)

train.drop(columns=["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"], inplace=True)
test.drop(columns=["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"], inplace=True)

del a0, a1, b0, b1
gc.collect()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1637881825.py in <cell line: 0>()
----> 1 for df in (train, test):
      2     for c in ["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"]:
      3         if df[c].isna().any():
      4             df[c] = df[c].fillna(0.0)
      5 

NameError: name 'train' is not defined

## === cell 14
for df in (train, test):
    left = df["atom_nm_0"].astype("string")
    right = df["atom_nm_1"].astype("string")
    df["atom_pair"] = (left + "_" + right).astype("category")

for col in ["type", "atom_nm_0", "atom_nm_1", "atom_pair"]:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")
    all_cats = pd.Index(train[col].cat.categories).union(
        pd.Index(test[col].cat.categories)
    )
    train[col] = train[col].cat.set_categories(all_cats)
    test[col] = test[col].cat.set_categories(all_cats)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1486274835.py in <cell line: 0>()
      1 # SPEED FIX (equivalent): build atom_pair using category values without per-row astype(str)
      2 # and unify categories once. Resulting categories/values are equivalent to string concatenation.
----> 3 for df in (train, test):
      4     left = df["atom_nm_0"].astype("string")
      5     right = df["atom_nm_1"].astype("string")

NameError: name 'train' is not defined

## === cell 15
train.head()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2745801949.py in <cell line: 0>()
----> 1 train.head()
      2 

NameError: name 'train' is not defined

## === cell 16
test.head()



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3214727096.py in <cell line: 0>()
----> 1 test.head()
      2 

NameError: name 'test' is not defined

## === cell 17
feature_cols = [
    "atom_0",
    "atom_1",
    "type",
    "atom_nm_0",
    "atom_nm_1",
    "atom_pair",
    "C",
    "F",
    "H",
    "N",
    "O",
    "dist",
]
X = train[feature_cols]
y = train["scalar_coupling_constant"]



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1570839410.py in <cell line: 0>()
     13     "dist",
     14 ]
---> 15 X = train[feature_cols]
     16 y = train["scalar_coupling_constant"]
     17 

NameError: name 'train' is not defined

## === cell 18
rng = np.random.RandomState(420)

params = {
    "boosting_type": "gbdt",
    "objective": "regression_l1",
    "learning_rate": 0.05,
    "num_leaves": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "reg_alpha": 0.1,
    "reg_lambda": 0.3,
    "metric": "l1",
    "num_threads": int(os.environ.get("OMP_NUM_THREADS", "0")) or os.cpu_count(),
    "feature_pre_filter": False,
}

cat_features = ["type", "atom_nm_0", "atom_nm_1", "atom_pair"]
cat_idx = [feature_cols.index(c) for c in cat_features]

train_types = train["type"].astype(str)
test_types = test["type"].astype(str)
all_types = sorted(
    pd.Index(pd.concat([train_types, test_types], axis=0).unique()).tolist()
)

train_type_groups = train.groupby(train_types, sort=False).indices
test_type_groups = test.groupby(test_types, sort=False).indices

X_train_df = X  # no copy; we'll replace columns in a shallow copy to avoid modifying original frame
X_test_df = test[feature_cols]

X_train_enc = X_train_df.copy()
X_test_enc = X_test_df.copy()
for c in cat_features:
    X_train_enc[c] = X_train_enc[c].cat.codes.astype(np.int32, copy=False)
    X_test_enc[c] = X_test_enc[c].cat.codes.astype(np.int32, copy=False)

X_train_np = np.ascontiguousarray(X_train_enc.to_numpy(copy=False))
y_np = y.to_numpy(dtype=np.float32, copy=False)
X_test_np = np.ascontiguousarray(X_test_enc.to_numpy(copy=False))

train_mol_codes = train["molecule_name"].cat.codes.to_numpy(copy=False)

preds_test = np.empty(test.shape[0], dtype=np.float64)
preds_test[:] = np.nan

oof_mae = []

for t in all_types:
    train_rows = train_type_groups.get(t, np.array([], dtype=np.int64))
    test_rows = test_type_groups.get(t, np.array([], dtype=np.int64))

    if train_rows.size == 0:
        if test_rows.size:
            preds_test[test_rows] = 0.0
        continue

    mol_codes_t = train_mol_codes[train_rows]
    uniq_codes = np.unique(mol_codes_t)
    rng.shuffle(uniq_codes)

    split = int(len(uniq_codes) * 0.8)
    train_codes_arr = uniq_codes[:split]
    valid_codes_arr = uniq_codes[split:]  # kept for semantic equivalence (unused)

    train_codes_arr.sort()
    pos = np.searchsorted(train_codes_arr, mol_codes_t)
    train_mask_t = (pos < train_codes_arr.size) & (train_codes_arr[pos] == mol_codes_t)
    valid_mask_t = ~train_mask_t

    tr_idx = train_rows[train_mask_t]
    va_idx = train_rows[valid_mask_t]

    lgb_train = lgb.Dataset(
        X_train_np[tr_idx],
        y_np[tr_idx],
        categorical_feature=cat_idx,
        free_raw_data=True,
    )
    lgb_eval = lgb.Dataset(
        X_train_np[va_idx],
        y_np[va_idx],
        categorical_feature=cat_idx,
        free_raw_data=True,
    )

    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_eval],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
        keep_training_booster=False,
    )

    y_pred_valid = gbm.predict(X_train_np[va_idx], num_iteration=gbm.best_iteration)
    mae_t = metrics.mean_absolute_error(y_np[va_idx], y_pred_valid)
    oof_mae.append(mae_t)

    if test_rows.size:
        preds_test[test_rows] = gbm.predict(
            X_test_np[test_rows], num_iteration=gbm.best_iteration
        )

    del lgb_train, lgb_eval, gbm, y_pred_valid, train_rows, test_rows, tr_idx, va_idx
    if (len(oof_mae) % 2) == 0:
        gc.collect()

print("Trained per-type models:", len(all_types))
if len(oof_mae) > 0:
    print("Mean per-type valid MAE (not Kaggle metric):", float(np.mean(oof_mae)))



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2713004098.py in <cell line: 0>()
     25 cat_idx = [feature_cols.index(c) for c in cat_features]
     26 
---> 27 train_types = train["type"].astype(str)
     28 test_types = test["type"].astype(str)
     29 all_types = sorted(

NameError: name 'train' is not defined

## === cell 19
if np.isnan(preds_test).any():
    missing = int(np.isnan(preds_test).sum())
    raise ValueError(
        f"Missing predictions for {missing} test rows; cannot write a valid submission."
    )

submission_df = pd.DataFrame(
    {
        "id": test["id"].to_numpy(copy=False),
        "scalar_coupling_constant": preds_test,
    }
)

submission_df.sort_values("id", inplace=True)
submission_df.to_csv("submission.csv", header=True, index=False)

print("Wrote submission.csv with shape:", submission_df.shape)
submission_df.head(10)

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1323702686.py in <cell line: 0>()
----> 1 if np.isnan(preds_test).any():
      2     missing = int(np.isnan(preds_test).sum())
      3     raise ValueError(
      4         f"Missing predictions for {missing} test rows; cannot write a valid submission."
      5     )

NameError: name 'preds_test' is not defined
