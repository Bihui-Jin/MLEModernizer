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

2.93849

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.99777) has done: 'I make the notebook run end-to-end and actually train/predict by fixing three blocking issues: (1) replace deprecated `normed` in `plt.hist` with `density`, (2) update LightGBM early-stopping to the v4.6 callback API so `fit()` works, and (3) fix feature engineering/column alignment bugs (your test `distance` mistakenly used `train`, and dummy columns differed between train/test). I also ensure the submission uses the correct filename suffix `.csv` and required columns (`id, scalar_coupling_constant`) with ids aligned to the original `df_test`. These are correctness/stability fixes and should also improve score versus the current “not yielded” state (since the model finally train and the test features be correct).'
- What this solution (achieved 1.99777) has done: 'The timeout is dominated by two huge pandas merges that create very wide intermediate frames and by Python-level loops for atom encoding, plus LightGBM training using suboptimal threading settings. I keep the exact same feature set and modeling approach, but speed up feature creation by replacing the merges with equivalent vectorized index-based joins on a pre-indexed `structures` table and by using a vectorized map for atom-to-number encoding. I also enforce aligned dummy columns in one step for both train/test (same semantics) and configure LightGBM to use all CPU cores and avoid extra overhead while preserving the same early-stopping training logic and evaluation behavior. All file paths, model type, and core training flow remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

np.random.seed(42)



## === cell 1
DATA_DIR = "../input/champs-scalar-coupling"

train_cols = [
    "id",
    "molecule_name",
    "atom_index_0",
    "atom_index_1",
    "type",
    "scalar_coupling_constant",
]
test_cols = ["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]
structures_cols = ["molecule_name", "atom_index", "atom", "x", "y", "z"]

df_train = pd.read_csv(
    f"{DATA_DIR}/train.csv",
    usecols=train_cols,
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
        "scalar_coupling_constant": np.float32,
    },
)
df_test = pd.read_csv(
    f"{DATA_DIR}/test.csv",
    usecols=test_cols,
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
structures = pd.read_csv(
    f"{DATA_DIR}/structures.csv",
    usecols=structures_cols,
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
)



## === cell 2
sample_submission = pd.read_csv(
    f"{DATA_DIR}/sample_submission.csv", usecols=["id", "scalar_coupling_constant"]
)



## === cell 3
pass



## === cell 4
pass



## === cell 5
pass



## === cell 6
pass



## === cell 7
pass



## === cell 8
pass



## === cell 9
pass



## === cell 10
plt = None
sns = None



## === cell 11
pass



## === cell 12
pass



## === cell 13
train_mol_cats = df_train["molecule_name"].cat.categories
df_test["molecule_name"] = df_test["molecule_name"].cat.set_categories(train_mol_cats)
structures["molecule_name"] = structures["molecule_name"].cat.set_categories(
    train_mol_cats
)

s = structures.copy()
s["atom"] = s["atom"].astype("category")
s = s.set_index(["molecule_name", "atom_index"]).sort_index()

s0 = s.rename(columns={"atom": "atom_x", "x": "x_x", "y": "y_x", "z": "z_x"})
train = df_train.join(
    s0[["atom_x", "x_x", "y_x", "z_x"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
test = df_test.join(
    s0[["atom_x", "x_x", "y_x", "z_x"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)

s1 = s.rename(columns={"atom": "atom_y", "x": "x_y", "y": "y_y", "z": "z_y"})
train = train.join(
    s1[["atom_y", "x_y", "y_y", "z_y"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)
test = test.join(
    s1[["atom_y", "x_y", "y_y", "z_y"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)



## === cell 14
pass



## === cell 15
pass



## === cell 16
pass



## === cell 17
train = train.drop(["molecule_name"], axis=1)
test = test.drop(["molecule_name"], axis=1)



## === cell 18
pass




## === cell 19
def atom_number(atom):
    if atom == "H":
        return 0
    elif atom == "C":
        return 1
    elif atom == "N":
        return 2
    elif atom == "O":
        return 3
    elif atom == "F":
        return 4
    return -1




## === cell 20
_atom_map = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}

train["atom_y"] = train["atom_y"].map(_atom_map).fillna(-1).astype(np.int8)
train["atom_x"] = train["atom_x"].map(_atom_map).fillna(-1).astype(np.int8)
test["atom_y"] = test["atom_y"].map(_atom_map).fillna(-1).astype(np.int8)
test["atom_x"] = test["atom_x"].map(_atom_map).fillna(-1).astype(np.int8)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2507891035.py in <cell line: 0>()
      2 _atom_map = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
      3 
----> 4 train["atom_y"] = train["atom_y"].map(_atom_map).fillna(-1).astype(np.int8)
      5 train["atom_x"] = train["atom_x"].map(_atom_map).fillna(-1).astype(np.int8)
      6 test["atom_y"] = test["atom_y"].map(_atom_map).fillna(-1).astype(np.int8)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in fillna(self, value, method, axis, inplace, limit, downcast)
   7347                     )
   7348 
-> 7349                 new_data = self._mgr.fillna(
   7350                     value=value, limit=limit, inplace=inplace, downcast=downcast
   7351                 )

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/base.py in fillna(self, value, limit, inplace, downcast)
    184             limit = libalgos.validate_limit(None, limit=limit)
    185 
--> 186         return self.apply_with_block(
    187             "fillna",
    188             value=value,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in fillna(self, value, limit, inplace, downcast, using_cow, already_warned)
   2332                 # 3rd party EA that has not implemented copy keyword yet
   2333                 refs = None
-> 2334                 new_values = self.values.fillna(value=value, method=None, limit=limit)
   2335                 # issue the warning *after* retrying, in case the TypeError
   2336                 #  was caused by an invalid fill_value

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/_mixins.py in fillna(self, value, method, limit, copy)
    374             # We validate the fill_value even if there is nothing to fill
    375             if value is not None:
--> 376                 self._validate_setitem_value(value)
    377 
    378             if not copy:

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/categorical.py in _validate_setitem_value(self, value)
   1587             return self._validate_listlike(value)
   1588         else:
-> 1589             return self._validate_scalar(value)
   1590 
   1591     def _validate_scalar(self, fill_value):

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/categorical.py in _validate_scalar(self, fill_value)
   1612             fill_value = self._unbox_scalar(fill_value)
   1613         else:
-> 1614             raise TypeError(
   1615                 "Cannot setitem on a Categorical with a new "
   1616                 f"category ({fill_value}), set the categories first"

TypeError: Cannot setitem on a Categorical with a new category (-1), set the categories first

## === cell 21
type_cats = pd.api.types.union_categoricals(
    [df_train["type"], df_test["type"]]
).categories
train["type"] = train["type"].cat.set_categories(type_cats)
test["type"] = test["type"].cat.set_categories(type_cats)

X_train_full = pd.get_dummies(
    train.drop(columns=["scalar_coupling_constant"]),
    columns=["type"],
    drop_first=True,
)
y_train_full = train["scalar_coupling_constant"].copy()

X_test_full = pd.get_dummies(
    test,
    columns=["type"],
    drop_first=True,
)

X_train_full, X_test_full = X_train_full.align(
    X_test_full, join="outer", axis=1, fill_value=0
)

uint8_cols_tr = X_train_full.dtypes == np.uint8
if uint8_cols_tr.any():
    cols = X_train_full.columns[uint8_cols_tr]
    X_train_full[cols] = X_train_full[cols].astype(np.int8, copy=False)

uint8_cols_te = X_test_full.dtypes == np.uint8
if uint8_cols_te.any():
    cols = X_test_full.columns[uint8_cols_te]
    X_test_full[cols] = X_test_full[cols].astype(np.int8, copy=False)



## === cell 22
pass



## === cell 23
dx = train["x_y"].to_numpy(dtype=np.float32) - train["x_x"].to_numpy(dtype=np.float32)
dy = train["y_y"].to_numpy(dtype=np.float32) - train["y_x"].to_numpy(dtype=np.float32)
dz = train["z_y"].to_numpy(dtype=np.float32) - train["z_x"].to_numpy(dtype=np.float32)
train_dist = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32, copy=False)
X_train_full["distance"] = train_dist

dx = test["x_y"].to_numpy(dtype=np.float32) - test["x_x"].to_numpy(dtype=np.float32)
dy = test["y_y"].to_numpy(dtype=np.float32) - test["y_x"].to_numpy(dtype=np.float32)
dz = test["z_y"].to_numpy(dtype=np.float32) - test["z_x"].to_numpy(dtype=np.float32)
test_dist = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32, copy=False)
X_test_full["distance"] = test_dist



## === cell 24
pass



## === cell 25
X_train_full = X_train_full.drop(["id"], axis=1)
X_test_full = X_test_full.drop(["id"], axis=1)



## === cell 26
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)



## === cell 27
from lightgbm import LGBMRegressor
import lightgbm as lgb



## === cell 28
X_tr_np = np.ascontiguousarray(X_tr.to_numpy())
X_val_np = np.ascontiguousarray(X_val.to_numpy())
y_tr_np = y_tr.to_numpy(dtype=np.float32, copy=False)
y_val_np = y_val.to_numpy(dtype=np.float32, copy=False)

model = LGBMRegressor(
    random_state=42,
    n_estimators=10000,
    n_jobs=-1,
)
model.fit(
    X_tr_np,
    y_tr_np,
    eval_set=[(X_val_np, y_val_np)],
    eval_metric="l1",
    callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=50)],
)



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2820424567.py in <cell line: 0>()
     11     n_jobs=-1,
     12 )
---> 13 model.fit(
     14     X_tr_np,
     15     y_tr_np,

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in fit(self, X, y, sample_weight, init_score, eval_set, eval_names, eval_sample_weight, eval_init_score, eval_metric, feature_name, categorical_feature, callbacks, init_model)
   1396     ) -> "LGBMRegressor":
   1397         """Docstring is inherited from the LGBMModel."""
-> 1398         super().fit(
   1399             X,
   1400             y,

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in fit(self, X, y, sample_weight, init_score, group, eval_set, eval_names, eval_sample_weight, eval_class_weight, eval_init_score, eval_group, eval_metric, feature_name, categorical_feature, callbacks, init_model)
    947 
    948         if not isinstance(X, (pd_DataFrame, dt_DataTable)):
--> 949             _X, _y = _LGBMValidateData(
    950                 self,
    951                 X,

/usr/local/lib/python3.11/dist-packages/lightgbm/compat.py in validate_data(_estimator, X, y, accept_sparse, ensure_all_finite, ensure_min_samples, **ignored_kwargs)
     76                 )
     77             else:
---> 78                 X, y = check_X_y(
     79                     X,
     80                     y,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1104         )
   1105 
-> 1106     X = check_array(
   1107         X,
   1108         accept_sparse=accept_sparse,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    877                     array = xp.astype(array, dtype, copy=False)
    878                 else:
--> 879                     array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
    880             except ComplexWarning as complex_warning:
    881                 raise ValueError(

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_array_api.py in _asarray_with_order(array, dtype, order, copy, xp)
    183     if xp.__name__ in {"numpy", "numpy.array_api"}:
    184         # Use NumPy API to support order
--> 185         array = numpy.asarray(array, order=order, dtype=dtype)
    186         return xp.asarray(array, copy=copy)
    187     else:

ValueError: could not convert string to float: 'H'

## === cell 29
preds_val = model.predict(X_val_np, num_iteration=model.best_iteration_)



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/1964169295.py in <cell line: 0>()
----> 1 preds_val = model.predict(X_val_np, num_iteration=model.best_iteration_)
      2 

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in best_iteration_(self)
   1202         """:obj:`int`: The best iteration of fitted model if ``early_stopping()`` callback has been specified."""
   1203         if not self.__sklearn_is_fitted__():
-> 1204             raise LGBMNotFittedError(
   1205                 "No best_iteration found. Need to call fit with early_stopping callback beforehand."
   1206             )

NotFittedError: No best_iteration found. Need to call fit with early_stopping callback beforehand.

## === cell 30
X_test_np = np.ascontiguousarray(X_test_full.to_numpy())
test_predictions = model.predict(X_test_np, num_iteration=model.best_iteration_)



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/3984316571.py in <cell line: 0>()
      1 # Speed: predict on contiguous numpy once.
      2 X_test_np = np.ascontiguousarray(X_test_full.to_numpy())
----> 3 test_predictions = model.predict(X_test_np, num_iteration=model.best_iteration_)
      4 

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in best_iteration_(self)
   1202         """:obj:`int`: The best iteration of fitted model if ``early_stopping()`` callback has been specified."""
   1203         if not self.__sklearn_is_fitted__():
-> 1204             raise LGBMNotFittedError(
   1205                 "No best_iteration found. Need to call fit with early_stopping callback beforehand."
   1206             )

NotFittedError: No best_iteration found. Need to call fit with early_stopping callback beforehand.

## === cell 31
pass



## === cell 32
submission = pd.DataFrame(
    {"id": df_test["id"].values, "scalar_coupling_constant": test_predictions}
)
submission = submission[["id", "scalar_coupling_constant"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3746907053.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"id": df_test["id"].values, "scalar_coupling_constant": test_predictions}
      3 )
      4 submission = submission[["id", "scalar_coupling_constant"]]
      5 submission.to_csv("submission.csv", index=False)

NameError: name 'test_predictions' is not defined
