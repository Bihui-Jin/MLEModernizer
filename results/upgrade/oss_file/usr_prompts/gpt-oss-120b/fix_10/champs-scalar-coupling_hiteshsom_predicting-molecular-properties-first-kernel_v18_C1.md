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

category_encoders==2.7.0
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

0.76707

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 2.21775) has done: 'I fixed the merge operations to keep all test rows (using left‑joins), removed the extra “index” column before modeling, and streamlined the workflow so the script runs without errors and writes a proper `submission.csv`. These changes keep the original feature set and model while ensuring a valid submission file is produced.'

# 9. Code solution

## === cell 0
import gc
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgbm

train = pd.read_csv(
    "../input/train.csv",
    dtype={
        "id": np.int32,
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "scalar_coupling_constant": np.float32,
    },
    usecols=[
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
)
test = pd.read_csv(
    "../input/test.csv",
    dtype={"id": np.int32, "atom_index_0": np.int16, "atom_index_1": np.int16},
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
)
sample_sub = pd.read_csv("../input/sample_submission.csv")
structures = pd.read_csv(
    "../input/structures.csv",
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
)

print(f"train.shape: {train.shape}")
print(f"test.shape: {test.shape}")



## === cell 1
y_train = train["scalar_coupling_constant"].copy()
X_train = train.drop(columns=["scalar_coupling_constant", "id"]).copy()
X_test = test.drop(columns=["id"]).copy()



## === cell 2
X_train = X_train.reset_index().rename(columns={"index": "row_idx"})
X_test = X_test.reset_index().rename(columns={"index": "row_idx"})




## === cell 3
def convert_object_to_categories(df_train, df_test):
    for col in df_train.columns:
        if df_train[col].dtype == "O":
            df_train[col] = df_train[col].astype("category")
            df_test[col] = df_test[col].astype("category")
    return df_train, df_test


X_train, X_test = convert_object_to_categories(X_train, X_test)



## === cell 4
structures_idx = structures.set_index(["molecule_name", "atom_index"])

X_train = X_train.join(
    structures_idx,
    on=["molecule_name", "atom_index_0"],
    rsuffix="_0",
    how="left",
)
X_test = X_test.join(
    structures_idx,
    on=["molecule_name", "atom_index_0"],
    rsuffix="_0",
    how="left",
)

X_train = X_train.rename(
    columns={
        "atom_index": "atom_index_0_tmp",
        "x": "atom_index_0_x",
        "y": "atom_index_0_y",
        "z": "atom_index_0_z",
        "atom": "atom_0",
    }
)
X_test = X_test.rename(
    columns={
        "atom_index": "atom_index_0_tmp",
        "x": "atom_index_0_x",
        "y": "atom_index_0_y",
        "z": "atom_index_0_z",
        "atom": "atom_0",
    }
)

X_train = X_train.join(
    structures_idx,
    on=["molecule_name", "atom_index_1"],
    rsuffix="_1",
    how="left",
)
X_test = X_test.join(
    structures_idx,
    on=["molecule_name", "atom_index_1"],
    rsuffix="_1",
    how="left",
)

X_train = X_train.rename(
    columns={
        "atom_index": "atom_index_1_tmp",
        "x": "atom_index_1_x",
        "y": "atom_index_1_y",
        "z": "atom_index_1_z",
        "atom": "atom_1",
    }
)
X_test = X_test.rename(
    columns={
        "atom_index": "atom_index_1_tmp",
        "x": "atom_index_1_x",
        "y": "atom_index_1_y",
        "z": "atom_index_1_z",
        "atom": "atom_1",
    }
)

del structures_idx
gc.collect()



## === cell 5
X_train = X_train.drop(columns=["atom_index_0_tmp", "atom_index_1_tmp"])
X_test = X_test.drop(columns=["atom_index_0_tmp", "atom_index_1_tmp"])



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/801753685.py in <cell line: 0>()
----> 1 X_train = X_train.drop(columns=["atom_index_0_tmp", "atom_index_1_tmp"])
      2 X_test = X_test.drop(columns=["atom_index_0_tmp", "atom_index_1_tmp"])
      3 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   5579                 weight  1.0     0.8
   5580         """
-> 5581         return super().drop(
   5582             labels=labels,
   5583             axis=axis,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   4786         for axis, labels in axes.items():
   4787             if labels is not None:
-> 4788                 obj = obj._drop_axis(labels, axis, level=level, errors=errors)
   4789 
   4790         if inplace:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _drop_axis(self, labels, axis, level, errors, only_slice)
   4828                 new_axis = axis.drop(labels, level=level, errors=errors)
   4829             else:
-> 4830                 new_axis = axis.drop(labels, errors=errors)
   4831             indexer = axis.get_indexer(new_axis)
   4832 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in drop(self, labels, errors)
   7068         if mask.any():
   7069             if errors != "ignore":
-> 7070                 raise KeyError(f"{labels[mask].tolist()} not found in axis")
   7071             indexer = indexer[~mask]
   7072         return self.delete(indexer)

KeyError: "['atom_index_0_tmp', 'atom_index_1_tmp'] not found in axis"

## === cell 6
for col in [
    "atom_index_0_x",
    "atom_index_0_y",
    "atom_index_0_z",
    "atom_index_1_x",
    "atom_index_1_y",
    "atom_index_1_z",
]:
    X_train[col] = X_train[col].astype(np.float32)
    X_test[col] = X_test[col].astype(np.float32)

X_train["distance"] = np.sqrt(
    (X_train["atom_index_0_x"] - X_train["atom_index_1_x"]) ** 2
    + (X_train["atom_index_0_y"] - X_train["atom_index_1_y"]) ** 2
    + (X_train["atom_index_0_z"] - X_train["atom_index_1_z"]) ** 2
)
X_test["distance"] = np.sqrt(
    (X_test["atom_index_0_x"] - X_test["atom_index_1_x"]) ** 2
    + (X_test["atom_index_0_y"] - X_test["atom_index_1_y"]) ** 2
    + (X_test["atom_index_0_z"] - X_test["atom_index_1_z"]) ** 2
)

X_train["join_type"] = X_train["type"].str.slice(0, 2)
X_test["join_type"] = X_test["type"].str.slice(0, 2)

X_train["num_bonds"] = X_train["type"].str.slice(0, 1).astype(np.int8)
X_test["num_bonds"] = X_test["type"].str.slice(0, 1).astype(np.int8)



## === cell 7
X_train["num_atoms"] = (
    X_train.groupby("molecule_name")["atom_index_0"].transform("max") + 1
)
X_test["num_atoms"] = (
    X_test.groupby("molecule_name")["atom_index_0"].transform("max") + 1
)




## === cell 8
def angle_between_vectors(df):
    dot = (
        df["atom_index_0_x"] * df["atom_index_1_x"]
        + df["atom_index_0_y"] * df["atom_index_1_y"]
        + df["atom_index_0_z"] * df["atom_index_1_z"]
    )
    mag0 = np.sqrt(
        df["atom_index_0_x"] ** 2
        + df["atom_index_0_y"] ** 2
        + df["atom_index_0_z"] ** 2
    )
    mag1 = np.sqrt(
        df["atom_index_1_x"] ** 2
        + df["atom_index_1_y"] ** 2
        + df["atom_index_1_z"] ** 2
    )
    cos_angle = dot / (mag0 * mag1 + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    df["angle"] = np.arccos(cos_angle).astype(np.float32)
    return df


X_train = angle_between_vectors(X_train)
X_test = angle_between_vectors(X_test)



## === cell 9
X_train, X_test = convert_object_to_categories(X_train, X_test)



## === cell 10
if "type" in X_train.columns:
    X_train = X_train.drop(columns=["type"])
if "type" in X_test.columns:
    X_test = X_test.drop(columns=["type"])

if "row_idx" in X_train.columns:
    X_train = X_train.drop(columns=["row_idx"])
if "row_idx" in X_test.columns:
    X_test = X_test.drop(columns=["row_idx"])

X_test = X_test.reindex(columns=X_train.columns)

numeric_cols = X_train.select_dtypes(
    include=["int16", "int32", "int64", "float32", "float64"]
).columns
X_train[numeric_cols] = X_train[numeric_cols].fillna(-999)
X_test[numeric_cols] = X_test[numeric_cols].fillna(-999)

categorical_cols = X_train.select_dtypes(include=["category"]).columns
for col in categorical_cols:
    if "missing" not in X_train[col].cat.categories:
        X_train[col] = X_train[col].cat.add_categories("missing")
        X_test[col] = X_test[col].cat.add_categories("missing")
    X_train[col] = X_train[col].fillna("missing")
    X_test[col] = X_test[col].fillna("missing")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2542753496.py in <cell line: 0>()
     22     if "missing" not in X_train[col].cat.categories:
     23         X_train[col] = X_train[col].cat.add_categories("missing")
---> 24         X_test[col] = X_test[col].cat.add_categories("missing")
     25     X_train[col] = X_train[col].fillna("missing")
     26     X_test[col] = X_test[col].fillna("missing")

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
kf = KFold(n_splits=5, shuffle=True, random_state=42)
preds = np.zeros(len(X_test))

lgb_params = {
    "n_estimators": 1500,  # reduced max trees; early stopping still caps actual count
    "learning_rate": 0.03,
    "num_leaves": 255,
    "max_bin": 255,  # limits histogram bins -> faster training
    "objective": "regression",
    "random_state": 42,
    "n_jobs": -1,
    "metric": "mae",
    "bagging_fraction": 0.8,
    "feature_fraction": 0.8,
    "verbosity": -1,
}

cat_features = [c for c in X_train.columns if X_train[c].dtype.name == "category"]

for train_idx, val_idx in kf.split(X_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = lgbm.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        categorical_feature=cat_features,
        callbacks=[lgbm.early_stopping(stopping_rounds=50, verbose=False)],
    )
    preds += model.predict(X_test) / kf.n_splits

y_predict = preds



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2363835895.py in <cell line: 0>()
     31         callbacks=[lgbm.early_stopping(stopping_rounds=50, verbose=False)],
     32     )
---> 33     preds += model.predict(X_test) / kf.n_splits
     34 
     35 y_predict = preds

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in predict(self, X, raw_score, start_iteration, num_iteration, pred_leaf, pred_contrib, validate_features, **kwargs)
   1142         predict_params["num_threads"] = self._process_n_jobs(predict_params["num_threads"])
   1143 
-> 1144         return self._Booster.predict(  # type: ignore[union-attr]
   1145             X,
   1146             raw_score=raw_score,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in predict(self, data, start_iteration, num_iteration, raw_score, pred_leaf, pred_contrib, data_has_header, validate_features, **kwargs)
   4765             else:
   4766                 num_iteration = -1
-> 4767         return predictor.predict(
   4768             data=data,
   4769             start_iteration=start_iteration,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in predict(self, data, start_iteration, num_iteration, raw_score, pred_leaf, pred_contrib, data_has_header, validate_features)
   1156 
   1157         if isinstance(data, pd_DataFrame):
-> 1158             data = _data_from_pandas(
   1159                 data=data,
   1160                 feature_name="auto",

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _data_from_pandas(data, feature_name, categorical_feature, pandas_categorical)
    849     else:
    850         if len(cat_cols) != len(pandas_categorical):
--> 851             raise ValueError("train and valid dataset categorical_feature do not match.")
    852         for col, category in zip(cat_cols, pandas_categorical):
    853             if list(data[col].cat.categories) != list(category):

ValueError: train and valid dataset categorical_feature do not match.

## === cell 12
sample_sub["scalar_coupling_constant"] = y_predict
sample_sub.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1026294186.py in <cell line: 0>()
----> 1 sample_sub["scalar_coupling_constant"] = y_predict
      2 sample_sub.to_csv("submission.csv", index=False)
      3 print("Submission saved to submission.csv")

NameError: name 'y_predict' is not defined
