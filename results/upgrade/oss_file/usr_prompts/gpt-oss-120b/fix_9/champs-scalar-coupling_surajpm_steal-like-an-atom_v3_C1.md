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
numpy==1.26.4
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

-1.314363717973649

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.23566) has done: 'I replaced the missing external submission files with a simple but valid baseline: the mean `scalar_coupling_constant` for each coupling type computed from the training data. The script now reads the competition data, creates these per‑type averages, merges them onto the test set, and writes a correctly formatted `submission.csv`. This fixes the FileNotFoundError and ensures a proper submission file is produced while keeping the original modelling logic untouched.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import gc

input_dir = "/kaggle/input"
comp_dir = os.path.join(input_dir, "champs-scalar-coupling")
if not os.path.isdir(comp_dir):
    comp_dir = os.path.join(input_dir, "data", "champs-scalar-coupling")
print("Using competition directory:", comp_dir)

ATOM_NUM = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}


def atom_to_num(sym):
    return ATOM_NUM.get(sym, 0)




## === cell 1
train_path = os.path.join(comp_dir, "train.csv")
test_path = os.path.join(comp_dir, "test.csv")

train_df = pd.read_csv(
    train_path,
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
        "scalar_coupling_constant": np.float32,
    },
)
test_df = pd.read_csv(
    test_path,
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)




## === cell 2
structures = pd.read_csv(
    os.path.join(comp_dir, "structures.csv"),
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

potential = pd.read_csv(
    os.path.join(comp_dir, "potential_energy.csv"),
    dtype={"molecule_name": "category", "potential_energy": np.float32},
)

dipole = pd.read_csv(
    os.path.join(comp_dir, "dipole_moments.csv"),
    dtype={
        "molecule_name": "category",
        "X": np.float32,
        "Y": np.float32,
        "Z": np.float32,
    },
)

mulliken = pd.read_csv(
    os.path.join(comp_dir, "mulliken_charges.csv"),
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "mulliken_charge": np.float32,
    },
)


def merge_atom_info(df, atom_idx_col, suffix):
    atom_info = (
        structures.rename(
            columns={
                "atom": f"atom_{suffix}",
                "x": f"x{suffix}",
                "y": f"y{suffix}",
                "z": f"z{suffix}",
            }
        )
    )[
        [
            "molecule_name",
            "atom_index",
            f"atom_{suffix}",
            f"x{suffix}",
            f"y{suffix}",
            f"z{suffix}",
        ]
    ]
    return pd.merge(
        df,
        atom_info,
        left_on=["molecule_name", atom_idx_col],
        right_on=["molecule_name", "atom_index"],
        how="left",
    ).drop(columns=["atom_index"])


train_len = len(train_df)
combined = pd.concat([train_df, test_df], ignore_index=True)

combined = merge_atom_info(combined, "atom_index_0", "0")
combined = merge_atom_info(combined, "atom_index_1", "1")


def compute_distance(df):
    return np.sqrt(
        (df["x0"] - df["x1"]) ** 2
        + (df["y0"] - df["y1"]) ** 2
        + (df["z0"] - df["z1"]) ** 2
    ).astype(np.float32)


combined["distance"] = compute_distance(combined)

combined["atom0_num"] = (
    combined["atom_0"].astype(str).map(ATOM_NUM).fillna(0).astype(np.int16)
)
combined["atom1_num"] = (
    combined["atom_1"].astype(str).map(ATOM_NUM).fillna(0).astype(np.int16)
)

potential = potential.rename(columns={"potential_energy": "pot_energy"})
combined = combined.merge(potential, on="molecule_name", how="left")

dipole["dipole_mag"] = np.sqrt(
    dipole["X"] ** 2 + dipole["Y"] ** 2 + dipole["Z"] ** 2
).astype(np.float32)
combined = combined.merge(
    dipole[["molecule_name", "dipole_mag"]], on="molecule_name", how="left"
)

mulliken0 = mulliken.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "charge0"}
)
mulliken1 = mulliken.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "charge1"}
)

combined = combined.merge(
    mulliken0[["molecule_name", "atom_index_0", "charge0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
combined = combined.merge(
    mulliken1[["molecule_name", "atom_index_1", "charge1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)

train_df = combined.iloc[:train_len].reset_index(drop=True)
test_df = combined.iloc[train_len:].reset_index(drop=True)

train_df["type_code"], type_mapping = pd.factorize(train_df["type"])
type_map = {lbl: idx for idx, lbl in enumerate(type_mapping)}
test_df["type_code"] = test_df["type"].map(type_map).fillna(-1).astype(np.int16)

feature_cols = [
    "distance",
    "atom0_num",
    "atom1_num",
    "pot_energy",
    "dipole_mag",
    "charge0",
    "charge1",
    "type_code",
]

train_X = train_df[feature_cols].fillna(-1).astype(np.float32).values
test_X = test_df[feature_cols].fillna(-1).astype(np.float32).values
train_y = train_df["scalar_coupling_constant"].values.astype(np.float32)

del combined, structures, potential, dipole, mulliken, train_df, test_df
gc.collect()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/30168919.py in <cell line: 0>()
    124 train_df["type_code"], type_mapping = pd.factorize(train_df["type"])
    125 type_map = {lbl: idx for idx, lbl in enumerate(type_mapping)}
--> 126 test_df["type_code"] = test_df["type"].map(type_map).fillna(-1).astype(np.int16)
    127 
    128 feature_cols = [

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

## === cell 3
X_tr, X_val, y_tr, y_val = train_test_split(
    train_X, train_y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(
    n_estimators=250, learning_rate=0.05, max_depth=4, random_state=42
)

model.fit(X_tr, y_tr)

val_pred = model.predict(X_val)
val_mae = mean_absolute_error(y_val, val_pred)
print(f"Validation MAE: {val_mae:.5f}, log‑MAE: {np.log(val_mae):.5f}")




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1916890164.py in <cell line: 0>()
      1 X_tr, X_val, y_tr, y_val = train_test_split(
----> 2     train_X, train_y, test_size=0.2, random_state=42
      3 )
      4 
      5 model = GradientBoostingRegressor(

NameError: name 'train_X' is not defined

## === cell 4
model.fit(train_X, train_y)

test_pred_model = model.predict(test_X)

type_mean = (
    pd.Series(train_y, index=train_X[:, -1].astype(np.int16)).groupby(level=0).mean()
)
test_type_mean = pd.Series(type_mean).reindex(
    test_X[:, -1].astype(np.int16), fill_value=train_y.mean()
)

test_pred = 0.7 * test_pred_model + 0.3 * test_type_mean.values

submission = pd.DataFrame(
    {
        "id": pd.read_csv(test_path, usecols=["id"])["id"],
        "scalar_coupling_constant": test_pred,
    }
)
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}, shape {submission.shape}")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2444182749.py in <cell line: 0>()
      1 # Retrain on the full training set
----> 2 model.fit(train_X, train_y)
      3 
      4 test_pred_model = model.predict(test_X)
      5 

NameError: name 'model' is not defined
