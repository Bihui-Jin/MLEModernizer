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

-1.6819062841656574

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the failing blending code with a simple, reliable baseline that reads the available training data, computes the average scalar_coupling_constant for each coupling type, and uses these averages to predict the test set (fall‑back to the overall mean when a type is unseen). The script writes a valid CSV (`my_blend_1.csv`) containing the required `id` and `scalar_coupling_constant` columns, eliminating the FileNotFoundError and producing a usable submission.'
- What this solution (achieved 1.23566) has done: 'The fix addresses the KeyError caused by trying to select a non‑existent `scalar_coupling_constant` column from the test set. Instead of indexing that column, we build a MultiIndex from the test rows (`type`, `element_0`, `element_1`) and reindex the pre‑computed combo means to obtain predictions, then fall back to type‑level and global means. This corrects the prediction generation while preserving the original blending logic and ensures a valid CSV submission is written.'
- What this solution (achieved 1.23566) has done: 'I keep the original data loading and merging steps, but add a second‑level fallback based on the mean coupling for each element‑pair (ignoring the coupling type). The prediction hierarchy now tries: (type + elements) → (elements only) → (type) → global mean. This modest extension uses only simple aggregations, preserving the baseline logic while expected to lower the log‑MAE and move the score toward the target.'
- What this solution (achieved 4.00165) has done: 'I keep the original data‑loading and merging steps, add a few cheap numeric features (inter‑atomic distance, dipole magnitude, potential energy, atomic numbers and a numeric encoding of the coupling type) and train a lightweight GradientBoostingRegressor on the residuals of the existing hierarchical mean‑baseline. By predicting a correction to the baseline and adding it back, the model can capture simple patterns missed by the pure averaging scheme, which should lower the log‑MAE toward the target while preserving the overall workflow and still producing a valid `my_blend_2.csv` submission.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

print("Available input directories:", os.listdir("../input"))




## === cell 1
train_path = "../input/champs-scalar-coupling/train.csv"
test_path = "../input/champs-scalar-coupling/test.csv"
structures_path = "../input/champs-scalar-coupling/structures.csv"
dipole_path = "../input/champs-scalar-coupling/dipole_moments.csv"
potential_path = "../input/champs-scalar-coupling/potential_energy.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

struct_df = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "category",
        "atom_index": "int32",
        "atom": "category",
        "x": "float32",
        "y": "float32",
        "z": "float32",
    },
)

struct_df = struct_df.rename(columns={"atom": "element"})[
    ["molecule_name", "atom_index", "element", "x", "y", "z"]
]

dipole_df = pd.read_csv(dipole_path)
potential_df = pd.read_csv(potential_path)

dipole_df["dipole_mag"] = np.sqrt(
    dipole_df["X"] ** 2 + dipole_df["Y"] ** 2 + dipole_df["Z"] ** 2
)




## === cell 2
struct_idx = struct_df.set_index(["molecule_name", "atom_index"])


def enrich_with_struct(df):
    """Add element, x, y, z for both atoms via fast reindex."""
    df = df.copy()
    idx0 = pd.MultiIndex.from_arrays([df["molecule_name"], df["atom_index_0"]])
    df["element_0"] = struct_idx["element"].reindex(idx0).values
    df["x_0"] = struct_idx["x"].reindex(idx0).values
    df["y_0"] = struct_idx["y"].reindex(idx0).values
    df["z_0"] = struct_idx["z"].reindex(idx0).values
    idx1 = pd.MultiIndex.from_arrays([df["molecule_name"], df["atom_index_1"]])
    df["element_1"] = struct_idx["element"].reindex(idx1).values
    df["x_1"] = struct_idx["x"].reindex(idx1).values
    df["y_1"] = struct_idx["y"].reindex(idx1).values
    df["z_1"] = struct_idx["z"].reindex(idx1).values
    return df


train_merged = enrich_with_struct(train_df)
test_merged = enrich_with_struct(test_df)

train_merged = train_merged.merge(
    dipole_df[["molecule_name", "dipole_mag"]], on="molecule_name", how="left"
)
train_merged = train_merged.merge(potential_df, on="molecule_name", how="left")

test_merged = test_merged.merge(
    dipole_df[["molecule_name", "dipole_mag"]], on="molecule_name", how="left"
)
test_merged = test_merged.merge(potential_df, on="molecule_name", how="left")




## === cell 3
atomic_number = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Cl": 17,
    "Br": 35,
    "I": 53,
    "S": 16,
    "P": 15,
}
train_merged["atomic_num_0"] = train_merged["element_0"].map(atomic_number).fillna(0)
train_merged["atomic_num_1"] = train_merged["element_1"].map(atomic_number).fillna(0)
test_merged["atomic_num_0"] = test_merged["element_0"].map(atomic_number).fillna(0)
test_merged["atomic_num_1"] = test_merged["element_1"].map(atomic_number).fillna(0)

train_merged["distance"] = np.sqrt(
    (train_merged["x_0"] - train_merged["x_1"]) ** 2
    + (train_merged["y_0"] - train_merged["y_1"]) ** 2
    + (train_merged["z_0"] - train_merged["z_1"]) ** 2
)
test_merged["distance"] = np.sqrt(
    (test_merged["x_0"] - test_merged["x_1"]) ** 2
    + (test_merged["y_0"] - test_merged["y_1"]) ** 2
    + (test_merged["z_0"] - test_merged["z_1"]) ** 2
)

train_merged["type_code"], type_uniques = pd.factorize(train_merged["type"])
test_merged["type_code"] = type_uniques.get_indexer(test_merged["type"])
unseen = test_merged["type_code"] == -1
if unseen.any():
    new_code = len(type_uniques)
    test_merged.loc[unseen, "type_code"] = new_code




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2640300197.py in <cell line: 0>()
     11     "P": 15,
     12 }
---> 13 train_merged["atomic_num_0"] = train_merged["element_0"].map(atomic_number).fillna(0)
     14 train_merged["atomic_num_1"] = train_merged["element_1"].map(atomic_number).fillna(0)
     15 test_merged["atomic_num_0"] = test_merged["element_0"].map(atomic_number).fillna(0)

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

TypeError: Cannot setitem on a Categorical with a new category (0), set the categories first

## === cell 4
combo_means = train_merged.groupby(["type", "element_0", "element_1"])[
    "scalar_coupling_constant"
].mean()
pair_means = train_merged.groupby(["element_0", "element_1"])[
    "scalar_coupling_constant"
].mean()
type_means = train_df.groupby("type")["scalar_coupling_constant"].mean()
global_mean = train_df["scalar_coupling_constant"].mean()


def baseline_predict(df, combo_means, pair_means, type_means, global_mean):
    key_combo = df.set_index(["type", "element_0", "element_1"]).index
    pred = combo_means.reindex(key_combo).reset_index(drop=True)

    missing = pred.isna()
    if missing.any():
        key_pair = (
            df.loc[missing, ["element_0", "element_1"]]
            .set_index(["element_0", "element_1"])
            .index
        )
        pred_pair = pair_means.reindex(key_pair).values
        pred[missing] = pred_pair

    missing = pd.isna(pred)
    if missing.any():
        pred[missing] = df.loc[missing, "type"].map(type_means)

    return pred.fillna(global_mean).values


train_baseline = baseline_predict(
    train_merged, combo_means, pair_means, type_means, global_mean
)
train_residual = train_merged["scalar_coupling_constant"].values - train_baseline

test_baseline = baseline_predict(
    test_merged, combo_means, pair_means, type_means, global_mean
)




## === cell 5
train_merged["atomic_num_sum"] = (
    train_merged["atomic_num_0"] + train_merged["atomic_num_1"]
)
train_merged["atomic_num_diff"] = (
    train_merged["atomic_num_0"] - train_merged["atomic_num_1"]
).abs()
test_merged["atomic_num_sum"] = (
    test_merged["atomic_num_0"] + test_merged["atomic_num_1"]
)
test_merged["atomic_num_diff"] = (
    test_merged["atomic_num_0"] - test_merged["atomic_num_1"]
).abs()

feature_cols = [
    "distance",
    "dipole_mag",
    "potential_energy",
    "atomic_num_0",
    "atomic_num_1",
    "type_code",
    "atomic_num_sum",
    "atomic_num_diff",
]

X = train_merged[feature_cols].fillna(-1)
y = train_residual

gbr = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
)

gbr.fit(X, y)

test_features = test_merged[feature_cols].fillna(-1)
test_residual_pred = gbr.predict(test_features)

final_pred = test_baseline + test_residual_pred

final_pred = np.nan_to_num(final_pred, nan=global_mean)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'atomic_num_0'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2399069326.py in <cell line: 0>()
      1 train_merged["atomic_num_sum"] = (
----> 2     train_merged["atomic_num_0"] + train_merged["atomic_num_1"]
      3 )
      4 train_merged["atomic_num_diff"] = (
      5     train_merged["atomic_num_0"] - train_merged["atomic_num_1"]

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'atomic_num_0'

## === cell 6
test_df["scalar_coupling_constant"] = final_pred

submission = test_df[["id", "scalar_coupling_constant"]].copy()
submission_path = "my_blend_2.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission written to {submission_path}, shape: {submission.shape}")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1242341312.py in <cell line: 0>()
----> 1 test_df["scalar_coupling_constant"] = final_pred
      2 
      3 submission = test_df[["id", "scalar_coupling_constant"]].copy()
      4 submission_path = "my_blend_2.csv"
      5 submission.to_csv(submission_path, index=False)

NameError: name 'final_pred' is not defined
