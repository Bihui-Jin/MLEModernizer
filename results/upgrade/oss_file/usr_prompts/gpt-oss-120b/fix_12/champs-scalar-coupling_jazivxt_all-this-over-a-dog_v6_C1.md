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

# 5. Target score

0.5536

# 6. Current score

2.1529

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 2.86732) has done: 'The fix adds simple imputation to replace NaNs (which cause ExtraTreesRegressor to error) with a constant value, and clarifies imports. No core modeling logic is changed, ensuring the script runs end‑to‑end and creates a valid `submission.csv`.'
- What this solution (achieved 2.1529) has done: 'I speed up the most time‑consuming steps: compute the inter‑atomic distance with a memory‑efficient formula and halve the number of trees in the ExtraTrees model (still the same algorithm). These changes keep the same feature set and prediction logic, only reducing constant‑factor work, so the results remain virtually identical while fitting well under the 600‑second limit.'

# 9. Code solution

## === cell 0
import numpy as np, pandas as pd, gc
from sklearn import preprocessing, ensemble, model_selection, metrics, impute

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

train = pd.read_csv(
    "../input/train.csv",
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
    "../input/test.csv",
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
sub = pd.read_csv("../input/sample_submission.csv")
print(train.shape, test.shape, sub.shape)

train = train.dropna(subset=["scalar_coupling_constant"]).reset_index(drop=True)

train["atom1"] = train["type"].map(lambda x: str(x)[2])
train["atom2"] = train["type"].map(lambda x: str(x)[3])
test["atom1"] = test["type"].map(lambda x: str(x)[2])
test["atom2"] = test["type"].map(lambda x: str(x)[3])

lbl = preprocessing.LabelEncoder()
for i in range(4):
    train[f"type{i}"] = lbl.fit_transform(train["type"].map(lambda x: str(x)[i]))
    test[f"type{i}"] = lbl.transform(test["type"].map(lambda x: str(x)[i]))

struct = pd.read_csv(
    "../input/structures.csv",
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
)

struct0 = struct.rename(
    columns={
        "atom_index": "atom_index_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
        "atom": "atom1",
    }
)[["molecule_name", "atom_index_0", "atom1", "x0", "y0", "z0"]]

struct1 = struct.rename(
    columns={
        "atom_index": "atom_index_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
        "atom": "atom2",
    }
)[["molecule_name", "atom_index_1", "atom2", "x1", "y1", "z1"]]

train = pd.merge(
    train,
    struct0,
    how="left",
    on=["molecule_name", "atom_index_0", "atom1"],
    sort=False,
)
train = pd.merge(
    train,
    struct1,
    how="left",
    on=["molecule_name", "atom_index_1", "atom2"],
    sort=False,
)

test = pd.merge(
    test,
    struct0,
    how="left",
    on=["molecule_name", "atom_index_0", "atom1"],
    sort=False,
)
test = pd.merge(
    test,
    struct1,
    how="left",
    on=["molecule_name", "atom_index_1", "atom2"],
    sort=False,
)

train = train.dropna(subset=["scalar_coupling_constant"]).reset_index(drop=True)
print(train.shape, test.shape, sub.shape)




## === cell 1
train_coords0 = train[["x0", "y0", "z0"]].values.astype(np.float32)
train_coords1 = train[["x1", "y1", "z1"]].values.astype(np.float32)
test_coords0 = test[["x0", "y0", "z0"]].values.astype(np.float32)
test_coords1 = test[["x1", "y1", "z1"]].values.astype(np.float32)

train["dist"] = np.sqrt(((train_coords0 - train_coords1) ** 2).sum(axis=1))
test["dist"] = np.sqrt(((test_coords0 - test_coords1) ** 2).sum(axis=1))

type_means = train.groupby("type")["dist"].mean()
train["dist_to_type_mean"] = train["dist"] / train["type"].map(type_means)
test["dist_to_type_mean"] = test["dist"] / test["type"].map(type_means)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2931420413.py in <cell line: 0>()
      9 
     10 type_means = train.groupby("type")["dist"].mean()
---> 11 train["dist_to_type_mean"] = train["dist"] / train["type"].map(type_means)
     12 test["dist_to_type_mean"] = test["dist"] / test["type"].map(type_means)
     13 

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/common.py in new_method(self, other)
     74         other = item_from_zerodim(other)
     75 
---> 76         return method(self, other)
     77 
     78     return new_method

/usr/local/lib/python3.11/dist-packages/pandas/core/arraylike.py in __truediv__(self, other)
    208     @unpack_zerodim_and_defer("__truediv__")
    209     def __truediv__(self, other):
--> 210         return self._arith_method(other, operator.truediv)
    211 
    212     @unpack_zerodim_and_defer("__rtruediv__")

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _arith_method(self, other, op)
   6133     def _arith_method(self, other, op):
   6134         self, other = self._align_for_op(other)
-> 6135         return base.IndexOpsMixin._arith_method(self, other, op)
   6136 
   6137     def _align_for_op(self, right, align_asobject: bool = False):

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _arith_method(self, other, op)
   1380 
   1381         with np.errstate(all="ignore"):
-> 1382             result = ops.arithmetic_op(lvalues, rvalues, op)
   1383 
   1384         return self._construct_result(result, name=res_name)

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in arithmetic_op(left, right, op)
    271         # Timedelta/Timestamp and other custom scalars are included in the check
    272         # because numexpr will fail on it, see GH#31457
--> 273         res_values = op(left, right)
    274     else:
    275         # TODO we should handle EAs consistently and move this check before the if/else

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/categorical.py in __array_ufunc__(self, ufunc, method, *inputs, **kwargs)
   1692         # for all other cases, raise for now (similarly as what happens in
   1693         # Series.__array_prepare__)
-> 1694         raise TypeError(
   1695             f"Object with dtype {self.dtype} cannot perform "
   1696             f"the numpy op {ufunc.__name__}"

TypeError: Object with dtype category cannot perform the numpy op divide

## === cell 2
exclude_cols = {
    "id",
    "molecule_name",
    "scalar_coupling_constant",
    "type",
    "atom1",
    "atom2",
    "atom_index_0",
    "atom_index_1",
}
col = [c for c in train.columns if c not in exclude_cols]

train[col] = train[col].astype(np.float32)
test[col] = test[col].astype(np.float32)

train[col] = train[col].fillna(-1)
test[col] = test[col].fillna(-1)

X = train[col].values
X_test = test[col].values

y_log = np.log1p(train["scalar_coupling_constant"].astype(float))
y_log = y_log.replace([np.inf, -np.inf], np.nan)
if y_log.isnull().any():
    median_val = np.median(y_log[~np.isnan(y_log)])
    y_log = y_log.fillna(median_val)

y = y_log.values.astype(np.float32)

X_train, X_val, y_train, y_val = model_selection.train_test_split(
    X, y, test_size=0.2, random_state=99
)

del train, test, struct, struct0, struct1, X, y
gc.collect()

reg = ensemble.ExtraTreesRegressor(
    n_estimators=200,  # halved from 400 for speed, algorithm unchanged
    max_features="sqrt",
    n_jobs=-1,
    random_state=4,
)

reg.fit(X_train, y_train)

val_pred = np.expm1(reg.predict(X_val))
val_mae = metrics.mean_absolute_error(np.expm1(y_val), val_pred)
print("Validation log‑MAE:", np.log(val_mae))

test_pred = np.expm1(reg.predict(X_test))
submission = pd.DataFrame({"id": sub["id"], "scalar_coupling_constant": test_pred})
submission.to_csv("submission.csv", index=False)
