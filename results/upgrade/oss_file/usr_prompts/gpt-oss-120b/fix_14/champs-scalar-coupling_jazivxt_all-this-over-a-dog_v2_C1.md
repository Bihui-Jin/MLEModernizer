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

0.90666

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.32733) has done: 'I added a simple median imputation step to remove NaNs that caused the ExtraTreesRegressor to fail, and kept the original modeling pipeline unchanged. The imputer is fitted on the training features and applied to both train and test sets before any split or model fitting, ensuring a valid prediction and CSV submission.'
- What this solution (achieved 1.48175) has done: 'The main slowdown comes from repeatedly merging the large auxiliary tables into both the train and test frames. By concatenating train and test, performing each merge once on the combined frame, and then splitting back, we cut the merge work roughly in half while keeping all columns identical. The rest of the pipeline—including feature engineering, filling missing values, and model training—remains unchanged, preserving exact model logic and results.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, ensemble, metrics

dtype_train = {
    "id": np.int32,
    "molecule_name": "object",
    "atom_index_0": np.int16,
    "atom_index_1": np.int16,
    "type": "object",
    "scalar_coupling_constant": np.float32,
}
dtype_test = {
    "id": np.int32,
    "molecule_name": "object",
    "atom_index_0": np.int16,
    "atom_index_1": np.int16,
    "type": "object",
}
train = pd.read_csv("../input/train.csv", dtype=dtype_train, usecols=dtype_train.keys())
test = pd.read_csv("../input/test.csv", dtype=dtype_test, usecols=dtype_test.keys())
sub = pd.read_csv("../input/sample_submission.csv")
print(train.shape, test.shape, sub.shape)

mol_codes, mol_unique = pd.factorize(
    pd.concat([train["molecule_name"], test["molecule_name"]], ignore_index=True)
)
train["mol_id"] = mol_codes[: len(train)].astype(np.int32)
test["mol_id"] = mol_codes[len(train) :].astype(np.int32)

train["atom"] = train["type"].str[3].astype("category")
test["atom"] = test["type"].str[3].astype("category")

lbl = preprocessing.LabelEncoder()
for i in range(4):
    combined = pd.concat([train["type"].str[i], test["type"].str[i]], ignore_index=True)
    lbl.fit(combined)
    train[f"type{i}"] = lbl.transform(train["type"].str[i])
    test[f"type{i}"] = lbl.transform(test["type"].str[i])

train["atom_code"] = train["atom"].cat.codes.astype(np.int16)
test["atom_code"] = test["atom"].cat.codes.astype(np.int16)


def map_mol_id(df, col="molecule_name"):
    df["mol_id"] = pd.Categorical(df[col], categories=mol_unique).codes.astype(np.int32)
    return df


train["is_train"] = True
test["is_train"] = False
combined = pd.concat([train, test], ignore_index=True)

structures = pd.read_csv(
    "../input/structures.csv",
    dtype={
        "molecule_name": "object",
        "atom_index": np.int16,
        "atom": "object",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
)
structures = map_mol_id(structures)
structures["atom_code"] = (
    structures["atom"].astype("category").cat.codes.astype(np.int16)
)

struct1 = structures.rename(
    columns={"atom_index": "atom_index_1", "x": "x1", "y": "y1", "z": "z1"}
)[["mol_id", "atom_index_1", "atom_code", "x1", "y1", "z1"]]
combined = combined.merge(
    struct1, how="left", on=["mol_id", "atom_index_1", "atom_code"]
)

struct0 = structures.rename(
    columns={"atom_index": "atom_index_0", "x": "x0", "y": "y0", "z": "z0"}
)[["mol_id", "atom_index_0", "atom_code", "x0", "y0", "z0"]]
combined = combined.merge(
    struct0, how="left", on=["mol_id", "atom_index_0", "atom_code"]
)

combined["dist"] = np.sqrt(
    (combined["x1"] - combined["x0"]) ** 2
    + (combined["y1"] - combined["y0"]) ** 2
    + (combined["z1"] - combined["z0"]) ** 2
)

del structures, struct0, struct1

pe = pd.read_csv(
    "../input/potential_energy.csv",
    dtype={"molecule_name": "object", "potential_energy": np.float32},
    usecols=["molecule_name", "potential_energy"],
)
pe = map_mol_id(pe)
pe = pe[["mol_id", "potential_energy"]]
combined = combined.merge(pe, how="left", on="mol_id")
del pe

mc = pd.read_csv(
    "../input/mulliken_charges.csv",
    dtype={
        "molecule_name": "object",
        "atom_index": np.int16,
        "mulliken_charge": np.float32,
    },
    usecols=["molecule_name", "atom_index", "mulliken_charge"],
)
mc = map_mol_id(mc, col="molecule_name")
mc.rename(columns={"atom_index": "atom_index_0"}, inplace=True)
mc = mc[["mol_id", "atom_index_0", "mulliken_charge"]]
combined = combined.merge(mc, how="left", on=["mol_id", "atom_index_0"])
del mc

dm = pd.read_csv(
    "../input/dipole_moments.csv",
    dtype={
        "molecule_name": "object",
        "X": np.float32,
        "Y": np.float32,
        "Z": np.float32,
    },
    usecols=["molecule_name", "X", "Y", "Z"],
)
dm = map_mol_id(dm)
dm = dm[["mol_id", "X", "Y", "Z"]]
combined = combined.merge(dm, how="left", on="mol_id")
del dm

mst = pd.read_csv(
    "../input/magnetic_shielding_tensors.csv",
    dtype={
        "molecule_name": "object",
        "atom_index": np.int16,
        "XX": np.float32,
        "YX": np.float32,
        "ZX": np.float32,
        "XY": np.float32,
        "YY": np.float32,
        "ZY": np.float32,
        "XZ": np.float32,
        "YZ": np.float32,
        "ZZ": np.float32,
    },
    usecols=[
        "molecule_name",
        "atom_index",
        "XX",
        "YX",
        "ZX",
        "XY",
        "YY",
        "ZY",
        "XZ",
        "YZ",
        "ZZ",
    ],
)
mst = map_mol_id(mst)
mst.rename(columns={"atom_index": "atom_index_0"}, inplace=True)
mst = mst[
    ["mol_id", "atom_index_0", "XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
]
combined = combined.merge(mst, how="left", on=["mol_id", "atom_index_0"])
del mst

scc = pd.read_csv(
    "../input/scalar_coupling_contributions.csv",
    dtype={
        "molecule_name": "object",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "object",
        "fc": np.float32,
        "sd": np.float32,
        "pso": np.float32,
        "dso": np.float32,
    },
    usecols=[
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "fc",
        "sd",
        "pso",
        "dso",
    ],
)
scc = map_mol_id(scc)
combined = combined.merge(
    scc,
    how="left",
    on=["mol_id", "atom_index_0", "atom_index_1", "type"],
)
del scc

combined.drop(columns=["mol_id", "atom_code", "type"], inplace=True)

train = combined[combined["is_train"]].copy()
test = combined[~combined["is_train"]].copy()
train.drop(columns=["is_train"], inplace=True)
test.drop(columns=["is_train"], inplace=True)

print(train.shape, test.shape, sub.shape)



## === cell 1
train.head()



## === cell 2
exclude_cols = ["id", "molecule_name", "scalar_coupling_constant", "type", "atom"]
feature_cols = [c for c in train.columns if c not in exclude_cols]

medians = train[feature_cols].median()
train[feature_cols] = train[feature_cols].fillna(medians)
test[feature_cols] = test[feature_cols].fillna(medians)

X_full = train[feature_cols].astype(np.float32).values
y_full = train["scalar_coupling_constant"].values

sample_size = 200_000
rng = np.random.RandomState(99)
if sample_size < len(train):
    sample_idx = rng.choice(len(train), size=sample_size, replace=False)
else:
    sample_idx = np.arange(len(train))
X_sample = X_full[sample_idx]
y_sample = y_full[sample_idx]

X_samp, X_val, y_samp, y_val = model_selection.train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=99
)

reg = ensemble.ExtraTreesRegressor(
    n_jobs=-1,
    random_state=4,
    n_estimators=300,  # increased a bit for better performance
)

reg.fit(X_samp, y_samp)
val_pred = reg.predict(X_val)
score = np.log(metrics.mean_absolute_error(y_val, val_pred))
print("Log MAE on quick validation:", score)

reg.fit(X_full, y_full)

X_test = test[feature_cols].astype(np.float32).values
test["scalar_coupling_constant"] = reg.predict(X_test)

test[["id", "scalar_coupling_constant"]].to_csv(
    "submission.csv", float_format="%.9f", index=False
)

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2946579128.py in <cell line: 0>()
      2 feature_cols = [c for c in train.columns if c not in exclude_cols]
      3 
----> 4 medians = train[feature_cols].median()
      5 train[feature_cols] = train[feature_cols].fillna(medians)
      6 test[feature_cols] = test[feature_cols].fillna(medians)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in median(self, axis, skipna, numeric_only, **kwargs)
  11704         **kwargs,
  11705     ):
> 11706         result = super().median(axis, skipna, numeric_only, **kwargs)
  11707         if isinstance(result, Series):
  11708             result = result.__finalize__(self, method="median")

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in median(self, axis, skipna, numeric_only, **kwargs)
  12429         **kwargs,
  12430     ) -> Series | float:
> 12431         return self._stat_function(
  12432             "median", nanops.nanmedian, axis, skipna, numeric_only, **kwargs
  12433         )

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _stat_function(self, name, func, axis, skipna, numeric_only, **kwargs)
  12375         validate_bool_kwarg(skipna, "skipna", none_allowed=False)
  12376 
> 12377         return self._reduce(
  12378             func, name=name, axis=axis, skipna=skipna, numeric_only=numeric_only
  12379         )

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _reduce(self, op, name, axis, skipna, numeric_only, filter_type, **kwds)
  11560         # After possibly _get_data and transposing, we are now in the
  11561         #  simple case where we can use BlockManager.reduce
> 11562         res = df._mgr.reduce(blk_func)
  11563         out = df._constructor_from_mgr(res, axes=res.axes).iloc[0]
  11564         if out_dtype is not None and out.dtype != "boolean":

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in reduce(self, func)
   1498         res_blocks: list[Block] = []
   1499         for blk in self.blocks:
-> 1500             nbs = blk.reduce(func)
   1501             res_blocks.extend(nbs)
   1502 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in reduce(self, func)
    402         assert self.ndim == 2
    403 
--> 404         result = func(self.values)
    405 
    406         if self.values.ndim == 1:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in blk_func(values, axis)
  11479                     return np.array([result])
  11480             else:
> 11481                 return op(values, axis=axis, skipna=skipna, **kwds)
  11482 
  11483         def _get_data() -> DataFrame:

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in f(values, axis, skipna, **kwds)
    145                     result = alt(values, axis=axis, skipna=skipna, **kwds)
    146             else:
--> 147                 result = alt(values, axis=axis, skipna=skipna, **kwds)
    148 
    149             return result

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in nanmedian(values, axis, skipna, mask)
    785             inferred = lib.infer_dtype(values)
    786             if inferred in ["string", "mixed"]:
--> 787                 raise TypeError(f"Cannot convert {values} to numeric")
    788         try:
    789             values = values.astype("f8")

TypeError: Cannot convert [['dsgdb9nsd_109986' 'dsgdb9nsd_109986' 'dsgdb9nsd_109986' ...
  'dsgdb9nsd_107330' 'dsgdb9nsd_107330' 'dsgdb9nsd_107330']
 ['dsgdb9nsd_109986' 'dsgdb9nsd_109986' 'dsgdb9nsd_109986' ...
  'dsgdb9nsd_107330' 'dsgdb9nsd_107330' 'dsgdb9nsd_107330']] to numeric
