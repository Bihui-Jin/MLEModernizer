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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

-2.4128442927016835

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd

INPUT_ROOT = "../input"
COMP_ROOT = os.path.join(INPUT_ROOT, "champs-scalar-coupling")
TARGET = "scalar_coupling_constant"

print("INPUT_ROOT exists:", os.path.exists(INPUT_ROOT))
print("COMP_ROOT exists:", os.path.exists(COMP_ROOT))
print(
    "Some input dirs:",
    sorted([p for p in glob.glob(os.path.join(INPUT_ROOT, "*")) if os.path.isdir(p)])[
        :20
    ],
)



## === cell 1
test_path = os.path.join(COMP_ROOT, "test.csv")
sample_path = os.path.join(COMP_ROOT, "sample_submission.csv")

test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

assert "id" in test.columns
assert list(sample.columns) == ["id", TARGET]

print("test shape:", test.shape)
print("sample shape:", sample.shape)
test.head()




## === cell 2
def _read_pred_file_as_series(path: str, test_ids: pd.Series) -> pd.Series:
    """
    Robustly read a prediction CSV and return a Series aligned to test_ids (index = id).
    Supports files with:
      - columns: ['id', 'scalar_coupling_constant']
      - only prediction column, with id stored as index
      - first unnamed column as id + one prediction column
    """
    df = pd.read_csv(path)

    if "id" in df.columns and TARGET in df.columns:
        s = df.set_index("id")[TARGET]

    else:
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "id"})
        if "id" in df.columns:
            if TARGET in df.columns:
                pred_col = TARGET
            else:
                cand_cols = [c for c in df.columns if c != "id"]
                pred_col = cand_cols[0] if len(cand_cols) else None
                if pred_col is None:
                    raise ValueError(f"No prediction column found in {path}")
            s = df.set_index("id")[pred_col]
        else:
            if df.shape[1] < 2:
                raise ValueError(
                    f"Cannot infer id/pred columns in {path}; columns={df.columns.tolist()}"
                )
            df2 = df.copy()
            df2.columns = ["id"] + list(df2.columns[1:])
            pred_col = TARGET if TARGET in df2.columns else df2.columns[1]
            s = df2.set_index("id")[pred_col]

    s = s.reindex(test_ids.values)
    return s


def get_median_from_files(files, test_ids: pd.Series) -> pd.Series:
    """
    Compute per-id median across a list of prediction files.
    Skips files that can't be read/aligned; returns NaNs if none usable.
    """
    usable = []
    for f in files:
        try:
            s = _read_pred_file_as_series(f, test_ids)
            coverage = s.notna().mean()
            if coverage >= 0.90:
                usable.append(s)
            else:
                pass
        except Exception:
            pass

    print(f"Requested {len(files)} files; usable {len(usable)}")
    if len(usable) == 0:
        return pd.Series(
            [float("nan")] * len(test_ids), index=test_ids.values, name="median_pred"
        )

    concat = pd.concat(usable, axis=1)
    med = concat.median(axis=1)
    med.name = "median_pred"
    return med


all_csvs = glob.glob(os.path.join(INPUT_ROOT, "**", "*.csv"), recursive=True)
exclude = {
    os.path.join(COMP_ROOT, "train.csv"),
    os.path.join(COMP_ROOT, "test.csv"),
    os.path.join(COMP_ROOT, "sample_submission.csv"),
    os.path.join(COMP_ROOT, "structures.csv"),
    os.path.join(COMP_ROOT, "scalar_coupling_contributions.csv"),
    os.path.join(COMP_ROOT, "dipole_moments.csv"),
    os.path.join(COMP_ROOT, "magnetic_shielding_tensors.csv"),
    os.path.join(COMP_ROOT, "mulliken_charges.csv"),
    os.path.join(COMP_ROOT, "potential_energy.csv"),
}
pred_csvs = [p for p in all_csvs if p not in exclude]

print("Total CSVs found:", len(all_csvs))
print("Candidate pred CSVs:", len(pred_csvs))
print("Example candidates:", pred_csvs[:20])



## === cell 3

test_ids = test["id"]

test["ens_median"] = get_median_from_files(pred_csvs, test_ids)

fallback = sample.set_index("id")[TARGET].reindex(test_ids.values)

if test["ens_median"].isna().all():
    print(
        "No usable external prediction files found; falling back to sample_submission baseline."
    )
    test["final_preds"] = fallback.values
else:
    test["final_preds"] = test["ens_median"].fillna(fallback).values

test[["id", "final_preds"]].head()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in nanmedian(values, axis, skipna, mask)
    788         try:
--> 789             values = values.astype("f8")
    790         except ValueError as err:

ValueError: could not convert string to float: 'dsgdb9nsd_071451'

The above exception was the direct cause of the following exception:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1033698893.py in <cell line: 0>()
      5 
      6 # Try to use all candidate prediction CSVs; median will be robust to outliers
----> 7 test["ens_median"] = get_median_from_files(pred_csvs, test_ids)
      8 
      9 # Fallback baseline: sample_submission (all zeros in the original competition)

/tmp/ipykernel_11/4056938968.py in get_median_from_files(files, test_ids)
     69 
     70     concat = pd.concat(usable, axis=1)
---> 71     med = concat.median(axis=1)
     72     med.name = "median_pred"
     73     return med

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
    790         except ValueError as err:
    791             # e.g. "could not convert string to float: 'a'"
--> 792             raise TypeError(str(err)) from err
    793     if not using_nan_sentinel and mask is not None:
    794         if not values.flags.writeable:

TypeError: could not convert string to float: 'dsgdb9nsd_071451'

## === cell 4
submission = pd.DataFrame(
    {"id": test["id"].astype(int), TARGET: test["final_preds"].astype(float)}
)

submission = submission.sort_values("id").reset_index(drop=True)

out_path = "ensemble_sub.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path, "shape:", submission.shape)
submission.head()



## --- ERROR in cell 4, traceback:
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

KeyError: 'final_preds'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/90533142.py in <cell line: 0>()
      1 # Produce a valid submission CSV
      2 submission = pd.DataFrame(
----> 3     {"id": test["id"].astype(int), TARGET: test["final_preds"].astype(float)}
      4 )
      5 

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

KeyError: 'final_preds'

## === cell 5
assert submission.columns.tolist() == ["id", TARGET]
assert submission["id"].isna().sum() == 0
assert submission[TARGET].isna().sum() == 0
assert submission.shape[0] == test.shape[0]
print(submission.describe(include="all"))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2403142258.py in <cell line: 0>()
      1 # Basic sanity checks for submission format
----> 2 assert submission.columns.tolist() == ["id", TARGET]
      3 assert submission["id"].isna().sum() == 0
      4 assert submission[TARGET].isna().sum() == 0
      5 assert submission.shape[0] == test.shape[0]

NameError: name 'submission' is not defined

## === cell 6
submission.head(20)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2834431645.py in <cell line: 0>()
----> 1 submission.head(20)

NameError: name 'submission' is not defined
