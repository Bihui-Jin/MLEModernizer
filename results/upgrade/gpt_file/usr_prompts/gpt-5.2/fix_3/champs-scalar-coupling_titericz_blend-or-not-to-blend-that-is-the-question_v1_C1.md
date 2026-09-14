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

-1.2781586533261562

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.23578) has done: 'The current notebook fails because it tries to read two external submissions from Kaggle Datasets that are not available in your environment, so `sub1`/`sub2` are never created and downstream cells crash. To make it run end-to-end and always produce a valid `.csv` submission, I replace those missing inputs with a simple, deterministic baseline model trained only from the provided `train.csv`/`test.csv` (group mean by coupling `type`). This preserves the “blending” idea by creating two lightweight baseline predictors and then weighted-averaging them, but without relying on unavailable files. The result writes `weighted-avg-blend-lgb-keras-1.csv` with the required `id,scalar_coupling_constant` columns.'

# 9. Code solution

## === cell 0
import os

DATA_ROOT = "/kaggle/data/champs-scalar-coupling"
ALT_DATA_ROOT = "/kaggle/input/champs-scalar-coupling"

if not os.path.exists(DATA_ROOT) and os.path.exists(ALT_DATA_ROOT):
    DATA_ROOT = ALT_DATA_ROOT

print("DATA_ROOT =", DATA_ROOT)
print("Files in DATA_ROOT (sample):", sorted(os.listdir(DATA_ROOT))[:10])



## === cell 1
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("/kaggle/data"))



## === cell 2
train_path = os.path.join(DATA_ROOT, "train.csv")
test_path = os.path.join(DATA_ROOT, "test.csv")
structures_path = os.path.join(DATA_ROOT, "structures.csv")

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
)
test = pd.read_csv(
    test_path, usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]
)

structures = pd.read_csv(
    structures_path, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
)

mol_agg = (
    structures.groupby("molecule_name")
    .agg(
        mol_x_mean=("x", "mean"),
        mol_y_mean=("y", "mean"),
        mol_z_mean=("z", "mean"),
        mol_n_atoms=("atom_index", "count"),
    )
    .reset_index()
)

coords = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)
coords2 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(
        coords[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    df = df.merge(
        coords2[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )
    df = df.merge(mol_agg, on="molecule_name", how="left")

    dx = (df["x0"] - df["x1"]).astype(np.float64)
    dy = (df["y0"] - df["y1"]).astype(np.float64)
    dz = (df["z0"] - df["z1"]).astype(np.float64)
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float64)

    cdx0 = (df["x0"] - df["mol_x_mean"]).astype(np.float64)
    cdy0 = (df["y0"] - df["mol_y_mean"]).astype(np.float64)
    cdz0 = (df["z0"] - df["mol_z_mean"]).astype(np.float64)
    cdx1 = (df["x1"] - df["mol_x_mean"]).astype(np.float64)
    cdy1 = (df["y1"] - df["mol_y_mean"]).astype(np.float64)
    cdz1 = (df["z1"] - df["mol_z_mean"]).astype(np.float64)
    df["r0"] = np.sqrt(cdx0 * cdx0 + cdy0 * cdy0 + cdz0 * cdz0).astype(np.float64)
    df["r1"] = np.sqrt(cdx1 * cdx1 + cdy1 * cdy1 + cdz1 * cdz1).astype(np.float64)

    df["pair"] = df["atom_0"].astype(str) + "-" + df["atom_1"].astype(str)
    return df


train_f = add_features(train.copy())
test_f = add_features(test.copy())

bin_width = 0.1
train_f["dist_bin"] = np.floor(train_f["dist"] / bin_width).astype(np.int32)
test_f["dist_bin"] = np.floor(test_f["dist"] / bin_width).astype(np.int32)

global_mean = float(train_f["scalar_coupling_constant"].mean())

type_stats = (
    train_f.groupby("type")["scalar_coupling_constant"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "type_mean", "count": "type_count"})
)

train_f["mol_n_atoms_bin"] = (train_f["mol_n_atoms"] // 5).astype(np.int32)
test_f["mol_n_atoms_bin"] = (test_f["mol_n_atoms"] // 5).astype(np.int32)

grp_cols = ["type", "dist_bin", "pair", "mol_n_atoms_bin"]
grp_stats = (
    train_f.groupby(grp_cols)["scalar_coupling_constant"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "grp_mean", "count": "grp_count"})
)

test_stats = test_f.merge(type_stats, how="left", left_on="type", right_index=True)
test_stats = test_stats.merge(grp_stats, how="left", on=grp_cols)

pred1 = test_stats["type_mean"].fillna(global_mean).astype(np.float64)

k_grp = 25.0
grp_count = test_stats["grp_count"].fillna(0.0).astype(np.float64)
grp_mean = (
    test_stats["grp_mean"]
    .fillna(test_stats["type_mean"])
    .fillna(global_mean)
    .astype(np.float64)
)
type_mean = test_stats["type_mean"].fillna(global_mean).astype(np.float64)

w_grp = grp_count / (grp_count + k_grp)
pred2 = (w_grp * grp_mean + (1.0 - w_grp) * type_mean).astype(np.float64)

sub1 = pd.DataFrame(
    {"id": test_stats["id"].values, "scalar_coupling_constant": pred1.values}
)
sub2 = pd.DataFrame(
    {"id": test_stats["id"].values, "scalar_coupling_constant": pred2.values}
)

print("pred1 describe:\n", sub1["scalar_coupling_constant"].describe())
print("pred2 describe:\n", sub2["scalar_coupling_constant"].describe())



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
IntCastingNaNError                        Traceback (most recent call last)
/tmp/ipykernel_11/2867879633.py in <cell line: 0>()
     99 bin_width = 0.1
    100 train_f["dist_bin"] = np.floor(train_f["dist"] / bin_width).astype(np.int32)
--> 101 test_f["dist_bin"] = np.floor(test_f["dist"] / bin_width).astype(np.int32)
    102 
    103 global_mean = float(train_f["scalar_coupling_constant"].mean())

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in astype(self, dtype, copy, errors)
   6641         else:
   6642             # else, only a single dtype is given
-> 6643             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
   6644             res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   6645             return res.__finalize__(self, method="astype")

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in astype(self, dtype, copy, errors)
    428             copy = False
    429 
--> 430         return self.apply(
    431             "astype",
    432             dtype=dtype,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in astype(self, dtype, copy, errors, using_cow, squeeze)
    756             values = values[0, :]  # type: ignore[call-overload]
    757 
--> 758         new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    759 
    760         new_values = maybe_coerce_values(new_values)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array_safe(values, dtype, copy, errors)
    235 
    236     try:
--> 237         new_values = astype_array(values, dtype, copy=copy)
    238     except (ValueError, TypeError):
    239         # e.g. _astype_nansafe can fail on object-dtype of strings

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array(values, dtype, copy)
    180 
    181     else:
--> 182         values = _astype_nansafe(values, dtype, copy=copy)
    183 
    184     # in pandas we don't store numpy str dtypes, so convert to object

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_nansafe(arr, dtype, copy, skipna)
     99 
    100     elif np.issubdtype(arr.dtype, np.floating) and dtype.kind in "iu":
--> 101         return _astype_float_to_int_nansafe(arr, dtype, copy)
    102 
    103     elif arr.dtype == object:

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_float_to_int_nansafe(values, dtype, copy)
    143     """
    144     if not np.isfinite(values).all():
--> 145         raise IntCastingNaNError(
    146             "Cannot convert non-finite values (NA or inf) to integer"
    147         )

IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer

## === cell 3
(sub1["scalar_coupling_constant"] - sub2["scalar_coupling_constant"]).abs().mean()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4050153359.py in <cell line: 0>()
----> 1 (sub1["scalar_coupling_constant"] - sub2["scalar_coupling_constant"]).abs().mean()
      2 

NameError: name 'sub1' is not defined

## === cell 4
sub1["scalar_coupling_constant"] = (
    0.35 * sub1["scalar_coupling_constant"] + 0.65 * sub2["scalar_coupling_constant"]
)

sub1 = sub1[["id", "scalar_coupling_constant"]].sort_values("id")
sub1.to_csv("weighted-avg-blend-lgb-keras-1.csv", index=False)

sub1["scalar_coupling_constant"].describe()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3192709797.py in <cell line: 0>()
      2 # while still blending two predictors as in the original core logic.
      3 sub1["scalar_coupling_constant"] = (
----> 4     0.35 * sub1["scalar_coupling_constant"] + 0.65 * sub2["scalar_coupling_constant"]
      5 )
      6 

NameError: name 'sub1' is not defined
