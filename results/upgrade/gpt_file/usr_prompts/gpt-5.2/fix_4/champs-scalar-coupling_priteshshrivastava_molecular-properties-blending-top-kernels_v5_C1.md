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

-1.5801604792430517

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.99777) has done: 'I remove the broken dependency on external “blending” input files that aren’t present in your environment and replace it with a minimal, self-contained baseline that trains and predicts from the provided CHAMPS data. To keep changes minimal while ensuring a valid end-to-end run, the new code build a small set of safe geometric features from `structures.csv` (pairwise distance and atom types) and train a separate lightweight model per coupling `type`. This preserves the original intent (predict `scalar_coupling_constant` from engineered structure features) and produces a correctly formatted `submission.csv`. The approach is deterministic and should yield a meaningful (though not SOTA) score instead of failing before submission creation.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

INPUT_DIR = "/kaggle/input/champs-scalar-coupling"
WORKING_DIR = "/kaggle/working"

print("Listing /kaggle/input (top):", os.listdir("/kaggle/input")[:10])
print("Using INPUT_DIR:", INPUT_DIR)
print("Files in INPUT_DIR (sample):", os.listdir(INPUT_DIR)[:10])



## === cell 1
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")
structures_path = os.path.join(INPUT_DIR, "structures.csv")
sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

train = pd.read_csv(
    train_path,
    dtype={
        "id": np.int32,
        "molecule_name": "string",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "string",
        "scalar_coupling_constant": np.float32,
    },
)
test = pd.read_csv(
    test_path,
    dtype={
        "id": np.int32,
        "molecule_name": "string",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "string",
    },
)
structures = pd.read_csv(
    structures_path,
    dtype={
        "molecule_name": "string",
        "atom_index": np.int16,
        "atom": "string",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
)
sample_sub = pd.read_csv(
    sample_sub_path, dtype={"id": np.int32, "scalar_coupling_constant": np.float32}
)

print(train.shape, test.shape, structures.shape, sample_sub.shape)
print(train.columns.tolist())
print(test.columns.tolist())
print(structures.columns.tolist())



## === cell 2

all_types = pd.Index(
    pd.concat([train["type"], test["type"]], axis=0).unique()
).sort_values()
all_atoms = pd.Index(structures["atom"].unique()).sort_values()

type_dtype = pd.CategoricalDtype(categories=list(all_types), ordered=False)
atom_dtype = pd.CategoricalDtype(categories=list(all_atoms), ordered=False)

train["type"] = train["type"].astype(type_dtype)
test["type"] = test["type"].astype(type_dtype)
structures["atom"] = structures["atom"].astype(atom_dtype)

s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]].copy()

s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]].copy()

s0 = s0.set_index(["molecule_name", "atom_index_0"], drop=True)
s1 = s1.set_index(["molecule_name", "atom_index_1"], drop=True)

atom_cats = list(atom_dtype.categories)
if "UNK" not in atom_cats:
    atom_cats.append("UNK")
new_atom_dtype = pd.CategoricalDtype(categories=atom_cats, ordered=False)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    base = base.set_index(["molecule_name", "atom_index_0", "atom_index_1"], drop=False)

    j0 = base.join(s0, on=["molecule_name", "atom_index_0"])
    j01 = j0.join(s1, on=["molecule_name", "atom_index_1"])

    for c in ["x0", "y0", "z0", "x1", "y1", "z1"]:
        if c not in j01.columns:
            j01[c] = np.float32(0.0)
        j01[c] = j01[c].astype(np.float32, copy=False)
    j01[["x0", "y0", "z0", "x1", "y1", "z1"]] = j01[
        ["x0", "y0", "z0", "x1", "y1", "z1"]
    ].fillna(np.float32(0.0))

    for ac in ["atom_0", "atom_1"]:
        if ac not in j01.columns:
            j01[ac] = "UNK"
        j01[ac] = j01[ac].astype("string")
        j01[ac] = j01[ac].fillna("UNK").astype(new_atom_dtype)

    x0 = j01["x0"].to_numpy(dtype=np.float32, copy=False)
    y0 = j01["y0"].to_numpy(dtype=np.float32, copy=False)
    z0 = j01["z0"].to_numpy(dtype=np.float32, copy=False)
    x1 = j01["x1"].to_numpy(dtype=np.float32, copy=False)
    y1 = j01["y1"].to_numpy(dtype=np.float32, copy=False)
    z1 = j01["z1"].to_numpy(dtype=np.float32, copy=False)

    dx = x0 - x1
    dy = y0 - y1
    dz = z0 - z1
    dist = np.sqrt(dx * dx + dy * dy + dz * dz, dtype=np.float32).astype(np.float32)

    j01["dx"] = dx
    j01["dy"] = dy
    j01["dz"] = dz
    j01["dist"] = dist

    j01["type"] = j01["type"].astype(type_dtype)
    j01 = j01.reset_index(drop=True)
    return j01


train_fe = add_features(train)
test_fe = add_features(test)

check_cols = [
    "atom_0",
    "atom_1",
    "x0",
    "y0",
    "z0",
    "x1",
    "y1",
    "z1",
    "dx",
    "dy",
    "dz",
    "dist",
    "type",
]
print(
    "Missing values after feature build - train:",
    int(train_fe[check_cols].isna().sum().sum()),
    "test:",
    int(test_fe[check_cols].isna().sum().sum()),
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3464996552.py in <cell line: 0>()
     93 
     94 
---> 95 train_fe = add_features(train)
     96 test_fe = add_features(test)
     97 

/tmp/ipykernel_11/3464996552.py in add_features(df)
     50 
     51     # Join on (molecule_name, atom_index_k)
---> 52     j0 = base.join(s0, on=["molecule_name", "atom_index_0"])
     53     j01 = j0.join(s1, on=["molecule_name", "atom_index_1"])
     54 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in join(self, other, on, how, lsuffix, rsuffix, sort, validate)
  10755                     validate=validate,
  10756                 )
> 10757             return merge(
  10758                 self,
  10759                 other,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    168         )
    169     else:
--> 170         op = _MergeOperation(
    171             left_df,
    172             right_df,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in __init__(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)
    792             left_drop,
    793             right_drop,
--> 794         ) = self._get_merge_keys()
    795 
    796         if left_drop:

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _get_merge_keys(self)
   1325                     #  the latter of which will raise
   1326                     k = cast(Hashable, k)
-> 1327                     left_keys.append(left._get_label_or_level_values(k))
   1328                     join_names.append(k)
   1329             if isinstance(self.right.index, MultiIndex):

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _get_label_or_level_values(self, key, axis)
   1904 
   1905         if self._is_label_reference(key, axis=axis):
-> 1906             self._check_label_or_level_ambiguity(key, axis=axis)
   1907             values = self.xs(key, axis=other_axes[0])._values
   1908         elif self._is_level_reference(key, axis=axis):

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _check_label_or_level_ambiguity(self, key, axis)
   1866                 f"{label_article} {label_type} label, which is ambiguous."
   1867             )
-> 1868             raise ValueError(msg)
   1869 
   1870     @final

ValueError: 'molecule_name' is both an index level and a column label, which is ambiguous.

## === cell 3
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

FEATURE_NUM = ["dist", "dx", "dy", "dz"]
FEATURE_CAT = ["type", "atom_0", "atom_1"]

all_data = pd.concat(
    [train_fe[FEATURE_NUM + FEATURE_CAT], test_fe[FEATURE_NUM + FEATURE_CAT]],
    axis=0,
    ignore_index=True,
)

X_num = all_data[FEATURE_NUM].to_numpy(dtype=np.float32, copy=False)
X_num = np.nan_to_num(X_num, nan=0.0, posinf=0.0, neginf=0.0)

X_cat = all_data[FEATURE_CAT].copy()
for c in FEATURE_CAT:
    X_cat[c] = X_cat[c].astype("string")

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
X_cat_sp = ohe.fit_transform(X_cat)

X_num_sp = sparse.csr_matrix(X_num)
X_all_sp = sparse.hstack([X_num_sp, X_cat_sp], format="csr")

n_train = len(train_fe)
X_train_all = X_all_sp[:n_train]
X_test_all = X_all_sp[n_train:]

y_train_all = train_fe["scalar_coupling_constant"].to_numpy(
    dtype=np.float32, copy=False
)

pred_test = np.zeros(len(test_fe), dtype=np.float32)

types = list(all_types)
print("Types:", types)

rf_params = dict(
    n_estimators=120, random_state=42, n_jobs=-1, max_depth=None, min_samples_leaf=1
)

train_type_codes = train_fe["type"].cat.codes.to_numpy()
test_type_codes = test_fe["type"].cat.codes.to_numpy()
type_categories = train_fe["type"].cat.categories.astype(str).tolist()

for code in np.unique(train_type_codes):
    tr_idx = np.flatnonzero(train_type_codes == code)
    te_idx = np.flatnonzero(test_type_codes == code)
    if te_idx.size == 0:
        continue

    Xtr = X_train_all[tr_idx]
    ytr = y_train_all[tr_idx]
    Xte = X_test_all[te_idx]

    model = RandomForestRegressor(**rf_params)
    model.fit(Xtr, ytr)
    pred_test[te_idx] = model.predict(Xte).astype(np.float32, copy=False)

    t = type_categories[code] if 0 <= code < len(type_categories) else str(code)
    print(f"type={t:4s} train_rows={tr_idx.size:7d} test_rows={te_idx.size:7d}")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3543117092.py in <cell line: 0>()
      9 # This is equivalent to get_dummies with dummy_na=False: unseen categories become all-zeros (like missing dummy columns).
     10 all_data = pd.concat(
---> 11     [train_fe[FEATURE_NUM + FEATURE_CAT], test_fe[FEATURE_NUM + FEATURE_CAT]],
     12     axis=0,
     13     ignore_index=True,

NameError: name 'train_fe' is not defined

## === cell 4
submission = pd.DataFrame(
    {
        "id": test["id"].to_numpy(copy=False),
        "scalar_coupling_constant": pred_test.astype(np.float32, copy=False),
    }
)

submission = sample_sub[["id"]].merge(submission, on="id", how="left")
assert submission.shape[0] == sample_sub.shape[0]
assert submission["scalar_coupling_constant"].isna().sum() == 0

out_path = os.path.join(WORKING_DIR, "submission.csv")
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2132646358.py in <cell line: 0>()
      2     {
      3         "id": test["id"].to_numpy(copy=False),
----> 4         "scalar_coupling_constant": pred_test.astype(np.float32, copy=False),
      5     }
      6 )

NameError: name 'pred_test' is not defined
