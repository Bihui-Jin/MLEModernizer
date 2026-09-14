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

1.19404

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 2.9981) has done: 'I (1) fix the broken hydrogen assertion by checking actual values rather than comparing category lists (which can be empty or differently shaped), (2) eliminate NaNs in test features by computing the `dist_to_type_mean` normalization using **train** type means and falling back safely for any unseen types, and (3) ensure train/test one-hot type columns are aligned (missing type columns get filled with 0). These changes are minimal, keep the same model and features, unblock inference, and should also improve score stability by preventing NaN-driven failures. Finally, the script write a valid Kaggle submission CSV with the required filename suffix and columns.'
- What this solution (achieved 3.82522) has done: 'I fix the NaN crash during per-type prediction by ensuring we sanitize (NaN/inf) not only the full test feature matrix but also every per-type slice passed into `HuberRegressor.predict`. I also add a small safeguard earlier to prevent NaNs created by missing structure merges (missing coordinates) from propagating into `dist` and `dist_to_type_mean`. These changes keep the same features, same models, and same training approach, but make inference robust so the notebook finishes and writes a valid `submission.csv`. Finally, I ensure `resultSet` is always created so the submission cell cannot fail with `NameError`.'
- What this solution (achieved 3.30922) has done: 'We keep your exact modeling approach (HuberRegressor global + per-type) and the same base features, but fix two high-impact issues that are currently depressing score: (1) you hardcoded `feature_cols` to only 7 coupling types, so rows of other types effectively get almost-zero “type” info (all type_* = 0) and become much harder to learn; we instead include one-hot columns for all train types, aligned into test. (2) We add the two categorical atom identities (`atom_0`, `atom_1`) as one-hot features (still linear, same training loop), which is a minimal feature extension that usually gives a large jump on CHAMPS with this baseline. These changes should move your logMAE substantially down toward the 1.19 target while preserving core logic and producing the same submission format.'
- What this solution (achieved 2.02456) has done: 'Your score is much worse than the target (lower is better), so we should improve accuracy with minimal, safe feature additions while keeping the same HuberRegressor setup and per-type training/prediction loop. The biggest low-risk gain here is adding molecule-level and atom-level physics features you already have available: Mulliken charge, magnetic shielding tensor summary stats, dipole moment components, and potential energy, merged onto the same rows you already build (no change to model class or training approach). We also add a couple of strictly-derived geometric features (dx, dy, dz) from the existing coordinates, which are cheap and often help linear models. Finally, we keep the same type/atom one-hots and add robust NaN/inf sanitization after merges so these new joins can’t break inference.'
- What this solution (achieved 4.20761) has done: 'Your current score (2.02456) is much worse than the target (1.19404, lower-is-better), so we should make small, safe changes that improve accuracy without changing the HuberRegressor-per-type core logic. The biggest low-risk gain here is to standardize/scale the numeric features (geometry + external physics features) using training statistics, which usually helps robust linear models like Huber converge to better coefficients when feature magnitudes differ a lot. We keep all one-hot columns unchanged, compute scaling parameters on train only, apply to both train/test, and keep the same per-type training/prediction loop and submission format. This is minimal and should move the score down toward the target.'
- What this solution (achieved 11.60383) has done: 'Your current score (4.20761, lower is better) is far from the target (1.19404), so we need a modest accuracy gain without changing your core approach (HuberRegressor global + per-type). The biggest issue is that standardization is applied but the **model intercept is not adjusted**, meaning predictions are shifted/miscalibrated (a common large error source with linear models); we fix this by setting `fit_intercept=False` after standardization so the model doesn’t relearn an unnecessary offset. We also add one minimal, strictly geometric feature (`inv_dist = 1/(dist+1e-3)`) which is cheap, stable, and often improves linear fits for coupling constants without changing the model class or training loop. Finally, we keep all your NaN/inf safeguards and submission writing intact.'
- What this solution (achieved 11.16568) has done: 'Your score got much worse than the target (lower-is-better), and the biggest likely cause is a calibration/centering mistake: you standardized features but forced `fit_intercept=False`, which generally biases linear predictions because standardized features are centered but the target is not. I make the smallest corrective change by letting Huber learn an intercept again (`fit_intercept=True`) while keeping the exact same features, per-type training loop, and prediction logic. I also add a very small, safe stabilization to the standardization step (reuse train `num_mean/num_std` as-is, but ensure we don’t accidentally overwrite dtypes) without changing semantics. This should move the logMAE back down toward your target without changing the overall approach.'
- What this solution (achieved 11.56831) has done: 'Your current score (11.16568, lower-is-better) is far worse than the target (1.19404), so we should make a small correction that fixes a likely modeling mistake without changing the overall approach (HuberRegressor global + per-type). The biggest issue is that you standardize the numeric feature columns, but you do **not** standardize the target; with `fit_intercept=True`, each per-type model must relearn an intercept from heavily one-hot + standardized features, which can become unstable and hurt generalization. I keep the exact same features and per-type training loop, but standardize `y` **within each type** during training and then unscale predictions back—this is a minimal calibration fix that typically reduces MAE substantially on CHAMPS for linear models. I also ensure the standardization parameters are computed on train only (already true) and keep all NaN/inf sanitization and the submission format unchanged.'
- What this solution (achieved 11.16568) has done: 'We need to move your score down toward 1.19404 (lower is better), and your current 11.56831 indicates something is badly miscalibrated rather than just “weak features.” The smallest high-impact fix that preserves your exact model/training loop is to stop standardizing the target (`y`) entirely—Huber already learns an intercept, and your per-type `y` standardization can easily inflate errors if the model underfits variance and then gets unscaled. I keep all features, merges, and per-type training, but train/predict directly in the original target scale (no `y` scaling), while retaining your numeric feature standardization and NaN/inf sanitization. This should move performance back toward the ~2–4 range you previously achieved with the same core approach, reducing the absolute gap to the 1.19 target without introducing new modeling ideas.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

try:
    from sklearnex import patch_sklearn  # type: ignore

    patch_sklearn()
except Exception:
    pass

np.random.seed(0)

INPUT_CANDIDATES = [
    "../input",
    "/kaggle/input",
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling/champs-scalar-coupling",
]


def resolve_path(filename: str) -> str:
    for base in INPUT_CANDIDATES:
        p = os.path.join(base, filename)
        if os.path.exists(p):
            return p
    return os.path.join("../input", filename)


print("Listing ../input (if exists):")
try:
    print(os.listdir("../input"))
except Exception as e:
    print("Could not list ../input:", e)



## === cell 1
train_path = resolve_path("train.csv")
trainSet = pd.read_csv(
    train_path,
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
        "scalar_coupling_constant": np.float64,
    },
)
print(trainSet.head())



## === cell 2
test_path = resolve_path("test.csv")
testSet = pd.read_csv(
    test_path,
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
print(testSet.head())



## === cell 3
structures_path = resolve_path("structures.csv")
structures = pd.read_csv(
    structures_path,
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
)
print(structures.head())



## === cell 4
structures_small = structures[
    ["molecule_name", "atom_index", "atom", "x", "y", "z"]
].copy()

all_molecules = pd.Index(
    pd.concat(
        [trainSet["molecule_name"].astype(str), testSet["molecule_name"].astype(str)]
    ).unique()
)
mol_cat = pd.Categorical(
    pd.Index(all_molecules), categories=all_molecules, ordered=False
)

mol_code_map = {m: i for i, m in enumerate(all_molecules.tolist())}
structures_small["mol_code"] = (
    structures_small["molecule_name"].astype(str).map(mol_code_map).astype(np.int32)
)


def map_atom_info_fast_inplace(df: pd.DataFrame) -> None:
    df["mol_code"] = df["molecule_name"].astype(str).map(mol_code_map).astype(np.int32)

    s0 = structures_small.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom_0",
            "x": "x_0",
            "y": "y_0",
            "z": "z_0",
        }
    )[["mol_code", "atom_index_0", "atom_0", "x_0", "y_0", "z_0"]]
    df.merge(
        s0,
        on=["mol_code", "atom_index_0"],
        how="left",
        copy=False,
        sort=False,
        inplace=True,
    )

    s1 = structures_small.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom_1",
            "x": "x_1",
            "y": "y_1",
            "z": "z_1",
        }
    )[["mol_code", "atom_index_1", "atom_1", "x_1", "y_1", "z_1"]]
    df.merge(
        s1,
        on=["mol_code", "atom_index_1"],
        how="left",
        copy=False,
        sort=False,
        inplace=True,
    )


map_atom_info_fast_inplace(trainSet)
map_atom_info_fast_inplace(testSet)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2758106171.py in <cell line: 0>()
     64 
     65 
---> 66 map_atom_info_fast_inplace(trainSet)
     67 map_atom_info_fast_inplace(testSet)
     68 

/tmp/ipykernel_11/2758106171.py in map_atom_info_fast_inplace(df)
     36         }
     37     )[["mol_code", "atom_index_0", "atom_0", "x_0", "y_0", "z_0"]]
---> 38     df.merge(
     39         s0,
     40         on=["mol_code", "atom_index_0"],

TypeError: DataFrame.merge() got an unexpected keyword argument 'inplace'

## === cell 5
print(trainSet.head())
print(testSet.head())



## === cell 6
coord_cols = ["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"]
for df_name, df in [("trainSet", trainSet), ("testSet", testSet)]:
    missing_coords = df[coord_cols].isna().any(axis=1).sum()
    if missing_coords > 0:
        print(
            f"Warning: {df_name} has rows with missing coordinates:",
            int(missing_coords),
        )
        df.loc[:, coord_cols] = df[coord_cols].fillna(0.0)

train_p0 = trainSet[["x_0", "y_0", "z_0"]].to_numpy(dtype=np.float32, copy=False)
train_p1 = trainSet[["x_1", "y_1", "z_1"]].to_numpy(dtype=np.float32, copy=False)
test_p0 = testSet[["x_0", "y_0", "z_0"]].to_numpy(dtype=np.float32, copy=False)
test_p1 = testSet[["x_1", "y_1", "z_1"]].to_numpy(dtype=np.float32, copy=False)

train_delta = train_p0 - train_p1
test_delta = test_p0 - test_p1

trainSet["dx"] = train_delta[:, 0]
trainSet["dy"] = train_delta[:, 1]
trainSet["dz"] = train_delta[:, 2]

testSet["dx"] = test_delta[:, 0]
testSet["dy"] = test_delta[:, 1]
testSet["dz"] = test_delta[:, 2]

train_dist = np.sqrt(
    (train_delta * train_delta).sum(axis=1, dtype=np.float32), dtype=np.float32
)
test_dist = np.sqrt(
    (test_delta * test_delta).sum(axis=1, dtype=np.float32), dtype=np.float32
)
trainSet["dist"] = train_dist
testSet["dist"] = test_dist

eps = 1e-3
trainSet["inv_dist"] = 1.0 / (trainSet["dist"].astype(np.float64) + eps)
testSet["inv_dist"] = 1.0 / (testSet["dist"].astype(np.float64) + eps)

type_mean_dist = trainSet.groupby("type", observed=True)["dist"].mean()
global_mean_dist = float(trainSet["dist"].mean())

train_den = trainSet["type"].map(type_mean_dist)
test_den = testSet["type"].map(type_mean_dist).fillna(global_mean_dist)

trainSet["dist_to_type_mean"] = trainSet["dist"].astype(np.float64) / train_den.astype(
    np.float64
)
testSet["dist_to_type_mean"] = testSet["dist"].astype(np.float64) / test_den.astype(
    np.float64
)

for df in (trainSet, testSet):
    df["dist_to_type_mean"] = (
        df["dist_to_type_mean"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    )
    df["inv_dist"] = df["inv_dist"].replace([np.inf, -np.inf], np.nan).fillna(0.0)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1451518578.py in <cell line: 0>()
      3 coord_cols = ["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"]
      4 for df_name, df in [("trainSet", trainSet), ("testSet", testSet)]:
----> 5     missing_coords = df[coord_cols].isna().any(axis=1).sum()
      6     if missing_coords > 0:
      7         print(

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index(['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1'], dtype='object')] are in the [columns]"

## === cell 7
train_atom0_unique = trainSet["atom_0"].dropna().unique()
test_atom0_unique = testSet["atom_0"].dropna().unique()

if not (len(train_atom0_unique) == 1 and train_atom0_unique[0] == "H"):
    print("Warning: train atom_0 is not always H. Unique values:", train_atom0_unique)
if not (len(test_atom0_unique) == 1 and test_atom0_unique[0] == "H"):
    print("Warning: test atom_0 is not always H. Unique values:", test_atom0_unique)



## --- ERROR in cell 7, traceback:
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

KeyError: 'atom_0'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3185041195.py in <cell line: 0>()
----> 1 train_atom0_unique = trainSet["atom_0"].dropna().unique()
      2 test_atom0_unique = testSet["atom_0"].dropna().unique()
      3 
      4 if not (len(train_atom0_unique) == 1 and train_atom0_unique[0] == "H"):
      5     print("Warning: train atom_0 is not always H. Unique values:", train_atom0_unique)

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

KeyError: 'atom_0'

## === cell 8
print(trainSet["atom_1"].astype("category").cat.categories)
print(testSet["atom_1"].astype("category").cat.categories)



## --- ERROR in cell 8, traceback:
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

KeyError: 'atom_1'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1934077235.py in <cell line: 0>()
----> 1 print(trainSet["atom_1"].astype("category").cat.categories)
      2 print(testSet["atom_1"].astype("category").cat.categories)
      3 

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

KeyError: 'atom_1'

## === cell 9
print(testSet["type"].astype("category").cat.categories)
print(trainSet["type"].astype("category").cat.categories)



## === cell 10
type_categories = trainSet["type"].astype("category").cat.categories.to_numpy()
type_to_code = {t: i for i, t in enumerate(type_categories)}

train_type_codes = (
    trainSet["type"].map(type_to_code).to_numpy(dtype=np.int32, copy=False)
)
test_type_codes = (
    testSet["type"].map(type_to_code).to_numpy(dtype=np.float64, copy=False)
)
test_type_codes = np.where(
    np.isfinite(test_type_codes), test_type_codes.astype(np.int32), -1
)

extra_test_types = set(testSet["type"].unique()) - set(type_categories)
if len(extra_test_types) > 0:
    print(
        "Warning: unseen types in test not present in train:",
        sorted(list(extra_test_types)),
    )



## === cell 11
base_model_params = dict(
    fit_intercept=True,
    max_iter=1000,
    tol=1e-5,
    warm_start=False,
)



## === cell 12
atom0_cats = trainSet["atom_0"].astype("category").cat.categories.to_numpy()
atom1_cats = trainSet["atom_1"].astype("category").cat.categories.to_numpy()
atom0_to_code = {t: i for i, t in enumerate(atom0_cats)}
atom1_to_code = {t: i for i, t in enumerate(atom1_cats)}

train_atom0_codes = (
    trainSet["atom_0"].map(atom0_to_code).to_numpy(dtype=np.int32, copy=False)
)
train_atom1_codes = (
    trainSet["atom_1"].map(atom1_to_code).to_numpy(dtype=np.int32, copy=False)
)

test_atom0_codes = (
    testSet["atom_0"].map(atom0_to_code).to_numpy(dtype=np.float64, copy=False)
)
test_atom1_codes = (
    testSet["atom_1"].map(atom1_to_code).to_numpy(dtype=np.float64, copy=False)
)
test_atom0_codes = np.where(
    np.isfinite(test_atom0_codes), test_atom0_codes.astype(np.int32), -1
)
test_atom1_codes = np.where(
    np.isfinite(test_atom1_codes), test_atom1_codes.astype(np.int32), -1
)



## --- ERROR in cell 12, traceback:
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

KeyError: 'atom_0'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1108529446.py in <cell line: 0>()
      1 # CHANGED: Same approach for atom one-hots: keep codes for later sparse encoding.
----> 2 atom0_cats = trainSet["atom_0"].astype("category").cat.categories.to_numpy()
      3 atom1_cats = trainSet["atom_1"].astype("category").cat.categories.to_numpy()
      4 atom0_to_code = {t: i for i, t in enumerate(atom0_cats)}
      5 atom1_to_code = {t: i for i, t in enumerate(atom1_cats)}

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

KeyError: 'atom_0'

## === cell 13
dipole_path = resolve_path("dipole_moments.csv")
pe_path = resolve_path("potential_energy.csv")
mull_path = resolve_path("mulliken_charges.csv")
mst_path = resolve_path("magnetic_shielding_tensors.csv")

dipole = pd.read_csv(
    dipole_path,
    dtype={
        "molecule_name": "category",
        "X": np.float32,
        "Y": np.float32,
        "Z": np.float32,
    },
).rename(columns={"X": "dipole_X", "Y": "dipole_Y", "Z": "dipole_Z"})
dipole["mol_code"] = (
    dipole["molecule_name"].astype(str).map(mol_code_map).astype(np.int32)
)
dipole_idx = dipole.set_index("mol_code")[["dipole_X", "dipole_Y", "dipole_Z"]]

pe = pd.read_csv(
    pe_path, dtype={"molecule_name": "category", "potential_energy": np.float32}
)
pe["mol_code"] = pe["molecule_name"].astype(str).map(mol_code_map).astype(np.int32)
pe_idx = pe.set_index("mol_code")[["potential_energy"]]

mull = pd.read_csv(
    mull_path,
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "mulliken_charge": np.float32,
    },
)
mull["mol_code"] = mull["molecule_name"].astype(str).map(mol_code_map).astype(np.int32)
mull_idx = mull.set_index(["mol_code", "atom_index"])[["mulliken_charge"]]

mst = pd.read_csv(
    mst_path,
    dtype={
        "molecule_name": "category",
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
)
mst["mol_code"] = mst["molecule_name"].astype(str).map(mol_code_map).astype(np.int32)

tensor_cols = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
mst["mst_trace"] = mst["XX"] + mst["YY"] + mst["ZZ"]
mst["mst_mean"] = mst[tensor_cols].mean(axis=1)
mst["mst_std"] = mst[tensor_cols].std(axis=1)
mst_small = mst[["mol_code", "atom_index", "mst_trace", "mst_mean", "mst_std"]]
mst_idx = mst_small.set_index(["mol_code", "atom_index"])[
    ["mst_trace", "mst_mean", "mst_std"]
]


def add_external_features_fast_inplace(df: pd.DataFrame) -> None:
    df.merge(
        dipole_idx.reset_index(),
        on="mol_code",
        how="left",
        copy=False,
        sort=False,
        inplace=True,
    )
    df.merge(
        pe_idx.reset_index(),
        on="mol_code",
        how="left",
        copy=False,
        sort=False,
        inplace=True,
    )

    mc = mull_idx.reset_index()
    ms = mst_idx.reset_index()

    mc0 = mc.rename(
        columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_charge_0"}
    )
    mc1 = mc.rename(
        columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_charge_1"}
    )
    df.merge(
        mc0[["mol_code", "atom_index_0", "mulliken_charge_0"]],
        on=["mol_code", "atom_index_0"],
        how="left",
        copy=False,
        sort=False,
        inplace=True,
    )
    df.merge(
        mc1[["mol_code", "atom_index_1", "mulliken_charge_1"]],
        on=["mol_code", "atom_index_1"],
        how="left",
        copy=False,
        sort=False,
        inplace=True,
    )

    ms0 = ms.rename(
        columns={
            "atom_index": "atom_index_0",
            "mst_trace": "mst_trace_0",
            "mst_mean": "mst_mean_0",
            "mst_std": "mst_std_0",
        }
    )
    ms1 = ms.rename(
        columns={
            "atom_index": "atom_index_1",
            "mst_trace": "mst_trace_1",
            "mst_mean": "mst_mean_1",
            "mst_std": "mst_std_1",
        }
    )
    df.merge(
        ms0[["mol_code", "atom_index_0", "mst_trace_0", "mst_mean_0", "mst_std_0"]],
        on=["mol_code", "atom_index_0"],
        how="left",
        copy=False,
        sort=False,
        inplace=True,
    )
    df.merge(
        ms1[["mol_code", "atom_index_1", "mst_trace_1", "mst_mean_1", "mst_std_1"]],
        on=["mol_code", "atom_index_1"],
        how="left",
        copy=False,
        sort=False,
        inplace=True,
    )

    df["mulliken_charge_diff"] = df["mulliken_charge_0"] - df["mulliken_charge_1"]
    df["mulliken_charge_sum"] = df["mulliken_charge_0"] + df["mulliken_charge_1"]
    df["mst_trace_diff"] = df["mst_trace_0"] - df["mst_trace_1"]
    df["mst_trace_sum"] = df["mst_trace_0"] + df["mst_trace_1"]


add_external_features_fast_inplace(trainSet)
add_external_features_fast_inplace(testSet)

ext_num_cols = [
    "dipole_X",
    "dipole_Y",
    "dipole_Z",
    "potential_energy",
    "mulliken_charge_0",
    "mulliken_charge_1",
    "mulliken_charge_diff",
    "mulliken_charge_sum",
    "mst_trace_0",
    "mst_mean_0",
    "mst_std_0",
    "mst_trace_1",
    "mst_mean_1",
    "mst_std_1",
    "mst_trace_diff",
    "mst_trace_sum",
]
for df_name, df in [("trainSet", trainSet), ("testSet", testSet)]:
    for c in ext_num_cols:
        if c not in df.columns:
            df[c] = 0.0
    arr = df[ext_num_cols].to_numpy(dtype=np.float64, copy=False)
    nonfinite = (~np.isfinite(arr)).sum()
    if nonfinite > 0:
        print(
            f"Warning: {df_name} has non-finite external feature values; filling with 0. Count:",
            int(nonfinite),
        )
        df.loc[:, ext_num_cols] = (
            df[ext_num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3580156267.py in <cell line: 0>()
    147 
    148 
--> 149 add_external_features_fast_inplace(trainSet)
    150 add_external_features_fast_inplace(testSet)
    151 

/tmp/ipykernel_11/3580156267.py in add_external_features_fast_inplace(df)
     65 # CHANGED: Avoid MultiIndex.from_frame + reindex per df; use merges with prepared key columns.
     66 def add_external_features_fast_inplace(df: pd.DataFrame) -> None:
---> 67     df.merge(
     68         dipole_idx.reset_index(),
     69         on="mol_code",

TypeError: DataFrame.merge() got an unexpected keyword argument 'inplace'

## === cell 14
from scipy import (
    sparse,
)  # available in Kaggle python images; required for CSR efficiency

numeric_feature_cols = [
    "dist",
    "dist_to_type_mean",
    "inv_dist",
    "dx",
    "dy",
    "dz",
] + ext_num_cols

train_num = trainSet[numeric_feature_cols].to_numpy(dtype=np.float64, copy=False)
train_num = np.nan_to_num(train_num, nan=0.0, posinf=0.0, neginf=0.0)

num_mean = train_num.mean(axis=0)
num_std = train_num.std(axis=0)
num_std = np.where(num_std == 0.0, 1.0, num_std)


def standardized_numeric(df: pd.DataFrame) -> np.ndarray:
    arr = df[numeric_feature_cols].to_numpy(dtype=np.float64, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return (arr - num_mean) / num_std


X_train_num = standardized_numeric(trainSet)
X_test_num = standardized_numeric(testSet)


def one_hot_csr(codes: np.ndarray, n_classes: int) -> sparse.csr_matrix:
    n = codes.shape[0]
    valid = codes >= 0
    rows = np.nonzero(valid)[0]
    cols = codes[valid].astype(np.int32, copy=False)
    data = np.ones(rows.shape[0], dtype=np.float64)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n_classes))


X_train_type = one_hot_csr(train_type_codes, len(type_categories))
X_test_type = one_hot_csr(test_type_codes, len(type_categories))
X_train_atom0 = one_hot_csr(train_atom0_codes, len(atom0_cats))
X_test_atom0 = one_hot_csr(test_atom0_codes, len(atom0_cats))
X_train_atom1 = one_hot_csr(train_atom1_codes, len(atom1_cats))
X_test_atom1 = one_hot_csr(test_atom1_codes, len(atom1_cats))

X_train_all = sparse.hstack(
    [X_train_type, X_train_atom0, X_train_atom1, sparse.csr_matrix(X_train_num)],
    format="csr",
)
X_test_all = sparse.hstack(
    [X_test_type, X_test_atom0, X_test_atom1, sparse.csr_matrix(X_test_num)],
    format="csr",
)

y_train_all = trainSet["scalar_coupling_constant"].to_numpy(
    dtype=np.float64, copy=False
)

if not np.all(np.isfinite(X_train_num)):
    print(
        "Warning: non-finite values in training numeric features after sanitization (unexpected)."
    )




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2065906854.py in <cell line: 0>()
     13     "dy",
     14     "dz",
---> 15 ] + ext_num_cols
     16 
     17 train_num = trainSet[numeric_feature_cols].to_numpy(dtype=np.float64, copy=False)

NameError: name 'ext_num_cols' is not defined

## === cell 15
def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    types = pd.Series(types)
    maes = (y_true - y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()




## === cell 16
global_model = HuberRegressor(**base_model_params).fit(X_train_all, y_train_all)

train_types_arr = trainSet["type"].to_numpy()
unique_types = np.array(sorted(trainSet["type"].unique()))

order = np.argsort(train_types_arr, kind="mergesort")
types_sorted = train_types_arr[order]

unique_types_sorted, start_idx, counts = np.unique(
    types_sorted, return_index=True, return_counts=True
)

type_to_slice = {
    t: (start, start + cnt)
    for t, start, cnt in zip(unique_types_sorted, start_idx, counts)
}

type_models = {}
for t in unique_types:
    start, end = type_to_slice[t]
    idx_sorted = order[start:end]
    X_t = X_train_all[idx_sorted]
    y_t = y_train_all[idx_sorted]
    type_models[t] = HuberRegressor(**base_model_params).fit(X_t, y_t)

print("Trained per-type models:", len(type_models), "Global fallback model: yes")



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1736870137.py in <cell line: 0>()
----> 1 global_model = HuberRegressor(**base_model_params).fit(X_train_all, y_train_all)
      2 
      3 train_types_arr = trainSet["type"].to_numpy()
      4 unique_types = np.array(sorted(trainSet["type"].unique()))
      5 

NameError: name 'X_train_all' is not defined

## === cell 17
train_pred_global = global_model.predict(X_train_all)

train_pred_per_type = np.empty(len(trainSet), dtype=np.float64)
for t in unique_types:
    start, end = type_to_slice[t]
    idx_sorted = order[start:end]
    train_pred_per_type[idx_sorted] = type_models[t].predict(X_train_all[idx_sorted])

print(
    "In-sample group logMAE (global model):",
    group_mean_log_mae(y_train_all, train_pred_global, trainSet["type"]),
)
print(
    "In-sample group logMAE (per-type models):",
    group_mean_log_mae(y_train_all, train_pred_per_type, trainSet["type"]),
)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3231788658.py in <cell line: 0>()
----> 1 train_pred_global = global_model.predict(X_train_all)
      2 
      3 train_pred_per_type = np.empty(len(trainSet), dtype=np.float64)
      4 for t in unique_types:
      5     start, end = type_to_slice[t]

NameError: name 'global_model' is not defined

## === cell 18
test_pred = np.empty(len(testSet), dtype=np.float64)
test_types = testSet["type"].to_numpy()

test_order = np.argsort(test_types, kind="mergesort")
test_types_sorted = test_types[test_order]
test_unique_sorted, test_start, test_counts = np.unique(
    test_types_sorted, return_index=True, return_counts=True
)
test_slice = {
    t: (s, s + c) for t, s, c in zip(test_unique_sorted, test_start, test_counts)
}

for t in test_unique_sorted:
    s, e = test_slice[t]
    idx_sorted = test_order[s:e]
    X_m = X_test_all[idx_sorted]
    if t in type_models:
        test_pred[idx_sorted] = type_models[t].predict(X_m)
    else:
        test_pred[idx_sorted] = global_model.predict(X_m)

resultSet = pd.DataFrame(
    {
        "id": testSet["id"].to_numpy(),
        "scalar_coupling_constant": test_pred,
    }
)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3961843332.py in <cell line: 0>()
     15     s, e = test_slice[t]
     16     idx_sorted = test_order[s:e]
---> 17     X_m = X_test_all[idx_sorted]
     18     if t in type_models:
     19         test_pred[idx_sorted] = type_models[t].predict(X_m)

NameError: name 'X_test_all' is not defined

## === cell 19
out_path = "submission.csv"
resultSet.to_csv(out_path, index=False)

with open(out_path, "r") as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i > 5:
            break
print("Wrote:", out_path, "rows:", len(resultSet))

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2372780226.py in <cell line: 0>()
      1 out_path = "submission.csv"
----> 2 resultSet.to_csv(out_path, index=False)
      3 
      4 with open(out_path, "r") as f:
      5     for i, line in enumerate(f):

NameError: name 'resultSet' is not defined
