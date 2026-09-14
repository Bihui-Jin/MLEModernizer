# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb

np.random.seed(99)

BASE = "../input"
if not os.path.exists(BASE):
    if os.path.exists("/kaggle/data/champs-scalar-coupling"):
        BASE = "/kaggle/data/champs-scalar-coupling"
    elif os.path.exists("/kaggle/input/champs-scalar-coupling"):
        BASE = "/kaggle/input/champs-scalar-coupling"
    elif os.path.exists("/kaggle/input"):
        BASE = "/kaggle/input"

train = pd.read_csv(os.path.join(BASE, "train.csv"))
test = pd.read_csv(os.path.join(BASE, "test.csv"))
sub = pd.read_csv(os.path.join(BASE, "sample_submission.csv"))
print(train.shape, test.shape, sub.shape)

train_type = train["type"].astype(str)
test_type = test["type"].astype(str)
train["atom1"] = train_type.str[2]
train["atom2"] = train_type.str[3]
test["atom1"] = test_type.str[2]
test["atom2"] = test_type.str[3]

for i in range(4):
    lbl = preprocessing.LabelEncoder()
    all_vals = (
        pd.concat([train_type.str[i], test_type.str[i]], axis=0).astype(str).values
    )
    lbl.fit(all_vals)
    train[f"type{i}"] = lbl.transform(train_type.str[i].astype(str).values)
    test[f"type{i}"] = lbl.transform(test_type.str[i].astype(str).values)

structures = pd.read_csv(os.path.join(BASE, "structures.csv"))

structures_small = structures[["molecule_name", "atom_index", "x", "y", "z"]].copy()
structures_small["molecule_name"] = structures_small["molecule_name"].astype("category")
structures_small["atom_index"] = structures_small["atom_index"].astype(np.int32)
structures_small[["x", "y", "z"]] = structures_small[["x", "y", "z"]].astype(np.float32)


def attach_coords_fast(df, atom_index_col, prefix):
    tmp = structures_small.rename(
        columns={
            "atom_index": atom_index_col,
            "x": f"x{prefix}",
            "y": f"y{prefix}",
            "z": f"z{prefix}",
        }
    )
    out = df.merge(
        tmp, on=["molecule_name", atom_index_col], how="left", sort=False, copy=False
    )
    return out


train = attach_coords_fast(train, "atom_index_0", "0")
test = attach_coords_fast(test, "atom_index_0", "0")
train = attach_coords_fast(train, "atom_index_1", "1")
test = attach_coords_fast(test, "atom_index_1", "1")

del structures, structures_small
print(train.shape, test.shape, sub.shape)



## === cell 1
train_p0 = train[["x0", "y0", "z0"]].to_numpy(dtype=np.float64, copy=False)
train_p1 = train[["x1", "y1", "z1"]].to_numpy(dtype=np.float64, copy=False)
test_p0 = test[["x0", "y0", "z0"]].to_numpy(dtype=np.float64, copy=False)
test_p1 = test[["x1", "y1", "z1"]].to_numpy(dtype=np.float64, copy=False)

train_dist = np.linalg.norm(train_p0 - train_p1, axis=1)
test_dist = np.linalg.norm(test_p0 - test_p1, axis=1)
train["dist"] = train_dist
test["dist"] = test_dist

type_mean_dist = train.groupby("type", sort=False)["dist"].mean()
train["dist_to_type_mean"] = train_dist / train["type"].map(type_mean_dist).to_numpy()
test["dist_to_type_mean"] = test_dist / test["type"].map(type_mean_dist).to_numpy()



## === cell 2
col = [
    c
    for c in train.columns
    if c
    not in ["id", "molecule_name", "scalar_coupling_constant", "type", "atom1", "atom2"]
]


def lgb_lmae(preds, dtrain):
    labels = dtrain.get_label()
    score = np.log(metrics.mean_absolute_error(labels, preds))
    return "lmae", score, False


params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.2,
    "num_leaves": 64,
    "num_threads": -1,
    "seed": 99,
    "feature_fraction_seed": 99,
    "bagging_seed": 99,
    "data_random_seed": 99,
    "force_col_wise": True,
    "verbosity": -1,
    "histogram_pool_size": 2048,
}

X_train_full = train[col].to_numpy()
y_train_full = train["scalar_coupling_constant"].to_numpy()
X_test_full = test[col].to_numpy()

pred = np.zeros(len(test), dtype=np.float64)

train_type_vals = train["type"].values
test_type_vals = test["type"].values

for t in train["type"].unique():
    trn_mask = train_type_vals == t
    tst_mask = test_type_vals == t

    trn_groups = train.loc[trn_mask, "molecule_name"]
    gss = model_selection.GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=99)
    X_all = X_train_full[trn_mask]
    y_all = y_train_full[trn_mask]

    train_idx, valid_idx = next(gss.split(X_all, y_all, groups=trn_groups))

    x1 = X_all[train_idx]
    y1 = y_all[train_idx]
    x2 = X_all[valid_idx]
    y2 = y_all[valid_idx]

    dtrain = lgb.Dataset(x1, label=y1, free_raw_data=False)
    dvalid = lgb.Dataset(x2, label=y2, reference=dtrain, free_raw_data=False)

    model = lgb.train(
        params,
        dtrain,
        20000,
        valid_sets=[dvalid],
        feval=lgb_lmae,
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=1000),
        ],
    )

    pred[tst_mask] = model.predict(
        X_test_full[tst_mask], num_iteration=model.best_iteration
    )

test["scalar_coupling_constant"] = pred

out = sub[["id"]].merge(test[["id", "scalar_coupling_constant"]], on="id", how="left")
out[["id", "scalar_coupling_constant"]].to_csv(
    "submission.csv", float_format="%.9f", index=False
)
print(out.shape, out.head())
