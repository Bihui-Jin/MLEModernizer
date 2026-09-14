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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os

print(os.listdir("../input"))



## === cell 1
structures = pd.read_csv("../input/structures.csv")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")




## === cell 2
def map_atom_info(df, atom_idx):
    df = pd.merge(
        df,
        structures,
        how="left",
        left_on=["molecule_name", f"atom_index_{atom_idx}"],
        right_on=["molecule_name", "atom_index"],
    )

    df = df.drop("atom_index", axis=1)
    df = df.rename(
        columns={
            "atom": f"atom_{atom_idx}",
            "x": f"x_{atom_idx}",
            "y": f"y_{atom_idx}",
            "z": f"z_{atom_idx}",
        }
    )
    return df


train = map_atom_info(train, 0)
train = map_atom_info(train, 1)

test = map_atom_info(test, 0)
test = map_atom_info(test, 1)



## === cell 3
train.head()



## === cell 4
train["dist"] = (
    (train["x_1"] - train["x_0"]) ** 2
    + (train["y_1"] - train["y_0"]) ** 2
    + (train["z_1"] - train["z_0"]) ** 2
) ** 0.5
test["dist"] = (
    (test["x_1"] - test["x_0"]) ** 2
    + (test["y_1"] - test["y_0"]) ** 2
    + (test["z_1"] - test["z_0"]) ** 2
) ** 0.5



## === cell 5
print(train["atom_0"].value_counts())
train = train.drop(["atom_0", "atom_index_1", "atom_index_0"], axis=1)
test = test.drop(["atom_0", "atom_index_1", "atom_index_0"], axis=1)



## === cell 6
train.head()



## === cell 7
test.head()



## === cell 8
sns.distplot(train.scalar_coupling_constant)



## === cell 9
sns.countplot(x=train["type"])



## === cell 10
sns.countplot(x=train["atom_1"])



## === cell 11
sns.boxplot(x=train.atom_1, y=train.scalar_coupling_constant, palette="rainbow")



## === cell 12
sns.boxplot(x=train.type, y=train.scalar_coupling_constant, palette="rainbow")



## === cell 13
sns.distplot(train.dist)



## === cell 14
plt.scatter(train.dist, train.scalar_coupling_constant)



## === cell 15
pass



## === cell 16
train = pd.get_dummies(train)
test = pd.get_dummies(test)



## === cell 17
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge


def champs_metric_from_arrays(y_true, y_pred, type_onehot_df):
    type_cols = [c for c in type_onehot_df.columns if c.startswith("type_")]
    maes = []
    for c in type_cols:
        mask = type_onehot_df[c].values == 1
        if mask.sum() == 0:
            continue
        mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
        maes.append(mae)
    if len(maes) == 0:
        return float(np.log(np.mean(np.abs(y_true - y_pred)) + 1e-9))
    return float(np.mean(np.log(np.array(maes) + 1e-9)))


def cv_champs_score_grouped(estimator, X_all, y_all, groups, n_splits=3):
    gkf = GroupKFold(n_splits=n_splits)
    type_cols = [c for c in X_all.columns if c.startswith("type_")]
    scores = []
    for tr_idx, va_idx in gkf.split(X_all, y_all, groups=groups):
        X_tr = X_all.iloc[tr_idx]
        y_tr = y_all.iloc[tr_idx].values
        X_va = X_all.iloc[va_idx]
        y_va = y_all.iloc[va_idx].values

        m = estimator
        m.fit(X_tr, y_tr)
        pred_va = m.predict(X_va)

        score_va = champs_metric_from_arrays(y_va, pred_va, X_va[type_cols])
        scores.append(score_va)
    return np.array(scores)




## === cell 18
X_full = train.drop(["scalar_coupling_constant", "id"], axis=1)
Y = train["scalar_coupling_constant"]

mol_cols = [c for c in X_full.columns if c.startswith("molecule_name_")]
if len(mol_cols) == 0:
    groups = np.zeros(len(X_full), dtype=np.int64)
else:
    mol_mat = X_full[mol_cols].to_numpy()
    groups = mol_mat.argmax(axis=1).astype(np.int64)

alpha_list = np.concatenate(
    [
        np.array([1e-4, 3e-4, 1e-3, 3e-3]),
        np.array([1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0]),
    ]
)

L = []
for alpha_val in alpha_list:
    model = Ridge(alpha=float(alpha_val), random_state=42)
    L.append(
        cv_champs_score_grouped(model, X_full, Y, groups=groups, n_splits=3).mean()
    )

plt.plot(alpha_list, L)
best_alpha = float(alpha_list[int(np.argmin(L))])
print("best_cv_score(logMAE_by_type, GroupKFold by molecule):", float(np.min(L)))
print("best_alpha:", best_alpha)



## === cell 19
drop_mol_cols_train = [c for c in X_full.columns if c.startswith("molecule_name_")]
X = X_full.drop(columns=drop_mol_cols_train, errors="ignore")

id_test = test["id"]
test = test.drop(["id"], axis=1)
drop_mol_cols_test = [c for c in test.columns if c.startswith("molecule_name_")]
test = test.drop(columns=drop_mol_cols_test, errors="ignore")

test = test.reindex(columns=X.columns, fill_value=0)
test = test.fillna(0)
X_filled = X.fillna(0)

type_cols = [c for c in X.columns if c.startswith("type_")]
if len(type_cols) == 0:
    model = Ridge(alpha=best_alpha, random_state=42)
    model.fit(X_filled, Y)
    pred = model.predict(test)
else:
    pred = np.zeros(len(test), dtype=np.float64)
    for tc in type_cols:
        tr_mask = X_filled[tc].values == 1
        te_mask = test[tc].values == 1

        if tr_mask.sum() == 0:
            continue

        model = Ridge(alpha=best_alpha, random_state=42)
        model.fit(X_filled.loc[tr_mask, :], Y.loc[tr_mask])

        if te_mask.sum() > 0:
            pred[te_mask] = model.predict(test.loc[te_mask, :])



## === cell 20
test_output = pd.DataFrame({"id": id_test, "scalar_coupling_constant": pred})
test_output.set_index("id", inplace=True)
test_output.to_csv("prediction1.csv")
print("Wrote prediction1.csv with shape:", test_output.shape)
