# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

1.24022

# 6. Current score

2.16556

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 2.16541) has done: 'We fix the Ridge prediction error by ensuring train/test feature matrices have no NaNs after the merges and distance calculation (left-joins can create missing coordinates/atoms). The minimal score-neutral approach is to impute missing numeric features with the training median and missing categorical values with a constant before one-hot encoding, then align columns exactly. We also make the CV helper correct (it currently passes an integer instead of a CV splitter) while keeping the same Ridge model and training flow. Finally, we guarantee a valid `submission.csv` is written with the required columns.'
- What this solution (achieved 2.16556) has done: 'I fix the crash in cell 17 by ensuring the post–one-hot-encoding matrices are strictly numeric before checking for NaNs (the error indicates some columns remain non-numeric/object). I also make sure `train` and `test` are aligned and converted to `float32` consistently so Ridge and CV run without dtype issues, without changing the model or feature logic. Finally, I keep the same training/inference flow and guarantee a correctly formatted `submission.csv` is written. These fixes are execution/stability-oriented and should be score-neutral to slightly positive (by preventing silent dtype/object issues).'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os

INPUT_DIR = (
    "/kaggle/input/champs-scalar-coupling"
    if os.path.exists("/kaggle/input/champs-scalar-coupling")
    else "../input"
)
print("INPUT_DIR =", INPUT_DIR)
print("Top-level /kaggle/input exists:", os.path.exists("/kaggle/input"))
if os.path.exists("/kaggle/input"):
    print("Available datasets under /kaggle/input:", os.listdir("/kaggle/input")[:20])



## === cell 1
structures = pd.read_csv(f"{INPUT_DIR}/structures.csv")
train = pd.read_csv(f"{INPUT_DIR}/train.csv")
test = pd.read_csv(f"{INPUT_DIR}/test.csv")




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
try:
    sns.histplot(train["scalar_coupling_constant"], kde=True)
    plt.show()
except Exception as e:
    print("Plot skipped (hist):", repr(e))



## === cell 9
try:
    sns.countplot(x=train["type"])
    plt.xticks(rotation=90)
    plt.show()
except Exception as e:
    print("Plot skipped (count type):", repr(e))



## === cell 10
try:
    sns.countplot(x=train["atom_1"])
    plt.show()
except Exception as e:
    print("Plot skipped (count atom_1):", repr(e))



## === cell 11
try:
    sns.boxplot(
        x=train["atom_1"], y=train["scalar_coupling_constant"], palette="rainbow"
    )
    plt.show()
except Exception as e:
    print("Plot skipped (box atom_1):", repr(e))



## === cell 12
try:
    sns.boxplot(x=train["type"], y=train["scalar_coupling_constant"], palette="rainbow")
    plt.xticks(rotation=90)
    plt.show()
except Exception as e:
    print("Plot skipped (box type):", repr(e))



## === cell 13
try:
    sns.histplot(train["dist"], kde=True)
    plt.show()
except Exception as e:
    print("Plot skipped (dist):", repr(e))



## === cell 14
try:
    plt.scatter(train["dist"], train["scalar_coupling_constant"], s=1, alpha=0.2)
    plt.show()
except Exception as e:
    print("Plot skipped (scatter):", repr(e))



## === cell 15
train = train.drop(["molecule_name"], axis=1)
test = test.drop(["molecule_name"], axis=1)



## === cell 16
cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in train.columns if c not in cat_cols]

for c in cat_cols:
    train[c] = train[c].fillna("Unknown")
    if c in test.columns:
        test[c] = test[c].fillna("Unknown")

num_fill_cols = [c for c in num_cols if c not in ["scalar_coupling_constant", "id"]]
medians = train[num_fill_cols].median(numeric_only=True)
train[num_fill_cols] = train[num_fill_cols].fillna(medians)
test[num_fill_cols] = test[num_fill_cols].fillna(medians)

test = test.fillna(0)

train = pd.get_dummies(train)
test = pd.get_dummies(test)



## === cell 17
X = train.drop(["scalar_coupling_constant", "id"], axis=1)
Y = train["scalar_coupling_constant"]

id_test = test["id"]
test = test.drop(["id"], axis=1)

test = test.reindex(columns=X.columns, fill_value=0)

X = X.apply(pd.to_numeric, errors="coerce")
test = test.apply(pd.to_numeric, errors="coerce")

fill_vals = X.median(numeric_only=True)
X = X.fillna(fill_vals)
test = test.fillna(fill_vals)

X = X.astype(np.float32)
test = test.astype(np.float32)
Y = Y.astype(np.float32)

print("X shape:", X.shape, "test shape:", test.shape)
print(
    "NaNs in X:",
    int(pd.isna(X).to_numpy().sum()),
    "NaNs in test:",
    int(pd.isna(test).to_numpy().sum()),
)



## === cell 18
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge


def score(estimator):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse = np.sqrt(
        -cross_val_score(estimator, X, Y, scoring="neg_mean_squared_error", cv=cv)
    )
    return rmse




## === cell 19
alpha_list = np.linspace(start=0.1, stop=1, num=10)
L = []

for alpha_val in alpha_list:
    model = Ridge(alpha=alpha_val)
    L.append(score(model).mean())

plt.plot(alpha_list, L)
plt.show()
print(min(L))
print(alpha_list[np.argmin(L)])



## === cell 20
model = Ridge(alpha=0.1)
model.fit(X, Y)
pred = model.predict(test)



## === cell 21
submission = pd.DataFrame({"id": id_test.values, "scalar_coupling_constant": pred})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("submission.csv exists:", os.path.exists("submission.csv"))
