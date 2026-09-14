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
import pathlib
import pandas as pd
import numpy as np
import gc
from sklearn import preprocessing, impute, ensemble

from sklearnex import patch_sklearn

patch_sklearn()

base_path = pathlib.Path("../input")
if not (base_path / "train.csv").exists():
    base_path = pathlib.Path("./input")


def read_csv(rel_path, **kwargs):
    return pd.read_csv(base_path / rel_path, **kwargs)




## === cell 1
train = read_csv(
    "train.csv",
    usecols=[
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
    dtype={
        "id": "int64",
        "molecule_name": "category",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
        "scalar_coupling_constant": "float32",
    },
)
test = read_csv(
    "test.csv",
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    dtype={
        "id": "int64",
        "molecule_name": "category",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
    },
)

train["atom1"] = train["type"].str[2]
train["atom2"] = train["type"].str[3]
test["atom1"] = test["type"].str[2]
test["atom2"] = test["type"].str[3]

for i in range(4):
    col = train["type"].str[i]
    train[f"type{i}"], uniques = pd.factorize(col, sort=True)
    test[f"type{i}"] = pd.Categorical(test["type"].str[i], categories=uniques).codes

struct = read_csv(
    "structures.csv",
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "category",
        "atom_index": "int16",
        "atom": "category",
        "x": "float32",
        "y": "float32",
        "z": "float32",
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
)
struct1 = struct.rename(
    columns={
        "atom_index": "atom_index_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
        "atom": "atom2",
    }
)

struct0.set_index(["molecule_name", "atom_index_0", "atom1"], inplace=True)
struct1.set_index(["molecule_name", "atom_index_1", "atom2"], inplace=True)

train["is_train"] = 1
test["is_train"] = 0




## === cell 2
combined = pd.concat([train, test], ignore_index=True, sort=False)

combined = combined.join(
    struct0, on=["molecule_name", "atom_index_0", "atom1"], how="left"
)
combined = combined.join(
    struct1, on=["molecule_name", "atom_index_1", "atom2"], how="left"
)

del struct, struct0, struct1
gc.collect()

potential = read_csv(
    "potential_energy.csv",
    dtype={"molecule_name": "category", "potential_energy": "float32"},
)
combined = combined.merge(potential, how="left", on="molecule_name", sort=False)

dipole = read_csv(
    "dipole_moments.csv",
    dtype={"molecule_name": "category", "X": "float32", "Y": "float32", "Z": "float32"},
)
dipole["dipole_mag"] = np.sqrt(dipole["X"] ** 2 + dipole["Y"] ** 2 + dipole["Z"] ** 2)
dipole = dipole[["molecule_name", "dipole_mag"]]
combined = combined.merge(dipole, how="left", on="molecule_name", sort=False)

contrib = read_csv(
    "scalar_coupling_contributions.csv",
    dtype={
        "molecule_name": "category",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
        "fc": "float32",
        "sd": "float32",
        "pso": "float32",
        "dso": "float32",
    },
)
combined = combined.merge(
    contrib,
    how="left",
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    sort=False,
)

mulliken = read_csv(
    "mulliken_charges.csv",
    dtype={
        "molecule_name": "category",
        "atom_index": "int16",
        "mulliken_charge": "float32",
    },
)
m0 = mulliken.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_0"}
)
combined = combined.merge(
    m0, how="left", on=["molecule_name", "atom_index_0"], sort=False
)
m1 = mulliken.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_1"}
)
combined = combined.merge(
    m1, how="left", on=["molecule_name", "atom_index_1"], sort=False
)

del potential, dipole, contrib, mulliken, m0, m1
gc.collect()

p0 = combined[["x0", "y0", "z0"]].astype(np.float32).values
p1 = combined[["x1", "y1", "z1"]].astype(np.float32).values
combined["dist"] = np.sqrt(((p0 - p1) ** 2).sum(axis=1))

type_mean = combined.groupby("type")["dist"].transform("mean")
combined["dist_to_type_mean"] = combined["dist"] / type_mean

train = combined[combined["is_train"] == 1].copy()
test = combined[combined["is_train"] == 0].copy()
train.drop(columns=["is_train"], inplace=True)
test.drop(columns=["is_train"], inplace=True)




## === cell 3
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
feature_cols = [c for c in train.columns if c not in exclude_cols]

cat_cols = train.select_dtypes(include=["category"]).columns
if len(cat_cols):
    train[cat_cols] = train[cat_cols].apply(lambda s: s.cat.codes)
    test[cat_cols] = test[cat_cols].apply(lambda s: s.cat.codes)

numeric_cols = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
imputer = impute.SimpleImputer(strategy="median")
train[numeric_cols] = imputer.fit_transform(train[numeric_cols])
test[numeric_cols] = imputer.transform(test[numeric_cols])

X_train = train[feature_cols].values.astype(np.float32, copy=False)
y_train = train["scalar_coupling_constant"].values
X_test = test[feature_cols].values.astype(np.float32, copy=False)

reg = ensemble.ExtraTreesRegressor(
    n_estimators=200,
    n_jobs=-1,
    random_state=4,
)

reg.fit(X_train, y_train)
test["scalar_coupling_constant"] = reg.predict(X_test)

test[["id", "scalar_coupling_constant"]].to_csv("submission.csv", index=False)
