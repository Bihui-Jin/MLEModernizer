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

# 5. Target score

0.91797

# 6. Current score

1.98714

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39512) has done: 'I fixed the LightGBM training call which rejected the `early_stopping_rounds` argument by using the modern callback API, renamed the cells to start from 1, and kept all original feature engineering and model logic unchanged. The script now runs end‑to‑end and writes a correctly formatted `submission.csv`.'
- What this solution (achieved 1.72214) has done: 'The changes add several informative molecular features (dipole magnitude, potential energy, and per‑atom Mulliken charges) that are cheap to compute and help the LightGBM model better explain variance in the coupling constant. The training parameters are adjusted to use a custom log‑MAE metric for early stopping (setting `metric":"none"` and providing `feval`) and a smaller learning rate with more potential boosting rounds, allowing the model to improve without over‑fitting. These minimal, targeted adjustments keep the original workflow intact while steering the validation score closer to the target lower‑is‑better value.'
- What this solution (achieved 1.98714) has done: 'I add a few cheap but informative features (the scalar‑coupling contributions, charge sum/difference, and a numeric encoding of the whole coupling type) and slightly relax the LightGBM early‑stopping to let the model train a bit longer with a larger leaf count and smaller learning rate. These changes keep the original workflow and model type intact while providing the model with stronger signals to lower the log‑MAE toward the target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics, model_selection
import lightgbm as lgb

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sub = pd.read_csv("../input/sample_submission.csv")
print(train.shape, test.shape, sub.shape)

train["atom"] = train["type"].map(lambda x: str(x)[3])
test["atom"] = test["type"].map(lambda x: str(x)[3])

lbl = preprocessing.LabelEncoder()
for i in range(4):
    train[f"type{i}"] = lbl.fit_transform(train["type"].map(lambda x: str(x)[i]))
    test[f"type{i}"] = lbl.transform(test["type"].map(lambda x: str(x)[i]))

type_lbl = preprocessing.LabelEncoder().fit(train["type"])
train["type_enc"] = type_lbl.transform(train["type"])
test["type_enc"] = type_lbl.transform(test["type"])

structures = pd.read_csv("../input/structures.csv").rename(
    columns={"atom_index": "atom_index_0", "x": "x0", "y": "y0", "z": "z0"}
)
train = pd.merge(
    train, structures, how="left", on=["molecule_name", "atom_index_0", "atom"]
)
test = pd.merge(
    test, structures, how="left", on=["molecule_name", "atom_index_0", "atom"]
)
del structures

structures = pd.read_csv("../input/structures.csv").rename(
    columns={"atom_index": "atom_index_1", "x": "x1", "y": "y1", "z": "z1"}
)
train = pd.merge(
    train, structures, how="left", on=["molecule_name", "atom_index_1", "atom"]
)
test = pd.merge(
    test, structures, how="left", on=["molecule_name", "atom_index_1", "atom"]
)
del structures

dipole = pd.read_csv("../input/dipole_moments.csv")
dipole["dipole_mag"] = np.sqrt((dipole[["X", "Y", "Z"]] ** 2).sum(axis=1))
train = train.merge(
    dipole[["molecule_name", "dipole_mag"]], on="molecule_name", how="left"
)
test = test.merge(
    dipole[["molecule_name", "dipole_mag"]], on="molecule_name", how="left"
)

pot = pd.read_csv("../input/potential_energy.csv")
train = train.merge(pot, on="molecule_name", how="left")
test = test.merge(pot, on="molecule_name", how="left")

charges = pd.read_csv("../input/mulliken_charges.csv")
charges0 = charges.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "charge0"}
)
charges1 = charges.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "charge1"}
)
train = train.merge(
    charges0[["molecule_name", "atom_index_0", "charge0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
test = test.merge(
    charges0[["molecule_name", "atom_index_0", "charge0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
train = train.merge(
    charges1[["molecule_name", "atom_index_1", "charge1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)
test = test.merge(
    charges1[["molecule_name", "atom_index_1", "charge1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)

train["charge_sum"] = train["charge0"] + train["charge1"]
train["charge_diff"] = train["charge0"] - train["charge1"]
test["charge_sum"] = test["charge0"] + test["charge1"]
test["charge_diff"] = test["charge0"] - test["charge1"]

contrib = pd.read_csv("../input/scalar_coupling_contributions.csv")
train = train.merge(
    contrib, on=["molecule_name", "atom_index_0", "atom_index_1", "type"], how="left"
)
test = test.merge(
    contrib, on=["molecule_name", "atom_index_0", "atom_index_1", "type"], how="left"
)

print(train.shape, test.shape, sub.shape)




## === cell 1
train_p0 = train[["x0", "y0", "z0"]].fillna(0).values
train_p1 = train[["x1", "y1", "z1"]].fillna(0).values
test_p0 = test[["x0", "y0", "z0"]].fillna(0).values
test_p1 = test[["x1", "y1", "z1"]].fillna(0).values

train["dist"] = np.linalg.norm(train_p0 - train_p1, axis=1)
test["dist"] = np.linalg.norm(test_p0 - test_p1, axis=1)

train["dist_to_type_mean"] = train["dist"] / train.groupby("type")["dist"].transform(
    "mean"
)
test["dist_to_type_mean"] = test["dist"] / test.groupby("type")["dist"].transform(
    "mean"
)




## === cell 2
col = [
    c
    for c in train.columns
    if c not in ["id", "molecule_name", "scalar_coupling_constant", "type", "atom"]
]


def lgb_lmae(preds, dtrain):
    labels = dtrain.get_label()
    mae = metrics.mean_absolute_error(labels, preds)
    return "lmae", np.log(mae + 1e-15), False  # log‑MAE for early stopping


params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "none",
    "learning_rate": 0.03,
    "num_leaves": 128,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
}

x_train, x_valid, y_train, y_valid = model_selection.train_test_split(
    train[col], train["scalar_coupling_constant"], test_size=0.2, random_state=99
)

train_set = lgb.Dataset(x_train, label=y_train)
valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set)

model = lgb.train(
    params,
    train_set,
    num_boost_round=5000,
    valid_sets=[valid_set],
    feval=lgb_lmae,
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
)

test["scalar_coupling_constant"] = model.predict(
    test[col], num_iteration=model.best_iteration
)

test[["id", "scalar_coupling_constant"]].to_csv(
    "submission.csv", float_format="%.9f", index=False
)
