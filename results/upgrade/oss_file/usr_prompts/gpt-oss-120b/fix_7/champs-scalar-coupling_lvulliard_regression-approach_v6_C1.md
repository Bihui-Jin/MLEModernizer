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

1.27404

# 6. Current score

11.32161

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 3.00294) has done: 'I removed the incorrect assertions about atom types, added safe handling for missing values after merging atom information, and filled any NaNs in numeric columns with zeros (and a placeholder for missing atom symbols). I also streamlined the workflow into sequential cells, ensured the one‑hot encoding of coupling types is applied consistently, and cast feature data to float before fitting and predicting with LinearRegression. Finally, the script now creates and writes a valid `results.csv` submission file.'
- What this solution (achieved 29.22928) has done: 'I add simple one‑hot encoding for the atom elements (`atom_0` and `atom_1`) to give the linear model extra relevant information, while keeping the original workflow and LinearRegression unchanged. The new dummy columns are created for the training set, the test set is aligned to the same dummy layout, and the feature list is built dynamically to include these columns together with the existing type‑one‑hots and distance features. This modest enrichment should lower the log‑MAE from the current 3.00 toward the target 1.27 without altering the core model or training procedure.'
- What this solution (achieved 3.29312) has done: 'I add richer atomic and molecular features (dipole moments, potential energy, Mulliken charges, and isotropic magnetic shielding) by merging the appropriate supplemental CSV files into both the train and test sets. These extra numeric columns are then included in the linear‑regression feature list, while keeping the original model and workflow untouched. Filling missing values with 0 ensures the pipeline runs end‑to‑end and writes a valid `results.csv` file, and the added chemistry‑related information should move the log‑MAE much closer to the target score.'
- What this solution (achieved 6.07616) has done: 'I add a simple quadratic distance feature (`dist_sq`) and train a separate LinearRegression model for each coupling type, which often improves predictive accuracy without changing the overall linear‑regression approach. These minimal enhancements keep the core workflow intact while aiming to lower the log‑MAE toward the target.'
- What this solution (achieved 11.32161) has done: 'I added three simple interaction features—product of the two atomic charges, product of the two shielding isotropic values, and a log‑distance term—to give the linear models a bit more expressive power. These columns are created after the existing missing‑value handling so they contain no NaNs, and they are included in the feature list used for training and prediction. This modest enrichment should lower the log‑MAE toward the target without altering the overall linear‑regression workflow.'
- What this solution (achieved 11.32161) has done: 'I add a simple per‑type bias correction: after fitting each LinearRegression model I compute the average residual on the training rows of that type and store it. When making predictions for the test set I add the corresponding bias to the model output. This small adjustment keeps the core linear‑regression workflow unchanged while often reducing the log‑MAE, moving the score closer to the target.'

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

print(os.listdir("../input"))

dipole_moments = pd.read_csv("../input/dipole_moments.csv")
potential_energy = pd.read_csv("../input/potential_energy.csv")
mulliken_charges = pd.read_csv("../input/mulliken_charges.csv")
magnetic_shielding = pd.read_csv("../input/magnetic_shielding_tensors.csv")
magnetic_shielding["shield_iso"] = (
    magnetic_shielding["XX"] + magnetic_shielding["YY"] + magnetic_shielding["ZZ"]
) / 3.0




## === cell 1
trainSet = pd.read_csv("../input/train.csv")




## === cell 2
testSet = pd.read_csv("../input/test.csv")




## === cell 3
structures = pd.read_csv("../input/structures.csv")




## === cell 4
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


def add_atom_feature(df, atom_idx, feature_df, feature_col, new_col):
    df = pd.merge(
        df,
        feature_df,
        how="left",
        left_on=["molecule_name", f"atom_index_{atom_idx}"],
        right_on=["molecule_name", "atom_index"],
    )
    df = df.drop("atom_index", axis=1)
    df = df.rename(columns={feature_col: new_col})
    return df


trainSet = map_atom_info(trainSet, 0)
trainSet = map_atom_info(trainSet, 1)
testSet = map_atom_info(testSet, 0)
testSet = map_atom_info(testSet, 1)

trainSet = add_atom_feature(
    trainSet, 0, mulliken_charges, "mulliken_charge", "charge_0"
)
trainSet = add_atom_feature(
    trainSet, 1, mulliken_charges, "mulliken_charge", "charge_1"
)
testSet = add_atom_feature(testSet, 0, mulliken_charges, "mulliken_charge", "charge_0")
testSet = add_atom_feature(testSet, 1, mulliken_charges, "mulliken_charge", "charge_1")

trainSet = add_atom_feature(
    trainSet, 0, magnetic_shielding, "shield_iso", "shield_iso_0"
)
trainSet = add_atom_feature(
    trainSet, 1, magnetic_shielding, "shield_iso", "shield_iso_1"
)
testSet = add_atom_feature(testSet, 0, magnetic_shielding, "shield_iso", "shield_iso_0")
testSet = add_atom_feature(testSet, 1, magnetic_shielding, "shield_iso", "shield_iso_1")

trainSet = pd.merge(trainSet, dipole_moments, how="left", on="molecule_name")
trainSet = pd.merge(trainSet, potential_energy, how="left", on="molecule_name")
testSet = pd.merge(testSet, dipole_moments, how="left", on="molecule_name")
testSet = pd.merge(testSet, potential_energy, how="left", on="molecule_name")




## === cell 5
for df in (trainSet, testSet):
    df["atom_0"] = df["atom_0"].fillna("X")
    df["atom_1"] = df["atom_1"].fillna("X")
    coord_cols = [c for c in df.columns if c.startswith(("x_", "y_", "z_"))]
    df[coord_cols] = df[coord_cols].fillna(0)

train_p0 = trainSet[["x_0", "y_0", "z_0"]].values
train_p1 = trainSet[["x_1", "y_1", "z_1"]].values
test_p0 = testSet[["x_0", "y_0", "z_0"]].values
test_p1 = testSet[["x_1", "y_1", "z_1"]].values

trainSet["dist"] = np.linalg.norm(train_p0 - train_p1, axis=1)
testSet["dist"] = np.linalg.norm(test_p0 - test_p1, axis=1)

trainSet["dist_to_type_mean"] = trainSet["dist"] / trainSet.groupby("type")[
    "dist"
].transform("mean")
testSet["dist_to_type_mean"] = testSet["dist"] / testSet.groupby("type")[
    "dist"
].transform("mean")

trainSet = trainSet.fillna(0)
testSet = testSet.fillna(0)

trainSet["dist_sq"] = trainSet["dist"] ** 2
testSet["dist_sq"] = testSet["dist"] ** 2

trainSet["charge_product"] = trainSet["charge_0"] * trainSet["charge_1"]
testSet["charge_product"] = testSet["charge_0"] * testSet["charge_1"]

trainSet["shield_product"] = trainSet["shield_iso_0"] * trainSet["shield_iso_1"]
testSet["shield_product"] = testSet["shield_iso_0"] * testSet["shield_iso_1"]

trainSet["log_dist"] = np.log(trainSet["dist"] + 1e-6)
testSet["log_dist"] = np.log(testSet["dist"] + 1e-6)




## === cell 6
train_atom0_dummies = pd.get_dummies(trainSet["atom_0"], prefix="atom_0")
train_atom1_dummies = pd.get_dummies(trainSet["atom_1"], prefix="atom_1")
trainSet = pd.concat([trainSet, train_atom0_dummies, train_atom1_dummies], axis=1)

test_atom0_dummies = pd.get_dummies(testSet["atom_0"], prefix="atom_0")
test_atom1_dummies = pd.get_dummies(testSet["atom_1"], prefix="atom_1")
test_atom0_dummies = test_atom0_dummies.reindex(
    columns=train_atom0_dummies.columns, fill_value=0
)
test_atom1_dummies = test_atom1_dummies.reindex(
    columns=train_atom1_dummies.columns, fill_value=0
)
testSet = pd.concat([testSet, test_atom0_dummies, test_atom1_dummies], axis=1)




## === cell 7
for typ in trainSet["type"].astype("category").cat.categories:
    col_name = f"type_{typ}"
    trainSet[col_name] = (trainSet["type"] == typ).astype(int)
    testSet[col_name] = (testSet["type"] == typ).astype(int)




## === cell 8
extra_numeric = [
    "dipole_X",
    "dipole_Y",
    "dipole_Z",
    "potential_energy",
    "charge_0",
    "charge_1",
    "shield_iso_0",
    "shield_iso_1",
    "charge_product",
    "shield_product",
    "log_dist",
]
feature_cols = [
    col
    for col in trainSet.columns
    if (
        col.startswith("type_")
        or col.startswith("atom_0_")
        or col.startswith("atom_1_")
        or col in ["dist", "dist_to_type_mean", "dist_sq"]
        or col in extra_numeric
    )
]

X_train = trainSet[feature_cols].astype(float)
y_train = trainSet["scalar_coupling_constant"]

type_models = {}
type_biases = {}
for typ in trainSet["type"].unique():
    idx = trainSet["type"] == typ
    mdl = LinearRegression()
    mdl.fit(X_train.loc[idx], y_train.loc[idx])
    train_pred = mdl.predict(X_train.loc[idx])
    bias = (y_train.loc[idx] - train_pred).mean()
    type_models[typ] = mdl
    type_biases[typ] = bias




## === cell 9
def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    maes = (y_true - y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()




## === cell 10
X_test = testSet[feature_cols].astype(float)

test_pred = np.zeros(len(testSet))
for typ in testSet["type"].unique():
    idx = testSet["type"] == typ
    pred = type_models[typ].predict(X_test.loc[idx])
    test_pred[idx] = pred + type_biases[typ]

resultSet = pd.DataFrame({"id": testSet["id"], "scalar_coupling_constant": test_pred})




## === cell 11
resultSet.to_csv("results.csv", index=False, header=True)
