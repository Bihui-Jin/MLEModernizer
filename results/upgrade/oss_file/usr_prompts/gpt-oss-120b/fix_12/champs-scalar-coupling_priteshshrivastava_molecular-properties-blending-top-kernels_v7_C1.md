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

-1.65028845049191

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the missing blend‑file reads with a simple, fully reproducible baseline: compute the mean `scalar_coupling_constant` for each coupling `type` from the training data and use that mean as the prediction for every test row (fallbacking to the global mean when a type is unseen). This fixes the FileNotFoundError, guarantees a valid `id,scalar_coupling_constant` CSV output, and provides a reasonable score without altering any core model logic that isn’t present.'
- What this solution (achieved 1.23566) has done: 'I load the atom element information from structures.csv and enrich both the training and test rows with the element symbols of the two atoms. Then I compute a more granular mean target for each `(type, atom0, atom1)` combination and use it as the primary prediction, falling back to the per‑type mean and finally the global mean. This adds discriminative power while keeping the original mean‑per‑type baseline logic intact, moving the evaluation score toward the lower target.'
- What this solution (achieved 3.37023) has done: 'I add a lightweight numeric encoding of the categorical fields and a Euclidean‑distance feature derived from the atom coordinates, then train a simple LinearRegression model on these features. This keeps the overall pipeline unchanged while giving the model extra predictive signal, which is expected to lower the log‑MAE toward the target. The script now builds the distance feature for both train and test, encodes ‘type’, ‘atom0’, and ‘atom1’ as integer codes, fits the regression, and writes the predictions to the required CSV.'
- What this solution (achieved 1.23566) has done: 'I replace the linear‑regression step with the more reliable per‑type + (atom0, atom1) mean baseline that was shown to give a much lower score (≈1.23). This keeps the existing feature‑engineering (distance, atom symbols) but removes the over‑fitting regression, yielding predictions that move the log‑MAE closer to the target. The script now computes the group means, falls back to per‑type and global means, and writes the submission.'
- What this solution (achieved 2.60668) has done: 'I fix the NaN handling that caused label‑encoding to fail by filling missing atom symbols with a placeholder and fitting the encoders on both train and test atom values. I also fill missing distances before training the Ridge model. These minimal changes restore the pipeline, generate the predictions, and write a proper `id,scalar_coupling_constant` CSV submission.'
- What this solution (achieved 1.23566) has done: 'I lower the influence of the Ridge model, which was hurting performance, by using only the reliable mean‑based baseline (or heavily weighting it). This change keeps the overall pipeline intact but moves the log‑MAE closer to the target lower value.'
- What this solution (achieved 1.32442) has done: 'I added the missing Ridge predictions for the training set so that the blending step can access `train_enriched["pred_ridge"]`. This resolves the KeyError and allows the script to compute the best blend weight, produce predictions for the test set, and write a valid `id,scalar_coupling_constant` CSV submission.'
- What this solution (achieved 1.23566) has done: 'I add a light feature‑scaling step (StandardScaler) before fitting the Ridge model and expand the blend‑weight search grid to finer 0.05 increments. Scaling helps the linear model use the distance and encoded categorical features more effectively, and a finer weight grid lets the script choose a blend that more closely minimizes the training log‑MAE, moving the score toward the lower target. These changes keep the overall pipeline intact (baseline means, ridge model, blending) while modestly improving prediction quality.'
- What this solution (achieved 1.23566) has done: 'I slightly increase the ridge regularisation (alpha = 1.0) to reduce possible over‑fitting and search the blending weight on a finer 0.01 grid instead of 0.05. These minimal tweaks keep the overall pipeline unchanged while giving the model a better chance to lower the log‑MAE, moving the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

print("Available files:", os.listdir("../input/champs-scalar-coupling")[:5])



## === cell 1
train_path = "../input/champs-scalar-coupling/train.csv"
test_path = "../input/champs-scalar-coupling/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)



## === cell 2
structures_path = "../input/champs-scalar-coupling/structures.csv"
structures_df = pd.read_csv(
    structures_path, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
)

train_atoms0 = (
    train_df.merge(
        structures_df,
        left_on=["molecule_name", "atom_index_0"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    .rename(columns={"atom": "atom0", "x": "x0", "y": "y0", "z": "z0"})
    .drop(columns=["atom_index"])
)

train_atoms1 = (
    train_atoms0.merge(
        structures_df,
        left_on=["molecule_name", "atom_index_1"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    .rename(columns={"atom": "atom1", "x": "x1", "y": "y1", "z": "z1"})
    .drop(columns=["atom_index"])
)

train_atoms1["distance"] = np.sqrt(
    (train_atoms1["x0"] - train_atoms1["x1"]) ** 2
    + (train_atoms1["y0"] - train_atoms1["y1"]) ** 2
    + (train_atoms1["z0"] - train_atoms1["z1"]) ** 2
)

train_enriched = train_atoms1



## === cell 3
test_atoms0 = (
    test_df.merge(
        structures_df,
        left_on=["molecule_name", "atom_index_0"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    .rename(columns={"atom": "atom0", "x": "x0", "y": "y0", "z": "z0"})
    .drop(columns=["atom_index"])
)

test_atoms1 = (
    test_atoms0.merge(
        structures_df,
        left_on=["molecule_name", "atom_index_1"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    .rename(columns={"atom": "atom1", "x": "x1", "y": "y1", "z": "z1"})
    .drop(columns=["atom_index"])
)

test_atoms1["distance"] = np.sqrt(
    (test_atoms1["x0"] - test_atoms1["x1"]) ** 2
    + (test_atoms1["y0"] - test_atoms1["y1"]) ** 2
    + (test_atoms1["z0"] - test_atoms1["z1"]) ** 2
)

test_enriched = test_atoms1



## === cell 4
train_enriched["atom0"] = train_enriched["atom0"].fillna("UNK")
train_enriched["atom1"] = train_enriched["atom1"].fillna("UNK")
test_enriched["atom0"] = test_enriched["atom0"].fillna("UNK")
test_enriched["atom1"] = test_enriched["atom1"].fillna("UNK")

median_dist = train_enriched["distance"].median()
train_enriched["distance"] = train_enriched["distance"].fillna(median_dist)
test_enriched["distance"] = test_enriched["distance"].fillna(median_dist)

type_atom_means = (
    train_enriched.groupby(["type", "atom0", "atom1"])["scalar_coupling_constant"]
    .mean()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "pred_type_atom"})
)

type_means = (
    train_enriched.groupby("type")["scalar_coupling_constant"]
    .mean()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "pred_type"})
)

global_mean = train_enriched["scalar_coupling_constant"].mean()

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge

atom_le = LabelEncoder()
all_atoms = pd.concat(
    [
        train_enriched["atom0"],
        train_enriched["atom1"],
        test_enriched["atom0"],
        test_enriched["atom1"],
    ]
)
atom_le.fit(all_atoms)

train_enriched["atom0_enc"] = atom_le.transform(train_enriched["atom0"])
train_enriched["atom1_enc"] = atom_le.transform(train_enriched["atom1"])
test_enriched["atom0_enc"] = atom_le.transform(test_enriched["atom0"])
test_enriched["atom1_enc"] = atom_le.transform(test_enriched["atom1"])

type_le = LabelEncoder()
all_types = pd.concat([train_enriched["type"], test_enriched["type"]])
type_le.fit(all_types)

train_enriched["type_enc"] = type_le.transform(train_enriched["type"])
test_enriched["type_enc"] = type_le.transform(test_enriched["type"])

feat_cols = ["distance", "atom0_enc", "atom1_enc", "type_enc"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_enriched[feat_cols])
X_test_scaled = scaler.transform(test_enriched[feat_cols])

ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_scaled, train_enriched["scalar_coupling_constant"])
train_enriched["pred_ridge"] = ridge.predict(X_train_scaled)
test_enriched["pred_ridge"] = ridge.predict(X_test_scaled)

test_pred = test_enriched.merge(
    type_atom_means, on=["type", "atom0", "atom1"], how="left"
).merge(type_means, on="type", how="left")
test_pred["baseline"] = (
    test_pred["pred_type_atom"].fillna(test_pred["pred_type"]).fillna(global_mean)
)

train_pred = train_enriched.merge(
    type_atom_means, on=["type", "atom0", "atom1"], how="left"
).merge(type_means, on="type", how="left")
train_pred["baseline"] = (
    train_pred["pred_type_atom"].fillna(train_pred["pred_type"]).fillna(global_mean)
)

weights = np.arange(0.0, 1.001, 0.01)
best_w = 1.0
best_score = np.inf

for w in weights:
    blended = w * train_pred["baseline"] + (1 - w) * train_enriched["pred_ridge"]
    mae_per_type = (
        pd.Series(np.abs(blended - train_enriched["scalar_coupling_constant"]))
        .groupby(train_enriched["type"])
        .mean()
    )
    log_mae = np.log(mae_per_type + 1e-9)
    score = log_mae.mean()
    if score < best_score:
        best_score = score
        best_w = w

test_pred["scalar_coupling_constant"] = (
    best_w * test_pred["baseline"] + (1 - best_w) * test_enriched["pred_ridge"]
)
test_pred_vals = test_pred["scalar_coupling_constant"].values



## === cell 5
submission = pd.DataFrame(
    {"id": test_enriched["id"], "scalar_coupling_constant": test_pred_vals}
)

submission_path = "my_enhanced_baseline_submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
