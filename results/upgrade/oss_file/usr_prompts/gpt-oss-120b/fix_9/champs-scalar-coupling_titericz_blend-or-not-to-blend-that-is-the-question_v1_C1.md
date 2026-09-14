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

1.23528

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the failing blend logic with a simple, self‑contained baseline that reads the competition data, computes the average scalar coupling constant for each coupling type in the training set, applies those averages to the test set (using a global mean for any missing types), and saves a correctly formatted `submission.csv`. This removes the missing‑file errors, guarantees a valid CSV output, and provides a reasonable low‑score baseline without altering any core modeling logic.'
- What this solution (achieved 1.23566) has done: 'I enhance the baseline by incorporating atom‑type information from the structures file.  
The script now loads `structures.csv`, extracts the element symbols for each atom in a coupling pair, and computes mean coupling constants grouped by `(type, atom_0, atom_1)`.  
During prediction the hierarchy of fall‑backs is: exact `(type, atom_0, atom_1)` mean → type‑only mean → global mean. This adds useful chemistry‑aware detail with minimal changes and is expected to lower the log‑MAE toward the target score.'
- What this solution (achieved 1.23566) has done: 'I fix the NaN handling when building the atom‑pair identifier for the test set, which caused a TypeError, and ensure the `submission` DataFrame is always created before it is written. This involves filling missing element symbols with a placeholder before creating the `pair` column and keeping the existing logic unchanged. The updated script now runs end‑to‑end and writes a valid `submission.csv` file.'
- What this solution (achieved 1.23566) has done: 'I load the molecule‑level numeric data (potential energy and dipole magnitude), merge them into the training and test frames, compute the same hierarchy‑based baseline predictions, then fit a tiny linear correction on the train residuals using these two features. The learned intercept and slopes are applied to the test baseline, giving a modest adjusted prediction that should lower the log‑MAE toward the target without changing the core averaging logic.'
- What this solution (achieved 1.23528) has done: 'I keep the overall baseline‑plus‑linear‑correction structure but replace the single global linear adjustment with a small per‑type linear model (intercept + potential + dipole). This adds just a few lines that fit separate coefficients for each coupling type on the training residuals and then apply the matching coefficients to the test set, falling back to the original global coefficients when a type has too few samples. The change retains all original features and merges, while giving the model more flexibility to reduce the log‑MAE and move the score closer to the target.'
- What this solution (achieved 1.23528) has done: 'I added a simple fix so the training DataFrame gets a `final_pred` column before the bias‑correction step. By setting `train_pred["final_pred"] = train_pred["baseline"]` (the baseline prediction), the subsequent error calculation works without altering the model logic. The rest of the pipeline remains unchanged, ensuring a valid `submission.csv` is written.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

BASE_PATH = "../input/champs-scalar-coupling"

train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")
structures_path = os.path.join(BASE_PATH, "structures.csv")
potential_path = os.path.join(BASE_PATH, "potential_energy.csv")
dipole_path = os.path.join(BASE_PATH, "dipole_moments.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)
structures_df = pd.read_csv(structures_path)
potential_df = pd.read_csv(potential_path)
dipole_df = pd.read_csv(dipole_path)

dipole_df["dipole_mag"] = np.sqrt(
    dipole_df["X"] ** 2 + dipole_df["Y"] ** 2 + dipole_df["Z"] ** 2
)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print("Structures shape:", structures_df.shape)
print("Potential shape:", potential_df.shape)
print("Dipole shape:", dipole_df.shape)




## === cell 1
structures_df = structures_df.rename(columns={"atom": "element"})

atom0 = structures_df.rename(
    columns={"atom_index": "atom_index_0", "element": "element_0"}
)[["molecule_name", "atom_index_0", "element_0"]]

atom1 = structures_df.rename(
    columns={"atom_index": "atom_index_1", "element": "element_1"}
)[["molecule_name", "atom_index_1", "element_1"]]

train_df = train_df.merge(atom0, on=["molecule_name", "atom_index_0"], how="left")
train_df = train_df.merge(atom1, on=["molecule_name", "atom_index_1"], how="left")
test_df = test_df.merge(atom0, on=["molecule_name", "atom_index_0"], how="left")
test_df = test_df.merge(atom1, on=["molecule_name", "atom_index_1"], how="left")

train_df = train_df.merge(potential_df, on="molecule_name", how="left")
train_df = train_df.merge(
    dipole_df[["molecule_name", "dipole_mag"]], on="molecule_name", how="left"
)
test_df = test_df.merge(potential_df, on="molecule_name", how="left")
test_df = test_df.merge(
    dipole_df[["molecule_name", "dipole_mag"]], on="molecule_name", how="left"
)




## === cell 2
global_mean = train_df["scalar_coupling_constant"].mean()
type_means = (
    train_df.groupby("type")["scalar_coupling_constant"]
    .mean()
    .rename("type_mean")
    .reset_index()
)

type_atom_means = (
    train_df.groupby(["type", "element_0", "element_1"])["scalar_coupling_constant"]
    .mean()
    .rename("type_atom_mean")
    .reset_index()
)

train_df["pair"] = train_df.apply(
    lambda r: "_".join(sorted([r["element_0"], r["element_1"]])), axis=1
)
type_pair_means = (
    train_df.groupby(["type", "pair"])["scalar_coupling_constant"]
    .mean()
    .rename("type_pair_mean")
    .reset_index()
)

print("Global mean:", global_mean)

test_df["element_0"] = test_df["element_0"].fillna("X").astype(str)
test_df["element_1"] = test_df["element_1"].fillna("X").astype(str)

test_df["pair"] = test_df.apply(
    lambda r: "_".join(sorted([r["element_0"], r["element_1"]])), axis=1
)


def add_baseline(df):
    df = df.merge(type_atom_means, on=["type", "element_0", "element_1"], how="left")
    df = df.merge(type_pair_means, on=["type", "pair"], how="left")
    df = df.merge(type_means, on="type", how="left")
    df["baseline"] = (
        df["type_atom_mean"]
        .fillna(df["type_pair_mean"])
        .fillna(df["type_mean"])
        .fillna(global_mean)
    )
    return df


train_pred = add_baseline(train_df.copy())
test_pred = add_baseline(test_df.copy())

pot_mean = train_pred["potential_energy"].mean()
dip_mean = train_pred["dipole_mag"].mean()

train_feat = pd.DataFrame(
    {
        "pot": train_pred["potential_energy"].fillna(pot_mean),
        "dip": train_pred["dipole_mag"].fillna(dip_mean),
    }
)
train_residual = train_pred["scalar_coupling_constant"] - train_pred["baseline"]

X_global = np.vstack(
    [np.ones(len(train_feat)), train_feat["pot"].values, train_feat["dip"].values]
).T
y_global = train_residual.values
coef_global = np.linalg.lstsq(X_global, y_global, rcond=None)[
    0
]  # [intercept, w_pot, w_dip]

type_coefs = []
for t, grp in train_pred.groupby("type"):
    if len(grp) < 2:
        continue  # not enough data, will use global coefficients
    X_t = np.vstack(
        [
            np.ones(len(grp)),
            grp["potential_energy"].fillna(pot_mean).values,
            grp["dipole_mag"].fillna(dip_mean).values,
        ]
    ).T
    y_t = (grp["scalar_coupling_constant"] - grp["baseline"]).values
    c_t = np.linalg.lstsq(X_t, y_t, rcond=None)[0]
    type_coefs.append({"type": t, "c0": c_t[0], "c1": c_t[1], "c2": c_t[2]})

type_coefs_df = pd.DataFrame(type_coefs)

test_pred = test_pred.merge(type_coefs_df, on="type", how="left")

test_pred["c0"] = test_pred["c0"].fillna(coef_global[0])
test_pred["c1"] = test_pred["c1"].fillna(coef_global[1])
test_pred["c2"] = test_pred["c2"].fillna(coef_global[2])

test_pred["adjustment"] = (
    test_pred["c0"]
    + test_pred["c1"] * test_pred["potential_energy"].fillna(pot_mean)
    + test_pred["c2"] * test_pred["dipole_mag"].fillna(dip_mean)
)

test_pred["final_pred"] = test_pred["baseline"] + test_pred["adjustment"]

train_pred["final_pred"] = train_pred["baseline"]

train_pred["error"] = train_pred["scalar_coupling_constant"] - train_pred["final_pred"]
type_bias = (
    train_pred.groupby("type")["error"]
    .mean()
    .reset_index()
    .rename(columns={"error": "bias"})
)
global_bias = train_pred["error"].mean()

test_pred = test_pred.merge(type_bias, on="type", how="left")
test_pred["bias"] = test_pred["bias"].fillna(global_bias)

test_pred["final_pred_adj"] = test_pred["final_pred"] + test_pred["bias"]
submission = pd.DataFrame(
    {
        "id": test_pred["id"],
        "scalar_coupling_constant": test_pred["final_pred_adj"],
    }
)

submission = submission.set_index("id").reindex(sample_sub["id"]).reset_index()
submission["scalar_coupling_constant"] = submission["scalar_coupling_constant"].astype(
    np.float32
)




## === cell 3
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
print(submission.head())
