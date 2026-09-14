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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

-2.4128442927016835

# 6. Current score

2.46705

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'Implemented a safe fallback pipeline that avoids missing external prediction files. The script now reads the official training and test data from the Kaggle input directory, computes a simple per‑type mean baseline for `scalar_coupling_constant`, applies it to the test set, and writes a valid `ensemble_sub.csv` submission. All previous median‑based stacking logic that relied on unavailable files has been removed, ensuring the notebook runs end‑to‑end and produces a correctly formatted CSV.'
- What this solution (achieved 1.23566) has done: 'We enrich the baseline by adding the element types of the two atoms in each pair (using structures.csv) and compute a mean target per *(type, atom_0, atom_1)* group. Predictions first use this detailed mean, then fall back to the per‑type mean, and finally to the global mean. This small feature addition keeps the original logic while giving a more specific estimate, which should lower the log‑MAE toward the target. The script now loads the structures file, merges atom symbols into train and test, builds the hierarchical means, and writes the submission as before.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight geometric feature – the inter‑atom distance – to the hierarchical mean calculation. By merging the x,y,z coordinates for each atom pair, computing a rounded distance bin, and grouping on `(type, atom_0, atom_1, dist_bin)` we obtain more specific “detailed” means while keeping the same overall baseline logic. Missing values still fall back to the per‑type and global means, so the pipeline remains stable and produces a valid CSV, but the added specificity should lower the log‑MAE and move the score toward the target.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight bias correction: after the hierarchical mean predictions are built, I compute the same predictions on the training set, calculate the average residual (actual – predicted), and then shift all test predictions by this global residual. This small adjustment keeps the original mean‑based logic while correcting systematic under‑ or over‑estimation, which should lower the log‑MAE and move the score toward the target.'
- What this solution (achieved 1.23566) has done: 'I keep the overall mean‑based hierarchy but add a per‑type bias correction (instead of only a global residual) which is a tiny, targeted tweak that usually lowers the log‑MAE without altering the core model logic. The code now computes a residual mean for each coupling type on the training set and adds the appropriate correction to the test predictions, falling back to the overall residual if a type‑specific value is missing.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight “pair‑only” mean level between the detailed distance‑bin means and the per‑type means. This keeps the original hierarchical‑mean logic while giving the model another chance to obtain a non‑missing estimate, which should lower the log‑MAE and move the score closer to the target. The new pair‑only means are merged after the detailed means and before falling back to the type‑mean, preserving all existing bias‑correction steps.'
- What this solution (achieved 2.28618) has done: 'Implemented targeted speed‑ups while keeping the overall modeling approach unchanged.  
Key changes:  
* Convert feature frames to NumPy float32 arrays before fitting to avoid pandas overhead.  
* Drop unused columns early to reduce memory pressure.  
* Switch to `HistGradientBoostingRegressor`, which implements the same gradient‑boosting logic but is far more efficient on millions of rows, preserving hyper‑parameters and deterministic behavior.'
- What this solution (achieved 2.2865) has done: 'I add a lightweight calibration step after the model’s predictions: fit a simple linear regression on the training predictions versus the true targets and apply this correction to the test predictions. This tiny post‑processing tweak keeps the original model unchanged while shifting the predictions toward the true distribution, which should reduce the log‑MAE and move the score closer to the target.'
- What this solution (achieved 2.42709) has done: 'I add a per‑type mean target feature and a per‑type residual correction, both of which are tiny post‑processing tweaks that keep the original HistGradientBoostingRegressor pipeline intact while moving the log‑MAE closer to the target (lower is better). The new “type_mean” column is merged into the training and test sets and included in the model features, then after the model prediction we compute the average residual for each coupling type on the training data and shift the test predictions accordingly. These changes are minimal, preserve the core logic, and are aimed at reducing the evaluation score toward the requested target.'
- What this solution (achieved 2.46705) has done: 'I fixed the NaN handling that broke the distance‑bin creation, ensured the encoding of unseen coupling types does not raise errors, and kept the rest of the pipeline unchanged so it still produces predictions and writes a proper CSV submission.'
- What this solution (achieved 2.46705) has done: 'I add a lightweight hierarchical mean feature `type_atom_pair_mean` that captures the average coupling for each (type, atom 0, atom 1) group and include it in the model’s feature set. This small addition preserves the original pipeline while giving the regressor a more specific signal, which should lower the log‑MAE and move the score closer to the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import HistGradientBoostingRegressor

INPUT_DIR = "/kaggle/input/champs-scalar-coupling"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
STRUCTURES_PATH = os.path.join(INPUT_DIR, "structures.csv")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

required_cols = {"id", "type", "scalar_coupling_constant"}
assert required_cols.issubset(train.columns), "Train file missing required columns"
assert {"id", "type"}.issubset(test.columns), "Test file missing required columns"

structures = pd.read_csv(
    STRUCTURES_PATH, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
)


def merge_atom(df, suffix):
    return df.merge(
        structures.rename(
            columns={
                "atom_index": f"atom_index_{suffix}",
                "atom": f"atom_{suffix}",
                "x": f"x{suffix}",
                "y": f"y{suffix}",
                "z": f"z{suffix}",
            }
        ),
        on=["molecule_name", f"atom_index_{suffix}"],
        how="left",
    )


train = merge_atom(train, "0")
train = merge_atom(train, "1")
test = merge_atom(test, "0")
test = merge_atom(test, "1")

for df in (train, test):
    df["dist"] = np.sqrt(
        (df["x0"] - df["x1"]) ** 2
        + (df["y0"] - df["y1"]) ** 2
        + (df["z0"] - df["z1"]) ** 2
    )

_, bin_edges = pd.cut(train["dist"], bins=10, retbins=True)

train_dist_bin = pd.cut(
    train["dist"], bins=bin_edges, labels=False, include_lowest=True
)
train["dist_bin"] = train_dist_bin.fillna(-1).astype(int)

test_dist_bin = pd.cut(test["dist"], bins=bin_edges, labels=False, include_lowest=True)
test["dist_bin"] = test_dist_bin.fillna(-1).astype(int)

element_to_num = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
}
for col in ["atom_0", "atom_1"]:
    train[f"{col}_num"] = train[col].map(element_to_num).fillna(0).astype(int)
    test[f"{col}_num"] = test[col].map(element_to_num).fillna(0).astype(int)

train["type_enc"], type_uniques = pd.factorize(train["type"])
test_type_idx = type_uniques.get_indexer(test["type"])
test["type_enc"] = np.where(test_type_idx == -1, -1, test_type_idx).astype(int)

type_mean_target = train.groupby("type_enc")["scalar_coupling_constant"].mean()
train["type_mean"] = train["type_enc"].map(type_mean_target)
global_mean = train["scalar_coupling_constant"].mean()
test["type_mean"] = test["type_enc"].map(type_mean_target).fillna(global_mean)

pair_mean_map = train.groupby(["type_enc", "atom_0", "atom_1"])[
    "scalar_coupling_constant"
].mean()
train["type_atom_pair_mean"] = (
    train.set_index(["type_enc", "atom_0", "atom_1"])
    .index.map(pair_mean_map)
    .fillna(global_mean)
)
test["type_atom_pair_mean"] = (
    test.set_index(["type_enc", "atom_0", "atom_1"])
    .index.map(pair_mean_map)
    .fillna(global_mean)
)

feature_cols = [
    "type_enc",
    "atom_0_num",
    "atom_1_num",
    "dist",
    "dist_bin",
    "type_mean",
    "type_atom_pair_mean",  # added feature
]

X_train = train[feature_cols].fillna(-1).astype(np.float32).values
y_train = train["scalar_coupling_constant"].astype(np.float32).values
X_test = test[feature_cols].fillna(-1).astype(np.float32).values

gbr = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    verbose=0,
)

gbr.fit(X_train, y_train)

train_pred = gbr.predict(X_train)
test_pred = gbr.predict(X_test)

train_residual = y_train - train_pred
type_residual = (
    pd.DataFrame({"type_enc": train["type_enc"], "resid": train_residual})
    .groupby("type_enc")["resid"]
    .mean()
)
test["type_resid"] = test["type_enc"].map(type_residual).fillna(0.0)

global_residual = train_residual.mean()
test["global_resid"] = global_residual

test["final_preds"] = test_pred + test["type_resid"] + test["global_resid"]



## === cell 1
submission = pd.DataFrame(
    {"id": test["id"], "scalar_coupling_constant": test["final_preds"]}
)

submission_path = "ensemble_sub.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission saved to {submission_path}")
print(submission.head())
