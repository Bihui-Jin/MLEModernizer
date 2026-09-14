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

-1.3547912745601447

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 3.00563) has done: 'The script originally tried to load several non‑existent past submissions, causing FileNotFound and NameError failures. I replaced those steps with a simple baseline: read the training data, compute the overall mean of *scalar_coupling_constant*, and assign that value to every row in the test set. The resulting predictions are saved to `submission.csv` with the required columns, ensuring a valid submission file is produced. An optional histogram of the generated predictions is also included.'
- What this solution (achieved 1.23566) has done: 'I replace the single global‑mean baseline with a per‑coupling‑type mean.  
The training set’s “type” column captures systematic differences, so mapping each test row to the mean constant of its type (falling back to the global mean when a type is missing) should lower the MAE‑based log score, moving it nearer the target −1.35479 while keeping the core logic unchanged.'
- What this solution (achieved 1.23566) has done: 'I load the atom element information from **structures.csv**, create a composite key that combines the coupling type with the two atom elements (sorted to keep symmetry), compute the mean scalar coupling constant for each key, and use that as the prediction. If a key is missing in the test set we fall back to the per‑type mean and then to the global mean, preserving the original baseline while adding a useful feature that should lower the log‑MAE toward the target.'
- What this solution (achieved 1.23566) has done: 'I keep the overall baseline logic but add a simple distance‑based refinement: after merging atom coordinates I compute the inter‑atomic distance, bucket it, and use the mean coupling for each (type, distance‑bucket) pair as a second‑level lookup. Predictions first use the detailed element‑type key, then fall back to the distance‑bucket mean, then to the type mean, and finally to the global mean. This adds useful geometric information while preserving the original approach and should lower the log‑MAE toward the target.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight charge‑difference feature to the existing hierarchical mean baseline: load the Mulliken charges, merge them for each atom, compute the charge‑difference bucket, and build a (type, elem_key, charge‑bucket) mean lookup that is consulted before the distance‑bucket fallback. This small extra lookup refines predictions without altering the overall model logic, and should move the log‑MAE toward the lower target score.'
- What this solution (achieved 1.23566) has done: 'I added a couple of low‑cost hierarchical look‑ups that keep the original mean‑based strategy but give it more specific information: a (type, elem_key) mean and a proper (type, dist_bucket) dictionary lookup. These extra fall‑backs are inserted before the broader type‑mean and global‑mean steps, which should lower the log‑MAE and move the score toward the target without altering the core baseline logic.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight bias correction step that learns the average residual per coupling type from the training data using the same hierarchical‑mean logic, then applies this per‑type adjustment to the test predictions. This preserves the original baseline while modestly shifting predictions toward the true values, helping lower the log‑MAE score toward the target.'
- What this solution (achieved 1.23566) has done: 'I add a small residual‑bias correction based on the detailed element‑type key. After computing the per‑type residual means (already present), I also compute the average residual for each `elem_key` in the training data and add this correction to the test predictions. This keeps the hierarchy unchanged while providing a modest, targeted adjustment that should lower the log‑MAE and move the score closer to the negative target. The rest of the pipeline, including data merging and CSV output, remains identical.'
- What this solution (achieved 1.23566) has done: 'The changes reorder the hierarchical lookup so that the distance‑bucket mean is consulted first (it captures geometry better than the element‑key alone) and add a residual correction per (type, dist_bucket). This small adjustment keeps the overall mean‑based strategy while giving a tighter fit, which should lower the log‑MAE and move the score toward the target.'
- What this solution (achieved 1.23566) has done: 'Implemented a robust path fix and renumbered cells so the notebook runs start‑to‑finish.  
* Cell 1 now sets `BASE_PATH` to the Kaggle input directory (`/kaggle/input/champs-scalar-coupling`) with a fallback to the original relative path.  
* The rest of the logic is unchanged – hierarchical mean look‑ups, residual corrections, and CSV output are retained.  
* Cells 3 and 4 write the submission file and optionally plot the prediction histogram.'
- What this solution (achieved 1.23566) has done: 'I added a data‑loading cell that safely reads the required CSV files (train, test, structures, Mulliken charges) from the Kaggle input directory or a local fallback.  
The original baseline logic is kept but wrapped in subsequent cells: we compute global and per‑type means, enrich the training data with element, distance and charge‑difference features, and build hierarchical lookup dictionaries.  
For the test set we recreate the same features, then fill predictions using a cascade of look‑ups (charge → distance → type‑elem → elem → type → global).  
Finally the script writes a proper `submission.csv` and plots a histogram of the predictions.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_PATH = "/kaggle/input/champs-scalar-coupling"
if not os.path.isdir(BASE_PATH):
    BASE_PATH = "./data/champs-scalar-coupling"


def load_csv(name):
    return pd.read_csv(os.path.join(BASE_PATH, name))


train_df = load_csv("train.csv")
test_df = load_csv("test.csv")
structures_df = load_csv("structures.csv")
charges_df = load_csv("mulliken_charges.csv")



## === cell 1
global_mean = train_df["scalar_coupling_constant"].mean()
type_means = train_df.groupby("type")["scalar_coupling_constant"].mean()


def make_elem_key(row):
    elems = sorted([row["elem0"], row["elem1"]])
    return f"{row['type']}_{elems[0]}_{elems[1]}"


train_tmp = (
    train_df.merge(
        structures_df,
        left_on=["molecule_name", "atom_index_0"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    .rename(columns={"atom": "elem0", "x": "x0", "y": "y0", "z": "z0"})
    .drop(columns=["atom_index"])
)

train_tmp = (
    train_tmp.merge(
        structures_df,
        left_on=["molecule_name", "atom_index_1"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    .rename(columns={"atom": "elem1", "x": "x1", "y": "y1", "z": "z1"})
    .drop(columns=["atom_index"])
)

train_tmp = (
    train_tmp.merge(
        charges_df,
        left_on=["molecule_name", "atom_index_0"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    .rename(columns={"mulliken_charge": "charge0"})
    .drop(columns=["atom_index"])
)

train_tmp = (
    train_tmp.merge(
        charges_df,
        left_on=["molecule_name", "atom_index_1"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    .rename(columns={"mulliken_charge": "charge1"})
    .drop(columns=["atom_index"])
)

train_tmp["elem_key"] = train_tmp.apply(make_elem_key, axis=1)

train_tmp["distance"] = np.sqrt(
    (train_tmp["x0"] - train_tmp["x1"]) ** 2
    + (train_tmp["y0"] - train_tmp["y1"]) ** 2
    + (train_tmp["z0"] - train_tmp["z1"]) ** 2
)
train_tmp["dist_bucket"] = train_tmp["distance"].round(2)

train_tmp["charge_diff"] = train_tmp["charge0"] - train_tmp["charge1"]
train_tmp["charge_bucket"] = train_tmp["charge_diff"].round(2)

elem_key_means = train_tmp.groupby("elem_key")["scalar_coupling_constant"].mean()
type_elem_key_means = train_tmp.groupby(["type", "elem_key"])[
    "scalar_coupling_constant"
].mean()
dist_means = train_tmp.groupby(["type", "dist_bucket"])[
    "scalar_coupling_constant"
].mean()
charge_means = train_tmp.groupby(["type", "elem_key", "charge_bucket"])[
    "scalar_coupling_constant"
].mean()

type_elem_key_dict = type_elem_key_means.to_dict()
dist_means_dict = dist_means.to_dict()
charge_means_dict = charge_means.to_dict()
elem_key_means_dict = elem_key_means.to_dict()



## === cell 2
test_tmp = (
    test_df.merge(
        structures_df,
        left_on=["molecule_name", "atom_index_0"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    .rename(columns={"atom": "elem0", "x": "x0", "y": "y0", "z": "z0"})
    .drop(columns=["atom_index"])
)

test_tmp = (
    test_tmp.merge(
        structures_df,
        left_on=["molecule_name", "atom_index_1"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    .rename(columns={"atom": "elem1", "x": "x1", "y": "y1", "z": "z1"})
    .drop(columns=["atom_index"])
)

test_tmp = (
    test_tmp.merge(
        charges_df,
        left_on=["molecule_name", "atom_index_0"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    .rename(columns={"mulliken_charge": "charge0"})
    .drop(columns=["atom_index"])
)

test_tmp = (
    test_tmp.merge(
        charges_df,
        left_on=["molecule_name", "atom_index_1"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    .rename(columns={"mulliken_charge": "charge1"})
    .drop(columns=["atom_index"])
)

test_tmp["elem_key"] = test_tmp.apply(make_elem_key, axis=1)

test_tmp["distance"] = np.sqrt(
    (test_tmp["x0"] - test_tmp["x1"]) ** 2
    + (test_tmp["y0"] - test_tmp["y1"]) ** 2
    + (test_tmp["z0"] - test_tmp["z1"]) ** 2
)
test_tmp["dist_bucket"] = test_tmp["distance"].round(2)

test_tmp["charge_diff"] = test_tmp["charge0"] - test_tmp["charge1"]
test_tmp["charge_bucket"] = test_tmp["charge_diff"].round(2)

preds = pd.Series(np.full(len(test_tmp), np.nan), index=test_tmp.index)

mask = preds.isna()
charge_keys = list(
    zip(
        test_tmp.loc[mask, "type"],
        test_tmp.loc[mask, "elem_key"],
        test_tmp.loc[mask, "charge_bucket"],
    )
)
preds.loc[mask] = [charge_means_dict.get(k, np.nan) for k in charge_keys]

mask = preds.isna()
dist_keys = list(zip(test_tmp.loc[mask, "type"], test_tmp.loc[mask, "dist_bucket"]))
preds.loc[mask] = [dist_means_dict.get(k, np.nan) for k in dist_keys]

mask = preds.isna()
type_elem_keys = list(zip(test_tmp.loc[mask, "type"], test_tmp.loc[mask, "elem_key"]))
preds.loc[mask] = [type_elem_key_dict.get(k, np.nan) for k in type_elem_keys]

mask = preds.isna()
elem_keys = test_tmp.loc[mask, "elem_key"]
preds.loc[mask] = elem_keys.map(elem_key_means_dict)

mask = preds.isna()
type_keys = test_tmp.loc[mask, "type"]
preds.loc[mask] = type_keys.map(type_means)

preds.fillna(global_mean, inplace=True)



## === cell 3
submission = test_df[["id"]].copy()
submission["scalar_coupling_constant"] = preds.values

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
print(f"Global mean: {global_mean:.5f}")
print(f"Distinct types: {type_means.shape[0]}")



## === cell 4
submission["scalar_coupling_constant"].plot(
    kind="hist", bins=100, title="Prediction Histogram"
)
plt.xlabel("scalar_coupling_constant")
plt.tight_layout()
plt.show()
