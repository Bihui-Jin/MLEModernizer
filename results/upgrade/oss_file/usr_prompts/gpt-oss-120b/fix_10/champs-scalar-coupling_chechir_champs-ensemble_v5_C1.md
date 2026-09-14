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

-2.058625093818288

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'The fix replaces the missing external prediction files with a self‑contained baseline that reads the official training and test data, computes the average scalar coupling constant for each coupling type, and uses those averages as predictions. This eliminates the FileNotFoundError, ensures a valid `submission.csv` with the correct columns is written, and provides a reasonable score without altering any core model logic beyond the simple baseline.'
- What this solution (achieved 1.23566) has done: 'I load the atom element information from `structures.csv`, create a combined key of coupling type + sorted atom‑pair elements, compute the mean target for each key, and use those means for predictions (falling back to the type mean and finally the global mean). This adds only a lightweight feature that is known to improve the baseline while keeping the original logic intact, moving the score closer to the lower target.'
- What this solution (achieved 1.23566) has done: 'The changes add a simple distance feature to the existing mean‑based baseline.  
First, the Euclidean distance between the two atoms of each pair is computed and binned (0.5 Å steps).  
A mean scalar‑coupling constant is then calculated for each *(type, element‑pair, distance‑bin)* combination.  
During prediction we first try to use this more specific mean; if it is missing we fall back to the original *(type, element‑pair)* mean, then to the per‑type mean, and finally to the global mean.  
These extra, still‑lightweight statistics give a finer calibration and are expected to lower the MAE (hence the log‑MAE) toward the target without altering the core logic.'
- What this solution (achieved 1.23566) has done: 'I keep the overall baseline logic but make the predictions a small weighted blend of the most specific mean available and the next‑most‑general mean. This modest smoothing usually lowers MAE (hence the log‑MAE) without altering the core feature engineering, moving the score closer to the negative target.'
- What this solution (achieved 1.23566) has done: 'I simplify the prediction logic to use the most specific available mean without blending, which generally reduces error by avoiding unnecessary smoothing. The function now returns the specific (type, elem_pair, dist_bin) mean when present, falls back to the (type, elem_pair) mean, then to the type mean, and finally to the global mean. This small change keeps the overall pipeline intact while moving the score closer to the lower target.'
- What this solution (achieved 1.68334) has done: 'I adjust the prediction function to use a light weighted blend of the most specific mean (type‑elem‑dist) and the next‑most‑specific mean (type‑elem) when both are available, and also blend the type‑level mean with the global mean. This tiny smoothing often reduces over‑fitting of very specific groups, lowering the log‑MAE and moving the score nearer the target without changing the overall baseline logic.'
- What this solution (achieved 1.23566) has done: 'I simplify the prediction logic to use the most specific available mean without blending. This removes the extra smoothing that was inflating the error, so the model fall back step‑by‑step from the (type, element‑pair, distance) mean to the (type, element‑pair) mean, then the type mean, and finally the global mean, which should lower the log‑MAE toward the target.'
- What this solution (achieved 1.93235) has done: 'Implemented finer distance bins (0.25 Å steps) to capture more detailed spatial patterns and introduced a lightweight blending scheme that combines the most specific available mean with the next‑most‑general mean (e.g., type‑elem‑dist + type‑elem, then elem + type, then type + global). This adds just enough smoothing to reduce over‑fitting while preserving the original mean‑based logic, moving the log‑MAE toward the lower target score. The script now writes a proper `submission.csv` with the required columns.'
- What this solution (achieved 1.23566) has done: 'I simplify the prediction logic to use the most specific available mean without any blending, falling back step‑by‑step from the (type, element‑pair, distance‑bin) mean to the (type, element‑pair) mean, then to the per‑type mean, and finally to the global mean. This reduces over‑smoothing and is expected to lower the log‑MAE, moving the score closer to the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import os
import numpy as np

BASE_INPUT = "../input/champs-scalar-coupling"

train_path = os.path.join(BASE_INPUT, "train.csv")
test_path = os.path.join(BASE_INPUT, "test.csv")
structures_path = os.path.join(BASE_INPUT, "structures.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
structures_df = pd.read_csv(structures_path)


def make_elem_pair(row):
    elems = sorted([row["atom_0"], row["atom_1"]])
    return f"{elems[0]}_{elems[1]}"


train_merged = train_df.merge(
    structures_df[["molecule_name", "atom_index", "atom", "x", "y", "z"]].rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom_0",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
    ),
    on=["molecule_name", "atom_index_0"],
    how="left",
).merge(
    structures_df[["molecule_name", "atom_index", "atom", "x", "y", "z"]].rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom_1",
            "x": "x1",
            "y": "y1",
            "z": "z1",
        }
    ),
    on=["molecule_name", "atom_index_1"],
    how="left",
)

train_merged["elem_pair"] = train_merged.apply(make_elem_pair, axis=1)

train_merged["distance"] = np.sqrt(
    (train_merged["x0"] - train_merged["x1"]) ** 2
    + (train_merged["y0"] - train_merged["y1"]) ** 2
    + (train_merged["z0"] - train_merged["z1"]) ** 2
)

bins = np.arange(0, 5.5, 0.25)
labels = [f"{b:.2f}" for b in bins[:-1]]
train_merged["dist_bin"] = pd.cut(
    train_merged["distance"],
    bins=np.append(bins, np.inf),
    labels=labels + ["5+"],
    right=False,
)

type_elem_dist_means = (
    train_merged.groupby(["type", "elem_pair", "dist_bin"])["scalar_coupling_constant"]
    .mean()
    .reset_index()
)

type_elem_means = (
    train_merged.groupby(["type", "elem_pair"])["scalar_coupling_constant"]
    .mean()
    .reset_index()
)

type_means = train_df.groupby("type")["scalar_coupling_constant"].mean()
global_mean = train_df["scalar_coupling_constant"].mean()


test_merged = test_df.merge(
    structures_df[["molecule_name", "atom_index", "atom", "x", "y", "z"]].rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom_0",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
    ),
    on=["molecule_name", "atom_index_0"],
    how="left",
).merge(
    structures_df[["molecule_name", "atom_index", "atom", "x", "y", "z"]].rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom_1",
            "x": "x1",
            "y": "y1",
            "z": "z1",
        }
    ),
    on=["molecule_name", "atom_index_1"],
    how="left",
)

test_merged["elem_pair"] = test_merged.apply(make_elem_pair, axis=1)

test_merged["distance"] = np.sqrt(
    (test_merged["x0"] - test_merged["x1"]) ** 2
    + (test_merged["y0"] - test_merged["y1"]) ** 2
    + (test_merged["z0"] - test_merged["z1"]) ** 2
)

test_merged["dist_bin"] = pd.cut(
    test_merged["distance"],
    bins=np.append(bins, np.inf),
    labels=labels + ["5+"],
    right=False,
)

type_elem_dist_series = pd.Series(
    data=type_elem_dist_means["scalar_coupling_constant"].values,
    index=pd.MultiIndex.from_arrays(
        [
            type_elem_dist_means["type"],
            type_elem_dist_means["elem_pair"],
            type_elem_dist_means["dist_bin"],
        ],
        names=["type", "elem_pair", "dist_bin"],
    ),
)

type_elem_series = pd.Series(
    data=type_elem_means["scalar_coupling_constant"].values,
    index=pd.MultiIndex.from_arrays(
        [type_elem_means["type"], type_elem_means["elem_pair"]],
        names=["type", "elem_pair"],
    ),
)


def predict_row(row):
    """Return the most specific available mean without blending."""
    specific = type_elem_dist_series.get(
        (row["type"], row["elem_pair"], row["dist_bin"])
    )
    elem_mean = type_elem_series.get((row["type"], row["elem_pair"]))
    type_mean = type_means.get(row["type"])

    if pd.notna(specific):
        return specific
    if pd.notna(elem_mean):
        return elem_mean
    if pd.notna(type_mean):
        return type_mean
    return global_mean


test_predictions = test_merged.apply(predict_row, axis=1)




## === cell 1
submission = pd.DataFrame(
    {"id": test_df["id"], "scalar_coupling_constant": test_predictions}
)

output_path = "submission.csv"
submission.to_csv(output_path, index=False)

print(f"Submission written to {output_path}")
