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

-2.872362622891446

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the failing ensemble loading code with a simple, self‑contained baseline that computes the mean coupling constant for each coupling type from the provided training data and uses it to predict the test set. This removes missing‑file errors, guarantees all required columns exist, and creates a valid `ensemble_sub.csv` submission. The approach respects the original task and metric while keeping core logic minimal and reproducible.'
- What this solution (achieved 1.23566) has done: 'I keep the original simple mean‑by‑type baseline but add a cheap, physics‑inspired feature – the Euclidean distance between the two atoms in each pair – using the provided `structures.csv`. By grouping the training targets by both coupling `type` and a rounded distance bin, the model can capture systematic distance effects without changing the overall architecture. Predictions first use the type‑plus‑distance mean, fall back to the type mean, and finally to the global mean, which should lower the validation score toward the target while preserving the original workflow.'
- What this solution (achieved 1.23566) has done: 'I keep the overall workflow unchanged but add a simple type‑specific linear regression on the inter‑atomic distance.  
First, after computing the type‑ and distance‑based means, I fit a 1‑st degree model (scalar ≈ a·distance + b) for each coupling type and store the coefficients.  
During prediction I use this linear estimate (`prediction_lr`) and fall back to the distance‑bin mean, the type mean, and finally the global mean, exactly as before. This small calibration step should lower the log‑MAE toward the target without altering the core baseline logic.'
- What this solution (achieved 1.23566) has done: 'I fixed the `fillna` usage that was causing a `TypeError` by converting the indexed map results into a `Series` aligned with the test DataFrame, and also made the earlier pair‑element‑distance mapping assign a proper Series. These adjustments keep the original baseline logic intact while ensuring the pipeline runs to completion and produces a valid `ensemble_sub.csv` submission.'
- What this solution (achieved 1.23566) has done: 'I add a simple “pair‑element” fallback mean and blend it into the existing hierarchy – after the type‑specific fallback but before the global mean – so predictions gain an extra chemistry‑aware cue without altering the core baseline logic. This small calibration is expected to lower the log‑MAE toward the target while keeping the overall workflow unchanged.'
- What this solution (achieved 1.23566) has done: 'I reduce the distance‑bin granularity (round to 1 decimal instead of 2) to give each bin more samples, and I make the type‑specific linear‑regression estimate the primary prediction instead of a last‑resort fallback. The hierarchy now starts with the distance‑based linear model and then falls back to the more granular means, which should lower the MAE and move the log‑MAE toward the target score while preserving the original baseline logic.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from pathlib import Path


def find_file(filename):
    matches = list(Path(".").rglob(filename))
    if not matches:
        raise FileNotFoundError(f"{filename} not found in any subdirectory.")
    return matches[0]


train_path = find_file("train.csv")
test_path = find_file("test.csv")
struct_path = find_file("structures.csv")

structures = pd.read_csv(
    struct_path, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)




## === cell 1
struct0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)
train = train.merge(struct0, on=["molecule_name", "atom_index_0"], how="left")
test = test.merge(struct0, on=["molecule_name", "atom_index_0"], how="left")

struct1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)
train = train.merge(struct1, on=["molecule_name", "atom_index_1"], how="left")
test = test.merge(struct1, on=["molecule_name", "atom_index_1"], how="left")




## === cell 2
def compute_distance(df):
    return np.sqrt(
        (df["x0"] - df["x1"]) ** 2
        + (df["y0"] - df["y1"]) ** 2
        + (df["z0"] - df["z1"]) ** 2
    )


train["distance"] = compute_distance(train)
test["distance"] = compute_distance(test)

train["dist_bin"] = train["distance"].round(1)
test["dist_bin"] = test["distance"].round(1)


def make_pair_elem(row):
    a0, a1 = row["atom0"], row["atom1"]
    if pd.isna(a0) or pd.isna(a1):
        return "unknown"
    return "_".join(sorted([str(a0), str(a1)]))


train["pair_elem"] = train.apply(make_pair_elem, axis=1)
test["pair_elem"] = test.apply(make_pair_elem, axis=1)




## === cell 3
contrib_path = find_file("scalar_coupling_contributions.csv")
contrib = pd.read_csv(contrib_path)

contrib["contrib_sum"] = contrib["fc"] + contrib["sd"] + contrib["pso"] + contrib["dso"]

train = train.merge(
    contrib[["molecule_name", "atom_index_0", "atom_index_1", "type", "contrib_sum"]],
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
)
test = test.merge(
    contrib[["molecule_name", "atom_index_0", "atom_index_1", "type", "contrib_sum"]],
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
)

train["contrib_residual"] = train["scalar_coupling_constant"] - train["contrib_sum"]
type_resid_means = train.groupby("type")["contrib_residual"].mean()
test["contrib_corrected"] = test["contrib_sum"] + test["type"].map(
    type_resid_means
).fillna(0)




## === cell 4
global_mean = train["scalar_coupling_constant"].mean()
type_means = train.groupby("type")["scalar_coupling_constant"].mean()
type_dist_means = train.groupby(["type", "dist_bin"])["scalar_coupling_constant"].mean()
type_pair_dist_means = train.groupby(["type", "pair_elem", "dist_bin"])[
    "scalar_coupling_constant"
].mean()
pair_elem_means = train.groupby("pair_elem")["scalar_coupling_constant"].mean()

type_coeffs = {}
for t, grp in train.groupby("type"):
    if grp["distance"].nunique() > 1:
        a, b = np.polyfit(grp["distance"], grp["scalar_coupling_constant"], 1)
    else:
        a = 0.0
        b = grp["scalar_coupling_constant"].mean()
    type_coeffs[t] = (a, b)

coeffs_df = pd.DataFrame.from_dict(type_coeffs, orient="index", columns=["a", "b"])
coeffs_df.index.name = "type"
coeffs_df = coeffs_df.reset_index()

test = test.merge(coeffs_df, on="type", how="left")
test["prediction_lr"] = test["a"] * test["distance"] + test["b"]

test["prediction"] = test["prediction_lr"]

type_pair_dist_map = type_pair_dist_means.to_dict()
fallback_tp = pd.Series(
    test.set_index(["type", "pair_elem", "dist_bin"]).index.map(type_pair_dist_map),
    index=test.index,
)
test["prediction"] = test["prediction"].fillna(fallback_tp)

type_dist_map = type_dist_means.to_dict()
fallback_td = pd.Series(
    test.set_index(["type", "dist_bin"]).index.map(type_dist_map), index=test.index
)
test["prediction"] = test["prediction"].fillna(fallback_td)

pair_elem_map = pair_elem_means.to_dict()
fallback_pe = pd.Series(test["pair_elem"].map(pair_elem_map), index=test.index)
test["prediction"] = test["prediction"].fillna(fallback_pe)

test["prediction"] = test["prediction"].fillna(test["type"].map(type_means))
test["prediction"] = test["prediction"].fillna(global_mean)

test["prediction"] = 0.9 * test["contrib_corrected"] + 0.1 * test["prediction"]




## === cell 5
submission = pd.DataFrame(
    {"id": test["id"], "scalar_coupling_constant": test["prediction"]}
)
submission_path = Path("ensemble_sub.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path.resolve()} with {len(submission)} rows.")
