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

-1.356722768415636

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'The fix replaces the missing‑folder logic with a straightforward baseline: it loads the competition data from the correct input directory, computes the average `scalar_coupling_constant` for each coupling `type` in the training set, applies these averages to the test rows (fall‑back to the global mean if a type is unseen), and writes a valid `submission.csv` with the required columns. All unnecessary cells are removed and the path references are corrected so the script runs end‑to‑end without errors.'
- What this solution (achieved 1.23566) has done: 'I enhance the baseline by incorporating atom element information from the structures file. By merging atom types for each coupling pair and computing means per (type, atom0, atom1), predictions become more specific. The hierarchy falls back to per‑type mean then the global mean, keeping the core averaging logic while improving the score toward the target.'
- What this solution (achieved 1.23566) has done: 'I merge the scalar coupling contributions data, compute a per‑type scaling factor that relates the sum of the four contribution terms to the target, and use this calibrated prediction as the primary estimate (falling back to the original type‑atom‑pair means and then the global mean). This adds only a lightweight calibration step, keeping the original averaging logic while moving predictions closer to the true values and thus lowering the log‑MAE toward the target.'
- What this solution (achieved 1.23566) has done: 'I replace the simple mean‑ratio scaling with a linear‑fit scaling (slope = Σ(y·x)/Σ(x²)) for each coupling type and add a global fallback slope. This keeps the same overall pipeline but usually gives a more accurate calibration of the contribution‑sum, which should lower the log‑MAE and move the score closer to the target.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight linear‑fit calibration instead of the simple origin‑only scaling.  
In **cell 0** I import NumPy.  
In **cell 3** I replace the “type_scaling” computation with per‑type slope & intercept using a least‑squares fit (and a global fallback) and build `calib_pred = slope*contrib_sum + intercept`.  
All other logic (hierarchical means, fallback, CSV output) stays unchanged, so the core pipeline is preserved while giving a more accurate calibration that should lower the log‑MAE toward the negative target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

BASE_PATH = "../input/champs-scalar-coupling"

required_files = [
    "train.csv",
    "test.csv",
    "sample_submission.csv",
    "structures.csv",
    "scalar_coupling_contributions.csv",
]
for f in required_files:
    fp = os.path.join(BASE_PATH, f)
    if not os.path.isfile(fp):
        raise FileNotFoundError(f"Expected file not found: {fp}")



## === cell 1
train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

assert {
    "id",
    "type",
    "scalar_coupling_constant",
    "molecule_name",
    "atom_index_0",
    "atom_index_1",
}.issubset(train.columns)
assert {"id", "type", "molecule_name", "atom_index_0", "atom_index_1"}.issubset(
    test.columns
)



## === cell 2
structures_path = os.path.join(BASE_PATH, "structures.csv")
structures = pd.read_csv(structures_path)

atom_info = structures[["molecule_name", "atom_index", "atom"]].rename(
    columns={"atom": "element"}
)

train = train.merge(
    atom_info.rename(columns={"atom_index": "atom_index_0", "element": "atom_0"}),
    on=["molecule_name", "atom_index_0"],
    how="left",
)
test = test.merge(
    atom_info.rename(columns={"atom_index": "atom_index_0", "element": "atom_0"}),
    on=["molecule_name", "atom_index_0"],
    how="left",
)

train = train.merge(
    atom_info.rename(columns={"atom_index": "atom_index_1", "element": "atom_1"}),
    on=["molecule_name", "atom_index_1"],
    how="left",
)
test = test.merge(
    atom_info.rename(columns={"atom_index": "atom_index_1", "element": "atom_1"}),
    on=["molecule_name", "atom_index_1"],
    how="left",
)

type_means = train.groupby("type")["scalar_coupling_constant"].mean()
global_mean = train["scalar_coupling_constant"].mean()
group_means = train.groupby(["type", "atom_0", "atom_1"])[
    "scalar_coupling_constant"
].mean()



## === cell 3
contrib_path = os.path.join(BASE_PATH, "scalar_coupling_contributions.csv")
contrib = pd.read_csv(contrib_path)

merge_cols = ["molecule_name", "atom_index_0", "atom_index_1", "type"]
train = train.merge(contrib, on=merge_cols, how="left")
test = test.merge(contrib, on=merge_cols, how="left")

for df in (train, test):
    df["contrib_sum"] = df[["fc", "sd", "pso", "dso"]].sum(axis=1)

valid = train["contrib_sum"] != 0


def fit_line(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, intercept


type_params = (
    train.loc[valid]
    .groupby("type")
    .apply(
        lambda g: pd.Series(
            fit_line(g["contrib_sum"].values, g["scalar_coupling_constant"].values),
            index=["slope", "intercept"],
        )
    )
)

global_slope, global_intercept = fit_line(
    train.loc[valid, "contrib_sum"].values,
    train.loc[valid, "scalar_coupling_constant"].values,
)

type_params["slope"] = type_params["slope"].fillna(global_slope)
type_params["intercept"] = type_params["intercept"].fillna(global_intercept)

calib_pred = test["contrib_sum"] * test["type"].map(type_params["slope"]) + test[
    "type"
].map(type_params["intercept"])



## === cell 4
test_key = list(zip(test["type"], test["atom_0"], test["atom_1"]))
hier_pred = pd.Series(test_key).map(group_means)
hier_pred = hier_pred.fillna(test["type"].map(type_means))
hier_pred = hier_pred.fillna(global_mean)

final_pred = calib_pred.where(test["contrib_sum"] != 0, hier_pred)
final_pred = final_pred.fillna(hier_pred)  # safety net

submission = pd.DataFrame({"id": test["id"], "scalar_coupling_constant": final_pred})



## === cell 5
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False, float_format="%.6f")
print(f"Submission written to {submission_path}, shape: {submission.shape}")
