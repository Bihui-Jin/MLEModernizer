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

-1.5209593019916507

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 3.00563) has done: 'I fix the broken file paths, handle missing external submission files gracefully, and fall back to a simple baseline (global mean of the training target) so that a valid `stackers_blend.csv` is always produced. This resolves the runtime errors and guarantees a submission file, while keeping the original blending logic when the external files are available.'
- What this solution (achieved 3.00563) has done: 'I fix the KeyError by merging the missing **type** column from the training data into the sample submission before applying the type‑based fallback. I also align any external submission files by `id` to ensure correct blending. The rest of the logic stays unchanged, guaranteeing a valid `stackers_blend.csv` is produced.'
- What this solution (achieved 3.00563) has done: 'The script failed because the sample submission file does not contain the required `molecule_name`, `atom_index_0`, `atom_index_1`, and `type` columns, causing a merge error with the structures data. I added a merge of the sample submission with the training set to bring in these columns, renamed the original prediction column to avoid conflicts, and kept the rest of the blending logic unchanged. This fixes the runtime error and ensures a valid `stackers_blend.csv` is written.'
- What this solution (achieved 3.00563) has done: 'I added a safeguard so that any NaN values produced when applying the per‑type bias are replaced with the original prediction, and then I also applied the overall bias (which was previously computed but unused). These tiny adjustments keep the core model unchanged while ensuring a complete, non‑NaN submission CSV is written.'
- What this solution (achieved 3.00563) has done: 'I add a simple distance‑based feature derived from the atom coordinates and use the mean coupling per (type, distance‑bin) as a fallback when the atom‑pair mean is missing. This keeps the original blending and bias corrections intact while giving the model more informative information, which should lower the log‑MAE toward the target without changing the core logic.'
- What this solution (achieved 3.00563) has done: 'I remove the post‑prediction bias corrections (type‑wise and overall bias) because they can over‑adjust the already simple mean‑based predictions and are likely inflating the error. Keeping only the pair / distance / type / global means (and optional external blending) should bring the log‑MAE closer to the target while preserving the original modeling logic.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import seaborn as sns

print("Input folders:", os.listdir("../input"))

base_path = "../input/champs-scalar-coupling"

train_path = os.path.join(base_path, "train.csv")
sample_path = os.path.join(base_path, "sample_submission.csv")
sub1_path = os.path.join(
    "../input", "lgb-public-kernels-plus-more-features", "sub_lgb_model_individual.csv"
)
sub2_path = os.path.join(
    "../input", "staking-and-stealing-like-a-molecule", "submission.csv"
)

train = pd.read_csv(train_path)
sample = pd.read_csv(sample_path)

sample = sample.rename(columns={"scalar_coupling_constant": "pred_placeholder"})
sample = sample.merge(
    train[["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]],
    on="id",
    how="left",
)


def safe_read(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"File not found (optional): {path}")
        return None


sub1 = safe_read(sub1_path)
sub2 = safe_read(sub2_path)

type_means = train.groupby("type")["scalar_coupling_constant"].mean()
global_mean = train["scalar_coupling_constant"].mean()




## === cell 1
structures_path = os.path.join(base_path, "structures.csv")
structures = pd.read_csv(structures_path)[
    ["molecule_name", "atom_index", "atom", "x", "y", "z"]
]

struct0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)
struct1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)

train_elem = train.merge(
    struct0, on=["molecule_name", "atom_index_0"], how="left"
).merge(struct1, on=["molecule_name", "atom_index_1"], how="left")

train_elem["distance"] = np.sqrt(
    (train_elem["x0"] - train_elem["x1"]) ** 2
    + (train_elem["y0"] - train_elem["y1"]) ** 2
    + (train_elem["z0"] - train_elem["z1"]) ** 2
)

train_elem["dist_bin"] = (train_elem["distance"] / 0.1).round() * 0.1


def fit_type(df):
    if len(df) > 1:
        slope, intercept = np.polyfit(df["distance"], df["scalar_coupling_constant"], 1)
    else:
        slope, intercept = 0.0, df["scalar_coupling_constant"].mean()
    return pd.Series({"slope": slope, "intercept": intercept})


type_dist_params = train_elem.groupby("type").apply(fit_type).reset_index()

type_elem_means = (
    train_elem.groupby(["type", "atom0", "atom1"])["scalar_coupling_constant"]
    .mean()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "pair_mean"})
)

dist_means = (
    train_elem.groupby(["type", "dist_bin"])["scalar_coupling_constant"]
    .mean()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "dist_mean"})
)

train_elem = train_elem.merge(
    type_elem_means, on=["type", "atom0", "atom1"], how="left"
)
train_elem = train_elem.merge(dist_means, on=["type", "dist_bin"], how="left")
train_elem = train_elem.merge(type_dist_params, on="type", how="left")

train_elem["dist_pred"] = (
    train_elem["slope"] * train_elem["distance"] + train_elem["intercept"]
)
train_base_pred = (
    train_elem["pair_mean"]
    .fillna(train_elem["dist_mean"])
    .fillna(train_elem["type"].map(type_means))
    .fillna(global_mean)
)
train_elem["pred"] = 0.6 * train_elem["dist_pred"] + 0.4 * train_base_pred

type_bias = (
    (train_elem["scalar_coupling_constant"] - train_elem["pred"])
    .groupby(train_elem["type"])
    .mean()
)

sample = sample.merge(struct0, on=["molecule_name", "atom_index_0"], how="left").merge(
    struct1, on=["molecule_name", "atom_index_1"], how="left"
)

sample["distance"] = np.sqrt(
    (sample["x0"] - sample["x1"]) ** 2
    + (sample["y0"] - sample["y1"]) ** 2
    + (sample["z0"] - sample["z1"]) ** 2
)
sample["dist_bin"] = (sample["distance"] / 0.1).round() * 0.1

sample = sample.merge(type_elem_means, on=["type", "atom0", "atom1"], how="left")
sample = sample.merge(dist_means, on=["type", "dist_bin"], how="left")
sample = sample.merge(type_dist_params, on="type", how="left")
sample["dist_pred"] = sample["slope"] * sample["distance"] + sample["intercept"]

base_pred = (
    sample["pair_mean"]
    .fillna(sample["dist_mean"])
    .fillna(sample["type"].map(type_means))
    .fillna(global_mean)
)

sample["scalar_coupling_constant"] = 0.6 * sample["dist_pred"] + 0.4 * base_pred

sample["scalar_coupling_constant"] += sample["type"].map(type_bias).fillna(0.0)

if sub1 is not None and sub2 is not None:
    sub1 = sub1.set_index("id")
    sub2 = sub2.set_index("id")
    sub1 = sub1.reindex(sample["id"])
    sub2 = sub2.reindex(sample["id"])
    blended = (
        0.5 * sub2["scalar_coupling_constant"] + 0.5 * sub1["scalar_coupling_constant"]
    )
    sample["scalar_coupling_constant"] = blended.fillna(
        sample["scalar_coupling_constant"]
    )

output_path = "stackers_blend.csv"
sample[["id", "scalar_coupling_constant"]].to_csv(output_path, index=False)
print(f"Submission written to {output_path}")




## === cell 2
print(sample["scalar_coupling_constant"].describe())




## === cell 3
sns.histplot(sample["scalar_coupling_constant"], kde=True)
