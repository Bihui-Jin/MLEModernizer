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

-2.0809984084130706

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I fixed the file‑path typo so the script now points to the correct “champs‑scalar‑coupling” directory, allowing the training and test CSVs to be loaded. With the data loaded, the simple type‑mean baseline runs and writes a proper `submission.csv` containing the required columns.'
- What this solution (achieved 1.23566) has done: 'I enrich the simple type‑mean baseline by also averaging within each coupling type together with the two atom element symbols. This adds only a lightweight grouping step (no new model) and is expected to lower the log‑MAE, moving the score toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 1.23566) has done: 'The fix fills missing atom symbols after the structure merges so string operations work without errors, and then proceeds to generate the submission CSV as before. A small safety conversion to string is added, and the script now prints the first few rows of the created submission.'
- What this solution (achieved 1.23566) has done: 'I add a couple of lightweight aggregations – a mean per `(molecule_name, type)` and a mean per `atom_pair` regardless of type – and look them up in the prediction order before falling back to the broader means. This keeps the rule‑based baseline intact while giving the model a finer‑grained reference that should lower the log‑MAE toward the target score.'
- What this solution (achieved 1.23566) has done: 'We add a finer‑grained mean aggregation (by molecule + type + atom‑pair) and change the predictor to average all available group means instead of picking the first match. This keeps the rule‑based baseline while giving it more relevant information, which should lower the log‑MAE and move the score closer to the target.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight calibration step that blends the rule‑based prediction with the overall mean using a small scaling factor.  
A quick validation split is used to pick the blending weight that gives the lowest MAE, then the same weight is applied to the test predictions. This keeps the original grouping logic intact while reducing error and moving the score toward the negative target.'
- What this solution (achieved 1.23566) has done: 'I replace the simple unweighted averaging in `predict_row` / `predict_row_val` with a lightweight weighted scheme that gives higher importance to the most specific group‑level means (e.g., molecule + type + atom‑pair) and lower weight to broader aggregates. This keeps the overall rule‑based baseline unchanged while making predictions more precise, which should reduce the log‑MAE and move the score closer to the negative target. I also keep the calibration blending step (re‑computed after weighting) and ensure the script still writes a correct `submission.csv`.'
- What this solution (achieved 1.23566) has done: 'I adjust the calibration step to optimise the blending factor α for the **log‑MAE** (the actual competition metric) instead of plain MAE. This tiny change keeps the whole rule‑based pipeline intact while steering predictions toward a lower log‑MAE, moving the score closer to the negative target.'

# 9. Code solution

## === cell 0
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_absolute_error


def get_path(relative_path):
    possible_paths = [
        os.path.join("..", "input", relative_path),  # original notebook style
        os.path.join("/kaggle", "input", relative_path),  # typical Kaggle mount
        relative_path,  # fallback to current dir
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {relative_path} in any known location.")


train_path = get_path("champs-scalar-coupling/train.csv")
test_path = get_path("champs-scalar-coupling/test.csv")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

struct_path = get_path("champs-scalar-coupling/structures.csv")
struct_df = pd.read_csv(struct_path)[["molecule_name", "atom_index", "atom"]]


def merge_atom(df, suffix):
    df = (
        df.merge(
            struct_df,
            left_on=["molecule_name", f"atom_index_{suffix}"],
            right_on=["molecule_name", "atom_index"],
            how="left",
        )
        .rename(columns={"atom": f"atom_{suffix}"})
        .drop(columns=["atom_index"])
    )
    return df


train_df = merge_atom(train_df, 0)
test_df = merge_atom(test_df, 0)
train_df = merge_atom(train_df, 1)
test_df = merge_atom(test_df, 1)

train_df["atom_0"] = train_df["atom_0"].fillna("X").astype(str)
train_df["atom_1"] = train_df["atom_1"].fillna("X").astype(str)
test_df["atom_0"] = test_df["atom_0"].fillna("X").astype(str)
test_df["atom_1"] = test_df["atom_1"].fillna("X").astype(str)

train_df["atom_pair"] = train_df.apply(
    lambda r: "_".join(sorted([r["atom_0"], r["atom_1"]])), axis=1
)
test_df["atom_pair"] = test_df.apply(
    lambda r: "_".join(sorted([r["atom_0"], r["atom_1"]])), axis=1
)

type_means = train_df.groupby("type")["scalar_coupling_constant"].mean()
overall_mean = train_df["scalar_coupling_constant"].mean()
triple_means = train_df.groupby(["type", "atom_0", "atom_1"])[
    "scalar_coupling_constant"
].mean()
pair_means = train_df.groupby(["type", "atom_pair"])["scalar_coupling_constant"].mean()
pair_means_all = train_df.groupby("atom_pair")["scalar_coupling_constant"].mean()
molecule_means = train_df.groupby("molecule_name")["scalar_coupling_constant"].mean()
mol_type_means = train_df.groupby(["molecule_name", "type"])[
    "scalar_coupling_constant"
].mean()
mol_pair_means = train_df.groupby(["molecule_name", "type", "atom_pair"])[
    "scalar_coupling_constant"
].mean()


_WEIGHT_MAP = {
    "mol_pair": 5,
    "pair": 4,
    "pair_all": 3,
    "triple": 2,
    "type": 1,
    "mol_type": 2,
    "molecule": 1,
}


def weighted_predict(row, maps):
    """Return weighted average of all available group means for a row."""
    weighted_sum = 0.0
    total_weight = 0.0

    val = maps["mol_pair"].get((row["molecule_name"], row["type"], row["atom_pair"]))
    if pd.notnull(val):
        w = _WEIGHT_MAP["mol_pair"]
        weighted_sum += val * w
        total_weight += w

    val = maps["pair"].get((row["type"], row["atom_pair"]))
    if pd.notnull(val):
        w = _WEIGHT_MAP["pair"]
        weighted_sum += val * w
        total_weight += w

    val = maps["pair_all"].get(row["atom_pair"])
    if pd.notnull(val):
        w = _WEIGHT_MAP["pair_all"]
        weighted_sum += val * w
        total_weight += w

    val = maps["triple"].get((row["type"], row["atom_0"], row["atom_1"]))
    if pd.notnull(val):
        w = _WEIGHT_MAP["triple"]
        weighted_sum += val * w
        total_weight += w

    val = maps["type"].get(row["type"])
    if pd.notnull(val):
        w = _WEIGHT_MAP["type"]
        weighted_sum += val * w
        total_weight += w

    val = maps["mol_type"].get((row["molecule_name"], row["type"]))
    if pd.notnull(val):
        w = _WEIGHT_MAP["mol_type"]
        weighted_sum += val * w
        total_weight += w

    val = maps["molecule"].get(row["molecule_name"])
    if pd.notnull(val):
        w = _WEIGHT_MAP["molecule"]
        weighted_sum += val * w
        total_weight += w

    if total_weight > 0:
        return weighted_sum / total_weight
    else:
        return overall_mean


full_maps = {
    "mol_pair": mol_pair_means,
    "pair": pair_means,
    "pair_all": pair_means_all,
    "triple": triple_means,
    "type": type_means,
    "mol_type": mol_type_means,
    "molecule": molecule_means,
}

validation_frac = 0.10
val_df = train_df.sample(frac=validation_frac, random_state=42)
train_subset = train_df.drop(val_df.index)

type_means_val = train_subset.groupby("type")["scalar_coupling_constant"].mean()
overall_mean_val = train_subset["scalar_coupling_constant"].mean()
triple_means_val = train_subset.groupby(["type", "atom_0", "atom_1"])[
    "scalar_coupling_constant"
].mean()
pair_means_val = train_subset.groupby(["type", "atom_pair"])[
    "scalar_coupling_constant"
].mean()
pair_means_all_val = train_subset.groupby("atom_pair")[
    "scalar_coupling_constant"
].mean()
molecule_means_val = train_subset.groupby("molecule_name")[
    "scalar_coupling_constant"
].mean()
mol_type_means_val = train_subset.groupby(["molecule_name", "type"])[
    "scalar_coupling_constant"
].mean()
mol_pair_means_val = train_subset.groupby(["molecule_name", "type", "atom_pair"])[
    "scalar_coupling_constant"
].mean()

val_maps = {
    "mol_pair": mol_pair_means_val,
    "pair": pair_means_val,
    "pair_all": pair_means_all_val,
    "triple": triple_means_val,
    "type": type_means_val,
    "mol_type": mol_type_means_val,
    "molecule": molecule_means_val,
}

val_predictions = val_df.apply(lambda r: weighted_predict(r, val_maps), axis=1)

best_alpha = 1.0
best_log_mae = np.log(
    mean_absolute_error(val_df["scalar_coupling_constant"], val_predictions) + 1e-12
)

alphas = np.linspace(0.0, 1.0, 21)  # 0.0, 0.05, …, 1.0
for a in alphas:
    blended = a * val_predictions + (1 - a) * overall_mean_val
    mae = mean_absolute_error(val_df["scalar_coupling_constant"], blended)
    log_mae = np.log(mae + 1e-12)  # protect against log(0)
    if log_mae < best_log_mae:
        best_log_mae = log_mae
        best_alpha = a

print(
    f"Calibration: chosen blending factor α = {best_alpha:.3f} (log‑MAE on validation = {best_log_mae:.5f})"
)

test_df["scalar_coupling_constant"] = test_df.apply(
    lambda r: weighted_predict(r, full_maps), axis=1
)
test_df["scalar_coupling_constant"] = (
    best_alpha * test_df["scalar_coupling_constant"] + (1 - best_alpha) * overall_mean
)

submission = test_df[["id", "scalar_coupling_constant"]]
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")



## === cell 1
print(submission.head())
