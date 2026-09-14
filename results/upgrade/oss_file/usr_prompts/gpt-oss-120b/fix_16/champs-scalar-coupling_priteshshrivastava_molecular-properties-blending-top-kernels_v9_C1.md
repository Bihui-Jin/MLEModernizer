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

-1.67498

# 6. Current score

1.23574

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the missing blending files with a simple, deterministic baseline: compute the mean scalar coupling constant for each coupling type from the training data and use those means as predictions for the test set (fallback to the overall mean for any missing type). This fixes the FileNotFoundError, ensures a valid “id,scalar_coupling_constant” submission CSV is written, and keeps the core logic simple and reproducible.'
- What this solution (achieved 1.23566) has done: 'The fix changes the way predictions are built: the original code attempted to call `fillna` on an Index object, causing a TypeError. We now create a proper pandas Series for the type‑element mean predictions, then safely fill missing values with the type mean and finally the overall mean. This restores the pipeline, produces a valid `id,scalar_coupling_constant` CSV, and leaves the core modeling logic unchanged.'
- What this solution (achieved 1.49278) has done: 'I replace the simple mean‑based baseline with a very lightweight GradientBoostingRegressor that uses only the coupling type and the two elemental symbols as features. The categorical values are factorized into integer codes (unseen categories get –1), and a small random subset (5 % of the training rows) is used to keep training fast. This modest model improves prediction quality enough to lower the log‑MAE toward the target while keeping the overall pipeline unchanged. The script still merges the structural information, creates the submission file, and respects the original file paths.'
- What this solution (achieved 1.46861) has done: 'I fix the training‑sample mismatch (X and y were sampled independently) and use a larger, more representative subset (20 %). I also add a simple interaction feature (type_code × elem0_code) that keeps the original lightweight model but gives it a bit more signal. These minimal changes should lower the validation MAE and move the log‑MAE score closer to the target while preserving the overall pipeline.'
- What this solution (achieved 1.45278) has done: 'I increase the training data proportion from 20 % to 50 % and give the GradientBoostingRegressor a bit more capacity (more trees, slightly lower learning rate, deeper trees). I also add a second interaction feature between the coupling type and the second atom element, which keeps the original lightweight modeling approach while providing extra signal. These modest changes should lower the validation error and move the Kaggle log‑MAE closer to the target without altering the overall pipeline.'
- What this solution (achieved 1.32033) has done: 'I add a strong signal – the mean coupling constant for each coupling type – as an additional feature for both training and test data, and raise the training‑sample fraction slightly (from 0.50 to 0.60) so the model sees more data. This preserves the original GradientBoostingRegressor pipeline while giving it clearer information, which should lower the validation error and move the log‑MAE toward the target score.'
- What this solution (achieved 1.23569) has done: 'I keep the overall pipeline but improve the prediction by training the GradientBoostingRegressor on the residual (target – type‑mean) instead of the raw target, then add the type‑mean back to the model output. I also use the full training set (no sampling) to give the model more data. These minimal changes keep the same features and model type while providing a stronger baseline that should lower the log‑MAE toward the target.'
- What this solution (achieved 1.23574) has done: 'We replace the slow `GradientBoostingRegressor` with its histogram‑based counterpart, which yields the same gradient‑boosting logic but runs orders of magnitude faster on large numeric tables. The placeholder slice that was never used is removed, and the validation‑MAE computation is kept identical so the metric stays unchanged. All other steps (data loading, merging, feature engineering) are left untouched.'
- What this solution (achieved 1.23574) has done: 'Implemented a proper float conversion for the `type_mean` column in the test features, fixing the `TypeError` caused by attempting to fill NaNs in a categorical series. The column is now cast to `float32` after mapping and before `fillna`, ensuring the pipeline runs end‑to‑end and produces a valid submission CSV.'

# 9. Code solution

## === cell 0
import os, pandas as pd

print("Available input directories:", os.listdir("../input"))



## === cell 1
BASE_PATH = "../input/champs-scalar-coupling/"

train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
structures_path = os.path.join(BASE_PATH, "structures.csv")

train_df = pd.read_csv(
    train_path,
    usecols=[
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
    dtype={
        "id": "int64",
        "molecule_name": "category",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
        "scalar_coupling_constant": "float32",
    },
)

test_df = pd.read_csv(
    test_path,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    dtype={
        "id": "int64",
        "molecule_name": "category",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
    },
)

structures = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "atom"],
    dtype={
        "molecule_name": "category",
        "atom_index": "int16",
        "atom": "category",
    },
)

structures = structures.rename(columns={"atom": "element"})

train_df = train_df.merge(
    structures[["molecule_name", "atom_index", "element"]].rename(
        columns={"atom_index": "atom_index_0", "element": "element_0"}
    ),
    on=["molecule_name", "atom_index_0"],
    how="left",
    sort=False,
)
test_df = test_df.merge(
    structures[["molecule_name", "atom_index", "element"]].rename(
        columns={"atom_index": "atom_index_0", "element": "element_0"}
    ),
    on=["molecule_name", "atom_index_0"],
    how="left",
    sort=False,
)

train_df = train_df.merge(
    structures[["molecule_name", "atom_index", "element"]].rename(
        columns={"atom_index": "atom_index_1", "element": "element_1"}
    ),
    on=["molecule_name", "atom_index_1"],
    how="left",
    sort=False,
)
test_df = test_df.merge(
    structures[["molecule_name", "atom_index", "element"]].rename(
        columns={"atom_index": "atom_index_1", "element": "element_1"}
    ),
    on=["molecule_name", "atom_index_1"],
    how="left",
    sort=False,
)



## === cell 2
test_df["type"] = test_df["type"].cat.set_categories(train_df["type"].cat.categories)
test_df["element_0"] = test_df["element_0"].cat.set_categories(
    train_df["element_0"].cat.categories
)
test_df["element_1"] = test_df["element_1"].cat.set_categories(
    train_df["element_1"].cat.categories
)

type_code = train_df["type"].cat.codes.astype("int16")
elem0_code = train_df["element_0"].cat.codes.astype("int16")
elem1_code = train_df["element_1"].cat.codes.astype("int16")

train_features = pd.DataFrame(
    {
        "type_code": type_code,
        "elem0_code": elem0_code,
        "elem1_code": elem1_code,
    }
)

train_features["type_elem0_inter"] = train_features["type_code"].astype(
    "int32"
) * train_features["elem0_code"].astype("int32")
train_features["type_elem1_inter"] = train_features["type_code"].astype(
    "int32"
) * train_features["elem1_code"].astype("int32")
train_features["elem0_elem1_inter"] = train_features["elem0_code"].astype(
    "int32"
) * train_features["elem1_code"].astype("int32")

type_mean_series = train_df.groupby("type")["scalar_coupling_constant"].mean()
overall_type_mean = type_mean_series.mean()
train_features["type_mean"] = train_df["type"].map(type_mean_series).astype("float32")

target = train_df["scalar_coupling_constant"]
residual = target - train_features["type_mean"]

train_sample = train_features
train_sample_y = residual

test_type_code = test_df["type"].cat.codes.astype("int16")
test_elem0_code = test_df["element_0"].cat.codes.astype("int16")
test_elem1_code = test_df["element_1"].cat.codes.astype("int16")

test_features = pd.DataFrame(
    {
        "type_code": test_type_code,
        "elem0_code": test_elem0_code,
        "elem1_code": test_elem1_code,
    }
)

test_features["type_elem0_inter"] = test_features["type_code"].astype(
    "int32"
) * test_features["elem0_code"].astype("int32")
test_features["type_elem1_inter"] = test_features["type_code"].astype(
    "int32"
) * test_features["elem1_code"].astype("int32")
test_features["elem0_elem1_inter"] = test_features["elem0_code"].astype(
    "int32"
) * test_features["elem1_code"].astype("int32")

test_features["type_mean"] = (
    test_df["type"]
    .map(type_mean_series)
    .astype("float32")  # ensure numeric dtype
    .fillna(overall_type_mean)
    .astype("float32")
)



## === cell 3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

X = train_sample.astype(np.float32).to_numpy()
y = train_sample_y.astype(np.float32).to_numpy()

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

gbr = HistGradientBoostingRegressor(
    max_iter=1200,
    learning_rate=0.03,
    max_depth=5,
    random_state=42,
)

gbr.fit(X_tr, y_tr)

type_mean_idx = train_sample.columns.get_loc("type_mean")
val_pred = X_val[:, type_mean_idx] + gbr.predict(X_val)
mae = mean_absolute_error(y_val + X_val[:, type_mean_idx], val_pred)
print(f"Validation MAE (raw): {mae:.4f}")



## === cell 4
X_test = test_features.astype(np.float32).to_numpy()
type_mean_idx_test = test_features.columns.get_loc("type_mean")
test_pred = X_test[:, type_mean_idx_test] + gbr.predict(X_test)



## === cell 5
submission = pd.DataFrame(
    {
        "id": test_df["id"],
        "scalar_coupling_constant": test_pred,
    }
)

submission_path = "my_blend_1.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")
