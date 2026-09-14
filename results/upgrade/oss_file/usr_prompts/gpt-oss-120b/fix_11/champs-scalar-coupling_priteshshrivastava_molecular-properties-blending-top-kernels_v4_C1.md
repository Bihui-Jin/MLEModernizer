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

-1.5735603655829546

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'We replace the missing blending files with a simple baseline that predicts the mean `scalar_coupling_constant` for each coupling `type` (and a global mean fallback). This fixes the FileNotFoundError, guarantees a valid `.csv` submission, and still yields a reasonable score without altering any core modeling logic.'
- What this solution (achieved 1.23566) has done: 'I replace the simple type‑mean baseline with a slightly richer prediction that uses the average contribution components (fc, sd, pso, dso) for each coupling type. By summing these averaged contributions we obtain a more informed estimate while keeping the same overall structure and without adding complex modeling. If a type is missing, we fall back to the global mean, ensuring a valid submission.'
- What this solution (achieved 1.23566) has done: 'Improved the baseline by first predicting the per‑type mean scalar coupling constant (directly from the training target) which is usually far more accurate than using summed contribution averages. The prediction now uses the type‑mean when available, falls back to the contribution‑based estimate for unseen types, and finally to the global mean. This modest change keeps the original workflow intact while moving the validation score lower (closer to the negative target).'
- What this solution (achieved 1.23566) has done: 'I add a simple linear regression that learns how the four contribution terms (fc, sd, pso, dso) map to the target scalar coupling constant, then use those predictions for any test rows where the contributions are available. For rows without contributions I fall back to the per‑type mean and finally to the global mean, keeping the original workflow intact while aiming to lower the validation score toward the negative target.'
- What this solution (achieved 1.23566) has done: 'I keep the overall workflow the same but make the linear‑regression prediction apply to more rows. First, I compute per‑type averages of the four contribution features and merge those averages into the test set, filling any missing contribution values with the corresponding type‑mean. This allows the linear model to produce predictions for rows that previously fell back to the simple type‑mean, which should lower the MAE (and thus move the log‑MAE score toward the negative target). The rest of the code—including the global linear fit, type‑mean fallback and global‑mean fallback—remains unchanged.'
- What this solution (achieved 1.23566) has done: 'I keep the overall workflow unchanged but add a per‑type mean baseline and blend it with the linear‑regression prediction. By merging the type‑mean target into the test set, computing both predictions, and then averaging them when both are present (otherwise falling back to whichever is available and finally to the global mean), we obtain a more accurate estimate that should lower the log‑MAE toward the negative target while preserving the original model logic.'
- What this solution (achieved 1.23566) has done: 'I add a lightweight validation step that searches for the best linear blend weight between the contribution‑based linear regression and the per‑type mean predictions. By picking the weight that minimizes MAE on a held‑out split of the training data, the same weight is applied to the test set, which should lower the log‑MAE toward the negative target while keeping the original workflow unchanged.'
- What this solution (achieved 1.23566) has done: 'I keep the overall workflow unchanged but add a tiny calibration step: after finding the optimal linear‑type blend weight, I search for a simple shrinkage factor α that pulls the blended predictions toward the overall global mean. α is chosen on the same validation split that was used for the blend weight, so it only fine‑tunes the existing model without altering its core logic. The final test predictions are then α * prediction + (1‑α) * global_mean, which should reduce MAE a bit and move the log‑MAE closer to the negative target.'
- What this solution (achieved 1.23566) has done: 'We fine‑tune the blending weight and shrinkage factor with a finer grid and evaluate the blend on the validation split (instead of the training split) so the calibration better matches the out‑of‑sample error, which should lower the log‑MAE and move the score toward the negative target. No changes are made to the core model or feature engineering.'
- What this solution (achieved 1.23566) has done: 'I keep the overall workflow and linear‑regression blending unchanged, but add a small per‑type residual correction learned from the validation split. After the best blend weight and shrinkage factor are selected, I compute the average residual (target – prediction) for each coupling type on the validation data and then apply this correction to the test predictions. This modest adjustment is expected to lower the MAE (and thus the log‑MAE) and move the score closer to the negative target without altering the core model logic.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

data_dir = "../input/champs-scalar-coupling"
print("Available files:", os.listdir(data_dir)[:5])

train_path = os.path.join(data_dir, "train.csv")
test_path = os.path.join(data_dir, "test.csv")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

type_target_means = (
    train_df.groupby("type")["scalar_coupling_constant"]
    .mean()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "type_pred"})
)

contrib_path = os.path.join(data_dir, "scalar_coupling_contributions.csv")
contrib_df = pd.read_csv(contrib_path)

train_merged = train_df.merge(
    contrib_df, on=["molecule_name", "atom_index_0", "atom_index_1", "type"], how="left"
)

feat_cols = ["fc", "sd", "pso", "dso"]
train_feat = train_merged[feat_cols].values
train_target = train_merged["scalar_coupling_constant"].values

mask = ~np.isnan(train_feat).any(axis=1)
train_feat = train_feat[mask]
train_target = train_target[mask]

X = np.hstack([np.ones((train_feat.shape[0], 1)), train_feat])
coeffs, *_ = np.linalg.lstsq(X, train_target, rcond=None)
intercept, coef_fc, coef_sd, coef_pso, coef_dso = coeffs

feat_means_type = train_merged.groupby("type")[feat_cols].mean().reset_index()

lin_pred_full = (
    intercept
    + coef_fc * train_merged["fc"]
    + coef_sd * train_merged["sd"]
    + coef_pso * train_merged["pso"]
    + coef_dso * train_merged["dso"]
)

valid_rows = ~np.isnan(lin_pred_full)

lin_vals = lin_pred_full[valid_rows].values
type_vals = (
    train_merged.loc[valid_rows, "type"]
    .map(type_target_means.set_index("type")["type_pred"])
    .values
)
target_vals = train_target[valid_rows]

rng = np.random.RandomState(42)
perm = rng.permutation(len(lin_vals))
train_idx = perm[: int(0.8 * len(lin_vals))]
val_idx = perm[int(0.8 * len(lin_vals)) :]

best_w = 0.5
best_mae = np.inf
for w in np.linspace(0, 1, 101):  # 0.00 … 1.00 step 0.01
    pred = w * lin_vals[val_idx] + (1 - w) * type_vals[val_idx]
    mae = np.mean(np.abs(pred - target_vals[val_idx]))
    if mae < best_mae:
        best_mae = mae
        best_w = w

blend_weight = best_w
print(
    f"Selected blend weight (linear * {blend_weight:.2f} + type * {1-blend_weight:.2f})"
)

global_mean = train_df["scalar_coupling_constant"].mean()

combined_valid = blend_weight * lin_vals + (1 - blend_weight) * type_vals

best_alpha = 1.0
best_alpha_mae = np.inf
for a in np.linspace(0, 1, 101):  # 0.00 … 1.00 step 0.01
    pred_alpha = a * combined_valid[val_idx] + (1 - a) * global_mean
    mae_alpha = np.mean(np.abs(pred_alpha - target_vals[val_idx]))
    if mae_alpha < best_alpha_mae:
        best_alpha_mae = mae_alpha
        best_alpha = a

print(f"Selected shrinkage factor α = {best_alpha:.2f} (toward global mean)")

combined_valid_alpha = best_alpha * combined_valid + (1 - best_alpha) * global_mean

valid_types = train_merged.loc[valid_rows, "type"].reset_index(drop=True)

val_residuals = target_vals[val_idx] - combined_valid_alpha[val_idx]
val_types = valid_types.iloc[val_idx]

type_correction = (
    pd.DataFrame({"type": val_types, "resid": val_residuals})
    .groupby("type")["resid"]
    .mean()
    .to_dict()
)
print(f"Computed per‑type correction for {len(type_correction)} types.")



## === cell 1
submission = test_df[["id", "type"]].copy()

test_merged = test_df.merge(
    contrib_df,
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
)

test_merged = test_merged.merge(
    feat_means_type, on="type", suffixes=("", "_type_mean"), how="left"
)

for col in feat_cols:
    test_merged[col].fillna(test_merged[f"{col}_type_mean"], inplace=True)

lin_pred = (
    intercept
    + coef_fc * test_merged["fc"]
    + coef_sd * test_merged["sd"]
    + coef_pso * test_merged["pso"]
    + coef_dso * test_merged["dso"]
)

test_merged = test_merged.merge(type_target_means, on="type", how="left")

combined_pred = lin_pred.copy()

mask_lin = ~np.isnan(lin_pred)
mask_type = ~test_merged["type_pred"].isna()
both_present = mask_lin & mask_type

combined_pred[both_present] = (
    blend_weight * lin_pred[both_present]
    + (1 - blend_weight) * test_merged.loc[both_present, "type_pred"]
)

combined_pred[mask_type & ~mask_lin] = test_merged.loc[
    mask_type & ~mask_lin, "type_pred"
]

combined_pred = best_alpha * combined_pred + (1 - best_alpha) * global_mean

combined_pred += test_merged["type"].map(type_correction).fillna(0)

submission["scalar_coupling_constant"] = combined_pred

submission["scalar_coupling_constant"].fillna(global_mean, inplace=True)



## === cell 2
output_path = "my_blend_1.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}, shape: {submission.shape}")
