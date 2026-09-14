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

-2.100464760474246

# 6. Current score

1.18497

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the failing ensemble code with a simple baseline that predicts the average scalar coupling constant for each coupling type using the training data. This ensures the script runs end‑to‑end, creates a correctly‑named `submission.csv` file, and provides reasonable predictions without external files.'
- What this solution (achieved 1.23566) has done: 'I replace the simple type‑wise average prediction with a slightly richer baseline that uses the average contributions (`fc`, `sd`, `pso`, `dso`) for each coupling type (these sum to the target constant). The script now loads the `scalar_coupling_contributions.csv`, computes the mean sum of contributions per type, and uses that as the prediction. If a type is missing in the contributions file it falls back to the original type‑wise mean, and finally to the global mean, ensuring every row gets a value. This small augmentation is expected to lower the MAE‑log metric, moving the score closer to the target while keeping the core logic unchanged.'
- What this solution (achieved 1.23566) has done: 'I calibrate the contribution‑based predictions by scaling each type’s summed contributions to match the actual mean scalar coupling constant observed in the training set. This small adjustment keeps the original averaging logic while making the predictions more aligned with the true targets, which should lower the log‑MAE and move the score toward the negative target value.'
- What this solution (achieved 1.18497) has done: 'I replace the type‑wise averaging with median‑based aggregations (more robust to outliers) for both the scalar coupling constant and the contributions, recalibrate the contribution sum using these medians, and keep the same fallback logic. This minimal change keeps the core workflow unchanged while likely lowering the log‑MAE and moving the score toward the negative target.'
- What this solution (achieved 1.23566) has done: 'I switch the aggregations from medians to means for both the target scalar coupling constant per type and the contribution components, then recompute the scaling factor using these means. This small statistical tweak should reduce the average error (bringing the log‑MAE closer to the negative target) while preserving the overall workflow and fallback logic.'
- What this solution (achieved 1.18497) has done: 'I replace the mean‑based aggregations with median‑based ones, which are more robust to outliers and should lower the log‑MAE, moving the score closer to the negative target. The fallback logic now uses the type‑wise median and a global median instead of means, while keeping the overall workflow unchanged.'
- What this solution (achieved 1.18497) has done: 'I add a simple bias correction: after generating predictions with the median‑based contribution scaling, I compute the median residual on the training set (actual – predicted) and add this offset to all test predictions. This keeps the original workflow intact while adjusting for systematic under‑ or over‑prediction, which should lower the log‑MAE and move the score toward the negative target.'
- What this solution (achieved 1.18497) has done: 'I adjust the baseline by applying a per‑type median residual correction instead of a single global median offset. After computing the initial predictions on the training set, I calculate the median residual (actual − predicted) for each coupling type and add this type‑specific adjustment to the test predictions (fallbacking to the global median residual when a type is missing). This fine‑grained bias correction should reduce the MAE and therefore move the log‑MAE score closer to the lower target value while keeping the overall workflow unchanged.'
- What this solution (achieved 1.18497) has done: 'I replace the median‑only baseline with a per‑row contribution baseline: join the contribution file to each pair, sum the four contribution columns, and scale that sum by a per‑type factor derived from the median scalar‑coupling constant and median contribution sum. Missing contributions still fall back to the type‑wise median and then the global median. The residual bias corrections remain unchanged, giving a more accurate prediction while preserving the original workflow.'
- What this solution (achieved 1.18497) has done: 'I keep the overall workflow unchanged but improve the scaling factor used to convert summed contributions into predicted coupling constants.  
Instead of relying only on the median‑based scale, I also compute a mean‑based scale per coupling type and then average the two scales. This small, statistically‑sound tweak can reduce prediction error (lowering the log‑MAE) and moves the score closer to the negative target without altering the core logic or I/O behavior.'
- What this solution (achieved 1.18497) has done: 'I add a per‑type bias term computed from the median scalar coupling and the median contribution sum, then apply this bias to the contribution‑based prediction (while keeping the existing fallback logic). This small calibration should shift predictions closer to the true values and therefore lower the log‑MAE, moving the score toward the negative target without altering the overall workflow.'
- What this solution (achieved 1.18497) has done: 'I replace the median/mean‑based calibration with a simple per‑type linear regression that directly fits a slope (scale) and intercept (bias) between the summed contributions and the true coupling constant on the training data. This keeps the overall workflow unchanged, still falls back to type‑wise medians when a regression cannot be computed, and adds only a small, well‑behaved statistical tweak expected to lower the log‑MAE and move the score closer to the negative target.'
- What this solution (achieved 1.18497) has done: 'I keep the overall workflow unchanged but improve the calibration step: compute a global linear regression (scale + bias) on all rows with contributions and use it as a fallback when a coupling type lacks its own regression parameters. I also replace the median‑based residual corrections with mean‑based ones, which better align with the MAE‑based metric. These small statistical tweaks are expected to lower the log‑MAE (move the score toward the negative target) while preserving the original pipeline and output format.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
from pathlib import Path


def locate_file(relative_path):
    possible_paths = [
        Path("../input") / relative_path,
        Path("/kaggle/input") / relative_path,
        Path("..") / relative_path,
        Path(relative_path),
    ]
    for p in possible_paths:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Could not find {relative_path} in any known input directories."
    )


train_path = locate_file("champs-scalar-coupling/train.csv")
test_path = locate_file("champs-scalar-coupling/test.csv")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

contrib_path = locate_file("champs-scalar-coupling/scalar_coupling_contributions.csv")
contrib_df = pd.read_csv(contrib_path)




## === cell 1
type_medians = (
    train_df.groupby("type")["scalar_coupling_constant"]
    .median()
    .reset_index()
    .rename(columns={"scalar_coupling_constant": "type_median"})
)
global_median = train_df["scalar_coupling_constant"].median()

contrib_df["contrib_sum"] = (
    contrib_df["fc"] + contrib_df["sd"] + contrib_df["pso"] + contrib_df["dso"]
)

train_join = train_df.merge(
    contrib_df[
        ["molecule_name", "atom_index_0", "atom_index_1", "type", "contrib_sum"]
    ],
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
)

type_params = []
for typ, grp in train_join.groupby("type"):
    grp = grp.dropna(subset=["contrib_sum", "scalar_coupling_constant"])
    if len(grp) < 2:
        continue
    x = grp["contrib_sum"]
    y = grp["scalar_coupling_constant"]
    x_mean = x.mean()
    y_mean = y.mean()
    var_x = ((x - x_mean) ** 2).mean()
    cov_xy = ((x - x_mean) * (y - y_mean)).mean()
    if var_x == 0:
        continue
    scale = cov_xy / var_x
    bias = y_mean - scale * x_mean
    type_params.append({"type": typ, "scale": scale, "bias": bias})

type_params = pd.DataFrame(type_params)

type_contrib_medians = (
    contrib_df.groupby("type")["contrib_sum"]
    .median()
    .reset_index()
    .rename(columns={"contrib_sum": "type_contrib_median"})
)
type_medians_for_scale = type_medians.merge(type_contrib_medians, on="type", how="left")
type_medians_for_scale["scale_median"] = (
    type_medians_for_scale["type_median"]
    / type_medians_for_scale["type_contrib_median"]
)
type_medians_for_scale["bias_median"] = (
    type_medians_for_scale["type_median"]
    - type_medians_for_scale["scale_median"]
    * type_medians_for_scale["type_contrib_median"]
)

type_params = pd.concat(
    [
        type_params,
        type_medians_for_scale[["type", "scale_median", "bias_median"]].rename(
            columns={"scale_median": "scale", "bias_median": "bias"}
        ),
    ],
    ignore_index=True,
    sort=False,
)

type_params = type_params.drop_duplicates(subset=["type"], keep="first")

global_grp = train_join.dropna(subset=["contrib_sum", "scalar_coupling_constant"])
x_glob = global_grp["contrib_sum"]
y_glob = global_grp["scalar_coupling_constant"]
xg_mean = x_glob.mean()
yg_mean = y_glob.mean()
var_xg = ((x_glob - xg_mean) ** 2).mean()
cov_xyg = ((x_glob - xg_mean) * (y_glob - yg_mean)).mean()
global_scale = cov_xyg / var_xg if var_xg != 0 else 1.0
global_bias = yg_mean - global_scale * xg_mean

train_pred = None  # placeholder, will be filled after make_predictions is defined




## === cell 2
def make_predictions(df):
    pred_df = df.merge(
        contrib_df[
            ["molecule_name", "atom_index_0", "atom_index_1", "type", "contrib_sum"]
        ],
        on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
        how="left",
    )
    pred_df = pred_df.merge(type_params, on="type", how="left")
    pred_df["scale"].fillna(global_scale, inplace=True)
    pred_df["bias"].fillna(global_bias, inplace=True)

    pred_df["pred"] = pred_df["contrib_sum"] * pred_df["scale"] + pred_df["bias"]
    pred_df["pred"].fillna(pd.NA, inplace=True)
    pred_df = pred_df.merge(type_medians, on="type", how="left")
    pred_df["pred"].fillna(pred_df["type_median"], inplace=True)
    pred_df["pred"].fillna(global_median, inplace=True)
    return pred_df[["id", "type", "pred"]]


train_pred = make_predictions(train_df)

train_residuals = train_df["scalar_coupling_constant"] - train_pred["pred"]
type_residuals = (
    pd.DataFrame({"type": train_df["type"], "residual": train_residuals})
    .groupby("type")["residual"]
    .mean()
    .reset_index()
    .rename(columns={"residual": "type_residual"})
)
global_residual = train_residuals.mean()

submission = make_predictions(test_df)

submission = submission.merge(type_residuals, on="type", how="left")
submission["type_residual"].fillna(0.0, inplace=True)
submission["pred"] = submission["pred"] + submission["type_residual"] + global_residual

submission_file = "submission.csv"
submission[["id", "pred"]].rename(columns={"pred": "scalar_coupling_constant"}).to_csv(
    submission_file, index=False
)

print(f"Submission written to {submission_file}")
print("First 5 rows of the submission:")
print(submission.head())




## === cell 3
assert os.path.isfile(submission_file), "Submission file was not created."
sub_check = pd.read_csv(submission_file)
expected_cols = ["id", "scalar_coupling_constant"]
assert (
    list(sub_check.columns) == expected_cols
), f"Column names mismatch. Expected {expected_cols}"
print("Submission file verified successfully.")
