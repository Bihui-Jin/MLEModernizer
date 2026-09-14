# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

-2.357440763184845

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'I replace the missing external prediction files with a simple baseline that predicts the mean `scalar_coupling_constant` for each coupling `type` computed from the training data. This removes the FileNotFoundError, creates the required columns, and writes a valid `sub_ensemble.csv` submission file while preserving the overall workflow structure.'
- What this solution (achieved 1.23566) has done: 'I extend the simple mean‑by‑type baseline by adding a cheap geometric feature: the Euclidean distance between the two atoms (computed from the provided structures).  
For each coupling type I also compute the mean target in small distance bins (rounded to 0.1 Å).  
During prediction the code first looks up the type‑plus‑distance‑bin mean, then falls back to the type mean and finally to the global mean. This keeps the original workflow while providing a modest, data‑driven improvement that should lower the MAE toward the target.'
- What this solution (achieved 1.23566) has done: 'I add a very cheap linear‐distance correction per coupling type.  
For each type I fit a simple slope + intercept on the training distances, store these coefficients, and use the resulting regression prediction as the first guess.  
If the regression is unavailable it falls back to the existing type‑+‑distance‑bin mean, then the type mean, then the global mean – preserving the original workflow while nudging the MAE lower (moving the score toward the negative target).'
- What this solution (achieved 1.23566) has done: 'I compute a systematic bias from the training predictions and subtract it from the test predictions. This small calibration keeps the original modelling steps unchanged while nudging the scores lower (toward the negative target). The bias is calculated after merging the same type‑, distance‑, and regression‑based features into the train set, then applied to the test set predictions.'
- What this solution (achieved 1.23566) has done: 'I add a per‑type bias correction to the existing pipeline. After computing the global bias I also calculate the average residual for each coupling type on the training data and subtract this type‑specific bias from the test predictions (in addition to the existing global bias). This minimal tweak keeps the overall workflow unchanged while nudging the predictions closer to the true values, thereby lowering the log‑MAE toward the target score.'
- What this solution (achieved 2.13612) has done: 'I add a simple global scaling factor that aligns the magnitude of the predictions with the training targets. After computing the bias corrections in the training part, I calculate `scale_factor = (train["scalar_coupling_constant"] / train["pred"]).mean()` and then multiply the test predictions by this factor. This minimal adjustment keeps the existing workflow unchanged while nudging the predictions downward, moving the MAE‑log score closer to the negative target. The cells are renumbered starting from 1 to satisfy the required format.'
- What this solution (achieved 1.23566) has done: 'I keep the original feature‑engineering and bias‑correction steps but remove the final scaling factor multiplication, which was over‑adjusting predictions and increasing the MAE. By omitting this scaling the predictions stay closer to the calibrated values, moving the log‑MAE toward the lower target score while preserving the rest of the pipeline unchanged.'
- What this solution (achieved 2.13612) has done: 'I keep the existing workflow but apply the previously‑computed global scaling factor to the test predictions. Multiplying by `scale_factor` (which aligns the prediction magnitudes with the training targets) should reduce the MAE and therefore move the log‑MAE closer to the negative target score, while preserving all other logic and the final CSV output.'
- What this solution (achieved 1.41059) has done: 'I remove the over‑adjusting scaling factor multiplication from the test prediction pipeline and clip the final predictions to be non‑negative. This keeps the existing feature engineering, regression, and bias corrections while preventing predictions from being inflated, which should lower the log‑MAE and move the score closer to the negative target.'
- What this solution (achieved 1.41059) has done: 'I correct the data directory path so the CSV files are found, renumber the cells to start at 1, and keep the existing feature‑engineering and prediction logic unchanged. This fixes the FileNotFoundError and the subsequent NameError cascade, allowing the script to run end‑to‑end and write a valid `sub_ensemble.csv` submission file.'
- What this solution (achieved 1.23566) has done: 'I compute a global bias, per‑type bias, and a scaling factor from the training data using the existing feature‑engineered predictions, then apply these corrections to the test predictions. This adds only a small calibration step (no model changes) and is expected to lower the log‑MAE, moving the score closer to the negative target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = "/kaggle/input/champs-scalar-coupling"
DATA_DIR = os.getenv("DATA_DIR", DEFAULT_DATA_DIR)

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
struct_path = os.path.join(DATA_DIR, "structures.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
struct = pd.read_csv(struct_path)

global_mean = train["scalar_coupling_constant"].mean()




## === cell 1
def add_coordinates(df, struct_df):
    df = df.merge(
        struct_df.rename(
            columns={
                "atom_index": "atom_index_0",
                "atom": "atom_0",
                "x": "x0",
                "y": "y0",
                "z": "z0",
            }
        )[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    df = df.merge(
        struct_df.rename(
            columns={
                "atom_index": "atom_index_1",
                "atom": "atom_1",
                "x": "x1",
                "y": "y1",
                "z": "z1",
            }
        )[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )
    df["distance"] = np.sqrt(
        (df["x0"] - df["x1"]) ** 2
        + (df["y0"] - df["y1"]) ** 2
        + (df["z0"] - df["z1"]) ** 2
    )
    df["dist_bin"] = (df["distance"] * 10).round() / 10.0
    return df


train = add_coordinates(train, struct)

type_mean = (
    train.groupby("type")["scalar_coupling_constant"]
    .mean()
    .reset_index(name="pred_type_mean")
)

type_dist_mean = (
    train.groupby(["type", "dist_bin"])["scalar_coupling_constant"]
    .mean()
    .reset_index(name="pred_type_dist_mean")
)

type_reg_list = []
for typ, grp in train.groupby("type"):
    if len(grp) > 1:
        slope, intercept = np.polyfit(
            grp["distance"], grp["scalar_coupling_constant"], 1
        )
    else:
        slope, intercept = np.nan, np.nan
    type_reg_list.append({"type": typ, "slope": slope, "intercept": intercept})
type_reg = pd.DataFrame(type_reg_list)

train_feat = train.merge(type_dist_mean, on=["type", "dist_bin"], how="left")
train_feat = train_feat.merge(type_mean, on="type", how="left")
train_feat = train_feat.merge(type_reg, on="type", how="left")
train_feat["bias_type"] = 0.0  # placeholder, will be overwritten later

train_feat["pred"] = (
    train_feat["slope"] * train_feat["distance"] + train_feat["intercept"]
)
train_feat["pred"].fillna(train_feat["pred_type_dist_mean"], inplace=True)
train_feat["pred"].fillna(train_feat["pred_type_mean"], inplace=True)
train_feat["pred"].fillna(global_mean, inplace=True)

train_feat["residual"] = train_feat["pred"] - train_feat["scalar_coupling_constant"]

bias = train_feat["residual"].mean()

bias_type = (
    train_feat.groupby("type")["residual"]
    .mean()
    .reset_index()
    .rename(columns={"residual": "bias_type"})
)

train_feat["pred_adj"] = train_feat["pred"] - bias
train_feat = train_feat.merge(bias_type, on="type", how="left")
train_feat["pred_adj"] = train_feat["pred_adj"] - train_feat["bias_type"]

ratio = train_feat["scalar_coupling_constant"] / train_feat["pred_adj"]
scale_factor = ratio.replace([np.inf, -np.inf], np.nan).mean()




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'bias_type'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1420469258.py in <cell line: 0>()
     91 train_feat["pred_adj"] = train_feat["pred"] - bias
     92 train_feat = train_feat.merge(bias_type, on="type", how="left")
---> 93 train_feat["pred_adj"] = train_feat["pred_adj"] - train_feat["bias_type"]
     94 
     95 # compute a simple scaling factor aligning magnitude of predictions to targets

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'bias_type'

## === cell 2
test = add_coordinates(test, struct)

test = test.merge(type_dist_mean, on=["type", "dist_bin"], how="left")
test = test.merge(type_mean, on="type", how="left")
test = test.merge(type_reg, on="type", how="left")
test = test.merge(bias_type, on="type", how="left")
test["bias_type"].fillna(0.0, inplace=True)

test["pred"] = test["slope"] * test["distance"] + test["intercept"]
test["pred"].fillna(test["pred_type_dist_mean"], inplace=True)
test["pred"].fillna(test["pred_type_mean"], inplace=True)
test["pred"].fillna(global_mean, inplace=True)

test["pred"] -= bias
test["pred"] -= test["bias_type"]
test["pred"] *= scale_factor
test["pred"] = np.clip(test["pred"], 0, None)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3881080877.py in <cell line: 0>()
     15 test["pred"] -= bias
     16 test["pred"] -= test["bias_type"]
---> 17 test["pred"] *= scale_factor
     18 test["pred"] = np.clip(test["pred"], 0, None)
     19 

NameError: name 'scale_factor' is not defined

## === cell 3
submission = pd.DataFrame()
submission["id"] = test["id"]
submission["scalar_coupling_constant"] = test["pred"]

submission_path = "sub_ensemble.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")




## === cell 4
print(submission.head())
