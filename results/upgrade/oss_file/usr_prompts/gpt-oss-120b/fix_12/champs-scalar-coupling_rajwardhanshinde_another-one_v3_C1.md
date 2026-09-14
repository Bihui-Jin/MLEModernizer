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

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 3.00563) has done: 'I replace the missing external submission reads with a simple baseline that uses the training data’s overall mean scalar coupling constant. This ensures the script runs end‑to‑end, creates the required `stackers_blend.csv` with correct columns, and removes all NameError/Filename errors. The rest of the workflow (imports, data loading, and optional plotting) is kept unchanged.'
- What this solution (achieved 1.23566) has done: 'Implemented a type‑specific baseline: loaded the test set to obtain each record’s coupling type, computed the mean scalar coupling constant per type from the training data, and used these means for predictions (falling back to the global mean when a type is missing). This replaces the single global mean with a more informative per‑type mean, which should lower the MAE and therefore reduce the log‑MAE score toward the target. The script still writes the required `stackers_blend.csv` and keeps the original histogram visualization unchanged.'
- What this solution (achieved 1.23566) has done: 'I enhance the baseline by also using the mean scalar coupling constant for each combination of coupling type and the two atom elements (e.g., “1JHC C‑H”). This adds a simple, informative feature while keeping the overall mean‑per‑type approach. The script loads the atom element information from `structures.csv`, builds a mapping for `(type, atom0, atom1)`, falls back to the per‑type mean, and finally to the global mean. The rest of the workflow and output format remain unchanged.'
- What this solution (achieved 1.23566) has done: 'Implemented a distance‑based refinement to the mean‑per‑type/atom‑pair baseline. After merging atom coordinates, the script now computes the Euclidean distance between coupled atoms, bins it to 0.1 Å, and builds a mean target value for each (type, atom0, atom1, distance‑bin) group. During prediction it first looks up this more specific mean, then falls back to the pair mean, type mean, and finally the global mean. This modestly improves prediction accuracy, moving the log‑MAE score toward the target while preserving the original workflow and output format.'
- What this solution (achieved 1.23566) has done: 'I improve the baseline by adding a nearest‑bin fallback for the distance‑based means. If an exact distance bin is missing for a `(type, atom0, atom1)` group, the code now looks up the closest available bin, which provides a more informed prediction than falling back to the coarser pair or type means. This small change is expected to lower the log‑MAE (move the score toward the negative target) while leaving the overall workflow unchanged.'
- What this solution (achieved 1.23566) has done: 'I add a simple scaling factor derived from the training set to slightly adjust the baseline predictions. After computing the original predictions on the training data (using the existing per‑type/atom‑pair/distance means), I calculate the ratio of the true mean to the predicted mean and multiply all test predictions by this factor. This tiny calibration often lowers the MAE without altering the core grouping logic, moving the log‑MAE score closer to the negative target.'
- What this solution (achieved 1.23566) has done: 'I add a per‑type calibration factor (instead of a single global scaling) so predictions are adjusted more accurately for each coupling type. This keeps the original grouping logic untouched, only refines the final scaling step, which should lower the MAE and thus move the log‑MAE score closer to the negative target.'
- What this solution (achieved 1.23566) has done: 'I keep the existing grouping‑based baseline and its per‑type scaling, but add a tiny per‑type additive correction derived from the training residuals (actual – predicted). This small adjustment usually lowers the MAE without altering the core grouping logic, moving the log‑MAE score closer to the negative target.'
- What this solution (achieved 1.2386) has done: 'I tighten the calibration step by using medians (more robust to outliers) for the global and per‑type scaling factors and replace the per‑type additive correction with the median residual instead of the mean. This small change keeps the original grouping‑based baseline untouched while likely reducing the MAE and moving the log‑MAE score toward the negative target.'
- What this solution (achieved 1.23566) has done: 'I replace the median‑based calibration with a lightweight per‑type linear regression (slope + intercept) and a global fallback. This keeps the original grouping logic untouched while providing a more accurate scaling of the baseline predictions, which should lower the MAE and thus move the log‑MAE score closer to the negative target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import seaborn as sns
import os

print(os.listdir("../input"))




## === cell 1
train_path = "../input/champs-scalar-coupling/train.csv"
sample_path = "../input/champs-scalar-coupling/sample_submission.csv"
test_path = "../input/champs-scalar-coupling/test.csv"
structures_path = "../input/champs-scalar-coupling/structures.csv"

train = pd.read_csv(train_path)
sample = pd.read_csv(sample_path)
test = pd.read_csv(test_path)
structures = pd.read_csv(structures_path)




## === cell 2
global_mean = train["scalar_coupling_constant"].mean()
type_means = train.groupby("type")["scalar_coupling_constant"].mean()

train_atoms0 = train.merge(
    structures.rename(columns={"atom_index": "atom_index_0", "atom": "atom0"}),
    on=["molecule_name", "atom_index_0"],
    how="left",
)
train_full = train_atoms0.merge(
    structures.rename(columns={"atom_index": "atom_index_1", "atom": "atom1"}),
    on=["molecule_name", "atom_index_1"],
    how="left",
)

train_full["atom_a"] = train_full.apply(lambda r: min(r["atom0"], r["atom1"]), axis=1)
train_full["atom_b"] = train_full.apply(lambda r: max(r["atom0"], r["atom1"]), axis=1)

pair_means = (
    train_full.groupby(["type", "atom_a", "atom_b"])["scalar_coupling_constant"]
    .mean()
    .reset_index()
)

pair_mean_dict = {
    (row.type, row.atom_a, row.atom_b): row.scalar_coupling_constant
    for row in pair_means.itertuples(index=False)
}

train_full["distance"] = np.sqrt(
    (train_full["x_x"] - train_full["x_y"]) ** 2
    + (train_full["y_x"] - train_full["y_y"]) ** 2
    + (train_full["z_x"] - train_full["z_y"]) ** 2
)
train_full["dist_bin"] = (train_full["distance"] * 10).round() / 10

pair_dist_means = (
    train_full.groupby(["type", "atom_a", "atom_b", "dist_bin"])[
        "scalar_coupling_constant"
    ]
    .mean()
    .reset_index()
)

pair_dist_dict = {
    (row.type, row.atom_a, row.atom_b, row.dist_bin): row.scalar_coupling_constant
    for row in pair_dist_means.itertuples(index=False)
}

pair_dist_group = {}
for row in pair_dist_means.itertuples(index=False):
    key = (row.type, row.atom_a, row.atom_b)
    if key not in pair_dist_group:
        pair_dist_group[key] = {}
    pair_dist_group[key][row.dist_bin] = row.scalar_coupling_constant

test_atoms0 = test.merge(
    structures.rename(columns={"atom_index": "atom_index_0", "atom": "atom0"}),
    on=["molecule_name", "atom_index_0"],
    how="left",
)
test_full = test_atoms0.merge(
    structures.rename(columns={"atom_index": "atom_index_1", "atom": "atom1"}),
    on=["molecule_name", "atom_index_1"],
    how="left",
)

test_full["atom_a"] = test_full.apply(lambda r: min(r["atom0"], r["atom1"]), axis=1)
test_full["atom_b"] = test_full.apply(lambda r: max(r["atom0"], r["atom1"]), axis=1)

test_full["distance"] = np.sqrt(
    (test_full["x_x"] - test_full["x_y"]) ** 2
    + (test_full["y_x"] - test_full["y_y"]) ** 2
    + (test_full["z_x"] - test_full["z_y"]) ** 2
)
test_full["dist_bin"] = (test_full["distance"] * 10).round() / 10

submission = sample.merge(
    test_full[["id", "type", "atom0", "atom1", "dist_bin"]], on="id", how="left"
)


def predict_row(row):
    key_dist = (row.type, row.atom_a, row.atom_b, row.dist_bin)
    if key_dist in pair_dist_dict:
        return pair_dist_dict[key_dist]

    group_key = (row.type, row.atom_a, row.atom_b)
    if group_key in pair_dist_group:
        dist_dict = pair_dist_group[group_key]
        nearest_bin = min(dist_dict.keys(), key=lambda b: abs(b - row.dist_bin))
        return dist_dict[nearest_bin]

    key_pair = (row.type, row.atom_a, row.atom_b)
    if key_pair in pair_mean_dict:
        return pair_mean_dict[key_pair]

    if row.type in type_means:
        return type_means[row.type]

    return global_mean


train_pred = train_full.apply(predict_row, axis=1)

type_coeffs = {}
for t, df in train.assign(pred=train_pred).groupby("type"):
    x = df["pred"].values
    y = df["scalar_coupling_constant"].values
    if len(x) < 2:
        a, b = 1.0, 0.0
    else:
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    type_coeffs[t] = (a, b)

A_glob = np.vstack([train_pred.values, np.ones_like(train_pred.values)]).T
global_a, global_b = np.linalg.lstsq(
    A_glob, train["scalar_coupling_constant"].values, rcond=None
)[0]


def predict_row_calibrated(row):
    base = predict_row(row)
    a, b = type_coeffs.get(row.type, (global_a, global_b))
    return a * base + b


submission["scalar_coupling_constant"] = submission.apply(
    predict_row_calibrated, axis=1
)

output_path = "stackers_blend.csv"
submission[["id", "scalar_coupling_constant"]].to_csv(output_path, index=False)
print(f"Submission written to {output_path}")




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2106258997.py in <cell line: 0>()
    142 
    143 
--> 144 submission["scalar_coupling_constant"] = submission.apply(
    145     predict_row_calibrated, axis=1
    146 )

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in apply(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)
  10372             kwargs=kwargs,
  10373         )
> 10374         return op.apply().__finalize__(self, method="apply")
  10375 
  10376     def map(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
    914             return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)
    915 
--> 916         return self.apply_standard()
    917 
    918     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1061     def apply_standard(self):
   1062         if self.engine == "python":
-> 1063             results, res_index = self.apply_series_generator()
   1064         else:
   1065             results, res_index = self.apply_series_numba()

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_series_generator(self)
   1079             for i, v in enumerate(series_gen):
   1080                 # ignore SettingWithCopy here in case the user mutates
-> 1081                 results[i] = self.func(v, *self.args, **self.kwargs)
   1082                 if isinstance(results[i], ABCSeries):
   1083                     # If we have a view on v, we need to make a copy because

/tmp/ipykernel_11/2106258997.py in predict_row_calibrated(row)
    137 
    138 def predict_row_calibrated(row):
--> 139     base = predict_row(row)
    140     a, b = type_coeffs.get(row.type, (global_a, global_b))
    141     return a * base + b

/tmp/ipykernel_11/2106258997.py in predict_row(row)
     90 def predict_row(row):
     91     # 1. Exact distance‑binned mean
---> 92     key_dist = (row.type, row.atom_a, row.atom_b, row.dist_bin)
     93     if key_dist in pair_dist_dict:
     94         return pair_dist_dict[key_dist]

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'Series' object has no attribute 'atom_a'

## === cell 3
sns.histplot(sample["scalar_coupling_constant"], kde=True)
