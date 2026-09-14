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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.5536

# 6. Current score

1.21757

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.49181) has done: 'Diagnosis: The crash in cell 2 happens when calling `reg.predict(test[col])` because `test[col]` contains NaN values. These NaNs come from earlier feature engineering/merges (e.g., failed joins on structures or division in `dist_to_type_mean`), and `ExtraTreesRegressor` cannot handle NaNs.  
Patch summary: In cell 2 only, impute missing values in the selected feature columns using deterministic statistics computed from the training split used for fitting (column-wise medians), then apply the same fill values to validation and test features. This preserves the model and training semantics while preventing NaNs at fit/predict time.  
Updated cells: Only cell 2 is modified as required.  
Compatibility notes for cell k+1: No interfaces are changed; `test['scalar_coupling_constant']` and the written `submission.csv` remain exactly as expected.  
Assumptions: Using median imputation is acceptable as minimal preprocessing to satisfy sklearn’s non-NaN requirement; no extra columns are added and all original features are retained.'
- What this solution (achieved 1.49181) has done: 'Your current score (1.49181, lower-is-better) is far from the target (0.5536), so we should improve performance but with minimal, low-risk changes that preserve the same model and training loop. The biggest issue is feature inconsistency: `dist_to_type_mean` in test is computed using *test* type means, which won’t match the training distribution and degrades generalization; we instead compute type means from train and apply them to both train and test (with a safe global fallback). We also avoid any remaining NaN/inf from division by zero by using a small epsilon and replacing inf with NaN before the existing median imputation. These changes keep the exact same model (ExtraTreesRegressor) and same overall approach, but should move the score substantially closer to the target.'
- What this solution (achieved 1.57224) has done: 'We need to move the score down (lower is better) from 1.49181 toward 0.5536, so we make small, low-risk feature fixes without changing the model or training loop. The biggest current issue is that the `LabelEncoder` is fit separately for each `type{i}` but reused across positions, which makes the encodings inconsistent and harms generalization; we use a separate encoder per character position. We also add a very small set of “free” geometric features (dx/dy/dz and squared distance) derived from already-merged coordinates; this preserves the core feature approach and typically improves CHAMPS performance. Finally, we keep the existing NaN/inf handling and ensure train-derived statistics (type means, medians) are consistently applied to validation and test.'
- What this solution (achieved 1.64218) has done: 'We keep your same ExtraTrees model and train/test split, but add two minimal, competition-relevant fixes that typically reduce log-MAE here. First, we make the `type0..type3` encodings consistent by fitting a separate `LabelEncoder` per character position on the concatenated train+test values (avoids unseen-category issues and keeps mappings stable). Second, we add one very low-risk geometric feature (`inv_dist`) derived from your already-computed distance, which often helps tree models without changing the approach. Everything else (feature set, imputation, training size, and submission writing) stays the same to move the score down toward the 0.5536 target.'
- What this solution (achieved 1.80345) has done: 'We make two minimal, score-relevant fixes while keeping the same ExtraTrees model and train/test split. First, we stop dropping `atom_index_0` and `atom_index_1` from the feature set (they are molecule-local but still informative and commonly help tree models here), which should reduce error without changing the overall approach. Second, we make the NaN imputation fully consistent by computing medians on the exact numeric feature columns used by the model and applying them to train/valid/test, avoiding any silent non-numeric median omissions. Everything else (feature engineering, model, training loop, and submission writing) stays the same.'
- What this solution (achieved 1.89347) has done: 'Diagnosis: The crash happens at `reg.fit(x1, y1_t)` because `y1_t = np.log1p(y1.values)` produces NaNs when `y1` contains values `<= -1` (log1p domain violation), and scikit-learn refuses NaNs in the target. This is consistent with the traceback “Input y contains NaN.” Since the core modeling logic is to train on `log1p(target)` and invert with `expm1`, the minimal safe fix is to filter out rows where `scalar_coupling_constant <= -1` (or NaN) from both `X` and `y` before splitting and transforming. This preserves the same approach while preventing invalid log transforms.

Patch summary: In cell 2, before `train_test_split`, subset the training frame to only valid target rows (`finite` and `> -1`) and use that subset for both features and target. No other logic (model, features, metric proxy, inference) is changed.

Updated cells: cell 2 only.

Compatibility notes for cell k+1: No new variables are introduced and existing outputs (`cv_metric`, `pred_test`, `submission.csv`) remain the same names/types. The split indices still map correctly to `train` via `x2.index` (they now be a subset of the original indices).

Assumptions: The NaNs in `y1_t` come from invalid `log1p` inputs (`<= -1`) or existing NaNs in `scalar_coupling_constant`, and dropping only those rows is acceptable for continuing execution without changing the intended transform-based training.'
- What this solution (achieved 1.88805) has done: 'You’re far above the target (1.89347 vs 0.5536, lower-is-better), so we should improve score with the smallest, safest fixes that keep your ExtraTrees + log1p/expm1 approach intact. The biggest remaining issue is your training subset: using `train.tail(2_000_000)` is arbitrary and can exclude many coupling types/molecules, hurting the per-type log-MAE; switching to a deterministic random sample of the same size keeps runtime similar but yields a much more representative training distribution. Second, your split is row-wise, but the competition splits by molecule; using a molecule-wise split (still a simple train/valid split, same training loop) better match the evaluation distribution and typically reduces type-wise errors. Everything else (features, model, log transform, median imputation, submission writing) stays the same.'
- What this solution (achieved 1.88952) has done: 'We need to move the score down (lower-is-better) from 1.88805 toward 0.5536, so I make two minimal, model-preserving improvements that typically reduce CHAMPS log-MAE without changing your ExtraTrees/log1p approach. First, I add the most “free” high-signal baseline feature: the atom types for atom_index_0 and atom_index_1 from `structures.csv`, encoded as integers (this is consistent with your existing structures merge and helps trees a lot). Second, I add one stable, low-cost feature: the integer bond separation inferred from the coupling `type` (e.g., `1J`, `2J`, `3J`) to help the model differentiate regimes; everything else (split-by-molecule, imputation, regressor, submission writing) stays the same.'
- What this solution (achieved 1.42425) has done: 'We’re still far above the target (1.88952 vs 0.5536, lower-is-better), so we should improve generalization without changing your core ExtraTrees + log1p/expm1 approach. The most impactful minimal fix here is to train *separate models per coupling `type`* (same regressor, same features, same transform) because the competition metric is averaged per type and the physics differs strongly by type; this typically reduces log-MAE substantially. To keep it stable and within the time limit, we keep the same 2M row cap but apply it *stratified by type* so all types are represented, then do a molecule-wise split within each type. Finally, we keep your existing deterministic median imputation and ensure the submission aligns exactly to `sample_submission` order.'
- What this solution (achieved 1.23886) has done: 'Your current log-MAE (1.42425, lower-is-better) is still far from the target (0.5536), so we should improve generalization with minimal, low-risk changes that keep your per-type ExtraTrees + log1p/expm1 approach intact. The biggest mismatch with the competition metric is that we evaluate MAE on the original scale while we train on log1p; switching the per-type model target to **signed log1p** (log1p of absolute value with sign) keeps the same transform-based semantics but better handles negative coupling constants without filtering them out. We also fix a subtle imputation bug: the numeric columns used for filling are derived from `test[col]`, which can exclude numeric columns that exist in train (or include ones with all-NaN in test); instead compute numeric columns from `train[col]` and align medians to each split, which stabilizes per-type fits. Finally, we keep the same features/model, but ensure the per-type sample allocation sums exactly to `MAX_ROWS` so no type is unintentionally under/overrepresented.'
- What this solution (achieved 1.21958) has done: 'Your current score (1.23886, lower-is-better) is still far from the target (0.5536), so we should improve generalization with the smallest safe changes that keep the same per-type ExtraTrees + signed-log1p target. The biggest remaining weakness is that the model never sees molecule-level context beyond the two atom coordinates; we can add low-cost, high-signal molecule/atom auxiliary features already provided (mulliken charges, magnetic shielding tensors, dipole moments, potential energy) via simple left-joins, which is consistent with your existing merge-based feature engineering. To keep runtime under 600s, we only load the needed columns and merge onto atom_index_0/1 and molecule_name, then reuse the same median imputation pipeline. This should reduce per-type MAE substantially and move the score closer to the target without changing the core modeling/training approach.'
- What this solution (achieved 1.21757) has done: 'Your score (1.21958, lower-is-better) is still far from the target (0.5536), so we should improve it with minimal, low-risk changes that keep your per-type ExtraTrees + signed-log1p approach intact. The biggest remaining issue is that you compute one global `num_cols` list from the whole training frame, but then compute per-type medians on `X_train[num_cols]`; for some types, many columns can be all-NaN, producing NaN fill values that leak into `X_valid/X_test` and degrade (or break) predictions. I make the imputation fully robust per type by (1) selecting numeric columns from the per-type `X_train`, (2) replacing any NaN medians with global (all-train) medians, and (3) finally filling any remaining NaNs with 0.0—this preserves your pipeline while ensuring deterministic, complete imputation. I also add a tiny per-type bias correction (using the median residual on that type’s validation split) to improve MAE without changing the model/loop or using any extra data.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn import *

INPUT_DIR = "../input"

train = pd.read_csv(f"{INPUT_DIR}/train.csv")
test = pd.read_csv(f"{INPUT_DIR}/test.csv")
sub = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")
print(train.shape, test.shape, sub.shape)

train["atom1"] = train["type"].map(lambda x: str(x)[2])
train["atom2"] = train["type"].map(lambda x: str(x)[3])
test["atom1"] = test["type"].map(lambda x: str(x)[2])
test["atom2"] = test["type"].map(lambda x: str(x)[3])

for i in range(4):
    lbl_i = preprocessing.LabelEncoder()
    all_vals = pd.concat(
        [
            train["type"].map(lambda x: str(x)[i]),
            test["type"].map(lambda x: str(x)[i]),
        ],
        axis=0,
        ignore_index=True,
    )
    lbl_i.fit(all_vals)
    train["type" + str(i)] = lbl_i.transform(train["type"].map(lambda x: str(x)[i]))
    test["type" + str(i)] = lbl_i.transform(test["type"].map(lambda x: str(x)[i]))

structures = pd.read_csv(f"{INPUT_DIR}/structures.csv").rename(
    columns={
        "atom_index": "atom_index_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
        "atom": "atom1",
    }
)
train = pd.merge(
    train, structures, how="left", on=["molecule_name", "atom_index_0", "atom1"]
)
test = pd.merge(
    test, structures, how="left", on=["molecule_name", "atom_index_0", "atom1"]
)
del structures

structures = pd.read_csv(f"{INPUT_DIR}/structures.csv").rename(
    columns={
        "atom_index": "atom_index_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
        "atom": "atom2",
    }
)
train = pd.merge(
    train, structures, how="left", on=["molecule_name", "atom_index_1", "atom2"]
)
test = pd.merge(
    test, structures, how="left", on=["molecule_name", "atom_index_1", "atom2"]
)
del structures
print(train.shape, test.shape, sub.shape)

atom_map = pd.read_csv(
    f"{INPUT_DIR}/structures.csv", usecols=["molecule_name", "atom_index", "atom"]
).rename(columns={"atom_index": "atom_index_0", "atom": "atom0"})
train = pd.merge(train, atom_map, how="left", on=["molecule_name", "atom_index_0"])
test = pd.merge(test, atom_map, how="left", on=["molecule_name", "atom_index_0"])
del atom_map

atom_map = pd.read_csv(
    f"{INPUT_DIR}/structures.csv", usecols=["molecule_name", "atom_index", "atom"]
).rename(columns={"atom_index": "atom_index_1", "atom": "atom1s"})
train = pd.merge(train, atom_map, how="left", on=["molecule_name", "atom_index_1"])
test = pd.merge(test, atom_map, how="left", on=["molecule_name", "atom_index_1"])
del atom_map

atom_lbl = preprocessing.LabelEncoder()
all_atoms = pd.concat(
    [train["atom0"], train["atom1s"], test["atom0"], test["atom1s"]],
    axis=0,
    ignore_index=True,
).astype(str)
atom_lbl.fit(all_atoms)
train["atom0_enc"] = atom_lbl.transform(train["atom0"].astype(str))
train["atom1s_enc"] = atom_lbl.transform(train["atom1s"].astype(str))
test["atom0_enc"] = atom_lbl.transform(test["atom0"].astype(str))
test["atom1s_enc"] = atom_lbl.transform(test["atom1s"].astype(str))

mulliken = pd.read_csv(
    f"{INPUT_DIR}/mulliken_charges.csv",
    usecols=["molecule_name", "atom_index", "mulliken_charge"],
)
m0 = mulliken.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_0"}
)
m1 = mulliken.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_1"}
)
train = pd.merge(train, m0, how="left", on=["molecule_name", "atom_index_0"])
test = pd.merge(test, m0, how="left", on=["molecule_name", "atom_index_0"])
train = pd.merge(train, m1, how="left", on=["molecule_name", "atom_index_1"])
test = pd.merge(test, m1, how="left", on=["molecule_name", "atom_index_1"])
del mulliken, m0, m1

mst = pd.read_csv(
    f"{INPUT_DIR}/magnetic_shielding_tensors.csv",
    usecols=["molecule_name", "atom_index", "XX", "YY", "ZZ"],
)
mst0 = mst.rename(
    columns={
        "atom_index": "atom_index_0",
        "XX": "mst_xx_0",
        "YY": "mst_yy_0",
        "ZZ": "mst_zz_0",
    }
)
mst1 = mst.rename(
    columns={
        "atom_index": "atom_index_1",
        "XX": "mst_xx_1",
        "YY": "mst_yy_1",
        "ZZ": "mst_zz_1",
    }
)
train = pd.merge(train, mst0, how="left", on=["molecule_name", "atom_index_0"])
test = pd.merge(test, mst0, how="left", on=["molecule_name", "atom_index_0"])
train = pd.merge(train, mst1, how="left", on=["molecule_name", "atom_index_1"])
test = pd.merge(test, mst1, how="left", on=["molecule_name", "atom_index_1"])
del mst, mst0, mst1

dipole = pd.read_csv(
    f"{INPUT_DIR}/dipole_moments.csv", usecols=["molecule_name", "X", "Y", "Z"]
).rename(columns={"X": "dipole_x", "Y": "dipole_y", "Z": "dipole_z"})
pe = pd.read_csv(
    f"{INPUT_DIR}/potential_energy.csv", usecols=["molecule_name", "potential_energy"]
)
train = pd.merge(train, dipole, how="left", on="molecule_name")
test = pd.merge(test, dipole, how="left", on="molecule_name")
train = pd.merge(train, pe, how="left", on="molecule_name")
test = pd.merge(test, pe, how="left", on="molecule_name")
del dipole, pe



## === cell 1
train["dx"] = train["x0"] - train["x1"]
train["dy"] = train["y0"] - train["y1"]
train["dz"] = train["z0"] - train["z1"]
test["dx"] = test["x0"] - test["x1"]
test["dy"] = test["y0"] - test["y1"]
test["dz"] = test["z0"] - test["z1"]

train["dist2"] = (
    train["dx"] * train["dx"] + train["dy"] * train["dy"] + train["dz"] * train["dz"]
)
test["dist2"] = (
    test["dx"] * test["dx"] + test["dy"] * test["dy"] + test["dz"] * test["dz"]
)

train["dist"] = np.sqrt(train["dist2"])
test["dist"] = np.sqrt(test["dist2"])

train_type_mean = train.groupby("type")["dist"].mean()
global_mean = train["dist"].mean()
eps = 1e-12

train_mean_for_row = train["type"].map(train_type_mean).fillna(global_mean)
test_mean_for_row = test["type"].map(train_type_mean).fillna(global_mean)

train["dist_to_type_mean"] = train["dist"] / (train_mean_for_row + eps)
test["dist_to_type_mean"] = test["dist"] / (test_mean_for_row + eps)

train["inv_dist"] = 1.0 / (train["dist"] + eps)
test["inv_dist"] = 1.0 / (test["dist"] + eps)

train["J_order"] = (
    train["type"]
    .map(lambda s: int(str(s)[0]) if str(s)[0].isdigit() else -1)
    .astype(np.int16)
)
test["J_order"] = (
    test["type"]
    .map(lambda s: int(str(s)[0]) if str(s)[0].isdigit() else -1)
    .astype(np.int16)
)

train.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)



## === cell 2
col = [
    c
    for c in train.columns
    if c
    not in [
        "id",
        "molecule_name",
        "scalar_coupling_constant",
        "type",
        "atom1",
        "atom2",
        "atom0",
        "atom1s",
    ]
]


def make_regressor():
    return ensemble.ExtraTreesRegressor(n_jobs=-1, n_estimators=20, random_state=4)


MAX_ROWS = 2_000_000

y_all = train["scalar_coupling_constant"].values
valid_mask = np.isfinite(y_all)
train_valid = train.loc[valid_mask].copy()

if len(train_valid) > MAX_ROWS:
    type_counts = train_valid["type"].value_counts()
    min_per_type = 2000
    alloc = (type_counts / type_counts.sum() * MAX_ROWS).astype(int)
    alloc = np.maximum(alloc, min_per_type)

    if alloc.sum() != MAX_ROWS:
        if alloc.sum() > MAX_ROWS:
            scale = MAX_ROWS / alloc.sum()
            alloc = np.maximum((alloc * scale).astype(int), 1)

        diff = int(MAX_ROWS - alloc.sum())
        if diff != 0:
            order = alloc.sort_values(ascending=False).index.tolist()
            i = 0
            step = 1 if diff > 0 else -1
            diff_abs = abs(diff)
            while diff_abs > 0 and i < 10_000_000:
                t = order[i % len(order)]
                new_val = int(alloc.loc[t] + step)
                if new_val >= 1:
                    alloc.loc[t] = new_val
                    diff_abs -= 1
                i += 1

    parts = []
    for t, n in alloc.items():
        df_t = train_valid[train_valid["type"] == t]
        if len(df_t) <= n:
            parts.append(df_t)
        else:
            parts.append(df_t.sample(n=int(n), random_state=99))
    train_sub = pd.concat(parts, axis=0, ignore_index=False)
else:
    train_sub = train_valid

X_global = train_sub[col]
global_num_cols = X_global.columns[
    X_global.dtypes.apply(lambda dt: np.issubdtype(dt, np.number))
]
global_fill_values = X_global[global_num_cols].median()

test_X_full = test[col].copy()
pred_test_full = np.zeros(len(test), dtype=np.float64)
cv_mae_by_type = {}

types = sorted(train_sub["type"].unique().tolist())
print("Training per-type models for", len(types), "types:", types)

for t in types:
    df_t = train_sub[train_sub["type"] == t].copy()
    if df_t.empty:
        continue

    mols = df_t["molecule_name"].unique()
    if len(mols) < 5:
        is_valid = np.zeros(len(df_t), dtype=bool)
    else:
        mols_train, mols_valid = model_selection.train_test_split(
            mols, test_size=0.2, random_state=99
        )
        is_valid = df_t["molecule_name"].isin(mols_valid).values

    X_train = df_t.loc[~is_valid, col].copy()
    y_train = df_t.loc[~is_valid, "scalar_coupling_constant"].copy()

    if len(X_train) == 0:
        X_train = df_t[col].copy()
        y_train = df_t["scalar_coupling_constant"].copy()
        X_valid = None
        y_valid = None
    else:
        X_valid = df_t.loc[is_valid, col].copy()
        y_valid = df_t.loc[is_valid, "scalar_coupling_constant"].copy()

    num_cols_t = X_train.columns[
        X_train.dtypes.apply(lambda dt: np.issubdtype(dt, np.number))
    ]
    fill_values_t = X_train[num_cols_t].median()
    fill_values_t = fill_values_t.fillna(global_fill_values.reindex(num_cols_t))
    fill_values_t = fill_values_t.fillna(0.0)

    X_train[num_cols_t] = X_train[num_cols_t].fillna(fill_values_t)
    X_train[num_cols_t] = X_train[num_cols_t].fillna(0.0)

    if X_valid is not None and len(X_valid) > 0:
        X_valid[num_cols_t] = X_valid[num_cols_t].fillna(fill_values_t)
        X_valid[num_cols_t] = X_valid[num_cols_t].fillna(0.0)

    y_train_vals = y_train.values.astype(np.float64)
    y_train_t = np.sign(y_train_vals) * np.log1p(np.abs(y_train_vals))

    reg = make_regressor()
    reg.fit(X_train, y_train_t)

    bias = 0.0
    if X_valid is not None and len(X_valid) > 0:
        pred_valid_t = reg.predict(X_valid).astype(np.float64)
        pred_valid = np.sign(pred_valid_t) * np.expm1(np.abs(pred_valid_t))
        resid = y_valid.values.astype(np.float64) - pred_valid.astype(np.float64)
        bias = (
            float(np.median(resid[np.isfinite(resid)]))
            if np.any(np.isfinite(resid))
            else 0.0
        )

        pred_valid_bc = pred_valid + bias
        mae_t = float(np.mean(np.abs(y_valid.values - pred_valid_bc)))
        cv_mae_by_type[t] = mae_t

    test_mask = (test["type"] == t).values
    if np.any(test_mask):
        X_test_t = test_X_full.loc[test_mask, :].copy()
        X_test_t[num_cols_t] = X_test_t[num_cols_t].fillna(fill_values_t)
        X_test_t[num_cols_t] = X_test_t[num_cols_t].fillna(0.0)

        pred_t_t = reg.predict(X_test_t).astype(np.float64)
        pred_t = np.sign(pred_t_t) * np.expm1(np.abs(pred_t_t))
        pred_t = pred_t + bias

        pred_t = np.where(np.isfinite(pred_t), pred_t, np.nan)
        pred_t = np.nan_to_num(pred_t, nan=np.nanmedian(pred_t))
        pred_test_full[test_mask] = pred_t

if len(cv_mae_by_type) > 0:
    cv_metric = float(np.mean(np.log(np.array(list(cv_mae_by_type.values())) + 1e-12)))
    print("Approx CV metric (mean log MAE over types with valid split):", cv_metric)

test_pred_df = pd.DataFrame(
    {"id": test["id"].values, "scalar_coupling_constant": pred_test_full}
)
submission = sub[["id"]].merge(test_pred_df, on="id", how="left")

if submission["scalar_coupling_constant"].isna().any():
    fallback = float(np.nanmedian(submission["scalar_coupling_constant"].values))
    submission["scalar_coupling_constant"] = submission[
        "scalar_coupling_constant"
    ].fillna(fallback)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
