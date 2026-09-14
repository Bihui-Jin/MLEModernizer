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
lightgbm==4.6.0
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

0.91797

# 6. Current score

1.25506

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39512) has done: 'Diagnosis: The crash occurs because LightGBM 4.6.0 removed/deprecated the `early_stopping_rounds` and `verbose_eval` keyword arguments from `lgb.train()`. In this version, early stopping and logging must be configured via `callbacks`. The rest of the training call (datasets, params, custom feval, and prediction using `best_iteration`) can remain the same.

Patch summary: Replace `early_stopping_rounds=20` and `verbose_eval=20` with the equivalent `callbacks=[lgb.early_stopping(20), lgb.log_evaluation(20)]` inside the `lgb.train()` call. This keeps the same training semantics (early stopping after 20 rounds without improvement and logging every 20 iterations) while matching the LightGBM 4.6.0 API.

Updated cells: Below is the updated cell 2 only.

Compatibility notes for cell k+1: `model` is still a LightGBM Booster with `best_iteration` set by early stopping, so `model.predict(..., num_iteration=model.best_iteration)` remains compatible.

Assumptions: LightGBM callbacks `early_stopping` and `log_evaluation` are available in `lightgbm==4.6.0` (they are).'
- What this solution (achieved 1.39114) has done: 'The crash happens because `Dataset.set_group()` in LightGBM expects *query sizes* (group counts that sum to `num_data`), but the code passes per-row `type` labels, so LightGBM raises “Sum of query counts differs from the length of #data”. The intended behavior for the custom metric is to access a per-row “type code” vector; LightGBM’s `group` field is not appropriate for that. The minimal fix is to stop using `set_group()` and instead attach the per-row type codes as a custom dataset field (via a monkey-patched `get_group()` method) so `lgb_champs_metric()` continues to work unchanged. This keeps training/inference semantics identical while unblocking execution.'
- What this solution (achieved 1.27901) has done: 'Your current approach trains a single regressor across all coupling types, but the evaluation averages log(MAE) per type; that mismatch is a major reason the score is far from the target. With minimal core-logic change (still LightGBM regression, same feature set, same metric), we train one LightGBM model per `type` using the same molecule-based split within each type, then stitch predictions back together by `id`. This aligns training with the evaluation semantics and should substantially reduce the score toward the 0.91797 target. I also make the `dist_to_type_mean` normalization use train-derived type means for both train/test (avoids test-dependent normalization that can hurt generalization and stability).'
- What this solution (achieved 1.27013) has done: 'Your current score (1.27901, lower is better) is far from the target (0.91797), so we should make a small but meaningful improvement without changing the core LightGBM-per-type approach. The biggest issue is that your custom metric isn’t actually computing per-type log(MAE) because `dvalid.get_group()` is set to all zeros; we fix this by passing the real per-row type codes into both train/valid datasets via a minimal monkey-patch, so early stopping selects iterations aligned with the competition metric. We also switch the split to be deterministic and consistent by using `GroupShuffleSplit` (still molecule-based) and add a tiny amount of regularization (`min_data_in_leaf`, `feature_fraction`, `bagging_fraction`, `bagging_freq`) to reduce overfit and move the score downward toward the target while keeping the same model family and training loop. All I/O paths and submission schema remain unchanged.'
- What this solution (achieved 1.25506) has done: 'We make the custom metric actually compute the competition’s per-type log(MAE) by attaching the correct per-row `type` codes (instead of all zeros) to each LightGBM Dataset; this makes early stopping select iterations that better match Kaggle’s evaluation and should reduce the score toward your target. We also switch the objective to `regression_l1` (still standard regression in LightGBM) so training optimizes MAE, which aligns better with the metric than squared error and typically improves this competition without changing the overall approach. Finally, we keep the molecule-based split and per-type training loop intact, and preserve the exact submission format/path.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn import *
import lightgbm as lgb

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sub = pd.read_csv("../input/sample_submission.csv")
print(train.shape, test.shape, sub.shape)

train["atom"] = train["type"].map(lambda x: str(x)[3])
test["atom"] = test["type"].map(lambda x: str(x)[3])

lbl = preprocessing.LabelEncoder()
for i in range(4):
    train["type" + str(i)] = lbl.fit_transform(train["type"].map(lambda x: str(x)[i]))
    test["type" + str(i)] = lbl.transform(test["type"].map(lambda x: str(x)[i]))

structures = pd.read_csv("../input/structures.csv").rename(
    columns={"atom_index": "atom_index_0", "x": "x0", "y": "y0", "z": "z0"}
)
train = pd.merge(
    train, structures, how="left", on=["molecule_name", "atom_index_0", "atom"]
)
test = pd.merge(
    test, structures, how="left", on=["molecule_name", "atom_index_0", "atom"]
)
del structures

structures = pd.read_csv("../input/structures.csv").rename(
    columns={"atom_index": "atom_index_1", "x": "x1", "y": "y1", "z": "z1"}
)
train = pd.merge(
    train, structures, how="left", on=["molecule_name", "atom_index_1", "atom"]
)
test = pd.merge(
    test, structures, how="left", on=["molecule_name", "atom_index_1", "atom"]
)
del structures
print(train.shape, test.shape, sub.shape)



## === cell 1
train_p0 = train[["x0", "y0", "z0"]].fillna(0).values
train_p1 = train[["x1", "y1", "z1"]].fillna(0).values
test_p0 = test[["x0", "y0", "z0"]].fillna(0).values
test_p1 = test[["x1", "y1", "z1"]].fillna(0).values

train["dist"] = np.linalg.norm(train_p0 - train_p1, axis=1)
test["dist"] = np.linalg.norm(test_p0 - test_p1, axis=1)

type_dist_mean = train.groupby("type")["dist"].mean()
train["dist_to_type_mean"] = train["dist"] / train["type"].map(type_dist_mean)
test["dist_to_type_mean"] = test["dist"] / test["type"].map(type_dist_mean)
train["dist_to_type_mean"] = (
    train["dist_to_type_mean"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
)
test["dist_to_type_mean"] = (
    test["dist_to_type_mean"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
)



## === cell 2
col = [
    c
    for c in train.columns
    if c not in ["id", "molecule_name", "scalar_coupling_constant", "type", "atom"]
]


def lgb_champs_metric(preds, dtrain):
    labels = dtrain.get_label()
    type_idx = dtrain.get_group()
    if type_idx is None or len(type_idx) != len(labels):
        score = float(np.log(metrics.mean_absolute_error(labels, preds)))
        return "champs_lmae", score, False

    scores = []
    for t in np.unique(type_idx):
        m = type_idx == t
        mae = metrics.mean_absolute_error(labels[m], preds[m])
        scores.append(np.log(mae))
    return "champs_lmae", float(np.mean(scores)), False


params = {
    "boosting_type": "gbdt",
    "objective": "regression_l1",
    "metric": "None",
    "learning_rate": 0.2,
    "num_leaves": 64,
    "min_data_in_leaf": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 99,
}


def attach_row_group(ds, row_group_vec):
    ds._row_group = np.asarray(row_group_vec, dtype=np.int32)

    def _get_group_override(self):
        return getattr(self, "_row_group", None)

    ds.get_group = _get_group_override.__get__(ds, type(ds))
    return ds


type_to_code = {t: i for i, t in enumerate(sorted(train["type"].unique()))}

test_pred = np.zeros(len(test), dtype=np.float64)

types = train["type"].unique()
for t in sorted(types):
    tr_t = train[train["type"] == t].copy()
    te_t = test[test["type"] == t].copy()

    gss = model_selection.GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=99)
    split = next(gss.split(tr_t, groups=tr_t["molecule_name"].values))
    trn_idx, val_idx = split[0], split[1]

    x_tr = tr_t.iloc[trn_idx][col]
    y_tr = tr_t.iloc[trn_idx]["scalar_coupling_constant"].values
    x_va = tr_t.iloc[val_idx][col]
    y_va = tr_t.iloc[val_idx]["scalar_coupling_constant"].values

    dtrain = lgb.Dataset(x_tr, label=y_tr)
    dvalid = lgb.Dataset(x_va, label=y_va, reference=dtrain)

    t_code = type_to_code[t]
    attach_row_group(dtrain, np.full(len(y_tr), fill_value=t_code, dtype=np.int32))
    attach_row_group(dvalid, np.full(len(y_va), fill_value=t_code, dtype=np.int32))

    model = lgb.train(
        params,
        dtrain,
        400,
        valid_sets=[dvalid],
        feval=lgb_champs_metric,
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
    )

    te_pred_t = model.predict(te_t[col], num_iteration=model.best_iteration)
    test_pred[test["type"].values == t] = te_pred_t

test["scalar_coupling_constant"] = test_pred
test[["id", "scalar_coupling_constant"]].to_csv(
    "submission.csv", float_format="%.9f", index=False
)
print(
    "Wrote submission.csv with shape:", test[["id", "scalar_coupling_constant"]].shape
)
