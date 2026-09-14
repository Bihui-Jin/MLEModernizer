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

1.69586

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39669) has done: 'I fix the LightGBM v4.6 API break causing `early_stopping_rounds` to error by switching to the supported callback-based early stopping and logging, keeping the same training setup and objective. I also make the train/valid split molecule-wise (as the competition split is by molecule) to avoid leakage and improve the score in a legitimate way while preserving the same model/feature logic. Finally, I make the `dist_to_type_mean` computation robust for test (no group leakage/NaNs) by using train type means for both train and test. The script run end-to-end and write `submission.csv` with the required columns.'
- What this solution (achieved 2.92743) has done: 'I fix the LightGBM runtime error by ensuring all features passed into `lgb.Dataset()` are numeric, by label-encoding the `atom0` and `atom1` columns (the only non-numeric columns in `col`). I do this in-place for both train and test with a single encoder per column to keep train/test mappings consistent and preserve the existing model/training logic. I also add a small safety fill for any remaining NaNs in the feature matrix before training/prediction so LightGBM won’t error on missing values represented as object/nullable types. This should let the notebook run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 1.69851) has done: 'Your current score is much worse than the target (lower-is-better), and the biggest (still minimal) win without changing the core model is to align training/validation and modeling with the competition’s metric being averaged per `type`. I keep the same LightGBM regressor and features, but train one model per coupling `type` (same params/rounds/early stopping) so each type can fit its own scale/relationship, which typically reduces the per-type MAE and therefore the averaged log-MAE. I also compute `dist_to_type_mean` using only the training fold within each type to avoid subtle leakage into validation and keep validation feedback honest while still using full-train type means for the final test feature. Finally, I ensure predictions are written back in the original test row order into a valid `submission.csv`.'
- What this solution (achieved 1.69851) has done: 'You’re currently far from the target (lower-is-better), so the smallest reliable way to move the score down without changing the core model/feature set is to fix two issues that hurt generalization: (1) the `type0..type3` encoding is fit on train and then transformed on test, which can mis-encode unseen characters and destabilize splits; I fit the encoders on combined train+test so the mapping is consistent. (2) you’re training one model per `type` but you still include the `type0..type3` features inside each per-type model; those are constant within that subset and can add noise/overfitting, so I drop only those columns from `col` (keeping the rest identical). I also compute `dist_to_type_mean` for test using the training-fold type means (same idea you already applied to validation) to avoid using full-train stats that include validation molecules, which typically improves honest generalization a bit while preserving semantics. The rest (LightGBM params, per-type training loop, early stopping, features) stays the same.'
- What this solution (achieved 1.69674) has done: 'You’re far above the target (lower-is-better), so we should improve generalization with the smallest semantic change: keep the same LightGBM per-type training loop and the same features, but make `dist_to_type_mean` computed strictly from the training split within each coupling `type` (your current code accidentally uses validation rows when building the denominator for validation, which weakens the feature and can hurt per-type MAE). I also fix the test `dist_to_type_mean` to use full-train per-type means (after training) rather than split-only means so test rows for types with sparse train-split support don’t get zeroed out as often. Finally, I add `seed`/`feature_fraction_seed`/`bagging_seed` for determinism (no intended score boost, just stability while we move toward the target).'
- What this solution (achieved 1.71449) has done: 'I make two minimal, score-relevant fixes without changing your feature set or the per-type LightGBM training approach. First, your validation `dist_to_type_mean` is currently mapped using `train.loc[val_idx, "type"]` (valid types) instead of mapping with the training-fold type means on the valid rows; this misalignment can weaken that feature and hurt generalization, so I correct the mapping to use `train.loc[val_idx, "type"].map(type_mean_trn)`. Second, I ensure LightGBM doesn’t overfit each per-type subset by enabling bagging via `bagging_fraction` and `bagging_freq` (core model stays GBDT with same objective/metric/rounds), which typically reduces MAE and should move your score down toward the target. Everything else (data, merges, per-type loop, early stopping, submission writing) stays the same.'
- What this solution (achieved 1.69586) has done: 'Your current score (1.71449, lower-is-better) is still far from the target (0.91797), so we should make small, legitimate generalization improvements without changing the core per-type LightGBM approach or feature set. The biggest low-risk win here is to reduce leakage/overfitting by computing `dist_to_type_mean` strictly per-`type` using only the training molecules (not mixing types and not using validation molecules), and to use those per-type means for validation and test mapping. Second, we keep your same params but add minimal regularization (`min_data_in_leaf`, `lambda_l2`) which usually reduces MAE on this competition without changing the modeling approach. Finally, we keep everything else intact and still write a valid `submission.csv`.'
- What this solution (achieved 1.69586) has done: 'You’re still far above the target (lower-is-better), so the most score-relevant minimal change is to fix the remaining leakage/feature-mismatch around `dist_to_type_mean`: right now validation uses training-fold means, but test uses full-train means while the model was early-stopped on the split-based feature distribution, creating a train/valid/test shift that hurts MAE. I compute and use a consistent `dist_to_type_mean` for train/valid/test based only on the training-molecules fold (and safely fall back to full-train means only for types absent in the training fold). I keep the same per-type LightGBM training loop, params, rounds, and features otherwise, and ensure the submission format and row alignment remain correct. This should move the score down (better) toward your target without changing the core approach.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn import *
import lightgbm as lgb

BASE_PATH = "/kaggle/data/champs-scalar-coupling"

train = pd.read_csv(f"{BASE_PATH}/train.csv")
test = pd.read_csv(f"{BASE_PATH}/test.csv")
sub = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")
print(train.shape, test.shape, sub.shape)

for i in range(4):
    lbl = preprocessing.LabelEncoder()
    all_chars = pd.concat([train["type"], test["type"]], axis=0).map(
        lambda x: str(x)[i]
    )
    lbl.fit(all_chars.values)
    train["type" + str(i)] = lbl.transform(train["type"].map(lambda x: str(x)[i]))
    test["type" + str(i)] = lbl.transform(test["type"].map(lambda x: str(x)[i]))

structures = pd.read_csv(f"{BASE_PATH}/structures.csv")

s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)
train = pd.merge(
    train,
    s0[["molecule_name", "atom_index_0", "atom0", "x0", "y0", "z0"]],
    how="left",
    on=["molecule_name", "atom_index_0"],
)
test = pd.merge(
    test,
    s0[["molecule_name", "atom_index_0", "atom0", "x0", "y0", "z0"]],
    how="left",
    on=["molecule_name", "atom_index_0"],
)

s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)
train = pd.merge(
    train,
    s1[["molecule_name", "atom_index_1", "atom1", "x1", "y1", "z1"]],
    how="left",
    on=["molecule_name", "atom_index_1"],
)
test = pd.merge(
    test,
    s1[["molecule_name", "atom_index_1", "atom1", "x1", "y1", "z1"]],
    how="left",
    on=["molecule_name", "atom_index_1"],
)

del structures, s0, s1
print(train.shape, test.shape, sub.shape)



## === cell 1
train_p0 = train[["x0", "y0", "z0"]].fillna(0).values
train_p1 = train[["x1", "y1", "z1"]].fillna(0).values
test_p0 = test[["x0", "y0", "z0"]].fillna(0).values
test_p1 = test[["x1", "y1", "z1"]].fillna(0).values

train["dist"] = np.linalg.norm(train_p0 - train_p1, axis=1)
test["dist"] = np.linalg.norm(test_p0 - test_p1, axis=1)

type_mean_full = train.groupby("type")["dist"].mean()
train["dist_to_type_mean"] = train["dist"] / train["type"].map(type_mean_full)
test["dist_to_type_mean"] = test["dist"] / test["type"].map(type_mean_full)

train["dist_to_type_mean"] = (
    train["dist_to_type_mean"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
)
test["dist_to_type_mean"] = (
    test["dist_to_type_mean"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
)



## === cell 2
for c in ["atom0", "atom1"]:
    le = preprocessing.LabelEncoder()
    all_vals = pd.concat([train[c], test[c]], axis=0).astype(str).fillna("nan")
    le.fit(all_vals.values)
    train[c] = le.transform(train[c].astype(str).fillna("nan"))
    test[c] = le.transform(test[c].astype(str).fillna("nan"))

col = [
    c
    for c in train.columns
    if c not in ["id", "molecule_name", "scalar_coupling_constant", "type"]
]

drop_const_type_feats = {"type0", "type1", "type2", "type3"}
col = [c for c in col if c not in drop_const_type_feats]


def lgb_lmae(preds, dtrain):
    labels = dtrain.get_label()
    score = np.log(metrics.mean_absolute_error(labels, preds))
    return "lmae", score, False


params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.2,
    "num_leaves": 64,
    "seed": 99,
    "feature_fraction_seed": 99,
    "bagging_seed": 99,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "lambda_l2": 1.0,
}

rng = np.random.RandomState(99)
mols = train["molecule_name"].unique()
rng.shuffle(mols)
cut = int(len(mols) * 0.8)
train_mols = set(mols[:cut])
valid_mols = set(mols[cut:])

trn_idx = train["molecule_name"].isin(train_mols)
val_idx = train["molecule_name"].isin(valid_mols)

type_mean_trn = train.loc[trn_idx].groupby("type")["dist"].mean()
type_mean_full = train.groupby("type")["dist"].mean()
type_mean_trn_safe = type_mean_trn.combine_first(type_mean_full)

train.loc[trn_idx, "dist_to_type_mean"] = train.loc[trn_idx, "dist"] / train.loc[
    trn_idx, "type"
].map(type_mean_trn_safe)
train.loc[val_idx, "dist_to_type_mean"] = train.loc[val_idx, "dist"] / train.loc[
    val_idx, "type"
].map(type_mean_trn_safe)
test["dist_to_type_mean"] = test["dist"] / test["type"].map(type_mean_trn_safe)

train["dist_to_type_mean"] = (
    train["dist_to_type_mean"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
)
test["dist_to_type_mean"] = (
    test["dist_to_type_mean"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
)

x_test_full = test[col].copy()
for c in x_test_full.columns:
    if not (
        np.issubdtype(x_test_full[c].dtype, np.number) or x_test_full[c].dtype == bool
    ):
        x_test_full[c] = pd.to_numeric(x_test_full[c], errors="coerce")
x_test_full.fillna(0.0, inplace=True)

test_pred = np.zeros(len(test), dtype=np.float64)

types = train["type"].unique()
print("Training per type models:", len(types))

for t in sorted(types):
    trn_mask_t = trn_idx & (train["type"] == t)
    val_mask_t = val_idx & (train["type"] == t)
    tst_mask_t = test["type"] == t

    if tst_mask_t.sum() == 0:
        continue

    x1 = train.loc[trn_mask_t, col].copy()
    x2 = train.loc[val_mask_t, col].copy()
    y1 = train.loc[trn_mask_t, "scalar_coupling_constant"]
    y2 = train.loc[val_mask_t, "scalar_coupling_constant"]

    for df in (x1, x2):
        for cc in df.columns:
            if not (np.issubdtype(df[cc].dtype, np.number) or df[cc].dtype == bool):
                df[cc] = pd.to_numeric(df[cc], errors="coerce")
        df.fillna(0.0, inplace=True)

    dtrain = lgb.Dataset(x1, label=y1)
    dvalid = lgb.Dataset(x2, label=y2, reference=dtrain) if len(x2) > 0 else None

    valid_sets = [dvalid] if dvalid is not None else []
    valid_names = ["valid"] if dvalid is not None else []

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=400,
        valid_sets=valid_sets,
        valid_names=valid_names,
        feval=lgb_lmae,
        callbacks=[
            (
                lgb.early_stopping(
                    stopping_rounds=20, first_metric_only=False, verbose=True
                )
                if dvalid is not None
                else lgb.log_evaluation(period=0)
            ),
            lgb.log_evaluation(period=50),
        ],
    )

    test_pred[tst_mask_t.values] = model.predict(
        x_test_full.loc[tst_mask_t.values, :],
        num_iteration=getattr(model, "best_iteration", None) or 400,
    )

test["scalar_coupling_constant"] = test_pred

submission = test[["id", "scalar_coupling_constant"]].copy()
submission = submission.sort_values("id").reset_index(drop=True)
submission.to_csv("submission.csv", float_format="%.9f", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
