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

1.19404

# 6. Current score

2.73098

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.85222) has done: 'Diagnosis: The crash happens in cell 17 when calling `model.predict(...)` because the feature matrix built from `testSet[...]` contains NaNs. These NaNs come from earlier feature engineering: some rows in `testSet` have missing engineered columns (most commonly `dist_to_type_mean` due to `groupby('type')` producing NaN means for types not present/fully formed, or missing coordinates from merges). `HuberRegressor` cannot handle NaNs, so prediction fails.  
Patch summary: In cell 17 only, build the test feature matrix and then deterministically replace any NaNs (and infs) with training-derived feature medians (or 0.0 if a feature is entirely NaN), preserving the exact feature set/order used in training. This keeps the model and core logic unchanged while ensuring `predict` receives finite values.  
Updated cells: Only cell 17 is modified.  
Compatibility notes for cell k+1: `resultSet` remains a DataFrame with columns `id` and `scalar_coupling_constant`, so cell 18 continues to work unchanged.  
Assumptions: Using training feature medians for imputation is acceptable as minimal preprocessing to satisfy the estimator’s finite-input requirement; it does not change the model architecture or training procedure.'
- What this solution (achieved 1.85222) has done: 'We make one metric-aligned change that should improve the leaderboard score without changing the model or features: compute `dist_to_type_mean` in the test set using the *training* per-type mean distances (rather than the test’s own per-type mean), which removes a train/test mismatch and keeps the feature definition consistent. We keep your existing NaN/inf handling (needed for HuberRegressor) and only adjust how `dist_to_type_mean` is built so it becomes comparable between train and test. This is a minimal change that should reduce error (lower is better) and move the score toward your target. The script still run end-to-end and write `results.csv` in the required submission format.'
- What this solution (achieved 1.59949) has done: 'We make one minimal, metric-aligned improvement to reduce the current gap (1.85222 → target 1.19404, lower is better) without changing your model, features, or training loop: fit the `HuberRegressor` with `sample_weight` so each coupling `type` contributes equally, matching the competition’s per-type averaged log-MAE objective. This keeps the same estimator and inputs, but shifts optimization away from the majority types that otherwise dominate the loss and typically improves the per-type averaged score. We keep your existing train-derived `dist_to_type_mean` mapping and NaN/inf handling so prediction remains robust. The script still runs end-to-end and writes `results.csv` with the required columns.'
- What this solution (achieved 1.54143) has done: 'Your current gap to the target is large (1.59949 vs 1.19404, lower is better), so we need a small but meaningful accuracy improvement without changing the model or feature set. The biggest issue left is that HuberRegressor is being fit on raw, differently-scaled features, which can hurt convergence and coefficient balance; adding a `StandardScaler` in a sklearn `Pipeline` keeps the exact same model and features but typically reduces MAE for linear models. I keep your per-type sample weighting (metric-aligned) and your train-derived `dist_to_type_mean` mapping and NaN/inf handling, just ensuring the same scaling is applied at predict time. The script still run end-to-end and write `results.csv` in the required format.'
- What this solution (achieved 1.54422) has done: 'We make one minimal, score-improving change that keeps your model/features/training loop identical: add a per-type intercept correction computed on the training data and applied to both train/test predictions. This is a lightweight calibration that often reduces the competition’s per-type MAE (and thus log-MAE) without altering the estimator, feature extraction, or loss. We compute the correction using out-of-fold predictions grouped by molecule to avoid leakage (since the split is by molecule), then refit the same model on all data and apply the learned per-type offsets at inference. Everything else (including your NaN/inf handling and submission format) remains unchanged.'
- What this solution (achieved 2.72215) has done: 'Your current score (1.54422) is worse than the target (1.19404), so we should improve accuracy with a minimal, metric-aligned change. The biggest remaining mismatch with the competition metric is that a single global linear model is trying to fit multiple coupling `type`s with very different target scales; keeping the same model and features, we can instead train the same HuberRegressor *separately per type* and predict per type, which usually reduces per-type MAE and thus the averaged log-MAE. This preserves your feature engineering and estimator, but removes cross-type interference while still using your per-type sample weighting (which becomes constant inside each type). We also keep your molecule-grouped OOF type-offset calibration, now computed within each type to remain leakage-safe and consistent with the per-type models.'
- What this solution (achieved 2.72195) has done: 'We should move your score down (lower is better) with the smallest change that improves per-type accuracy without altering your feature set or model family. Right now, your per-type models in cell 17 are trained without the metric-aligned per-type sample weighting you already used globally, and your test fallback uses the global model trained only once earlier (before per-type training). I (1) apply the same per-type-equalizing `sample_weight` inside each per-type model fit (so the robust regression is less dominated by large-magnitude targets even within a type), and (2) refit a dedicated global fallback model on the fully-imputed `X_all` (to match the exact preprocessing used for the per-type models), which stabilizes unseen/edge cases. These are minimal, metric-aligned changes and should improve the score toward 1.19404 without changing features, estimator, or training approach.'
- What this solution (achieved 2.72195) has done: 'Your current score (2.72195, lower is better) is far from the target (1.19404), so we should improve accuracy with the smallest change that keeps your model/features/training semantics intact. The biggest correctness bug in your current code is that `pred_test` is preallocated with `np.empty`, so any row whose type is present but still not assigned (or any edge-case ordering) can retain uninitialized garbage values; `pd.isna` won’t reliably catch those, causing catastrophic submission errors. I initialize `pred_test` with `np.nan` so every unpredicted row is deterministically caught and filled by the fallback model. Additionally, I make test feature columns explicitly `float64` after NaN handling so the sklearn `Pipeline(StandardScaler+HuberRegressor)` always receives numeric arrays (avoids silent object-dtype issues that can degrade fits/preds).'
- What this solution (achieved 2.72195) has done: 'Your current score (2.72195, lower is better) is far worse than the target (1.19404), and the biggest remaining issue is that the per-type out-of-fold offset calibration in cell 17 is currently unreliable because `oof_t` is created with `np.empty`, so any samples not assigned (e.g., due to split edge-cases) can keep garbage values and corrupt `type_offsets`. I make this deterministic by initializing `oof_t` with `np.nan`, asserting all validation rows are filled, and computing the offset only from filled OOF predictions; this preserves the same model, features, and training approach but removes a major source of catastrophic error. I also make GroupKFold use `n_splits=min(3, n_groups)` so the split is always valid per type and doesn’t silently degrade behavior. These changes are minimal, metric-aligned, and should move the score down toward the target while keeping the rest of your pipeline identical and still producing `results.csv`.'
- What this solution (achieved 2.72738) has done: 'You’re far above the target (2.72195 vs 1.19404, lower is better), so we need a real accuracy fix but still minimal and within your existing approach. The main issue is that your “type offset” is calibrated as `median(y - oof_pred)` but then applied as `pred + offset`; this uses the wrong sign and systematically worsens predictions per type. I flip the sign consistently by defining the offset as `median(oof_pred - y)` and still applying it as `pred + offset` (equivalently `pred - median(y-oof)`), keeping your same models, features, splits, and training. Everything else (NaN handling, per-type models, fallback, and submission writing) stays unchanged.'
- What this solution (achieved 2.72957) has done: 'We make two minimal, metric-aligned fixes that should move the score down toward the target without changing your feature set or model family. First, we ensure per-type OOF offset calibration uses the same preprocessing as inference by recomputing `dist_to_type_mean` for test with a safe fallback (global mean) and by applying the same NaN/inf handling consistently before any per-type split. Second, we stabilize the per-type offset by using a mean (not median) on the OOF residual direction already corrected, which typically reduces MAE under this competition metric while keeping the same calibration idea. Everything else (HuberRegressor, StandardScaler pipeline, per-type models, GroupKFold by molecule, submission writing) remains the same.'
- What this solution (achieved 2.73098) has done: 'Your current score (2.72957) is much worse than the target (1.19404, lower is better), and the most likely cause is that the per-type “offset calibration” is adding a bias in the wrong direction and/or computed in a way that destabilizes predictions. I make the smallest correction to the offset definition so it matches the way you apply it at inference (i.e., compute an additive bias that maps predictions toward targets), while keeping the same per-type models, GroupKFold-by-molecule OOF scheme, features, and HuberRegressor pipeline. I also make the type list deterministic (sorted) to remove run-to-run variability that can cause score swings. Everything else (feature engineering, scaling, model family, training approach, and submission writing) remains unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import HuberRegressor
from sklearn import metrics

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GroupKFold

import os

print(os.listdir("../input"))



## === cell 1
trainSet = pd.read_csv("../input/train.csv")
display(trainSet.head())



## === cell 2
testSet = pd.read_csv("../input/test.csv")
display(testSet.head())



## === cell 3
structures = pd.read_csv("../input/structures.csv")
display(structures.head())




## === cell 4
def map_atom_info(df, atom_idx):
    df = pd.merge(
        df,
        structures,
        how="left",
        left_on=["molecule_name", f"atom_index_{atom_idx}"],
        right_on=["molecule_name", "atom_index"],
    )

    df = df.drop("atom_index", axis=1)
    df = df.rename(
        columns={
            "atom": f"atom_{atom_idx}",
            "x": f"x_{atom_idx}",
            "y": f"y_{atom_idx}",
            "z": f"z_{atom_idx}",
        }
    )
    return df


trainSet = map_atom_info(trainSet, 0)
trainSet = map_atom_info(trainSet, 1)

testSet = map_atom_info(testSet, 0)
testSet = map_atom_info(testSet, 1)



## === cell 5
display(trainSet.head())
display(testSet.head())



## === cell 6
train_p0 = trainSet[["x_0", "y_0", "z_0"]].values
train_p1 = trainSet[["x_1", "y_1", "z_1"]].values
test_p0 = testSet[["x_0", "y_0", "z_0"]].values
test_p1 = testSet[["x_1", "y_1", "z_1"]].values

trainSet["dist"] = np.linalg.norm(train_p0 - train_p1, axis=1)
testSet["dist"] = np.linalg.norm(test_p0 - test_p1, axis=1)

trainSet["dist_to_type_mean"] = trainSet["dist"] / trainSet.groupby("type")[
    "dist"
].transform("mean")

type_mean_dist_train = trainSet.groupby("type")["dist"].mean()
global_mean_dist_train = float(trainSet["dist"].mean())

den = testSet["type"].map(type_mean_dist_train).astype(np.float64)
den = den.fillna(global_mean_dist_train)
den = den.replace(0.0, global_mean_dist_train)  # avoid infs in rare degenerate cases
testSet["dist_to_type_mean"] = testSet["dist"].astype(np.float64) / den



## === cell 7
assert trainSet["atom_0"].eq("H").all()

if not testSet["atom_0"].eq("H").all():
    print(
        "Warning: testSet['atom_0'] is not always 'H'. "
        f"Found categories: {list(testSet['atom_0'].astype('category').cat.categories)}"
    )



## === cell 8
print(trainSet["atom_1"].astype("category").cat.categories)
print(testSet["atom_1"].astype("category").cat.categories)



## === cell 9
print(testSet["type"].astype("category").cat.categories)
print(trainSet["type"].astype("category").cat.categories)



## === cell 10
for i in trainSet["type"].astype("category").cat.categories.values:
    trainSet["type_" + str(i)] = trainSet["type"] == i
    testSet["type_" + str(i)] = testSet["type"] == i



## === cell 11
base_model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("huber", HuberRegressor()),
    ]
)



## === cell 12
type_counts = trainSet["type"].value_counts()
sample_weight = (
    trainSet["type"].map(lambda t: 1.0 / type_counts.loc[t]).astype(np.float64).values
)

feature_cols = [
    "type_1JHC",
    "type_1JHN",
    "type_2JHC",
    "type_2JHH",
    "type_2JHN",
    "type_3JHC",
    "type_3JHH",
    "dist",
    "dist_to_type_mean",
]

X_train = np.array(trainSet[feature_cols])

fitDist = base_model.fit(
    X_train,
    trainSet["scalar_coupling_constant"],
    huber__sample_weight=sample_weight,  # keep your metric-aligned weighting
)



## === cell 13
fitDist.named_steps["huber"].coef_




## === cell 14
def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    maes = (y_true - y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()




## === cell 15
group_mean_log_mae(
    trainSet["scalar_coupling_constant"],
    base_model.predict(np.array(trainSet[feature_cols])),
    trainSet["type"],
)



## === cell 16
print(
    group_mean_log_mae(
        trainSet["scalar_coupling_constant"],
        trainSet["scalar_coupling_constant"].median(),
        trainSet["type"],
    )
)
print(group_mean_log_mae(trainSet["scalar_coupling_constant"], 0.85, trainSet["type"]))



## === cell 17
X_all = trainSet[feature_cols].copy()
X_all = X_all.replace([np.inf, -np.inf], np.nan)

train_medians = X_all.median(numeric_only=True).fillna(0.0)
X_all = X_all.fillna(train_medians)

X_all = X_all.astype(np.float64)

y_all = trainSet["scalar_coupling_constant"].astype(np.float64).values
types_all = trainSet["type"].astype(str).values
groups_all = trainSet["molecule_name"].astype(str).values

unique_types = sorted(pd.Series(types_all).unique().tolist())

models_by_type = {}
type_offsets = {}

type_counts_all = pd.Series(types_all).value_counts()
sw_all = (
    pd.Series(types_all)
    .map(lambda t: 1.0 / type_counts_all.loc[t])
    .astype(np.float64)
    .values
)

for t in unique_types:
    idx_t = np.where(types_all == t)[0]
    X_t = X_all.iloc[idx_t]
    y_t = y_all[idx_t]
    g_t = groups_all[idx_t]
    sw_t = sw_all[idx_t]

    n_groups = pd.Series(g_t).nunique()
    oof_t = np.full(len(idx_t), np.nan, dtype=np.float64)

    if n_groups >= 2:
        n_splits = min(3, int(n_groups))
        gkf = GroupKFold(n_splits=n_splits)

        for tr_i, va_i in gkf.split(X_t, y_t, groups=g_t):
            m = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("huber", HuberRegressor()),
                ]
            )
            m.fit(
                np.array(X_t.iloc[tr_i]),
                y_t[tr_i],
                huber__sample_weight=sw_t[tr_i],
            )
            oof_t[va_i] = m.predict(np.array(X_t.iloc[va_i]))

        filled = ~np.isnan(oof_t)
        assert (
            filled.all()
        ), f"OOF predictions not fully assigned for type={t} (filled {filled.mean():.3f})"

        type_offsets[t] = float(np.mean(y_t - oof_t))
    else:
        type_offsets[t] = 0.0

    m_full = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("huber", HuberRegressor()),
        ]
    )
    m_full.fit(np.array(X_t), y_t, huber__sample_weight=sw_t)
    models_by_type[t] = m_full

fallback_model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("huber", HuberRegressor()),
    ]
)
fallback_model.fit(np.array(X_all), y_all, huber__sample_weight=sw_all)



## === cell 18
X_test = testSet[feature_cols].copy()
X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_test = X_test.fillna(train_medians)

X_test = X_test.astype(np.float64)

pred_test = np.full(len(testSet), np.nan, dtype=np.float64)

test_types = testSet["type"].astype(str).values

for t, m in models_by_type.items():
    idx = np.where(test_types == t)[0]
    if len(idx) == 0:
        continue
    p = m.predict(np.array(X_test.iloc[idx]))
    p = p + float(type_offsets.get(t, 0.0))
    pred_test[idx] = p

unseen_idx = np.where(np.isnan(pred_test))[0]
if len(unseen_idx) > 0:
    pred_test[unseen_idx] = fallback_model.predict(np.array(X_test.iloc[unseen_idx]))

resultSet = pd.DataFrame({"id": testSet["id"], "scalar_coupling_constant": pred_test})



## === cell 19
resultSet.to_csv("results.csv", index=False, header=True)



## === cell 20
with open("results.csv", "r") as f:
    for i, line in enumerate(f):
        print(line)
        if i > 5:
            break
