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

-1.31958

# 6. Current score

2.99778

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.99777) has done: 'The crash happens because cell 1 tries to read three submission CSVs from other Kaggle notebook datasets under `../input/...`, but those folders don’t exist in this environment (only the `champs-scalar-coupling` competition data is available). To keep the notebook runnable and preserve the downstream interface, the minimal fix is to load the available `sample_submission.csv` three times and assign it to `sub1`, `sub2`, and `sub3` (so the expected `scalar_coupling_constant` column exists). This avoids changing any later logic while ensuring the `.describe()` calls and the arithmetic in cell 2 work deterministically.'
- What this solution (achieved 1.18497) has done: 'Your current code is only averaging three identical `sample_submission.csv` files, which keeps every prediction at 0 and yields a poor score. To move the score toward the (much better) target, the smallest legitimate change is to replace the placeholder predictions with a simple, type-wise constant baseline learned from `train.csv` (median per coupling `type`), then apply it to `test.csv`. This keeps the overall “single-pass, no model training loop” core approach (just generating a submission from simple statistics) while making predictions non-trivial and typically much closer to reasonable values for this competition. The submission is written with the required columns and row alignment by `id`.'
- What this solution (achieved 1.23596) has done: 'Your current baseline uses the per-type median, which is robust but often underfits this competition and leaves a large gap to the target. To move the score down (better) toward the target with minimal logic change, I switch the per-type statistic from median to mean, which typically better matches the MAE-driven evaluation here. I also add a tiny Bayesian-style shrinkage toward the global mean to stabilize rare coupling types without changing the overall “type-wise constant prediction” approach. The submission format, alignment by `id`, and file path/output (`submission.csv`) remain unchanged.'
- What this solution (achieved 1.18497) has done: 'Your current approach is a type-wise constant baseline with shrinkage; the smallest way to move the score down (better) toward the target is to make those constants closer to what minimizes MAE for each type. Since MAE is minimized by the median (not the mean), we switch the per-type center back to median but keep the same shrinkage idea for stability. To better match the competition metric (average log-MAE per type), we also apply shrinkage in a type-adaptive way (more shrinkage for rare types, less for frequent ones) without changing the “no model training loop” core logic. The submission file name, columns, and id alignment remain unchanged.'
- What this solution (achieved 1.18497) has done: 'Your current baseline is a type-wise constant with adaptive shrinkage, but it is still too coarse for this metric because each coupling `type` has a different scale and offset. The smallest change that usually reduces log-MAE here without changing the “single-pass statistics baseline” core logic is to also use molecule-level information: add the molecule’s mean target (computed from train) as an offset, then shrink that molecule offset back toward 0 for unseen/rare molecules. This keeps the same style of solution (no model training loop; only groupby statistics) while typically moving the score downward (better) toward your target. The submission remains aligned by `id` and written to `submission.csv`.'
- What this solution (achieved 1.18497) has done: 'I keep your “groupby statistics + shrinkage” baseline intact but make the molecule adjustment better aligned to the competition metric by learning molecule residuals *per coupling type* (since log-MAE is averaged by type and each type has very different offsets). This is still the same core logic (type constant + molecule offset), just with a more granular residual table and the same kind of shrinkage you already use. I also make the shrinkage weight depend on the amount of data for each (molecule,type) pair so rare pairs don’t overfit. This should reduce MAE within each type and move your score downward (better) toward the target.'
- What this solution (achieved 1.18497) has done: 'Your current baseline is already a type constant plus a (molecule,type) residual table with shrinkage; the main low-risk way to move the score down toward the target is to make the residual correction closer to what minimizes MAE. Since MAE is minimized by the median (not the mean), we switch the (molecule,type) residual estimator from mean to median while keeping the same shrinkage structure and weights. This keeps the same core “groupby statistics + shrinkage” logic and usually reduce log-MAE on this competition by making the per-group correction more robust to outliers. The submission writing, columns, and id alignment remain unchanged and it still produces `submission.csv` end-to-end.'
- What this solution (achieved 2.93251) has done: 'Your current baseline is still too coarse for this metric because it only uses `type` and `(molecule_name,type)` information, leaving out the very informative atom-pair geometry. With minimal changes and without introducing a training loop or new model, we can add one strong physical feature: the inter-atomic distance computed from `structures.csv`, then learn a per-`type` linear correction `a*distance + b` by least squares on a sample of training rows (kept small for runtime). This preserves the “groupby statistics + shrinkage” core logic and simply adds an extra additive term that usually reduces log-MAE substantially. The submission format, id alignment, and output path (`submission.csv`) remain unchanged.'
- What this solution (achieved 2.26318) has done: 'Your current score (2.93251, lower is better) is far worse than the target (-1.31958), so we should improve performance, but with minimal changes and the same overall “groupby statistics + shrinkage + per-type linear distance correction” logic. The biggest issue is that the distance-based linear correction is being fit on the *residual after* the (type + molecule,type) baseline, which tends to be weak/noisy and can easily make things worse—so we instead fit the per-type linear model directly to the target with distance as the only regressor (still a per-type `a*dist+b`, same approach, just a safer target). We also fix the per-type sampling to be deterministic (so you’re not accidentally taking arbitrary indices) and make the distance join more robust by using `join` (same semantics, less risk of misalignment/NaNs). These are small, stability-focused changes that should move the score down toward the target without changing the core approach.'
- What this solution (achieved 2.90783) has done: 'Your current score (2.26318, lower is better) is far from the target (-1.31958), so we should improve it, but with minimal changes and the same “type baseline + (molecule,type) residual + per-type linear distance term” core logic. The biggest issue is that the distance linear term currently contributes an additional *type-level intercept* (`b`) and is blended with a fixed `lambda_dist`, which can double-count the already-strong `pred_type`/`pred_mol_type` intercepts and destabilize per-type errors. I (1) fit the distance line on the residual after the existing baseline (so it truly acts as a correction), (2) force the distance model to be centered (no intercept) to avoid reintroducing offsets, and (3) set `lambda_dist=1.0` since the shrinkage weight `w` already controls the correction strength. These are small, targeted changes that typically reduce log-MAE materially versus the current behavior without changing the overall approach or adding any training loop.'
- What this solution (achieved 2.90783) has done: 'I keep your exact “type center + (molecule,type) residual + per-type centered linear distance correction” pipeline, but fix the main reason it currently scores extremely poorly: the distance join is producing lots of NaNs (molecule_name/atom_index mismatch), so the distance term becomes noise and hurts every type’s log-MAE. The minimal, score-improving change is to ensure `atom_index` columns are the same integer dtype in **train/test/structures** before building the `(molecule_name, atom_index)` index, and to compute the distance via a direct `reindex` on a MultiIndex (more reliable than `join` here). I also add a small safety clamp for any remaining invalid distances and keep all your shrinkage and centered-no-intercept fitting exactly as-is. This should move the score downward (better) toward the target without changing the core logic or adding any training loop.'
- What this solution (achieved 2.91008) has done: 'Your current pipeline is being hurt because the distance correction is fit using a deterministic “first N rows per type” slice, which is not representative and can make the learned slope noisy/wrong for some types. I keep the exact same logic (type median + (molecule,type) residual median with shrinkage + centered no-intercept per-type distance slope on the residual), but change the per-type sampling to a deterministic random sample (fixed seed) so the distance fit better reflects each type distribution without adding any training loop. I also add a small clamp on pair distances to a sensible range to reduce the impact of any remaining coordinate/join anomalies on the slope fit and predictions. These are minimal, targeted changes aimed at moving your log-MAE score down (better) toward the target while preserving the overall semantics.'
- What this solution (achieved 3.0083) has done: 'Your score is far worse than the target (lower is better), so we should improve it while keeping the exact same “type median + (molecule,type) residual median with shrinkage + centered per-type linear distance correction on the residual” core logic. The biggest issue is that the distance correction is currently **over‑shrunk** (`gamma=20000` makes `w` tiny), so the distance term contributes almost nothing and you fall back to a weak baseline; reducing this shrinkage is a minimal, directly relevant change. I also align the fit to the competition metric by using a **type-dependent gamma** (more shrinkage for rare types, less for common ones) without changing the model form. Everything else (features, fitting method, no intercept, submission format/path) stays the same.'
- What this solution (achieved 2.99778) has done: 'Your current score is far worse than the target (lower is better), so the smallest safe way to move toward the target is to make the distance correction actually contribute signal, not noise. I keep your exact baseline (type median + (molecule,type) residual median with shrinkage + centered no‑intercept per‑type linear distance correction on the residual) but fix the main instability: the distance fit is overly sensitive to outliers and skewed distance distributions per type. Concretely, I (1) standardize the distance feature per type (robust IQR scaling) for fitting/prediction while keeping the same linear form, and (2) add a light, type-wise ridge stabilization to the centered slope estimate (same closed-form, still no training loop), which typically reduces log-MAE. Submission format/path stays identical and it still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))



## === cell 1
DATA_DIR = "../input/champs-scalar-coupling"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")

train = pd.read_csv(
    train_path,
    usecols=[
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
)
test = pd.read_csv(
    test_path,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
)
sample_sub = pd.read_csv(sample_path)

structures = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "x", "y", "z"],
)

for c in ["atom_index_0", "atom_index_1"]:
    train[c] = train[c].astype(np.int32, copy=False)
    test[c] = test[c].astype(np.int32, copy=False)
structures["atom_index"] = structures["atom_index"].astype(np.int32, copy=False)

print(train.head())
print(test.head())
print(sample_sub.head())
print(structures.head())



## === cell 2
type_center = train.groupby("type")["scalar_coupling_constant"].median()
type_count = train.groupby("type")["scalar_coupling_constant"].size().astype(float)
global_center = float(train["scalar_coupling_constant"].median())

base_alpha = 200.0
alpha_t = base_alpha / np.sqrt(type_count.clip(lower=1.0))
shrunken_type_center = (type_center * type_count + global_center * alpha_t) / (
    type_count + alpha_t
)

train_type_pred = (
    train["type"].map(shrunken_type_center).fillna(global_center).astype(float)
)
train_resid = train["scalar_coupling_constant"].astype(float) - train_type_pred

grp = [train["molecule_name"], train["type"]]
mol_type_resid_center = train_resid.groupby(grp).median()
mol_type_count = train_resid.groupby(grp).size().astype(float)

mol_alpha = 50.0
beta_mt = mol_alpha / np.sqrt(mol_type_count.clip(lower=1.0))
shrunken_mol_type_resid = (mol_type_resid_center * mol_type_count) / (
    mol_type_count + beta_mt
)

pred_type = test["type"].map(shrunken_type_center).fillna(global_center).astype(float)

test_key = pd.MultiIndex.from_frame(test[["molecule_name", "type"]])
pred_mol_type = (
    pd.Series(shrunken_mol_type_resid.reindex(test_key).values, index=test.index)
    .fillna(0.0)
    .astype(float)
)

coords = (
    structures.set_index(["molecule_name", "atom_index"])[["x", "y", "z"]]
    .astype(np.float32)
    .sort_index()
)


def add_distance(df, prefix):
    key0 = pd.MultiIndex.from_frame(
        df[["molecule_name", "atom_index_0"]].rename(
            columns={"atom_index_0": "atom_index"}
        )
    )
    key1 = pd.MultiIndex.from_frame(
        df[["molecule_name", "atom_index_1"]].rename(
            columns={"atom_index_1": "atom_index"}
        )
    )

    p0 = coords.reindex(key0).to_numpy()
    p1 = coords.reindex(key1).to_numpy()

    d = np.linalg.norm(p0 - p1, axis=1)

    df[prefix + "_dist"] = d.astype(np.float32)
    return df


train = add_distance(train, "pair")
test = add_distance(test, "pair")

dist_median = float(np.nanmedian(train["pair_dist"].values))


def clean_dist(s, fill_value):
    s = s.replace([np.inf, -np.inf], np.nan).fillna(fill_value).astype(np.float32)
    return s.clip(lower=0.0, upper=10.0)


train["pair_dist"] = clean_dist(train["pair_dist"], dist_median)
test["pair_dist"] = clean_dist(test["pair_dist"], dist_median)

train_key = pd.MultiIndex.from_frame(train[["molecule_name", "type"]])
train_mol_type = (
    pd.Series(shrunken_mol_type_resid.reindex(train_key).values, index=train.index)
    .fillna(0.0)
    .astype(float)
)
train_baseline = (train_type_pred + train_mol_type).astype(float)
train_resid2 = train["scalar_coupling_constant"].astype(float) - train_baseline

max_rows_per_type = 200000

rng = np.random.RandomState(2020)
train_idx = []
for t, idx in train.groupby("type", sort=False).indices.items():
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size > max_rows_per_type:
        idx = rng.choice(idx, size=max_rows_per_type, replace=False)
    idx = np.sort(idx)
    train_idx.append(idx)
train_idx = np.concatenate(train_idx)

fit_df = pd.DataFrame(
    {
        "type": train.loc[train_idx, "type"].values,
        "dist": train.loc[train_idx, "pair_dist"].astype(float).values,
        "y": train_resid2.loc[train_idx].astype(float).values,  # residual target
    }
)


def robust_type_stats(g):
    x = g["dist"].to_numpy(dtype=np.float64)
    med = float(np.median(x))
    q1 = float(np.percentile(x, 25))
    q3 = float(np.percentile(x, 75))
    iqr = float(q3 - q1)
    if not np.isfinite(iqr) or iqr < 1e-6:
        iqr = 1.0
    return pd.Series({"dist_med": med, "dist_iqr": iqr})


type_dist_stats = fit_df.groupby("type", sort=False).apply(robust_type_stats)

fit_df = fit_df.join(type_dist_stats, on="type")
fit_df["dist_s"] = (fit_df["dist"] - fit_df["dist_med"]) / fit_df["dist_iqr"]


def fit_a_centered_ridge(g, ridge=1e-2):
    x = g["dist_s"].to_numpy(dtype=np.float64)
    y = g["y"].to_numpy(dtype=np.float64)
    n = x.size
    if n < 2:
        return pd.Series({"a": 0.0, "x_mean": float(x.mean()) if n == 1 else 0.0})
    x_mean = float(x.mean())
    xc = x - x_mean
    denom = float(np.sum(xc**2) + ridge * n)
    if denom <= 1e-12:
        return pd.Series({"a": 0.0, "x_mean": x_mean})
    a = float(np.sum(xc * y) / denom)
    return pd.Series({"a": a, "x_mean": x_mean})


ab = fit_df.groupby("type", sort=False).apply(fit_a_centered_ridge)

type_n_fit = fit_df.groupby("type")["y"].size().astype(float)

gamma_base = 2000.0
gamma_t = (gamma_base / np.sqrt(type_n_fit.clip(lower=1.0))).astype(float)
w = (type_n_fit / (type_n_fit + gamma_t)).reindex(ab.index).fillna(0.0).astype(float)

ab["a"] = (ab["a"].astype(float) * w).astype(float)

test_type_stats = type_dist_stats.reindex(test["type"]).reset_index(drop=True)
test_dist_med = test_type_stats["dist_med"].fillna(dist_median).astype(float).values
test_dist_iqr = test_type_stats["dist_iqr"].fillna(1.0).astype(float).values
test_dist_s = (
    (test["pair_dist"].astype(float).values - test_dist_med) / test_dist_iqr
).astype(np.float64)

a_test = test["type"].map(ab["a"]).fillna(0.0).astype(float).values
xmean_test = test["type"].map(ab["x_mean"]).fillna(0.0).astype(float).values
pred_dist_corr = (a_test * (test_dist_s - xmean_test)).astype(float)

lambda_dist = 1.0
pred = (pred_type.values + pred_mol_type.values + lambda_dist * pred_dist_corr).astype(
    float
)

submission = pd.DataFrame(
    {
        "id": test["id"].astype(
            sample_sub["id"].dtype if "id" in sample_sub else test["id"].dtype
        ),
        "scalar_coupling_constant": pred,
    }
)

submission.to_csv("submission.csv", index=False)

print(submission["scalar_coupling_constant"].describe())
print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
