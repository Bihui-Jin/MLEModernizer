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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

-1.572352650827347

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.23566) has done: 'The notebook is trying to ensemble (“stack”) predictions from external datasets that aren’t present in your environment (`../input/top-mol`, `../input/another-one`, etc.), so it fails before producing any submission. I fix this by switching the input discovery to the actual provided competition path (`/kaggle/data/champs-scalar-coupling/`) and by adding a safe fallback that generates a baseline submission when no external prediction files exist. I also remove IPython-only magic (`%matplotlib inline`) so it runs as a plain Python script, and fix deprecated `np.bool`. This run end-to-end and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 1.23566) has done: 'Your current score (1.23566, lower-is-better) is far worse than the target (-1.5723), so we should improve performance while keeping your “groupby-mean fallback” core logic intact. The biggest issue is leakage/overfitting in the fallback: you’re using `molecule_name` at inference, but train/test are split by molecule so those lookups almost always miss and add noise; switching to a type-level baseline (plus a tiny amount of atom-pair geometry from `structures.csv`) is a minimal, legitimate improvement. We keep the same overall approach (compute simple aggregated statistics from train, merge into test, fill NAs, write submission), but add one robust feature: inter-atomic distance and type-specific distance→target calibration via per-type linear regression (closed-form, no iterative training loop). If any external submission-like CSVs are found, we keep your ensemble path unchanged.'
- What this solution (achieved 1.23566) has done: 'We keep your existing “type-wise distance→target linear calibration” fallback intact, but make two minimal, score-relevant fixes: (1) compute distances in float64 (then store as float32) to reduce numerical noise in the per-type regression coefficients, and (2) make the regression more robust by fitting on `1/dist` (physically more correlated with coupling strength than raw distance) while preserving the same closed-form, per-type linear model structure (still just `a + b*x` per type). This typically improves MAE/logMAE substantially versus raw distance while staying within your same feature-extraction/training semantics (no iterative training, no new models). Everything else (external-sub ensembling path, I/O paths, submission writing) remains unchanged and still always produce a valid `submission.csv`.'
- What this solution (achieved 1.24003) has done: 'Your current score (1.23566, lower-is-better) is far from the target (-1.5723), so we should legitimately improve predictions while keeping your “per-type closed-form linear calibration on 1/dist” core logic unchanged. The biggest minimal win is to fit the linear calibration per coupling `type` on a **log1p-transformed target** (still just `a + b*x` per type, still closed-form) and then invert back with `expm1`; this aligns better with the competition’s log-MAE behavior and reduces the effect of large outliers without changing the modeling approach. I also add a tiny, safe clamp on distance to avoid infinities/unstable extreme `1/dist`, and keep the external-submission ensembling path exactly as-is. The script still runs end-to-end and always writes a valid `submission.csv` with the required columns and row alignment.'
- What this solution (achieved 1.23982) has done: 'Your current score (1.24003, lower-is-better) is far worse than the target (-1.57235), so we should improve accuracy while keeping your same fallback core logic (per-type closed-form linear fit on `inv_dist` with a log1p-signed target transform). The largest minimal win is to make the linear fit less sensitive to outliers by winsorizing the transformed target **within each coupling type** before fitting (still the same model form `a + b*x`, still closed-form, just a more robust estimate). I also compute the regression coefficients in float64 (then store as float32) to reduce numerical noise, without changing the model structure. Everything else (external submission ensembling path, distance feature, submission format) remains the same and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 1.23982) has done: 'We keep your exact fallback modeling approach (per-type closed-form linear fit on `inv_dist` with signed-log1p target transform and per-type winsorization), but make one score-relevant improvement: add a second physically meaningful feature (`inv_dist2 = 1/dist^2`) and fit the same closed-form linear model on both features per coupling type. This preserves the “no iterative training loop” core logic while typically improving MAE because coupling strength often follows a steeper distance decay than 1/r. To avoid overfitting/noise, we also add a minimal ridge term in the 2×2 normal equations (still closed-form) and keep the existing small-sample fallback behavior. Everything else (external submission ensembling path, I/O paths, submission format) stays unchanged and it still always writes a valid `submission.csv`.'
- What this solution (achieved 1.23982) has done: 'We keep your exact fallback approach (type-wise closed-form linear regression on `inv_dist` and `inv_dist2` with signed-log1p target and per-type winsorization), but fix two score-relevant issues that can silently hurt MAE: (1) the per-type quantile clipping currently uses float32 and a slow/unstable `transform(lambda quantile)` path; we compute per-type clip thresholds once in float64 and merge them back (same semantics, more stable and faster), and (2) we make the regression numerically safer by solving the 2×2 system per type using sums (not means) and a slightly larger ridge to avoid near-singular types (still closed-form, same model). These are minimal changes that generally reduce noise/outliers sensitivity and should improve the score (lower is better), without changing the overall modeling logic or adding iterative training. Output format/paths remain identical and it still always writes `submission.csv`.'
- What this solution (achieved 1.23982) has done: 'We keep your exact fallback model (per-type closed-form regression on `inv_dist` and `inv_dist2` with signed-log1p target + per-type winsorization), but add one more geometrically meaningful, still-minimal feature: `inv_dist3 = 1/dist^3`, and fit the same closed-form linear model with three features per type using summed normal equations with a tiny ridge for stability. This preserves the training approach (no iterative loops, no new model class) while typically improving MAE because scalar couplings often decay faster than 1/r and benefit from a slightly richer distance basis. We also keep your external-submission ensembling path unchanged, only improving the fallback used when external predictions aren’t available/usable. The script still runs end-to-end and writes a valid `submission.csv` with the required schema and row alignment.'
- What this solution (achieved 1.23982) has done: 'The crash comes from how pairwise cross-product column names are generated for features like `is_C_0` and `is_H_0`: the code uses string ordering (`ci <= cj`) to pick a key name, which can mismatch the actually-created column (created only for `i<=j` in the original `x_cols` order). I fix this by generating cross-term keys using the same index-based ordering used when creating the columns, so the lookup always matches and the regression runs. I also add a small safeguard to fill any missing aggregated columns with zeros (defensive against unexpected missingness) and ensure `submission` is always defined so cell 5 can write `submission.csv`. These changes are score-neutral/positive (they restore the intended model) and keep the core logic identical.'
- What this solution (achieved 1.23982) has done: 'Your current score (1.23982, lower-is-better) is far from the target (-1.57235), so we should legitimately improve predictions while keeping the same fallback core logic (type-wise closed-form regression on distance-basis + simple atom-type indicators). The biggest minimal gain available within your constraints is to add a couple of additional, still “cheap” geometry features: (1) coordinate deltas (dx,dy,dz) to capture orientation effects, and (2) a very small set of per-molecule centering features (atom coords relative to molecule centroid) so the model can use local position patterns without any new model class. We keep the exact training semantics: per-type closed-form ridge regression on engineered features, with the same signed-log1p transform and per-type winsorization. External-submission ensembling remains unchanged; these new features only strengthen the fallback baseline used when external predictions aren’t present/usable.'
- What this solution (achieved 1.23982) has done: 'Your current score (1.23982, lower-is-better) is still very far from the target (-1.57235), so we should make a small, legitimate accuracy improvement while preserving your exact fallback approach (per-type closed-form ridge regression on engineered geometry + atom indicators with signed-log1p target transform and winsorization). The most direct minimal gain is to incorporate the provided per-atom physics tables (mulliken charges + shielding tensors) as a few additional pairwise features (sum/diff of charges; diagonal/trace shielding sums/diffs) without changing the training method. This keeps the same model family and fitting semantics (still per-type closed-form regression), but gives the model signal beyond distance/orientation that is known to correlate with couplings. All I/O paths and submission writing stay the same, and external-sub ensembling remains unchanged.'
- What this solution (achieved 1.23982) has done: 'Your current score (1.23982, lower-is-better) is far worse than the target (-1.57235), so we should improve accuracy while keeping the same per-type closed-form ridge regression fallback. The smallest high-signal change is to add a couple of additional physics tables already available (`dipole_moments.csv` per molecule and `potential_energy.csv` per molecule) as simple per-pair features, without changing the model family/training semantics. We keep your existing geometry/atom-indicator/charge/shielding features and the same signed-log1p target transform + per-type winsorization; we just augment `x_cols` and fill missing values robustly. External-submission ensembling stays unchanged, and the script still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

BASE_DIR = "/kaggle/data/champs-scalar-coupling"
TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")
STRUCTURES_PATH = os.path.join(BASE_DIR, "structures.csv")

print("Listing base dir:", BASE_DIR)
print("Exists:", os.path.exists(BASE_DIR))
print("Some files:", sorted(os.listdir(BASE_DIR))[:20])



## === cell 1
train = pd.read_csv(
    TRAIN_PATH,
    usecols=[
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
)
test = pd.read_csv(
    TEST_PATH,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print("train shape:", train.shape)
print("test shape:", test.shape)
print("sample_sub shape:", sample_sub.shape)
print("train columns:", train.columns.tolist())
print("test columns:", test.columns.tolist())



## === cell 2
SEARCH_ROOTS = [
    "/kaggle/data",
    "/kaggle/input",
    "/kaggle/working",
]


def find_candidate_prediction_csvs(roots, max_files=50):
    candidates = []
    for root in roots:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                if not fn.lower().endswith(".csv"):
                    continue
                fpath = os.path.join(dirpath, fn)
                if os.path.basename(fpath) in {
                    "train.csv",
                    "test.csv",
                    "structures.csv",
                    "scalar_coupling_contributions.csv",
                    "magnetic_shielding_tensors.csv",
                    "mulliken_charges.csv",
                    "dipole_moments.csv",
                    "potential_energy.csv",
                    "sample_submission.csv",
                }:
                    continue
                candidates.append(fpath)
                if len(candidates) >= max_files:
                    return candidates
    return candidates


candidate_csvs = find_candidate_prediction_csvs(SEARCH_ROOTS, max_files=200)
print("Found candidate external CSVs (up to 200):", len(candidate_csvs))
print("\n".join(candidate_csvs[:20]))




## === cell 3
def load_submission_like_csv(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    cols = set(df.columns)
    if "id" in cols and "scalar_coupling_constant" in cols:
        df = df[["id", "scalar_coupling_constant"]].copy()
        return df
    return None


external_subs = []
for p in candidate_csvs:
    df = load_submission_like_csv(p)
    if df is None:
        continue
    external_subs.append((p, df))

print("External submission-like CSVs loaded:", len(external_subs))
for p, df in external_subs[:10]:
    print(p, df.shape)




## === cell 4
def add_distance_and_atom_feature(df_pairs, structures_df):
    s = structures_df[["molecule_name", "atom_index", "atom", "x", "y", "z"]].copy()

    cent = (
        s.assign(
            x=s["x"].astype(np.float64),
            y=s["y"].astype(np.float64),
            z=s["z"].astype(np.float64),
        )
        .groupby("molecule_name", sort=False)[["x", "y", "z"]]
        .mean()
        .rename(columns={"x": "cx", "y": "cy", "z": "cz"})
        .reset_index()
    )
    cent[["cx", "cy", "cz"]] = cent[["cx", "cy", "cz"]].astype(np.float32)

    s0 = s.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom0",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
    )
    s1 = s.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom1",
            "x": "x1",
            "y": "y1",
            "z": "z1",
        }
    )

    out = df_pairs.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
    out = out.merge(s1, on=["molecule_name", "atom_index_1"], how="left")
    out = out.merge(cent, on="molecule_name", how="left")

    x0 = out["x0"].astype(np.float64)
    y0 = out["y0"].astype(np.float64)
    z0 = out["z0"].astype(np.float64)
    x1 = out["x1"].astype(np.float64)
    y1 = out["y1"].astype(np.float64)
    z1 = out["z1"].astype(np.float64)

    dx = x0 - x1
    dy = y0 - y1
    dz = z0 - z1
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)

    out["dx"] = dx.astype(np.float32)
    out["dy"] = dy.astype(np.float32)
    out["dz"] = dz.astype(np.float32)
    out["dist"] = dist.astype(np.float32)

    cx = out["cx"].astype(np.float64)
    cy = out["cy"].astype(np.float64)
    cz = out["cz"].astype(np.float64)
    x0c = x0 - cx
    y0c = y0 - cy
    z0c = z0 - cz
    x1c = x1 - cx
    y1c = y1 - cy
    z1c = z1 - cz

    out["x0c"] = x0c.astype(np.float32)
    out["y0c"] = y0c.astype(np.float32)
    out["z0c"] = z0c.astype(np.float32)
    out["x1c"] = x1c.astype(np.float32)
    out["y1c"] = y1c.astype(np.float32)
    out["z1c"] = z1c.astype(np.float32)

    r0 = np.sqrt(x0c * x0c + y0c * y0c + z0c * z0c)
    r1 = np.sqrt(x1c * x1c + y1c * y1c + z1c * z1c)
    dot01 = x0c * x1c + y0c * y1c + z0c * z1c
    denom = np.maximum(r0 * r1, 1e-12)  # numerical guard
    cos01 = dot01 / denom

    out["r0"] = r0.astype(np.float32)
    out["r1"] = r1.astype(np.float32)
    out["dot01"] = dot01.astype(np.float32)
    out["abs_dot01"] = np.abs(dot01).astype(np.float32)
    out["cos01"] = cos01.astype(np.float32)
    out["abs_cos01"] = np.abs(cos01).astype(np.float32)

    out = out.drop(columns=["x0", "y0", "z0", "x1", "y1", "z1", "cx", "cy", "cz"])
    return out


def fit_typewise_linear_kfeat(
    train_feat, x_cols, y_col, ridge=1e-5, min_n_for_slopes=50
):
    x_cols = list(x_cols)
    tmp = train_feat[["type"] + x_cols + [y_col]].copy()
    for c in x_cols:
        tmp[c] = tmp[c].astype(np.float64)
    tmp["y"] = tmp[y_col].astype(np.float64)

    g = tmp.groupby("type", sort=False)
    n = g.size().astype(np.int64)

    sum_y = g["y"].sum().astype(np.float64)
    sum_x = {c: g[c].sum().astype(np.float64) for c in x_cols}

    stats = pd.DataFrame({"type": n.index, "n": n.values})
    stats["sum_y"] = sum_y.values
    for c in x_cols:
        stats[f"sum_{c}"] = sum_x[c].values

    eps_n = 1e-30
    n_f = stats["n"].astype(np.float64).values
    inv_n = 1.0 / (n_f + eps_n)

    mean_y = stats["sum_y"].astype(np.float64).values * inv_n
    mean_x = {c: stats[f"sum_{c}"].astype(np.float64).values * inv_n for c in x_cols}

    for c in x_cols:
        tmp[f"{c}_y"] = tmp[c] * tmp["y"]
    for i, ci in enumerate(x_cols):
        for j in range(i, len(x_cols)):
            cj = x_cols[j]
            tmp[f"{ci}_{cj}"] = tmp[ci] * tmp[cj]

    agg_dict = {}
    for c in x_cols:
        agg_dict[f"sum_{c}_y"] = (f"{c}_y", "sum")
    for i, ci in enumerate(x_cols):
        for j in range(i, len(x_cols)):
            cj = x_cols[j]
            agg_dict[f"sum_{ci}_{cj}"] = (f"{ci}_{cj}", "sum")

    cross = tmp.groupby("type", sort=False).agg(**agg_dict).reset_index()
    stats = stats.merge(cross, on="type", how="left")

    expected_cols = []
    for c in x_cols:
        expected_cols.append(f"sum_{c}_y")
    for i, ci in enumerate(x_cols):
        for j in range(i, len(x_cols)):
            cj = x_cols[j]
            expected_cols.append(f"sum_{ci}_{cj}")
    for c in expected_cols:
        if c not in stats.columns:
            stats[c] = 0.0
    stats[expected_cols] = stats[expected_cols].fillna(0.0)

    k = len(x_cols)
    coef = np.zeros((stats.shape[0], k), dtype=np.float64)

    r = float(ridge)
    for row_i in range(stats.shape[0]):
        nn = float(n_f[row_i])
        if nn <= 0:
            continue

        A = np.zeros((k, k), dtype=np.float64)
        cvec = np.zeros((k,), dtype=np.float64)

        my = mean_y[row_i]
        for ii, ci in enumerate(x_cols):
            s_xy = float(stats.loc[row_i, f"sum_{ci}_y"])
            cvec[ii] = s_xy - nn * float(mean_x[ci][row_i]) * my

        for ii, ci in enumerate(x_cols):
            for jj in range(ii, k):
                cj = x_cols[jj]
                key = f"sum_{ci}_{cj}"
                s_xixj = float(stats.loc[row_i, key])
                val = s_xixj - nn * float(mean_x[ci][row_i]) * float(mean_x[cj][row_i])
                A[ii, jj] = val
                A[jj, ii] = val

        A.flat[:: k + 1] += r

        try:
            coef[row_i, :] = np.linalg.solve(A, cvec)
        except np.linalg.LinAlgError:
            coef[row_i, :] = 0.0

    a0 = mean_y.copy()
    for j, cj in enumerate(x_cols):
        a0 = a0 - coef[:, j] * mean_x[cj]

    out = stats[["type", "n"]].copy()
    out["a"] = a0.astype(np.float32)

    for j, cj in enumerate(x_cols):
        out[f"b_{cj}"] = coef[:, j].astype(np.float32)

    out["type_mean"] = mean_y.astype(np.float32)

    small = out["n"] < int(min_n_for_slopes)
    for j, cj in enumerate(x_cols):
        out.loc[small, f"b_{cj}"] = 0.0
    out.loc[small, "a"] = out.loc[small, "type_mean"].astype(np.float32)

    return out


def build_fallback_submission(train_df, test_df, sample_sub_df):
    structures = pd.read_csv(
        STRUCTURES_PATH,
        usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
        dtype={
            "molecule_name": "category",
            "atom_index": np.int16,
            "atom": "category",
            "x": np.float32,
            "y": np.float32,
            "z": np.float32,
        },
    )

    mc_path = os.path.join(BASE_DIR, "mulliken_charges.csv")
    mst_path = os.path.join(BASE_DIR, "magnetic_shielding_tensors.csv")

    mulliken = pd.read_csv(
        mc_path,
        usecols=["molecule_name", "atom_index", "mulliken_charge"],
        dtype={
            "molecule_name": "category",
            "atom_index": np.int16,
            "mulliken_charge": np.float32,
        },
    )
    shielding = pd.read_csv(
        mst_path,
        usecols=["molecule_name", "atom_index", "XX", "YY", "ZZ"],
        dtype={
            "molecule_name": "category",
            "atom_index": np.int16,
            "XX": np.float32,
            "YY": np.float32,
            "ZZ": np.float32,
        },
    )
    shielding["trace"] = (
        shielding["XX"].astype(np.float32)
        + shielding["YY"].astype(np.float32)
        + shielding["ZZ"].astype(np.float32)
    ).astype(np.float32)

    dip_path = os.path.join(BASE_DIR, "dipole_moments.csv")
    pe_path = os.path.join(BASE_DIR, "potential_energy.csv")
    dipole = pd.read_csv(
        dip_path,
        usecols=["molecule_name", "X", "Y", "Z"],
        dtype={
            "molecule_name": "category",
            "X": np.float32,
            "Y": np.float32,
            "Z": np.float32,
        },
    )
    pe = pd.read_csv(
        pe_path,
        usecols=["molecule_name", "potential_energy"],
        dtype={"molecule_name": "category", "potential_energy": np.float32},
    )
    dipole["dip_norm"] = np.sqrt(
        dipole["X"].astype(np.float64) ** 2
        + dipole["Y"].astype(np.float64) ** 2
        + dipole["Z"].astype(np.float64) ** 2
    ).astype(np.float32)

    train_feat = add_distance_and_atom_feature(train_df, structures)
    test_feat = add_distance_and_atom_feature(test_df, structures)

    train_feat = train_feat.merge(dipole, on="molecule_name", how="left")
    test_feat = test_feat.merge(dipole, on="molecule_name", how="left")
    train_feat = train_feat.merge(pe, on="molecule_name", how="left")
    test_feat = test_feat.merge(pe, on="molecule_name", how="left")

    mc0 = mulliken.rename(
        columns={"atom_index": "atom_index_0", "mulliken_charge": "mc0"}
    )
    mc1 = mulliken.rename(
        columns={"atom_index": "atom_index_1", "mulliken_charge": "mc1"}
    )
    st0 = shielding.rename(
        columns={
            "atom_index": "atom_index_0",
            "XX": "sXX0",
            "YY": "sYY0",
            "ZZ": "sZZ0",
            "trace": "str0",
        }
    )
    st1 = shielding.rename(
        columns={
            "atom_index": "atom_index_1",
            "XX": "sXX1",
            "YY": "sYY1",
            "ZZ": "sZZ1",
            "trace": "str1",
        }
    )

    train_feat = train_feat.merge(mc0, on=["molecule_name", "atom_index_0"], how="left")
    train_feat = train_feat.merge(mc1, on=["molecule_name", "atom_index_1"], how="left")
    train_feat = train_feat.merge(st0, on=["molecule_name", "atom_index_0"], how="left")
    train_feat = train_feat.merge(st1, on=["molecule_name", "atom_index_1"], how="left")

    test_feat = test_feat.merge(mc0, on=["molecule_name", "atom_index_0"], how="left")
    test_feat = test_feat.merge(mc1, on=["molecule_name", "atom_index_1"], how="left")
    test_feat = test_feat.merge(st0, on=["molecule_name", "atom_index_0"], how="left")
    test_feat = test_feat.merge(st1, on=["molecule_name", "atom_index_1"], how="left")

    aux_cols = [
        "mc0",
        "mc1",
        "sXX0",
        "sYY0",
        "sZZ0",
        "str0",
        "sXX1",
        "sYY1",
        "sZZ1",
        "str1",
    ]
    for c in aux_cols:
        train_feat[c] = train_feat[c].astype(np.float32).fillna(np.float32(0.0))
        test_feat[c] = test_feat[c].astype(np.float32).fillna(np.float32(0.0))

    mol_cols = ["X", "Y", "Z", "dip_norm", "potential_energy"]
    for c in mol_cols:
        train_feat[c] = train_feat[c].astype(np.float32).fillna(np.float32(0.0))
        test_feat[c] = test_feat[c].astype(np.float32).fillna(np.float32(0.0))

    train_feat["mc_sum"] = (train_feat["mc0"] + train_feat["mc1"]).astype(np.float32)
    train_feat["mc_diff"] = (train_feat["mc0"] - train_feat["mc1"]).astype(np.float32)
    test_feat["mc_sum"] = (test_feat["mc0"] + test_feat["mc1"]).astype(np.float32)
    test_feat["mc_diff"] = (test_feat["mc0"] - test_feat["mc1"]).astype(np.float32)

    for name, a, b in [
        ("str", "str0", "str1"),
        ("sXX", "sXX0", "sXX1"),
        ("sYY", "sYY0", "sYY1"),
        ("sZZ", "sZZ0", "sZZ1"),
    ]:
        train_feat[f"{name}_sum"] = (train_feat[a] + train_feat[b]).astype(np.float32)
        train_feat[f"{name}_diff"] = (train_feat[a] - train_feat[b]).astype(np.float32)
        test_feat[f"{name}_sum"] = (test_feat[a] + test_feat[b]).astype(np.float32)
        test_feat[f"{name}_diff"] = (test_feat[a] - test_feat[b]).astype(np.float32)

    dist_floor = np.float32(1e-3)
    train_dist = np.maximum(train_feat["dist"].values.astype(np.float32), dist_floor)
    test_dist = np.maximum(test_feat["dist"].values.astype(np.float32), dist_floor)

    train_feat["inv_dist"] = (1.0 / train_dist).astype(np.float32)
    test_feat["inv_dist"] = (1.0 / test_dist).astype(np.float32)

    train_feat["inv_dist2"] = (
        train_feat["inv_dist"].values.astype(np.float32) ** 2
    ).astype(np.float32)
    test_feat["inv_dist2"] = (
        test_feat["inv_dist"].values.astype(np.float32) ** 2
    ).astype(np.float32)

    train_feat["inv_dist3"] = (
        train_feat["inv_dist"].values.astype(np.float32) ** 3
    ).astype(np.float32)
    test_feat["inv_dist3"] = (
        test_feat["inv_dist"].values.astype(np.float32) ** 3
    ).astype(np.float32)

    train_feat["inv_dist5"] = (
        train_feat["inv_dist"].values.astype(np.float32) ** 5
    ).astype(np.float32)
    test_feat["inv_dist5"] = (
        test_feat["inv_dist"].values.astype(np.float32) ** 5
    ).astype(np.float32)

    y = train_feat["scalar_coupling_constant"].astype(np.float64).values
    y_sign = np.sign(y)
    y_trans = y_sign * np.log1p(np.abs(y))
    train_feat["y_trans"] = y_trans.astype(np.float32)

    q = (
        train_feat[["type", "y_trans"]]
        .assign(y_trans=lambda d: d["y_trans"].astype(np.float64))
        .groupby("type", sort=False)["y_trans"]
        .quantile([0.01, 0.99])
        .unstack()
        .reset_index()
        .rename(columns={0.01: "q_low", 0.99: "q_high"})
    )
    train_feat = train_feat.merge(q, on="type", how="left")
    y_trans_f64 = train_feat["y_trans"].astype(np.float64)
    train_feat["y_trans_clip"] = y_trans_f64.clip(
        lower=train_feat["q_low"].astype(np.float64),
        upper=train_feat["q_high"].astype(np.float64),
    ).astype(np.float32)
    train_feat = train_feat.drop(columns=["q_low", "q_high"])

    common_atoms = ["H", "C", "N", "O", "F"]
    for a in common_atoms:
        train_feat[f"is_{a}_0"] = (train_feat["atom0"].astype(str) == a).astype(
            np.float32
        )
        train_feat[f"is_{a}_1"] = (train_feat["atom1"].astype(str) == a).astype(
            np.float32
        )
        test_feat[f"is_{a}_0"] = (test_feat["atom0"].astype(str) == a).astype(
            np.float32
        )
        test_feat[f"is_{a}_1"] = (test_feat["atom1"].astype(str) == a).astype(
            np.float32
        )

    bin_cols = [f"is_{a}_0" for a in common_atoms] + [f"is_{a}_1" for a in common_atoms]
    bin_means = train_feat.groupby("type", sort=False)[bin_cols].mean().reset_index()
    bin_means = bin_means.rename(columns={c: f"{c}_mean" for c in bin_cols})
    train_feat = train_feat.merge(bin_means, on="type", how="left")
    test_feat = test_feat.merge(bin_means, on="type", how="left")

    for c in bin_cols:
        train_feat[c] = (
            train_feat[c].astype(np.float32)
            - train_feat[f"{c}_mean"].astype(np.float32)
        ).astype(np.float32)
        test_feat[c] = (
            test_feat[c].astype(np.float32) - test_feat[f"{c}_mean"].astype(np.float32)
        ).astype(np.float32)
    train_feat = train_feat.drop(columns=[f"{c}_mean" for c in bin_cols])
    test_feat = test_feat.drop(columns=[f"{c}_mean" for c in bin_cols])

    geom_cols = ["dx", "dy", "dz", "x0c", "y0c", "z0c", "x1c", "y1c", "z1c"]
    angle_cols = ["r0", "r1", "dot01", "abs_dot01", "cos01", "abs_cos01"]
    aux_pair_cols = [
        "mc_sum",
        "mc_diff",
        "str_sum",
        "str_diff",
        "sXX_sum",
        "sXX_diff",
        "sYY_sum",
        "sYY_diff",
        "sZZ_sum",
        "sZZ_diff",
    ]
    mol_feat_cols = ["X", "Y", "Z", "dip_norm", "potential_energy"]

    x_cols = (
        ["inv_dist", "inv_dist2", "inv_dist3", "inv_dist5"]
        + geom_cols
        + angle_cols
        + bin_cols
        + aux_pair_cols
        + mol_feat_cols
    )

    for c in x_cols:
        train_feat[c] = train_feat[c].astype(np.float32).fillna(np.float32(0.0))
        test_feat[c] = test_feat[c].astype(np.float32).fillna(np.float32(0.0))

    type_mean_y = (
        train_feat.groupby("type", sort=False)["scalar_coupling_constant"]
        .mean()
        .reset_index()
        .rename(columns={"scalar_coupling_constant": "type_mean_y"})
    )

    type_lr = fit_typewise_linear_kfeat(
        train_feat,
        x_cols=x_cols,
        y_col="y_trans_clip",
        ridge=4e-5,
        min_n_for_slopes=50,
    )
    type_lr = type_lr.merge(type_mean_y, on="type", how="left")

    test_feat = test_feat.merge(type_lr, on="type", how="left")

    global_mean_y = float(train_df["scalar_coupling_constant"].mean())
    test_feat["a"] = test_feat["a"].astype(np.float64).fillna(0.0)
    for c in x_cols:
        bc = f"b_{c}"
        if bc in test_feat.columns:
            test_feat[bc] = test_feat[bc].astype(np.float64).fillna(0.0)
        else:
            test_feat[bc] = 0.0
    test_feat["type_mean_y"] = (
        test_feat["type_mean_y"].astype(np.float64).fillna(global_mean_y)
    )

    pred_trans = test_feat["a"].astype(np.float64).values
    for c in x_cols:
        pred_trans = (
            pred_trans
            + test_feat[f"b_{c}"].astype(np.float64).values
            * test_feat[c].astype(np.float64).values
        )

    pred = np.sign(pred_trans) * np.expm1(np.abs(pred_trans))
    test_feat["scalar_coupling_constant"] = pred.astype(np.float32)

    missing = test_feat["scalar_coupling_constant"].isna()
    if missing.any():
        test_feat.loc[missing, "scalar_coupling_constant"] = test_feat.loc[
            missing, "type_mean_y"
        ].astype(np.float32)

    merged = test_feat[["id", "scalar_coupling_constant"]].copy()
    out = sample_sub_df[["id"]].merge(merged, on="id", how="left")

    if out["scalar_coupling_constant"].isna().any():
        out["scalar_coupling_constant"] = out["scalar_coupling_constant"].fillna(
            global_mean_y
        )

    return out


if len(external_subs) >= 1:
    base = sample_sub[["id"]].copy()
    pred_cols = []
    for i, (p, df) in enumerate(external_subs):
        col = f"mol{i}"
        tmp = df.rename(columns={"scalar_coupling_constant": col})
        base = base.merge(tmp, on="id", how="left")
        pred_cols.append(col)

    non_null_cols = [c for c in pred_cols if base[c].notna().any()]
    base = base[["id"] + non_null_cols].copy()
    concat_sub = base

    if len(non_null_cols) == 0:
        print(
            "External files found but none had usable predictions for our ids; using fallback baseline."
        )
        submission = build_fallback_submission(train, test, sample_sub)
    else:
        concat_sub["m_mean"] = concat_sub.loc[:, non_null_cols].mean(axis=1)

        submission = concat_sub[["id"]].copy()
        submission["scalar_coupling_constant"] = concat_sub["m_mean"].astype(float)

        if submission["scalar_coupling_constant"].isna().any():
            fallback = build_fallback_submission(train, test, sample_sub)
            submission = submission.merge(
                fallback, on="id", how="left", suffixes=("", "_fb")
            )
            submission["scalar_coupling_constant"] = submission[
                "scalar_coupling_constant"
            ].fillna(submission["scalar_coupling_constant_fb"])
            submission = submission[["id", "scalar_coupling_constant"]]
else:
    print("No external prediction files available; using fallback baseline.")
    submission = build_fallback_submission(train, test, sample_sub)

print("submission shape:", submission.shape)
print(submission.head())



## === cell 5
submission.to_csv("submission.csv", index=False, float_format="%.6f")

chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["id", "scalar_coupling_constant"]
assert chk.shape[0] == sample_sub.shape[0]
assert chk["id"].isna().sum() == 0
print("Wrote submission.csv with rows:", chk.shape[0])
print(chk.head())
