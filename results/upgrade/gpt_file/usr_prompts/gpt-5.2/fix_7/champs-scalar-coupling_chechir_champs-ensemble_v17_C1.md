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

-1.7766920641782642

# 6. Current score

1.18497

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.18497) has done: 'The errors come from relying on external Kaggle “../input/champ-preds” and “../input/nnet-b-seed-*” folders that don’t exist in your environment, so the prediction columns never get created and downstream cells fail. I keep the ensemble/blending core logic intact, but make it robust by (1) auto-discovering which of those input folders/files are actually present, (2) reading any available prediction files and aligning them to `test` by `id` (not by row order), and (3) falling back to a safe baseline (type-wise median from `train.csv`) when some/all external preds are missing. Finally, I ensure a valid `submission.csv` with the required columns is always written.'
- What this solution (achieved 1.18497) has done: 'Your current score is far above (worse than) the target, so we should improve it rather than “match” it; the biggest issue is that in this environment none of the external prediction files exist, so your blend collapses to the weak type-median baseline. I keep your exact blending core logic and weights, but change how missing prediction sources are handled: instead of defaulting to NaNs, we auto-discover any available submission-like CSVs under `../input/` and `/kaggle/data/` and use them as substitutes, aligned by `id`. If no external preds are found, we still fall back to the type-median baseline and always write a valid `submission.csv`. These changes are minimal, preserve evaluation semantics, and are directly aimed at improving score by using any real available predictions instead of the baseline.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower is better) is far worse than the target (-1.7767), and the main reason is that the intended strong external prediction files are missing so the blend effectively collapses toward a weak type-median baseline. I keep the exact blending weights/core logic, but make the “discovered substitute” logic much stricter: only accept candidate CSVs that (a) contain `id` and prediction, (b) have ~exactly the test row count, and (c) have nearly complete overlap of test ids, so we don’t accidentally blend in unrelated CSVs that destroy performance. If no valid substitutes are found, it still fall back to the type-median baseline and always write a valid `submission.csv`. This should improve score versus the current accidental blending with wrong/disaligned files while preserving your ensemble semantics.'
- What this solution (achieved 1.18497) has done: 'Your current score is much worse than the target (lower-is-better), and the main cause is that the blend is almost certainly using “discovered” CSVs that aren’t actually CHAMPS predictions (or are misaligned), which can easily be worse than the simple type-median baseline. I keep your exact ensemble/blending logic and weights, but make the substitute-discovery logic *safer*: only use external files if they have near-perfect ID coverage, unique IDs, numeric prediction values, and non-degenerate variance; otherwise we drop them and fall back to the type-median baseline (which should improve score versus bad substitutes). I also make ID alignment robust by forcing an inner merge on `id` when reading predictions (still preserving your semantics of aligning by `id`). Finally, the script always write a valid `submission.csv`.'
- What this solution (achieved 1.18497) has done: 'We keep your blending weights and overall ensemble logic identical, but tighten the “substitute prediction discovery” so it no longer drags in unrelated CSVs that can be worse than the type-median baseline (which is likely why your score is stuck around 1.18). Concretely, we (1) require candidates to match the test `id` set by an inner-merge check (not just overlap ratio), (2) reject files with duplicated IDs or large missing ID fractions after reindexing to the full test set, and (3) prefer candidates whose distribution is plausible for CHAMPS (nontrivial variance and bounded extreme outliers). If no strong candidates exist, the pipeline reliably fall back to the type-median baseline (which should be better than blending in wrong files) and still always write a valid `submission.csv`. These are minimal changes focused only on improving score by avoiding harmful substitute files while preserving your current ensemble semantics.'
- What this solution (achieved 1.18497) has done: 'Your current score is far worse than the target (lower-is-better), so the most likely issue is that the blend is still not using any genuinely good external predictions and/or is accidentally accepting “valid-looking” but low-quality/misaligned substitutes. I keep your ensemble weights and blending exactly the same, but change the substitute discovery to be even safer and more CHAMPS-specific by (1) only considering CSVs that are very likely CHAMPS submissions (exact test id set match after alignment), and (2) preferring candidates whose values vary by coupling `type` in a realistic way (type-wise standard deviation check), which helps reject constant/near-baseline files. If no strong candidates exist, we still fall back to the type-median baseline to avoid harm, and we still always write a valid `submission.csv`. These changes are minimal and directly aimed at improving the score versus the current ~1.18 by preventing bad substitutes from contaminating the blend.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path


def safe_listdir(p):
    p = Path(p)
    if p.exists():
        items = sorted([x.name for x in p.iterdir()])
        print(f"{p} ({len(items)} items)")
        print(items[:50])
    else:
        print(f"{p} DOES NOT EXIST")


safe_listdir("../input/")
safe_listdir("../input/nnet-b-seed-10")
safe_listdir("../input/nnet-b-seed-11")
safe_listdir("../input/nnet-b-seed-12")
safe_listdir("../input/champ-preds")



## === cell 1
import numpy as np
import pandas as pd

BASE_INPUT = Path("../input/champs-scalar-coupling")
ALT_INPUT = Path("/kaggle/data/champs-scalar-coupling")  # safety for nonstandard mounts


def resolve_input_file(fname: str) -> Path:
    """Find file in known dataset locations."""
    for base in (BASE_INPUT, ALT_INPUT, Path("../input"), Path("/kaggle/input")):
        cand = base / fname
        if cand.exists():
            return cand
    cand = Path(fname)
    if cand.exists():
        return cand
    raise FileNotFoundError(f"Could not locate required file: {fname}")


def read_pred_file(path: str, target_col: str, id_col: str = "id") -> pd.Series:
    """
    Read a prediction CSV and return a Series indexed by id.
    Supports files with:
      - columns ['id', target_col]
      - or an unnamed first column as id (common in some kernels)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(p)

    if (
        id_col not in df.columns
        and len(df.columns) > 0
        and str(df.columns[0]).lower().startswith("unnamed")
    ):
        df = df.rename(columns={df.columns[0]: id_col})

    if id_col not in df.columns:
        raise ValueError(
            f"{path} missing '{id_col}' column. Columns: {df.columns.tolist()}"
        )

    if target_col not in df.columns:
        if "prediction" in df.columns:
            df = df.rename(columns={"prediction": target_col})
        else:
            raise ValueError(
                f"{path} missing '{target_col}' column. Columns: {df.columns.tolist()}"
            )

    df = df[[id_col, target_col]].copy()
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[id_col, target_col])
    df[id_col] = df[id_col].astype(np.int64, copy=False)

    s = df.set_index(id_col)[target_col]
    s.name = p.stem
    return s


def get_median_from_files(files, test_ids: pd.Series, target_col: str) -> pd.Series:
    """Take median across provided submission files, aligned by id."""
    series_list = []
    for f in files:
        try:
            s = read_pred_file(f, target_col=target_col).reindex(test_ids.values)
            series_list.append(s.reset_index(drop=True))
        except Exception as e:
            print(f"Skipping {f} due to: {type(e).__name__}: {e}")
    if len(series_list) == 0:
        raise FileNotFoundError(
            "None of the provided prediction files could be loaded."
        )
    mat = pd.concat(series_list, axis=1)
    return mat.median(axis=1)


def _typewise_std_check(
    s_full: pd.Series,
    test_types: pd.Series,
    min_nontrivial_types: int = 4,
    min_type_std: float = 1e-4,
) -> bool:
    """
    Score-improving safeguard: CHAMPS predictions should vary by coupling type.
    Reject substitutes that are almost constant within most types (often non-CHAMPS CSVs).
    """
    df = pd.DataFrame({"pred": s_full.values, "type": test_types.values})
    df = df.dropna(subset=["pred"])
    if len(df) < 1000:
        return False
    g = df.groupby("type")["pred"].std()
    g = g.replace([np.inf, -np.inf], np.nan).dropna()
    if g.empty:
        return False
    return int((g > min_type_std).sum()) >= int(min_nontrivial_types)


def _is_good_pred_series(
    s: pd.Series,
    test_ids: np.ndarray,
    expected_rows: int,
    test_types: pd.Series,
    min_id_coverage: float = 0.999999,  # CHANGE: require (effectively) exact id coverage to avoid harmful substitutes
    min_std: float = 1e-5,
    max_abs_quantile: float = 0.999,
    max_abs_value: float = 1e5,
) -> bool:
    """
    Score-improving safeguard: reject discovered substitutes unless they look like real CHAMPS
    predictions aligned by id with near-perfect coverage and plausible numeric distribution.
    """
    if s is None or len(s) == 0:
        return False
    s = s.dropna()
    if len(s) == 0:
        return False

    if not s.index.is_unique:
        return False

    test_id_set = set(map(int, test_ids.tolist()))
    in_test = np.fromiter((int(i) in test_id_set for i in s.index.values), dtype=bool)
    if in_test.mean() < min_id_coverage:
        return False

    s_full = s.reindex(test_ids)
    miss_frac = float(s_full.isna().mean())
    if miss_frac > (1.0 - min_id_coverage):
        return False

    vals = s_full.dropna().values.astype(float)
    if vals.size < expected_rows * min_id_coverage:
        return False

    if float(np.nanstd(vals)) < min_std:
        return False

    q = np.quantile(np.abs(vals), max_abs_quantile)
    if not np.isfinite(q) or q > max_abs_value:
        return False

    if not _typewise_std_check(s_full, test_types=test_types):
        return False

    return True


def discover_prediction_csvs(
    search_roots,
    target_col: str,
    test_ids: np.ndarray,
    test_types: pd.Series,
    expected_rows: int,
    max_files: int = 60,
    min_id_coverage: float = 0.999999,  # CHANGE: stricter to avoid unrelated CSVs hurting score
    accept_row_tol: int = 0,
):
    """
    Stricter discovery: only accept files that are extremely likely to be CHAMPS submissions.
    """
    found = []
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob("*.csv"):
            name = p.name.lower()
            if name in ("train.csv", "test.csv", "sample_submission.csv"):
                continue
            try:
                if p.stat().st_size < 200_000:
                    continue
            except Exception:
                continue

            try:
                try:
                    nrows = sum(1 for _ in open(p, "rb")) - 1
                except Exception:
                    nrows = None
                if nrows is not None and abs(nrows - expected_rows) > accept_row_tol:
                    continue

                s = read_pred_file(str(p), target_col=target_col)
                if nrows is None and abs(len(s) - expected_rows) > accept_row_tol:
                    continue

                if not _is_good_pred_series(
                    s,
                    test_ids=test_ids,
                    expected_rows=expected_rows,
                    test_types=test_types,
                    min_id_coverage=min_id_coverage,
                ):
                    continue

                found.append(str(p))
                if len(found) >= max_files:
                    return found
            except Exception:
                continue
    return found


test = pd.read_csv(resolve_input_file("test.csv"))
TARGET = "scalar_coupling_constant"

train = pd.read_csv(resolve_input_file("train.csv"), usecols=["type", TARGET])
type_median = train.groupby("type")[TARGET].median()

pred_sources = {}

n1_path = Path("../input/champ-preds/gnn_median_2279.csv")
n2_path = Path("../input/champ-preds/gnn_train_sep_2258.csv")

lgb_a_files = [
    "../input/champ-preds/submission_type_2085.csv",
    "../input/champ-preds/submission_type_2082.csv",
]
lgb_m_files = [
    "../input/champ-preds/lgb_type_full_f286_10.csv",
    "../input/champ-preds/lgb_type_full_f262_10.csv",
]
nnet_files = [
    "../input/nnet-b-seed-10/lgb_type_cv-1.72296_mae0.24019_bags-1_f120_fd5_10.csv",
    "../input/nnet-b-seed-11/lgb_type_cv-1.70647_mae0.2376_bags-1_f120_fd5_11.csv",
    "../input/nnet-b-seed-12/lgb_type_cv-1.69833_mae0.24106_bags-1_f120_fd5_12.csv",
]

search_roots = ["../input", "/kaggle/data", "/kaggle/input"]
discovered = discover_prediction_csvs(
    search_roots,
    target_col=TARGET,
    test_ids=test["id"].values,
    test_types=test["type"],
    expected_rows=len(test),
    max_files=60,
    min_id_coverage=0.999999,
    accept_row_tol=0,
)
print(f"Discovered {len(discovered)} STRICT+++ candidate prediction CSV(s).")
print(discovered[:20])


def try_read_one(path, name):
    try:
        s = read_pred_file(str(path), target_col=TARGET)
        if not _is_good_pred_series(
            s,
            test_ids=test["id"].values,
            expected_rows=len(test),
            test_types=test["type"],
            min_id_coverage=0.999999,
        ):
            raise ValueError(
                "Candidate prediction file failed strict validation (coverage/uniqueness/plausibility/typewise variability)."
            )
        s = s.reindex(test["id"].values).reset_index(drop=True)
        s.name = name
        return s
    except Exception as e:
        print(f"{name}: could not use {path} due to {type(e).__name__}: {e}")
        return pd.Series(np.nan, index=np.arange(len(test)), name=name)


def try_median(files, name):
    try:
        good = []
        for f in files:
            try:
                s = read_pred_file(str(f), target_col=TARGET)
                if _is_good_pred_series(
                    s,
                    test_ids=test["id"].values,
                    expected_rows=len(test),
                    test_types=test["type"],
                    min_id_coverage=0.999999,
                ):
                    good.append(str(f))
            except Exception:
                pass
        if len(good) == 0:
            raise FileNotFoundError(
                "No valid prediction files after strict validation."
            )
        s = get_median_from_files(good, test_ids=test["id"], target_col=TARGET)
        s.name = name
        return s.reset_index(drop=True)
    except Exception as e:
        print(f"{name}: falling back due to {type(e).__name__}: {e}")
        return pd.Series(np.nan, index=np.arange(len(test)), name=name)


discovered_queue = list(discovered)  # consume in order


def substitute_files_if_missing(original_files, want_n=1):
    existing = [f for f in original_files if Path(f).exists()]
    if len(existing) >= want_n:
        return existing
    out = list(existing)
    while len(out) < want_n and len(discovered_queue) > 0:
        cand = discovered_queue.pop(0)
        if cand not in out:
            out.append(cand)
    return out


n1_files = substitute_files_if_missing([str(n1_path)], want_n=1)
n2_files = substitute_files_if_missing([str(n2_path)], want_n=1)

pred_sources["n1"] = (
    try_read_one(n1_files[0], "n1")
    if len(n1_files)
    else pd.Series(np.nan, index=np.arange(len(test)), name="n1")
)
pred_sources["n2"] = (
    try_read_one(n2_files[0], "n2")
    if len(n2_files)
    else pd.Series(np.nan, index=np.arange(len(test)), name="n2")
)

lgb_a_files_use = substitute_files_if_missing(lgb_a_files, want_n=2)
lgb_m_files_use = substitute_files_if_missing(lgb_m_files, want_n=2)
nnet_files_use = substitute_files_if_missing(nnet_files, want_n=3)

pred_sources["lgb_a"] = try_median(lgb_a_files_use, "lgb_a")
pred_sources["lgb_m"] = try_median(lgb_m_files_use, "lgb_m")
pred_sources["nnet"] = try_median(nnet_files_use, "nnet")

for k, s in pred_sources.items():
    test[k] = s.values

baseline = test["type"].map(type_median).astype(float)
for c in ["n1", "n2", "lgb_a", "lgb_m", "nnet"]:
    test[c] = pd.to_numeric(test[c], errors="coerce").astype(float)
    miss = test[c].isna()
    if miss.any():
        test.loc[miss, c] = baseline.loc[miss].values

test.head(10)



## === cell 2
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
sns.heatmap(test[["n1", "n2", "lgb_a", "lgb_m", "nnet"]].corr(), ax=ax)
test[["n1", "n2", "lgb_a", "lgb_m", "nnet"]].corr()



## === cell 3
test["final_preds"] = (
    test["n1"] * 0.65
    + test["n2"] * 0.05
    + test["lgb_a"] * 0.15
    + test["lgb_m"] * 0.10
    + test["nnet"] * 0.05
)
test.head(20)



## === cell 4
submission = pd.DataFrame({"id": test["id"].astype(int)})
submission["scalar_coupling_constant"] = test["final_preds"].astype(float)

submission = submission.sort_values("id").reset_index(drop=True)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Wrote {submission_path} with shape {submission.shape}")
submission.head()



## === cell 5
submission.head(20)
