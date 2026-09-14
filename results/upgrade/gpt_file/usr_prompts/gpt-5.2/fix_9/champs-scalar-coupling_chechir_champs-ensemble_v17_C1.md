# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
    min_id_coverage: float = 0.999999,
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
    min_id_coverage: float = 0.999999,
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

train_type_only = pd.read_csv(resolve_input_file("train.csv"), usecols=["type", TARGET])
type_median = train_type_only.groupby("type")[TARGET].median()

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

search_roots = [
    BASE_INPUT,
    ALT_INPUT,
    Path("../input/champ-preds"),
    Path("../input/nnet-b-seed-10"),
    Path("../input/nnet-b-seed-11"),
    Path("../input/nnet-b-seed-12"),
]
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

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

train_full = pd.read_csv(
    resolve_input_file("train.csv"),
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type", TARGET],
)
structures = pd.read_csv(
    resolve_input_file("structures.csv"),
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
)

s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)
s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)


def add_geom_features(df_base: pd.DataFrame) -> pd.DataFrame:
    df = df_base.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(s1, on=["molecule_name", "atom_index_1"], how="left")
    dx = df["x0"] - df["x1"]
    dy = df["y0"] - df["y1"]
    dz = df["z0"] - df["z1"]
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)
    df["dist2"] = dx * dx + dy * dy + dz * dz
    df["inv_dist"] = 1.0 / (df["dist"] + 1e-6)
    return df


train_feat = add_geom_features(train_full.drop(columns=[TARGET]))
test_feat = add_geom_features(
    test[["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]].copy()
)

mol_agg = (
    train_feat.groupby("molecule_name")["dist"]
    .agg(["mean", "std", "min", "max"])
    .reset_index()
)
mol_agg = mol_agg.rename(
    columns={
        "mean": "mol_dist_mean",
        "std": "mol_dist_std",
        "min": "mol_dist_min",
        "max": "mol_dist_max",
    }
)
type_agg = (
    train_feat.groupby("type")["dist"].agg(["mean", "std", "min", "max"]).reset_index()
)
type_agg = type_agg.rename(
    columns={
        "mean": "type_dist_mean",
        "std": "type_dist_std",
        "min": "type_dist_min",
        "max": "type_dist_max",
    }
)

train_feat = train_feat.merge(mol_agg, on="molecule_name", how="left").merge(
    type_agg, on="type", how="left"
)
test_feat = test_feat.merge(mol_agg, on="molecule_name", how="left").merge(
    type_agg, on="type", how="left"
)

X = train_feat[
    [
        "molecule_name",
        "type",
        "atom_0",
        "atom_1",
        "x0",
        "y0",
        "z0",
        "x1",
        "y1",
        "z1",
        "dist",
        "dist2",
        "inv_dist",
        "mol_dist_mean",
        "mol_dist_std",
        "mol_dist_min",
        "mol_dist_max",
        "type_dist_mean",
        "type_dist_std",
        "type_dist_min",
        "type_dist_max",
    ]
]
y = train_full[TARGET].astype(float)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.05, random_state=42, shuffle=True
)

cat_cols = ["molecule_name", "type", "atom_0", "atom_1"]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
        (
            "cat",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            cat_cols,
        ),
    ],
    remainder="drop",
)

baseline_model = HistGradientBoostingRegressor(
    loss="absolute_error",
    learning_rate=0.08,
    max_depth=8,
    max_bins=255,
    random_state=42,
)

baseline_pipe = Pipeline([("prep", preprocess), ("model", baseline_model)])

baseline_pipe.fit(X_train, y_train)

X_test = test_feat[X.columns]
baseline_pred = pd.Series(
    baseline_pipe.predict(X_test), index=test.index, name="baseline_pred"
).astype(float)

fallback_type_median = test["type"].map(type_median).astype(float)
baseline_pred = baseline_pred.fillna(fallback_type_median)

baseline_pred = baseline_pred.fillna(type_median.median())

for c in ["n1", "n2", "lgb_a", "lgb_m", "nnet"]:
    test[c] = pd.to_numeric(test[c], errors="coerce").astype(float)
    miss = test[c].isna()
    if miss.any():
        test.loc[miss, c] = baseline_pred.loc[miss].values

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
