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

import numpy as np
import pandas as pd

INPUT_ROOT = Path("/kaggle/data/champs-scalar-coupling")
TARGET = "scalar_coupling_constant"

print("INPUT_ROOT exists:", INPUT_ROOT.exists())
print("Files in INPUT_ROOT:", sorted([p.name for p in INPUT_ROOT.glob("*")])[:20])

test_path = INPUT_ROOT / "test.csv"
train_path = INPUT_ROOT / "train.csv"
sample_path = INPUT_ROOT / "sample_submission.csv"

test = pd.read_csv(
    test_path,
    dtype={
        "id": "int32",
        "molecule_name": "category",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
    },
)
train = pd.read_csv(
    train_path,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type", TARGET],
    dtype={
        "id": "int32",
        "molecule_name": "category",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
        TARGET: "float32",
    },
)
sample_submission = pd.read_csv(sample_path, dtype={"id": "int32", TARGET: "float32"})

print("test:", test.shape, "train:", train.shape, "sample:", sample_submission.shape)
print(test.head())




## === cell 1
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor


def safe_read_csv(path: Path, usecols=None, dtype=None):
    if path.exists():
        return pd.read_csv(path, usecols=usecols, dtype=dtype)
    return None


structures = safe_read_csv(
    INPUT_ROOT / "structures.csv",
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "category",
        "atom_index": "int16",
        "atom": "category",
        "x": "float32",
        "y": "float32",
        "z": "float32",
    },
)
mulliken = safe_read_csv(
    INPUT_ROOT / "mulliken_charges.csv",
    usecols=["molecule_name", "atom_index", "mulliken_charge"],
    dtype={
        "molecule_name": "category",
        "atom_index": "int16",
        "mulliken_charge": "float32",
    },
)
mag = safe_read_csv(
    INPUT_ROOT / "magnetic_shielding_tensors.csv",
    usecols=[
        "molecule_name",
        "atom_index",
        "XX",
        "YY",
        "ZZ",
    ],  # keep same minimal subset
    dtype={
        "molecule_name": "category",
        "atom_index": "int16",
        "XX": "float32",
        "YY": "float32",
        "ZZ": "float32",
    },
)
dip = safe_read_csv(
    INPUT_ROOT / "dipole_moments.csv",
    usecols=["molecule_name", "X", "Y", "Z"],
    dtype={"molecule_name": "category", "X": "float32", "Y": "float32", "Z": "float32"},
)
pe = safe_read_csv(
    INPUT_ROOT / "potential_energy.csv",
    usecols=["molecule_name", "potential_energy"],
    dtype={"molecule_name": "category", "potential_energy": "float32"},
)

structures_idx = structures.set_index(["molecule_name", "atom_index"]).sort_index()
mulliken_idx = (
    None
    if mulliken is None
    else mulliken.set_index(["molecule_name", "atom_index"]).sort_index()
)
mag_idx = (
    None if mag is None else mag.set_index(["molecule_name", "atom_index"]).sort_index()
)
dip_idx = None if dip is None else dip.set_index("molecule_name").sort_index()
pe_idx = None if pe is None else pe.set_index("molecule_name").sort_index()


def build_pair_features(df_pairs: pd.DataFrame) -> pd.DataFrame:
    df = df_pairs[
        ["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]
    ].copy()

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

    a0 = structures_idx.reindex(key0).reset_index(drop=True)
    a1 = structures_idx.reindex(key1).reset_index(drop=True)

    df["atom_0"] = a0["atom"].values
    df["x_0"] = a0["x"].values
    df["y_0"] = a0["y"].values
    df["z_0"] = a0["z"].values

    df["atom_1"] = a1["atom"].values
    df["x_1"] = a1["x"].values
    df["y_1"] = a1["y"].values
    df["z_1"] = a1["z"].values

    if mulliken_idx is not None:
        m0 = mulliken_idx.reindex(key0).reset_index(drop=True)
        m1 = mulliken_idx.reindex(key1).reset_index(drop=True)
        df["mulliken_0"] = m0["mulliken_charge"].values
        df["mulliken_1"] = m1["mulliken_charge"].values

    if mag_idx is not None:
        t0 = mag_idx.reindex(key0).reset_index(drop=True)
        t1 = mag_idx.reindex(key1).reset_index(drop=True)
        df["mXX_0"] = t0["XX"].values
        df["mYY_0"] = t0["YY"].values
        df["mZZ_0"] = t0["ZZ"].values
        df["mXX_1"] = t1["XX"].values
        df["mYY_1"] = t1["YY"].values
        df["mZZ_1"] = t1["ZZ"].values

    if dip_idx is not None:
        d = dip_idx.reindex(df["molecule_name"]).reset_index(drop=True)
        df["dipX"] = d["X"].values
        df["dipY"] = d["Y"].values
        df["dipZ"] = d["Z"].values

    if pe_idx is not None:
        e = pe_idx.reindex(df["molecule_name"]).reset_index(drop=True)
        df["potential_energy"] = e["potential_energy"].values

    dx = df["x_0"] - df["x_1"]
    dy = df["y_0"] - df["y_1"]
    dz = df["z_0"] - df["z_1"]
    df["dx"] = dx
    df["dy"] = dy
    df["dz"] = dz
    dist2 = dx * dx + dy * dy + dz * dz
    df["dist2"] = dist2
    df["dist"] = np.sqrt(dist2)

    df["abs_dx"] = dx.abs()
    df["abs_dy"] = dy.abs()
    df["abs_dz"] = dz.abs()
    df["x_mean"] = (df["x_0"] + df["x_1"]) * 0.5
    df["y_mean"] = (df["y_0"] + df["y_1"]) * 0.5
    df["z_mean"] = (df["z_0"] + df["z_1"]) * 0.5

    if "mulliken_0" in df.columns and "mulliken_1" in df.columns:
        diff = df["mulliken_0"] - df["mulliken_1"]
        df["mulliken_diff"] = diff
        df["mulliken_absdiff"] = diff.abs()

    for base in ["mXX", "mYY", "mZZ"]:
        c0 = f"{base}_0"
        c1 = f"{base}_1"
        if c0 in df.columns and c1 in df.columns:
            d = df[c0] - df[c1]
            df[f"{base}_diff"] = d
            df[f"{base}_absdiff"] = d.abs()

    return df


train_feat = build_pair_features(train.drop(columns=[TARGET]))
test_feat = build_pair_features(test)

drop_cols = ["id", "molecule_name"]  # keep type/atom_0/atom_1 as categorical
feature_cols = [c for c in train_feat.columns if c not in drop_cols]
cat_cols = [c for c in ["type", "atom_0", "atom_1"] if c in feature_cols]
num_cols = [c for c in feature_cols if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                ]
            ),
            num_cols,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "enc",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                    ),
                ]
            ),
            cat_cols,
        ),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

reg = HistGradientBoostingRegressor(
    loss="absolute_error",
    learning_rate=0.08,
    max_depth=8,
    max_leaf_nodes=63,
    min_samples_leaf=50,
    l2_regularization=0.0,
    max_iter=250,
    random_state=42,
)

model = Pipeline(steps=[("prep", preprocess), ("reg", reg)])

gkf = GroupKFold(n_splits=3)
y = train[TARGET].values
groups = train["molecule_name"].values

X_all = train_feat[feature_cols]
cv_maes = []
for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_all, y, groups=groups), 1):
    X_tr = X_all.take(tr_idx, axis=0)
    X_va = X_all.take(va_idx, axis=0)
    model.fit(X_tr, y[tr_idx])
    pred_va = model.predict(X_va)
    mae = mean_absolute_error(y[va_idx], pred_va)
    cv_maes.append(mae)
    print(f"Fold {fold} MAE: {mae:.6f}")
print("CV MAE mean:", float(np.mean(cv_maes)))

model.fit(X_all, y)
baseline_pred = model.predict(test_feat[feature_cols]).astype("float64")

test = test.copy()
test["baseline_type_median"] = (
    baseline_pred  # keep column name to minimize downstream changes
)
print(test[["id", "type", "baseline_type_median"]].head())




## === cell 2
def read_pred_file(path: str, target_col: str = TARGET) -> pd.Series:
    """
    Read a prediction file and return a Series indexed by id.
    Accepts:
      - submission-like CSV with columns ['id', target_col]
      - CSV whose first column is id (index) and one prediction column (or target_col)
    """
    path = Path(path)
    df = pd.read_csv(path)

    if "id" in df.columns:
        id_col = "id"
        if target_col in df.columns:
            pred_col = target_col
        else:
            non_id_cols = [c for c in df.columns if c != "id"]
            if len(non_id_cols) != 1:
                raise ValueError(
                    f"Can't infer prediction column in {path}. Columns: {df.columns.tolist()}"
                )
            pred_col = non_id_cols[0]
        s = df.set_index(id_col)[pred_col]
        s.name = path.stem
        return s

    df = pd.read_csv(path, index_col=0)
    if target_col in df.columns:
        s = df[target_col]
    elif df.shape[1] == 1:
        s = df.iloc[:, 0]
    else:
        raise ValueError(
            f"Can't infer prediction column in {path}. Columns: {df.columns.tolist()}"
        )
    s.index.name = "id"
    s.name = path.stem
    return s


def get_median_from_files(files) -> pd.Series:
    series_list = []
    for f in files:
        f = Path(f)
        if not f.exists():
            print(f"Missing optional file, skipping: {f}")
            continue
        try:
            series_list.append(read_pred_file(str(f)))
            print(f"Loaded: {f}")
        except Exception as e:
            print(f"Failed reading {f} ({type(e).__name__}: {e}), skipping.")
    if len(series_list) == 0:
        return pd.Series(dtype="float64")
    concat_sub = pd.concat(series_list, axis=1, sort=True)
    med = concat_sub.median(axis=1)
    med.name = "median_ens"
    return med


def maybe_merge_series(colname: str, path: str):
    p = Path(path)
    if not p.exists():
        print(f"Missing optional file, skipping: {p}")
        return None
    s = read_pred_file(str(p))
    test_merge = test.merge(
        s.rename(colname), left_on="id", right_index=True, how="left"
    )
    return test_merge


for c in [
    "n1",
    "n2",
    "lgb_a",
    "lgb_m",
    "nnet",
    "nnet_cont",
    "final_mpnn",
    "mpnn",
    "lb",
]:
    test[c] = np.nan

n1_med = get_median_from_files(
    [
        "../input/champ-preds/gnn_median_2302.csv",
        "../input/champ-preds/gnn_median_2301.csv",
        "../input/champ-preds/gnn_median_adjusted_1JHC_2296.csv",
    ]
)
if len(n1_med) > 0:
    test = test.merge(
        n1_med.rename("n1_med"), left_on="id", right_index=True, how="left"
    )
else:
    test["n1_med"] = np.nan

tmp = maybe_merge_series("gnn_2312", "../input/champ-preds/gnn_median_65_68_2312.csv")
if tmp is not None:
    test = tmp
tmp = maybe_merge_series("lastgnn", "../input/champ-preds/gnn_median_69_73.csv")
if tmp is not None:
    test = tmp

if (
    ("lastgnn" in test.columns)
    or ("gnn_2312" in test.columns)
    or ("n1_med" in test.columns)
):
    lastgnn = test["lastgnn"] if "lastgnn" in test.columns else np.nan
    gnn_2312 = test["gnn_2312"] if "gnn_2312" in test.columns else np.nan
    n1_med_col = test["n1_med"]
    test["n1"] = lastgnn * 0.5 + gnn_2312 * 0.3 + n1_med_col * 0.2

tmp = maybe_merge_series("n2", "../input/champ-preds/gnn0_median_2068.csv")
if tmp is not None:
    test = tmp

lgb_a_med = get_median_from_files(
    [
        "../input/champ-preds/submission_type_2100.csv",
        "../input/champ-preds/submission_type_2085.csv",
    ]
)
if len(lgb_a_med) > 0:
    test = test.merge(
        lgb_a_med.rename("lgb_a"), left_on="id", right_index=True, how="left"
    )

lgb_m_med = get_median_from_files(
    [
        "../input/champ-preds/lgb_type_full_f286_10.csv",
        "../input/champ-preds/lgb_type_full_f262_10.csv",
    ]
)
if len(lgb_m_med) > 0:
    test = test.merge(
        lgb_m_med.rename("lgb_m"), left_on="id", right_index=True, how="left"
    )

nnet_med = get_median_from_files(
    [
        "../input/nnpvals-seed-20/nnet_sub_s11.csv",
        "../input/champ-preds/nnet_sub.csv",
        "../input/nn-seed-10/nnet_sub.csv",
        "../input/nn-seed-11/nnet_sub.csv",
        "../input/nnet-c-seed-10/lgb_type_cv-1.7126_mae0.23572_fd5_10.csv",
        "../input/nnet-c-seed-11/lgb_type_cv-1.70994_mae0.23497_fd5_11.csv",
        "../input/nnet-c-seed-12/lgb_type_cv-1.71029_mae0.23523_fd5_12.csv",
        "../input/nnet-b-seed-10/lgb_type_cv-1.72296_mae0.24019_bags-1_f120_fd5_10.csv",
        "../input/nnet-b-seed-11/lgb_type_cv-1.70647_mae0.2376_bags-1_f120_fd5_11.csv",
        "../input/nnet-b-seed-12/lgb_type_cv-1.69833_mae0.24106_bags-1_f120_fd5_12.csv",
        "../input/nnet-try-seed-11/lgb_type_cv-1.64944_mae0.23965_bags-1_f120_fd5_11.csv",
    ]
)
if len(nnet_med) > 0:
    test = test.merge(
        nnet_med.rename("nnet"), left_on="id", right_index=True, how="left"
    )

nnet_cont_med = get_median_from_files(
    [
        "../input/nncont-seed-23-p/nnetCont_sub_only_predict.csv",
        "../input/nncont-seed-22/nnetCont_sub-1.8213.csv",
    ]
)
if len(nnet_cont_med) > 0:
    test = test.merge(
        nnet_cont_med.rename("nnet_cont_med"),
        left_on="id",
        right_index=True,
        how="left",
    )
else:
    test["nnet_cont_med"] = np.nan

tmp = maybe_merge_series(
    "nn_contof", "../input/nncont-seed-26/nnetCont_sub_-2.6677.csv"
)
if tmp is not None:
    test = tmp

if ("nn_contof" in test.columns) or ("nnet_cont_med" in test.columns):
    nn_contof = test["nn_contof"] if "nn_contof" in test.columns else np.nan
    test["nnet_cont"] = nn_contof * 0.6 + test["nnet_cont_med"] * 0.4

tmp = maybe_merge_series("final_mpnn", "../input/champ-preds/final_mpnn.csv")
if tmp is not None:
    test = tmp
tmp = maybe_merge_series("mpnn", "../input/champ-preds/mpnn_5_fold_pseudo_1449.csv")
if tmp is not None:
    test = tmp
tmp = maybe_merge_series(
    "lb", "../input/chemistry-of-best-models-1-895/stack_median.csv"
)
if tmp is not None:
    test = tmp

for c in [
    "n1",
    "n2",
    "lgb_a",
    "lgb_m",
    "nnet",
    "nnet_cont",
    "final_mpnn",
    "mpnn",
    "lb",
]:
    test[c] = test[c].astype("float64")
    test[c] = test[c].fillna(test["baseline_type_median"])

print(test[["id", "type", "baseline_type_median", "n1", "n2"]].head())




## === cell 3
test["nnet_ens"] = test["nnet_cont"] * 0.6 + test["nnet"] * 0.4
test["lgb_ens"] = test["lgb_a"] * 0.8 + test["lgb_m"] * 0.2
print("Built ensemble helper columns: nnet_ens, lgb_ens")




## === cell 4
test["final_preds"] = (
    test["n1"] * 0.65
    + test["n2"] * 0.06
    + test["lgb_ens"] * 0.14
    + test["nnet_ens"] * 0.06
    + test["lb"] * 0.07
    + test["final_mpnn"] * 0.02
)

test["final_preds"] = (
    test["final_preds"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(test["baseline_type_median"])
)
print(test[["id", "final_preds"]].head(10))




## === cell 5
submission = pd.DataFrame(
    {"id": test["id"].astype(int), TARGET: test["final_preds"].astype("float64")}
)

submission = sample_submission[["id"]].merge(submission, on="id", how="left")
submission[TARGET] = submission[TARGET].fillna(submission[TARGET].median())

out_path = "ensemble_sub.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", submission.shape)
print(submission.head())




## === cell 6
submission.head(20)
