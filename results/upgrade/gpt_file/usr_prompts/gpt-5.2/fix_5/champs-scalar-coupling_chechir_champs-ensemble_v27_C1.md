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

-2.3117754596968823

# 6. Current score

3.16394

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.99777) has done: 'I fix the pipeline so it no longer crashes when the referenced “../input/*” model-prediction files are missing in your environment. The main change is to make prediction loading robust: if an external file isn’t found, we fall back to a safe baseline (zeros) while keeping the ensembling logic intact and ensuring all required columns exist. I also align all loaded predictions by `id` (instead of relying on row order/index_col quirks) to prevent silent misalignment bugs. Finally, I always write a valid `submission.csv` with the required columns and `.csv` suffix.'
- What this solution (achieved 1.99777) has done: 'Your current score is far worse than the target (lower is better), and the biggest reason is that almost all referenced `../input/...` prediction files are missing here, so the ensemble collapses to (mostly) zeros. To move the score toward the target while preserving your ensembling logic, I add a minimal fallback: train a fast per-`type` baseline model (ridge regression) on `train.csv` using only geometric + atom-type features from `structures.csv`, and use its predictions only for rows where the external predictions are missing (all-zero across ensemble sources). This keeps your original weighted blend intact when real OOF/submission predictions are available, but prevents the catastrophic all-zero submission when they aren’t. The approach is deterministic, runs within the time budget, and still writes a valid `submission.csv`.'
- What this solution (achieved 2.41977) has done: 'I fix the crash in the fallback feature builder by avoiding pandas `string`/`pd.NA` comparisons in `np.where`, which currently triggers “boolean value of NA is ambiguous” when any merged atom labels are missing. The minimal change is to ensure `atom_0`/`atom_1` are plain Python strings with a safe fill value before building `atom_pair`, making the comparison well-defined and deterministic. I also add a small safety check that merged structure coordinates exist (otherwise distance features become NaN), and fill any remaining NaNs in the final fallback design matrix with zeros so Ridge prediction can’t error. This should restore end-to-end execution and keep the ensemble logic identical, only improving score versus the all-zero external-preds case by enabling the intended Ridge fallback.'
- What this solution (achieved 3.16394) has done: 'Your current score (2.41977, lower-is-better) is far from the target (-2.3118), so we should improve (decrease) the error without changing your ensemble’s core logic. The biggest low-risk gain here is to make the Ridge fallback (used when external prediction files are missing) stronger while keeping the same “only replace when all external preds are zero” behavior. I minimally extend the fallback features by merging in Mulliken charges and magnetic shielding tensors for the two atoms and adding simple aggregate tensor statistics; this is still the same training approach (single Ridge fit) and same prediction semantics. I also ensure the fallback design matrix remains fully numeric and NaN-safe so it runs end-to-end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os, glob
import pandas as pd
import numpy as np

for p in [
    "../input/1-mpn",
    "../input/",
    "../input/nnet-b-seed-10",
    "../input/nnet-b-seed-11",
    "../input/nnet-b-seed-12",
    "../input/champ-preds",
]:
    print(f"Listing: {p}")
    if os.path.exists(p):
        print("\n".join(os.listdir(p)[:20]))
    else:
        print("Path not found (OK for this environment).")



## === cell 1
import pandas as pd
import numpy as np

BASE_PATH = "/kaggle/data/champs-scalar-coupling"
test_path = os.path.join(BASE_PATH, "test.csv")
sample_path = os.path.join(BASE_PATH, "sample_submission.csv")

test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

TARGET = "scalar_coupling_constant"

assert "id" in test.columns, "test.csv must have an 'id' column"
assert (
    "id" in sample_sub.columns and TARGET in sample_sub.columns
), "sample_submission must have id and scalar_coupling_constant"

test.shape, sample_sub.shape




## === cell 2
def _safe_read_pred_series(path, ids, target_col=TARGET):
    """
    Robustly load predictions from a CSV and align to provided ids.
    Supports two common formats:
      - columns: ['id', 'scalar_coupling_constant']
      - index as id (first column), with a prediction column
    If file is missing/unreadable, returns zeros (score will be poor but submission valid).
    """
    ids = pd.Index(ids)
    if (path is None) or (not os.path.exists(path)):
        return pd.Series(
            np.zeros(len(ids), dtype=np.float64), index=ids, name=target_col
        )

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.Series(
            np.zeros(len(ids), dtype=np.float64), index=ids, name=target_col
        )

    if "id" in df.columns:
        if target_col not in df.columns:
            if df.shape[1] == 2:
                pred_col = [c for c in df.columns if c != "id"][0]
                s = df.set_index("id")[pred_col]
            else:
                return pd.Series(
                    np.zeros(len(ids), dtype=np.float64), index=ids, name=target_col
                )
        else:
            s = df.set_index("id")[target_col]
        return s.reindex(ids).fillna(0.0).astype(np.float64)

    if df.shape[1] >= 2:
        first = df.columns[0]
        pred_col = target_col if target_col in df.columns else df.columns[-1]
        try:
            tmp = df.set_index(first)[pred_col]
            tmp.index = pd.to_numeric(tmp.index, errors="ignore")
            return tmp.reindex(ids).fillna(0.0).astype(np.float64)
        except Exception:
            return pd.Series(
                np.zeros(len(ids), dtype=np.float64), index=ids, name=target_col
            )

    return pd.Series(np.zeros(len(ids), dtype=np.float64), index=ids, name=target_col)


def get_median_from_files(files, ids):
    """
    Load multiple prediction files robustly and take median per id.
    Missing files contribute a zero series; this keeps the pipeline runnable.
    """
    series_list = []
    for f in files:
        series_list.append(_safe_read_pred_series(f, ids))
    concat_sub = pd.concat(series_list, axis=1)
    champ_median = concat_sub.median(axis=1).values
    return champ_median


ids = test["id"].values

test["n1"] = get_median_from_files(
    [
        "../input/champ-preds/gnn_median_2302.csv",
        "../input/champ-preds/gnn_median_2301.csv",
        "../input/champ-preds/gnn_median_adjusted_1JHC_2296.csv",
    ],
    ids=ids,
)

gnn_2312 = _safe_read_pred_series(
    "../input/champ-preds/gnn_median_65_68_2312.csv", ids
).values
test["lastgnn"] = _safe_read_pred_series(
    "../input/champ-preds/gnn_median_69_73.csv", ids
).values
test["n1"] = gnn_2312 * 0.6 + test["n1"] * 0.4

test["n2"] = _safe_read_pred_series(
    "../input/champ-preds/gnn0_median_2068.csv", ids
).values

test["lgb_a"] = get_median_from_files(
    [
        "../input/champ-preds/submission_type_2100.csv",
        "../input/champ-preds/submission_type_2085.csv",
    ],
    ids=ids,
)

test["lgb_m"] = get_median_from_files(
    [
        "../input/champ-preds/lgb_type_full_f286_10.csv",
        "../input/champ-preds/lgb_type_full_f262_10.csv",
    ],
    ids=ids,
)

test["nnet"] = get_median_from_files(
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
    ],
    ids=ids,
)

test["nnet_cont"] = get_median_from_files(
    [
        "../input/nncont-seed-23-p/nnetCont_sub_only_predict.csv",
        "../input/nncont-seed-22/nnetCont_sub-1.8213.csv",
    ],
    ids=ids,
)

nn_contof = _safe_read_pred_series(
    "../input/nncont-seed-26/nnetCont_sub_-2.6677.csv", ids
).values
test["nnet_cont"] = nn_contof * 0.6 + test["nnet_cont"] * 0.4

test["final_mpnn"] = _safe_read_pred_series(
    "../input/champ-preds/final_mpnn.csv", ids
).values
test["mpnn"] = _safe_read_pred_series(
    "../input/champ-preds/mpnn_5_fold_pseudo_1449.csv", ids
).values

test["lb"] = _safe_read_pred_series(
    "../input/chemistry-of-best-models-1-895/stack_median.csv", ids
).values

test.head(3)



## === cell 3
import matplotlib.pyplot as plt
import seaborn as sns

test["nnet_ens"] = test["nnet_cont"] * 0.6 + test["nnet"] * 0.4
test["lgb_ens"] = test["lgb_a"] * 0.8 + test["lgb_m"] * 0.2

cols = [
    c
    for c in test.columns
    if c not in ["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]
]
corr = test[cols].corr(numeric_only=True)

plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap="viridis", square=True)
plt.tight_layout()
corr



## === cell 4
test["final_preds"] = (
    test["n1"] * 0.65
    + test["n2"] * 0.07
    + test["lgb_ens"] * 0.15
    + test["nnet_ens"] * 0.06
    + test["lb"] * 0.07
)

test[["id", "final_preds"]].head(5)



## === cell 5
from sklearn.linear_model import Ridge

train_path = os.path.join(BASE_PATH, "train.csv")
structures_path = os.path.join(BASE_PATH, "structures.csv")
mulliken_path = os.path.join(BASE_PATH, "mulliken_charges.csv")
mst_path = os.path.join(BASE_PATH, "magnetic_shielding_tensors.csv")

train = pd.read_csv(
    train_path,
    usecols=["molecule_name", "atom_index_0", "atom_index_1", "type", TARGET],
)
structures = pd.read_csv(
    structures_path, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
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

mulliken = pd.read_csv(
    mulliken_path, usecols=["molecule_name", "atom_index", "mulliken_charge"]
)
m0 = mulliken.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_0"}
)
m1 = mulliken.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_1"}
)

mst_cols = [
    "molecule_name",
    "atom_index",
    "XX",
    "YX",
    "ZX",
    "XY",
    "YY",
    "ZY",
    "XZ",
    "YZ",
    "ZZ",
]
mst = pd.read_csv(mst_path, usecols=mst_cols)
t0 = mst.rename(
    columns={c: f"t0_{c}" for c in mst_cols if c not in ["molecule_name", "atom_index"]}
    | {"atom_index": "atom_index_0"}
)
t1 = mst.rename(
    columns={c: f"t1_{c}" for c in mst_cols if c not in ["molecule_name", "atom_index"]}
    | {"atom_index": "atom_index_1"}
)

tensor_elems = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
t0_elem_cols = [f"t0_{c}" for c in tensor_elems]
t1_elem_cols = [f"t1_{c}" for c in tensor_elems]


def _add_geo_atom_features(df):
    df = df.merge(s0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(s1, on=["molecule_name", "atom_index_1"], how="left")

    df = df.merge(m0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(m1, on=["molecule_name", "atom_index_1"], how="left")
    df = df.merge(t0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(t1, on=["molecule_name", "atom_index_1"], how="left")

    for c in ["x0", "y0", "z0", "x1", "y1", "z1"]:
        if c not in df.columns:
            df[c] = np.nan

    dx = df["x0"] - df["x1"]
    dy = df["y0"] - df["y1"]
    dz = df["z0"] - df["z1"]
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)
    df["dist2"] = (df["dist"] * df["dist"]).astype(np.float32)

    a0 = df["atom_0"].astype("object").fillna("X").astype(str)
    a1 = df["atom_1"].astype("object").fillna("X").astype(str)
    df["atom_pair"] = np.where(a0 <= a1, a0 + "_" + a1, a1 + "_" + a0)

    df["mulliken_sum"] = (df["mulliken_0"] + df["mulliken_1"]).astype(np.float32)
    df["mulliken_diff"] = (df["mulliken_0"] - df["mulliken_1"]).astype(np.float32)

    df["t0_trace"] = (df.get("t0_XX") + df.get("t0_YY") + df.get("t0_ZZ")).astype(
        np.float32
    )
    df["t1_trace"] = (df.get("t1_XX") + df.get("t1_YY") + df.get("t1_ZZ")).astype(
        np.float32
    )

    df["t0_abs_mean"] = df[t0_elem_cols].abs().mean(axis=1).astype(np.float32)
    df["t1_abs_mean"] = df[t1_elem_cols].abs().mean(axis=1).astype(np.float32)

    keep = [
        "type",
        "atom_pair",
        "dist",
        "dist2",
        "mulliken_sum",
        "mulliken_diff",
        "t0_trace",
        "t1_trace",
        "t0_abs_mean",
        "t1_abs_mean",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    return df[keep]


Xtr_cat = _add_geo_atom_features(train)
ytr = train[TARGET].astype(np.float32).values

Xtr_ohe = pd.get_dummies(Xtr_cat[["type", "atom_pair"]], drop_first=False, sparse=False)
Xtr_num = (
    Xtr_cat.drop(columns=["type", "atom_pair"])
    .astype(np.float32)
    .reset_index(drop=True)
)
Xtr = np.hstack([Xtr_num.values, Xtr_ohe.values]).astype(np.float32)
Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)

ridge = Ridge(alpha=1.0, random_state=0)
ridge.fit(Xtr, ytr)

Xte_cat = _add_geo_atom_features(test)
Xte_ohe = pd.get_dummies(Xte_cat[["type", "atom_pair"]], drop_first=False, sparse=False)
Xte_ohe = Xte_ohe.reindex(columns=Xtr_ohe.columns, fill_value=0.0)
Xte_num = (
    Xte_cat.drop(columns=["type", "atom_pair"])
    .astype(np.float32)
    .reset_index(drop=True)
)
Xte = np.hstack([Xte_num.values, Xte_ohe.values]).astype(np.float32)
Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)

fallback_preds = ridge.predict(Xte).astype(np.float64)

pred_source_cols = ["n1", "n2", "lgb_ens", "nnet_ens", "lb"]
all_zero_mask = (test[pred_source_cols].abs().sum(axis=1) < 1e-12).values

print(
    "Rows with missing external preds (using fallback):",
    int(all_zero_mask.sum()),
    "/",
    len(test),
)

test.loc[all_zero_mask, "final_preds"] = fallback_preds[all_zero_mask]

test[["id", "final_preds"]].head(5)



## === cell 6
submission = pd.DataFrame({"id": test["id"].values, TARGET: test["final_preds"].values})

submission = submission.sort_values("id").reset_index(drop=True)

submission.to_csv("submission.csv", index=False)
submission.to_csv("last_gnn.csv", index=False)

submission.head(10), submission.shape



## === cell 7
assert submission.columns.tolist() == ["id", TARGET]
assert submission["id"].isna().sum() == 0
assert submission[TARGET].isna().sum() == 0
print("Wrote submission.csv with", len(submission), "rows")
print(submission.describe(include="all"))
