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

# 8. Previous improvement plan

- What this solution (achieved 1.18497) has done: 'The errors come from relying on external Kaggle “../input/champ-preds” and “../input/nnet-b-seed-*” folders that don’t exist in your environment, so the prediction columns never get created and downstream cells fail. I keep the ensemble/blending core logic intact, but make it robust by (1) auto-discovering which of those input folders/files are actually present, (2) reading any available prediction files and aligning them to `test` by `id` (not by row order), and (3) falling back to a safe baseline (type-wise median from `train.csv`) when some/all external preds are missing. Finally, I ensure a valid `submission.csv` with the required columns is always written.'

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
safe_listdir("../input/champ-preds")



## === cell 2
import numpy as np
import pandas as pd

BASE_INPUT = Path("../input/champs-scalar-coupling")
ALT_INPUT = Path("/kaggle/data/champs-scalar-coupling")  # safety for nonstandard mounts


def resolve_input_file(fname: str) -> Path:
    """Find file in known dataset locations."""
    for base in (BASE_INPUT, ALT_INPUT, Path("../input")):
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
      - or an unnamed first column as index (common in some kernels)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(p)

    if id_col not in df.columns and df.columns[0].lower().startswith("unnamed"):
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

    s = df.set_index(id_col)[target_col]
    s.name = p.stem
    return s


def get_median_from_files(files, test_ids: pd.Series, target_col: str) -> pd.Series:
    """
    Take median across provided submission files, aligned by id.
    """
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


test = pd.read_csv(resolve_input_file("test.csv"))
TARGET = "scalar_coupling_constant"

train = pd.read_csv(resolve_input_file("train.csv"), usecols=["type", TARGET])
type_median = train.groupby("type")[TARGET].median()

pred_sources = {}

n1_path = Path("../input/champ-preds/gnn_median_2279.csv")
n2_path = Path("../input/champ-preds/gnn_train_sep_2258.csv")

if n1_path.exists():
    pred_sources["n1"] = (
        read_pred_file(str(n1_path), target_col=TARGET)
        .reindex(test["id"].values)
        .reset_index(drop=True)
    )
else:
    pred_sources["n1"] = pd.Series(np.nan, index=np.arange(len(test)), name="n1")

if n2_path.exists():
    pred_sources["n2"] = (
        read_pred_file(str(n2_path), target_col=TARGET)
        .reindex(test["id"].values)
        .reset_index(drop=True)
    )
else:
    pred_sources["n2"] = pd.Series(np.nan, index=np.arange(len(test)), name="n2")

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


def try_median(files, name):
    try:
        s = get_median_from_files(files, test_ids=test["id"], target_col=TARGET)
        s.name = name
        return s.reset_index(drop=True)
    except Exception as e:
        print(f"{name}: falling back due to {type(e).__name__}: {e}")
        return pd.Series(np.nan, index=np.arange(len(test)), name=name)


pred_sources["lgb_a"] = try_median(lgb_a_files, "lgb_a")
pred_sources["lgb_m"] = try_median(lgb_m_files, "lgb_m")
pred_sources["nnet"] = try_median(nnet_files, "nnet")

for k, s in pred_sources.items():
    test[k] = s.values

baseline = test["type"].map(type_median).astype(float)
for c in ["n1", "n2", "lgb_a", "lgb_m", "nnet"]:
    test[c] = test[c].astype(float)
    miss = test[c].isna()
    if miss.any():
        test.loc[miss, c] = baseline.loc[miss].values

test.head(10)



## === cell 3
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
sns.heatmap(test[["n1", "n2", "lgb_a", "lgb_m", "nnet"]].corr(), ax=ax)
test[["n1", "n2", "lgb_a", "lgb_m", "nnet"]].corr()



## === cell 4
test["final_preds"] = (
    test["n1"] * 0.65
    + test["n2"] * 0.05
    + test["lgb_a"] * 0.15
    + test["lgb_m"] * 0.10
    + test["nnet"] * 0.05
)
test.head(20)



## === cell 5
submission = pd.DataFrame({"id": test["id"].astype(int)})

submission["scalar_coupling_constant"] = test["final_preds"].astype(float)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Wrote {submission_path} with shape {submission.shape}")
submission.head()



## === cell 6
submission.head(20)
