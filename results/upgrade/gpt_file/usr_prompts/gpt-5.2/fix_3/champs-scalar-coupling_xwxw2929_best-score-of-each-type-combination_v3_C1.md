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

-1.6777209112242684

# 6. Current score

3.00563

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 3.00563) has done: 'The notebook fails because it tries to read two external submissions from `../input/keras-*` datasets that are not present in your environment, so `sub1/sub2` are undefined and the concatenation never builds a full set of test ids. To make it run end-to-end and always output a valid submission, I replace those missing inputs with a simple, deterministic baseline model trained from the provided competition files only (no extra datasets). The model preserves the original “combine predictions into one submission” intent by predicting per coupling `type` and then filling every test `id`, guaranteeing no missing ids. This should also produce a non-trivial score (better than a constant/empty submission) while keeping changes minimal and focused on correctness.'
- What this solution (achieved 3.00563) has done: 'Your current model uses only distance, which is too weak for this competition and explains the large gap to the target (lower is better). I keep the same “fit per coupling `type` and predict for test” core approach, but upgrade the per-type model from 1D linear regression on `dist` to a small closed-form linear regression on a few safe, cheap geometric features (distance powers and coordinate deltas), which better matches known signal while staying deterministic and fast. I also add light ridge regularization for numerical stability and clip extreme predictions per type to the training range to reduce MAE blow-ups that disproportionately hurt the log-MAE metric. The pipeline remains end-to-end, uses only provided competition files, and still outputs a valid `submission.csv` with all test ids.'

# 9. Code solution

## === cell 0
import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

BASE_CANDIDATES = [
    "../input/champs-scalar-coupling",
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/data/input/champs-scalar-coupling",
]
BASE_PATH = None
for p in BASE_CANDIDATES:
    if os.path.exists(p):
        BASE_PATH = p
        break

if BASE_PATH is None:
    for root, dirs, files in os.walk("/kaggle"):
        if "train.csv" in files and "test.csv" in files:
            BASE_PATH = root
            break

print("BASE_PATH:", BASE_PATH)
print("Listing BASE_PATH (first 20):", sorted(os.listdir(BASE_PATH))[:20])



## === cell 1
train = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
test = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
structures = pd.read_csv(os.path.join(BASE_PATH, "structures.csv"))

print(train.shape, test.shape, structures.shape)
print(train.head())
print(test.head())




## === cell 2
def add_pair_distance(df_pairs, structures_df):
    s = structures_df[["molecule_name", "atom_index", "atom", "x", "y", "z"]].copy()

    s0 = s.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom_0",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
    )
    s1 = s.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom_1",
            "x": "x1",
            "y": "y1",
            "z": "z1",
        }
    )

    out = df_pairs.merge(s0, how="left", on=["molecule_name", "atom_index_0"])
    out = out.merge(s1, how="left", on=["molecule_name", "atom_index_1"])

    dx = out["x0"] - out["x1"]
    dy = out["y0"] - out["y1"]
    dz = out["z0"] - out["z1"]
    out["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)

    if out["dist"].isna().any():
        out["dist"] = out["dist"].fillna(out["dist"].median())

    return out


train_feat = add_pair_distance(train, structures)
test_feat = add_pair_distance(test, structures)

print(
    train_feat[
        ["id", "type", "atom_0", "atom_1", "dist", "scalar_coupling_constant"]
    ].head()
)
print(test_feat[["id", "type", "atom_0", "atom_1", "dist"]].head())




## === cell 3
def make_design_matrix(df):
    dx = (df["x0"] - df["x1"]).to_numpy(np.float64)
    dy = (df["y0"] - df["y1"]).to_numpy(np.float64)
    dz = (df["z0"] - df["z1"]).to_numpy(np.float64)
    d = df["dist"].to_numpy(np.float64)

    d2 = d * d
    d3 = d2 * d
    invd = 1.0 / np.maximum(d, 1e-8)

    X = np.column_stack(
        [
            np.ones_like(d),
            d,
            d2,
            d3,
            invd,
            np.abs(dx),
            np.abs(dy),
            np.abs(dz),
        ]
    )
    return X


types = sorted(train_feat["type"].unique())
params = {}  # per-type: (coef vector, clip_min, clip_max)

ridge = 1e-3

for t in types:
    tr = train_feat[train_feat["type"] == t]
    y = tr["scalar_coupling_constant"].to_numpy(np.float64)
    X = make_design_matrix(tr)

    XtX = X.T @ X
    XtX.flat[:: XtX.shape[0] + 1] += ridge  # add ridge to diagonal in-place
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)

    y_lo = np.quantile(y, 0.001)
    y_hi = np.quantile(y, 0.999)

    params[t] = (beta, float(y_lo), float(y_hi))

pred = np.empty(len(test_feat), dtype=np.float64)

global_mean = float(train_feat["scalar_coupling_constant"].mean())

for t in types:
    beta, y_lo, y_hi = params[t]
    mask = test_feat["type"].values == t
    if not np.any(mask):
        continue
    Xte = make_design_matrix(test_feat.loc[mask])
    p = Xte @ beta
    p = np.clip(p, y_lo, y_hi)
    pred[mask] = p

if np.isnan(pred).any():
    pred[np.isnan(pred)] = global_mean

sub = pd.DataFrame({"id": test_feat["id"].values, "scalar_coupling_constant": pred})
sub = sub.sort_values("id").reset_index(drop=True)

print(sub.head())
print("Submission shape:", sub.shape, "Expected rows:", test.shape[0])



## === cell 4
missing = set(test["id"].values) - set(sub["id"].values)
extra = set(sub["id"].values) - set(test["id"].values)
dup = sub["id"].duplicated().sum()

print("Missing ids:", len(missing))
print("Extra ids:", len(extra))
print("Duplicate ids:", int(dup))

assert len(missing) == 0, "Invalid submission: missing required ids."
assert len(extra) == 0, "Invalid submission: extra ids."
assert dup == 0, "Invalid submission: duplicate ids."
assert list(sub.columns) == [
    "id",
    "scalar_coupling_constant",
], "Invalid submission columns."



## === cell 5
out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(pd.read_csv(out_path).head())
