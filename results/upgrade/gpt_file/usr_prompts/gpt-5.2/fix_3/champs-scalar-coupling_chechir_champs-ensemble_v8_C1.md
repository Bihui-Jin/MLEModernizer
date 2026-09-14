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

-2.1018426058695074

# 6. Current score

2.75915

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.222) has done: 'Your notebook fails because it depends on external Kaggle datasets (`../input/champ-preds` and `../input/1-mpnn`) that are not present in your provided environment; this prevents any submission from being written. I replace those missing-file reads with an in-notebook baseline that trains a fast, deterministic per-`type` linear regression using only `train.csv`/`test.csv` plus `structures.csv`-derived geometric features (distance and a few coordinate deltas), which preserves the overall “blend/ensemble-like” intent while making it runnable end-to-end. I also ensure strict ID alignment (merge back on `id`) and write a valid `submission.csv` with the exact required columns. The remaining cells be updated to display the produced submission and a preview of the test predictions, so no downstream `NameError` occurs.'
- What this solution (achieved 2.75915) has done: 'Your current Ridge-per-type baseline is leaving a lot of signal on the table because the metric is averaged per coupling `type`, and the data provides several strong, “allowed” auxiliary physics features (mulliken charges, shielding tensors, dipole moments, potential energy) that can be merged without changing the core modeling approach. I keep the exact same training loop (per-`type` Ridge in a Pipeline) and just extend the feature set with these extra merged features plus a couple of minimal, stable geometric transforms (`dist^2`, `1/dist`) that often help linear models. I also set `fit_intercept=False` since we already include scaling and type-wise training; this is a small calibration change that typically reduces MAE without altering the overall approach. The result still runs end-to-end within constraints and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

BASE = "/kaggle/data/champs-scalar-coupling"
print("Base exists:", os.path.exists(BASE))
print("Files (head):", sorted(os.listdir(BASE))[:10])



## === cell 1
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

DATA_DIR = "/kaggle/data/champs-scalar-coupling"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
struct_path = os.path.join(DATA_DIR, "structures.csv")

mulliken_path = os.path.join(DATA_DIR, "mulliken_charges.csv")
shield_path = os.path.join(DATA_DIR, "magnetic_shielding_tensors.csv")
dipole_path = os.path.join(DATA_DIR, "dipole_moments.csv")
energy_path = os.path.join(DATA_DIR, "potential_energy.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
structures = pd.read_csv(struct_path)

mulliken = pd.read_csv(mulliken_path)
shield = pd.read_csv(shield_path)
dipole = pd.read_csv(dipole_path)
energy = pd.read_csv(energy_path)

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

m0 = mulliken.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_0"}
)
m1 = mulliken.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_1"}
)

sh0 = shield.rename(columns={"atom_index": "atom_index_0"})
sh1 = shield.rename(columns={"atom_index": "atom_index_1"})
sh0 = sh0.add_prefix("sh0_").rename(
    columns={"sh0_molecule_name": "molecule_name", "sh0_atom_index_0": "atom_index_0"}
)
sh1 = sh1.add_prefix("sh1_").rename(
    columns={"sh1_molecule_name": "molecule_name", "sh1_atom_index_1": "atom_index_1"}
)


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(
        s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    out = out.merge(
        s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )

    out = out.merge(
        m0[["molecule_name", "atom_index_0", "mulliken_0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    out = out.merge(
        m1[["molecule_name", "atom_index_1", "mulliken_1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )
    out = out.merge(
        sh0,
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    out = out.merge(
        sh1,
        on=["molecule_name", "atom_index_1"],
        how="left",
    )
    out = out.merge(dipole, on="molecule_name", how="left")
    out = out.merge(energy, on="molecule_name", how="left")

    dx = out["x0"] - out["x1"]
    dy = out["y0"] - out["y1"]
    dz = out["z0"] - out["z1"]
    out["dx"] = dx
    out["dy"] = dy
    out["dz"] = dz
    out["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)

    eps = 1e-6
    out["dist2"] = out["dist"] ** 2
    out["inv_dist"] = 1.0 / (out["dist"] + eps)

    out["x0_plus_x1"] = out["x0"] + out["x1"]
    out["y0_plus_y1"] = out["y0"] + out["y1"]
    out["z0_plus_z1"] = out["z0"] + out["z1"]
    out["x0_minus_x1"] = dx
    out["y0_minus_y1"] = dy
    out["z0_minus_z1"] = dz
    return out


train_f = add_structure_features(train)
test_f = add_structure_features(test)

for col in ["atom_0", "atom_1"]:
    train_f[col] = train_f[col].astype("category")
    test_f[col] = test_f[col].astype("category")

all_atoms = pd.concat(
    [train_f[["atom_0", "atom_1"]], test_f[["atom_0", "atom_1"]]], axis=0
)
atom0_cats = all_atoms["atom_0"].astype("category").cat.categories
atom1_cats = all_atoms["atom_1"].astype("category").cat.categories
train_f["atom_0"] = pd.Categorical(train_f["atom_0"], categories=atom0_cats)
test_f["atom_0"] = pd.Categorical(test_f["atom_0"], categories=atom0_cats)
train_f["atom_1"] = pd.Categorical(train_f["atom_1"], categories=atom1_cats)
test_f["atom_1"] = pd.Categorical(test_f["atom_1"], categories=atom1_cats)

train_atoms = pd.get_dummies(train_f[["atom_0", "atom_1"]], dummy_na=True)
test_atoms = pd.get_dummies(test_f[["atom_0", "atom_1"]], dummy_na=True)
test_atoms = test_atoms.reindex(columns=train_atoms.columns, fill_value=0)

shield_cols0 = [
    c
    for c in train_f.columns
    if c.startswith("sh0_") and c not in ("sh0_molecule_name",)
]
shield_cols1 = [
    c
    for c in train_f.columns
    if c.startswith("sh1_") and c not in ("sh1_molecule_name",)
]

num_features = (
    [
        "dx",
        "dy",
        "dz",
        "dist",
        "dist2",
        "inv_dist",
        "x0",
        "y0",
        "z0",
        "x1",
        "y1",
        "z1",
        "x0_plus_x1",
        "y0_plus_y1",
        "z0_plus_z1",
        "x0_minus_x1",
        "y0_minus_y1",
        "z0_minus_z1",
        "mulliken_0",
        "mulliken_1",
        "X",
        "Y",
        "Z",
        "potential_energy",
    ]
    + shield_cols0
    + shield_cols1
)

X_train_num = train_f[num_features].astype("float32")
X_test_num = test_f[num_features].astype("float32")

X_train_full = pd.concat(
    [X_train_num.reset_index(drop=True), train_atoms.reset_index(drop=True)], axis=1
)
X_test_full = pd.concat(
    [X_test_num.reset_index(drop=True), test_atoms.reset_index(drop=True)], axis=1
)

y = train_f["scalar_coupling_constant"].astype("float32").values
train_type = train_f["type"].values
test_type = test_f["type"].values

model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
        ("ridge", Ridge(alpha=1.0, fit_intercept=False, random_state=0)),
    ]
)

pred = np.zeros(len(test_f), dtype=np.float32)

for t in pd.unique(test_type):
    idx_test = np.where(test_type == t)[0]
    idx_train = np.where(train_type == t)[0]
    if len(idx_train) == 0:
        pred[idx_test] = float(np.mean(y))
        continue
    model.fit(X_train_full.iloc[idx_train], y[idx_train])
    pred[idx_test] = model.predict(X_test_full.iloc[idx_test]).astype(np.float32)

submission = pd.DataFrame({"id": test["id"].values, "scalar_coupling_constant": pred})
submission["id"] = submission["id"].astype(np.int64)
submission = submission.sort_values("id").reset_index(drop=True)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(submission), "cols:", list(submission.columns))



## === cell 2
submission.head(20)



## === cell 3
submission.describe()
