# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

-1.65028845049191

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import zipfile
import numpy as np
import pandas as pd

INPUT_DIR = "/kaggle/input/champs-scalar-coupling"
WORKING_DIR = "/kaggle/working"


def resolve_comp_dir(base_dir: str) -> str:
    direct_train = os.path.join(base_dir, "train.csv")
    if os.path.exists(direct_train):
        return base_dir
    nested = os.path.join(base_dir, "champs-scalar-coupling")
    nested_train = os.path.join(nested, "train.csv")
    if os.path.exists(nested_train):
        return nested
    for name in os.listdir(base_dir):
        cand = os.path.join(base_dir, name)
        if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "train.csv")):
            return cand
    raise FileNotFoundError(
        f"Could not find train.csv under {base_dir} or its subdirectories."
    )


COMP_DIR = resolve_comp_dir(INPUT_DIR)

print("Listing input dir:", INPUT_DIR)
print("Resolved competition dir:", COMP_DIR)
print("Top-level contents:", os.listdir(INPUT_DIR)[:30])
print("Comp dir contents:", os.listdir(COMP_DIR)[:30])



## === cell 1
train_path = os.path.join(COMP_DIR, "train.csv")
test_path = os.path.join(COMP_DIR, "test.csv")
structures_csv_path = os.path.join(COMP_DIR, "structures.csv")
structures_zip_path = os.path.join(COMP_DIR, "structures.zip")
sample_path = os.path.join(COMP_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

print(train.shape, test.shape, sample_sub.shape)
print("train cols:", train.columns.tolist())
print("test cols:", test.columns.tolist())

structures_csv = None
if os.path.exists(structures_csv_path):
    structures_csv = pd.read_csv(structures_csv_path)
    print("structures.csv shape:", structures_csv.shape)
    print("structures.csv cols:", structures_csv.columns.tolist())
else:
    print("structures.csv not found; will rely on structures.zip")

print(
    "structures.zip exists:", os.path.exists(structures_zip_path), structures_zip_path
)



## === cell 2
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

for df in (train, test):
    df["atom_index_0"] = df["atom_index_0"].astype(np.int32)
    df["atom_index_1"] = df["atom_index_1"].astype(np.int32)
    df["molecule_name"] = df["molecule_name"].astype(str)
    df["type"] = df["type"].astype(str)

if structures_csv is not None:
    structures_csv["atom_index"] = structures_csv["atom_index"].astype(np.int32)
    structures_csv["molecule_name"] = structures_csv["molecule_name"].astype(str)
    structures_csv["atom"] = structures_csv["atom"].astype(str)
    for c in ["x", "y", "z"]:
        structures_csv[c] = structures_csv[c].astype(np.float32)


def load_structures_from_zip(zip_path: str, molecules) -> pd.DataFrame:
    """
    Bugfix (root cause of merge NaNs):
    The XYZ files in this competition include an explicit atom index column:
        <atom_index> <atom_symbol> <x> <y> <z>
    The previous implementation used enumerate(atom_lines) as atom_index, which does NOT
    match train/test atom_index values, causing massive missing merges.
    Here we parse the explicit index when present; otherwise fall back to enumerate.
    """
    molecules = pd.Index(pd.unique(pd.Series(list(molecules)).astype(str)))
    mol_set = set(molecules.tolist())
    rows = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        xyz_members = [n for n in names if n.lower().endswith(".xyz")]

        needed = []
        for member in xyz_members:
            base = os.path.basename(member)
            mol = os.path.splitext(base)[0]
            if mol in mol_set:
                needed.append((mol, member))

        if len(needed) == 0:
            preview = xyz_members[:10]
            raise FileNotFoundError(
                "No required molecule XYZ files found inside structures.zip. "
                f"Looked for molecule basenames in zip members. Example xyz members: {preview}"
            )

        for mol, member in needed:
            with zf.open(member) as f:
                content = f.read().decode("utf-8", errors="replace").splitlines()
            if len(content) < 3:
                continue
            try:
                n_atoms = int(content[0].strip())
            except Exception:
                continue

            atom_lines = content[2 : 2 + n_atoms]
            for fallback_idx, line in enumerate(atom_lines):
                parts = line.strip().split()
                if len(parts) < 4:
                    continue

                atom_index = None
                atom = None
                x = y = z = None

                if len(parts) >= 5:
                    try:
                        atom_index = int(parts[0])
                        atom = parts[1]
                        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    except Exception:
                        atom_index = None

                if atom_index is None:
                    try:
                        atom_index = fallback_idx
                        atom = parts[0]
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    except Exception:
                        continue

                rows.append((mol, atom_index, atom, x, y, z))

    out = pd.DataFrame(
        rows, columns=["molecule_name", "atom_index", "atom", "x", "y", "z"]
    )
    if out.empty:
        raise ValueError(
            "Parsed 0 structure rows from structures.zip; zip content may be malformed."
        )

    out["atom_index"] = out["atom_index"].astype(np.int32)
    out["x"] = out["x"].astype(np.float32)
    out["y"] = out["y"].astype(np.float32)
    out["z"] = out["z"].astype(np.float32)
    out["molecule_name"] = out["molecule_name"].astype(str)
    out["atom"] = out["atom"].astype(str)
    return out


needed_molecules = pd.Index(
    pd.concat([train["molecule_name"], test["molecule_name"]]).unique()
)

use_zip = True
if structures_csv is not None:
    have_mols = set(structures_csv["molecule_name"].unique().tolist())
    missing_mols = [m for m in needed_molecules.tolist() if m not in have_mols]
    if len(missing_mols) == 0:
        structures_small = structures_csv[
            ["molecule_name", "atom_index", "atom", "x", "y", "z"]
        ].copy()

        a0_try = structures_small.rename(
            columns={
                "atom_index": "atom_index_0",
                "atom": "atom_0",
                "x": "x0",
                "y": "y0",
                "z": "z0",
            }
        )
        a1_try = structures_small.rename(
            columns={
                "atom_index": "atom_index_1",
                "atom": "atom_1",
                "x": "x1",
                "y": "y1",
                "z": "z1",
            }
        )

        test_sample = test.head(20000).copy()
        merged = test_sample.merge(
            a0_try, on=["molecule_name", "atom_index_0"], how="left"
        ).merge(a1_try, on=["molecule_name", "atom_index_1"], how="left")

        if not merged[["x0", "y0", "z0", "x1", "y1", "z1"]].isna().any().any():
            use_zip = False

if use_zip:
    if not os.path.exists(structures_zip_path):
        raise FileNotFoundError(
            "structures.zip not found, and structures.csv appears incomplete. "
            f"Looked for {structures_zip_path}"
        )
    print(
        "Using structures.zip to build structures for required molecules:",
        len(needed_molecules),
    )
    structures = load_structures_from_zip(structures_zip_path, needed_molecules)
else:
    print("Using structures.csv (coverage check passed).")
    structures = structures_csv[
        ["molecule_name", "atom_index", "atom", "x", "y", "z"]
    ].copy()

structures["molecule_name"] = structures["molecule_name"].astype(str)
structures["atom"] = structures["atom"].astype(str)

structures_small = structures[
    ["molecule_name", "atom_index", "atom", "x", "y", "z"]
].copy()

a0 = structures_small.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)
a1 = structures_small.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(a0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(a1, on=["molecule_name", "atom_index_1"], how="left")

    if df[["x0", "y0", "z0", "x1", "y1", "z1"]].isna().any().any():
        missing = int(df[["x0", "y0", "z0", "x1", "y1", "z1"]].isna().sum().sum())
        bad = df[df[["x0", "y0", "z0", "x1", "y1", "z1"]].isna().any(axis=1)].head(5)
        raise ValueError(
            f"Found {missing} missing coordinate values after merging structures. "
            f"Example rows:\n{bad[['molecule_name','atom_index_0','atom_index_1','type']].to_string(index=False)}"
        )

    dx = (df["x0"] - df["x1"]).astype(np.float32)
    dy = (df["y0"] - df["y1"]).astype(np.float32)
    dz = (df["z0"] - df["z1"]).astype(np.float32)
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)
    return df


train_fe = build_features(train)
test_fe = build_features(test)

feature_cols_num = ["dist"]
feature_cols_cat = ["type", "atom_0", "atom_1"]

X_train = train_fe[feature_cols_num + feature_cols_cat]
y_train = train_fe["scalar_coupling_constant"].astype(np.float32)
X_test = test_fe[feature_cols_num + feature_cols_cat]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
        ("num", "passthrough", feature_cols_num),
    ],
    remainder="drop",
)

model = Ridge(alpha=1.0, random_state=0)
pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])

max_rows = 1_200_000
if len(X_train) > max_rows:
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X_train), size=max_rows, replace=False)
    X_fit = X_train.iloc[idx]
    y_fit = y_train.iloc[idx]
else:
    X_fit = X_train
    y_fit = y_train

print("Fitting rows:", len(X_fit), " / total:", len(X_train))
pipe.fit(X_fit, y_fit)

test_pred = pipe.predict(X_test).astype(np.float32)
print(
    "Predictions:",
    test_pred.shape,
    "min/max:",
    float(np.min(test_pred)),
    float(np.max(test_pred)),
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/972262269.py in <cell line: 0>()
    211 
    212 train_fe = build_features(train)
--> 213 test_fe = build_features(test)
    214 
    215 feature_cols_num = ["dist"]

/tmp/ipykernel_11/972262269.py in build_features(df)
    198         missing = int(df[["x0", "y0", "z0", "x1", "y1", "z1"]].isna().sum().sum())
    199         bad = df[df[["x0", "y0", "z0", "x1", "y1", "z1"]].isna().any(axis=1)].head(5)
--> 200         raise ValueError(
    201             f"Found {missing} missing coordinate values after merging structures. "
    202             f"Example rows:\n{bad[['molecule_name','atom_index_0','atom_index_1','type']].to_string(index=False)}"

ValueError: Found 2806878 missing coordinate values after merging structures. Example rows:
   molecule_name  atom_index_0  atom_index_1 type
dsgdb9nsd_071451             9             0 1JHC
dsgdb9nsd_071451             9             1 2JHC
dsgdb9nsd_071451             9             4 3JHC
dsgdb9nsd_071451             9             5 3JHC
dsgdb9nsd_071451             9            10 2JHH

## === cell 3
if "test_pred" not in globals():
    raise NameError(
        "test_pred is not defined; model training/prediction did not complete."
    )

sub = pd.DataFrame({"id": test["id"].values, "scalar_coupling_constant": test_pred})

sub = sample_sub[["id"]].merge(sub, on="id", how="left")
if sub["scalar_coupling_constant"].isna().any():
    missing_ids = sub.loc[sub["scalar_coupling_constant"].isna(), "id"].head(5).tolist()
    raise ValueError(
        f"Submission has missing predictions after id alignment. Example ids: {missing_ids}"
    )

out_path = os.path.join(WORKING_DIR, "submission.csv")
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())
print("Submission shape:", sub.shape)
print("Columns:", sub.columns.tolist())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3358528899.py in <cell line: 0>()
      1 if "test_pred" not in globals():
----> 2     raise NameError(
      3         "test_pred is not defined; model training/prediction did not complete."
      4     )
      5 

NameError: test_pred is not defined; model training/prediction did not complete.
