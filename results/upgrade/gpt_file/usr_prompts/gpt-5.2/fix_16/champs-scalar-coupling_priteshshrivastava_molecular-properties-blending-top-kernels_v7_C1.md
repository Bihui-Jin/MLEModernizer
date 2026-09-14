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

WORKING_DIR = "/kaggle/working"


def resolve_comp_dir(base_dir: str) -> str:
    """Resolve directory that directly contains train.csv/test.csv/structures.csv."""
    direct_train = os.path.join(base_dir, "train.csv")
    if os.path.exists(direct_train):
        return base_dir

    nested = os.path.join(base_dir, "champs-scalar-coupling")
    nested_train = os.path.join(nested, "train.csv")
    if os.path.exists(nested_train):
        return nested

    if os.path.isdir(base_dir):
        for name in os.listdir(base_dir):
            cand = os.path.join(base_dir, name)
            if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "train.csv")):
                return cand

    raise FileNotFoundError(
        f"Could not find train.csv under {base_dir} or its subdirectories."
    )


def find_dataset_root() -> str:
    """
    Return the directory that actually contains train.csv/test.csv.
    Note: structures may live in a sibling mount; we handle that separately.
    """
    candidates = [
        "/kaggle/input/champs-scalar-coupling",
        "/kaggle/input/champs-scalar-coupling/champs-scalar-coupling",
        "/kaggle/data/champs-scalar-coupling",
        "/kaggle/data/champs-scalar-coupling/champs-scalar-coupling",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for c in candidates:
        try:
            if os.path.isdir(c):
                d = resolve_comp_dir(c)
                if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
                    os.path.join(d, "test.csv")
                ):
                    return d
        except FileNotFoundError:
            pass

    for root in ["/kaggle/input", "/kaggle/data"]:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            cand = os.path.join(root, name)
            if not os.path.isdir(cand):
                continue
            try:
                d = resolve_comp_dir(cand)
                if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
                    os.path.join(d, "test.csv")
                ):
                    return d
            except FileNotFoundError:
                continue

    raise FileNotFoundError(
        "Could not locate champs-scalar-coupling dataset under /kaggle/input or /kaggle/data."
    )


def _normalize_molecule_name(s: pd.Series) -> pd.Series:
    """
    Bugfix: Do NOT strip '.xyz' here.
    In this competition CSVs use molecule_name like 'dsgdb9nsd_000001' already.
    Stripping '.xyz' can corrupt names that legitimately contain the substring 'xyz'
    (e.g., 'dsgdb9nsd_071451'), causing 100% merge failure with structures.csv.
    """
    return s.astype(str).str.strip()


def read_structures_csv(path: str) -> pd.DataFrame:
    """Read structures.csv and validate columns."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"structures.csv not found at {path}")
    df = pd.read_csv(path)
    need = {"molecule_name", "atom_index", "atom", "x", "y", "z"}
    if not need.issubset(df.columns):
        raise ValueError(f"structures.csv at {path} missing required columns: {need}")
    return df


def read_structures_from_zip(structures_zip_path: str) -> pd.DataFrame:
    """Fallback loader from structures.zip (slower, but ensures full coverage)."""
    if not os.path.exists(structures_zip_path):
        raise FileNotFoundError(f"structures.zip not found at {structures_zip_path}")

    rows = []
    with zipfile.ZipFile(structures_zip_path, "r") as zf:
        names = [n for n in zf.namelist() if n.endswith(".xyz")]
        for n in names:
            base = n.rsplit("/", 1)[-1]
            mol = base[:-4] if base.lower().endswith(".xyz") else base
            mol = mol.strip()

            with zf.open(n) as f:
                content = f.read().decode("utf-8", errors="replace").splitlines()
            if len(content) < 3:
                continue
            try:
                natoms = int(content[0].strip())
            except Exception:
                continue
            atom_lines = content[2 : 2 + natoms]
            for i, line in enumerate(atom_lines):
                parts = line.split()
                if len(parts) < 4:
                    continue
                atom = parts[0]
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                except Exception:
                    continue
                rows.append((mol, i, atom, x, y, z))

    df = pd.DataFrame(
        rows, columns=["molecule_name", "atom_index", "atom", "x", "y", "z"]
    )
    df["molecule_name"] = _normalize_molecule_name(df["molecule_name"])
    return df


def find_any_existing_file(filename: str) -> str:
    """
    Sometimes COMP_DIR resolves to a copy containing train/test but not structures.
    Search common roots for the missing file and return the first hit.
    """
    roots = [
        "/kaggle/input/champs-scalar-coupling",
        "/kaggle/input/champs-scalar-coupling/champs-scalar-coupling",
        "/kaggle/data/champs-scalar-coupling",
        "/kaggle/data/champs-scalar-coupling/champs-scalar-coupling",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for r in roots:
        p = os.path.join(r, filename)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find {filename} under /kaggle/input or /kaggle/data."
    )


COMP_DIR = find_dataset_root()

print("Resolved competition dir:", COMP_DIR)
print("Comp dir contents (head):", os.listdir(COMP_DIR)[:30])

train_path = os.path.join(COMP_DIR, "train.csv")
test_path = os.path.join(COMP_DIR, "test.csv")
sample_path = os.path.join(COMP_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

train["molecule_name"] = _normalize_molecule_name(train["molecule_name"])
test["molecule_name"] = _normalize_molecule_name(test["molecule_name"])

print("train/test/sample shapes:", train.shape, test.shape, sample_sub.shape)

structures_csv_path = os.path.join(COMP_DIR, "structures.csv")
structures_zip_path = os.path.join(COMP_DIR, "structures.zip")
if not os.path.exists(structures_csv_path):
    structures_csv_path = find_any_existing_file("structures.csv")
if not os.path.exists(structures_zip_path):
    try:
        structures_zip_path = find_any_existing_file("structures.zip")
    except FileNotFoundError:
        structures_zip_path = structures_zip_path  # keep as-is

print(
    "Using structures.csv:",
    structures_csv_path,
    "| exists:",
    os.path.exists(structures_csv_path),
)
print(
    "Using structures.zip:",
    structures_zip_path,
    "| exists:",
    os.path.exists(structures_zip_path),
)

structures_csv = None
if os.path.exists(structures_csv_path):
    structures_csv = read_structures_csv(structures_csv_path)
    structures_csv["molecule_name"] = _normalize_molecule_name(
        structures_csv["molecule_name"]
    )

    needed_mols = pd.Index(
        pd.concat([train["molecule_name"], test["molecule_name"]], axis=0).unique()
    )
    have_mols = pd.Index(structures_csv["molecule_name"].unique())
    coverage = float(needed_mols.isin(have_mols).mean())
    print(
        "structures.csv shape:",
        structures_csv.shape,
        "| unique molecules:",
        len(have_mols),
        "| needed molecules:",
        len(needed_mols),
        "| coverage:",
        coverage,
    )
    if coverage < 0.999:
        print("structures.csv coverage insufficient; will fall back to structures.zip.")
        structures_csv = None
else:
    print("structures.csv not found; will try structures.zip fallback.")
    structures_csv = None

if structures_csv is None:
    structures_csv = read_structures_from_zip(structures_zip_path)
    print(
        "Loaded structures from zip. shape:",
        structures_csv.shape,
        "| unique molecules:",
        structures_csv["molecule_name"].nunique(),
    )

print(
    "Unique molecules - train/test/structures:",
    train["molecule_name"].nunique(),
    test["molecule_name"].nunique(),
    structures_csv["molecule_name"].nunique(),
)



## === cell 1
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

for df in (train, test):
    df["atom_index_0"] = df["atom_index_0"].astype(np.int32)
    df["atom_index_1"] = df["atom_index_1"].astype(np.int32)
    df["molecule_name"] = df["molecule_name"].astype(str)
    df["type"] = df["type"].astype(str)

structures_csv["atom_index"] = structures_csv["atom_index"].astype(np.int32)
structures_csv["molecule_name"] = _normalize_molecule_name(
    structures_csv["molecule_name"]
)
structures_csv["atom"] = structures_csv["atom"].astype(str)
for c in ["x", "y", "z"]:
    structures_csv[c] = structures_csv[c].astype(np.float32)

structures_small = structures_csv[
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


def build_features(df: pd.DataFrame, name: str = "df") -> pd.DataFrame:
    df = df.merge(a0, on=["molecule_name", "atom_index_0"], how="left")
    df = df.merge(a1, on=["molecule_name", "atom_index_1"], how="left")

    coord_cols = ["x0", "y0", "z0", "x1", "y1", "z1"]
    if df[coord_cols].isna().any().any():
        n_missing_rows = int(df[coord_cols].isna().any(axis=1).sum())
        n_total = len(df)
        bad = df[df[coord_cols].isna().any(axis=1)].head(5)
        example_mols = bad["molecule_name"].astype(str).unique().tolist()[:3]
        in_struct = structures_small["molecule_name"].isin(example_mols).any()
        raise ValueError(
            f"[{name}] Missing coordinates after merging structures: "
            f"{n_missing_rows}/{n_total} rows have NaNs. Example rows:\n"
            f"{bad[['molecule_name','atom_index_0','atom_index_1','type']].to_string(index=False)}\n"
            f"Example molecules present in structures? {in_struct}"
        )

    dx = (df["x0"] - df["x1"]).astype(np.float32)
    dy = (df["y0"] - df["y1"]).astype(np.float32)
    dz = (df["z0"] - df["z1"]).astype(np.float32)
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)
    return df


train_fe = build_features(train, name="train")
test_fe = build_features(test, name="test")

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

model = Ridge(alpha=1.0)
pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])

print("Fitting rows:", len(X_train))
pipe.fit(X_train, y_train)

test_pred = pipe.predict(X_test).astype(np.float32)
print(
    "Predictions:",
    test_pred.shape,
    "min/max:",
    float(np.min(test_pred)),
    float(np.max(test_pred)),
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4243884342.py in <cell line: 0>()
     68 
     69 train_fe = build_features(train, name="train")
---> 70 test_fe = build_features(test, name="test")
     71 
     72 feature_cols_num = ["dist"]

/tmp/ipykernel_11/4243884342.py in build_features(df, name)
     53         example_mols = bad["molecule_name"].astype(str).unique().tolist()[:3]
     54         in_struct = structures_small["molecule_name"].isin(example_mols).any()
---> 55         raise ValueError(
     56             f"[{name}] Missing coordinates after merging structures: "
     57             f"{n_missing_rows}/{n_total} rows have NaNs. Example rows:\n"

ValueError: [test] Missing coordinates after merging structures: 467813/467813 rows have NaNs. Example rows:
   molecule_name  atom_index_0  atom_index_1 type
dsgdb9nsd_071451             9             0 1JHC
dsgdb9nsd_071451             9             1 2JHC
dsgdb9nsd_071451             9             4 3JHC
dsgdb9nsd_071451             9             5 3JHC
dsgdb9nsd_071451             9            10 2JHH
Example molecules present in structures? False

## === cell 2
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

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3358528899.py in <cell line: 0>()
      1 if "test_pred" not in globals():
----> 2     raise NameError(
      3         "test_pred is not defined; model training/prediction did not complete."
      4     )
      5 

NameError: test_pred is not defined; model training/prediction did not complete.
