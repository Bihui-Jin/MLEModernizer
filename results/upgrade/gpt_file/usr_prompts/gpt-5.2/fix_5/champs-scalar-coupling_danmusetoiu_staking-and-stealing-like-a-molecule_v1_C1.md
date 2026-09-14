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

-1.35224452227701

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.17332) has done: 'I remove the dependency on missing external “../input/…” submissions (the cause of the FileNotFoundError) and replace it with an in-notebook baseline model that uses only the provided competition data. To keep changes minimal and stable, the solution build lightweight, leakage-safe aggregate features from `train.csv` (grouped by coupling `type` and atom indices) and apply them to `test.csv`, with sensible fallbacks to avoid NaNs. This run end-to-end in the given environment and always write a valid `submission.csv` with the required columns. Since no current score exists, the focus is on correctness and producing a reasonable baseline score without changing any “core model” (the original code was only ensembling submissions, not training a model).'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_DIR_CANDIDATES = [
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/data/input/champs-scalar-coupling",
    "../input/champs-scalar-coupling",
    "../input",
    "/kaggle/input",
]


def find_data_dir():
    for d in DATA_DIR_CANDIDATES:
        if os.path.exists(d):
            if os.path.isfile(os.path.join(d, "train.csv")):
                return d
            cand = os.path.join(d, "champs-scalar-coupling")
            if os.path.isfile(os.path.join(cand, "train.csv")):
                return cand
    if os.path.isfile("/kaggle/data/champs-scalar-coupling/train.csv"):
        return "/kaggle/data/champs-scalar-coupling"
    raise FileNotFoundError("Could not locate champs-scalar-coupling data directory.")


DATA_DIR = find_data_dir()
print("Using DATA_DIR:", DATA_DIR)




## === cell 1
def safe_listdir(path):
    try:
        return os.listdir(path)
    except Exception as e:
        return f"Could not list {path}: {e}"


print("List /kaggle/input:", safe_listdir("/kaggle/input"))
print("List /kaggle/data:", safe_listdir("/kaggle/data"))
print("List DATA_DIR:", safe_listdir(DATA_DIR))



## === cell 2
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")

usecols_train = [
    "id",
    "molecule_name",
    "atom_index_0",
    "atom_index_1",
    "type",
    "scalar_coupling_constant",
]
usecols_test = ["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]

train = pd.read_csv(
    train_path,
    usecols=usecols_train,
    dtype={
        "id": np.int32,
        "molecule_name": "object",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
        "scalar_coupling_constant": np.float32,
    },
)
test = pd.read_csv(
    test_path,
    usecols=usecols_test,
    dtype={
        "id": np.int32,
        "molecule_name": "object",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
sample_sub = pd.read_csv(
    sample_path, dtype={"id": np.int32, "scalar_coupling_constant": np.float32}
)

print(
    "train shape:",
    train.shape,
    "test shape:",
    test.shape,
    "sample_sub shape:",
    sample_sub.shape,
)
print("train columns:", train.columns.tolist())
print("test columns:", test.columns.tolist())

if not set(sample_sub.columns) >= {"id", "scalar_coupling_constant"}:
    raise ValueError("sample_submission.csv does not have required columns.")
if sample_sub["id"].nunique() != len(sample_sub):
    raise ValueError("Duplicate IDs found in sample_submission.")



## === cell 3
structures = pd.read_csv(
    structures_path,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "object",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
)


def add_distance_and_atoms(df, structures_df):
    df = df.copy()

    df["molecule_name"] = df["molecule_name"].astype("object")
    structures_df = structures_df.copy()
    structures_df["molecule_name"] = structures_df["molecule_name"].astype("object")

    df["atom_index_0"] = df["atom_index_0"].astype(np.int16)
    df["atom_index_1"] = df["atom_index_1"].astype(np.int16)
    structures_df["atom_index"] = structures_df["atom_index"].astype(np.int16)

    s0 = structures_df.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom_0",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
    )
    s1 = structures_df.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom_1",
            "x": "x1",
            "y": "y1",
            "z": "z1",
        }
    )

    df = df.merge(
        s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
        validate="many_to_one",
    )
    df = df.merge(
        s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
        validate="many_to_one",
    )

    missing = df[["x0", "y0", "z0", "x1", "y1", "z1"]].isna().any(axis=1).sum()
    if missing:
        bad = df[df[["x0", "x1"]].isna().any(axis=1)][
            ["molecule_name", "atom_index_0", "atom_index_1"]
        ].head(5)
        raise ValueError(
            f"Missing coordinates after merge for {missing} rows. Sample bad keys:\n{bad}"
        )

    dx = (df["x0"].values - df["x1"].values).astype(np.float32)
    dy = (df["y0"].values - df["y1"].values).astype(np.float32)
    dz = (df["z0"].values - df["z1"].values).astype(np.float32)
    df["distance"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)

    df.drop(columns=["x0", "y0", "z0", "x1", "y1", "z1"], inplace=True)
    return df


train_d = add_distance_and_atoms(train, structures)
test_d = add_distance_and_atoms(test, structures)

print("Distance stats (train):", train_d["distance"].describe())
print("Distance stats (test):", test_d["distance"].describe())
print("Atom_0/Atom_1 train value counts (head):")
print(train_d["atom_0"].value_counts().head(10))
print(train_d["atom_1"].value_counts().head(10))



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1204444011.py in <cell line: 0>()
     78 
     79 train_d = add_distance_and_atoms(train, structures)
---> 80 test_d = add_distance_and_atoms(test, structures)
     81 
     82 print("Distance stats (train):", train_d["distance"].describe())

/tmp/ipykernel_11/1204444011.py in add_distance_and_atoms(df, structures_df)
     64             ["molecule_name", "atom_index_0", "atom_index_1"]
     65         ].head(5)
---> 66         raise ValueError(
     67             f"Missing coordinates after merge for {missing} rows. Sample bad keys:\n{bad}"
     68         )

ValueError: Missing coordinates after merge for 467813 rows. Sample bad keys:
      molecule_name  atom_index_0  atom_index_1
0  dsgdb9nsd_071451             9             0
1  dsgdb9nsd_071451             9             1
2  dsgdb9nsd_071451             9             4
3  dsgdb9nsd_071451             9             5
4  dsgdb9nsd_071451             9            10

## === cell 4
type_mean = train_d.groupby("type")["scalar_coupling_constant"].mean()
global_mean = float(train_d["scalar_coupling_constant"].mean())

type_ai0_mean = train_d.groupby(["type", "atom_index_0"])[
    "scalar_coupling_constant"
].mean()
type_ai1_mean = train_d.groupby(["type", "atom_index_1"])[
    "scalar_coupling_constant"
].mean()
type_pair_mean = train_d.groupby(["type", "atom_index_0", "atom_index_1"])[
    "scalar_coupling_constant"
].mean()

type_a0_mean = train_d.groupby(["type", "atom_0"])["scalar_coupling_constant"].mean()
type_a1_mean = train_d.groupby(["type", "atom_1"])["scalar_coupling_constant"].mean()
type_a0a1_mean = train_d.groupby(["type", "atom_0", "atom_1"])[
    "scalar_coupling_constant"
].mean()

N_BINS = 60

bin_edges_by_type = {}
for t, g in train_d.groupby("type", sort=False):
    qs = np.linspace(0.0, 1.0, N_BINS + 1)
    edges = np.quantile(g["distance"].values, qs).astype(np.float32)
    edges = np.unique(edges)
    if edges.size < 3:
        edges = np.unique(
            np.array(
                [g["distance"].min(), g["distance"].median(), g["distance"].max()],
                dtype=np.float32,
            )
        )
    bin_edges_by_type[t] = edges

train_bins = np.empty(len(train_d), dtype=np.int16)
for t, edges in bin_edges_by_type.items():
    m = train_d["type"].values == t
    dvals = train_d.loc[m, "distance"].values
    idx = np.searchsorted(edges, dvals, side="right") - 1
    max_bin = max(0, len(edges) - 2)
    idx = np.clip(idx, 0, max_bin).astype(np.int16)
    train_bins[m] = idx

train_d = train_d.assign(dist_bin=train_bins)
type_distbin_mean = train_d.groupby(["type", "dist_bin"])[
    "scalar_coupling_constant"
].mean()

test_bins = np.empty(len(test_d), dtype=np.int16)
for t, edges in bin_edges_by_type.items():
    m = test_d["type"].values == t
    dvals = test_d.loc[m, "distance"].values
    idx = np.searchsorted(edges, dvals, side="right") - 1
    max_bin = max(0, len(edges) - 2)
    idx = np.clip(idx, 0, max_bin).astype(np.int16)
    test_bins[m] = idx

test_d = test_d.assign(dist_bin=test_bins)

test_key_pair = list(
    zip(
        test_d["type"].values,
        test_d["atom_index_0"].values,
        test_d["atom_index_1"].values,
    )
)
test_key_ai0 = list(zip(test_d["type"].values, test_d["atom_index_0"].values))
test_key_ai1 = list(zip(test_d["type"].values, test_d["atom_index_1"].values))
test_key_distbin = list(zip(test_d["type"].values, test_d["dist_bin"].values))

test_key_a0 = list(zip(test_d["type"].values, test_d["atom_0"].values))
test_key_a1 = list(zip(test_d["type"].values, test_d["atom_1"].values))
test_key_a0a1 = list(
    zip(test_d["type"].values, test_d["atom_0"].values, test_d["atom_1"].values)
)

pred_pair = pd.Series(test_key_pair).map(type_pair_mean)
pred_ai0 = pd.Series(test_key_ai0).map(type_ai0_mean)
pred_ai1 = pd.Series(test_key_ai1).map(type_ai1_mean)
pred_type = test_d["type"].map(type_mean)
pred_dist = pd.Series(test_key_distbin).map(type_distbin_mean)

pred_a0 = pd.Series(test_key_a0).map(type_a0_mean)
pred_a1 = pd.Series(test_key_a1).map(type_a1_mean)
pred_a0a1 = pd.Series(test_key_a0a1).map(type_a0a1_mean)

pred = pred_dist.copy()
pred = pred.fillna(pred_pair)
pred = pred.fillna(pred_a0a1)
pred = pred.fillna(0.55 * pred_a0 + 0.45 * pred_a1)
pred = pred.fillna(0.6 * pred_ai0 + 0.4 * pred_ai1)
pred = pred.fillna(pred_type)
pred = pred.fillna(global_mean)

test_pred = pred.astype(np.float32).values
print("Prediction stats:", pd.Series(test_pred).describe())



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3449379903.py in <cell line: 0>()
     49 ].mean()
     50 
---> 51 test_bins = np.empty(len(test_d), dtype=np.int16)
     52 for t, edges in bin_edges_by_type.items():
     53     m = test_d["type"].values == t

NameError: name 'test_d' is not defined

## === cell 5
try:
    ax = pd.Series(test_pred).plot(
        kind="hist", bins=100, title="Predicted scalar_coupling_constant (test)"
    )
    fig = ax.get_figure()
    fig.tight_layout()
except Exception as e:
    print("Plotting skipped:", e)



## === cell 6
submission = pd.DataFrame(
    {"id": test_d["id"].values, "scalar_coupling_constant": test_pred}
)

submission = submission.set_index("id").reindex(sample_sub["id"].values).reset_index()

if submission.isna().any().any():
    raise ValueError(
        "Submission contains NaNs after reindexing; check feature mapping logic."
    )
if list(submission.columns) != ["id", "scalar_coupling_constant"]:
    raise ValueError(f"Unexpected submission columns: {submission.columns.tolist()}")

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("submission shape:", submission.shape)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2494067762.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"id": test_d["id"].values, "scalar_coupling_constant": test_pred}
      3 )
      4 
      5 # Keep alignment exactly like sample_submission

NameError: name 'test_d' is not defined
