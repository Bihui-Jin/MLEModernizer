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

-1.3300563636911908

# 6. Current score

1.18497

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.18497) has done: 'Your notebook is trying to blend several external submission files that don’t exist in this Kaggle environment, which causes the FileNotFoundError and prevents any submission from being written. I keep the blending core idea, but make it robust: it attempt to load those files if present, otherwise fall back to a simple, valid baseline built from the provided competition data. To nudge score toward your target (and avoid a terrible constant baseline), the fallback train a lightweight per-`type` median model on `train.csv` and apply it to `test.csv`, which is score-improving while staying minimal and fast. Finally, it always write a valid `submission1236.csv` with the required columns and row alignment.'

# 9. Code solution

## === cell 0
import os

BASE_INPUT = "/kaggle/data/champs-scalar-coupling"
ALT_INPUT = "/kaggle/input/champs-scalar-coupling"

if os.path.exists(BASE_INPUT):
    INPUT_DIR = BASE_INPUT
elif os.path.exists(ALT_INPUT):
    INPUT_DIR = ALT_INPUT
else:
    INPUT_DIR = "../input/champs-scalar-coupling"

INPUT_DIR



## === cell 1
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for p in ["../input", "/kaggle/input", "/kaggle/data"]:
    if os.path.exists(p):
        print(p, "->", os.listdir(p)[:20])

print("Using INPUT_DIR =", INPUT_DIR)
print("INPUT_DIR exists:", os.path.exists(INPUT_DIR))
print(
    "INPUT_DIR files sample:",
    os.listdir(INPUT_DIR)[:20] if os.path.exists(INPUT_DIR) else "N/A",
)



## === cell 2


def safe_read_submission(path):
    """Return (df, err). df is None if read fails."""
    try:
        df = pd.read_csv(path)
        if not {"id", "scalar_coupling_constant"}.issubset(df.columns):
            return (
                None,
                f"Missing required columns in {path}. Found: {list(df.columns)}",
            )
        return df[["id", "scalar_coupling_constant"]].copy(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


blend_paths = [
    "../input/blender/LGB_2019-07-11_-1.4378.csv",
    "../input/blender/submission-2.csv",
    "../input/blender/stack_minmax_median.csv",
    "../input/blender2/submission.csv",
    "../input/blender2/submission-giba-1.csv",
    "../input/blender/workingsubmission-test.csv",
]

subs = []
errors = []
for p in blend_paths:
    df, err = safe_read_submission(p)
    if df is not None:
        subs.append(df)
    else:
        errors.append((p, err))

print(f"Found {len(subs)} blendable submission files.")
if errors:
    print(
        "Missing/unreadable blend inputs (expected in original notebook, not required here):"
    )
    for p, err in errors:
        print(" -", p, "->", err)

if len(subs) >= 2:
    base = subs[0].set_index("id")
    aligned = [base]
    for s in subs[1:]:
        aligned.append(s.set_index("id").reindex(base.index))

    for i in range(len(aligned)):
        if aligned[i]["scalar_coupling_constant"].isna().any():
            aligned[i]["scalar_coupling_constant"] = aligned[i][
                "scalar_coupling_constant"
            ].fillna(aligned[0]["scalar_coupling_constant"])

    weights = [0.25, 0.25, 0.10, 0.20, 0.15, 0.05]
    weights = weights[: len(aligned)]
    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()

    blended = np.zeros(len(base), dtype=np.float64)
    for w, a in zip(weights, aligned):
        blended += w * a["scalar_coupling_constant"].to_numpy(dtype=np.float64)

    sub1 = pd.DataFrame({"id": base.index.values, "scalar_coupling_constant": blended})
    print(sub1["scalar_coupling_constant"].describe())
else:
    train_path = os.path.join(INPUT_DIR, "train.csv")
    test_path = os.path.join(INPUT_DIR, "test.csv")
    sample_path = os.path.join(INPUT_DIR, "sample_submission.csv")

    train = pd.read_csv(train_path, usecols=["type", "scalar_coupling_constant"])
    test = pd.read_csv(test_path, usecols=["id", "type"])
    sample = pd.read_csv(sample_path, usecols=["id"])

    type_median = train.groupby("type")["scalar_coupling_constant"].median()
    global_median = float(train["scalar_coupling_constant"].median())

    preds = test["type"].map(type_median).fillna(global_median).astype(np.float64)

    sub1 = pd.DataFrame(
        {"id": test["id"].values, "scalar_coupling_constant": preds.values}
    )
    sub1 = sample.merge(sub1, on="id", how="left")
    sub1["scalar_coupling_constant"] = (
        sub1["scalar_coupling_constant"].fillna(global_median).astype(np.float64)
    )

    print("Fallback baseline built (per-type median).")
    print(sub1["scalar_coupling_constant"].describe())



## === cell 3
sub1.to_csv("submission1236.csv", index=False)
print("Wrote submission1236.csv with shape:", sub1.shape)
print(sub1.head())



## === cell 4
try:
    ax = sub1["scalar_coupling_constant"].plot(
        kind="hist", bins=100, title="scalar_coupling_constant histogram"
    )
    fig = ax.get_figure()
    fig.show()
except Exception as e:
    print("Plot skipped due to:", type(e).__name__, str(e))
