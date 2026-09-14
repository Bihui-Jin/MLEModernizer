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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.33542

# 6. Current score

1.23566

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.4908) has done: 'I fix the NaN crash by ensuring the feature engineering creates `dist_to_type_mean` using *train-derived* type means (so test never gets NaNs for unseen/rare types) and by adding a simple median imputation step for any remaining missing values after merges. I also make the file paths robust to both `../input/...` (classic Kaggle) and your provided `/kaggle/data/champs-scalar-coupling/...` layout without changing the core model or features. These changes are score-neutral-to-positive (they prevent broken predictions and avoid injecting NaNs). The script then train and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 1.45368) has done: 'Your current score (1.4908, lower-is-better) is far from the target (0.33542), so we need a meaningful but still minimal change that keeps the same overall pipeline. The biggest issue is that you’re fitting one regressor across all coupling `type`s, while the metric is averaged per-type; training separate models per `type` is a small structural change but keeps the same model/feature logic and typically reduces error a lot on this competition. I keep the exact same features and ExtraTrees model, but fit/predict per-type and then assemble predictions back in original test row order. I also make LabelEncoder usage safe by fitting on the union of train+test `type` strings to avoid potential transform issues.'
- What this solution (achieved 1.41441) has done: 'We keep your exact feature set and per-type ExtraTrees approach, but make two small changes that typically reduce MAE on this competition: (1) add a couple of low-risk geometric features (coordinate deltas and squared distance) derived from the already-merged structure coordinates, and (2) slightly increase `n_estimators` so the ExtraTrees predictions are less noisy without changing the model family or training loop. These are minimal, metric-aligned improvements (distance geometry is central for scalar couplings) and should move the log-MAE down toward your target. The submission writing and row alignment remain unchanged.'
- What this solution (achieved 1.23566) has done: 'The timeout is dominated by fitting 8 separate `ExtraTreesRegressor` models on very large per-type subsets (millions of rows) and by repeated boolean slicing that materializes large temporary arrays. I keep the exact same model, hyperparameters, and per-type training loop, but speed it up by (1) converting types to compact integer codes once, (2) sorting/grouping indices by type so each subset is sliced via contiguous index ranges rather than expensive boolean masks, and (3) avoiding repeated dtype/object conversions during atom/type parsing. I also reduce overhead in NaN imputation by using precomputed column indices and in-place fills, and I release large intermediates early to lower memory pressure (which also improves runtime stability). All changes are equivalence-preserving (no sampling, no fewer trees, no early stopping, no precision reduction of the learning algorithm).'
- What this solution (achieved 1.23566) has done: 'The crash comes from using NumPy’s `astype("string")`, which isn’t a valid NumPy dtype; this stops cell 0 before `x0..z1` are created, which then triggers the downstream KeyError. I fix the atom-type consistency masks by comparing via object/unicode arrays (equivalent intent, but compatible with the Kaggle/Python environment) and add a small safety check to fail fast if any structure lookup is missing. This is execution-stability only and should be score-neutral; the model, per-type training loop, features, and submission formatting remain unchanged. After this, the pipeline should run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 1.23566) has done: 'The crash is because the `(molecule_name, atom_index)` lookup key collides: `SHIFT=256` is smaller than the maximum atom_index in this dataset, so different atoms can map to the same key and `get_indexer` can return `-1` for some valid pairs. I fix this by computing a safe `SHIFT` from `structures.atom_index.max()+1` and by building the lookup index from unique keys (drop duplicates defensively), which preserves the same feature logic but makes the merges correct. Then the downstream geometry features (`x0..z1`, distances, etc.) exist so cells 1–2 run. This is a correctness fix and should also improve score (your model was effectively training on corrupted/missing structure features before).'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn  # type: ignore

    patch_sklearn()
except Exception:
    pass

from sklearn import preprocessing, ensemble

np.random.seed(4)


def resolve_input_dir():
    candidates = [
        "../input/champs-scalar-coupling",
        "../input",
        "/kaggle/input/champs-scalar-coupling",
        "/kaggle/input",
        "/kaggle/data/champs-scalar-coupling",
        "/kaggle/data/input/champs-scalar-coupling",
        "/kaggle/data/input",
    ]
    needed = {"train.csv", "test.csv", "structures.csv", "sample_submission.csv"}
    for d in candidates:
        if os.path.isdir(d) and needed.issubset(set(os.listdir(d))):
            return d
    if os.path.isdir(".") and needed.issubset(set(os.listdir("."))):
        return "."
    raise FileNotFoundError(
        "Could not locate input directory containing train/test/structures/sample_submission CSV files."
    )


INPUT_DIR = resolve_input_dir()

train_dtypes = {
    "id": "int32",
    "molecule_name": "category",
    "atom_index_0": "int16",
    "atom_index_1": "int16",
    "type": "category",
    "scalar_coupling_constant": "float32",
}
test_dtypes = {
    "id": "int32",
    "molecule_name": "category",
    "atom_index_0": "int16",
    "atom_index_1": "int16",
    "type": "category",
}

train = pd.read_csv(
    os.path.join(INPUT_DIR, "train.csv"),
    usecols=list(train_dtypes.keys()),
    dtype=train_dtypes,
)
test = pd.read_csv(
    os.path.join(INPUT_DIR, "test.csv"),
    usecols=list(test_dtypes.keys()),
    dtype=test_dtypes,
)
sub = pd.read_csv(
    os.path.join(INPUT_DIR, "sample_submission.csv"),
    usecols=["id", "scalar_coupling_constant"],
)
print(train.shape, test.shape, sub.shape)

train_type_str = train["type"].astype("string")
test_type_str = test["type"].astype("string")

train["atom1"] = train_type_str.str[2]
train["atom2"] = train_type_str.str[3]
test["atom1"] = test_type_str.str[2]
test["atom2"] = test_type_str.str[3]

lbl = preprocessing.LabelEncoder()
all_types = pd.concat([train_type_str, test_type_str], axis=0)
for i in range(4):
    s_all = all_types.str[i]
    lbl.fit(s_all)
    train["type" + str(i)] = lbl.transform(train_type_str.str[i]).astype("int8")
    test["type" + str(i)] = lbl.transform(test_type_str.str[i]).astype("int8")

structures = pd.read_csv(
    os.path.join(INPUT_DIR, "structures.csv"),
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

all_mols = pd.Categorical(
    pd.concat(
        [train["molecule_name"], test["molecule_name"], structures["molecule_name"]],
        axis=0,
    )
)
train["molecule_name"] = pd.Categorical(
    train["molecule_name"], categories=all_mols.categories
)
test["molecule_name"] = pd.Categorical(
    test["molecule_name"], categories=all_mols.categories
)
structures["molecule_name"] = pd.Categorical(
    structures["molecule_name"], categories=all_mols.categories
)

mol_code_struct = structures["molecule_name"].cat.codes.to_numpy(np.int32, copy=False)
mol_code_train = train["molecule_name"].cat.codes.to_numpy(np.int32, copy=False)
mol_code_test = test["molecule_name"].cat.codes.to_numpy(np.int32, copy=False)

SHIFT = int(structures["atom_index"].max()) + 1

struct_key = mol_code_struct.astype(np.int64) * SHIFT + structures[
    "atom_index"
].to_numpy(np.int64, copy=False)

if struct_key.hasnans:
    raise ValueError("Unexpected NaNs in structures key construction.")

if pd.Index(struct_key).has_duplicates:
    keep = ~pd.Index(struct_key).duplicated(keep="first")
    structures = structures.loc[keep].reset_index(drop=True)
    mol_code_struct = structures["molecule_name"].cat.codes.to_numpy(
        np.int32, copy=False
    )
    struct_key = mol_code_struct.astype(np.int64) * SHIFT + structures[
        "atom_index"
    ].to_numpy(np.int64, copy=False)

struct_index = pd.Index(struct_key)

_struct_atom = structures["atom"].to_numpy(copy=False)
_struct_x = structures["x"].to_numpy(copy=False)
_struct_y = structures["y"].to_numpy(copy=False)
_struct_z = structures["z"].to_numpy(copy=False)


def take_struct_rows(mol_codes: np.ndarray, atom_idx: pd.Series):
    keys = mol_codes.astype(np.int64) * SHIFT + atom_idx.to_numpy(np.int64, copy=False)
    pos = struct_index.get_indexer(keys)  # -1 if missing
    if (pos < 0).any():
        bad = np.flatnonzero(pos < 0)[:5]
        raise KeyError(
            f"Some (molecule, atom_index) pairs were not found in structures.csv. "
            f"Example row indices: {bad.tolist()}"
        )
    atom = _struct_atom[pos]
    x = _struct_x[pos].astype(np.float32, copy=False)
    y = _struct_y[pos].astype(np.float32, copy=False)
    z = _struct_z[pos].astype(np.float32, copy=False)
    return atom, x, y, z


tr_atom0, tr_x0, tr_y0, tr_z0 = take_struct_rows(mol_code_train, train["atom_index_0"])
tr_atom1, tr_x1, tr_y1, tr_z1 = take_struct_rows(mol_code_train, train["atom_index_1"])
te_atom0, te_x0, te_y0, te_z0 = take_struct_rows(mol_code_test, test["atom_index_0"])
te_atom1, te_x1, te_y1, te_z1 = take_struct_rows(mol_code_test, test["atom_index_1"])

train_atom1_arr = train["atom1"].astype(str).to_numpy(copy=False)
train_atom2_arr = train["atom2"].astype(str).to_numpy(copy=False)
test_atom1_arr = test["atom1"].astype(str).to_numpy(copy=False)
test_atom2_arr = test["atom2"].astype(str).to_numpy(copy=False)

tr_atom0_arr = tr_atom0.astype(str, copy=False)
tr_atom1_arr = tr_atom1.astype(str, copy=False)
te_atom0_arr = te_atom0.astype(str, copy=False)
te_atom1_arr = te_atom1.astype(str, copy=False)

mask_tr0 = tr_atom0_arr == train_atom1_arr
mask_tr1 = tr_atom1_arr == train_atom2_arr
mask_te0 = te_atom0_arr == test_atom1_arr
mask_te1 = te_atom1_arr == test_atom2_arr

for df, x0, y0, z0, x1, y1, z1, m0, m1 in [
    (
        train,
        tr_x0.copy(),
        tr_y0.copy(),
        tr_z0.copy(),
        tr_x1.copy(),
        tr_y1.copy(),
        tr_z1.copy(),
        mask_tr0,
        mask_tr1,
    ),
    (
        test,
        te_x0.copy(),
        te_y0.copy(),
        te_z0.copy(),
        te_x1.copy(),
        te_y1.copy(),
        te_z1.copy(),
        mask_te0,
        mask_te1,
    ),
]:
    x0[~m0] = np.nan
    y0[~m0] = np.nan
    z0[~m0] = np.nan
    x1[~m1] = np.nan
    y1[~m1] = np.nan
    z1[~m1] = np.nan
    df["x0"], df["y0"], df["z0"] = x0, y0, z0
    df["x1"], df["y1"], df["z1"] = x1, y1, z1

train_key0 = mol_code_train.astype(np.int64) * SHIFT + train["atom_index_0"].to_numpy(
    np.int64, copy=False
)
train_key1 = mol_code_train.astype(np.int64) * SHIFT + train["atom_index_1"].to_numpy(
    np.int64, copy=False
)
test_key0 = mol_code_test.astype(np.int64) * SHIFT + test["atom_index_0"].to_numpy(
    np.int64, copy=False
)
test_key1 = mol_code_test.astype(np.int64) * SHIFT + test["atom_index_1"].to_numpy(
    np.int64, copy=False
)

mulliken = pd.read_csv(
    os.path.join(INPUT_DIR, "mulliken_charges.csv"),
    usecols=["molecule_name", "atom_index", "mulliken_charge"],
    dtype={
        "molecule_name": "category",
        "atom_index": "int16",
        "mulliken_charge": "float32",
    },
)
mulliken["molecule_name"] = pd.Categorical(
    mulliken["molecule_name"], categories=all_mols.categories
)
mull_key = mulliken["molecule_name"].cat.codes.to_numpy(np.int32, copy=False).astype(
    np.int64
) * SHIFT + mulliken["atom_index"].to_numpy(np.int64, copy=False)
mull_index = pd.Index(mull_key)
mull_vals = mulliken["mulliken_charge"].to_numpy(np.float32, copy=False)
del mulliken, mull_key

pos = mull_index.get_indexer(train_key0)
train["mulliken_charge_0"] = mull_vals[pos]
pos = mull_index.get_indexer(train_key1)
train["mulliken_charge_1"] = mull_vals[pos]
pos = mull_index.get_indexer(test_key0)
test["mulliken_charge_0"] = mull_vals[pos]
pos = mull_index.get_indexer(test_key1)
test["mulliken_charge_1"] = mull_vals[pos]
del mull_index, mull_vals, pos
gc.collect()

mst = pd.read_csv(
    os.path.join(INPUT_DIR, "magnetic_shielding_tensors.csv"),
    dtype={"molecule_name": "category", "atom_index": "int16"},
)
mst["molecule_name"] = pd.Categorical(
    mst["molecule_name"], categories=all_mols.categories
)
mst_cols = [c for c in mst.columns if c not in ["molecule_name", "atom_index"]]
for c in mst_cols:
    if mst[c].dtype != np.float32:
        mst[c] = mst[c].astype("float32", copy=False)

mst_key = mst["molecule_name"].cat.codes.to_numpy(np.int32, copy=False).astype(
    np.int64
) * SHIFT + mst["atom_index"].to_numpy(np.int64, copy=False)
mst_index = pd.Index(mst_key)
mst_matrix = mst[mst_cols].to_numpy(dtype=np.float32, copy=False)
del mst, mst_key
gc.collect()


def take_mst(keys: np.ndarray):
    p = mst_index.get_indexer(keys)
    return mst_matrix[p]


mst0_cols = [f"mst0_{c}" for c in mst_cols]
mst1_cols = [f"mst1_{c}" for c in mst_cols]

train[mst0_cols] = take_mst(train_key0)
train[mst1_cols] = take_mst(train_key1)
test[mst0_cols] = take_mst(test_key0)
test[mst1_cols] = take_mst(test_key1)

del mst_index, mst_matrix, mst_cols, mst0_cols, mst1_cols
gc.collect()

dip = pd.read_csv(
    os.path.join(INPUT_DIR, "dipole_moments.csv"),
    usecols=["molecule_name", "X", "Y", "Z"],
    dtype={"molecule_name": "category", "X": "float32", "Y": "float32", "Z": "float32"},
).rename(columns={"X": "dipole_X", "Y": "dipole_Y", "Z": "dipole_Z"})
pe = pd.read_csv(
    os.path.join(INPUT_DIR, "potential_energy.csv"),
    usecols=["molecule_name", "potential_energy"],
    dtype={"molecule_name": "category", "potential_energy": "float32"},
)

dip["molecule_name"] = pd.Categorical(
    dip["molecule_name"], categories=all_mols.categories
)
pe["molecule_name"] = pd.Categorical(
    pe["molecule_name"], categories=all_mols.categories
)

dip = dip.set_index("molecule_name", drop=True)
pe = pe.set_index("molecule_name", drop=True)

train[["dipole_X", "dipole_Y", "dipole_Z"]] = dip.reindex(train["molecule_name"])[
    ["dipole_X", "dipole_Y", "dipole_Z"]
].to_numpy(dtype=np.float32, copy=False)
test[["dipole_X", "dipole_Y", "dipole_Z"]] = dip.reindex(test["molecule_name"])[
    ["dipole_X", "dipole_Y", "dipole_Z"]
].to_numpy(dtype=np.float32, copy=False)

train["potential_energy"] = pe.reindex(train["molecule_name"])[
    "potential_energy"
].to_numpy(dtype=np.float32, copy=False)
test["potential_energy"] = pe.reindex(test["molecule_name"])[
    "potential_energy"
].to_numpy(dtype=np.float32, copy=False)

del dip, pe, structures, struct_index, struct_key, mol_code_struct, all_mols
del _struct_atom, _struct_x, _struct_y, _struct_z
gc.collect()

test = test.sort_values("id").reset_index(drop=True)
print("After physics merges:", train.shape, test.shape)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1923372375.py in <cell line: 0>()
    129 
    130 # Defensive: ensure 1:1 mapping for indexer even if duplicates exist (shouldn't, but safe)
--> 131 if struct_key.hasnans:
    132     raise ValueError("Unexpected NaNs in structures key construction.")
    133 

AttributeError: 'numpy.ndarray' object has no attribute 'hasnans'

## === cell 1
train_p0 = train[["x0", "y0", "z0"]].to_numpy(dtype=np.float32, copy=False)
train_p1 = train[["x1", "y1", "z1"]].to_numpy(dtype=np.float32, copy=False)
test_p0 = test[["x0", "y0", "z0"]].to_numpy(dtype=np.float32, copy=False)
test_p1 = test[["x1", "y1", "z1"]].to_numpy(dtype=np.float32, copy=False)

dtrain = train_p0 - train_p1
dtest = test_p0 - test_p1

train["dx"] = dtrain[:, 0]
train["dy"] = dtrain[:, 1]
train["dz"] = dtrain[:, 2]
test["dx"] = dtest[:, 0]
test["dy"] = dtest[:, 1]
test["dz"] = dtest[:, 2]

train_dist = np.linalg.norm(dtrain, axis=1).astype(np.float32, copy=False)
test_dist = np.linalg.norm(dtest, axis=1).astype(np.float32, copy=False)

train["dist"] = train_dist
test["dist"] = test_dist
train["dist2"] = train_dist * train_dist
test["dist2"] = test_dist * test_dist

type_mean = train.groupby("type")["dist"].mean()
global_mean = float(train_dist.mean())

train["dist_to_type_mean"] = train["dist"] / train["type"].map(type_mean)
test_denom = test["type"].map(type_mean).fillna(global_mean)
test["dist_to_type_mean"] = test["dist"] / test_denom

del train_p0, train_p1, test_p0, test_p1, dtrain, dtest
gc.collect()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3185848466.py in <cell line: 0>()
----> 1 train_p0 = train[["x0", "y0", "z0"]].to_numpy(dtype=np.float32, copy=False)
      2 train_p1 = train[["x1", "y1", "z1"]].to_numpy(dtype=np.float32, copy=False)
      3 test_p0 = test[["x0", "y0", "z0"]].to_numpy(dtype=np.float32, copy=False)
      4 test_p1 = test[["x1", "y1", "z1"]].to_numpy(dtype=np.float32, copy=False)
      5 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index(['x0', 'y0', 'z0'], dtype='object')] are in the [columns]"

## === cell 2
col = [
    c
    for c in train.columns
    if c
    not in [
        "id",
        "molecule_name",
        "scalar_coupling_constant",
        "type",
        "atom1",
        "atom2",
        "atom_index_0",
        "atom_index_1",
    ]
]

train_X_df = train[col]
test_X_df = test[col]

med = train_X_df.median(numeric_only=True)
med_vec = med.reindex(col).to_numpy(dtype=np.float32, copy=False)

y = train["scalar_coupling_constant"].to_numpy(dtype=np.float32, copy=False)

train_type_codes = train["type"].cat.codes.to_numpy(np.int16, copy=False)
test_type_codes = test["type"].cat.codes.to_numpy(np.int16, copy=False)

test_pred = np.zeros(len(test), dtype=np.float64)

X_train_all = np.ascontiguousarray(train_X_df.to_numpy(dtype=np.float32, copy=True))
X_test_all = np.ascontiguousarray(test_X_df.to_numpy(dtype=np.float32, copy=True))

nan_tr = np.isnan(X_train_all)
if nan_tr.any():
    rr, cc = np.nonzero(nan_tr)
    X_train_all[rr, cc] = med_vec[cc]

nan_te = np.isnan(X_test_all)
if nan_te.any():
    rr, cc = np.nonzero(nan_te)
    X_test_all[rr, cc] = med_vec[cc]

np.nan_to_num(X_test_all, nan=0.0, copy=False)

order_tr = np.argsort(train_type_codes, kind="mergesort")  # stable, deterministic
order_te = np.argsort(test_type_codes, kind="mergesort")

tr_codes_sorted = train_type_codes[order_tr]
te_codes_sorted = test_type_codes[order_te]

unique_tr, tr_starts = np.unique(tr_codes_sorted, return_index=True)
unique_te, te_starts = np.unique(te_codes_sorted, return_index=True)

tr_ends = np.r_[tr_starts[1:], len(order_tr)]
te_ends = np.r_[te_starts[1:], len(order_te)]

tr_bounds = {int(c): (int(s), int(e)) for c, s, e in zip(unique_tr, tr_starts, tr_ends)}
te_bounds = {int(c): (int(s), int(e)) for c, s, e in zip(unique_te, te_starts, te_ends)}

for code, (s_tr, e_tr) in tr_bounds.items():
    b = te_bounds.get(code)
    if b is None:
        continue
    s_te, e_te = b
    if s_te == e_te:
        continue

    reg = ensemble.ExtraTreesRegressor(n_jobs=-1, n_estimators=80, random_state=4)

    tr_idx = order_tr[s_tr:e_tr]
    te_idx = order_te[s_te:e_te]

    Xtr = X_train_all[tr_idx]
    ytr = y[tr_idx]
    Xte = X_test_all[te_idx]

    reg.fit(Xtr, ytr)
    test_pred[te_idx] = reg.predict(Xte)

test["scalar_coupling_constant"] = test_pred

submission = test[["id", "scalar_coupling_constant"]].sort_values("id")
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
