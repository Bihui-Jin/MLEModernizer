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

category_encoders==2.7.0
geopandas==0.14.4
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.7869

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.12944) has done: 'I replace the object‑to‑category conversion with a safe integer‑encoding that keeps all columns numeric, ensuring LightGBM receives a non‑empty 2‑D array for prediction. I also add a quick shape print before training and keep the rest of the pipeline unchanged, then write the predictions to a proper `submission.csv`.'
- What this solution (achieved 3.00455) has done: 'The fix changes the merges that add structural information to use a left join, preventing the test set from being emptied when some atom‑index matches are missing. This keeps all rows in `X_test`, allowing LightGBM to make predictions and the script to write a valid `submission.csv`.'
- What this solution (achieved 3.01061) has done: 'The fix converts the “type” column to a categorical dtype before training so LightGBM accepts it, and then proceeds with fitting, predicting, and writing a proper `submission.csv`. This resolves the datatype error and allows the pipeline to generate a valid submission file.'
- What this solution (achieved 3.00815) has done: 'I add three simple geometric delta features (dx, dy, dz) alongside the existing distance, and make the LightGBM model a bit more expressive by increasing `num_leaves` and adding modest subsampling parameters. These changes keep the original pipeline intact while giving the model richer information to reduce the log‑MAE toward the target score.'
- What this solution (achieved 1.99777) has done: 'The fix removes the unsupported `verbose` argument from LightGBM’s `fit` call, allowing the model to train correctly and produce predictions. No other logic changes are made, preserving the original feature engineering and pipeline while ensuring a valid `submission.csv` is written.'
- What this solution (achieved 3.00572) has done: 'I keep the overall pipeline unchanged but train the model on the original target values instead of a log‑transformed version. The competition metric is based on a log of MAE, so training on the raw scale aligns the model’s loss (MAE) with the evaluation more closely, which should lower the final score. Accordingly, I remove the `log1p` transformation and its inverse (`expm1`) in the prediction step.'

# 9. Code solution

## === cell 0
gc.collect()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2278732043.py in <cell line: 0>()
----> 1 gc.collect()
      2 

NameError: name 'gc' is not defined

## === cell 1
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample_sub = pd.read_csv("../input/sample_submission.csv")
structures = pd.read_csv("../input/structures.csv")
contributions = pd.read_csv("../input/scalar_coupling_contributions.csv")
dipole = pd.read_csv("../input/dipole_moments.csv")
potential_energy = pd.read_csv("../input/potential_energy.csv")
mulliken = pd.read_csv("../input/mulliken_charges.csv")
print(f"train.shape: {train.shape}")
print(f"test.shape: {test.shape}")



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3204751123.py in <cell line: 0>()
----> 1 train = pd.read_csv("../input/train.csv")
      2 test = pd.read_csv("../input/test.csv")
      3 sample_sub = pd.read_csv("../input/sample_submission.csv")
      4 structures = pd.read_csv("../input/structures.csv")
      5 contributions = pd.read_csv("../input/scalar_coupling_contributions.csv")

NameError: name 'pd' is not defined

## === cell 2
X_train = train.drop(columns=["scalar_coupling_constant"]).copy()
y_train = train["scalar_coupling_constant"].copy()
X_test = test.copy()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1024720760.py in <cell line: 0>()
----> 1 X_train = train.drop(columns=["scalar_coupling_constant"]).copy()
      2 y_train = train["scalar_coupling_constant"].copy()
      3 X_test = test.copy()
      4 

NameError: name 'train' is not defined

## === cell 3
print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1645962322.py in <cell line: 0>()
----> 1 print(f"X_train.shape: {X_train.shape}")
      2 print(f"X_test.shape: {X_test.shape}")
      3 

NameError: name 'X_train' is not defined

## === cell 4
X_train = X_train.drop(columns=["id"])
X_test = X_test.drop(columns=["id"])



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/757085028.py in <cell line: 0>()
----> 1 X_train = X_train.drop(columns=["id"])
      2 X_test = X_test.drop(columns=["id"])
      3 

NameError: name 'X_train' is not defined

## === cell 5
X_train = X_train.reset_index()
X_test = X_test.reset_index()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3666974169.py in <cell line: 0>()
----> 1 X_train = X_train.reset_index()
      2 X_test = X_test.reset_index()
      3 
      4 

NameError: name 'X_train' is not defined

## === cell 6
def encode_object_to_int(df_train, df_test, exclude_cols=None):
    """
    Encode all object columns as integer codes, except those listed in exclude_cols.
    Categories are aligned between train and test to avoid unseen categories.
    """
    if exclude_cols is None:
        exclude_cols = []
    obj_cols = [
        c
        for c in df_train.select_dtypes(include="object").columns
        if c not in exclude_cols
    ]
    for col in obj_cols:
        combined = pd.concat([df_train[col], df_test[col]], axis=0)
        cat = combined.astype("category").cat
        df_train[col] = cat.codes[: len(df_train)]
        df_test[col] = cat.codes[len(df_train) :]
    return df_train, df_test


X_train, X_test = encode_object_to_int(
    X_train, X_test, exclude_cols=["molecule_name", "type"]
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2041734083.py in <cell line: 0>()
     20 
     21 X_train, X_test = encode_object_to_int(
---> 22     X_train, X_test, exclude_cols=["molecule_name", "type"]
     23 )
     24 

NameError: name 'X_train' is not defined

## === cell 7
print(f"Unique types in train: {X_train['type'].unique()}")
print(f"Unique types in test: {X_test['type'].unique()}")




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1679449741.py in <cell line: 0>()
----> 1 print(f"Unique types in train: {X_train['type'].unique()}")
      2 print(f"Unique types in test: {X_test['type'].unique()}")
      3 
      4 

NameError: name 'X_train' is not defined

## === cell 8
def calc_score(X_val, y_true, y_pred):
    df = X_val.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["error"] = (df["y_true"] - df["y_pred"]).abs()
    agg = df.groupby("type").agg({"error": "mean"})
    agg["log_error"] = np.log(agg["error"])
    return agg["log_error"].mean()




## === cell 9
def cross_val(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        model = lgbm.LGBMRegressor(random_state=42)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[val_idx])
        score = calc_score(X.iloc[val_idx], y.iloc[val_idx], pred)
        print(f"fold {fold} score: {score}")




## === cell 10
X_train = X_train.merge(
    structures,
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    how="left",
    sort=True,
    suffixes=("", "_0"),
)
X_test = X_test.merge(
    structures,
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    how="left",
    sort=True,
    suffixes=("", "_0"),
)

X_train = X_train.rename(
    columns={
        "atom_index": "atom_index_0_0",
        "x": "atom_index_0_x",
        "y": "atom_index_0_y",
        "z": "atom_index_0_z",
        "atom": "atom_0",
    }
)
X_test = X_test.rename(
    columns={
        "atom_index": "atom_index_0_0",
        "x": "atom_index_0_x",
        "y": "atom_index_0_y",
        "z": "atom_index_0_z",
        "atom": "atom_0",
    }
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/897068182.py in <cell line: 0>()
----> 1 X_train = X_train.merge(
      2     structures,
      3     left_on=["molecule_name", "atom_index_0"],
      4     right_on=["molecule_name", "atom_index"],
      5     how="left",

NameError: name 'X_train' is not defined

## === cell 11
X_train = X_train.merge(
    structures,
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    how="left",
    sort=True,
    suffixes=("", "_1"),
)
X_test = X_test.merge(
    structures,
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    how="left",
    sort=True,
    suffixes=("", "_1"),
)

X_train = X_train.rename(
    columns={
        "atom_index": "atom_index_1_1",
        "x": "atom_index_1_x",
        "y": "atom_index_1_y",
        "z": "atom_index_1_z",
        "atom": "atom_1",
    }
)
X_test = X_test.rename(
    columns={
        "atom_index": "atom_index_1_1",
        "x": "atom_index_1_x",
        "y": "atom_index_1_y",
        "z": "atom_index_1_z",
        "atom": "atom_1",
    }
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2999296677.py in <cell line: 0>()
----> 1 X_train = X_train.merge(
      2     structures,
      3     left_on=["molecule_name", "atom_index_1"],
      4     right_on=["molecule_name", "atom_index"],
      5     how="left",

NameError: name 'X_train' is not defined

## === cell 12
X_train = X_train.drop(columns=["atom_index_0_0", "atom_index_1_1"])
X_test = X_test.drop(columns=["atom_index_0_0", "atom_index_1_1"])

X_train = X_train.merge(
    contributions,
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
)
X_test = X_test.merge(
    contributions,
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
)

X_train = X_train.merge(potential_energy, on="molecule_name", how="left")
X_test = X_test.merge(potential_energy, on="molecule_name", how="left")

dipole_renamed = dipole.rename(columns={"X": "dip_X", "Y": "dip_Y", "Z": "dip_Z"})
X_train = X_train.merge(dipole_renamed, on="molecule_name", how="left")
X_test = X_test.merge(dipole_renamed, on="molecule_name", how="left")

charges0 = mulliken.rename(columns={"mulliken_charge": "charge_0"})
X_train = X_train.merge(
    charges0,
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    how="left",
)
X_train = X_train.drop(columns=["atom_index"])
X_test = X_test.merge(
    charges0,
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    how="left",
)
X_test = X_test.drop(columns=["atom_index"])

charges1 = mulliken.rename(columns={"mulliken_charge": "charge_1"})
X_train = X_train.merge(
    charges1,
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    how="left",
)
X_train = X_train.drop(columns=["atom_index"])
X_test = X_test.merge(
    charges1,
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    how="left",
)
X_test = X_test.drop(columns=["atom_index"])

X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/810598367.py in <cell line: 0>()
----> 1 X_train = X_train.drop(columns=["atom_index_0_0", "atom_index_1_1"])
      2 X_test = X_test.drop(columns=["atom_index_0_0", "atom_index_1_1"])
      3 
      4 X_train = X_train.merge(
      5     contributions,

NameError: name 'X_train' is not defined

## === cell 13
X_train["distance"] = np.sqrt(
    (X_train["atom_index_0_x"] - X_train["atom_index_1_x"]) ** 2
    + (X_train["atom_index_0_y"] - X_train["atom_index_1_y"]) ** 2
    + (X_train["atom_index_0_z"] - X_train["atom_index_1_z"]) ** 2
)
X_test["distance"] = np.sqrt(
    (X_test["atom_index_0_x"] - X_test["atom_index_1_x"]) ** 2
    + (X_test["atom_index_0_y"] - X_test["atom_index_1_y"]) ** 2
    + (X_test["atom_index_0_z"] - X_test["atom_index_1_z"]) ** 2
)

X_train["dx"] = X_train["atom_index_0_x"] - X_train["atom_index_1_x"]
X_train["dy"] = X_train["atom_index_0_y"] - X_train["atom_index_1_y"]
X_train["dz"] = X_train["atom_index_0_z"] - X_train["atom_index_1_z"]
X_test["dx"] = X_test["atom_index_0_x"] - X_test["atom_index_1_x"]
X_test["dy"] = X_test["atom_index_0_y"] - X_test["atom_index_1_y"]
X_test["dz"] = X_test["atom_index_0_z"] - X_test["atom_index_1_z"]

X_train["join_type"] = X_train["type"].str.slice(0, 2)
X_test["join_type"] = X_test["type"].str.slice(0, 2)

X_train["num_atoms"] = (
    X_train.groupby("molecule_name")["atom_index_0"].transform("max") + 1
)
X_test["num_atoms"] = (
    X_test.groupby("molecule_name")["atom_index_0"].transform("max") + 1
)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2400649157.py in <cell line: 0>()
----> 1 X_train["distance"] = np.sqrt(
      2     (X_train["atom_index_0_x"] - X_train["atom_index_1_x"]) ** 2
      3     + (X_train["atom_index_0_y"] - X_train["atom_index_1_y"]) ** 2
      4     + (X_train["atom_index_0_z"] - X_train["atom_index_1_z"]) ** 2
      5 )

NameError: name 'np' is not defined

## === cell 14
X_train = X_train.drop(columns=["index"])
X_test = X_test.drop(columns=["index"])

X_train, X_test = encode_object_to_int(X_train, X_test, exclude_cols=["type"])

type_mean = y_train.groupby(X_train["type"]).mean()
X_train["type_mean"] = X_train["type"].map(type_mean)
X_test["type_mean"] = X_test["type"].map(type_mean)
global_mean = y_train.mean()
X_test["type_mean"].fillna(global_mean, inplace=True)

X_train["type"] = X_train["type"].astype("category")
X_test["type"] = X_test["type"].astype("category")

print(f"Final shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}")

X_train = X_train.drop(columns=["molecule_name"])
X_test = X_test.drop(columns=["molecule_name"])

y_train_processed = y_train  # keep original values

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train_processed, test_size=0.10, random_state=42
)

lgbm_model = lgbm.LGBMRegressor(
    n_estimators=4000,
    learning_rate=0.02,
    num_leaves=128,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=4,
)

lgbm_model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="l1",
    callbacks=[lgbm.early_stopping(stopping_rounds=200, verbose=False)],
    categorical_feature=["type"],
)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/914658192.py in <cell line: 0>()
----> 1 X_train = X_train.drop(columns=["index"])
      2 X_test = X_test.drop(columns=["index"])
      3 
      4 # encode remaining object columns (except the categorical 'type')
      5 X_train, X_test = encode_object_to_int(X_train, X_test, exclude_cols=["type"])

NameError: name 'X_train' is not defined

## === cell 15
y_predict = lgbm_model.predict(X_test)
y_predict = np.maximum(y_predict, 0)  # enforce non‑negative predictions



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3544083052.py in <cell line: 0>()
----> 1 y_predict = lgbm_model.predict(X_test)
      2 y_predict = np.maximum(y_predict, 0)  # enforce non‑negative predictions
      3 

NameError: name 'lgbm_model' is not defined

## === cell 16
assert len(y_predict) == X_test.shape[0], "Prediction length mismatch!"

sample_sub["scalar_coupling_constant"] = y_predict
output_path = "submission.csv"
sample_sub.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4071970224.py in <cell line: 0>()
----> 1 assert len(y_predict) == X_test.shape[0], "Prediction length mismatch!"
      2 
      3 sample_sub["scalar_coupling_constant"] = y_predict
      4 output_path = "submission.csv"
      5 sample_sub.to_csv(output_path, index=False)

NameError: name 'y_predict' is not defined
