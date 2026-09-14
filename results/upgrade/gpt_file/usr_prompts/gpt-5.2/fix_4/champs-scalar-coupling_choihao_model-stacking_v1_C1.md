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
scipy==1.15.3
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

-1.965121149403092

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.99777) has done: 'I fix the immediate runtime failure by removing the hardcoded `../input/models` dependency and instead reading the competition’s `sample_submission.csv` so the notebook can always produce a valid submission. Since your current code is a stacking/blending notebook that expects external model prediction CSVs (which do not exist in this environment), the minimal correct fallback is to generate a baseline prediction file with the required columns and `.csv` suffix. I also remove notebook-only/IPython commands (`%matplotlib inline`) and deprecation issues (`np.bool`) so it runs as a plain Python script on Kaggle. The output be a valid `stack_median.csv` submission.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.99777, lower-is-better) is far worse than the target (-1.9651), and the reason is that the notebook is producing an all-zeros fallback because no external model prediction files exist. To move the score toward the target with minimal change, I keep the same “single submission CSV generation” flow but replace the zero fallback with a simple, legitimate baseline: train a per-`type` median on `train.csv` and predict those medians for `test.csv` by `type`. This preserves the overall lightweight approach (no new model architecture/training loop) while aligning predictions with the competition’s per-type error behavior, which should materially reduce MAE. I also keep the existing stacking logic intact when `../input/models` exists, and only use the baseline when it doesn’t.'

# 9. Code solution

## === cell 0
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "../input/champs-scalar-coupling",
    "../input",  # sometimes files are directly here
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(filename: str) -> str:
    for root in DATA_ROOT_CANDIDATES:
        p = os.path.join(root, filename)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find {filename} under any of: {DATA_ROOT_CANDIDATES}"
    )




## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import gmean

for root in ["/kaggle/input", "/kaggle/data", "../input", "."]:
    if os.path.isdir(root):
        try:
            print(root, "->", sorted(os.listdir(root))[:20])
        except Exception as e:
            print(root, "-> (could not list)", e)



## === cell 2
sub_path = "../input/models"
all_files = []
if os.path.isdir(sub_path):
    all_files = [f for f in os.listdir(sub_path) if f.lower().endswith(".csv")]
all_files



## === cell 3
to_remove = [
    "submission-1.701.csv",
    "submission-1.643.csv",
    "submission-1.481.csv",
    "submission-1.302.csv",
    "submission-1.619.csv",
    "submission-1.662.csv",
    "submission-1.696.csv",
    "submission-1.780.csv",
    "submission-1.708.csv",
    "submission-1.714.csv",
]
all_files = [f for f in all_files if f not in set(to_remove)]
all_files



## === cell 4
all_files



## === cell 5
if len(all_files) > 0:
    outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
    concat_sub = pd.concat(outs, axis=1)
    cols = list(map(lambda x: "mol" + str(x), range(len(concat_sub.columns))))
    concat_sub.columns = cols
    concat_sub.reset_index(inplace=True)
    if "index" in concat_sub.columns and "id" not in concat_sub.columns:
        concat_sub.rename(columns={"index": "id"}, inplace=True)
else:
    from sklearn.linear_model import LinearRegression

    train_path = find_file("train.csv")
    test_path = find_file("test.csv")
    structures_path = find_file("structures.csv")

    train = pd.read_csv(
        train_path,
        usecols=[
            "molecule_name",
            "atom_index_0",
            "atom_index_1",
            "type",
            "scalar_coupling_constant",
        ],
    )
    test = pd.read_csv(
        test_path,
        usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    )
    structures = pd.read_csv(
        structures_path,
        usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
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

    train = train.merge(
        s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    train = train.merge(
        s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )

    test = test.merge(
        s0[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    test = test.merge(
        s1[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )

    def add_dist(df: pd.DataFrame) -> pd.DataFrame:
        dx = (df["x0"] - df["x1"]).astype(np.float64)
        dy = (df["y0"] - df["y1"]).astype(np.float64)
        dz = (df["z0"] - df["z1"]).astype(np.float64)
        df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)
        df["inv_dist"] = 1.0 / (df["dist"] + 1e-6)
        df["dist2"] = df["dist"] * df["dist"]
        return df

    train = add_dist(train)
    test = add_dist(test)

    atom_values = pd.concat(
        [train["atom_0"], train["atom_1"], test["atom_0"], test["atom_1"]], axis=0
    )
    atom_values = atom_values.dropna().unique().tolist()
    atom_to_int = {a: i for i, a in enumerate(sorted(atom_values))}
    for df in (train, test):
        df["atom0_i"] = df["atom_0"].map(atom_to_int).fillna(-1).astype(np.int16)
        df["atom1_i"] = df["atom_1"].map(atom_to_int).fillna(-1).astype(np.int16)

    feature_cols = ["dist", "inv_dist", "dist2", "atom0_i", "atom1_i"]

    type_median = (
        train.groupby("type")["scalar_coupling_constant"].median().astype(np.float64)
    )
    global_median = float(train["scalar_coupling_constant"].median())

    preds = np.empty(len(test), dtype=np.float64)
    preds[:] = np.nan

    for t, idx in test.groupby("type").indices.items():
        te_idx = np.array(list(idx), dtype=np.int64)
        tr = train[train["type"] == t]
        if len(tr) < 50 or tr[feature_cols].isna().any().any():
            fallback = float(type_median.get(t, global_median))
            preds[te_idx] = fallback
            continue

        X_tr = tr[feature_cols].astype(np.float64).values
        y_tr = tr["scalar_coupling_constant"].astype(np.float64).values
        X_te = test.loc[te_idx, feature_cols].astype(np.float64).values

        model = LinearRegression(n_jobs=None)
        model.fit(X_tr, y_tr)
        preds[te_idx] = model.predict(X_te)

    preds = np.where(np.isfinite(preds), preds, global_median)

    concat_sub = pd.DataFrame(
        {"id": test["id"].astype(int), "mol0": preds.astype(np.float64)}
    )

concat_sub.head()
ncol = concat_sub.shape[1]
ncol



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2454642312.py in <cell line: 0>()
    132         model = LinearRegression(n_jobs=None)
    133         model.fit(X_tr, y_tr)
--> 134         preds[te_idx] = model.predict(X_te)
    135 
    136     # Any residual NaNs (shouldn't happen) -> global median

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in predict(self, X)
    352             Returns predicted values.
    353         """
--> 354         return self._decision_function(X)
    355 
    356     def _set_intercept(self, X_offset, y_offset, X_scale):

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in _decision_function(self, X)
    335         check_is_fitted(self)
    336 
--> 337         X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset=False)
    338         return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
    339 

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    563             raise ValueError("Validation should be done on X, y or both.")
    564         elif not no_val_X and no_val_y:
--> 565             X = check_array(X, input_name="X", **check_params)
    566             out = X
    567         elif no_val_X and not no_val_y:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    919 
    920         if force_all_finite:
--> 921             _assert_all_finite(
    922                 array,
    923                 input_name=input_name,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
    159                 "#estimators-that-handle-nan-values"
    160             )
--> 161         raise ValueError(msg_err)
    162 
    163 

ValueError: Input X contains NaN.
LinearRegression does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values

## === cell 6
if concat_sub.shape[1] > 2:
    corr = concat_sub.iloc[:, 1:].corr()
else:
    corr = pd.DataFrame()
corr



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1217246646.py in <cell line: 0>()
----> 1 if concat_sub.shape[1] > 2:
      2     corr = concat_sub.iloc[:, 1:].corr()
      3 else:
      4     corr = pd.DataFrame()
      5 corr

NameError: name 'concat_sub' is not defined

## === cell 7
if concat_sub.shape[1] > 3:
    corr = concat_sub.iloc[:, 1:].corr()
    mask = np.zeros_like(corr, dtype=bool)  # fix deprecated np.bool
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="prism",
        vmin=0.96,
        center=0,
        square=True,
        linewidths=1,
        annot=True,
        fmt=".4f",
        ax=ax,
    )
    plt.show()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1036941174.py in <cell line: 0>()
----> 1 if concat_sub.shape[1] > 3:
      2     corr = concat_sub.iloc[:, 1:].corr()
      3     mask = np.zeros_like(corr, dtype=bool)  # fix deprecated np.bool
      4     mask[np.triu_indices_from(mask)] = True
      5     f, ax = plt.subplots(figsize=(11, 9))

NameError: name 'concat_sub' is not defined

## === cell 8
concat_sub["m_max"] = concat_sub.iloc[:, 1:].max(axis=1)
concat_sub["m_min"] = concat_sub.iloc[:, 1:].min(axis=1)
concat_sub["m_median"] = concat_sub.iloc[:, 1:].median(axis=1)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2215861327.py in <cell line: 0>()
----> 1 concat_sub["m_max"] = concat_sub.iloc[:, 1:].max(axis=1)
      2 concat_sub["m_min"] = concat_sub.iloc[:, 1:].min(axis=1)
      3 concat_sub["m_median"] = concat_sub.iloc[:, 1:].median(axis=1)
      4 

NameError: name 'concat_sub' is not defined

## === cell 9
concat_sub.describe()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2125283769.py in <cell line: 0>()
----> 1 concat_sub.describe()
      2 

NameError: name 'concat_sub' is not defined

## === cell 10
concat_sub.head(10)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4210472026.py in <cell line: 0>()
----> 1 concat_sub.head(10)
      2 

NameError: name 'concat_sub' is not defined

## === cell 11
cutoff_lo = 0.8
cutoff_hi = 0.2



## === cell 12
concat_sub["scalar_coupling_constant"] = concat_sub["m_median"]

sub = concat_sub[["id", "scalar_coupling_constant"]].copy()
sub["id"] = sub["id"].astype(int)

out_path = "stack_median.csv"
sub.to_csv(out_path, index=False, float_format="%.6f")

print("Wrote submission:", out_path)
print(sub.head())
print("Rows:", len(sub))

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3607038130.py in <cell line: 0>()
----> 1 concat_sub["scalar_coupling_constant"] = concat_sub["m_median"]
      2 
      3 sub = concat_sub[["id", "scalar_coupling_constant"]].copy()
      4 sub["id"] = sub["id"].astype(int)
      5 

NameError: name 'concat_sub' is not defined
