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

2.980106608063688

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV



## === cell 1
SEED = 1234
FOLDS = 3




## === cell 2
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(SEED)




## === cell 3
def resolve_data_dir():
    candidates = [
        "/kaggle/input/champs-scalar-coupling",
        "/kaggle/data/champs-scalar-coupling",
        "/kaggle/data/input/champs-scalar-coupling",
        "../input/champs-scalar-coupling",
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.exists(os.path.join(p, "train.csv")):
            return p
    raise FileNotFoundError(
        "Could not find champs-scalar-coupling data directory in known locations."
    )


file_folder = resolve_data_dir()
train = pd.read_csv(f"{file_folder}/train.csv")
test = pd.read_csv(f"{file_folder}/test.csv")
structures = pd.read_csv(f"{file_folder}/structures.csv")

print("data_dir=", file_folder)
print(
    "train={}, test={}, structures={}".format(
        repr(train.shape), repr(test.shape), repr(structures.shape)
    )
)



## === cell 4


def add_structure_features(df, structures_df):
    s0 = structures_df.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "type_0",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
    )
    s1 = structures_df.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "type_1",
            "x": "x1",
            "y": "y1",
            "z": "z1",
        }
    )

    out = df.merge(
        s0[["molecule_name", "atom_index_0", "type_0", "x0", "y0", "z0"]],
        on=["molecule_name", "atom_index_0"],
        how="left",
    )
    out = out.merge(
        s1[["molecule_name", "atom_index_1", "type_1", "x1", "y1", "z1"]],
        on=["molecule_name", "atom_index_1"],
        how="left",
    )

    for c in ["x0", "y0", "z0", "x1", "y1", "z1"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["dist_x"] = out["x0"] - out["x1"]
    out["dist_y"] = out["y0"] - out["y1"]
    out["dist_z"] = out["z0"] - out["z1"]
    out["dist"] = np.sqrt(out["dist_x"] ** 2 + out["dist_y"] ** 2 + out["dist_z"] ** 2)

    out = out.drop(columns=["x0", "y0", "z0", "x1", "y1", "z1"])

    return out


train_fe = add_structure_features(train, structures)
test_fe = add_structure_features(test, structures)

missing_train = train_fe[["type_0", "type_1", "dist"]].isna().mean()
missing_test = test_fe[["type_0", "type_1", "dist"]].isna().mean()
print("Missing rates (train):\n", missing_train)
print("Missing rates (test):\n", missing_test)




## === cell 5
def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition.
    """
    maes = (y_true - y_pred).abs().groupby(types).mean()
    maes = np.log(maes.map(lambda x: max(x, floor)))
    print(maes)
    return maes.mean()




## === cell 6

type1_dist_mean = train_fe.groupby("type_1")["dist"].mean()
mol_type_dist_mean = train_fe.groupby(["molecule_name", "type"])["dist"].mean()

train_fe["dist_to_type_1_mean"] = train_fe["type_1"].map(type1_dist_mean)
test_fe["dist_to_type_1_mean"] = test_fe["type_1"].map(type1_dist_mean)

train_fe["molecule_type_dist_mean"] = list(
    zip(train_fe["molecule_name"], train_fe["type"])
)
test_fe["molecule_type_dist_mean"] = list(
    zip(test_fe["molecule_name"], test_fe["type"])
)

train_fe["molecule_type_dist_mean"] = train_fe["molecule_type_dist_mean"].map(
    mol_type_dist_mean
)
test_fe["molecule_type_dist_mean"] = test_fe["molecule_type_dist_mean"].map(
    mol_type_dist_mean
)

global_dist_mean = float(train_fe["dist"].mean())
for col in ["dist_to_type_1_mean", "molecule_type_dist_mean"]:
    train_fe[col] = train_fe[col].fillna(global_dist_mean)
    test_fe[col] = test_fe[col].fillna(global_dist_mean)

predictor_cols = [
    "type_0",
    "type_1",
    "dist",
    "dist_to_type_1_mean",
    "molecule_type_dist_mean",
    "dist_x",
    "dist_y",
    "dist_z",
]
print("Any NaNs in train predictors:", train_fe[predictor_cols].isna().any().any())
print("Any NaNs in test predictors:", test_fe[predictor_cols].isna().any().any())



## === cell 7
y_train = np.log(train_fe["scalar_coupling_constant"].values + 1000.0)

x_train = train_fe[predictor_cols].copy()
x_test = test_fe[predictor_cols].copy()

x_all = pd.concat([x_train, x_test], axis=0, ignore_index=True)
x_all = pd.get_dummies(x_all, columns=["type_0", "type_1"], drop_first=False)

x_train_enc = x_all.iloc[: len(x_train)].reset_index(drop=True)
x_test_enc = x_all.iloc[len(x_train) :].reset_index(drop=True)

model = Lasso(alpha=1.0, random_state=SEED)
pipe = Pipeline([("model", model)])
param_grid = {"model__max_iter": [10, 20]}

cv = GridSearchCV(
    pipe,
    cv=FOLDS,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)
cv.fit(x_train_enc, y_train)
print(
    "best_params_={}\nbest_score_={}".format(
        repr(cv.best_params_), repr(cv.best_score_)
    )
)



## === cell 8
y_true = train_fe["scalar_coupling_constant"].values
y_pred_train = np.exp(cv.predict(x_train_enc)) - 1000.0
gmlm = group_mean_log_mae(pd.Series(y_true), pd.Series(y_pred_train), train_fe["type"])
print("group_mean_log_mae={}".format(gmlm))



## === cell 9
preds_log = cv.predict(x_test_enc)
preds = np.exp(preds_log) - 1000.0

submission = pd.DataFrame(
    {"id": test_fe["id"].values, "scalar_coupling_constant": preds}
)
submission.head()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/821723124.py in <cell line: 0>()
----> 1 preds_log = cv.predict(x_test_enc)
      2 preds = np.exp(preds_log) - 1000.0
      3 
      4 submission = pd.DataFrame(
      5     {"id": test_fe["id"].values, "scalar_coupling_constant": preds}

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in predict(self, X)
    497         """
    498         check_is_fitted(self)
--> 499         return self.best_estimator_.predict(X)
    500 
    501     @available_if(_estimator_has("predict_proba"))

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in predict(self, X, **predict_params)
    479         for _, name, transform in self._iter(with_final=False):
    480             Xt = transform.transform(Xt)
--> 481         return self.steps[-1][1].predict(Xt, **predict_params)
    482 
    483     @available_if(_final_estimator_has("fit_predict"))

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in predict(self, X)
    352             Returns predicted values.
    353         """
--> 354         return self._decision_function(X)
    355 
    356     def _set_intercept(self, X_offset, y_offset, X_scale):

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_coordinate_descent.py in _decision_function(self, X)
   1072             return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
   1073         else:
-> 1074             return super()._decision_function(X)
   1075 
   1076 

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
Lasso does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values

## === cell 10
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Files in working dir:", os.listdir("."))

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/852795567.py in <cell line: 0>()
      1 # Ensure valid CSV submission
----> 2 submission.to_csv("submission.csv", index=False)
      3 print("Wrote submission.csv with shape:", submission.shape)
      4 print("Files in working dir:", os.listdir("."))

NameError: name 'submission' is not defined
