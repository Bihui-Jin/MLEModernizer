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

0.66661

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc
import os

import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


## === cell 1
print(os.listdir("../input"))


## === cell 2
train_original = pd.read_csv("../input/train.csv")
structures_original = pd.read_csv("../input/structures.csv")
test_original = pd.read_csv("../input/test.csv")


## === cell 3
train_original.head()


## === cell 4
structures_original.head()


## === cell 5
test_original.head()


## === cell 6
structures_original[structures_original['molecule_name'] == 'dsgdb9nsd_000015']


## === cell 7
moleculeCount = structures_original.groupby(by=['molecule_name','atom'])[['atom']].count()
moleculeCount.rename(columns={'atom':'count'},inplace = True)
moleculeCount = moleculeCount.unstack(fill_value=0)
moleculeCount = moleculeCount['count'].reset_index()

moleculeCount.head()


## === cell 8
moleculeCount[moleculeCount['molecule_name'] == 'dsgdb9nsd_000015']


## === cell 9
structures = pd.DataFrame.merge(structures_original,moleculeCount
                               ,how='inner'
                               ,left_on = ['molecule_name'] 
                               ,right_on = ['molecule_name']
                              )

structures.head()


## === cell 10
tmp_merge = pd.DataFrame.merge(train_original,structures
                               ,how='left'
                               ,left_on = ['molecule_name','atom_index_0'] 
                               ,right_on = ['molecule_name','atom_index']
                              )

tmp_merge = tmp_merge.merge(structures
                ,how='left'
                ,left_on = ['molecule_name','atom_index_1'] 
                ,right_on = ['molecule_name','atom_index']
               )

tmp_merge.drop(columns=['atom_index_x','atom_index_y','C_x','F_x','H_x','N_x','O_x'],inplace=True)
tmp_merge.columns = ['id' , 'molecule_name' , 'atom_0' , 'atom_1' , 'type' , 'scalar_coupling_constant' , 
                      'atom_nm_0' , 'x_0' , 'y_0' , 'z_0' , 'atom_nm_1' , 'x_1' , 'y_1' , 'z_1','C','F','H','N','O']

train = tmp_merge[['id' , 'molecule_name' , 'atom_0' , 'atom_1' , 'type'  , 'atom_nm_0' , 'x_0' ,
           'y_0' , 'z_0' , 'atom_nm_1' , 'x_1' , 'y_1' , 'z_1','C','F','H','N','O', 'scalar_coupling_constant']]
train.sort_values(by=['id','molecule_name'],inplace=True)
train.reset_index(inplace=True,drop=True)

tmp_merge = None

train.head()


## === cell 11
tmp_merge = pd.DataFrame.merge(test_original,structures
                               ,how='inner'
                               ,left_on = ['molecule_name','atom_index_0'] 
                               ,right_on = ['molecule_name','atom_index']
                              )
tmp_merge = tmp_merge.merge(structures
                ,how='inner'
                ,left_on = ['molecule_name','atom_index_1'] 
                ,right_on = ['molecule_name','atom_index']
               )

tmp_merge.drop(columns=['atom_index_x','atom_index_y','C_x','F_x','H_x','N_x','O_x'],inplace=True)
tmp_merge.columns = ['id' , 'molecule_name' , 'atom_0' , 'atom_1' , 'type' ,  
                      'atom_nm_0' , 'x_0' , 'y_0' , 'z_0' , 'atom_nm_1' , 'x_1' , 'y_1' , 'z_1','C','F','H','N','O']


test = tmp_merge[['id' , 'molecule_name' , 'atom_0' , 'atom_1' , 'type'  , 'atom_nm_0' , 'x_0' ,
           'y_0' , 'z_0' , 'atom_nm_1' , 'x_1' , 'y_1' , 'z_1', 'C','F','H','N','O']]


test.sort_values(by=['id','molecule_name'],inplace=True)
test.reset_index(inplace=True,drop=True)

tmp_merge = None

test.head()


## === cell 12
train_original = None
del train_original
structures_original = None
del structures_original
test_original = None
del test_original
structures = None
del structures
gc.collect()


## === cell 13
train['dist'] = np.linalg.norm(train[['x_0', 'y_0', 'z_0']].values - train[['x_1', 'y_1', 'z_1']].values, axis=1)
test['dist'] = np.linalg.norm(test[['x_0', 'y_0', 'z_0']].values - test[['x_1', 'y_1', 'z_1']].values, axis=1)

train.drop(columns=['x_0', 'y_0', 'z_0','x_1', 'y_1', 'z_1'],inplace=True)
test.drop(columns=['x_0', 'y_0', 'z_0','x_1', 'y_1', 'z_1'],inplace=True)


## === cell 14
train['type'] = pd.Categorical(train['type'])
train['atom_nm_1'] = pd.Categorical(train['atom_nm_1'])
test['type'] = pd.Categorical(test['type'])
test['atom_nm_1'] = pd.Categorical(test['atom_nm_1'])


## === cell 15
train.head()


## === cell 16
test.head()


## === cell 17
X = train[['atom_0' ,  'atom_1' , 'type', 'atom_nm_1', 'C' ,  'F' ,  'H' ,  'N' ,  'O' , 'dist' ]]

y = train['scalar_coupling_constant']


## === cell 18
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=420)


## === cell 19
lgb_train = lgb.Dataset(X_train,y_train,free_raw_data=True)
lgb_eval = lgb.Dataset(X_test,y_test,free_raw_data=True)


## === cell 20
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'learning_rate': 0.05,
    'num_leaves': 50, 
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'num_boost_round':5000,
    'reg_alpha': 0.1,
    'reg_lambda': 0.3,
'early_stopping_rounds':5
         }


## === cell 21
gbm = lgb.train(
    params,
    lgb_train,

    valid_sets=lgb_eval
)


## === cell 22
y_predict = gbm.predict(X_test)
mse = np.sqrt(metrics.mean_squared_error(y_predict,y_test))

print('Mean Squared Error is : '+str(mse))


## === cell 23
submission_df = pd.DataFrame(columns=['id', 'scalar_coupling_constant'])
submission_df['id'] = test['id']
submission_df['scalar_coupling_constant'] = gbm.predict(test[['atom_0' ,  'atom_1' , 'type', 'atom_nm_1', 'C' ,  'F' ,  'H' ,  'N' ,  'O' , 'dist' ]])
submission_df.to_csv('submissions.csv', header=True, index=False)
submission_df.head(10)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3295861900.py in <cell line: 0>()
      1 submission_df = pd.DataFrame(columns=['id', 'scalar_coupling_constant'])
      2 submission_df['id'] = test['id']
----> 3 submission_df['scalar_coupling_constant'] = gbm.predict(test[['atom_0' ,  'atom_1' , 'type', 'atom_nm_1', 'C' ,  'F' ,  'H' ,  'N' ,  'O' , 'dist' ]])
      4 submission_df.to_csv('submissions.csv', header=True, index=False)
      5 submission_df.head(10)

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in predict(self, data, start_iteration, num_iteration, raw_score, pred_leaf, pred_contrib, data_has_header, validate_features, **kwargs)
   4765             else:
   4766                 num_iteration = -1
-> 4767         return predictor.predict(
   4768             data=data,
   4769             start_iteration=start_iteration,

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in predict(self, data, start_iteration, num_iteration, raw_score, pred_leaf, pred_contrib, data_has_header, validate_features)
   1156 
   1157         if isinstance(data, pd_DataFrame):
-> 1158             data = _data_from_pandas(
   1159                 data=data,
   1160                 feature_name="auto",

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _data_from_pandas(data, feature_name, categorical_feature, pandas_categorical)
    832 ) -> Tuple[np.ndarray, List[str], Union[List[str], List[int]], List[List]]:
    833     if len(data.shape) != 2 or data.shape[0] < 1:
--> 834         raise ValueError("Input data must be 2 dimensional and non empty.")
    835 
    836     # take shallow copy in case we modify categorical columns

ValueError: Input data must be 2 dimensional and non empty.
