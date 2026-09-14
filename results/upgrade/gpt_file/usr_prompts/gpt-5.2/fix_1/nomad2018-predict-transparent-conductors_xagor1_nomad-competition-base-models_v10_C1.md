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
Predict the formation energy and bandgap energy of a material.

## Metric
Column-wise root mean squared logarithmic error.

## Submission Format
For each id in the test set, you must predict a value for both formation_energy_ev_natom and bandgap_energy_ev. The file should contain a header and have the following format:
```
id,formation_energy_ev_natom,bandgap_energy_ev
1,0.1779,1.8892
2,0.1779,1.8892
3,0.1779,1.8892
...
```

## Dataset
The following information has been included:

- Spacegroup (a label identifying the symmetry of the material)
- Total number of Al, Ga, In and O atoms in the unit cell ($\N_{total}$)
- Relative compositions of Al, Ga, and In (x, y, z)
- Lattice vectors and angles: lv1, lv2, lv3 (which are lengths given in units of angstroms ($10^{-10}$ meters) and $\alpha, \beta, \gamma$ (which are angles in degrees between 0° and 360°)

Note: For each line of the CSV file, the corresponding spatial positions of all of the atoms in the unit cell (expressed in Cartesian coordinates) are provided as a separate file.

train.csv - contains a set of materials for which the bandgap and formation energies are provided

test.csv - contains the set of materials for which you must predict the bandgap and formation energies

/{train|test}/{id}/geometry.xyz - files with spatial information about the material. The file name corresponds to the id in the respective csv files.

# 2. Python version

3.6

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (89 lines)
            sample_submission.csv (241 lines)
            sample_submission.csv.zip (765 Bytes)
            test.csv (241 lines)
            test.csv.zip (6.0 kB)
            test.zip (505.0 kB)
            train.csv (2161 lines)
            train.csv.zip (56.7 kB)
            train.zip (4.5 MB)
            nomad2018-predict-transparent-conductors/
                description.md (89 lines)
                sample_submission.csv (241 lines)
                ... and 7 other files
                nomad2018-predict-transparent-conductors/
                test/
                    1/
                        geometry.xyz (3.0 kB)
                    10/
                        geometry.xyz (3.0 kB)
                    ... and 239 other folders
                train/
                    1/
                        geometry.xyz (5.5 kB)
                    10/
                        geometry.xyz (2.3 kB)
                    ... and 2159 other folders
            test/
                1/
                    geometry.xyz (3.0 kB)
                10/
                    geometry.xyz (3.0 kB)
                ... and 239 other folders
            train/
                1/
                    geometry.xyz (5.5 kB)
                10/
                    geometry.xyz (2.3 kB)
                ... and 2159 other folders
        input/
            description.md (89 lines)
            sample_submission.csv (241 lines)
            sample_submission.csv.zip (765 Bytes)
            test.csv (241 lines)
            test.csv.zip (6.0 kB)
            test.zip (505.0 kB)
            train.csv (2161 lines)
            train.csv.zip (56.7 kB)
            train.zip (4.5 MB)
            nomad2018-predict-transparent-conductors/
                description.md (89 lines)
                sample_submission.csv (241 lines)
                ... and 7 other files
                nomad2018-predict-transparent-conductors/
                test/
                    1/
                        geometry.xyz (3.0 kB)
                    10/
                        geometry.xyz (3.0 kB)
                    ... and 239 other folders
                train/
                    1/
                        geometry.xyz (5.5 kB)
                    10/
                        geometry.xyz (2.3 kB)
                    ... and 2159 other folders
            test/
                1/
                    geometry.xyz (3.0 kB)
                10/
                    geometry.xyz (3.0 kB)
                ... and 239 other folders
            train/
                1/
                    geometry.xyz (5.5 kB)
                10/
                    geometry.xyz (2.3 kB)
                ... and 2159 other folders
        working/
            nomad2018-predict-transparent-conductors/
                description.md (89 lines)
                sample_submission.csv (241 lines)
                ... and 7 other files
                nomad2018-predict-transparent-conductors/
                test/
                    1/
                        geometry.xyz (3.0 kB)
                    10/
                        geometry.xyz (3.0 kB)
                    ... and 239 other folders
                train/
                    1/
                        geometry.xyz (5.5 kB)
                    10/
                        geometry.xyz (2.3 kB)
                    ... and 2159 other folders
```

-> data/nomad2018-predict-transparent-conductors/sample_submission.csv has 240 rows and 3 columns.
The columns are: id, formation_energy_ev_natom, bandgap_energy_ev

-> data/nomad2018-predict-transparent-conductors/test.csv has 240 rows and 12 columns.
The columns are: id, spacegroup, number_of_total_atoms, percent_atom_al, percent_atom_ga, percent_atom_in, lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang, lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree

-> data/nomad2018-predict-transparent-conductors/train.csv has 2160 rows and 14 columns.
The columns are: id, spacegroup, number_of_total_atoms, percent_atom_al, percent_atom_ga, percent_atom_in, lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang, lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree, formation_energy_ev_natom, bandgap_energy_ev

-> data/sample_submission.csv has 240 rows and 3 columns.
The columns are: id, formation_energy_ev_natom, bandgap_energy_ev

-> data/test.csv has 240 rows and 12 columns.
The columns are: id, spacegroup, number_of_total_atoms, percent_atom_al, percent_atom_ga, percent_atom_in, lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang, lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree

-> data/train.csv has 2160 rows and 14 columns.
The columns are: id, spacegroup, number_of_total_atoms, percent_atom_al, percent_atom_ga, percent_atom_in, lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang, lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree, formation_energy_ev_natom, bandgap_energy_ev

-> (stopped after 10 files for performance)

# 5. Target score

0.07034

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import gc
import time
import numpy as np
import pandas as pd
import glob
import io
import math
import matplotlib


from sklearn.cross_validation import train_test_split
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, LinearRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import seaborn as sns
print(os.listdir("../input"))
from sklearn import tree
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1190775903.py in <cell line: 0>()
     10 
     11 
---> 12 from sklearn.cross_validation import train_test_split
     13 import xgboost as xgb
     14 from xgboost import plot_importance

ModuleNotFoundError: No module named 'sklearn.cross_validation'

## === cell 1
path = '../input/'
train_df = pd.read_csv(path+"/train.csv")
test_df = pd.read_csv(path+"/test.csv")


## === cell 2
print("Training data shape")
print(train_df.shape)
print("Testing data shape")
print(test_df.shape)


## === cell 3
print("Training columns")
print(train_df.columns)
print("Testing columns")
print(test_df.columns)


## === cell 4
print(train_df.dtypes)
print(test_df.dtypes)


## === cell 5
Targets_df=pd.DataFrame()
Targets_df["bandgap_energy_ev"]=train_df["bandgap_energy_ev"].copy()
Targets_df["formation_energy_ev_natom"]=train_df["formation_energy_ev_natom"].copy()
train_df=train_df.drop(["formation_energy_ev_natom","bandgap_energy_ev"],axis=1)


## === cell 6
train_id_df=pd.DataFrame()
train_id_df["id"]=train_df["id"].copy()
train_df=train_df.drop(["id"],axis=1)
test_id_df=pd.DataFrame()
test_id_df["id"]=test_df["id"].copy()
test_df=test_df.drop(["id"],axis=1)


## === cell 7
combined_df = pd.concat([train_df, test_df], ignore_index=True)


## === cell 8
numerical_df=pd.DataFrame.copy(combined_df[['number_of_total_atoms', 'percent_atom_al',
       'percent_atom_ga', 'percent_atom_in', 'lattice_vector_1_ang',
       'lattice_vector_2_ang', 'lattice_vector_3_ang',
       'lattice_angle_alpha_degree', 'lattice_angle_beta_degree',
       'lattice_angle_gamma_degree']])

one_hot_df=pd.DataFrame.copy(combined_df[["spacegroup"]])

one_hot_df=pd.get_dummies(one_hot_df,prefix=["spacegroup"],
                       columns=["spacegroup"])

features_df=pd.concat([numerical_df,one_hot_df],axis=1)







features_df=pd.concat([numerical_df,one_hot_df],axis=1)


## === cell 10
print("Total number of null values in the df")
print(features_df.isna().sum().sum())


## === cell 11
training_examples=features_df.iloc[0:2400].copy()
test_examples=features_df.iloc[2400:3000].copy()


## === cell 12
def rmsle_cv(model):
    rmsle= np.sqrt(-cross_val_score(model, training_examples, training_targets, scoring="neg_mean_squared_log_error", cv = 5))
    return(rmsle)


## === cell 13
training_targets=Targets_df["bandgap_energy_ev"].copy()
model_ridge=RidgeCV(alphas = [0,0.001,0.01,0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75],cv=5).fit(training_examples, training_targets)
print(model_ridge.alpha_)
BG_rmsle=rmsle_cv(model_ridge).mean()
print(BG_rmsle)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3773416773.py in <cell line: 0>()
      1 training_targets=Targets_df["bandgap_energy_ev"].copy()
      2 #Ridge cv model for range of alphas. Can also set cv number, with cv=X
----> 3 model_ridge=RidgeCV(alphas = [0,0.001,0.01,0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75],cv=5).fit(training_examples, training_targets)
      4 #model_ridge=RidgeCV(alphas = [0.001],cv=5).fit(training_examples, training_targets)
      5 #Some possible outputs to print, like best alpha, a list of the coefficients, RMSE etc

NameError: name 'RidgeCV' is not defined

## === cell 14
ridge_BG_preds = model_ridge.predict(test_examples)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1813306024.py in <cell line: 0>()
      1 #Make predictions
----> 2 ridge_BG_preds = model_ridge.predict(test_examples)

NameError: name 'model_ridge' is not defined

## === cell 15
training_targets=Targets_df["formation_energy_ev_natom"].copy()
model_ridge=RidgeCV(alphas = [0,0.001,0.01,0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75],cv=5).fit(training_examples, training_targets)
print(model_ridge.alpha_)
EF_rmsle=rmsle_cv(model_ridge).mean()
print(EF_rmsle)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/116075011.py in <cell line: 0>()
      1 training_targets=Targets_df["formation_energy_ev_natom"].copy()
      2 #Ridge cv model for range of alphas. Can also set cv number, with cv=X
----> 3 model_ridge=RidgeCV(alphas = [0,0.001,0.01,0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75],cv=5).fit(training_examples, training_targets)
      4 #model_ridge=RidgeCV(alphas = [0.001],cv=5).fit(training_examples, training_targets)
      5 #Some possible outputs to print, like best alpha, a list of the coefficients, RMSE etc

NameError: name 'RidgeCV' is not defined

## === cell 16
print("Expected combined error")
combined_rmsle=(EF_rmsle+BG_rmsle)/2
print(combined_rmsle)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2185493966.py in <cell line: 0>()
      1 print("Expected combined error")
----> 2 combined_rmsle=(EF_rmsle+BG_rmsle)/2
      3 print(combined_rmsle)
      4 #Expected .1095 for no reg results, got .1051. Delta = 0.0044
      5 #Expect .0920 with 0.5 skew, actual is 0.0933. Delta = 0.0013 (v close!)

NameError: name 'EF_rmsle' is not defined

## === cell 17
ridge_EF_preds = model_ridge.predict(test_examples)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3100701816.py in <cell line: 0>()
      1 #Make predictions
----> 2 ridge_EF_preds = model_ridge.predict(test_examples)

NameError: name 'model_ridge' is not defined

## === cell 18
Predictions_df=pd.DataFrame()
Predictions_df["id"]=test_id_df["id"].copy()
Predictions_df["formation_energy_ev_natom"]=ridge_EF_preds
Predictions_df["bandgap_energy_ev"]=ridge_BG_preds
Predictions_df.to_csv("Ridge_Nomad.csv",index=False)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1514882775.py in <cell line: 0>()
      1 Predictions_df=pd.DataFrame()
      2 Predictions_df["id"]=test_id_df["id"].copy()
----> 3 Predictions_df["formation_energy_ev_natom"]=ridge_EF_preds
      4 Predictions_df["bandgap_energy_ev"]=ridge_BG_preds
      5 Predictions_df.to_csv("Ridge_Nomad.csv",index=False)

NameError: name 'ridge_EF_preds' is not defined

## === cell 20
training_targets=Targets_df["bandgap_energy_ev"].copy()
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005,0.0001,0.00005,1e-5,1e-6,0],cv=5).fit(training_examples, np.ravel(training_targets))
print(model_lasso.alpha_)
BG_rmsle=rmsle_cv(model_lasso).mean()
print(BG_rmsle)
lasso_coef = pd.Series(model_lasso.coef_, index = training_examples.columns)
print("Lasso picked " + str(sum(lasso_coef != 0)) + " variables and eliminated the other " +  str(sum(lasso_coef == 0)) + " variables")


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1642144331.py in <cell line: 0>()
      1 #Lasso CV for a range of alphas
      2 training_targets=Targets_df["bandgap_energy_ev"].copy()
----> 3 model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005,0.0001,0.00005,1e-5,1e-6,0],cv=5).fit(training_examples, np.ravel(training_targets))
      4 #print(model_lasso.coef_)
      5 print(model_lasso.alpha_)

NameError: name 'LassoCV' is not defined

## === cell 21
lasso_BG_preds = model_lasso.predict(test_examples)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1894016635.py in <cell line: 0>()
      1 #Make predictions
----> 2 lasso_BG_preds = model_lasso.predict(test_examples)

NameError: name 'model_lasso' is not defined

## === cell 22
training_targets=Targets_df["formation_energy_ev_natom"].copy()
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005,1e-5,0],cv=5).fit(training_examples, np.ravel(training_targets))
print(model_lasso.alpha_)
EF_rmsle=rmsle_cv(model_lasso).mean()
print(EF_rmsle)
lasso_coef = pd.Series(model_lasso.coef_, index = training_examples.columns)
print("Lasso picked " + str(sum(lasso_coef != 0)) + " variables and eliminated the other " +  str(sum(lasso_coef == 0)) + " variables")


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/951602114.py in <cell line: 0>()
      1 #Lasso CV for a range of alphas
      2 training_targets=Targets_df["formation_energy_ev_natom"].copy()
----> 3 model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005,1e-5,0],cv=5).fit(training_examples, np.ravel(training_targets))
      4 #print(model_lasso.coef_)
      5 print(model_lasso.alpha_)

NameError: name 'LassoCV' is not defined

## === cell 23
lasso_EF_preds = model_lasso.predict(test_examples)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/234626119.py in <cell line: 0>()
      1 #Make predictions
----> 2 lasso_EF_preds = model_lasso.predict(test_examples)

NameError: name 'model_lasso' is not defined

## === cell 24
print("Expected combined error")
combined_rmsle=(EF_rmsle+BG_rmsle)/2
print(combined_rmsle)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2535082309.py in <cell line: 0>()
      1 print("Expected combined error")
----> 2 combined_rmsle=(EF_rmsle+BG_rmsle)/2
      3 print(combined_rmsle)
      4 
      5 #Lasso alpha ->0 seems to give the best results, i.e. no lasso.

NameError: name 'EF_rmsle' is not defined

## === cell 25
Predictions_df=pd.DataFrame()
Predictions_df["id"]=test_id_df["id"].copy()
Predictions_df["formation_energy_ev_natom"]=lasso_EF_preds
Predictions_df["bandgap_energy_ev"]=lasso_BG_preds
Predictions_df.to_csv("Lasso_Nomad.csv",index=False)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2661849264.py in <cell line: 0>()
      1 Predictions_df=pd.DataFrame()
      2 Predictions_df["id"]=test_id_df["id"].copy()
----> 3 Predictions_df["formation_energy_ev_natom"]=lasso_EF_preds
      4 Predictions_df["bandgap_energy_ev"]=lasso_BG_preds
      5 Predictions_df.to_csv("Lasso_Nomad.csv",index=False)

NameError: name 'lasso_EF_preds' is not defined

## === cell 28
last=len(model_xgb.loc[:])-1
model_xgb.loc[last]


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1098063572.py in <cell line: 0>()
----> 1 last=len(model_xgb.loc[:])-1
      2 model_xgb.loc[last]

NameError: name 'model_xgb' is not defined

## === cell 29
training_targets=np.log1p(Targets_df["bandgap_energy_ev"].copy())
dtrain = xgb.DMatrix(training_examples, label = training_targets)
dtest = xgb.DMatrix(test_examples)
params = {"max_depth":2,
          "eta":0.1,
          'gamma':0,  
          'subsample':0.6,
          'colsample_bytree':0.6,
          'min_child_weight':7,
         }
model_xgb = xgb.cv(params, dtrain,  num_boost_round=500,early_stopping_rounds=100)
model_xgb.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
last=len(model_xgb.loc[:])-1
print(model_xgb.loc[last:,["test-rmse-mean"]])


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2248814308.py in <cell line: 0>()
      2 #training_targets=np.log1p(Targets_df["bandgap_energy_ev"].copy())
      3 training_targets=np.log1p(Targets_df["bandgap_energy_ev"].copy())
----> 4 dtrain = xgb.DMatrix(training_examples, label = training_targets)
      5 dtest = xgb.DMatrix(test_examples)
      6 #tune wrt params

NameError: name 'xgb' is not defined

## === cell 30
model_xgb = xgb.XGBRegressor(n_estimators=367, max_depth=2, learning_rate=0.1,gamma=0,subsample=0.6,colsample_bytree=0.6,min_child_weight=7)
model_xgb.fit(training_examples, training_targets)
xgb.plot_importance(model_xgb)

xgb_BG_preds = np.expm1(model_xgb.predict(test_examples))


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/348423491.py in <cell line: 0>()
      1 #Fit a model with optimized parameters
----> 2 model_xgb = xgb.XGBRegressor(n_estimators=367, max_depth=2, learning_rate=0.1,gamma=0,subsample=0.6,colsample_bytree=0.6,min_child_weight=7)
      3 model_xgb.fit(training_examples, training_targets)
      4 xgb.plot_importance(model_xgb)
      5 

NameError: name 'xgb' is not defined

## === cell 31
training_targets=Targets_df["formation_energy_ev_natom"].copy()
dtrain = xgb.DMatrix(training_examples, label = training_targets)
dtest = xgb.DMatrix(test_examples)
params = {"max_depth":3,
          "eta":0.1,
          'gamma':0,  
          'subsample':1,
          'colsample_bytree':1,
          'min_child_weight':1,
         }
model_xgb = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model_xgb.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3883554517.py in <cell line: 0>()
      1 #XGB model for EF
      2 training_targets=Targets_df["formation_energy_ev_natom"].copy()
----> 3 dtrain = xgb.DMatrix(training_examples, label = training_targets)
      4 dtest = xgb.DMatrix(test_examples)
      5 #tune wrt params

NameError: name 'xgb' is not defined

## === cell 32
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(training_examples, training_targets)
xgb.plot_importance(model_xgb)

xgb_EF_preds = model_xgb.predict(test_examples)


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/705179210.py in <cell line: 0>()
      1 #Fit a model with optimized parameters
----> 2 model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
      3 model_xgb.fit(training_examples, training_targets)
      4 xgb.plot_importance(model_xgb)
      5 

NameError: name 'xgb' is not defined

## === cell 33
Predictions_df=pd.DataFrame()
Predictions_df["id"]=test_id_df["id"].copy()
Predictions_df["formation_energy_ev_natom"]=xgb_EF_preds
Predictions_df["bandgap_energy_ev"]=xgb_BG_preds
Predictions_df.to_csv("XGB_Nomad.csv",index=False)


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/400627243.py in <cell line: 0>()
      1 Predictions_df=pd.DataFrame()
      2 Predictions_df["id"]=test_id_df["id"].copy()
----> 3 Predictions_df["formation_energy_ev_natom"]=xgb_EF_preds
      4 Predictions_df["bandgap_energy_ev"]=xgb_BG_preds
      5 Predictions_df.to_csv("XGB_Nomad.csv",index=False)

NameError: name 'xgb_EF_preds' is not defined
