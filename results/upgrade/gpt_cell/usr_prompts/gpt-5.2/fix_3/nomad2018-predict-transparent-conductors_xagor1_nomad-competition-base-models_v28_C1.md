# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.6

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

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

from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.linear_model import (
    Ridge,
    RidgeCV,
    ElasticNet,
    LassoCV,
    LassoLarsCV,
    LinearRegression,
)
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
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings("ignore")


## === cell 1
path = '../input/'
train_df = pd.read_csv(path+"/train.csv")
test_df = pd.read_csv(path+"/test.csv")


## === cell 2
print("Training data shape","\n")
print(train_df.shape,"\n")
print("Testing data shape","\n")
print(test_df.shape,"\n")

print("Training columns","\n")
print(train_df.columns,"\n")
print("Testing columns","\n")
print(test_df.columns,"\n")



## === cell 3
Targets_df=pd.DataFrame()
Targets_df["bandgap_energy_ev"]=train_df["bandgap_energy_ev"].copy()
Targets_df["formation_energy_ev_natom"]=train_df["formation_energy_ev_natom"].copy()
train_df=train_df.drop(["formation_energy_ev_natom","bandgap_energy_ev"],axis=1)

train_id_df=pd.DataFrame()
train_id_df["id"]=train_df["id"].copy()
train_df=train_df.drop(["id"],axis=1)
test_id_df=pd.DataFrame()
test_id_df["id"]=test_df["id"].copy()
test_df=test_df.drop(["id"],axis=1)

combined_df = pd.concat([train_df, test_df], ignore_index=True)
print("Total number of null values in the df","\n")
print(combined_df.isna().sum().sum())


## === cell 4
numerical_df=pd.DataFrame.copy(combined_df[['number_of_total_atoms', 'percent_atom_al',
       'percent_atom_ga', 'percent_atom_in', 'lattice_vector_1_ang',
       'lattice_vector_2_ang', 'lattice_vector_3_ang',
       'lattice_angle_alpha_degree', 'lattice_angle_beta_degree',
       'lattice_angle_gamma_degree']])

one_hot_df=pd.DataFrame.copy(combined_df[["spacegroup"]])

one_hot_df=pd.get_dummies(one_hot_df,prefix=["spacegroup"],
                       columns=["spacegroup"])

features_df=pd.concat([numerical_df,one_hot_df],axis=1)


skewed_feats= numerical_df.skew()
skewed_feats = skewed_feats[skewed_feats > 0.1]
skewed_feats = skewed_feats.index

unskewed_feats= numerical_df.skew()
unskewed_feats = unskewed_feats[unskewed_feats < 0.1]
unskewed_feats = unskewed_feats.index


transform_df=pd.DataFrame()
transform_df[unskewed_feats]=(numerical_df[unskewed_feats]
                               - numerical_df[unskewed_feats].mean()) / (numerical_df[unskewed_feats].max() - numerical_df[unskewed_feats].min())
transform_df[skewed_feats] = np.log1p(numerical_df[skewed_feats])


features_transform_df=pd.concat([transform_df,one_hot_df],axis=1)
features_df=pd.concat([numerical_df,one_hot_df],axis=1)


## === cell 5
training_examples=features_df.iloc[0:2400].copy()
test_examples=features_df.iloc[2400:3000].copy()
training_examples_transform=features_transform_df.iloc[0:2400].copy()
test_examples_transform=features_transform_df.iloc[2400:3000].copy()


## === cell 6
def rmsle_cv(model):
    rmsle= np.sqrt(-cross_val_score(model, training_examples_transform, training_targets, scoring="neg_mean_squared_log_error", cv = 5))
    return(rmsle)


## === cell 7
n_train = len(Targets_df)
X_train = features_transform_df.iloc[:n_train].copy()
X_test = features_transform_df.iloc[n_train:].copy()
test_ids = test_id_df["id"].copy()

training_targets = Targets_df["bandgap_energy_ev"].copy()
model_linear = LinearRegression().fit(X_train, training_targets)
linear_BG_pred = model_linear.predict(X_test)
BG_rmsle = np.sqrt(
    -cross_val_score(
        model_linear,
        X_train,
        training_targets,
        scoring="neg_mean_squared_log_error",
        cv=5,
    )
).mean()
print("Band gap RMSLE:")
print(BG_rmsle, "\n")

training_targets = Targets_df["formation_energy_ev_natom"].copy()
model_linear = LinearRegression().fit(X_train, training_targets)
linear_EF_pred = model_linear.predict(X_test)
EF_rmsle = np.sqrt(
    -cross_val_score(
        model_linear,
        X_train,
        training_targets,
        scoring="neg_mean_squared_log_error",
        cv=5,
    )
).mean()
print("Formation Energy RMSLE:")
print(EF_rmsle, "\n")

print("Expected combined RMSLE")
combined_rmsle = (EF_rmsle + BG_rmsle) / 2
print(combined_rmsle)

Predictions_df = pd.DataFrame()
Predictions_df["id"] = test_ids
Predictions_df["formation_energy_ev_natom"] = linear_EF_pred
Predictions_df["bandgap_energy_ev"] = linear_BG_pred
Predictions_df.to_csv("Linear_Nomad.csv", index=False)


## === cell 8
training_targets=np.log1p(Targets_df["bandgap_energy_ev"].copy())
dtrain = xgb.DMatrix(training_examples, label = training_targets)
dtest = xgb.DMatrix(test_examples)
params = {"max_depth":2,
          "eta":0.1,
          'gamma':0,  
          'subsample':0.8,
          'colsample_bytree':1,
          'min_child_weight':10,
         }
model_xgb = xgb.cv(params, dtrain,  num_boost_round=500,early_stopping_rounds=100)
model_xgb.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
last=len(model_xgb.loc[:])-1
print(model_xgb.loc[last:,["test-rmse-mean"]])


## --- ERROR in cell 8, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mXGBoostError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4013060069.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;31m#XGB model for BG[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0mtraining_targets[0m[0;34m=[0m[0mnp[0m[0;34m.[0m[0mlog1p[0m[0;34m([0m[0mTargets_df[0m[0;34m[[0m[0;34m"bandgap_energy_ev"[0m[0;34m][0m[0;34m.[0m[0mcopy[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 3[0;31m [0mdtrain[0m [0;34m=[0m [0mxgb[0m[0;34m.[0m[0mDMatrix[0m[0;34m([0m[0mtraining_examples[0m[0;34m,[0m [0mlabel[0m [0;34m=[0m [0mtraining_targets[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0mdtest[0m [0;34m=[0m [0mxgb[0m[0;34m.[0m[0mDMatrix[0m[0;34m([0m[0mtest_examples[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;31m#tune wrt params[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36minner_f[0;34m(*args, **kwargs)[0m
[1;32m    728[0m             [0;32mfor[0m [0mk[0m[0;34m,[0m [0marg[0m [0;32min[0m [0mzip[0m[0;34m([0m[0msig[0m[0;34m.[0m[0mparameters[0m[0;34m,[0m [0margs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    729[0m                 [0mkwargs[0m[0;34m[[0m[0mk[0m[0;34m][0m [0;34m=[0m [0marg[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 730[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    731[0m [0;34m[0m[0m
[1;32m    732[0m         [0;32mreturn[0m [0minner_f[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36m__init__[0;34m(self, data, label, weight, base_margin, missing, silent, feature_names, feature_types, nthread, group, qid, label_lower_bound, label_upper_bound, feature_weights, enable_categorical, data_split_mode)[0m
[1;32m    867[0m         [0mself[0m[0;34m.[0m[0mhandle[0m [0;34m=[0m [0mhandle[0m[0;34m[0m[0;34m[0m[0m
[1;32m    868[0m [0;34m[0m[0m
[0;32m--> 869[0;31m         self.set_info(
[0m[1;32m    870[0m             [0mlabel[0m[0;34m=[0m[0mlabel[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    871[0m             [0mweight[0m[0;34m=[0m[0mweight[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36minner_f[0;34m(*args, **kwargs)[0m
[1;32m    728[0m             [0;32mfor[0m [0mk[0m[0;34m,[0m [0marg[0m [0;32min[0m [0mzip[0m[0;34m([0m[0msig[0m[0;34m.[0m[0mparameters[0m[0;34m,[0m [0margs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    729[0m                 [0mkwargs[0m[0;34m[[0m[0mk[0m[0;34m][0m [0;34m=[0m [0marg[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 730[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    731[0m [0;34m[0m[0m
[1;32m    732[0m         [0;32mreturn[0m [0minner_f[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36mset_info[0;34m(self, label, weight, base_margin, group, qid, label_lower_bound, label_upper_bound, feature_names, feature_types, feature_weights)[0m
[1;32m    930[0m [0;34m[0m[0m
[1;32m    931[0m         [0;32mif[0m [0mlabel[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 932[0;31m             [0mself[0m[0;34m.[0m[0mset_label[0m[0;34m([0m[0mlabel[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    933[0m         [0;32mif[0m [0mweight[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    934[0m             [0mself[0m[0;34m.[0m[0mset_weight[0m[0;34m([0m[0mweight[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36mset_label[0;34m(self, label)[0m
[1;32m   1068[0m         [0;32mfrom[0m [0;34m.[0m[0mdata[0m [0;32mimport[0m [0mdispatch_meta_backend[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1069[0m [0;34m[0m[0m
[0;32m-> 1070[0;31m         [0mdispatch_meta_backend[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mlabel[0m[0;34m,[0m [0;34m"label"[0m[0;34m,[0m [0;34m"float"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1071[0m [0;34m[0m[0m
[1;32m   1072[0m     [0;32mdef[0m [0mset_weight[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mweight[0m[0;34m:[0m [0mArrayLike[0m[0;34m)[0m [0;34m->[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/data.py[0m in [0;36mdispatch_meta_backend[0;34m(matrix, data, name, dtype)[0m
[1;32m   1223[0m         [0;32mreturn[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1224[0m     [0;32mif[0m [0m_is_pandas_series[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1225[0;31m         [0m_meta_from_pandas_series[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mname[0m[0;34m,[0m [0mdtype[0m[0;34m,[0m [0mhandle[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1226[0m         [0;32mreturn[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1227[0m     [0;32mif[0m [0m_is_dlpack[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/data.py[0m in [0;36m_meta_from_pandas_series[0;34m(data, name, dtype, handle)[0m
[1;32m    543[0m         [0mdata[0m [0;34m=[0m [0mdata[0m[0;34m.[0m[0mto_dense[0m[0;34m([0m[0;34m)[0m  [0;31m# type: ignore[0m[0;34m[0m[0;34m[0m[0m
[1;32m    544[0m     [0;32massert[0m [0mlen[0m[0;34m([0m[0mdata[0m[0;34m.[0m[0mshape[0m[0;34m)[0m [0;34m==[0m [0;36m1[0m [0;32mor[0m [0mdata[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m1[0m[0;34m][0m [0;34m==[0m [0;36m0[0m [0;32mor[0m [0mdata[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m1[0m[0;34m][0m [0;34m==[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 545[0;31m     [0m_meta_from_numpy[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mname[0m[0;34m,[0m [0mdtype[0m[0;34m,[0m [0mhandle[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    546[0m [0;34m[0m[0m
[1;32m    547[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/data.py[0m in [0;36m_meta_from_numpy[0;34m(data, field, dtype, handle)[0m
[1;32m   1157[0m         [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"Masked array is not supported."[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1158[0m     [0minterface_str[0m [0;34m=[0m [0m_array_interface[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1159[0;31m     [0m_check_call[0m[0;34m([0m[0m_LIB[0m[0;34m.[0m[0mXGDMatrixSetInfoFromInterface[0m[0;34m([0m[0mhandle[0m[0;34m,[0m [0mc_str[0m[0;34m([0m[0mfield[0m[0;34m)[0m[0;34m,[0m [0minterface_str[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1160[0m [0;34m[0m[0m
[1;32m   1161[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36m_check_call[0;34m(ret)[0m
[1;32m    280[0m     """
[1;32m    281[0m     [0;32mif[0m [0mret[0m [0;34m!=[0m [0;36m0[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 282[0;31m         [0;32mraise[0m [0mXGBoostError[0m[0;34m([0m[0mpy_str[0m[0;34m([0m[0m_LIB[0m[0;34m.[0m[0mXGBGetLastError[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    283[0m [0;34m[0m[0m
[1;32m    284[0m [0;34m[0m[0m

[0;31mXGBoostError[0m: [08:38:32] /workspace/src/data/data.cc:501: Check failed: this->labels.Size() % this->num_row_ == 0 (2160 vs. 0) : Incorrect size for labels.
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3588ca) [0x7fff8386c8ca]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x389af7) [0x7fff8389daf7]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x38ab51) [0x7fff8389eb51]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGDMatrixSetInfoFromInterface+0xb0) [0x7fff836723a0]
  [bt] (4) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (5) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]
  [bt] (6) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7ffff63bc4d8]
  [bt] (7) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0x9c8e) [0x7ffff63bbc8e]
  [bt] (8) /usr/bin/python3(_PyObject_MakeTpCall+0x27c) [0x52f85c]



## === cell 9
model_xgb = xgb.XGBRegressor(n_estimators=last, max_depth=2, learning_rate=0.1,
                             gamma=0,subsample=0.8,colsample_bytree=1,min_child_weight=10)
model_xgb.fit(training_examples, training_targets)
xgb.plot_importance(model_xgb)

xgb_BG_preds = np.expm1(model_xgb.predict(test_examples))
