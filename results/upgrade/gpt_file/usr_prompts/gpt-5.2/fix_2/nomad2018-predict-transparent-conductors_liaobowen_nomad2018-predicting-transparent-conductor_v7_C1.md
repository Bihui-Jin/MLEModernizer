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
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0

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

0.07578

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



## === cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import time
from sklearn.svm import LinearSVR, SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor

import warnings

warnings.filterwarnings("ignore")




## === cell 2
def load_data(road="../input/"):
    """
    Bugfix: Kaggle dataset is nested under ../input/nomad2018-predict-transparent-conductors/.
    Keep behavior but auto-detect the correct folder so reads don't fail.
    """
    gc.collect()

    cand = os.path.join(road, "nomad2018-predict-transparent-conductors")
    base = cand if os.path.exists(os.path.join(cand, "train.csv")) else road

    df_train = pd.read_csv(os.path.join(base, "train.csv"))
    df_test = pd.read_csv(os.path.join(base, "test.csv"))

    sub = pd.DataFrame(columns=["id", "formation_energy_ev_natom", "bandgap_energy_ev"])
    sub["id"] = df_test["id"]

    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train.drop("id", axis=1, inplace=True)
    df_test.drop("id", axis=1, inplace=True)

    gc.collect()
    return df_train, df_test, sub, base


df_train, df_test, sub, DATA_BASE = load_data()



## === cell 3
df_train.head()



## === cell 4
df_train.describe()




## === cell 5
def feature_engine(df):
    df = df.copy()
    df["al"] = df["number_of_total_atoms"] * df["percent_atom_al"]
    df["ga"] = df["number_of_total_atoms"] * df["percent_atom_ga"]
    df["in"] = df["number_of_total_atoms"] * df["percent_atom_in"]
    df["all"] = df["al"] + df["ga"] + df["in"]
    df["spacegroup"] = df["spacegroup"].astype("object")

    for c in ["formation_energy_ev_natom", "bandgap_energy_ev"]:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)
    return pd.get_dummies(df)


Y1 = df_train["formation_energy_ev_natom"].copy()
Y2 = df_train["bandgap_energy_ev"].copy()

X_train_fe = feature_engine(df_train)
X_test_fe = feature_engine(df_test)

X_train_fe, X_test_fe = X_train_fe.align(X_test_fe, join="left", axis=1, fill_value=0)

df_train = X_train_fe
df_test = X_test_fe

df_train.head()




## === cell 6
def plot_norm(feature):
    sns.distplot(df_train[feature], fit=norm)
    plt.show()
    plt.scatter(
        df_train[feature], Y1, label="formation_energy_ev_natom", alpha=0.1, c="r"
    )
    plt.legend()
    plt.title(feature)
    plt.show()
    plt.scatter(df_train[feature], Y2, label="bandgap_energy_ev", alpha=0.1, c="g")
    plt.legend()
    plt.title(feature)
    plt.show()


for col in [
    "number_of_total_atoms",
    "percent_atom_al",
    "percent_atom_ga",
    "percent_atom_in",
    "lattice_vector_1_ang",
    "lattice_vector_1_ang",
    "lattice_vector_3_ang",
]:
    pass




## === cell 7
def plot_plot(feature):
    plt.subplot(121)
    plt.scatter(df_train[feature], Y1, label="formation_energy_ev_natom")
    plt.legend()
    plt.subplot(122)
    plt.scatter(df_train[feature], Y2, label="bandgap_energy_ev")
    plt.legend()


plot_plot("percent_atom_al")



## === cell 8
x_train, x_test, y_train, y_test = train_test_split(
    df_train, Y2, test_size=0.2, random_state=7
)
rs = StandardScaler()
x_train = rs.fit_transform(x_train)
x_test = rs.transform(x_test)


def validate_data(
    model, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test
):
    """
    Bugfix: RMSLE needs non-negative predictions; clamp to 0 to avoid log1p domain issues.
    Score-impact: prevents NaNs/inf and aligns with competition metric semantics.
    """
    model.fit(x_train, y_train)
    y_pred_tr = model.predict(x_train)
    y_pred_te = model.predict(x_test)

    y_pred_tr = np.maximum(y_pred_tr, 0)
    y_pred_te = np.maximum(y_pred_te, 0)

    print("train:\n {}".format(np.sqrt(MSE(np.log1p(y_train), np.log1p(y_pred_tr)))))
    print("test :\n {}".format(np.sqrt(MSE(np.log1p(y_test), np.log1p(y_pred_te)))))


def make_sub(model1, model2):
    """
    Bugfix: apply same non-negativity constraint before writing submission (RMSLE stability).
    """
    model1.fit(df_train, Y1)
    model2.fit(df_train, Y2)

    y_pred1 = np.maximum(model1.predict(df_test), 0)
    y_pred2 = np.maximum(model2.predict(df_test), 0)

    sub["formation_energy_ev_natom"] = y_pred1
    sub["bandgap_energy_ev"] = y_pred2
    sub.to_csv("sub.csv", index=False)
    print("submit finished -> sub.csv")




## === cell 9
lr = LinearRegression()
validate_data(lr)



## === cell 10
# '''
# train:
#  0.39189838428500523
# test :
#  0.3961722538975962
#
#



## === cell 11
sns.distplot(Y2, fit=norm)
plt.show()



## === cell 12
start = time.time()

params = {
    "max_features": [0.5, 0.8],
    "min_samples_split": [3, 6],
    "min_samples_leaf": [8, 10, 13],
}
rfr = RandomForestRegressor(bootstrap=False, n_estimators=300, random_state=7)
grid = GridSearchCV(rfr, params, cv=10, scoring="neg_mean_squared_error")

print("spend time :{:.2f}s".format(time.time() - start))



## === cell 13
from lightgbm import LGBMRegressor
import lightgbm as lgb


def model_fit_lgb(model, model_params, x_train, y_train, early_stop_rounds=5):
    """
    Bugfix for lightgbm>=4.x: lgb.cv removed early_stopping_rounds kwarg.
    Use callbacks=[lgb.early_stopping(...)] and set a valid num_boost_round.
    """
    model_train = lgb.Dataset(x_train, y_train)
    print("cving...")
    cv_result = lgb.cv(
        params=model_params,
        train_set=model_train,
        nfold=50,
        stratified=False,
        shuffle=True,
        num_boost_round=5000,
        seed=0,
        metrics="rmse",
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stop_rounds, verbose=False)
        ],
    )
    print("cv finished.")
    n_estimators = len(cv_result["rmse-mean"])
    print(n_estimators)
    model.set_params(n_estimators=n_estimators)


lgb_params = {
    "learning_rate": 0.1,
    "bagging_fraction": 0.8,
    "feature_fraction": 0.8,
    "num_leaves": 50,
    "metric": "rmse",
    "bagging_freq": 10,
    "objective": "regression",
    "random_state": 7,
}

lgb1 = LGBMRegressor(**lgb_params)

model_fit_lgb(lgb1, lgb_params, x_train, y_train)
validate_data(lgb1)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3006888680.py in <cell line: 0>()
     43 lgb1 = LGBMRegressor(**lgb_params)
     44 
---> 45 model_fit_lgb(lgb1, lgb_params, x_train, y_train)
     46 validate_data(lgb1)
     47 

/tmp/ipykernel_11/3006888680.py in model_fit_lgb(model, model_params, x_train, y_train, early_stop_rounds)
     24     )
     25     print("cv finished.")
---> 26     n_estimators = len(cv_result["rmse-mean"])
     27     print(n_estimators)
     28     model.set_params(n_estimators=n_estimators)

KeyError: 'rmse-mean'

## === cell 14
params = {"max_depth": range(3, 11), "num_leaves": range(55, 65)}
grid = GridSearchCV(lgb1, params, cv=10, scoring="neg_mean_squared_error")



## === cell 15
lgb2 = LGBMRegressor(
    n_estimators=61,
    learning_rate=0.1,
    bagging_fraction=0.8,
    feature_fraction=0.8,
    max_depth=7,
    num_leaves=60,
    metric="rmse",
    bagging_freq=10,
    objective="regression",
    random_state=7,
)
validate_data(lgb2)



## === cell 16
params = {
    "min_child_samples": range(10, 50, 10),
    "min_child_weight": np.linspace(0.00005, 0.0005, 10),
}
grid = GridSearchCV(lgb2, params, cv=10, scoring="neg_mean_squared_error")



## === cell 17
lgb3 = LGBMRegressor(
    n_estimators=61,
    learning_rate=0.1,
    bagging_fraction=0.8,
    feature_fraction=0.8,
    max_depth=7,
    num_leaves=60,
    metric="rmse",
    bagging_freq=10,
    min_child_samples=20,
    min_child_weight=0.0001,
    objective="regression",
    random_state=7,
)
validate_data(lgb3)



## === cell 18
params = {
    "bagging_fraction": np.linspace(0.4, 0.5, 10),
    "feature_fraction": np.linspace(0.4, 0.5, 10),
}
grid = GridSearchCV(lgb3, params, cv=10, scoring="neg_mean_squared_error")



## === cell 19
lgb4 = LGBMRegressor(
    n_estimators=61,
    learning_rate=0.1,
    bagging_fraction=0.4,
    feature_fraction=0.4555555,
    max_depth=7,
    num_leaves=60,
    metric="rmse",
    bagging_freq=10,
    min_child_samples=20,
    min_child_weight=0.0001,
    objective="regression",
    random_state=7,
)



## === cell 20
params = {"reg_alpha": np.linspace(0.04, 0.05, 50), "reg_gamma": [0]}
grid = GridSearchCV(lgb4, params, cv=10, scoring="neg_mean_squared_error")



## === cell 21
lgb5 = LGBMRegressor(
    n_estimators=61,
    learning_rate=0.1,
    bagging_fraction=0.4,
    feature_fraction=0.4555555,
    max_depth=7,
    num_leaves=60,
    metric="rmse",
    bagging_freq=10,
    min_child_samples=20,
    min_child_weight=0.0001,
    reg_alpha=0.04204081632653062,
    reg_gamma=0,
    objective="regression",
    random_state=7,
)



## === cell 22
params = {"bagging_freq": range(10, 70, 5)}
grid = GridSearchCV(lgb4, params, cv=10, scoring="neg_mean_squared_error")



## === cell 23
lgb6 = LGBMRegressor(
    n_estimators=61,
    learning_rate=0.1,
    bagging_fraction=0.4,
    feature_fraction=0.4555555,
    max_depth=7,
    num_leaves=60,
    metric="rmse",
    bagging_freq=10,
    min_child_samples=20,
    min_child_weight=0.0001,
    reg_alpha=0.04204081632653062,
    reg_gamma=0,
    objective="regression",
    random_state=7,
)
validate_data(lgb6)



## === cell 24
params = dict(
    learning_rate=0.1,
    bagging_fraction=0.4,
    feature_fraction=0.4555555,
    max_depth=7,
    num_leaves=60,
    metric="rmse",
    bagging_freq=10,
    min_child_samples=20,
    min_child_weight=0.0001,
    reg_alpha=0.04204081632653062,
    reg_gamma=0,
    objective="regression",
    random_state=7,
)

model_fit_lgb(lgb5, params, x_train, y_train)

make_sub(
    SVR(kernel="rbf", **{"C": 80.0, "gamma": 0.00043333333333333337}),
    lgb5,
)

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1407117750.py in <cell line: 0>()
     16 )
     17 
---> 18 model_fit_lgb(lgb5, params, x_train, y_train)
     19 
     20 make_sub(

/tmp/ipykernel_11/3006888680.py in model_fit_lgb(model, model_params, x_train, y_train, early_stop_rounds)
     24     )
     25     print("cv finished.")
---> 26     n_estimators = len(cv_result["rmse-mean"])
     27     print(n_estimators)
     28     model.set_params(n_estimators=n_estimators)

KeyError: 'rmse-mean'
