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

3.10

# 2. Installed packages

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0
xgboost==2.0.3

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        input/
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        working/
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
```

-> data/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/tabular-playground-series-dec-2021/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> data/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> input/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.stats import mode

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold


## === cell 1
train = pd.read_csv('../input/tabular-playground-series-dec-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-dec-2021/test.csv')


## === cell 2
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)


## === cell 3
train.head()


## === cell 4
train.describe()


## === cell 5
print("Columns: \n{0}".format(list(train.columns)))


## === cell 6
print('Train data shape:', train.shape)
print('Test data shape:', test.shape)


## === cell 7
missing_values_train = train.isna().any().sum()
print('Missing values in train data: {0}'.format(missing_values_train[missing_values_train > 0]))

missing_values_test = test.isna().any().sum()
print('Missing values in test data: {0}'.format(missing_values_test[missing_values_test > 0]))


## === cell 8
duplicates_train = train.duplicated().sum()
print('Duplicates in train data: {0}'.format(duplicates_train))

duplicates_test = test.duplicated().sum()
print('Duplicates in test data: {0}'.format(duplicates_test))


## === cell 9
categorical_features = train.columns[11:-1:]
print("Categorical Columns: \n{0}".format(list(categorical_features)))


## === cell 10
numerical_features = train.columns[1:11]
print("Numerical Columns: \n{0}".format(list(train.columns[1:11])))
train[numerical_features].describe()


## === cell 11
plt.figure(figsize=(10, 6))
plt.title('Target distribution')
sns.countplot(x=train['Cover_Type'], data=train)


## === cell 12
cType5 = train[train['Cover_Type'] == 5].index
print("Number of rows with Cover_Type = 5: {0}".format(len(cType5)))


## === cell 13
print("Unique values in Soil_Type7 column train data: {0}".format(train['Soil_Type7'].unique()))
print("Unique values in Soil_Type15 column train data: {0}".format(train['Soil_Type15'].unique()))

print("Unique values in Soil_Type7 column test data: {0}".format(test['Soil_Type7'].unique()))
print("Unique values in Soil_Type15 column test data: {0}".format(test['Soil_Type15'].unique()))


## === cell 14
train.drop(cType5, axis=0, inplace=True)

train.drop(['Soil_Type7', 'Soil_Type15'], axis=1, inplace=True)
test.drop(['Soil_Type7', 'Soil_Type15'], axis=1, inplace=True)


## === cell 15
X = train.iloc[:, 1:-1].copy()
y = train.Cover_Type.copy()



## === cell 16
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

def run_model(model):
    model.fit(train_X, train_y, eval_set=[(val_X, val_y)],
              early_stopping_rounds=40, eval_metric='mlogloss',
              verbose=False)
    predictions = model.predict(val_X)
    score = accuracy_score(val_y, predictions)
    return score, "Accuracy score:  {:.6f}".format(score)

def evaluate_model(model):
    print("Accuracy score:", accuracy_score(train_y, model.predict(train_X)))


## === cell 17
def run_xgboost_model(c, max_score):
    try:
        value = run_model(XGBClassifier(seed=1, tree_method='gpu_hist', predictor='gpu_predictor',
                                        learning_rate=float(c[0]), gamma=float(c[1]), max_depth=int(c[2]),
                                        reg_alpha=float(c[3]), reg_lambda=float(c[4]), n_estimators=int(c[5])))
        if value[0] > max_score[0]:
            max_score[0] = value[0]
            max_score[1] = c
        print("Combination: learning_rate: {0}, gamma: {1}, max_depth: {2}, reg_alpha: {3}, reg_lambda: {4}, n_estimators: {5}, {6}".format(c[0], c[1], c[2], c[3], c[4], c[5], value[1]))
    except:
        print("Invalid combination: learning_rate: {0}, gamma: {1}, max_depth: {2}, reg_alpha: {3}, reg_lambda: {4}, n_estimators: {5}.".format(c[0], c[1], c[2], c[3], c[4], c[5]))
        pass


## === cell 18
learning_rate = [0.5]
gamma         = [1.0]
max_depth     = [8]

reg_alpha     = [0]
reg_lambda    = [0, 0.1, 0.2]
n_estimators  = [100]


## === cell 19
max_score = [0, []]
combinations = np.array(
    np.meshgrid(learning_rate, gamma, max_depth, reg_alpha, reg_lambda, n_estimators)
).T.reshape(-1, 6)
for combination in combinations:
    run_xgboost_model(combination, max_score)

if isinstance(max_score[1], (list, tuple)) and len(max_score[1]) == 6:
    print(
        "Best score: learning_rate: {0}, gamma: {1}, max_depth: {2}, reg_alpha: {3}, reg_lambda: {4}, n_estimators: {5}, Accuracy score: {6:.6f}.".format(
            max_score[1][0],
            max_score[1][1],
            max_score[1][2],
            max_score[1][3],
            max_score[1][4],
            max_score[1][5],
            max_score[0],
        )
    )
else:
    print(
        "Best score: unavailable (no valid hyperparameter combination completed successfully)."
    )


## === cell 20
test_X = test.iloc[:, 1:]
model = XGBClassifier(
    seed=1,
    tree_method="gpu_hist",
    predictor="gpu_predictor",
    eval_metric="mlogloss",
    learning_rate=0.3,
    gamma=1.6,
    max_depth=10,
)

le = LabelEncoder()
y_enc = le.fit_transform(y)

fold = 1
accuracy_scores = []
test_predictions = []
skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
for train_idx, test_idx in skf.split(X, y_enc):
    train_X, val_X = X.iloc[train_idx], X.iloc[test_idx]
    train_y, val_y = y_enc[train_idx], y_enc[test_idx]
    print(len(train_X.columns))
    print(len(val_X.columns))

    model.fit(
        train_X,
        train_y,
        early_stopping_rounds=40,
        eval_set=[(val_X, val_y)],
        verbose=False,
    )
    predictions = model.predict(val_X)
    score = accuracy_score(val_y, predictions)
    print("Fold: {0}  \t\t Accuracy score:  {1:.6f}".format(fold, score))
    accuracy_scores.append(score)

    test_predictions.append(le.inverse_transform(model.predict(test_X).astype(int)))
    fold += 1

test_predictions = np.squeeze(mode(np.column_stack(test_predictions), axis=1)[0])
print("Mean accuracy score: {0:.6f}".format(np.mean(accuracy_scores)))


## --- ERROR in cell 20, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mXGBoostError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4133605713.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     26[0m     [0mprint[0m[0;34m([0m[0mlen[0m[0;34m([0m[0mval_X[0m[0;34m.[0m[0mcolumns[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     27[0m [0;34m[0m[0m
[0;32m---> 28[0;31m     model.fit(
[0m[1;32m     29[0m         [0mtrain_X[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     30[0m         [0mtrain_y[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36minner_f[0;34m(*args, **kwargs)[0m
[1;32m    728[0m             [0;32mfor[0m [0mk[0m[0;34m,[0m [0marg[0m [0;32min[0m [0mzip[0m[0;34m([0m[0msig[0m[0;34m.[0m[0mparameters[0m[0;34m,[0m [0margs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    729[0m                 [0mkwargs[0m[0;34m[[0m[0mk[0m[0;34m][0m [0;34m=[0m [0marg[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 730[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    731[0m [0;34m[0m[0m
[1;32m    732[0m         [0;32mreturn[0m [0minner_f[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py[0m in [0;36mfit[0;34m(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)[0m
[1;32m   1517[0m             )
[1;32m   1518[0m [0;34m[0m[0m
[0;32m-> 1519[0;31m             self._Booster = train(
[0m[1;32m   1520[0m                 [0mparams[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1521[0m                 [0mtrain_dmatrix[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36minner_f[0;34m(*args, **kwargs)[0m
[1;32m    728[0m             [0;32mfor[0m [0mk[0m[0;34m,[0m [0marg[0m [0;32min[0m [0mzip[0m[0;34m([0m[0msig[0m[0;34m.[0m[0mparameters[0m[0;34m,[0m [0margs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    729[0m                 [0mkwargs[0m[0;34m[[0m[0mk[0m[0;34m][0m [0;34m=[0m [0marg[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 730[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    731[0m [0;34m[0m[0m
[1;32m    732[0m         [0;32mreturn[0m [0minner_f[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/training.py[0m in [0;36mtrain[0;34m(params, dtrain, num_boost_round, evals, obj, feval, maximize, early_stopping_rounds, evals_result, verbose_eval, xgb_model, callbacks, custom_metric)[0m
[1;32m    179[0m         [0;32mif[0m [0mcb_container[0m[0;34m.[0m[0mbefore_iteration[0m[0;34m([0m[0mbst[0m[0;34m,[0m [0mi[0m[0;34m,[0m [0mdtrain[0m[0;34m,[0m [0mevals[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    180[0m             [0;32mbreak[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 181[0;31m         [0mbst[0m[0;34m.[0m[0mupdate[0m[0;34m([0m[0mdtrain[0m[0;34m,[0m [0mi[0m[0;34m,[0m [0mobj[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    182[0m         [0;32mif[0m [0mcb_container[0m[0;34m.[0m[0mafter_iteration[0m[0;34m([0m[0mbst[0m[0;34m,[0m [0mi[0m[0;34m,[0m [0mdtrain[0m[0;34m,[0m [0mevals[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    183[0m             [0;32mbreak[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36mupdate[0;34m(self, dtrain, iteration, fobj)[0m
[1;32m   2048[0m [0;34m[0m[0m
[1;32m   2049[0m         [0;32mif[0m [0mfobj[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2050[0;31m             _check_call(
[0m[1;32m   2051[0m                 _LIB.XGBoosterUpdateOneIter(
[1;32m   2052[0m                     [0mself[0m[0;34m.[0m[0mhandle[0m[0;34m,[0m [0mctypes[0m[0;34m.[0m[0mc_int[0m[0;34m([0m[0miteration[0m[0;34m)[0m[0;34m,[0m [0mdtrain[0m[0;34m.[0m[0mhandle[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36m_check_call[0;34m(ret)[0m
[1;32m    280[0m     """
[1;32m    281[0m     [0;32mif[0m [0mret[0m [0;34m!=[0m [0;36m0[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 282[0;31m         [0;32mraise[0m [0mXGBoostError[0m[0;34m([0m[0mpy_str[0m[0;34m([0m[0m_LIB[0m[0;34m.[0m[0mXGBGetLastError[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    283[0m [0;34m[0m[0m
[1;32m    284[0m [0;34m[0m[0m

[0;31mXGBoostError[0m: [07:28:49] /workspace/src/tree/updater_gpu_hist.cu:781: Exception in gpu_hist: [07:28:49] /workspace/src/tree/updater_gpu_hist.cu:787: Check failed: ctx_->gpu_id >= 0 (-1 vs. 0) : Must have at least one device
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7fdd59bcbf2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb3e95a) [0x7fdd59be295a]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb483cd) [0x7fdd59bec3cd]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7fdd59504c79]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x461d09) [0x7fdd59505d09]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7fdd595694f7]
  [bt] (6) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7fdd59205ef0]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7fddcc3f9e2e]
  [bt] (8) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7fddcc3f6493]



Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7fdd59bcbf2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb485c9) [0x7fdd59bec5c9]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7fdd59504c79]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x461d09) [0x7fdd59505d09]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7fdd595694f7]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7fdd59205ef0]
  [bt] (6) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7fddcc3f9e2e]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7fddcc3f6493]
  [bt] (8) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7fddcc4094d8]



## === cell 21
output = pd.DataFrame({'Id': test.Id, 'Cover_Type': test_predictions})
output.to_csv('submission.csv', index=False)
output
