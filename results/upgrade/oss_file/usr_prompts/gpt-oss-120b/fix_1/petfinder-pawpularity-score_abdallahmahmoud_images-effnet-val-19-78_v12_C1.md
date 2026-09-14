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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.10

# 3. Installed packages

catboost==1.2.8
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
pillow==11.3.0
plotly==5.24.1
plotly-express==0.4.1
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

20.4571

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

sns.set_palette(sns.color_palette("Set2"))

import sklearn
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as tfl

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, DaskLGBMRegressor

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans

from PIL import Image
import os

np.random.seed(0)
tf.random.set_seed(0)


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_csv('/kaggle/input/petfinder-pawpularity-score/train.csv')
train['path'] = '/kaggle/input/petfinder-pawpularity-score/train/' + train['Id'] + '.jpg'
train.head(3)


## === cell 2
test = pd.read_csv('/kaggle/input/petfinder-pawpularity-score/test.csv')
test['path'] = '/kaggle/input/petfinder-pawpularity-score/test/' + test['Id'] + '.jpg'


## === cell 3
print(f'shape: {train.shape}')
print(train.info())


## === cell 4
train.describe()


## === cell 5
plt.figure(figsize=(10,4))

sns.histplot(data=train, x='Pawpularity');


## === cell 6
plt.figure(figsize=(20,10))
plt.tight_layout()

for i in range(12):
    plt.subplot(3, 4, i+1)
    sns.countplot(data=train, x=train.columns[1:13][i])


## === cell 7
def size_and_shape(row):
    img = Image.open(row['path'])
    return pd.Series([img.size[0], img.size[1], os.path.getsize(row['path'])])


## === cell 8
scale = MinMaxScaler()

train[['width', 'height', 'size']] = pd.DataFrame(scale.fit_transform(train.apply(size_and_shape, axis=1).values))
test[['width', 'height', 'size']] = pd.DataFrame(scale.fit_transform(test.apply(size_and_shape, axis=1).values))


## === cell 9
X = train.drop(['Id', 'Pawpularity', 'path'], axis=1)
y = train['Pawpularity']


## === cell 10
k = KMeans(8, random_state=0)

k.fit(X)

X['cluster'] = k.predict(X)
test['cluster'] = k.predict(test.drop(['Id', 'path'], axis=1))


## === cell 11
p = PCA()

p.fit(X)

X = X.join(pd.DataFrame(p.transform(X)))
test = test.join(pd.DataFrame(p.transform(test.drop(['Id', 'path'], axis=1))))


## === cell 12
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=86, test_size=0.1)


## === cell 13
def eval_model(model):
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)

    return mean_squared_error(y_test, preds, squared=False)


## === cell 14
xgb = XGBRegressor(seed=0,
                   learning_rate =0.1,
                     n_estimators=70,
                     max_depth=2,
                     min_child_weight=2,
                     gamma=0,
                     subsample=0.9,
                     colsample_bytree=0.2)

light = LGBMRegressor(random_state=0,
                      num_leaves=4,
                      subsample_for_bin=20, 
                      min_split_gain=0.1,
                      min_child_samples=25,
                      reg_lambda=0.4)

cat = CatBoostRegressor(random_seed=0, 
                          verbose=0, 
                          num_trees=11, 
                          learning_rate=0.26,
                          l2_leaf_reg=2.5, 
                          random_strength=0.9)

extra = ExtraTreesRegressor(random_state=0, 
                            max_depth=4, 
                            min_samples_leaf=2, 
                            max_features=6, 
                            max_leaf_nodes=20)

rf = RandomForestRegressor(random_state=0, 
                              n_estimators=10,
                              max_depth=5, 
                              max_features=9)

gb = GradientBoostingRegressor(random_state=0, 
                               n_estimators=40, 
                               subsample=0.44)


models = {'rf': rf, 'gb': gb, 'extra': extra, 'cat': cat, 'light': light, 'xgb': xgb}

for model in models:
    print(model, 'RMSE:', eval_model(models[model]))


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/45692343.py in <cell line: 0>()
     41 
     42 for model in models:
---> 43     print(model, 'RMSE:', eval_model(models[model]))

/tmp/ipykernel_11/440510418.py in eval_model(model)
      1 def eval_model(model):
----> 2     model.fit(X_train, y_train)
      3 
      4     preds = model.predict(X_test)
      5 

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_forest.py in fit(self, X, y, sample_weight)
    343         if issparse(y):
    344             raise ValueError("sparse multilabel-indicator for y is not supported.")
--> 345         X, y = self._validate_data(
    346             X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE
    347         )

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    546             validated.
    547         """
--> 548         self._check_feature_names(X, reset=reset)
    549 
    550         if y is None and self._get_tags()["requires_y"]:

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _check_feature_names(self, X, reset)
    413 
    414         if reset:
--> 415             feature_names_in = _get_feature_names(X)
    416             if feature_names_in is not None:
    417                 self.feature_names_in_ = feature_names_in

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _get_feature_names(X)
   1901     # mixed type of string and non-string is not supported
   1902     if len(types) > 1 and "str" in types:
-> 1903         raise TypeError(
   1904             "Feature names are only supported if all input features have string names, "
   1905             f"but your input has {types} as feature name / column name types. "

TypeError: Feature names are only supported if all input features have string names, but your input has ['int', 'str'] as feature name / column name types. If you want feature names to be stored and validated, you must convert them all to strings, by using X.columns = X.columns.astype(str) for example. Otherwise you can remove feature / column names from your input data, or convert them all to a non-string data type.

## === cell 15
models = {'cat': cat, 'light': light, 'gb': gb}

models_vote = [(model, models[model]) for model in models]

vote = VotingRegressor(estimators=models_vote, weights=[0.5, 0.3, 0.2])

vote.fit(X_train, y_train)

eval_model(vote)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3377292675.py in <cell line: 0>()
      5 vote = VotingRegressor(estimators=models_vote, weights=[0.5, 0.3, 0.2])
      6 
----> 7 vote.fit(X_train, y_train)
      8 
      9 eval_model(vote)

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_voting.py in fit(self, X, y, sample_weight)
    596         self._validate_params()
    597         y = column_or_1d(y, warn=True)
--> 598         return super().fit(X, y, sample_weight)
    599 
    600     def predict(self, X):

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_voting.py in fit(self, X, y, sample_weight)
     79             )
     80 
---> 81         self.estimators_ = Parallel(n_jobs=self.n_jobs)(
     82             delayed(_fit_single_estimator)(
     83                 clone(clf),

/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py in __call__(self, iterable)
     61             for delayed_func, args, kwargs in iterable
     62         )
---> 63         return super().__call__(iterable_with_config)
     64 
     65 

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in __call__(self, iterable)
   1984             output = self._get_sequential_output(iterable)
   1985             next(output)
-> 1986             return output if self.return_generator else list(output)
   1987 
   1988         # Let's create an ID that uniquely identifies the current call. If the

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _get_sequential_output(self, iterable)
   1912                 self.n_dispatched_batches += 1
   1913                 self.n_dispatched_tasks += 1
-> 1914                 res = func(*args, **kwargs)
   1915                 self.n_completed_tasks += 1
   1916                 self.print_progress()

/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py in __call__(self, *args, **kwargs)
    121             config = {}
    122         with config_context(**config):
--> 123             return self.function(*args, **kwargs)

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_base.py in _fit_single_estimator(estimator, X, y, sample_weight, message_clsname, message)
     44     else:
     45         with _print_elapsed_time(message_clsname, message):
---> 46             estimator.fit(X, y)
     47     return estimator
     48 

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in fit(self, X, y, sample_weight, monitor)
    427         # trees use different types for X and y, checking them separately.
    428 
--> 429         X, y = self._validate_data(
    430             X, y, accept_sparse=["csr", "csc", "coo"], dtype=DTYPE, multi_output=True
    431         )

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    546             validated.
    547         """
--> 548         self._check_feature_names(X, reset=reset)
    549 
    550         if y is None and self._get_tags()["requires_y"]:

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _check_feature_names(self, X, reset)
    413 
    414         if reset:
--> 415             feature_names_in = _get_feature_names(X)
    416             if feature_names_in is not None:
    417                 self.feature_names_in_ = feature_names_in

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _get_feature_names(X)
   1901     # mixed type of string and non-string is not supported
   1902     if len(types) > 1 and "str" in types:
-> 1903         raise TypeError(
   1904             "Feature names are only supported if all input features have string names, "
   1905             f"but your input has {types} as feature name / column name types. "

TypeError: Feature names are only supported if all input features have string names, but your input has ['int', 'str'] as feature name / column name types. If you want feature names to be stored and validated, you must convert them all to strings, by using X.columns = X.columns.astype(str) for example. Otherwise you can remove feature / column names from your input data, or convert them all to a non-string data type.

## === cell 16
test['Pawpularity'] = vote.predict(test.drop(['Id', 'path'], axis=1))
test[['Id', 'Pawpularity']].to_csv('submission.csv', index=False)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/2454847720.py in <cell line: 0>()
----> 1 test['Pawpularity'] = vote.predict(test.drop(['Id', 'path'], axis=1))
      2 test[['Id', 'Pawpularity']].to_csv('submission.csv', index=False)

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_voting.py in predict(self, X)
    614             The predicted values.
    615         """
--> 616         check_is_fitted(self)
    617         return np.average(self._predict(X), axis=1, weights=self._weights_not_none)
    618 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This VotingRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
