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

20.56755

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np

train_data = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/train.csv")
test_data = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/test.csv")
import seaborn as sns

sns.distplot(train_data["Pawpularity"])


## === cell 1
value = train_data["Pawpularity"].quantile(0.9)
value2 = train_data["Pawpularity"].quantile(0.2)
print(value, value2)


## === cell 2
train_data = train_data[train_data["Pawpularity"] < value]
train_data = train_data[train_data["Pawpularity"] > value2]
sns.distplot(train_data["Pawpularity"])


## === cell 3
import scipy.stats as stats
import matplotlib.pyplot as plt

stats.probplot(train_data["Pawpularity"], dist="norm", plot=plt)
plt.show()


## === cell 4
x_train = np.array(train_data.iloc[:, 1:13])
x_test = np.array(test_data.iloc[:, 1:])
y_train = np.array(train_data.iloc[:, 13])


## === cell 5
from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
x_train = stdsc.fit_transform(x_train)
x_test = stdsc.transform(x_test)


## === cell 6
from sklearn.model_selection import train_test_split

x_train2, x_valid2, y_train2, y_valid2 = train_test_split(
    x_train, y_train, test_size=0.1, random_state=123
)


## === cell 7
import xgboost as xgb

xgbm2 = xgb.XGBRegressor(
    base_score=0.5,
    booster="gbtree",
    colsample_bylevel=1,
    colsample_bynode=1,
    colsample_bytree=1,
    gamma=0,
    gpu_id=-1,
    importance_type="gain",
    interaction_constraints="",
    learning_rate=0.3,
    max_delta_step=0,
    max_depth=4,
    min_child_weight=1,
    monotone_constraints="()",
    n_estimators=500,
    n_jobs=2,
    num_parallel_tree=1,
    random_state=0,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    subsample=1,
    tree_method="exact",
    validate_parameters=1,
    verbosity=0,
)
model1 = xgbm2
model1.fit(x_train, y_train)
a1 = model1.predict(x_valid2)
p1 = model1.predict(x_test)
from sklearn.metrics import mean_squared_error

w1 = np.sqrt(mean_squared_error(a1, y_valid2))
print("XGB RMSE:", w1)


## === cell 8
import seaborn as sns

sns.scatterplot(x=a1, y=y_valid2)


## === cell 9
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(
    random_state=123,
    max_depth=12,
    min_samples_leaf=5,
    n_estimators=400,
    min_samples_split=2,
)
model2 = forest
model2.fit(x_train, y_train)
a2 = model2.predict(x_valid2)
p2 = model2.predict(x_test)
w2 = np.sqrt(mean_squared_error(a2, y_valid2))
print("RF RMSE:", w2)


## === cell 10
sns.scatterplot(x=a2, y=y_valid2)


## === cell 11
import lightgbm as lgb

gbm = lgb.LGBMRegressor(
    learning_rate=0.01,
    max_depth=10,
    min_child_samples=20,
    min_child_weight=0.001,
    num_leaves=31,
    reg_alpha=0,
    reg_lambda=1,
    random_state=0,
    n_estimators=600,
)
model3 = gbm
model3.fit(x_train, y_train, verbose=0)
a3 = model3.predict(x_valid2)
p3 = model3.predict(x_test)
w3 = np.sqrt(mean_squared_error(a3, y_valid2))
print("LGBM RMSE:", w3)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/809654106.py in <cell line: 0>()
     13 )
     14 model3 = gbm
---> 15 model3.fit(x_train, y_train, verbose=0)
     16 a3 = model3.predict(x_valid2)
     17 p3 = model3.predict(x_test)

TypeError: LGBMRegressor.fit() got an unexpected keyword argument 'verbose'

## === cell 12
sns.scatterplot(x=a3, y=y_valid2)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/534548187.py in <cell line: 0>()
----> 1 sns.scatterplot(x=a3, y=y_valid2)

NameError: name 'a3' is not defined

## === cell 13
from sklearn.neural_network import MLPRegressor

deep_learning = MLPRegressor(
    hidden_layer_sizes=(100, 100, 100, 100),
    random_state=123,
    verbose=False,
    activation="relu",
    early_stopping=True,
    max_iter=500,
    solver="adam",
    warm_start=True,
)
model4 = deep_learning
model4.fit(x_train, y_train)
a4 = model4.predict(x_valid2)
p4 = model4.predict(x_test)
w4 = np.sqrt(mean_squared_error(a4, y_valid2))
print("MLP RMSE:", w4)


## === cell 14
sns.scatterplot(x=a4, y=y_valid2)


## === cell 15
from sklearn import svm

svr = svm.SVR(C=1, epsilon=0.1, verbose=False, degree=3)
model7 = svr
model7.fit(x_train, y_train)
a7 = model7.predict(x_valid2)
p7 = model7.predict(x_test)
w7 = np.sqrt(mean_squared_error(a7, y_valid2))
print("SVR RMSE:", w7)


## === cell 16
sns.scatterplot(x=a7, y=y_valid2)


## === cell 17
pred_df = pd.DataFrame(
    {
        "Id": test_data["Id"],
        "XGBoost": p1,
        "RandomForest": p2,
        "LightGBM": p3,
        "SVR": p7,
        "MLP": p4,
    }
)
final_pred = pred_df.iloc[:, 1:].mean(axis=1)
submission = pd.DataFrame({"Id": test_data["Id"], "Pawpularity": final_pred})


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1126905451.py in <cell line: 0>()
      5         "XGBoost": p1,
      6         "RandomForest": p2,
----> 7         "LightGBM": p3,
      8         "SVR": p7,
      9         "MLP": p4,

NameError: name 'p3' is not defined

## === cell 18
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"File saved to {submission_path}")

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2501545878.py in <cell line: 0>()
      1 submission_path = "/kaggle/working/submission.csv"
----> 2 submission.to_csv(submission_path, index=False)
      3 print(f"File saved to {submission_path}")

NameError: name 'submission' is not defined
