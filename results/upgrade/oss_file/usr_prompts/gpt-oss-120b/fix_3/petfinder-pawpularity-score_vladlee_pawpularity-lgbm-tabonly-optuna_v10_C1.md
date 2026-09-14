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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
tqdm==4.67.1

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

20.48745

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from IPython import display as ipd
import gc, time, logging, re, math, os, warnings, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
def seeding(SEED, use_tf=False):
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TF_CUDNN_DETERMINISTIC"] = str(SEED)
    if use_tf:
        import tensorflow as tf

        tf.random.set_seed(SEED)
    print("seeding done!!!")




## === cell 2
RANDOM_SEED = 42
DATA_PATH = "/kaggle/input/petfinder-pawpularity-score/"

train_df = pd.read_csv(DATA_PATH + "train.csv")
test_df = pd.read_csv(DATA_PATH + "test.csv")
submission_template = pd.read_csv(DATA_PATH + "sample_submission.csv")

seeding(RANDOM_SEED)
display(train_df.head(2))



## === cell 3
FEATURES = [
    "Subject Focus",
    "Eyes",
    "Face",
    "Near",
    "Action",
    "Accessory",
    "Group",
    "Collage",
    "Human",
    "Occlusion",
    "Info",
    "Blur",
]
target = train_df["Pawpularity"]
train = train_df[FEATURES].copy()
test = test_df[FEATURES].copy()
print("train shape:", train.shape)
print("test shape:", test.shape)
print("target shape:", target.shape)



## === cell 4
CAT_FEATURES = FEATURES.copy()
ALL_CAT_FEATURES = FEATURES.copy()


def add_cross_features(df):
    for f1 in CAT_FEATURES:
        for f2 in CAT_FEATURES:
            if f1 != f2:
                name2 = f"{f1}-{f2}"
                df[name2] = df[f1].astype(str) + "_" + df[f2].astype(str)
                ALL_CAT_FEATURES.append(name2)
                for f3 in CAT_FEATURES:
                    if f3 not in (f1, f2):
                        name3 = f"{f1}-{f2}-{f3}"
                        df[name3] = (
                            df[f1].astype(str)
                            + "_"
                            + df[f2].astype(str)
                            + "_"
                            + df[f3].astype(str)
                        )
                        ALL_CAT_FEATURES.append(name3)
    return df


train = add_cross_features(train)
test = add_cross_features(test)
print("train shape after cross features:", train.shape)
print("test shape after cross features:", test.shape)



## === cell 5
for col in train.columns:
    train[col] = train[col].astype("category").cat.codes
    test[col] = test[col].astype("category").cat.codes



## === cell 6
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

pca_tmp = PCA(n_components=0.999, svd_solver="full")
pca_tmp.fit(train_scaled)
n_components = pca_tmp.n_components_
print("Selected PCA components:", n_components)

pca = PCA(n_components=n_components)
pca_train = pca.fit_transform(train_scaled)
pca_test = pca.transform(test_scaled)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")
plt.title("PCA explained variance")
plt.show()



## === cell 7
X = pd.DataFrame(
    pca_train, columns=[f"PCA{i}" for i in range(n_components)], index=train.index
)




## === cell 8
def run_train(X, y, run_params, splits, num_boost_round, early_stopping_rounds):
    """
    Train LightGBM models with KFold using the modern callback API.
    """
    models = []
    oof_predicted = []
    evals_results = {}
    folds = KFold(n_splits=splits, shuffle=True, random_state=RANDOM_SEED)
    for fold_n, (train_idx, valid_idx) in enumerate(folds.split(X)):
        print(f"Fold {fold_n+1} started")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),  # silence per‑iteration output
        ]

        model = lgb.train(
            run_params,
            train_set=train_data,
            num_boost_round=num_boost_round,
            valid_sets=[valid_data],
            valid_names=["valid"],
            callbacks=callbacks,
            evals_result=evals_results,
        )
        oof_predicted.append(model.predict(X_valid, num_iteration=model.best_iteration))
        models.append(model)
    return models, oof_predicted, evals_results




## === cell 9
LEARNING_RATE = 0.001
MAX_DEPTH = -1
NUM_LEAVES = 31
TOTAL_SPLITS = 4
NUM_BOOST_ROUND = 400
EARLY_STOPPING_ROUNDS = 50

run_params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": LEARNING_RATE,
    "num_leaves": NUM_LEAVES,
    "max_depth": MAX_DEPTH,
    "verbose": -1,
}

models, oof_predicted, evals_results = run_train(
    X, target, run_params, TOTAL_SPLITS, NUM_BOOST_ROUND, EARLY_STOPPING_ROUNDS
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3418820533.py in <cell line: 0>()
     16 }
     17 
---> 18 models, oof_predicted, evals_results = run_train(
     19     X, target, run_params, TOTAL_SPLITS, NUM_BOOST_ROUND, EARLY_STOPPING_ROUNDS
     20 )

/tmp/ipykernel_11/3903644330.py in run_train(X, y, run_params, splits, num_boost_round, early_stopping_rounds)
     20         ]
     21 
---> 22         model = lgb.train(
     23             run_params,
     24             train_set=train_data,

TypeError: train() got an unexpected keyword argument 'evals_result'

## === cell 10
predicted = [
    model.predict(pca_test, num_iteration=model.best_iteration) for model in models
]
avg_preds = np.mean(predicted, axis=0)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1044634396.py in <cell line: 0>()
      1 predicted = [
----> 2     model.predict(pca_test, num_iteration=model.best_iteration) for model in models
      3 ]
      4 avg_preds = np.mean(predicted, axis=0)
      5 

NameError: name 'models' is not defined

## === cell 11
submission = pd.DataFrame({"Id": submission_template["Id"], "Pawpularity": avg_preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False, float_format="%.6f")
print(f"Saved submission to {submission_path}")
submission.head(20)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4227041621.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"Id": submission_template["Id"], "Pawpularity": avg_preds})
      2 submission_path = "submission.csv"
      3 submission.to_csv(submission_path, index=False, float_format="%.6f")
      4 print(f"Saved submission to {submission_path}")
      5 submission.head(20)

NameError: name 'avg_preds' is not defined
