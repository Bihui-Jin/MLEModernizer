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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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

20.479513072030347

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
from pathlib import Path

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

## === cell 1
ls ../input/petfinder-pawpularity-score/

## === cell 2
train = pd.read_csv('../input/petfinder-pawpularity-score/train.csv')
test = pd.read_csv('../input/petfinder-pawpularity-score/test.csv')

INPUT = Path('../input/petfinder-pawpularity-score/')
TRAIN_IMG_DIR = INPUT / 'train'            
TEST_IMG_DIR = INPUT /'test'

train.shape, test.shape

## === cell 3
train.head()

## === cell 4
test.head()

## === cell 5
train['img_path'] = train['Id'].apply(lambda x: f'../input/petfinder-pawpularity-score/train/{str(x)}.jpg')
test['img_path'] = test['Id'].apply(lambda x: f'../input/petfinder-pawpularity-score/test/{str(x)}.jpg')
target_col = 'Pawpularity'
metadata_cols = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
                 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']
train[metadata_cols + [target_col]].mean()

## === cell 6
test[metadata_cols].mean()

## === cell 8
plt.hist(train[target_col], bins=50);

## === cell 10
def create_shape_feature(df):
    width_height_list = []
    file_size_list = []
    for path_ in tqdm(df['img_path']):
        width_height_list.append(Image.open(path_).size)
        file_size_list.append(os.path.getsize(path_))
    df['width_height'] = width_height_list
    df['file_size'] = file_size_list
    df['width'] = df['width_height'].apply(lambda x: x[0])
    df['height'] = df['width_height'].apply(lambda x: x[1])
    return df

## === cell 11
train = create_shape_feature(train)
test = create_shape_feature(test)

## === cell 12
train['width_height'].value_counts()[:20]

## === cell 13
test['width_height'].value_counts()

## === cell 15
im = Image.open(test['img_path'].values[0])
plt.imshow(im);

## === cell 17
!python -m pip install --no-index --find-links=../input/ipyplot ipyplot
import ipyplot

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3212520643.py in <cell line: 0>()
      1 # !pip install ipyplot
      2 get_ipython().system('python -m pip install --no-index --find-links=../input/ipyplot ipyplot')
----> 3 import ipyplot

ModuleNotFoundError: No module named 'ipyplot'

## === cell 19
image_paths = []
labels = []
custom_texts = []

for col in metadata_cols:
    tmp_df = train[train[col] == 1]
    for i in range(4):
        image_paths.append(tmp_df.iloc[i, :]['img_path'])
        labels.append(col)
        target = str(tmp_df.iloc[i, :][target_col])
        meta = tmp_df.iloc[i, :][metadata_cols + ['width', 'height']].values
        meta = ''.join([f'{col}:{m}, ' for m, col in zip(meta, metadata_cols + ['width', 'height'])])
        custom_texts.append(f'target: {target}\n{meta}')

## === cell 20
ipyplot.plot_class_tabs(image_paths, labels, custom_texts=custom_texts, force_b64=True, img_width=350)

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1479743405.py in <cell line: 0>()
----> 1 ipyplot.plot_class_tabs(image_paths, labels, custom_texts=custom_texts, force_b64=True, img_width=350)

NameError: name 'ipyplot' is not defined

## === cell 22
ipyplot.plot_images(test['img_path'].values, force_b64=True, img_width=100)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1936481289.py in <cell line: 0>()
----> 1 ipyplot.plot_images(test['img_path'].values, force_b64=True, img_width=100)

NameError: name 'ipyplot' is not defined

## === cell 24
train['area'] = train['width'] * train['height']
train['size_per_ pixel'] = train['file_size'] / train['area']

test['area'] = test['width'] * test['height']
test['size_per_ pixel'] = test['file_size'] / test['area']

## === cell 25
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


def calc_model_importance(model, feature_names=None, importance_type='gain'):
    importance_df = pd.DataFrame(model.feature_importance(importance_type=importance_type),
                                 index=feature_names,
                                 columns=['importance']).sort_values('importance')
    return importance_df


def calc_mean_importance(importance_df_list):
    mean_importance = np.mean(
        np.array([df['importance'].values for df in importance_df_list]), axis=0)
    mean_df = importance_df_list[0].copy()
    mean_df['importance'] = mean_importance
    return mean_df


def plot_importance(importance_df, title='',
                    save_filepath=None, figsize=(4, 6)):
    importance_df = importance_df.iloc[-50:, :]
    fig, ax = plt.subplots(figsize=figsize)
    importance_df.plot.barh(ax=ax)
    if title:
        plt.title(title)
    plt.tight_layout()
    if save_filepath is None:
        plt.show()
    else:
        plt.savefig(save_filepath)
    plt.close()


def do_train(all_feature, params):

    models = []
    scores = []

    gain_importance_list = []
    split_importance_list = []

    y = all_feature['Pawpularity'].values
    X = all_feature.drop(['Id', 'img_path', 'width_height', 'Pawpularity'], axis=1)
    print(f'features: {X.columns.values}')
    print(f'num features: {len(X.columns)}')

    oof = np.zeros(len(X))
    
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X)):

        print(f"Fold :{fold+1}")

        X_train, y_train = X.iloc[trn_idx], y[trn_idx]
        X_valid, y_valid = X.iloc[val_idx], y[val_idx]

        weights = None
        lgbm_train = lgb.Dataset(X_train, y_train, weight=weights)
        lgbm_valid = lgb.Dataset(X_valid, y_valid, reference=lgbm_train, weight=weights)

        model = lgb.train(params=params,
                          train_set=lgbm_train,
                          valid_sets=[lgbm_train, lgbm_valid],
                          num_boost_round=5000,
                          verbose_eval=100,
                          categorical_feature=metadata_cols,
                          early_stopping_rounds=30
                          )

        y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
        oof[val_idx] = y_pred

        score = round(np.sqrt(mean_squared_error(y_true=y_valid, y_pred=y_pred)), 3)
        print(f'RMSE: {score}')

        scores.append(score)
        models.append(model)
        print("*" * 5)

        feature_names = X_train.columns.values.tolist()
        gain_importance_df = calc_model_importance(
            model, feature_names=feature_names, importance_type='gain')
        gain_importance_list.append(gain_importance_df)

        split_importance_df = calc_model_importance(
            model, feature_names=feature_names, importance_type='split')
        split_importance_list.append(split_importance_df)

    print(scores)
    score = round(np.sqrt(mean_squared_error(y_true=y, y_pred=oof)), 3)
    print('score: ', score)

    gain_importance_df = calc_mean_importance(gain_importance_list)
    split_importance_df = calc_mean_importance(split_importance_list)

    return models, gain_importance_df, split_importance_df, oof, score

## === cell 26
lgb_params = {
    'objective': 'regression',
    'max_depth': 3,
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'lambda_l1': 1,
    'lambda_l2': 1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'verbosity': -1,
}

models, gain_importance_df, split_importance_df, oof, score = do_train(train, lgb_params)

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1902877284.py in <cell line: 0>()
     13 }
     14 
---> 15 models, gain_importance_df, split_importance_df, oof, score = do_train(train, lgb_params)

/tmp/ipykernel_11/1966887555.py in do_train(all_feature, params)
     68 
     69         # model
---> 70         model = lgb.train(params=params,
     71                           train_set=lgbm_train,
     72                           valid_sets=[lgbm_train, lgbm_valid],

TypeError: train() got an unexpected keyword argument 'verbose_eval'

## === cell 27
plot_importance(gain_importance_df, 'importance_gain')
plot_importance(split_importance_df, 'importance_split')

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3108999778.py in <cell line: 0>()
----> 1 plot_importance(gain_importance_df, 'importance_gain')
      2 plot_importance(split_importance_df, 'importance_split')

NameError: name 'gain_importance_df' is not defined

## === cell 28
plt.scatter(train[target_col], oof, s=2)
plt.xlabel('target')
plt.ylabel('oof')
plt.title(f'cv: {score}');

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1515729773.py in <cell line: 0>()
----> 1 plt.scatter(train[target_col], oof, s=2)
      2 plt.xlabel('target')
      3 plt.ylabel('oof')
      4 plt.title(f'cv: {score}');

NameError: name 'oof' is not defined

## === cell 29
mean_of_target = train[target_col].mean()
print(f'Target average: {mean_of_target}')

score = np.sqrt(mean_squared_error(y_true=train[target_col], y_pred = np.ones(len(train)) * mean_of_target))
print(f'RMSE when predicting the average value of the target: {score}')

## === cell 32
sample = pd.read_csv('../input/petfinder-pawpularity-score/sample_submission.csv')
test_feature = test.drop(['Id', 'img_path', 'width_height'], axis=1)

preds = []
for model in models:
    preds.append(model.predict(test_feature, num_iteration=model.best_iteration))

sample[target_col] = np.mean(preds, axis=0)
sample.to_csv('submission.csv', index=False)

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3832428005.py in <cell line: 0>()
      3 
      4 preds = []
----> 5 for model in models:
      6     preds.append(model.predict(test_feature, num_iteration=model.best_iteration))
      7 

NameError: name 'models' is not defined
