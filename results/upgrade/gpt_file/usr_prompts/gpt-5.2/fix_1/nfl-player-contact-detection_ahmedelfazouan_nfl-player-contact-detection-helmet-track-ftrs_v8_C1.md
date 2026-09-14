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
Predict moments of contact between football player pairs, as well as when players make non-foot contact with the ground, using game footage and tracking data.

## Metric
Matthews Correlation Coefficient between the predicted and actual contact events.

## Submission Format
For every allowable `contact_id` (formed by concatenating the `game_play_step_player1_player2`), you must predict whether the involved players are in contact at that moment in time. **sample_submission.csv** provides the exhaustive list of `contact_id`s. Note that the ground, denoted as player `G`, is included as a possible contact in place of player2. The player with the lower id is always listed first in the `contact_id`.

The file should contain a header and have the following format:

```
contact_id,contact
58168_003392_0_38590_43854,0
58168_003392_0_38590_41257,1
58168_003392_0_38590_41944,0
etc.

```

## Dataset 
**[train/test] mp4** videos of each play. Each play has three videos. The two main view are shot from the endzone and sideline. Sideline and Endzone video pairs are matched frame for frame in time, but different players may be visible in each view. This year, an additional view is provided, All29 which should include view of every player involved in the play. All29 video is not guaranteed to be time synced with the sideline and endzone.

These videos all contain a frame rate of 59.94 HZ. The moment of snap occurs 5 seconds into the video.

**train_labels.csv** Contains a row for every combination of players, and players with the ground for each 0.1 second timestamp in the play.

- `contact_id`: A combination of the game_play, player_ids and step columns.
- `game_play`: the unique ID for the game and play.
- `nfl_player_id_1` The lower numbered player id in the contact pair. If contact with ground then this is just the player id.
- `nfl_player_id_2`: The larger number player id in the contact pair. If for contact with the ground, this will contain an uppercase "G"
- `step`: A number representing each each timestep for each play, starting at 0 at the moment of the play starting, and incrementing by 1 every 0.1 seconds.
- `datetime`: The timetamp of the contact, at 10Hz
- `contact`: Whether contact occurred

Note: Labels may not be exact but are expected to be within +/-10Hz from the actual moment of contact. Labels were created in a multistep process including quality checks, however there still may be mislabels. You should expect the test labels to be of similar quality as the training set labels.

**sample_submission.csv** A valid sample submission file.

- `contact_id`: A combination of the game_key, play_id, nfl_player_ids and step (as explained above)
- `contact`: A binary value predicting contact. 1 indicates contact and 0 indicates no-contact.

**[train/test]_baseline_helmets.csv** contains imperfect baseline predictions for helmet boxes and player assignments for the Sideline and Endzone video view. The model used to create these predictions are from the winning solution from last year's competition and can be used to leverage your predictions.

- `game_play`: Unique game key and play id combination for the play.
- `game_key`: the ID code for the game.
- `play_id`: the ID code for the play.
- `view`: The video view, either `Sideline` or `Endzone`
- `video`: The filename of the associated video.
- `frame`: The associated frame within the video.
- `nfl_player_id`: The **imperfect** predicted player id.
- `player_label`: The player label. A combination of V/H (home or visiting team) and the player jersey number.
- `[left/width/top/height]`: the specification of the bounding box of the prediction.

**[train/test]_player_tracking.csv** Each player wears a sensor that allows us to locate them on the field; that information is reported in these two files.

- `game_play`: Unique game key and play id combination for the play.
- `game_key`: the ID code for the game.
- `play_id`: the ID code for the play.
- `nfl_player_id`: the player's ID code.
- `datetime`: timestamp at 10 Hz.
- `step`: timestep within play relative to the play start.
- `position`: the football position of the player.
- `team`: team of the player, either home or away.
- `jersey_number`: Player jersey number
- `x_position:` player position along the long axis of the field. See figure below.
- `y_position`: player position along the short axis of the field. See figure below.
- `speed`: speed in yards/second.
- `distance`: distance traveled from prior time point, in yards.
- `orientation`: orientation of player (deg).
- `direction`: angle of player motion (deg).
- `acceleration`: magnitiude of the total acceleration in yards/second^2.
- `sa`: Signed acceleration yards/second^2 in the direction the player is moving.

**[train/test]_video_metadata.csv** Metadata for each sideline and endzone video file including the timestamp information to be used to sync with player tracking data.

- `game_play`: Unique game key and play id combination for the play.
- `game_key`: the ID code for the game.
- `play_id`: the ID code for the play.
- `view`: The video view, either `Sideline` or `Endzone`
- `start_time`: The timestamp of the video start.
- `end_time`: The timestamp when the video ends.
- `snap_time`: The timestamp when the play starts within the video. This is 5 seconds (300 frames) into the video.

# 2. Python version

3.11

# 3. Installed packages

cudf-cu12==25.2.2
cudf-polars-cu12==25.6.0
cuml-cu12==25.2.1
cupy-cuda12x==13.6.0
dask-cudf-cu12==25.2.2
geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
libcudf-cu12==25.2.2
libcuml-cu12==25.2.1
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pylibcudf-cu12==25.2.2
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (165 lines)
            sample_submission.csv (463244 lines)
            sample_submission.csv.zip (1.4 MB)
            test.zip (381.3 MB)
            test_baseline_helmets.csv (371409 lines)
            test_baseline_helmets.csv.zip (2.9 MB)
            test_player_tracking.csv (127755 lines)
            test_player_tracking.csv.zip (2.9 MB)
            test_video_metadata.csv (49 lines)
            test_video_metadata.csv.zip (1.2 kB)
            train.zip (3.6 GB)
            train_baseline_helmets.csv (3412209 lines)
            train_baseline_helmets.csv.zip (26.1 MB)
            train_labels.csv (4258376 lines)
            train_labels.csv.zip (30.1 MB)
            train_player_tracking.csv (1225300 lines)
            train_player_tracking.csv.zip (28.0 MB)
            train_video_metadata.csv (433 lines)
            train_video_metadata.csv.zip (7.7 kB)
            nfl-player-contact-detection/
                description.md (165 lines)
                sample_submission.csv (463244 lines)
                ... and 17 other files
                nfl-player-contact-detection/
                test/
                    58187_001341_All29.mp4 (4.0 MB)
                    58187_001341_Endzone.mp4 (6.9 MB)
                    ... and 70 other files
                    test/
                train/
                    58168_003392_All29.mp4 (3.6 MB)
                    58168_003392_Endzone.mp4 (6.1 MB)
                    ... and 646 other files
                    train/
            test/
                58187_001341_All29.mp4 (4.0 MB)
                58187_001341_Endzone.mp4 (6.9 MB)
                ... and 70 other files
                test/
            train/
                58168_003392_All29.mp4 (3.6 MB)
                58168_003392_Endzone.mp4 (6.1 MB)
                ... and 646 other files
                train/
        input/
            description.md (165 lines)
            sample_submission.csv (463244 lines)
            sample_submission.csv.zip (1.4 MB)
            test.zip (381.3 MB)
            test_baseline_helmets.csv (371409 lines)
            test_baseline_helmets.csv.zip (2.9 MB)
            test_player_tracking.csv (127755 lines)
            test_player_tracking.csv.zip (2.9 MB)
            test_video_metadata.csv (49 lines)
            test_video_metadata.csv.zip (1.2 kB)
            train.zip (3.6 GB)
            train_baseline_helmets.csv (3412209 lines)
            train_baseline_helmets.csv.zip (26.1 MB)
            train_labels.csv (4258376 lines)
            train_labels.csv.zip (30.1 MB)
            train_player_tracking.csv (1225300 lines)
            train_player_tracking.csv.zip (28.0 MB)
            train_video_metadata.csv (433 lines)
            train_video_metadata.csv.zip (7.7 kB)
            nfl-player-contact-detection/
                description.md (165 lines)
                sample_submission.csv (463244 lines)
                ... and 17 other files
                nfl-player-contact-detection/
                test/
                    58187_001341_All29.mp4 (4.0 MB)
                    58187_001341_Endzone.mp4 (6.9 MB)
                    ... and 70 other files
                    test/
                train/
                    58168_003392_All29.mp4 (3.6 MB)
                    58168_003392_Endzone.mp4 (6.1 MB)
                    ... and 646 other files
                    train/
            test/
                58187_001341_All29.mp4 (4.0 MB)
                58187_001341_Endzone.mp4 (6.9 MB)
                ... and 70 other files
                test/
                    58187_001341_All29.mp4 (4.0 MB)
                    58187_001341_Endzone.mp4 (6.9 MB)
                    ... and 70 other files
                    test/
            train/
                58168_003392_All29.mp4 (3.6 MB)
                58168_003392_Endzone.mp4 (6.1 MB)
                ... and 646 other files
                train/
                    58168_003392_All29.mp4 (3.6 MB)
                    58168_003392_Endzone.mp4 (6.1 MB)
                    ... and 646 other files
                    train/
        working/
            nfl-player-contact-detection/
                description.md (165 lines)
                sample_submission.csv (463244 lines)
                ... and 17 other files
                nfl-player-contact-detection/
                test/
                    58187_001341_All29.mp4 (4.0 MB)
                    58187_001341_Endzone.mp4 (6.9 MB)
                    ... and 70 other files
                    test/
                train/
                    58168_003392_All29.mp4 (3.6 MB)
                    58168_003392_Endzone.mp4 (6.1 MB)
                    ... and 646 other files
                    train/
```

-> data/nfl-player-contact-detection/sample_submission.csv has 463243 rows and 2 columns.
The columns are: contact_id, contact

-> data/nfl-player-contact-detection/test_baseline_helmets.csv has 371408 rows and 12 columns.
The columns are: game_play, game_key, play_id, view, video, frame, nfl_player_id, player_label, left, width, top, height

-> data/nfl-player-contact-detection/test_player_tracking.csv has 127754 rows and 17 columns.
The columns are: game_play, game_key, play_id, nfl_player_id, datetime, step, team, position, jersey_number, x_position, y_position, speed, distance, direction, orientation... and 2 more columns

-> data/nfl-player-contact-detection/test_video_metadata.csv has 48 rows and 7 columns.
The columns are: game_play, game_key, play_id, view, start_time, end_time, snap_time

-> data/nfl-player-contact-detection/train_baseline_helmets.csv has 3412208 rows and 12 columns.
The columns are: game_play, game_key, play_id, view, video, frame, nfl_player_id, player_label, left, width, top, height

-> data/nfl-player-contact-detection/train_labels.csv has 4258375 rows and 7 columns.
The columns are: contact_id, game_play, datetime, step, nfl_player_id_1, nfl_player_id_2, contact

-> data/nfl-player-contact-detection/train_player_tracking.csv has 1225299 rows and 17 columns.
The columns are: game_play, game_key, play_id, nfl_player_id, datetime, step, team, position, jersey_number, x_position, y_position, speed, distance, direction, orientation... and 2 more columns

-> data/nfl-player-contact-detection/train_video_metadata.csv has 432 rows and 7 columns.
The columns are: game_play, game_key, play_id, view, start_time, end_time, snap_time

-> (stopped after 10 files for performance)

# 5. Target score

0.64447

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import torch

class Config:
    AUTHOR = "colum2131"

    NAME = "NFLC-" + "Exp001-simple-xgb-baseline"

    COMPETITION = "nfl-player-contact-detection"

    seed = 42
    num_fold = 5
    
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate':0.03,
        'tree_method':'hist' if not torch.cuda.is_available() else 'gpu_hist'
    }


## === cell 1
import os
import gc
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import Video, display

from scipy.optimize import minimize
import cv2
from glob import glob
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    matthews_corrcoef,
)

import xgboost as xgb

import torch

if torch.cuda.is_available():
    import cupy 
    import cudf
    from cuml import ForestInference


## === cell 2
def setup(cfg):
    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cfg.INPUT = f'../input/{cfg.COMPETITION}'
    cfg.EXP = cfg.NAME
    cfg.OUTPUT_EXP = cfg.NAME
    cfg.SUBMISSION = './'
    cfg.DATASET = '../input/'

    cfg.EXP_MODEL = os.path.join(cfg.EXP, 'model')
    cfg.EXP_FIG = os.path.join(cfg.EXP, 'fig')
    cfg.EXP_PREDS = os.path.join(cfg.EXP, 'preds')

    for d in [cfg.EXP_MODEL, cfg.EXP_FIG, cfg.EXP_PREDS]:
        os.makedirs(d, exist_ok=True)
        
    return cfg


## === cell 3
def add_contact_id(df):
    df["contact_id"] = (
        df["game_play"]
        + "_"
        + df["step"].astype("str")
        + "_"
        + df["nfl_player_id_1"].astype("str")
        + "_"
        + df["nfl_player_id_2"].astype("str")
    )
    return df

def expand_contact_id(df):
    """
    Splits out contact_id into seperate columns.
    """
    df["game_play"] = df["contact_id"].str[:12]
    df["step"] = df["contact_id"].str.split("_").str[-3].astype("int")
    df["nfl_player_id_1"] = df["contact_id"].str.split("_").str[-2]
    df["nfl_player_id_2"] = df["contact_id"].str.split("_").str[-1]
    return df

def get_groupkfold(train, target_col, group_col, n_splits):
    kf = GroupKFold(n_splits=n_splits)
    generator = kf.split(train, train[target_col], train[group_col])
    fold_series = []
    for fold, (idx_train, idx_valid) in enumerate(generator):
        fold_series.append(pd.Series(fold, index=idx_valid))
    fold_series = pd.concat(fold_series).sort_index()
    return fold_series

def fit_xgboost(cfg, X, y, params, add_suffix=''):
    """
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate':0.01,
        'tree_method':'gpu_hist'
    }
    """
    oof_pred = np.zeros(len(y), dtype=np.float32)
    for fold in sorted(cfg.folds.unique()):
        if fold == -1: continue
        idx_train = (cfg.folds!=fold)
        idx_valid = (cfg.folds==fold)
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]
        display(pd.Series(y_valid).value_counts())

        xgb_train = xgb.DMatrix(x_train, label=y_train)
        xgb_valid = xgb.DMatrix(x_valid, label=y_valid)
        evals = [(xgb_train,'train'),(xgb_valid,'eval')]

        model = xgb.train(
            params,
            xgb_train,
            num_boost_round=10_000,
            early_stopping_rounds=100,
            evals=evals,
            verbose_eval=100,
        )

        model_path = os.path.join(cfg.EXP_MODEL, f'xgb_fold{fold}{add_suffix}.model')
        model.save_model(model_path)
        if not torch.cuda.is_available():
            model = xgb.Booster().load_model(model_path)
        else:
            model = ForestInference.load(model_path, output_class=True, model_type='xgboost')
        pred_i = model.predict_proba(x_valid)[:, 1]
        oof_pred[x_valid.index] = pred_i
        score = round(roc_auc_score(y_valid, pred_i), 5)
        print(f'Performance of the prediction: {score}\n')
        del model; gc.collect()

    np.save(os.path.join(cfg.EXP_PREDS, f'oof_pred{add_suffix}'), oof_pred)
    score = round(roc_auc_score(y, oof_pred), 5)
    print(f'All Performance of the prediction: {score}')
    return oof_pred

def pred_xgboost(X, data_dir, add_suffix=''):
    models = glob(os.path.join(data_dir, f'xgb_fold*{add_suffix}.model'))
    if not torch.cuda.is_available():
         models = [xgb.Booster().load_model(model_path) for model in models]
    else:
        models = [ForestInference.load(model, output_class=True, model_type='xgboost') for model in models]
    preds = np.array([model.predict_proba(X)[:, 1] for model in models])
    preds = np.mean(preds, axis=0)
    return preds


## === cell 4
cfg = setup(Config)

if not torch.cuda.is_available():
    tr_tracking = pd.read_csv(os.path.join(cfg.INPUT, 'train_player_tracking.csv'), parse_dates=["datetime"])
    te_tracking = pd.read_csv(os.path.join(cfg.INPUT, 'test_player_tracking.csv'), parse_dates=["datetime"])
    sub = pd.read_csv(os.path.join(cfg.INPUT, 'sample_submission.csv'))

    train = pd.read_csv(os.path.join(cfg.INPUT, 'train_labels.csv'), parse_dates=["datetime"])
    test = expand_contact_id(sub)
    
else:
    tr_tracking = cudf.read_csv(os.path.join(cfg.INPUT, 'train_player_tracking.csv'), parse_dates=["datetime"])
    te_tracking = cudf.read_csv(os.path.join(cfg.INPUT, 'test_player_tracking.csv'), parse_dates=["datetime"])
    sub = pd.read_csv(os.path.join(cfg.INPUT, 'sample_submission.csv'))

    train = cudf.read_csv(os.path.join(cfg.INPUT, 'train_labels.csv'), parse_dates=["datetime"])
    test = cudf.DataFrame(expand_contact_id(sub))


## === cell 5
def create_features(df, tr_tracking, merge_col="step", use_cols=["x_position", "y_position"]):
    output_cols = []
    df_combo = (
        df.astype({"nfl_player_id_1": "str"})
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id",] + use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_1"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .rename(columns={c: c+"_1" for c in use_cols})
        .drop("nfl_player_id", axis=1)
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id"] + use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_2"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .drop("nfl_player_id", axis=1)
        .rename(columns={c: c+"_2" for c in use_cols})
        .sort_values(["game_play", merge_col, "nfl_player_id_1", "nfl_player_id_2"])
        .reset_index(drop=True)
    )
    output_cols += [c+"_1" for c in use_cols]
    output_cols += [c+"_2" for c in use_cols]
    
    if ("x_position" in use_cols) & ("y_position" in use_cols):
        index = df_combo['x_position_2'].notnull()
        if torch.cuda.is_available():
            index = index.to_array()
        distance_arr = np.full(len(index), np.nan)
        tmp_distance_arr = np.sqrt(
            np.square(df_combo.loc[index, "x_position_1"] - df_combo.loc[index, "x_position_2"])
            + np.square(df_combo.loc[index, "y_position_1"]- df_combo.loc[index, "y_position_2"])
        )
        if torch.cuda.is_available():
            tmp_distance_arr = tmp_distance_arr.to_array()
        distance_arr[index] = tmp_distance_arr
        df_combo['distance'] = distance_arr
        output_cols += ["distance"]
        
    df_combo['G_flug'] = (df_combo['nfl_player_id_2']=="G")
    output_cols += ["G_flug"]
    return df_combo, output_cols


use_cols = [
    'x_position', 'y_position', 'speed', 'distance',
    'direction', 'orientation', 'acceleration', 'sa'
]
train, feature_cols = create_features(train, tr_tracking, use_cols=use_cols)
test, feature_cols = create_features(test, te_tracking, use_cols=use_cols)
if torch.cuda.is_available():
    train = train.to_pandas()
    test = test.to_pandas()

display(train)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/335372663.py in <cell line: 0>()
     56     'direction', 'orientation', 'acceleration', 'sa'
     57 ]
---> 58 train, feature_cols = create_features(train, tr_tracking, use_cols=use_cols)
     59 test, feature_cols = create_features(test, te_tracking, use_cols=use_cols)
     60 if torch.cuda.is_available():

/tmp/ipykernel_11/335372663.py in create_features(df, tr_tracking, merge_col, use_cols)
     35         index = df_combo['x_position_2'].notnull()
     36         if torch.cuda.is_available():
---> 37             index = index.to_array()
     38         distance_arr = np.full(len(index), np.nan)
     39         tmp_distance_arr = np.sqrt(

AttributeError: 'Series' object has no attribute 'to_array'

## === cell 6
DISTANCE_THRESH = 2

train_y = train['contact'].values
oof_pred = np.zeros(len(train))
cond_dis_train = (train['distance']<=DISTANCE_THRESH) | (train['distance'].isna())
cond_dis_test = (test['distance']<=DISTANCE_THRESH) | (test['distance'].isna())

train = train[cond_dis_train]
train.reset_index(inplace = True, drop = True)

print('number of train data : ',len(train))

_ = gc.collect()


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2857072787.py in <cell line: 0>()
      3 train_y = train['contact'].values
      4 oof_pred = np.zeros(len(train))
----> 5 cond_dis_train = (train['distance']<=DISTANCE_THRESH) | (train['distance'].isna())
      6 cond_dis_test = (test['distance']<=DISTANCE_THRESH) | (test['distance'].isna())
      7 

/usr/local/lib/python3.11/dist-packages/cudf/utils/performance_tracking.py in wrapper(*args, **kwargs)
     49                     )
     50                 )
---> 51             return func(*args, **kwargs)
     52 
     53     return wrapper

/usr/local/lib/python3.11/dist-packages/cudf/core/dataframe.py in __getitem__(self, arg)
   1360         """
   1361         if _is_scalar_or_zero_d_array(arg) or isinstance(arg, tuple):
-> 1362             out = self._get_columns_by_label(arg)
   1363             if is_scalar(arg):
   1364                 nlevels = 1

/usr/local/lib/python3.11/dist-packages/cudf/utils/performance_tracking.py in wrapper(*args, **kwargs)
     49                     )
     50                 )
---> 51             return func(*args, **kwargs)
     52 
     53     return wrapper

/usr/local/lib/python3.11/dist-packages/cudf/core/frame.py in _get_columns_by_label(self, labels)
    404         Akin to cudf.DataFrame(...).loc[:, labels]
    405         """
--> 406         return self._from_data_like_self(self._data.select_by_label(labels))
    407 
    408     @property

/usr/local/lib/python3.11/dist-packages/cudf/core/column_accessor.py in select_by_label(self, key)
    411                 if any(isinstance(k, slice) for k in key):
    412                     return self._select_by_label_with_wildcard(key)
--> 413             return self._select_by_label_grouped(key)
    414 
    415     def get_labels_by_index(self, index: Any) -> tuple:

/usr/local/lib/python3.11/dist-packages/cudf/core/column_accessor.py in _select_by_label_grouped(self, key)
    573 
    574     def _select_by_label_grouped(self, key: abc.Hashable) -> Self:
--> 575         result = self._grouped_data[key]
    576         if isinstance(result, column.ColumnBase):
    577             # self._grouped_data[key] = self._data[key] so skip validation

KeyError: 'distance'

## === cell 7
CLUSTERS = [10, 50, 100, 500]

def add_step_pct(df, cluster):
    df['step_pct'] = cluster * (df['step']-min(df['step']))/(max(df['step'])-min(df['step']))
    df['step_pct'] = df['step_pct'].apply(np.ceil).astype(np.int32)
    return df

for cluster in CLUSTERS:
    train = train.groupby('game_play').apply(lambda x:add_step_pct(x,cluster))
    test = test.groupby('game_play').apply(lambda x:add_step_pct(x,cluster))

    for helmet_view in ['Sideline', 'Endzone']:
        helmet_train = pd.read_csv('/kaggle/input/nfl-player-contact-detection/train_baseline_helmets.csv')
        helmet_train.loc[helmet_train['view']=='Endzone2','view'] = 'Endzone'
        helmet_test = pd.read_csv('/kaggle/input/nfl-player-contact-detection/test_baseline_helmets.csv')
        helmet_test.loc[helmet_test['view']=='Endzone2','view'] = 'Endzone'

        helmet_train.rename(columns = {'frame': 'step'}, inplace = True)
        helmet_train = helmet_train.groupby('game_play').apply(lambda x:add_step_pct(x,cluster))
        helmet_test.rename(columns = {'frame': 'step'}, inplace = True)
        helmet_test = helmet_test.groupby('game_play').apply(lambda x:add_step_pct(x,cluster))
        helmet_train = helmet_train[helmet_train['view']==helmet_view]
        helmet_test = helmet_test[helmet_test['view']==helmet_view]

        helmet_train['helmet_id'] = helmet_train['game_play'] + '_' + helmet_train['nfl_player_id'].astype(str) + '_' + helmet_train['step_pct'].astype(str)
        helmet_test['helmet_id'] = helmet_test['game_play'] + '_' + helmet_test['nfl_player_id'].astype(str) + '_' + helmet_test['step_pct'].astype(str)

        helmet_train = helmet_train[['helmet_id', 'left', 'width', 'top', 'height']].groupby('helmet_id').mean().reset_index()
        helmet_test = helmet_test[['helmet_id', 'left', 'width', 'top', 'height']].groupby('helmet_id').mean().reset_index()
        for player_ind in [1, 2]:
            train['helmet_id'] = train['game_play'] + '_' + train['nfl_player_id_'+str(player_ind)].astype(str) + \
                                    '_' + train['step_pct'].astype(str)
            test['helmet_id'] = test['game_play'] + '_' + test['nfl_player_id_'+str(player_ind)].astype(str) + \
                                    '_' + test['step_pct'].astype(str)

            train = train.merge(helmet_train, how = 'left')
            test = test.merge(helmet_test, how = 'left')

            train.rename(columns = {i:i+'_'+helmet_view+'_'+str(cluster)+'_'+str(player_ind) for i in ['left', 'width', 'top', 'height']}, inplace = True)
            test.rename(columns = {i:i+'_'+helmet_view+'_'+str(cluster)+'_'+str(player_ind) for i in ['left', 'width', 'top', 'height']}, inplace = True)

            del train['helmet_id'], test['helmet_id']
            gc.collect()

            feature_cols += [i+'_'+helmet_view+'_'+str(cluster)+'_'+str(player_ind) for i in ['left', 'width', 'top', 'height']]
        del helmet_train, helmet_test
        gc.collect()


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4033002100.py in <cell line: 0>()
      7 
      8 for cluster in CLUSTERS:
----> 9     train = train.groupby('game_play').apply(lambda x:add_step_pct(x,cluster))
     10     test = test.groupby('game_play').apply(lambda x:add_step_pct(x,cluster))
     11 

/usr/local/lib/python3.11/dist-packages/cudf/utils/performance_tracking.py in wrapper(*args, **kwargs)
     49                     )
     50                 )
---> 51             return func(*args, **kwargs)
     52 
     53     return wrapper

/usr/local/lib/python3.11/dist-packages/cudf/core/groupby/groupby.py in apply(self, func, engine, include_groups, *args, **kwargs)
   1988             )
   1989         elif engine == "cudf":
-> 1990             result = self._iterative_groupby_apply(
   1991                 func,
   1992                 group_names,

/usr/local/lib/python3.11/dist-packages/cudf/utils/performance_tracking.py in wrapper(*args, **kwargs)
     49                     )
     50                 )
---> 51             return func(*args, **kwargs)
     52 
     53     return wrapper

/usr/local/lib/python3.11/dist-packages/cudf/core/groupby/groupby.py in _iterative_groupby_apply(self, function, group_names, offsets, group_keys, grouped_values, *args)
   1765 
   1766         chunks = [grouped_values[s:e] for s, e in itertools.pairwise(offsets)]
-> 1767         chunk_results = [function(chk, *args) for chk in chunks]
   1768         return self._post_process_chunk_results(
   1769             chunk_results, group_names, group_keys, grouped_values

/usr/local/lib/python3.11/dist-packages/cudf/core/groupby/groupby.py in <listcomp>(.0)
   1765 
   1766         chunks = [grouped_values[s:e] for s, e in itertools.pairwise(offsets)]
-> 1767         chunk_results = [function(chk, *args) for chk in chunks]
   1768         return self._post_process_chunk_results(
   1769             chunk_results, group_names, group_keys, grouped_values

/tmp/ipykernel_11/4033002100.py in <lambda>(x)
      7 
      8 for cluster in CLUSTERS:
----> 9     train = train.groupby('game_play').apply(lambda x:add_step_pct(x,cluster))
     10     test = test.groupby('game_play').apply(lambda x:add_step_pct(x,cluster))
     11 

/tmp/ipykernel_11/4033002100.py in add_step_pct(df, cluster)
      2 
      3 def add_step_pct(df, cluster):
----> 4     df['step_pct'] = cluster * (df['step']-min(df['step']))/(max(df['step'])-min(df['step']))
      5     df['step_pct'] = df['step_pct'].apply(np.ceil).astype(np.int32)
      6     return df

/usr/local/lib/python3.11/dist-packages/cudf/utils/utils.py in __iter__(self)
    243         information.
    244         """
--> 245         raise TypeError(
    246             f"{self.__class__.__name__} object is not iterable. "
    247             f"Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` "

TypeError: Series object is not iterable. Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` if you wish to iterate over the values.

## === cell 8
for cluster in CLUSTERS:
    for helmet_view in ['Sideline', 'Endzone']:
        train.loc[train['G_flug']==True,'left_'+helmet_view+'_'+str(cluster)+'_2'] = train.loc[train['G_flug']==True,'left_'+helmet_view+'_'+str(cluster)+'_1']
        train.loc[train['G_flug']==True,'top_'+helmet_view+'_'+str(cluster)+'_2'] = train.loc[train['G_flug']==True,'top_'+helmet_view+'_'+str(cluster)+'_1']
        train.loc[train['G_flug']==True,'width_'+helmet_view+'_'+str(cluster)+'_2'] = 0
        train.loc[train['G_flug']==True,'height_'+helmet_view+'_'+str(cluster)+'_2'] = 0
        
        test.loc[test['G_flug']==True,'left_'+helmet_view+'_'+str(cluster)+'_2'] = test.loc[test['G_flug']==True,'left_'+helmet_view+'_'+str(cluster)+'_1']
        test.loc[test['G_flug']==True,'top_'+helmet_view+'_'+str(cluster)+'_2'] = test.loc[test['G_flug']==True,'top_'+helmet_view+'_'+str(cluster)+'_1']
        test.loc[test['G_flug']==True,'width_'+helmet_view+'_'+str(cluster)+'_2'] = 0
        test.loc[test['G_flug']==True,'height_'+helmet_view+'_'+str(cluster)+'_2'] = 0


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3653189514.py in <cell line: 0>()
      1 for cluster in CLUSTERS:
      2     for helmet_view in ['Sideline', 'Endzone']:
----> 3         train.loc[train['G_flug']==True,'left_'+helmet_view+'_'+str(cluster)+'_2'] = train.loc[train['G_flug']==True,'left_'+helmet_view+'_'+str(cluster)+'_1']
      4         train.loc[train['G_flug']==True,'top_'+helmet_view+'_'+str(cluster)+'_2'] = train.loc[train['G_flug']==True,'top_'+helmet_view+'_'+str(cluster)+'_1']
      5         train.loc[train['G_flug']==True,'width_'+helmet_view+'_'+str(cluster)+'_2'] = 0

/usr/local/lib/python3.11/dist-packages/cudf/utils/performance_tracking.py in wrapper(*args, **kwargs)
     49                     )
     50                 )
---> 51             return func(*args, **kwargs)
     52 
     53     return wrapper

/usr/local/lib/python3.11/dist-packages/cudf/core/dataframe.py in __getitem__(self, arg)
   1360         """
   1361         if _is_scalar_or_zero_d_array(arg) or isinstance(arg, tuple):
-> 1362             out = self._get_columns_by_label(arg)
   1363             if is_scalar(arg):
   1364                 nlevels = 1

/usr/local/lib/python3.11/dist-packages/cudf/utils/performance_tracking.py in wrapper(*args, **kwargs)
     49                     )
     50                 )
---> 51             return func(*args, **kwargs)
     52 
     53     return wrapper

/usr/local/lib/python3.11/dist-packages/cudf/core/frame.py in _get_columns_by_label(self, labels)
    404         Akin to cudf.DataFrame(...).loc[:, labels]
    405         """
--> 406         return self._from_data_like_self(self._data.select_by_label(labels))
    407 
    408     @property

/usr/local/lib/python3.11/dist-packages/cudf/core/column_accessor.py in select_by_label(self, key)
    411                 if any(isinstance(k, slice) for k in key):
    412                     return self._select_by_label_with_wildcard(key)
--> 413             return self._select_by_label_grouped(key)
    414 
    415     def get_labels_by_index(self, index: Any) -> tuple:

/usr/local/lib/python3.11/dist-packages/cudf/core/column_accessor.py in _select_by_label_grouped(self, key)
    573 
    574     def _select_by_label_grouped(self, key: abc.Hashable) -> Self:
--> 575         result = self._grouped_data[key]
    576         if isinstance(result, column.ColumnBase):
    577             # self._grouped_data[key] = self._data[key] so skip validation

KeyError: 'G_flug'

## === cell 9
cols = [i[:-2] for i in train.columns if i[-2:]=='_1' and i!='nfl_player_id_1']
train[[i+'_diff' for i in cols]] = np.abs(train[[i+'_1' for i in cols]].values - train[[i+'_2' for i in cols]].values)
test[[i+'_diff' for i in cols]] = np.abs(test[[i+'_1' for i in cols]].values - test[[i+'_2' for i in cols]].values)
feature_cols += [i+'_diff' for i in cols]

cols = ['x_position', 'y_position', 'speed', 'distance', 'direction', 'orientation', 'acceleration', 'sa']
train[[i+'_prod' for i in cols]] = train[[i+'_1' for i in cols]].values * train[[i+'_2' for i in cols]].values
test[[i+'_prod' for i in cols]] = test[[i+'_1' for i in cols]].values * test[[i+'_2' for i in cols]].values
feature_cols += [i+'_prod' for i in cols]

print('number of features : ',len(feature_cols))
print('number of train data : ',len(train))


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3920378547.py in <cell line: 0>()
      2 train[[i+'_diff' for i in cols]] = np.abs(train[[i+'_1' for i in cols]].values - train[[i+'_2' for i in cols]].values)
      3 test[[i+'_diff' for i in cols]] = np.abs(test[[i+'_1' for i in cols]].values - test[[i+'_2' for i in cols]].values)
----> 4 feature_cols += [i+'_diff' for i in cols]
      5 
      6 cols = ['x_position', 'y_position', 'speed', 'distance', 'direction', 'orientation', 'acceleration', 'sa']

NameError: name 'feature_cols' is not defined

## === cell 10

cfg.folds = get_groupkfold(train, 'contact', 'game_play', cfg.num_fold)
cfg.folds.to_csv(os.path.join(cfg.EXP_PREDS, 'folds.csv'), index=False)

oof_pred[np.where(cond_dis_train)] = fit_xgboost(cfg, train[feature_cols], train['contact'], 
                                              cfg.xgb_params, add_suffix="_xgb_1st")
np.save('oof_pred.npy',oof_pred)
sub_pred = pred_xgboost(test.loc[cond_dis_test, feature_cols], cfg.EXP_MODEL, add_suffix="_xgb_1st")


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/300146003.py in <cell line: 0>()
      3 # ==============================
      4 
----> 5 cfg.folds = get_groupkfold(train, 'contact', 'game_play', cfg.num_fold)
      6 cfg.folds.to_csv(os.path.join(cfg.EXP_PREDS, 'folds.csv'), index=False)
      7 

/tmp/ipykernel_11/4262739256.py in get_groupkfold(train, target_col, group_col, n_splits)
     31     generator = kf.split(train, train[target_col], train[group_col])
     32     fold_series = []
---> 33     for fold, (idx_train, idx_valid) in enumerate(generator):
     34         fold_series.append(pd.Series(fold, index=idx_valid))
     35     fold_series = pd.concat(fold_series).sort_index()

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in split(self, X, y, groups)
    350             )
    351 
--> 352         for train, test in super().split(X, y, groups):
    353             yield train, test
    354 

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in split(self, X, y, groups)
     83         X, y, groups = indexable(X, y, groups)
     84         indices = np.arange(_num_samples(X))
---> 85         for test_index in self._iter_test_masks(X, y, groups):
     86             train_index = indices[np.logical_not(test_index)]
     87             test_index = indices[test_index]

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in _iter_test_masks(self, X, y, groups)
     95         By default, delegates to _iter_test_indices(X, y, groups)
     96         """
---> 97         for test_index in self._iter_test_indices(X, y, groups):
     98             test_mask = np.zeros(_num_samples(X), dtype=bool)
     99             test_mask[test_index] = True

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in _iter_test_indices(self, X, y, groups)
    529         if groups is None:
    530             raise ValueError("The 'groups' parameter should not be None.")
--> 531         groups = check_array(groups, input_name="groups", ensure_2d=False, dtype=None)
    532 
    533         unique_groups, groups = np.unique(groups, return_inverse=True)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    877                     array = xp.astype(array, dtype, copy=False)
    878                 else:
--> 879                     array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
    880             except ComplexWarning as complex_warning:
    881                 raise ValueError(

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_array_api.py in _asarray_with_order(array, dtype, order, copy, xp)
    183     if xp.__name__ in {"numpy", "numpy.array_api"}:
    184         # Use NumPy API to support order
--> 185         array = numpy.asarray(array, order=order, dtype=dtype)
    186         return xp.asarray(array, copy=copy)
    187     else:

/usr/local/lib/python3.11/dist-packages/cudf/utils/performance_tracking.py in wrapper(*args, **kwargs)
     49                     )
     50                 )
---> 51             return func(*args, **kwargs)
     52 
     53     return wrapper

/usr/local/lib/python3.11/dist-packages/cudf/core/frame.py in __array__(self, dtype, copy)
    440     @_performance_tracking
    441     def __array__(self, dtype=None, copy=None):
--> 442         raise TypeError(
    443             "Implicit conversion to a host NumPy array via __array__ is not "
    444             "allowed, To explicitly construct a GPU matrix, consider using "

TypeError: Implicit conversion to a host NumPy array via __array__ is not allowed, To explicitly construct a GPU matrix, consider using .to_cupy()
To explicitly construct a host matrix, consider using .to_numpy().

## === cell 11
def func(x_list):
    score = matthews_corrcoef(train_y, oof_pred>x_list[0])
    return -score

x0 = [0.5]
result = minimize(func, x0,  method="nelder-mead")
cfg.threshold = result.x[0]
print("score:", round(matthews_corrcoef(train_y, oof_pred>cfg.threshold), 5))
print("threshold", round(cfg.threshold, 5))
del train
gc.collect()

test = add_contact_id(test)
test['contact'] = 0
test.loc[cond_dis_test, 'contact'] = (sub_pred > cfg.threshold).astype(int)
test[['contact_id', 'contact']].to_csv('submission.csv', index=False)
display(test[['contact_id', 'contact']].head())


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/678705812.py in <cell line: 0>()
      7 
      8 x0 = [0.5]
----> 9 result = minimize(func, x0,  method="nelder-mead")
     10 cfg.threshold = result.x[0]
     11 print("score:", round(matthews_corrcoef(train_y, oof_pred>cfg.threshold), 5))

/usr/local/lib/python3.11/dist-packages/scipy/optimize/_minimize.py in minimize(fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options)
    724 
    725     if meth == 'nelder-mead':
--> 726         res = _minimize_neldermead(fun, x0, args, callback, bounds=bounds,
    727                                    **options)
    728     elif meth == 'powell':

/usr/local/lib/python3.11/dist-packages/scipy/optimize/_optimize.py in _minimize_neldermead(func, x0, args, callback, maxiter, maxfev, disp, return_all, initial_simplex, xatol, fatol, adaptive, bounds, **unknown_options)
    831     try:
    832         for k in range(N + 1):
--> 833             fsim[k] = func(sim[k])
    834     except _MaxFuncCallError:
    835         pass

/usr/local/lib/python3.11/dist-packages/scipy/optimize/_optimize.py in function_wrapper(x, *wrapper_args)
    540         ncalls[0] += 1
    541         # A copy of x is sent to the user function (gh13740)
--> 542         fx = function(np.copy(x), *(wrapper_args + args))
    543         # Ideally, we'd like to a have a true scalar returned from f(x). For
    544         # backwards-compatibility, also allow np.array([1.3]),

/tmp/ipykernel_11/678705812.py in func(x_list)
      3 # ==============================
      4 def func(x_list):
----> 5     score = matthews_corrcoef(train_y, oof_pred>x_list[0])
      6     return -score
      7 

/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py in matthews_corrcoef(y_true, y_pred, sample_weight)
    909     -0.33...
    910     """
--> 911     y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    912     check_consistent_length(y_true, y_pred, sample_weight)
    913     if y_type not in {"binary", "multiclass"}:

/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py in _check_targets(y_true, y_pred)
     85     """
     86     check_consistent_length(y_true, y_pred)
---> 87     type_true = type_of_target(y_true, input_name="y_true")
     88     type_pred = type_of_target(y_pred, input_name="y_pred")
     89 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/multiclass.py in type_of_target(y, input_name)
    307         raise ValueError("y cannot be class 'SparseSeries' or 'SparseArray'")
    308 
--> 309     if is_multilabel(y):
    310         return "multilabel-indicator"
    311 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/multiclass.py in is_multilabel(y)
    167             warnings.simplefilter("error", np.VisibleDeprecationWarning)
    168             try:
--> 169                 y = check_array(y, dtype=None, **check_y_kwargs)
    170             except (np.VisibleDeprecationWarning, ValueError) as e:
    171                 if str(e).startswith("Complex data not supported"):

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    877                     array = xp.astype(array, dtype, copy=False)
    878                 else:
--> 879                     array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
    880             except ComplexWarning as complex_warning:
    881                 raise ValueError(

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_array_api.py in _asarray_with_order(array, dtype, order, copy, xp)
    183     if xp.__name__ in {"numpy", "numpy.array_api"}:
    184         # Use NumPy API to support order
--> 185         array = numpy.asarray(array, order=order, dtype=dtype)
    186         return xp.asarray(array, copy=copy)
    187     else:

cupy/_core/core.pyx in cupy._core.core._ndarray_base.__array__()

TypeError: Implicit conversion to a NumPy array is not allowed. Please use `.get()` to construct a NumPy array explicitly.
