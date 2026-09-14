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

0.57012

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
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


class Config:
    COMPETITION = "nfl-player-contact-detection"
    NAME = "exp"
    num_fold = 5
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "nthread": 4,
        "seed": 42,
    }




## === cell 1
def setup(cfg):
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.INPUT = f"../input/{cfg.COMPETITION}"
    cfg.EXP = cfg.NAME
    cfg.OUTPUT_EXP = cfg.NAME
    cfg.SUBMISSION = "./"
    cfg.DATASET = "../input/"

    cfg.EXP_MODEL = os.path.join(cfg.EXP, "model")
    cfg.EXP_FIG = os.path.join(cfg.EXP, "fig")
    cfg.EXP_PREDS = os.path.join(cfg.EXP, "preds")

    for d in [cfg.EXP_MODEL, cfg.EXP_FIG, cfg.EXP_PREDS]:
        os.makedirs(d, exist_ok=True)

    return cfg




## === cell 2
cfg = setup(Config)

if not torch.cuda.is_available():
    tr_tracking = pd.read_csv(
        os.path.join(cfg.INPUT, "train_player_tracking.csv"), parse_dates=["datetime"]
    )
    te_tracking = pd.read_csv(
        os.path.join(cfg.INPUT, "test_player_tracking.csv"), parse_dates=["datetime"]
    )
    sub = pd.read_csv(os.path.join(cfg.INPUT, "sample_submission.csv"))

    train = pd.read_csv(
        os.path.join(cfg.INPUT, "train_labels.csv"), parse_dates=["datetime"]
    )
    test = None  # placeholder, will be created after expanding contact_id
else:
    tr_tracking = cudf.read_csv(
        os.path.join(cfg.INPUT, "train_player_tracking.csv"), parse_dates=["datetime"]
    )
    te_tracking = cudf.read_csv(
        os.path.join(cfg.INPUT, "test_player_tracking.csv"), parse_dates=["datetime"]
    )
    sub = pd.read_csv(os.path.join(cfg.INPUT, "sample_submission.csv"))

    train = cudf.read_csv(
        os.path.join(cfg.INPUT, "train_labels.csv"), parse_dates=["datetime"]
    )
    test = None




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
    Splits out contact_id into separate columns.
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


def fit_xgboost(cfg, X, y, params, add_suffix=""):
    oof_pred = np.zeros(len(y), dtype=np.float32)
    for fold in sorted(cfg.folds.unique()):
        if fold == -1:
            continue
        idx_train = cfg.folds != fold
        idx_valid = cfg.folds == fold
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]
        display(pd.Series(y_valid).value_counts())

        xgb_train = xgb.DMatrix(x_train, label=y_train)
        xgb_valid = xgb.DMatrix(x_valid, label=y_valid)
        evals = [(xgb_train, "train"), (xgb_valid, "eval")]

        model = xgb.train(
            params,
            xgb_train,
            num_boost_round=10_000,
            early_stopping_rounds=100,
            evals=evals,
            verbose_eval=100,
        )

        model_path = os.path.join(cfg.EXP_MODEL, f"xgb_fold{fold}{add_suffix}.model")
        model.save_model(model_path)

        if not torch.cuda.is_available():
            model = xgb.Booster()
            model.load_model(model_path)
            pred_i = model.predict(xgb.DMatrix(x_valid))
        else:
            model = ForestInference.load(
                model_path, output_class=True, model_type="xgboost"
            )
            pred_i = model.predict_proba(x_valid)[:, 1]

        oof_pred[idx_valid] = pred_i
        score = round(roc_auc_score(y_valid, pred_i), 5)
        print(f"Performance of the prediction: {score}\n")
        del model
        gc.collect()

    np.save(os.path.join(cfg.EXP_PREDS, f"oof_pred{add_suffix}"), oof_pred)
    score = round(roc_auc_score(y, oof_pred), 5)
    print(f"All Performance of the prediction: {score}")
    return oof_pred


def pred_xgboost(X, data_dir, add_suffix=""):
    model_paths = glob(os.path.join(data_dir, f"xgb_fold*{add_suffix}.model"))
    preds = []
    for path in model_paths:
        if not torch.cuda.is_available():
            model = xgb.Booster()
            model.load_model(path)
            pred = model.predict(xgb.DMatrix(X))
        else:
            model = ForestInference.load(path, output_class=True, model_type="xgboost")
            pred = model.predict_proba(X)[:, 1]
        preds.append(pred)
    if not preds:
        return np.zeros(len(X))
    return np.mean(np.column_stack(preds), axis=1)




## === cell 4
def create_features(
    df, tr_tracking, merge_col="step", use_cols=["x_position", "y_position"]
):
    output_cols = []
    df_combo = (
        df.astype({"nfl_player_id_1": "str"})
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id"] + use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_1"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .rename(columns={c: c + "_1" for c in use_cols})
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
        .rename(columns={c: c + "_2" for c in use_cols})
        .sort_values(["game_play", merge_col, "nfl_player_id_1", "nfl_player_id_2"])
        .reset_index(drop=True)
    )
    output_cols += [c + "_1" for c in use_cols]
    output_cols += [c + "_2" for c in use_cols]

    if ("x_position" in use_cols) and ("y_position" in use_cols):
        df_combo["distance"] = np.nan
        mask = df_combo["x_position_2"].notnull()
        distance = np.sqrt(
            (df_combo.loc[mask, "x_position_1"] - df_combo.loc[mask, "x_position_2"])
            ** 2
            + (df_combo.loc[mask, "y_position_1"] - df_combo.loc[mask, "y_position_2"])
            ** 2
        )
        df_combo.loc[mask, "distance"] = distance
        output_cols.append("distance")

    df_combo["G_flug"] = df_combo["nfl_player_id_2"] == "G"
    output_cols.append("G_flug")
    return df_combo, output_cols


test = expand_contact_id(sub)

use_cols = [
    "x_position",
    "y_position",
    "speed",
    "distance",
    "direction",
    "orientation",
    "acceleration",
    "sa",
]
train, feature_cols = create_features(train, tr_tracking, use_cols=use_cols)
test, _ = create_features(test, te_tracking, use_cols=use_cols)

if torch.cuda.is_available():
    train = train.to_pandas()
    test = test.to_pandas()

display(train.head())




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1063973867.py in <cell line: 0>()
     62 ]
     63 train, feature_cols = create_features(train, tr_tracking, use_cols=use_cols)
---> 64 test, _ = create_features(test, te_tracking, use_cols=use_cols)
     65 
     66 if torch.cuda.is_available():

/tmp/ipykernel_55/1063973867.py in create_features(df, tr_tracking, merge_col, use_cols)
      5     df_combo = (
      6         df.astype({"nfl_player_id_1": "str"})
----> 7         .merge(
      8             tr_tracking.astype({"nfl_player_id": "str"})[
      9                 ["game_play", merge_col, "nfl_player_id"] + use_cols

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in merge(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
  10830         from pandas.core.reshape.merge import merge
  10831 
> 10832         return merge(
  10833             self,
  10834             right,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    151 ) -> DataFrame:
    152     left_df = _validate_operand(left)
--> 153     right_df = _validate_operand(right)
    154     if how == "cross":
    155         return _cross_merge(

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _validate_operand(obj)
   2690         return obj.to_frame()
   2691     else:
-> 2692         raise TypeError(
   2693             f"Can only merge Series or DataFrame objects, a {type(obj)} was passed"
   2694         )

TypeError: Can only merge Series or DataFrame objects, a <class 'cudf.core.dataframe.DataFrame'> was passed

## === cell 5
train_X = train[feature_cols]
test_X = test[feature_cols]
train_y = train["contact"]

cfg.folds = get_groupkfold(train, "contact", "game_play", cfg.num_fold)
cfg.folds.to_csv(os.path.join(cfg.EXP_PREDS, "folds.csv"), index=False)

oof_pred = fit_xgboost(cfg, train_X, train_y, cfg.xgb_params, add_suffix="_xgb_1st")
sub_pred = pred_xgboost(test_X, cfg.EXP_MODEL, add_suffix="_xgb_1st")




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/3509878335.py in <cell line: 0>()
      1 train_X = train[feature_cols]
----> 2 test_X = test[feature_cols]
      3 train_y = train["contact"]
      4 
      5 cfg.folds = get_groupkfold(train, "contact", "game_play", cfg.num_fold)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index(['x_position_1', 'y_position_1', 'speed_1', 'distance_1', 'direction_1',\n       'orientation_1', 'acceleration_1', 'sa_1', 'x_position_2',\n       'y_position_2', 'speed_2', 'distance_2', 'direction_2', 'orientation_2',\n       'acceleration_2', 'sa_2', 'distance', 'G_flug'],\n      dtype='object')] are in the [columns]"

## === cell 6
def func(x_list):
    thresh = x_list[0]
    score = matthews_corrcoef(train["contact"], oof_pred > thresh)
    return -score


x0 = [0.5]
result = minimize(func, x0, method="nelder-mead")
cfg.threshold = result.x[0]
print("score:", round(matthews_corrcoef(train["contact"], oof_pred > cfg.threshold), 5))
print("threshold", round(cfg.threshold, 5))

test = add_contact_id(test)
test["contact"] = (sub_pred > cfg.threshold).astype(int)
test[["contact_id", "contact"]].to_csv("submission.csv", index=False)
display(test[["contact_id", "contact"]].head())

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3315263414.py in <cell line: 0>()
      6 
      7 x0 = [0.5]
----> 8 result = minimize(func, x0, method="nelder-mead")
      9 cfg.threshold = result.x[0]
     10 print("score:", round(matthews_corrcoef(train["contact"], oof_pred > cfg.threshold), 5))

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

/tmp/ipykernel_55/3315263414.py in func(x_list)
      1 def func(x_list):
      2     thresh = x_list[0]
----> 3     score = matthews_corrcoef(train["contact"], oof_pred > thresh)
      4     return -score
      5 

NameError: name 'oof_pred' is not defined
