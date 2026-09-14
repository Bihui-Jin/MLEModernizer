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

0.61754

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
    INPUT = "/kaggle/input"
    OUTPUT_EXP = "exp"
    SUBMISSION = "./"
    DATASET = "/kaggle/input"
    EXP_MODEL = os.path.join(OUTPUT_EXP, "model")
    EXP_FIG = os.path.join(OUTPUT_EXP, "fig")
    EXP_PREDS = os.path.join(OUTPUT_EXP, "preds")
    num_fold = 5
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }


def setup(cfg):
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.INPUT = f"/kaggle/input/{cfg.COMPETITION}"
    cfg.EXP = cfg.NAME
    cfg.OUTPUT_EXP = cfg.NAME
    cfg.SUBMISSION = "./"
    cfg.DATASET = "/kaggle/input/"

    cfg.EXP_MODEL = os.path.join(cfg.EXP, "model")
    cfg.EXP_FIG = os.path.join(cfg.EXP, "fig")
    cfg.EXP_PREDS = os.path.join(cfg.EXP, "preds")

    for d in [cfg.EXP_MODEL, cfg.EXP_FIG, cfg.EXP_PREDS]:
        os.makedirs(d, exist_ok=True)

    return cfg




## === cell 1
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
            pred_i = model.predict(xgb_valid)  # returns probabilities
        else:
            model_gpu = ForestInference.load(
                model_path, output_class=True, model_type="xgboost"
            )
            pred_i = model_gpu.predict_proba(x_valid)[:, 1]

        oof_pred[x_valid.index] = pred_i
        score = round(roc_auc_score(y_valid, pred_i), 5)
        print(f"Performance of the prediction: {score}\n")
        del model
        gc.collect()

    np.save(os.path.join(cfg.EXP_PREDS, f"oof_pred{add_suffix}"), oof_pred)
    score = round(roc_auc_score(y, oof_pred), 5)
    print(f"All Performance of the prediction: {score}")
    return oof_pred


def pred_xgboost(X, data_dir, add_suffix=""):
    models = glob(os.path.join(data_dir, f"xgb_fold*{add_suffix}.model"))
    preds = []
    for model_path in models:
        if not torch.cuda.is_available():
            model = xgb.Booster()
            model.load_model(model_path)
            pred = model.predict(xgb.DMatrix(X))
        else:
            model = ForestInference.load(
                model_path, output_class=True, model_type="xgboost"
            )
            pred = model.predict_proba(X)[:, 1]
        preds.append(pred)
    preds = np.mean(np.array(preds), axis=0)
    return preds




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
    test = expand_contact_id(sub)

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
    test = cudf.DataFrame(expand_contact_id(sub))




## === cell 3
def create_features(
    df, tr_tracking, merge_col="step", use_cols=["x_position", "y_position"]
):
    df = df.astype(
        {"nfl_player_id_1": "str", "nfl_player_id_2": "str", "game_play": "str"}
    )
    tr_tracking = tr_tracking.astype({"nfl_player_id": "str", "game_play": "str"})

    df_combo = (
        df.merge(
            tr_tracking[["game_play", merge_col, "nfl_player_id"] + use_cols],
            left_on=["game_play", merge_col, "nfl_player_id_1"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .rename(columns={c: c + "_1" for c in use_cols})
        .drop("nfl_player_id", axis=1)
        .merge(
            tr_tracking[["game_play", merge_col, "nfl_player_id"] + use_cols],
            left_on=["game_play", merge_col, "nfl_player_id_2"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .rename(columns={c: c + "_2" for c in use_cols})
        .drop("nfl_player_id", axis=1)
        .sort_values(["game_play", merge_col, "nfl_player_id_1", "nfl_player_id_2"])
        .reset_index(drop=True)
    )
    output_cols = [c + "_1" for c in use_cols] + [c + "_2" for c in use_cols]

    if ("x_position" in use_cols) and ("y_position" in use_cols):
        idx = df_combo["x_position_2"].notnull()
        distance_arr = np.full(len(df_combo), np.nan, dtype=np.float32)
        tmp = np.sqrt(
            (df_combo.loc[idx, "x_position_1"] - df_combo.loc[idx, "x_position_2"]) ** 2
            + (df_combo.loc[idx, "y_position_1"] - df_combo.loc[idx, "y_position_2"])
            ** 2
        )
        distance_arr[idx] = tmp
        df_combo["distance"] = distance_arr
        output_cols.append("distance")

    df_combo["G_flug"] = df_combo["nfl_player_id_2"] == "G"
    output_cols.append("G_flug")
    return df_combo, output_cols


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
test, feature_cols = create_features(test, te_tracking, use_cols=use_cols)

if torch.cuda.is_available():
    train = train.to_pandas()
    test = test.to_pandas()

display(train.head())




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3161232817.py in <cell line: 0>()
     56     "sa",
     57 ]
---> 58 train, feature_cols = create_features(train, tr_tracking, use_cols=use_cols)
     59 test, feature_cols = create_features(test, te_tracking, use_cols=use_cols)
     60 

/tmp/ipykernel_55/3161232817.py in create_features(df, tr_tracking, merge_col, use_cols)
     37             ** 2
     38         )
---> 39         distance_arr[idx] = tmp
     40         df_combo["distance"] = distance_arr
     41         output_cols.append("distance")

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

## === cell 4
def add_step_pct(df):
    df["step_pct"] = (
        100 * (df["step"] - df["step"].min()) / (df["step"].max() - df["step"].min())
    )
    df["step_pct"] = df["step_pct"].apply(np.ceil).astype(np.int32)
    return df


train = (
    train.groupby("game_play", group_keys=False)
    .apply(add_step_pct)
    .reset_index(drop=True)
)
test = (
    test.groupby("game_play", group_keys=False)
    .apply(add_step_pct)
    .reset_index(drop=True)
)

for helmet_view in ["Sideline", "Endzone"]:
    helmet_train = pd.read_csv(
        "/kaggle/input/nfl-player-contact-detection/train_baseline_helmets.csv"
    )
    helmet_train.loc[helmet_train["view"] == "Endzone2", "view"] = "Endzone"
    helmet_test = pd.read_csv(
        "/kaggle/input/nfl-player-contact-detection/test_baseline_helmets.csv"
    )
    helmet_test.loc[helmet_test["view"] == "Endzone2", "view"] = "Endzone"

    helmet_train.rename(columns={"frame": "step"}, inplace=True)
    helmet_train = (
        helmet_train.groupby("game_play", group_keys=False)
        .apply(add_step_pct)
        .reset_index(drop=True)
    )
    helmet_test.rename(columns={"frame": "step"}, inplace=True)
    helmet_test = (
        helmet_test.groupby("game_play", group_keys=False)
        .apply(add_step_pct)
        .reset_index(drop=True)
    )

    helmet_train = helmet_train[helmet_train["view"] == helmet_view]
    helmet_test = helmet_test[helmet_test["view"] == helmet_view]

    helmet_train["helmet_id"] = (
        helmet_train["game_play"]
        + "_"
        + helmet_train["nfl_player_id"].astype(str)
        + "_"
        + helmet_train["step_pct"].astype(str)
    )
    helmet_test["helmet_id"] = (
        helmet_test["game_play"]
        + "_"
        + helmet_test["nfl_player_id"].astype(str)
        + "_"
        + helmet_test["step_pct"].astype(str)
    )

    helmet_train = (
        helmet_train[["helmet_id", "left", "width", "top", "height"]]
        .groupby("helmet_id")
        .mean()
        .reset_index()
    )
    helmet_test = (
        helmet_test[["helmet_id", "left", "width", "top", "height"]]
        .groupby("helmet_id")
        .mean()
        .reset_index()
    )
    for player_ind in [1, 2]:
        train["helmet_id"] = (
            train["game_play"]
            + "_"
            + train["nfl_player_id_" + str(player_ind)].astype(str)
            + "_"
            + train["step_pct"].astype(str)
        )
        test["helmet_id"] = (
            test["game_play"]
            + "_"
            + test["nfl_player_id_" + str(player_ind)].astype(str)
            + "_"
            + test["step_pct"].astype(str)
        )

        train = train.merge(helmet_train, how="left", on="helmet_id")
        test = test.merge(helmet_test, how="left", on="helmet_id")

        train.rename(
            columns={
                i: i + "_" + helmet_view + "_" + str(player_ind)
                for i in ["left", "width", "top", "height"]
            },
            inplace=True,
        )
        test.rename(
            columns={
                i: i + "_" + helmet_view + "_" + str(player_ind)
                for i in ["left", "width", "top", "height"]
            },
            inplace=True,
        )

        del train["helmet_id"], test["helmet_id"]
        gc.collect()

        feature_cols += [
            i + "_" + helmet_view + "_" + str(player_ind)
            for i in ["left", "width", "top", "height"]
        ]
    del helmet_train, helmet_test
    gc.collect()




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/1743817806.py in <cell line: 0>()
      9 train = (
     10     train.groupby("game_play", group_keys=False)
---> 11     .apply(add_step_pct)
     12     .reset_index(drop=True)
     13 )

/usr/local/lib/python3.11/dist-packages/cudf/utils/performance_tracking.py in wrapper(*args, **kwargs)
     49                     )
     50                 )
---> 51             return func(*args, **kwargs)
     52 
     53     return wrapper

/usr/local/lib/python3.11/dist-packages/cudf/core/groupby/groupby.py in apply(self, func, engine, include_groups, *args, **kwargs)
   1974 
   1975         if engine == "auto":
-> 1976             if _can_be_jitted(grouped_values, func, args):
   1977                 engine = "jit"
   1978             else:

/usr/local/lib/python3.11/dist-packages/cudf/core/udf/groupby_utils.py in _can_be_jitted(frame, func, args)
    225     )
    226     try:
--> 227         _get_udf_return_type(dataframe_group_type, func, args)
    228         return True
    229     except (UDFError, TypingError):

/usr/local/lib/python3.11/dist-packages/cudf/utils/performance_tracking.py in wrapper(*args, **kwargs)
     49                     )
     50                 )
---> 51             return func(*args, **kwargs)
     52 
     53     return wrapper

/usr/local/lib/python3.11/dist-packages/cudf/core/udf/utils.py in _get_udf_return_type(argty, func, args)
    102     # needed here.
    103     with _CUDFNumbaConfig():
--> 104         ptx, output_type = cudautils.compile_udf(func, compile_sig)
    105 
    106     if not isinstance(output_type, MaskedType):

/usr/local/lib/python3.11/dist-packages/cudf/utils/cudautils.py in compile_udf(udf, type_signature)
    124     # We haven't compiled a function like this before, so need to fall back to
    125     # compilation with Numba
--> 126     ptx_code, return_type = cuda.compile_ptx_for_current_device(
    127         udf, type_signature, device=True
    128     )

/usr/local/lib/python3.11/dist-packages/numba_cuda/numba/cuda/compiler.py in compile_ptx_for_current_device(pyfunc, sig, debug, lineinfo, device, fastmath, opt, abi, abi_info)
    566     device's compute capabilility. See :func:`compile_ptx`."""
    567     cc = get_current_device().compute_capability
--> 568     return compile_ptx(pyfunc, sig, debug=debug, lineinfo=lineinfo,
    569                        device=device, fastmath=fastmath, cc=cc, opt=opt,
    570                        abi=abi, abi_info=abi_info)

/usr/local/lib/python3.11/dist-packages/numba_cuda/numba/cuda/compiler.py in compile_ptx(pyfunc, sig, debug, lineinfo, device, fastmath, cc, opt, abi, abi_info)
    555     with the Numba ABI, rather than :func:`compile`'s default of compiling a
    556     device function with the C ABI."""
--> 557     return compile(pyfunc, sig, debug=debug, lineinfo=lineinfo, device=device,
    558                    fastmath=fastmath, cc=cc, opt=opt, abi=abi,
    559                    abi_info=abi_info, output='ptx')

/usr/local/lib/python3.11/dist-packages/numba/core/compiler_lock.py in _acquire_compile_lock(*args, **kwargs)
     33         def _acquire_compile_lock(*args, **kwargs):
     34             with self:
---> 35                 return func(*args, **kwargs)
     36         return _acquire_compile_lock
     37 

/usr/local/lib/python3.11/dist-packages/numba_cuda/numba/cuda/compiler.py in compile(pyfunc, sig, debug, lineinfo, device, fastmath, cc, opt, abi, abi_info, output)
    508 
    509     cc = cc or config.CUDA_DEFAULT_PTX_CC
--> 510     cres = compile_cuda(pyfunc, return_type, args, debug=debug,
    511                         lineinfo=lineinfo, fastmath=fastmath,
    512                         nvvm_options=nvvm_options, cc=cc)

/usr/local/lib/python3.11/dist-packages/numba/core/compiler_lock.py in _acquire_compile_lock(*args, **kwargs)
     33         def _acquire_compile_lock(*args, **kwargs):
     34             with self:
---> 35                 return func(*args, **kwargs)
     36         return _acquire_compile_lock
     37 

/usr/local/lib/python3.11/dist-packages/numba_cuda/numba/cuda/compiler.py in compile_cuda(pyfunc, return_type, args, debug, lineinfo, inline, fastmath, nvvm_options, cc, max_registers, lto)
    224     from numba.core.target_extension import target_override
    225     with target_override('cuda'):
--> 226         cres = compiler.compile_extra(typingctx=typingctx,
    227                                       targetctx=targetctx,
    228                                       func=pyfunc,

/usr/local/lib/python3.11/dist-packages/numba/core/compiler.py in compile_extra(typingctx, targetctx, func, args, return_type, flags, locals, library, pipeline_class)
    742     pipeline = pipeline_class(typingctx, targetctx, library,
    743                               args, return_type, flags, locals)
--> 744     return pipeline.compile_extra(func)
    745 
    746 

/usr/local/lib/python3.11/dist-packages/numba/core/compiler.py in compile_extra(self, func)
    436         self.state.lifted = ()
    437         self.state.lifted_from = None
--> 438         return self._compile_bytecode()
    439 
    440     def compile_ir(self, func_ir, lifted=(), lifted_from=None):

/usr/local/lib/python3.11/dist-packages/numba/core/compiler.py in _compile_bytecode(self)
    504         """
    505         assert self.state.func_ir is None
--> 506         return self._compile_core()
    507 
    508     def _compile_ir(self):

/usr/local/lib/python3.11/dist-packages/numba/core/compiler.py in _compile_core(self)
    479                     if (utils.use_new_style_errors() and not
    480                             isinstance(e, errors.NumbaError)):
--> 481                         raise e
    482 
    483                     self.state.status.fail_reason = e

/usr/local/lib/python3.11/dist-packages/numba/core/compiler.py in _compile_core(self)
    470                 res = None
    471                 try:
--> 472                     pm.run(self.state)
    473                     if self.state.cr is not None:
    474                         break

/usr/local/lib/python3.11/dist-packages/numba/core/compiler_machinery.py in run(self, state)
    362                 if (utils.use_new_style_errors() and not
    363                         isinstance(e, errors.NumbaError)):
--> 364                     raise e
    365                 msg = "Failed in %s mode pipeline (step: %s)" % \
    366                     (self.pipeline_name, pass_desc)

/usr/local/lib/python3.11/dist-packages/numba/core/compiler_machinery.py in run(self, state)
    354                 pass_inst = _pass_registry.get(pss).pass_inst
    355                 if isinstance(pass_inst, CompilerPass):
--> 356                     self._runPass(idx, pass_inst, state)
    357                 else:
    358                     raise BaseException("Legacy pass in use")

/usr/local/lib/python3.11/dist-packages/numba/core/compiler_lock.py in _acquire_compile_lock(*args, **kwargs)
     33         def _acquire_compile_lock(*args, **kwargs):
     34             with self:
---> 35                 return func(*args, **kwargs)
     36         return _acquire_compile_lock
     37 

/usr/local/lib/python3.11/dist-packages/numba/core/compiler_machinery.py in _runPass(self, index, pss, internal_state)
    309                 mutated |= check(pss.run_initialization, internal_state)
    310             with SimpleTimer() as pass_time:
--> 311                 mutated |= check(pss.run_pass, internal_state)
    312             with SimpleTimer() as finalize_time:
    313                 mutated |= check(pss.run_finalizer, internal_state)

/usr/local/lib/python3.11/dist-packages/numba/core/compiler_machinery.py in check(func, compiler_state)
    271 
    272         def check(func, compiler_state):
--> 273             mangled = func(compiler_state)
    274             if mangled not in (True, False):
    275                 msg = ("CompilerPass implementations should return True/False. "

/usr/local/lib/python3.11/dist-packages/numba/core/typed_passes.py in run_pass(self, state)
    110                               % (state.func_id.func_name,)):
    111             # Type inference
--> 112             typemap, return_type, calltypes, errs = type_inference_stage(
    113                 state.typingctx,
    114                 state.targetctx,

/usr/local/lib/python3.11/dist-packages/numba/core/typed_passes.py in type_inference_stage(typingctx, targetctx, interp, args, return_type, locals, raise_errors)
     91         infer.build_constraint()
     92         # return errors in case of partial typing
---> 93         errs = infer.propagate(raise_errors=raise_errors)
     94         typemap, restype, calltypes = infer.unify(raise_errors=raise_errors)
     95 

/usr/local/lib/python3.11/dist-packages/numba/core/typeinfer.py in propagate(self, raise_errors)
   1081             # Errors can appear when the type set is incomplete; only
   1082             # raise them when there is no progress anymore.
-> 1083             errors = self.constraints.propagate(self)
   1084             newtoken = self.get_state_token()
   1085             self.debug.propagate_finished()

/usr/local/lib/python3.11/dist-packages/numba/core/typeinfer.py in propagate(self, typeinfer)
    180                         errors.append(utils.chain_exception(new_exc, e))
    181                     elif utils.use_new_style_errors():
--> 182                         raise e
    183                     else:
    184                         msg = ("Unknown CAPTURED_ERRORS style: "

/usr/local/lib/python3.11/dist-packages/numba/core/typeinfer.py in propagate(self, typeinfer)
    158                                                    lineno=loc.line):
    159                 try:
--> 160                     constraint(typeinfer)
    161                 except ForceLiteralArg as e:
    162                     errors.append(e)

/usr/local/lib/python3.11/dist-packages/numba/core/typeinfer.py in __call__(self, typeinfer)
    474             typevars = typeinfer.typevars
    475             for ty in typevars[self.value.name].get():
--> 476                 sig = typeinfer.context.resolve_static_getitem(
    477                     value=ty, index=self.index,
    478                 )

/usr/local/lib/python3.11/dist-packages/numba/core/typing/context.py in resolve_static_getitem(self, value, index)
    319         args = value, index
    320         kws = ()
--> 321         return self.resolve_function_type("static_getitem", args, kws)
    322 
    323     def resolve_static_setitem(self, target, index, value):

/usr/local/lib/python3.11/dist-packages/numba/core/typing/context.py in resolve_function_type(self, func, args, kws)
    207 
    208         # Check builtin functions
--> 209         res = self._resolve_builtin_function_type(func, args, kws)
    210 
    211         # Re-raise last_exception if no function type has been found

/usr/local/lib/python3.11/dist-packages/numba/core/typing/context.py in _resolve_builtin_function_type(self, func, args, kws)
    224                 for support_literals in [True, False]:
    225                     if support_literals:
--> 226                         res = defn.apply(args, kws)
    227                     else:
    228                         fixedargs = [types.unliteral(a) for a in args]

/usr/local/lib/python3.11/dist-packages/numba/core/typing/templates.py in apply(self, args, kws)
    348     def apply(self, args, kws):
    349         generic = getattr(self, "generic")
--> 350         sig = generic(args, kws)
    351         # Enforce that *generic()* must return None or Signature
    352         if sig is not None:

/usr/local/lib/python3.11/dist-packages/numba/core/typing/arraydecl.py in generic(self, args, kws)
    616         if isinstance(record, types.Record) and isinstance(idx, str):
    617             if idx not in record.fields:
--> 618                 raise KeyError(f"Field '{idx}' was not found in record with "
    619                                f"fields {tuple(record.fields.keys())}")
    620             ret = record.typeof(idx)

KeyError: "Field 'step_pct' was not found in record with fields ('step', 'nfl_player_id_1', 'contact')"

## === cell 5
train[feature_cols] = train[feature_cols].astype(np.float16)
test[feature_cols] = test[feature_cols].astype(np.float16)
gc.collect()

train_X = train[feature_cols]
test_X = test[feature_cols]
train_y = train["contact"]
cfg.folds = get_groupkfold(train, "contact", "game_play", cfg.num_fold)
cfg.folds.to_csv(os.path.join(cfg.EXP_PREDS, "folds.csv"), index=False)

oof_pred = fit_xgboost(cfg, train_X, train_y, cfg.xgb_params, add_suffix="_xgb_1st")
sub_pred = pred_xgboost(test_X, cfg.EXP_MODEL, add_suffix="_xgb_1st")




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2374670368.py in <cell line: 0>()
----> 1 train[feature_cols] = train[feature_cols].astype(np.float16)
      2 test[feature_cols] = test[feature_cols].astype(np.float16)
      3 gc.collect()
      4 
      5 train_X = train[feature_cols]

NameError: name 'feature_cols' is not defined

## === cell 6
def func(x_list):
    thresh = x_list[0]
    score = matthews_corrcoef(train["contact"], (oof_pred > thresh).astype(int))
    return -score


x0 = [0.5]
result = minimize(func, x0, method="nelder-mead")
cfg.threshold = result.x[0]
print(
    "score:",
    round(
        matthews_corrcoef(train["contact"], (oof_pred > cfg.threshold).astype(int)), 5
    ),
)
print("threshold", round(cfg.threshold, 5))

test = add_contact_id(test)
test["contact"] = (sub_pred > cfg.threshold).astype(int)
test[["contact_id", "contact"]].to_csv("submission.csv", index=False)
display(test[["contact_id", "contact"]].head())

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3218315278.py in <cell line: 0>()
      6 
      7 x0 = [0.5]
----> 8 result = minimize(func, x0, method="nelder-mead")
      9 cfg.threshold = result.x[0]
     10 print(

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

/tmp/ipykernel_55/3218315278.py in func(x_list)
      1 def func(x_list):
      2     thresh = x_list[0]
----> 3     score = matthews_corrcoef(train["contact"], (oof_pred > thresh).astype(int))
      4     return -score
      5 

NameError: name 'oof_pred' is not defined
