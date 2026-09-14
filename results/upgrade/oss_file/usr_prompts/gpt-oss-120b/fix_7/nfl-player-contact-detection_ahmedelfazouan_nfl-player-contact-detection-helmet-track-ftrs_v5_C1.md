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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from sklearn.metrics import matthews_corrcoef, roc_auc_score
import xgboost as xgb
import torch
from glob import glob


class Config:
    COMPETITION = "nfl-player-contact-detection"
    NAME = "exp"
    INPUT = "/kaggle/input"
    OUTPUT_EXP = "exp"
    SUBMISSION = "./"
    DATASET = "/kaggle/input"
    num_fold = 5
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",  # will become gpu_hist if CUDA is present
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
    if torch.cuda.is_available():
        cfg.xgb_params["tree_method"] = "gpu_hist"
    return cfg




## === cell 1
cfg = setup(Config)

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




## === cell 2
def add_contact_id(df):
    df["contact_id"] = (
        df["game_play"].astype(str)
        + "_"
        + df["step"].astype(str)
        + "_"
        + df["nfl_player_id_1"].astype(str)
        + "_"
        + df["nfl_player_id_2"].astype(str)
    )
    return df


def expand_contact_id(df):
    df["game_play"] = df["contact_id"].str.split("_").str[0]
    df["step"] = df["contact_id"].str.split("_").str[1].astype(int)
    df["nfl_player_id_1"] = df["contact_id"].str.split("_").str[2]
    df["nfl_player_id_2"] = df["contact_id"].str.split("_").str[3]
    return df


def get_groupkfold(train, target_col, group_col, n_splits):
    kf = GroupKFold(n_splits=n_splits)
    folds = []
    for fold, (_, idx_valid) in enumerate(
        kf.split(train, train[target_col], train[group_col])
    ):
        folds.append(pd.Series(fold, index=idx_valid))
    return pd.concat(folds).sort_index()


def fit_xgboost(cfg, X, y, params, add_suffix=""):
    oof_pred = np.zeros(len(y), dtype=np.float32)
    for fold in sorted(cfg.folds.unique()):
        if fold == -1:
            continue
        idx_train = cfg.folds != fold
        idx_valid = cfg.folds == fold
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        dtrain = xgb.DMatrix(x_train, label=y_train)
        dvalid = xgb.DMatrix(x_valid, label=y_valid)
        evals = [(dtrain, "train"), (dvalid, "eval")]

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=10_000,
            early_stopping_rounds=100,
            evals=evals,
            verbose_eval=100,
        )

        model_path = os.path.join(cfg.EXP_MODEL, f"xgb_fold{fold}{add_suffix}.model")
        model.save_model(model_path)

        pred_i = model.predict(xgb.DMatrix(x_valid))
        oof_pred[x_valid.index] = pred_i

        print(f"Fold {fold} AUC:", round(roc_auc_score(y_valid, pred_i), 5))
        del model
        gc.collect()

    np.save(os.path.join(cfg.EXP_PREDS, f"oof_pred{add_suffix}"), oof_pred)
    print("Overall AUC:", round(roc_auc_score(y, oof_pred), 5))
    return oof_pred


def pred_xgboost(X, data_dir, add_suffix=""):
    models = glob(os.path.join(data_dir, f"xgb_fold*{add_suffix}.model"))
    preds = []
    dmatrix = xgb.DMatrix(X)
    for model_path in models:
        model = xgb.Booster()
        model.load_model(model_path)
        preds.append(model.predict(dmatrix))
    return np.mean(np.column_stack(preds), axis=1)


def create_features(df, tr_tracking, merge_col="step", use_cols=None):
    if use_cols is None:
        use_cols = []
    df = df.astype(
        {
            "nfl_player_id_1": "category",
            "nfl_player_id_2": "category",
            "game_play": "category",
        }
    )
    tr_tracking = tr_tracking.astype(
        {"nfl_player_id": "category", "game_play": "category"}
    )

    left = df.merge(
        tr_tracking[["game_play", merge_col, "nfl_player_id"] + use_cols],
        left_on=["game_play", merge_col, "nfl_player_id_1"],
        right_on=["game_play", merge_col, "nfl_player_id"],
        how="left",
    )
    left = left.rename(columns={c: f"{c}_1" for c in use_cols}).drop(
        "nfl_player_id", axis=1
    )

    df_combo = left.merge(
        tr_tracking[["game_play", merge_col, "nfl_player_id"] + use_cols],
        left_on=["game_play", merge_col, "nfl_player_id_2"],
        right_on=["game_play", merge_col, "nfl_player_id"],
        how="left",
    )
    df_combo = df_combo.rename(columns={c: f"{c}_2" for c in use_cols}).drop(
        "nfl_player_id", axis=1
    )

    output_cols = [f"{c}_1" for c in use_cols] + [f"{c}_2" for c in use_cols]

    if "x_position" in use_cols and "y_position" in use_cols:
        idx = df_combo["x_position_2"].notnull()
        dist = np.full(len(df_combo), np.nan, dtype=np.float32)
        dist[idx] = np.sqrt(
            (df_combo.loc[idx, "x_position_1"] - df_combo.loc[idx, "x_position_2"]) ** 2
            + (df_combo.loc[idx, "y_position_1"] - df_combo.loc[idx, "y_position_2"])
            ** 2
        )
        df_combo["distance"] = dist
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
test, _ = create_features(
    test, te_tracking, use_cols=use_cols
)  # feature_cols unchanged


def add_step_pct(df):
    step_min = df["step"].min()
    step_max = df["step"].max()
    pct = 100 * (df["step"] - step_min) / (step_max - step_min)
    df["step_pct"] = np.ceil(pct).astype(np.int32)
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

helmet_train_full = pd.read_csv(
    "/kaggle/input/nfl-player-contact-detection/train_baseline_helmets.csv"
)
helmet_test_full = pd.read_csv(
    "/kaggle/input/nfl-player-contact-detection/test_baseline_helmets.csv"
)
helmet_train_full.loc[helmet_train_full["view"] == "Endzone2", "view"] = "Endzone"
helmet_test_full.loc[helmet_test_full["view"] == "Endzone2", "view"] = "Endzone"

helmet_train_full.rename(columns={"frame": "step"}, inplace=True)
helmet_train_full = (
    helmet_train_full.groupby("game_play", group_keys=False)
    .apply(add_step_pct)
    .reset_index(drop=True)
)
helmet_test_full.rename(columns={"frame": "step"}, inplace=True)
helmet_test_full = (
    helmet_test_full.groupby("game_play", group_keys=False)
    .apply(add_step_pct)
    .reset_index(drop=True)
)

for helmet_view in ["Sideline", "Endzone"]:
    helmet_train = helmet_train_full[helmet_train_full["view"] == helmet_view].copy()
    helmet_test = helmet_test_full[helmet_test_full["view"] == helmet_view].copy()

    helmet_train["helmet_id"] = (
        helmet_train["game_play"].astype(str)
        + "_"
        + helmet_train["nfl_player_id"].astype(str)
        + "_"
        + helmet_train["step_pct"].astype(str)
    )
    helmet_test["helmet_id"] = (
        helmet_test["game_play"].astype(str)
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
            train["game_play"].astype(str)
            + "_"
            + train[f"nfl_player_id_{player_ind}"].astype(str)
            + "_"
            + train["step_pct"].astype(str)
        )
        test["helmet_id"] = (
            test["game_play"].astype(str)
            + "_"
            + test[f"nfl_player_id_{player_ind}"].astype(str)
            + "_"
            + test["step_pct"].astype(str)
        )

        train = train.merge(helmet_train, how="left", on="helmet_id")
        test = test.merge(helmet_test, how="left", on="helmet_id")

        rename_map = {
            i: f"{i}_{helmet_view}_{player_ind}"
            for i in ["left", "width", "top", "height"]
        }
        train.rename(columns=rename_map, inplace=True)
        test.rename(columns=rename_map, inplace=True)

        train.drop(columns=["helmet_id"], inplace=True)
        test.drop(columns=["helmet_id"], inplace=True)

        feature_cols += list(rename_map.values())
    gc.collect()
del helmet_train_full, helmet_test_full
gc.collect()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
IntCastingNaNError                        Traceback (most recent call last)
/tmp/ipykernel_55/2158740118.py in <cell line: 0>()
    166 test = (
    167     test.groupby("game_play", group_keys=False)
--> 168     .apply(add_step_pct)
    169     .reset_index(drop=True)
    170 )

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in apply(self, func, include_groups, *args, **kwargs)
   1822         with option_context("mode.chained_assignment", None):
   1823             try:
-> 1824                 result = self._python_apply_general(f, self._selected_obj)
   1825                 if (
   1826                     not isinstance(self.obj, Series)

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in _python_apply_general(self, f, data, not_indexed_same, is_transform, is_agg)
   1883             data after applying f
   1884         """
-> 1885         values, mutated = self._grouper.apply_groupwise(f, data, self.axis)
   1886         if not_indexed_same is None:
   1887             not_indexed_same = mutated

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py in apply_groupwise(self, f, data, axis)
    917             # group might be modified
    918             group_axes = group.axes
--> 919             res = f(group)
    920             if not mutated and not _is_indexed_like(res, group_axes, axis):
    921                 mutated = True

/tmp/ipykernel_55/2158740118.py in add_step_pct(df)
    155     step_max = df["step"].max()
    156     pct = 100 * (df["step"] - step_min) / (step_max - step_min)
--> 157     df["step_pct"] = np.ceil(pct).astype(np.int32)
    158     return df
    159 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in astype(self, dtype, copy, errors)
   6641         else:
   6642             # else, only a single dtype is given
-> 6643             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
   6644             res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   6645             return res.__finalize__(self, method="astype")

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in astype(self, dtype, copy, errors)
    428             copy = False
    429 
--> 430         return self.apply(
    431             "astype",
    432             dtype=dtype,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in astype(self, dtype, copy, errors, using_cow, squeeze)
    756             values = values[0, :]  # type: ignore[call-overload]
    757 
--> 758         new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    759 
    760         new_values = maybe_coerce_values(new_values)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array_safe(values, dtype, copy, errors)
    235 
    236     try:
--> 237         new_values = astype_array(values, dtype, copy=copy)
    238     except (ValueError, TypeError):
    239         # e.g. _astype_nansafe can fail on object-dtype of strings

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array(values, dtype, copy)
    180 
    181     else:
--> 182         values = _astype_nansafe(values, dtype, copy=copy)
    183 
    184     # in pandas we don't store numpy str dtypes, so convert to object

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_nansafe(arr, dtype, copy, skipna)
     99 
    100     elif np.issubdtype(arr.dtype, np.floating) and dtype.kind in "iu":
--> 101         return _astype_float_to_int_nansafe(arr, dtype, copy)
    102 
    103     elif arr.dtype == object:

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_float_to_int_nansafe(values, dtype, copy)
    143     """
    144     if not np.isfinite(values).all():
--> 145         raise IntCastingNaNError(
    146             "Cannot convert non-finite values (NA or inf) to integer"
    147         )

IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer

## === cell 3
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




## === cell 4
def func(x_list):
    thresh = x_list[0]
    return -matthews_corrcoef(train["contact"], (oof_pred > thresh).astype(int))


x0 = [0.5]
result = minimize(func, x0, method="nelder-mead")
cfg.threshold = result.x[0]

print(
    "Best MCC:",
    round(
        matthews_corrcoef(train["contact"], (oof_pred > cfg.threshold).astype(int)),
        5,
    ),
)
print("Threshold:", round(cfg.threshold, 5))

test = add_contact_id(test)
test["contact"] = (sub_pred > cfg.threshold).astype(int)
test[["contact_id", "contact"]].to_csv("submission.csv", index=False)
display(test[["contact_id", "contact"]].head())

## --- ERROR in outputing the csv:
Invalid submission: Submission contact_id's inconsistent with answers contact_id's
