# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.64313

# 6. Current score

0.01979

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.01923) has done: 'I fix the GPU/CPU branch inconsistencies by converting all feature engineering to run in pandas (CPU) so we avoid cudf/cupy implicit conversions and unsupported APIs (`to_array`, `min/max` on cudf Series, sklearn expecting numpy). I correct the XGBoost model loading/prediction logic (the current `Booster().load_model()` returns `None`, and `predict_proba` isn’t available on `Booster`) and ensure predictions are produced consistently via `Booster.predict`. I also fix the distance feature creation (it was being overwritten by tracking “distance” and then sometimes missing), make the threshold optimization use numpy arrays to avoid cupy errors, and ensure a valid `submission.csv` with required columns is written end-to-end. These changes are runtime/correctness fixes and should yield a reasonable baseline score without altering the intended modeling approach (XGB + engineered features + MCC threshold calibration).'
- What this solution (achieved 0.01886) has done: 'Your current score is far below the target, so the most likely issue is not “model strength” but a submission misalignment/logic bug that collapses MCC (e.g., predicting almost-all zeros because most rows get filtered out by the distance gate, and/or thresholding is tuned on a filtered subset but applied inconsistently). I keep your XGB + engineered features + MCC-threshold calibration intact, but make two minimal corrections: (1) apply the distance filter only when a valid distance exists (don’t treat NaN distance as “keep”), so you don’t flood the submission with untrained/ambiguous cases, and (2) compute the optimal MCC threshold on the full OOF (all rows) rather than only the filtered subset, so the decision threshold matches how you submit. These changes should move MCC upward toward your target without changing the core modeling approach or adding new modeling components. The script still run end-to-end and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.01979) has done: 'Your MCC is collapsing mainly because the distance gate is excluding the majority of candidate pairs, so most submission rows are forced to 0 and you only ever score on a small subset; to move toward the target, we should keep your exact XGB + feature engineering + threshold calibration, but stop gating by distance at inference/training time. I minimally change the pipeline to train/predict on all rows by imputing missing numeric features (including distance for “G” rows) with a constant sentinel, and keep the MCC threshold search on the full OOF (as you already do). This preserves your model/loop and features but prevents “almost-all-zero” submissions, which should raise MCC materially from ~0.02 toward your target. I also avoid repeatedly re-reading the huge helmet CSVs inside loops (same data, just different grouping), which keeps runtime safely under the 600s limit without changing semantics.'
- What this solution (achieved 0.01979) has done: 'Your current MCC is far below the target, which strongly suggests a correctness issue rather than pure model weakness; here the biggest one is that `train_y_all` is taken from the pre-feature `train` dataframe, but `oof_pred_all` is produced on `train_f` after merges/sorts/resets, so labels and predictions can be misaligned and MCC collapses. I minimally fix this by deriving `train_y_all` directly from `train_f["contact"]` (aligned with `oof_pred_all`) and using it consistently in threshold optimization and reporting. I also remove the duplicate/unused `test_out` reconstruction and instead write predictions directly into the original `sub` (preserving exact row order/ids), which avoids any accidental `contact_id` reformatting or ordering issues. Core model, features, folds, and training loop remain unchanged.'

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
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.03,
        "tree_method": "hist" if not torch.cuda.is_available() else "gpu_hist",
    }




## === cell 1
import os
import gc

import numpy as np
import pandas as pd

from scipy.optimize import minimize

from glob import glob
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, matthews_corrcoef

import xgboost as xgb
import torch

pd.set_option("mode.chained_assignment", None)




## === cell 2
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
    """
    Use xgb.Booster predict (not predict_proba) and correct model loading.
    """
    oof_pred = np.zeros(len(y), dtype=np.float32)

    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = X
    y_values = np.asarray(y).astype(np.float32)

    for fold in sorted(cfg.folds.unique()):
        if fold == -1:
            continue
        idx_train = (cfg.folds != fold).values
        idx_valid = (cfg.folds == fold).values

        x_train, y_train = X_values[idx_train], y_values[idx_train]
        x_valid, y_valid = X_values[idx_valid], y_values[idx_valid]

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

        pred_i = model.predict(xgb_valid)
        oof_pred[np.where(idx_valid)[0]] = pred_i.astype(np.float32)

        score = round(roc_auc_score(y_valid, pred_i), 5)
        print(f"fold {fold} AUC: {score}\n")

        del model
        gc.collect()

    np.save(os.path.join(cfg.EXP_PREDS, f"oof_pred{add_suffix}.npy"), oof_pred)
    score = round(roc_auc_score(y_values, oof_pred), 5)
    print(f"All AUC: {score}")
    return oof_pred


def pred_xgboost(X, data_dir, add_suffix=""):
    """
    Correct Booster loading and use Booster.predict on DMatrix.
    """
    model_paths = sorted(glob(os.path.join(data_dir, f"xgb_fold*{add_suffix}.model")))
    if len(model_paths) == 0:
        raise FileNotFoundError(
            f"No model files found in {data_dir} for suffix={add_suffix}"
        )

    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = X

    dtest = xgb.DMatrix(X_values)
    preds = []
    for mp in model_paths:
        booster = xgb.Booster()
        booster.load_model(mp)
        preds.append(booster.predict(dtest))
        del booster
    preds = np.mean(np.vstack(preds), axis=0)
    return preds




## === cell 4
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
test = expand_contact_id(sub.copy())

print(
    "train:",
    train.shape,
    "test(sub expanded):",
    test.shape,
    "tracking:",
    tr_tracking.shape,
)




## === cell 5
def create_features(df, tracking, merge_col="step", use_cols=None):
    """
    Bugfixes:
    - Avoid cudf-only APIs and ensure 'distance' feature is always created when possible.
    - Prevent name collision with tracking 'distance' by renaming tracking distance -> trk_distance.
    """
    if use_cols is None:
        use_cols = ["x_position", "y_position"]

    df = df.copy()
    tracking = tracking.copy()

    if "distance" in tracking.columns and "distance" in use_cols:
        tracking = tracking.rename(columns={"distance": "trk_distance"})
        use_cols = ["trk_distance" if c == "distance" else c for c in use_cols]

    df["nfl_player_id_1"] = df["nfl_player_id_1"].astype(str)
    df["nfl_player_id_2"] = df["nfl_player_id_2"].astype(str)
    tracking["nfl_player_id"] = tracking["nfl_player_id"].astype(str)

    base_cols = ["game_play", merge_col, "nfl_player_id"] + use_cols
    tsmall = tracking[base_cols]

    df_combo = (
        df.merge(
            tsmall,
            left_on=["game_play", merge_col, "nfl_player_id_1"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .rename(columns={c: c + "_1" for c in use_cols})
        .drop(columns=["nfl_player_id"])
        .merge(
            tsmall,
            left_on=["game_play", merge_col, "nfl_player_id_2"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .rename(columns={c: c + "_2" for c in use_cols})
        .drop(columns=["nfl_player_id"])
        .sort_values(["game_play", merge_col, "nfl_player_id_1", "nfl_player_id_2"])
        .reset_index(drop=True)
    )

    output_cols = [c + "_1" for c in use_cols] + [c + "_2" for c in use_cols]

    if ("x_position" in tracking.columns) and ("y_position" in tracking.columns):
        if ("x_position_1" in df_combo.columns) and (
            "x_position_2" in df_combo.columns
        ):
            dx = df_combo["x_position_1"] - df_combo["x_position_2"]
            dy = df_combo["y_position_1"] - df_combo["y_position_2"]
            df_combo["distance"] = np.sqrt(dx * dx + dy * dy).astype(np.float32)
            df_combo.loc[
                df_combo["x_position_1"].isna() | df_combo["x_position_2"].isna(),
                "distance",
            ] = np.nan
            df_combo.loc[
                df_combo["y_position_1"].isna() | df_combo["y_position_2"].isna(),
                "distance",
            ] = np.nan
            output_cols += ["distance"]

    df_combo["G_flug"] = (df_combo["nfl_player_id_2"] == "G").astype(np.int8)
    output_cols += ["G_flug"]

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
test, _ = create_features(test, te_tracking, use_cols=use_cols)

print("features:", len(feature_cols))
print(train[feature_cols].head())



## === cell 6
SENTINEL_FILL = -999.0

oof_pred_all = np.zeros(len(train), dtype=np.float32)

DISTANCE_THRESH = 2
cond_dis_train = (train["distance"].notna()) & (train["distance"] <= DISTANCE_THRESH)
cond_dis_test = (test["distance"].notna()) & (test["distance"] <= DISTANCE_THRESH)

train_f = train.reset_index(drop=True)
test_f = test.reset_index(drop=True)

print("number of features:", len(feature_cols))
print("number of train data (NO filtering):", len(train_f))
gc.collect()



## === cell 7
CLUSTERS = [10, 50, 100, 500]


def add_step_pct(df, cluster):
    smin = df["step"].min()
    smax = df["step"].max()
    denom = (smax - smin) if (smax != smin) else 1
    step_pct = cluster * (df["step"] - smin) / denom
    df["step_pct"] = np.ceil(step_pct).astype(np.int32)
    return df


helmet_train_all = pd.read_csv(os.path.join(cfg.INPUT, "train_baseline_helmets.csv"))
helmet_train_all.loc[helmet_train_all["view"] == "Endzone2", "view"] = "Endzone"
helmet_test_all = pd.read_csv(os.path.join(cfg.INPUT, "test_baseline_helmets.csv"))
helmet_test_all.loc[helmet_test_all["view"] == "Endzone2", "view"] = "Endzone"

helmet_train_all = helmet_train_all.rename(columns={"frame": "step"})
helmet_test_all = helmet_test_all.rename(columns={"frame": "step"})

for cluster in CLUSTERS:
    train_f = train_f.groupby("game_play", group_keys=False).apply(
        lambda x: add_step_pct(x, cluster)
    )
    test_f = test_f.groupby("game_play", group_keys=False).apply(
        lambda x: add_step_pct(x, cluster)
    )

    helmet_train = helmet_train_all.groupby("game_play", group_keys=False).apply(
        lambda x: add_step_pct(x, cluster)
    )
    helmet_test = helmet_test_all.groupby("game_play", group_keys=False).apply(
        lambda x: add_step_pct(x, cluster)
    )

    for helmet_view in ["Sideline", "Endzone"]:
        hv_train = helmet_train[helmet_train["view"] == helmet_view].copy()
        hv_test = helmet_test[helmet_test["view"] == helmet_view].copy()

        hv_train["helmet_id"] = (
            hv_train["game_play"]
            + "_"
            + hv_train["nfl_player_id"].astype(str)
            + "_"
            + hv_train["step_pct"].astype(str)
        )
        hv_test["helmet_id"] = (
            hv_test["game_play"]
            + "_"
            + hv_test["nfl_player_id"].astype(str)
            + "_"
            + hv_test["step_pct"].astype(str)
        )

        hv_train = (
            hv_train[["helmet_id", "left", "width", "top", "height"]]
            .groupby("helmet_id", as_index=False)
            .mean()
        )
        hv_test = (
            hv_test[["helmet_id", "left", "width", "top", "height"]]
            .groupby("helmet_id", as_index=False)
            .mean()
        )

        for player_ind in [1, 2]:
            train_f["helmet_id"] = (
                train_f["game_play"]
                + "_"
                + train_f["nfl_player_id_" + str(player_ind)].astype(str)
                + "_"
                + train_f["step_pct"].astype(str)
            )
            test_f["helmet_id"] = (
                test_f["game_play"]
                + "_"
                + test_f["nfl_player_id_" + str(player_ind)].astype(str)
                + "_"
                + test_f["step_pct"].astype(str)
            )

            train_f = train_f.merge(hv_train, how="left", on="helmet_id")
            test_f = test_f.merge(hv_test, how="left", on="helmet_id")

            rename_map = {
                i: i + f"_{helmet_view}_{cluster}_{player_ind}"
                for i in ["left", "width", "top", "height"]
            }
            train_f = train_f.rename(columns=rename_map)
            test_f = test_f.rename(columns=rename_map)

            train_f = train_f.drop(columns=["helmet_id"])
            test_f = test_f.drop(columns=["helmet_id"])

            feature_cols += list(rename_map.values())

        del hv_train, hv_test
        gc.collect()

cols = [i[:-2] for i in train_f.columns if i.endswith("_1") and i != "nfl_player_id_1"]
train_f[[i + "_diff" for i in cols]] = np.abs(
    train_f[[i + "_1" for i in cols]].values - train_f[[i + "_2" for i in cols]].values
)
test_f[[i + "_diff" for i in cols]] = np.abs(
    test_f[[i + "_1" for i in cols]].values - test_f[[i + "_2" for i in cols]].values
)
feature_cols += [i + "_diff" for i in cols]

cols2 = [
    "x_position",
    "y_position",
    "speed",
    "distance",
    "direction",
    "orientation",
    "acceleration",
    "sa",
]
cols2 = [
    c for c in cols2 if (c + "_1") in train_f.columns and (c + "_2") in train_f.columns
]
train_f[[i + "_prod" for i in cols2]] = np.abs(
    train_f[[i + "_1" for i in cols2]].values
    - train_f[[i + "_2" for i in cols2]].values
)
test_f[[i + "_prod" for i in cols2]] = np.abs(
    test_f[[i + "_1" for i in cols2]].values - test_f[[i + "_2" for i in cols2]].values
)
feature_cols += [i + "_prod" for i in cols2]

print("number of features:", len(feature_cols))
print("number of train data (NO filtering):", len(train_f))



## === cell 8
X_train = train_f[feature_cols].astype(np.float32).fillna(SENTINEL_FILL)
X_test = test_f[feature_cols].astype(np.float32).fillna(SENTINEL_FILL)

cfg.folds = get_groupkfold(train_f, "contact", "game_play", cfg.num_fold)
cfg.folds.to_csv(os.path.join(cfg.EXP_PREDS, "folds.csv"), index=False)

oof_pred_f = fit_xgboost(
    cfg,
    X_train,
    train_f["contact"].astype(np.int8),
    cfg.xgb_params,
    add_suffix="_xgb_1st",
)

oof_pred_all = oof_pred_f.astype(np.float32)
np.save("oof_pred.npy", oof_pred_all)

sub_pred_f = pred_xgboost(X_test, cfg.EXP_MODEL, add_suffix="_xgb_1st")



## === cell 9
train_y_all = train_f["contact"].astype(np.int8).values
oof_pred_all = oof_pred_all.astype(np.float32)


def func(x_list):
    thr = float(x_list[0])
    score = matthews_corrcoef(train_y_all, (oof_pred_all > thr).astype(np.int8))
    return -score


x0 = [0.5]
result = minimize(func, x0, method="nelder-mead")
cfg.threshold = float(result.x[0])

print(
    "OOF MCC (all rows):",
    round(
        matthews_corrcoef(train_y_all, (oof_pred_all > cfg.threshold).astype(np.int8)),
        5,
    ),
)
print("threshold", round(cfg.threshold, 5))

submission = sub.copy()
submission["contact"] = (sub_pred_f > cfg.threshold).astype(np.int8)

sub_path = "submission.csv"
submission[["contact_id", "contact"]].to_csv(sub_path, index=False)
print("Wrote:", sub_path, "rows:", len(submission))
print(submission.head())
print("Predicted positive rate:", float(submission["contact"].mean()))
