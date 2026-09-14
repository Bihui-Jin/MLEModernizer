# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd

from glob import glob

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, matthews_corrcoef

import xgboost as xgb
import torch

if torch.cuda.is_available():
    import cudf
    from cuml import (
        ForestInference,
    )  # noqa: F401 (kept to preserve original environment intent)

import random


def setup(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

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
    s = df["contact_id"].astype(str)
    df["game_play"] = s.str[:12]
    parts = s.str.rsplit("_", n=3, expand=True)
    df["step"] = parts[1].astype("int")
    df["nfl_player_id_1"] = parts[2]
    df["nfl_player_id_2"] = parts[3]
    return df


def get_groupkfold(train, target_col, group_col, n_splits):
    kf = GroupKFold(n_splits=n_splits)
    generator = kf.split(train, train[target_col], train[group_col])
    fold_series = []
    for fold, (idx_train, idx_valid) in enumerate(generator):
        fold_series.append(pd.Series(fold, index=idx_valid))
    fold_series = pd.concat(fold_series).sort_index()
    return fold_series


def _make_dmatrix(X, y=None, ref=None, params=None):
    tree_method = (params or {}).get("tree_method", "hist")
    use_quantile = tree_method in ("hist", "gpu_hist")
    if y is None:
        if use_quantile:
            return xgb.QuantileDMatrix(X, ref=ref)
        return xgb.DMatrix(X)
    else:
        if use_quantile:
            return xgb.QuantileDMatrix(X, label=y, ref=ref)
        return xgb.DMatrix(X, label=y)


def fit_xgboost(cfg, X, y, params, add_suffix=""):
    oof_pred = np.zeros(len(y), dtype=np.float32)

    params = dict(params)
    params.setdefault(
        "nthread", int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 4))
    )
    params.setdefault("verbosity", 1)
    if torch.cuda.is_available():
        params.setdefault("predictor", "gpu_predictor")

    for fold in sorted(cfg.folds.unique()):
        if fold == -1:
            continue

        idx_train = (cfg.folds != fold).to_numpy()
        idx_valid = (cfg.folds == fold).to_numpy()

        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        dtrain = _make_dmatrix(x_train, y=y_train, ref=None, params=params)
        dvalid = _make_dmatrix(x_valid, y=y_valid, ref=dtrain, params=params)

        evals = [(dtrain, "train"), (dvalid, "eval")]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=10_000,
            evals=evals,
            early_stopping_rounds=100,
            verbose_eval=200,
        )

        best_it = model.best_iteration
        if best_it is None or best_it < 0:
            best_round = model.num_boosted_rounds()
        else:
            best_round = int(best_it) + 1

        model_path = os.path.join(cfg.EXP_MODEL, f"xgb_fold{fold}{add_suffix}.model")
        model.save_model(model_path)

        pred_i = model.predict(dvalid, iteration_range=(0, best_round)).astype(
            np.float32, copy=False
        )
        oof_pred[np.where(idx_valid)[0]] = pred_i

        score = round(roc_auc_score(y_valid, pred_i), 5)
        print(f"Fold {fold} AUC (sanity): {score}  best_round={best_round}\n")

        del dtrain, dvalid, x_train, y_train, x_valid, y_valid, pred_i, model
        gc.collect()

    np.save(os.path.join(cfg.EXP_PREDS, f"oof_pred{add_suffix}.npy"), oof_pred)
    score = round(roc_auc_score(y, oof_pred), 5)
    print(f"All AUC (sanity): {score}")
    return oof_pred


def pred_xgboost(X, data_dir, add_suffix="", params=None):
    model_paths = sorted(glob(os.path.join(data_dir, f"xgb_fold*{add_suffix}.model")))
    if len(model_paths) == 0:
        raise FileNotFoundError(
            f"No trained models found in {data_dir} with suffix {add_suffix}"
        )

    params = params or {}
    tree_method = params.get("tree_method", "hist")
    use_inplace = hasattr(xgb.Booster(), "inplace_predict")

    xgb_test = None
    if not use_inplace:
        xgb_test = _make_dmatrix(
            X, y=None, ref=None, params={"tree_method": tree_method}
        )

    preds = np.zeros(X.shape[0], dtype=np.float32)
    for mp in model_paths:
        booster = xgb.Booster()
        booster.load_model(mp)

        best_it = booster.best_iteration
        if best_it is not None and best_it >= 0:
            it_range = (0, int(best_it) + 1)
            if use_inplace:
                p = booster.inplace_predict(X, iteration_range=it_range).astype(
                    np.float32, copy=False
                )
            else:
                p = booster.predict(xgb_test, iteration_range=it_range).astype(
                    np.float32, copy=False
                )
        else:
            if use_inplace:
                p = booster.inplace_predict(X).astype(np.float32, copy=False)
            else:
                p = booster.predict(xgb_test).astype(np.float32, copy=False)

        preds += p
        del booster, p

    preds /= float(len(model_paths))
    return preds




## === cell 1
class Config:
    AUTHOR = "colum2131"
    NAME = "NFLC-" + "Exp001-simple-xgb-baseline"
    COMPETITION = "nfl-player-contact-detection"

    seed = 42
    num_fold = 5

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.03,
        "tree_method": "hist" if not torch.cuda.is_available() else "gpu_hist",
    }


cfg = setup(Config)

tracking_usecols = [
    "game_play",
    "step",
    "nfl_player_id",
    "x_position",
    "y_position",
    "speed",
    "distance",
    "direction",
    "orientation",
    "acceleration",
    "sa",
]
train_usecols = [
    "contact_id",
    "game_play",
    "datetime",
    "step",
    "nfl_player_id_1",
    "nfl_player_id_2",
    "contact",
]

tracking_dtypes = {
    "game_play": "string",
    "step": "int16",
    "nfl_player_id": "string",
    "x_position": "float32",
    "y_position": "float32",
    "speed": "float32",
    "distance": "float32",
    "direction": "float32",
    "orientation": "float32",
    "acceleration": "float32",
    "sa": "float32",
}
train_dtypes = {
    "contact_id": "string",
    "game_play": "string",
    "step": "int16",
    "nfl_player_id_1": "string",
    "nfl_player_id_2": "string",
    "contact": "int8",
}

if torch.cuda.is_available():
    tr_tracking = cudf.read_csv(
        os.path.join(cfg.INPUT, "train_player_tracking.csv"),
        usecols=tracking_usecols,
    ).to_pandas()
    te_tracking = cudf.read_csv(
        os.path.join(cfg.INPUT, "test_player_tracking.csv"),
        usecols=tracking_usecols,
    ).to_pandas()
    train = cudf.read_csv(
        os.path.join(cfg.INPUT, "train_labels.csv"),
        usecols=train_usecols,
        parse_dates=["datetime"],
    ).to_pandas()
    tr_tracking = tr_tracking.astype(tracking_dtypes, copy=False)
    te_tracking = te_tracking.astype(tracking_dtypes, copy=False)
    train = train.astype(train_dtypes, copy=False)
else:
    tr_tracking = pd.read_csv(
        os.path.join(cfg.INPUT, "train_player_tracking.csv"),
        usecols=tracking_usecols,
        dtype=tracking_dtypes,
    )
    te_tracking = pd.read_csv(
        os.path.join(cfg.INPUT, "test_player_tracking.csv"),
        usecols=tracking_usecols,
        dtype=tracking_dtypes,
    )
    train = pd.read_csv(
        os.path.join(cfg.INPUT, "train_labels.csv"),
        usecols=train_usecols,
        parse_dates=["datetime"],
        dtype=train_dtypes,
    )

sub = pd.read_csv(os.path.join(cfg.INPUT, "sample_submission.csv"))
test = expand_contact_id(sub.copy())

print(train.shape, test.shape, tr_tracking.shape, te_tracking.shape)




## === cell 2
def create_features(
    df, tracking, merge_col="step", use_cols=("x_position", "y_position")
):
    use_cols = list(use_cols)
    output_cols = []

    df = df.copy()
    df["nfl_player_id_1"] = df["nfl_player_id_1"].astype("string")
    df["nfl_player_id_2"] = df["nfl_player_id_2"].astype("string")

    tr = tracking[["game_play", merge_col, "nfl_player_id"] + use_cols].copy()
    tr["nfl_player_id"] = tr["nfl_player_id"].astype("string")

    tr = tr.set_index(["game_play", merge_col, "nfl_player_id"], drop=True)

    key1 = pd.MultiIndex.from_frame(
        df[["game_play", merge_col, "nfl_player_id_1"]].rename(
            columns={"nfl_player_id_1": "nfl_player_id"}
        )
    )
    key2 = pd.MultiIndex.from_frame(
        df[["game_play", merge_col, "nfl_player_id_2"]].rename(
            columns={"nfl_player_id_2": "nfl_player_id"}
        )
    )

    feat1 = tr.reindex(key1).reset_index(drop=True)
    feat2 = tr.reindex(key2).reset_index(drop=True)

    feat1.columns = [c + "_1" for c in use_cols]
    feat2.columns = [c + "_2" for c in use_cols]

    df_combo = pd.concat([df.reset_index(drop=True), feat1, feat2], axis=1)

    output_cols += [c + "_1" for c in use_cols]
    output_cols += [c + "_2" for c in use_cols]

    if ("x_position" in use_cols) and ("y_position" in use_cols):
        x1 = df_combo["x_position_1"].to_numpy()
        y1 = df_combo["y_position_1"].to_numpy()
        x2 = df_combo["x_position_2"].to_numpy()
        y2 = df_combo["y_position_2"].to_numpy()
        mask = ~np.isnan(x1) & ~np.isnan(x2) & ~np.isnan(y1) & ~np.isnan(y2)

        distance_arr = np.full(len(df_combo), np.nan, dtype=np.float32)
        dx = (x1[mask] - x2[mask]).astype(np.float32, copy=False)
        dy = (y1[mask] - y2[mask]).astype(np.float32, copy=False)
        distance_arr[mask] = np.sqrt(dx * dx + dy * dy, dtype=np.float32)
        df_combo["distance"] = distance_arr
        if "distance" not in output_cols:
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
test, feature_cols = create_features(test, te_tracking, use_cols=use_cols)

print("n_features:", len(feature_cols))
print("train cols ok:", set(feature_cols).issubset(train.columns))




## === cell 3
def add_step_pct_vectorized(df):
    out = df.copy()
    out["step"] = out["step"].astype(np.int32, copy=False)

    step = out["step"]
    smin = (
        out.groupby("game_play", sort=False)["step"]
        .transform("min")
        .astype(np.int32, copy=False)
    )
    smax = (
        out.groupby("game_play", sort=False)["step"]
        .transform("max")
        .astype(np.int32, copy=False)
    )

    denom = (smax - smin).astype(np.int32, copy=False)
    denom = denom.mask(denom == 0, 1)

    step_pct = 100.0 * (step - smin) / denom
    out["step_pct"] = np.ceil(step_pct).astype(np.int32)
    return out


train = add_step_pct_vectorized(train)
test = add_step_pct_vectorized(test)

helmet_usecols = [
    "game_play",
    "view",
    "frame",
    "nfl_player_id",
    "left",
    "width",
    "top",
    "height",
]
helmet_dtypes = {
    "game_play": "string",
    "view": "category",
    "frame": "int32",
    "nfl_player_id": "string",
    "left": "float32",
    "width": "float32",
    "top": "float32",
    "height": "float32",
}

helmet_train = pd.read_csv(
    os.path.join(cfg.INPUT, "train_baseline_helmets.csv"),
    usecols=helmet_usecols,
    dtype=helmet_dtypes,
)
helmet_train.loc[helmet_train["view"] == "Endzone2", "view"] = "Endzone"
helmet_test = pd.read_csv(
    os.path.join(cfg.INPUT, "test_baseline_helmets.csv"),
    usecols=helmet_usecols,
    dtype=helmet_dtypes,
)
helmet_test.loc[helmet_test["view"] == "Endzone2", "view"] = "Endzone"

helmet_train = helmet_train.rename(columns={"frame": "step"})
helmet_test = helmet_test.rename(columns={"frame": "step"})

helmet_train = add_step_pct_vectorized(helmet_train)
helmet_test = add_step_pct_vectorized(helmet_test)

helmet_train["nfl_player_id"] = helmet_train["nfl_player_id"].astype("string")
helmet_test["nfl_player_id"] = helmet_test["nfl_player_id"].astype("string")


def _build_helmet_id(
    game_play_s: pd.Series, pid_s: pd.Series, step_pct_s: pd.Series
) -> pd.Series:
    gp = game_play_s.astype(str).to_numpy()
    pid = pid_s.astype(str).to_numpy()
    sp = step_pct_s.astype(str).to_numpy()
    return pd.Series(
        np.char.add(np.char.add(np.char.add(gp, "_"), pid), np.char.add("_", sp)),
        index=game_play_s.index,
    )


helmet_train["helmet_id"] = _build_helmet_id(
    helmet_train["game_play"], helmet_train["nfl_player_id"], helmet_train["step_pct"]
)
helmet_test["helmet_id"] = _build_helmet_id(
    helmet_test["game_play"], helmet_test["nfl_player_id"], helmet_test["step_pct"]
)

agg_cols = ["left", "width", "top", "height"]

for player_ind in [1, 2]:
    train[f"helmet_id_{player_ind}"] = _build_helmet_id(
        train["game_play"], train[f"nfl_player_id_{player_ind}"], train["step_pct"]
    )
    test[f"helmet_id_{player_ind}"] = _build_helmet_id(
        test["game_play"], test[f"nfl_player_id_{player_ind}"], test["step_pct"]
    )

needed_train_ids = pd.Index(
    pd.unique(
        pd.concat([train["helmet_id_1"], train["helmet_id_2"]], ignore_index=True)
    )
)
needed_test_ids = pd.Index(
    pd.unique(pd.concat([test["helmet_id_1"], test["helmet_id_2"]], ignore_index=True))
)

helmet_train = helmet_train.loc[helmet_train["helmet_id"].isin(needed_train_ids)]
helmet_test = helmet_test.loc[helmet_test["helmet_id"].isin(needed_test_ids)]

helmet_train_agg = (
    helmet_train[["helmet_id", "view"] + agg_cols]
    .groupby(["view", "helmet_id"], sort=False, observed=True)[agg_cols]
    .mean()
)
helmet_test_agg = (
    helmet_test[["helmet_id", "view"] + agg_cols]
    .groupby(["view", "helmet_id"], sort=False, observed=True)[agg_cols]
    .mean()
)


def _join_helmet_features_join(base_df, helmet_agg, view_name, player_ind, agg_cols):
    hv = helmet_agg.xs(view_name, level=0, drop_level=True)  # index: helmet_id
    key = base_df[f"helmet_id_{player_ind}"]
    feat = hv.reindex(key)
    feat = feat.set_index(base_df.index)
    feat.columns = [f"{c}_{view_name}_{player_ind}" for c in agg_cols]
    out = base_df.join(feat)
    new_cols = list(feat.columns)
    return out, new_cols


for helmet_view in ["Sideline", "Endzone"]:
    for player_ind in [1, 2]:
        train, new_cols = _join_helmet_features_join(
            train, helmet_train_agg, helmet_view, player_ind, agg_cols
        )
        test, _ = _join_helmet_features_join(
            test, helmet_test_agg, helmet_view, player_ind, agg_cols
        )
        feature_cols += new_cols

del (
    helmet_train,
    helmet_test,
    helmet_train_agg,
    helmet_test_agg,
    needed_train_ids,
    needed_test_ids,
)
gc.collect()

train = train.drop(columns=["helmet_id_1", "helmet_id_2"])
test = test.drop(columns=["helmet_id_1", "helmet_id_2"])

print("final n_features:", len(feature_cols))




## === cell 4
use_float16 = (
    False  # keep deterministic + faster pipeline; XGBoost will handle internally
)
dtype = np.float32

train_feats = train[feature_cols].astype(dtype, copy=False)
test_feats = test[feature_cols].astype(dtype, copy=False)

med = train_feats.median(numeric_only=True)
train_feats = train_feats.fillna(med)
test_feats = test_feats.fillna(med)

train_X = np.asarray(train_feats.to_numpy(copy=False), dtype=np.float32, order="C")
test_X = np.asarray(test_feats.to_numpy(copy=False), dtype=np.float32, order="C")
train_y = train["contact"].astype(np.int8).to_numpy()

gc.collect()

neg = float((train_y == 0).sum())
pos = float((train_y == 1).sum())
scale_pos_weight = neg / max(pos, 1.0)
cfg.xgb_params = dict(cfg.xgb_params)
cfg.xgb_params["scale_pos_weight"] = scale_pos_weight
print(
    "scale_pos_weight:",
    round(scale_pos_weight, 4),
    "pos_rate:",
    round(pos / (pos + neg), 6),
)

cfg.folds = get_groupkfold(train, "contact", "game_play", cfg.num_fold)
cfg.folds.to_csv(os.path.join(cfg.EXP_PREDS, "folds.csv"), index=False)

oof_pred = fit_xgboost(cfg, train_X, train_y, cfg.xgb_params, add_suffix="_xgb_1st")
sub_pred = pred_xgboost(
    test_X, cfg.EXP_MODEL, add_suffix="_xgb_1st", params=cfg.xgb_params
)




## === cell 5
def _mcc_from_confmat(tp, fp, fn, tn):
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom <= 0:
        return 0.0
    return (tp * tn - fp * fn) / np.sqrt(denom)


def _best_mcc_threshold_grid(y_true, y_prob, n_grid=200):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    qs = np.linspace(0.0, 1.0, n_grid)
    ts = np.quantile(y_prob, qs)
    ts = np.unique(np.clip(ts, 0.0, 1.0))

    order = np.argsort(y_prob, kind="mergesort")
    y_sorted = y_true[order]
    p_sorted = y_prob[order]

    total_pos = int(y_true.sum())
    total_neg = int(len(y_true) - total_pos)

    tp = total_pos
    fp = total_neg
    fn = 0
    tn = 0

    best_t = 0.5
    best_mcc = -1.0

    idx = 0
    n = len(y_true)

    for t in ts:
        while idx < n and p_sorted[idx] <= t:
            if y_sorted[idx] == 1:
                tp -= 1
                fn += 1
            else:
                fp -= 1
                tn += 1
            idx += 1

        mcc = _mcc_from_confmat(tp, fp, fn, tn)
        if mcc > best_mcc:
            best_mcc = float(mcc)
            best_t = float(t)

    return best_t, float(best_mcc)


t_oof, mcc_oof = _best_mcc_threshold_grid(train_y, oof_pred, n_grid=300)
print("OOF-best threshold (grid):", round(t_oof, 5), "OOF MCC:", round(mcc_oof, 5))

oof_pos_rate = float((oof_pred > t_oof).mean())
print("oof_pos_rate @ oof-best:", round(oof_pos_rate, 6))

t_rate = float(np.quantile(sub_pred.astype(np.float64), 1.0 - oof_pos_rate))
t_rate = float(np.clip(t_rate, 0.0, 1.0))
print("threshold from test pos-rate matching:", round(t_rate, 5))

cfg.threshold = float(0.8 * t_oof + 0.2 * t_rate)

test_pos_rate = float((sub_pred > cfg.threshold).mean())
min_test_pos_rate = max(1e-4, 0.25 * oof_pos_rate)
if test_pos_rate < min_test_pos_rate:
    cfg.threshold = t_rate
    test_pos_rate = float((sub_pred > cfg.threshold).mean())
    print("guardrail triggered -> using t_rate as final threshold")

print(
    "final threshold:",
    round(cfg.threshold, 5),
    "test_pos_rate:",
    round(test_pos_rate, 6),
)

overall_mcc = matthews_corrcoef(train_y, (oof_pred > cfg.threshold).astype(int))
print("OOF MCC @ final threshold (for sanity):", round(overall_mcc, 5))

sub_out = sub.copy()
sub_out["contact"] = (sub_pred > cfg.threshold).astype(int)
sub_out[["contact_id", "contact"]].to_csv("submission.csv", index=False)

print(sub_out.head())
print("saved to submission.csv, shape:", sub_out.shape)
print("submission positives:", int(sub_out["contact"].sum()))
