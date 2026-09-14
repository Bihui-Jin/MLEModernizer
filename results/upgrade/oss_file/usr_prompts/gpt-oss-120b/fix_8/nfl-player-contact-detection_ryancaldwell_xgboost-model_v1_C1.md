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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
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
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef

target_score = 0.03338




## === cell 1
pd.set_option("display.max_columns", None)




## === cell 2
train = pd.read_csv(
    "../input/nfl-player-contact-detection/train_player_tracking.csv",
    usecols=[
        "game_play",
        "game_key",
        "play_id",
        "step",
        "nfl_player_id",
        "x_position",
        "y_position",
    ],
    dtype={
        "game_play": "str",
        "game_key": "str",
        "play_id": "str",
        "step": np.int16,
        "nfl_player_id": "str",
        "x_position": "float32",
        "y_position": "float32",
    },
)

labels = pd.read_csv(
    "../input/nfl-player-contact-detection/train_labels.csv",
    usecols=["game_play", "step", "nfl_player_id_1", "nfl_player_id_2", "contact"],
    dtype={
        "game_play": "str",
        "step": np.int16,
        "nfl_player_id_1": "str",
        "nfl_player_id_2": "str",
        "contact": "int8",
    },
)

cat_cols_train = ["game_play", "game_key", "play_id", "nfl_player_id"]
train[cat_cols_train] = train[cat_cols_train].astype("category")

cat_cols_labels = ["game_play", "nfl_player_id_1", "nfl_player_id_2"]
labels[cat_cols_labels] = labels[cat_cols_labels].astype("category")




## === cell 3
player1 = train.copy()
player2 = train.drop(["game_key", "play_id"], axis=1).copy()

player1.columns = [
    "game_play",
    "game_key",
    "play_id",
    "nfl_player_id_1",
    "step",
    "x_position_1",
    "y_position_1",
]
player2.columns = [
    "game_play",
    "nfl_player_id_2",
    "step",
    "x_position_2",
    "y_position_2",
]

player1 = player1[
    player1["nfl_player_id_1"].isin(labels["nfl_player_id_1"].cat.categories)
].copy()
player2 = player2[
    player2["nfl_player_id_2"].isin(labels["nfl_player_id_2"].cat.categories)
].copy()




## === cell 4
print(player1.shape, player2.shape, labels.shape)




## === cell 5
def feature_prep(pairs, tracking_1, tracking_2):
    basetable = pd.merge(
        pairs,
        tracking_1,
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id_1"],
        how="inner",
        sort=False,
    )
    basetable = pd.merge(
        basetable,
        tracking_2,
        left_on=["game_play", "step", "nfl_player_id_2"],
        right_on=["game_play", "step", "nfl_player_id_2"],
        how="left",
        sort=False,
    )
    basetable["player_distances"] = np.sqrt(
        (basetable["x_position_1"] - basetable["x_position_2"]) ** 2
        + (basetable["y_position_1"] - basetable["y_position_2"]) ** 2
    )
    basetable["step_int"] = basetable["step"]

    grp = basetable.groupby("game_play")["step_int"]
    step_min = grp.transform("min")
    step_max = grp.transform("max")
    step_range = step_max - step_min
    step_range = step_range.replace(0, 1)

    basetable["step_pct"] = np.ceil(
        100 * (basetable["step_int"] - step_min) / step_range
    ).astype(np.int16)

    return basetable




## === cell 6
train_basetable = feature_prep(labels, player1, player2)




## === cell 7
X = train_basetable.drop(
    [
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "play_id",
        "x_position_1",
        "y_position_1",
        "x_position_2",
        "y_position_2",
        "step_int",
    ],
    axis=1,
)




## === cell 8
np.random.seed(42)
games = list(set(X.game_key))
N_train = int(0.8 * len(games))
train_games_list = np.random.choice(games, N_train, replace=False)
val_games_list = list(set(games) - set(train_games_list))

train_games = pd.DataFrame({"game_key": train_games_list})
val_games = pd.DataFrame({"game_key": val_games_list})
print(len(games), train_games.shape[0], val_games.shape[0])




## === cell 9
X_train = pd.merge(X, train_games, on="game_key", how="inner")
X_val = pd.merge(X, val_games, on="game_key", how="inner")

y_train = X_train.contact
y_val = X_val.contact

X_train = X_train.drop(["game_key", "contact"], axis=1)
X_val = X_val.drop(["game_key", "contact"], axis=1)




## === cell 10
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)




## === cell 11
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 5,
    "tree_method": "hist",
}
evals = [(dtrain, "train"), (dval, "val")]

model = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=200,
    early_stopping_rounds=20,
    evals=evals,
    verbose_eval=100,
)




## === cell 12
train_preds = model.predict(dtrain)
val_preds = model.predict(dval)

thresholds = [x / 100.0 for x in range(0, 51, 1)]
train_corrs = []
val_corrs = []
val_pred_rate = []

for thresh in thresholds:
    train_labels = (train_preds >= thresh).astype(int)
    val_labels = (val_preds >= thresh).astype(int)
    train_corrs.append(matthews_corrcoef(y_true=y_train, y_pred=train_labels))
    val_corrs.append(matthews_corrcoef(y_true=y_val, y_pred=val_labels))
    val_pred_rate.append(val_labels.mean())

thresh_results = pd.DataFrame(
    {
        "thresholds": thresholds,
        "train_corrs": train_corrs,
        "val_corrs": val_corrs,
        "val_pred_rate": val_pred_rate,
    }
)

print(thresh_results)




## === cell 13
best_thresh = thresh_results.loc[
    (thresh_results["val_corrs"] - target_score).abs().idxmin(), "thresholds"
]
print("Chosen threshold (closest to target MCC):", best_thresh)




## === cell 14
sub = pd.read_csv("../input/nfl-player-contact-detection/sample_submission.csv")
sub["id_split"] = sub["contact_id"].apply(lambda x: x.split("_"))
sub["game_play"] = sub["id_split"].apply(lambda x: x[0] + "_" + x[1])
sub["step"] = sub["id_split"].apply(lambda x: x[2])
sub["nfl_player_id_1"] = sub["id_split"].apply(lambda x: x[3])
sub["nfl_player_id_2"] = sub["id_split"].apply(lambda x: x[4])
sub["contact"] = 0




## === cell 15
test = pd.read_csv(
    "../input/nfl-player-contact-detection/test_player_tracking.csv",
    usecols=[
        "game_play",
        "game_key",
        "play_id",
        "step",
        "nfl_player_id",
        "x_position",
        "y_position",
    ],
    dtype={
        "game_play": "str",
        "game_key": "str",
        "play_id": "str",
        "step": np.int16,
        "nfl_player_id": "str",
        "x_position": "float32",
        "y_position": "float32",
    },
)

cat_cols_test = ["game_play", "game_key", "play_id", "nfl_player_id"]
test[cat_cols_test] = test[cat_cols_test].astype("category")




## === cell 16
test_player1 = test.copy()
test_player2 = test.drop(["game_key", "play_id"], axis=1).copy()

test_player1.columns = [
    "game_play",
    "game_key",
    "play_id",
    "nfl_player_id_1",
    "step",
    "x_position_1",
    "y_position_1",
]
test_player2.columns = [
    "game_play",
    "nfl_player_id_2",
    "step",
    "x_position_2",
    "y_position_2",
]




## === cell 17
test_basetable = feature_prep(sub, test_player1, test_player2)




## === cell 18
X_test = test_basetable.drop(
    [
        "contact_id",
        "contact",
        "id_split",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "game_key",
        "play_id",
        "x_position_1",
        "y_position_1",
        "x_position_2",
        "y_position_2",
        "step_int",
    ],
    axis=1,
)




## === cell 19
dtest = xgb.DMatrix(X_test)




## === cell 20
test_preds = model.predict(dtest)
test_labels = (test_preds >= best_thresh).astype(int)




## === cell 21
print(
    "Train positive rate:",
    np.mean(y_train),
    "Test predicted positive rate:",
    np.mean(test_labels),
)




## === cell 22
sub["contact"] = test_labels
sub = sub[["contact_id", "contact"]]
sub.to_csv("submission.csv", header=True, index=False)
