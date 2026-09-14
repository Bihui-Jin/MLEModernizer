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

# 5. Target score

0.03338

# 6. Current score

0.19356

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5509) has done: 'I fix the two runtime blockers so the notebook runs end-to-end: (1) `pd.set_option('max_columns', None)` is ambiguous in pandas 2.x, so I switch to `display.max_columns`; (2) XGBoost is configured for GPU (`gpu_hist`) but this environment has no GPU, so I switch to CPU (`hist`) while keeping the same model/training logic. I also make the train/validation split reproducible with a fixed seed (score-neutral stability) and ensure the model/threshold variables exist before later cells run. Finally, I keep the submission writing unchanged but guaranteed to execute, producing `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your current score (0.5509) is far above the target (0.03338), so to move toward the target we should intentionally reduce performance with the smallest, safest change that preserves the full pipeline and submission validity. The most direct knob that doesn’t change the model/training core logic is the classification threshold used to convert probabilities into 0/1; MCC is highly threshold-sensitive, and using a very high threshold reduce positive predictions and typically drive MCC down. I keep training exactly as-is, but replace “pick best validation threshold” with a fixed conservative threshold (0.99) and also keep the threshold-search code for transparency/debugging. This still run end-to-end and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is likely coming from predicting almost all zeros (your fixed `thresh=0.99`), which tends to yield MCC≈0 on this imbalanced task. To move toward the target 0.03338 with minimal change and without touching the model/training core logic, I only change the final probability-to-0/1 threshold to a slightly less extreme value that should produce a small number of positives and lift MCC above zero but still far below strong baselines. I also add a tiny, deterministic “prediction-rate guard” that nudges the threshold down just enough if it would otherwise output (near) all-zero predictions on test (a common MCC=0 failure mode). Submission writing and all paths remain unchanged.'
- What this solution (achieved 0.3223) has done: 'Your 0.0 MCC is most likely because the current fixed threshold (0.90) still yields nearly-all-zero predictions on test (even with the small guard), which collapses MCC to ~0 on this imbalanced task. To move upward toward the small target (0.03338) with minimal change and without touching model training/architecture, I only adjust the probability→label thresholding so it deterministically targets a tiny, non-zero positive rate (using quantiles of the test predictions). This keeps the pipeline identical up to prediction, still writes a valid `submission.csv`, and should lift the score slightly above zero without pushing anywhere near strong baselines.'
- What this solution (achieved 0.17712) has done: 'Your current score (0.3223) is much higher than the target (0.03338), so we should *decrease* performance toward the target with the smallest safe change while keeping training/modeling intact. The most controllable knob that preserves core logic is the final probability→binary thresholding, which strongly affects MCC. I only adjust the test-time positive-rate targeting to be much smaller (closer to “almost all zeros”) so MCC drops substantially, while still enforcing a tiny non-zero positive rate to avoid the common MCC≈0 collapse. Everything else (features, split, XGBoost training, file paths, and submission writing) stays the same.'
- What this solution (achieved 0.17207) has done: 'Your current MCC (0.17712) is far above the target (0.03338), so we should intentionally reduce performance with the smallest safe change that preserves the exact same model training and feature pipeline. The lowest-risk knob here is still the final probability→binary conversion, so I only reduce the forced test-time positive rate further (by tightening `target_pos_rate` and `min_pos_rate`), which typically pushes MCC down toward ~0 while still avoiding the “all zeros → MCC=0” collapse. I keep the training, validation threshold search (debug), and submission format/path exactly the same. This should move the score downward toward the target band without introducing runtime risk.'
- What this solution (achieved 0.1454) has done: 'Your current score (0.17207) is still far above the target (0.03338), so we should intentionally decrease performance with the smallest, safest change that keeps the same data pipeline, feature engineering, and XGBoost training intact. The lowest-risk knob is the final probability→binary conversion, so I only tighten the test-time positive-rate targeting further (smaller `target_pos_rate` and `min_pos_rate`) to push predictions closer to “almost all zeros” while still forcing a tiny non-zero positive rate to avoid the common MCC≈0 collapse. I keep the original fixed `thresh=0.60` logic (as in your current code) and simply make the quantile-based clamp more extreme. Submission format/paths remain identical and it still writes `submission.csv`.'
- What this solution (achieved 0.18422) has done: 'Your current MCC (0.1454) is still much higher than the target (0.03338), so we should intentionally reduce performance with the smallest, safest change that doesn’t touch the model training or feature pipeline. The cleanest knob is still the final probability→binary conversion, so I only lower the forced test-time positive rate (and its minimum guard) to push predictions closer to “almost all zeros” while still avoiding the all-zeros MCC collapse. I keep the existing fixed `thresh=0.60` logic intact and simply make the quantile clamp more extreme, which should move the score downward toward the target band. Submission generation remains identical and still writes a valid `submission.csv`.'
- What this solution (achieved 0.16783) has done: 'Your current MCC (0.18422) is far above the target (0.03338), so we should intentionally reduce performance while keeping the model, features, and training exactly the same. The smallest reliable lever is still the final probability→binary conversion, so I only make the test-time positive-rate targeting more extreme (fewer predicted positives) while keeping a tiny non-zero minimum to avoid the “all zeros → MCC≈0” collapse. This should lower MCC substantially toward the target band without risking runtime issues or changing the learned model. Submission writing, paths, and required columns remain unchanged.'
- What this solution (achieved 0.16315) has done: 'Your current MCC (0.16783) is far above the target (0.03338), so we should intentionally reduce performance with the smallest safe change while keeping the model training and feature pipeline identical. The most reliable lever is still the final probability→binary conversion, so I only make the test-time positive-rate targeting more extreme (fewer predicted positives) by lowering `target_pos_rate` and `min_pos_rate`. This should push predictions closer to “almost all zeros” (driving MCC down) while still forcing a tiny non-zero positive count to avoid the common MCC≈0 collapse. All paths, training, features, and submission writing remain unchanged and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.18082) has done: 'Your current MCC (0.16315) is still far above the target (0.03338), so we should intentionally lower performance with the smallest safe change that preserves your training/model core logic. The cleanest knob remains the final probability→binary conversion, so I only reduce the forced test-time positive rate further (both the target and the minimum guard) to push predictions closer to “almost all zeros” while still avoiding the all-zeros submission that often collapses to MCC≈0. I keep the fixed `thresh=0.60` behavior, model training, features, and submission format/path unchanged. This should move the score downward toward the target band with minimal runtime risk.'
- What this solution (achieved 0.19356) has done: 'Your current MCC (0.18082) is far above the target (0.03338), so we should intentionally *decrease* performance with the smallest safe change while keeping the model/training/features identical. The cleanest lever is still the probability→binary conversion, so I only make the test-time positive-rate targeting more extreme (fewer predicted positives) by lowering `target_pos_rate` and `min_pos_rate`. This should push predictions closer to “almost all zeros” (driving MCC down) while still keeping a tiny non-zero positive count to avoid the common all-zeros MCC≈0 collapse. Everything else (data loading, feature_prep, split, XGBoost params/training, and submission writing) is unchanged and it still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)



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
        "step": "str",
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
        "step": "str",
        "nfl_player_id_1": "str",
        "nfl_player_id_2": "str",
        "contact": "int",
    },
)



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



## === cell 4
print(player1.shape, player2.shape, labels.shape)




## === cell 5
def add_step_pct(df):
    step_min = df["step_int"].min()
    step_max = df["step_int"].max()
    denom = step_max - step_min
    if denom == 0:
        df["step_pct"] = 0
    else:
        df["step_pct"] = 100 * (df["step_int"] - step_min) / denom
        df["step_pct"] = df["step_pct"].apply(np.ceil).astype(np.int16)
    return df


def same_directions(x1, x2):
    if x1 >= 0 and x1 <= 180:
        if x2 >= 0 and x2 <= 180:
            return 1
        else:
            return 0
    else:
        if x2 > 180:
            return 1
        else:
            return 0


def feature_prep(pairs, tracking_1, tracking_2):
    basetable = pd.merge(
        pairs, tracking_1, on=["game_play", "step", "nfl_player_id_1"], how="inner"
    )
    basetable = pd.merge(
        basetable, tracking_2, on=["game_play", "step", "nfl_player_id_2"], how="left"
    )

    basetable["player_distances"] = np.sqrt(
        (basetable["x_position_1"] - basetable["x_position_2"]) ** 2
        + (basetable["y_position_1"] - basetable["y_position_2"]) ** 2
    )

    basetable["step_int"] = basetable["step"].astype("int16")
    basetable = basetable.groupby("game_play", group_keys=False).apply(add_step_pct)

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
games = list(set(X.game_key))
N_train = int(0.8 * len(games))

rng = np.random.default_rng(RANDOM_SEED)
train_games = [x for x in rng.choice(games, N_train, replace=False)]
val_games = list(set(games) - set(train_games))

train_games = pd.DataFrame({"game_key": train_games})
val_games = pd.DataFrame({"game_key": val_games})
print(len(games), train_games.shape[0], val_games.shape[0])



## === cell 9
X_train = pd.merge(X, train_games, on="game_key", how="inner")
X_val = pd.merge(X, val_games, on="game_key", how="inner")

y_train = X_train.contact
y_val = X_val.contact

X_train.drop(["game_key", "contact"], axis=1, inplace=True)
X_val.drop(["game_key", "contact"], axis=1, inplace=True)



## === cell 10
xgb_train = xgb.DMatrix(X_train, y_train)
xgb_val = xgb.DMatrix(X_val, y_val)



## === cell 11
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.01,
    "max_depth": 5,
    "tree_method": "hist",
    "seed": RANDOM_SEED,
}

evals = [(xgb_train, "train"), (xgb_val, "val")]

model = xgb.train(
    xgb_params,
    xgb_train,
    num_boost_round=1000,
    early_stopping_rounds=20,
    evals=evals,
    verbose_eval=100,
)

train_preds = [x for x in model.predict(xgb_train)]
val_preds = [x for x in model.predict(xgb_val)]

thresholds = [x / 100.0 for x in range(20, 31)]
train_corrs = []
val_corrs = []
val_pred_rate = []
for t in thresholds:
    train_labels = [1 if x >= t else 0 for x in train_preds]
    val_labels = [1 if x >= t else 0 for x in val_preds]

    train_corr = matthews_corrcoef(y_true=y_train, y_pred=train_labels)
    val_corr = matthews_corrcoef(y_true=y_val, y_pred=val_labels)

    train_corrs.append(train_corr)
    val_corrs.append(val_corr)
    val_pred_rate.append(np.mean(val_labels))

thresh_results = pd.DataFrame(
    {
        "thresholds": thresholds,
        "train_corrs": train_corrs,
        "val_corrs": val_corrs,
        "val_pred_rate": val_pred_rate,
    }
)

print(thresh_results)



## === cell 12
try:
    xgb.plot_importance(model)
except Exception as e:
    print(f"plot_importance skipped: {e}")



## === cell 13
best_val_thresh = thresh_results.sort_values("val_corrs", ascending=False).reset_index(
    drop=True
)["thresholds"][0]

thresh = 0.60
print("best_val_thresh (not used):", best_val_thresh, "| initial thresh:", thresh)



## === cell 14
sub = pd.read_csv("../input/nfl-player-contact-detection/sample_submission.csv")
sub["id_split"] = sub["contact_id"].apply(lambda x: x.split("_"))
sub["game_play"] = sub["id_split"].apply(lambda x: x[0] + "_" + x[1])
sub["step"] = sub["id_split"].apply(lambda x: x[2])
sub["nfl_player_id_1"] = sub["id_split"].apply(lambda x: x[3])
sub["nfl_player_id_2"] = sub["id_split"].apply(lambda x: x[4])



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
        "step": "str",
        "nfl_player_id": "str",
        "x_position": "float32",
        "y_position": "float32",
    },
)



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
drop_cols = [
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
]
drop_cols = [c for c in drop_cols if c in test_basetable.columns]
X_test = test_basetable.drop(drop_cols, axis=1)



## === cell 19
xgb_test = xgb.DMatrix(X_test)



## === cell 20
test_preds = np.asarray(model.predict(xgb_test), dtype=np.float32)

target_pos_rate = 0.00000002  # tighter than 0.00000008 -> fewer predicted positives

q = float(np.clip(1.0 - target_pos_rate, 0.0, 1.0))
quant_thresh = float(np.quantile(test_preds, q))

thresh = max(thresh, quant_thresh)
test_labels = (test_preds >= thresh).astype(np.int8)
test_pos_rate = float(test_labels.mean())

min_pos_rate = 0.00000001  # tighter than 0.00000003
if test_pos_rate < min_pos_rate:
    q2 = float(np.clip(1.0 - min_pos_rate, 0.0, 1.0))
    thresh = float(np.quantile(test_preds, q2))
    test_labels = (test_preds >= thresh).astype(np.int8)
    test_pos_rate = float(test_labels.mean())



## === cell 21
print(
    "train_pred_rate@0.30 (debug only):",
    np.mean([1 if x >= 0.30 else 0 for x in train_preds]),
)
print("final_thresh:", float(thresh), "| test_pos_rate:", float(test_pos_rate))



## === cell 22
sub["contact"] = test_labels
sub = sub[["contact_id", "contact"]]
sub.to_csv("submission.csv", header=True, index=False)
print("Wrote submission.csv with shape:", sub.shape)
