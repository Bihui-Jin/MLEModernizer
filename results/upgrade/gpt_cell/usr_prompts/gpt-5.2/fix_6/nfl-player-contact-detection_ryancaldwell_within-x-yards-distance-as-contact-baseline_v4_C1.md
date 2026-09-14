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
sklearn-pandas==2.2.0

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

0.26302

# 6. Current score

0.39643

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.39643) has done: 'Diagnosis: The crash happens in cell 11 during `pd.merge(test_basetable, endzone_player1_t, ...)` because `test_basetable` already contains columns like `game_play` (and others) from `sub`, while `endzone_player1_t` (derived from `test_helmets`) also carries `game_play`/`step_10`/`nfl_player_id`. In pandas 2.2, merges that would create duplicate column names via the default `suffixes=('_x','_y')` can raise `MergeError` when the resulting suffixed name would itself collide (here `'game_play_x'`). The simplest fix is to ensure the right-hand helmet tables only include the join key and the coordinate columns needed for distance, so no overlapping non-key columns are introduced. This preserves the exact feature/distance logic while avoiding duplicate columns.

Patch summary: In cell 11, restrict `sideline_player*_t` and `endzone_player*_t` to only the columns required for the merges (`helmet_id_1/2` and the corresponding `sl_*`/`ez_*` coordinate columns) before merging into `test_basetable`. This prevents overlapping columns like `game_play` from being merged and eliminates the `MergeError` deterministically.

Updated cells: Only cell 11 is changed.

Compatibility notes for cell k+1: `test_basetable["sideline_player_distance"]` is still created with the same column names (`sl_x_1`, `sl_y_1`, `sl_x_2`, `sl_y_2`) and has identical semantics, so cell 12 continues to work unchanged.

Assumptions: The intended inference uses only sideline distances (as in cell 12), and endzone coordinates are not required for later cells beyond this merge; keeping endzone merges but trimming columns is therefore safe and minimally invasive.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef



## === cell 1
pd.set_option("display.max_columns", None)

DATA_DIR = "/kaggle/data/nfl-player-contact-detection"



## === cell 2
train_helmets = pd.read_csv(
    f"{DATA_DIR}/train_baseline_helmets.csv",
    usecols=[
        "game_play",
        "view",
        "frame",
        "nfl_player_id",
        "left",
        "width",
        "top",
        "height",
    ],
    dtype={
        "game_play": "str",
        "view": "str",
        "frame": "int32",
        "nfl_player_id": "str",
        "left": "int16",
        "width": "int16",
        "top": "int16",
        "height": "int16",
    },
)

labels = pd.read_csv(
    f"{DATA_DIR}/train_labels.csv",
    usecols=["game_play", "step", "nfl_player_id_1", "nfl_player_id_2", "contact"],
    dtype={
        "game_play": "str",
        "step": "int16",  # 10Hz step (0.1s)
        "nfl_player_id_1": "str",
        "nfl_player_id_2": "str",
        "contact": "int8",
    },
)




## === cell 3
def add_nearest_frame_map(helmets_df: pd.DataFrame) -> pd.DataFrame:
    helmets_df = helmets_df.copy()
    helmets_df["step_10"] = np.rint(
        (helmets_df["frame"].astype(np.float32) - 300.0) / 5.994
    ).astype(np.int16)
    helmets_df = helmets_df[helmets_df["step_10"] >= 0]
    return helmets_df


train_helmets = add_nearest_frame_map(train_helmets)



## === cell 4
train_helmets = train_helmets.groupby(
    ["game_play", "view", "step_10", "nfl_player_id"], as_index=False
)[["left", "width", "top", "height"]].mean()

train_helmets["x"] = train_helmets["left"] + (train_helmets["width"] / 2.0)
train_helmets["y"] = train_helmets["top"] - (train_helmets["height"] / 2.0)
train_helmets = train_helmets.drop(["left", "width", "top", "height"], axis=1)

train_helmets["helmet_id"] = (
    train_helmets["game_play"]
    + "_"
    + train_helmets["nfl_player_id"].astype(str)
    + "_"
    + train_helmets["step_10"].astype(str)
)

labels["helmet_id_1"] = (
    labels["game_play"]
    + "_"
    + labels["nfl_player_id_1"].astype(str)
    + "_"
    + labels["step"].astype(str)
)
labels["helmet_id_2"] = (
    labels["game_play"]
    + "_"
    + labels["nfl_player_id_2"].astype(str)
    + "_"
    + labels["step"].astype(str)
)



## === cell 5
sideline_player1 = (
    train_helmets[train_helmets["view"] == "Sideline"].drop("view", axis=1).copy()
)
sideline_player2 = (
    train_helmets[train_helmets["view"] == "Sideline"].drop("view", axis=1).copy()
)

endzone_player1 = (
    train_helmets[train_helmets["view"] == "Endzone"].drop("view", axis=1).copy()
)
endzone_player2 = (
    train_helmets[train_helmets["view"] == "Endzone"].drop("view", axis=1).copy()
)

sideline_player1 = sideline_player1.rename(
    columns={"helmet_id": "helmet_id_1", "x": "sl_x_1", "y": "sl_y_1"}
)
sideline_player2 = sideline_player2.rename(
    columns={"helmet_id": "helmet_id_2", "x": "sl_x_2", "y": "sl_y_2"}
)

endzone_player1 = endzone_player1.rename(
    columns={"helmet_id": "helmet_id_1", "x": "ez_x_1", "y": "ez_y_1"}
)
endzone_player2 = endzone_player2.rename(
    columns={"helmet_id": "helmet_id_2", "x": "ez_x_2", "y": "ez_y_2"}
)



## === cell 6
sideline_player1_m = sideline_player1[["helmet_id_1", "sl_x_1", "sl_y_1"]]
sideline_player2_m = sideline_player2[["helmet_id_2", "sl_x_2", "sl_y_2"]]
endzone_player1_m = endzone_player1[["helmet_id_1", "ez_x_1", "ez_y_1"]]
endzone_player2_m = endzone_player2[["helmet_id_2", "ez_x_2", "ez_y_2"]]

basetable = pd.merge(labels, sideline_player1_m, on=["helmet_id_1"], how="left")
basetable = pd.merge(basetable, sideline_player2_m, on=["helmet_id_2"], how="left")
basetable = pd.merge(basetable, endzone_player1_m, on=["helmet_id_1"], how="left")
basetable = pd.merge(basetable, endzone_player2_m, on=["helmet_id_2"], how="left")

basetable["sideline_player_distance"] = np.sqrt(
    (basetable["sl_x_1"] - basetable["sl_x_2"]) ** 2
    + (basetable["sl_y_1"] - basetable["sl_y_2"]) ** 2
)
basetable["endzone_player_distance"] = np.sqrt(
    (basetable["ez_x_1"] - basetable["ez_x_2"]) ** 2
    + (basetable["ez_y_1"] - basetable["ez_y_2"]) ** 2
)


## === cell 7
_ = basetable.groupby("contact")[
    ["sideline_player_distance", "endzone_player_distance"]
].describe()
print("Train basetable rows:", len(basetable))
print(
    "Sideline distance non-null:", basetable["sideline_player_distance"].notna().sum()
)
print("Endzone distance non-null:", basetable["endzone_player_distance"].notna().sum())



## === cell 8
cand = basetable["sideline_player_distance"].dropna()
if len(cand) == 0:
    best_thresh = 22.0
else:
    qs = np.linspace(0.01, 0.20, 40)  # same deterministic grid idea
    thresh_grid = np.unique(np.quantile(cand.to_numpy(), qs))
    best_mcc = -1.0
    best_thresh = 22.0
    y_true = basetable["contact"].to_numpy()
    dist_arr = basetable["sideline_player_distance"].to_numpy()
    valid = np.isfinite(dist_arr)
    for t in thresh_grid:
        y_pred = np.zeros_like(y_true)
        y_pred[valid] = (dist_arr[valid] <= t).astype(np.int8)
        mcc = matthews_corrcoef(y_true=y_true, y_pred=y_pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = float(t)

print("Selected sideline distance threshold:", best_thresh)
print("Best train MCC (sideline only):", best_mcc)



## === cell 9
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
parts = sub["contact_id"].str.split("_", expand=True)
sub["game_play"] = parts[0] + "_" + parts[1]
sub["step"] = parts[2].astype(np.int16)
sub["nfl_player_id_1"] = parts[3].astype(str)
sub["nfl_player_id_2"] = parts[4].astype(str)

sub["helmet_id_1"] = (
    sub["game_play"] + "_" + sub["nfl_player_id_1"] + "_" + sub["step"].astype(str)
)
sub["helmet_id_2"] = (
    sub["game_play"] + "_" + sub["nfl_player_id_2"] + "_" + sub["step"].astype(str)
)



## === cell 10
test_helmets = pd.read_csv(
    f"{DATA_DIR}/test_baseline_helmets.csv",
    usecols=[
        "game_play",
        "view",
        "frame",
        "nfl_player_id",
        "left",
        "width",
        "top",
        "height",
    ],
    dtype={
        "game_play": "str",
        "view": "str",
        "frame": "int32",
        "nfl_player_id": "str",
        "left": "int16",
        "width": "int16",
        "top": "int16",
        "height": "int16",
    },
)

test_helmets = add_nearest_frame_map(test_helmets)

test_helmets = test_helmets.groupby(
    ["game_play", "view", "step_10", "nfl_player_id"], as_index=False
)[["left", "width", "top", "height"]].mean()

test_helmets["x"] = test_helmets["left"] + (test_helmets["width"] / 2.0)
test_helmets["y"] = test_helmets["top"] - (test_helmets["height"] / 2.0)
test_helmets = test_helmets.drop(["left", "width", "top", "height"], axis=1)

test_helmets["helmet_id"] = (
    test_helmets["game_play"]
    + "_"
    + test_helmets["nfl_player_id"].astype(str)
    + "_"
    + test_helmets["step_10"].astype(str)
)



## === cell 11
sideline_player1_t = (
    test_helmets[test_helmets["view"] == "Sideline"].drop("view", axis=1).copy()
)
sideline_player2_t = (
    test_helmets[test_helmets["view"] == "Sideline"].drop("view", axis=1).copy()
)

endzone_player1_t = (
    test_helmets[test_helmets["view"] == "Endzone"].drop("view", axis=1).copy()
)
endzone_player2_t = (
    test_helmets[test_helmets["view"] == "Endzone"].drop("view", axis=1).copy()
)

sideline_player1_t = sideline_player1_t.rename(
    columns={"helmet_id": "helmet_id_1", "x": "sl_x_1", "y": "sl_y_1"}
)
sideline_player2_t = sideline_player2_t.rename(
    columns={"helmet_id": "helmet_id_2", "x": "sl_x_2", "y": "sl_y_2"}
)

endzone_player1_t = endzone_player1_t.rename(
    columns={"helmet_id": "helmet_id_1", "x": "ez_x_1", "y": "ez_y_1"}
)
endzone_player2_t = endzone_player2_t.rename(
    columns={"helmet_id": "helmet_id_2", "x": "ez_x_2", "y": "ez_y_2"}
)

sideline_player1_t = sideline_player1_t[["helmet_id_1", "sl_x_1", "sl_y_1"]]
sideline_player2_t = sideline_player2_t[["helmet_id_2", "sl_x_2", "sl_y_2"]]
endzone_player1_t = endzone_player1_t[["helmet_id_1", "ez_x_1", "ez_y_1"]]
endzone_player2_t = endzone_player2_t[["helmet_id_2", "ez_x_2", "ez_y_2"]]

test_basetable = pd.merge(sub, sideline_player1_t, on=["helmet_id_1"], how="left")
test_basetable = pd.merge(
    test_basetable, sideline_player2_t, on=["helmet_id_2"], how="left"
)
test_basetable = pd.merge(
    test_basetable, endzone_player1_t, on=["helmet_id_1"], how="left"
)
test_basetable = pd.merge(
    test_basetable, endzone_player2_t, on=["helmet_id_2"], how="left"
)

test_basetable["sideline_player_distance"] = np.sqrt(
    (test_basetable["sl_x_1"] - test_basetable["sl_x_2"]) ** 2
    + (test_basetable["sl_y_1"] - test_basetable["sl_y_2"]) ** 2
)


## === cell 12
dist = test_basetable["sideline_player_distance"].to_numpy()
pred = np.zeros(len(test_basetable), dtype=np.int8)
valid = np.isfinite(dist)
pred[valid] = (dist[valid] <= best_thresh).astype(np.int8)

sub_out = sub[["contact_id"]].copy()
sub_out["contact"] = pred

sub_out.to_csv("submission.csv", header=True, index=False)

print(sub_out.head())
print("Wrote submission.csv with rows:", len(sub_out))
print("Positive rate:", float(sub_out["contact"].mean()))
