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

0.17668

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.22921) has done: 'The update keeps the entire workflow unchanged but replaces the naïve 100‑iteration threshold search with a single sort‑and‑cumulative‑count algorithm, eliminating the O(N × 100) cost while yielding identical MCC values. The new vectorized search runs in O(N log N) time and uses only small auxiliary arrays, preserving exact predictions and the original model logic.'
- What this solution (achieved 0.20564) has done: 'I add a parallel distance‑based threshold for the endzone view, keep the original sideline threshold, and predict contact when either view indicates contact. This uses the same lightweight search as before, so the core model logic stays unchanged while gaining a modest MCC boost toward the target.'
- What this solution (achieved 0.20564) has done: 'I extend the distance‑threshold search from 1‑100 pixels to 1‑200 pixels for both sideline and endzone views. This small change lets the model consider larger contact distances that were previously ignored, which should raise the MCC closer to the target score while keeping the overall logic unchanged.'
- What this solution (achieved 0.17668) has done: 'I add a joint‑distance threshold search that looks at the minimum of the sideline and endzone distances, then use that single best threshold for predictions. This keeps the overall workflow and model unchanged while giving a higher MCC that moves the score toward the target.'
- What this solution (achieved 0.06339) has done: 'I increase the threshold search range to 300 pixels and use the separate best sideline and endzone thresholds (instead of the minimum‑distance rule) when generating predictions, which should raise the MCC toward the target while preserving the original workflow.'
- What this solution (achieved 0.17668) has done: 'I keep the overall workflow unchanged but switch the prediction rule to use the combined minimum distance (the one for which the best MCC was already computed in cell 8). This aligns the test‑time decision with the threshold that gave the highest MCC on the training split, which should raise the score toward the target.'
- What this solution (achieved 0.06339) has done: 'I keep the overall workflow unchanged and only adjust the final prediction rule. Instead of using a single combined distance threshold, I apply the separate optimal thresholds found for the sideline and endzone views and predict a contact when either view indicates proximity (OR logic). This small change aligns the test‑time decision with the best per‑view thresholds and is expected to raise the MCC toward the target while preserving the core model logic.'
- What this solution (achieved 0.17668) has done: 'I switch the final prediction rule to use the single combined minimum‑distance threshold that was already computed ( best_thr_combined ). This aligns the test‑time decision with the threshold that gave the highest MCC on the training split, replacing the previous OR‑logic of separate view thresholds and should move the MCC closer to the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef

cat_dtype = "category"
train_helmets = pd.read_csv(
    "/kaggle/input/nfl-player-contact-detection/train_baseline_helmets.csv",
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
        "game_play": cat_dtype,
        "view": cat_dtype,
        "frame": "int32",
        "nfl_player_id": cat_dtype,
        "left": "int8",
        "width": "int8",
        "top": "int8",
        "height": "int8",
    },
).rename(columns={"frame": "step"})

labels = pd.read_csv(
    "/kaggle/input/nfl-player-contact-detection/train_labels.csv",
    usecols=["game_play", "step", "nfl_player_id_1", "nfl_player_id_2", "contact"],
    dtype={
        "game_play": cat_dtype,
        "step": "int32",
        "nfl_player_id_1": cat_dtype,
        "nfl_player_id_2": cat_dtype,
        "contact": "int8",
    },
)




## === cell 1
def add_step_pct(df):
    step_min = df.groupby("game_play")["step"].transform("min")
    step_max = df.groupby("game_play")["step"].transform("max")
    denom = (step_max - step_min).replace(0, 1)  # avoid division by zero
    df["step_pct"] = np.ceil(100 * (df["step"] - step_min) / denom).astype(np.int32)
    df.drop(columns=["step"], inplace=True)
    return df


train_helmets = add_step_pct(train_helmets)
labels = add_step_pct(labels)



## === cell 2
train_helmets["helmet_id"] = (
    train_helmets["game_play"].astype(str)
    + "_"
    + train_helmets["nfl_player_id"].astype(str)
    + "_"
    + train_helmets["step_pct"].astype(str)
)

labels["helmet_id_1"] = (
    labels["game_play"].astype(str)
    + "_"
    + labels["nfl_player_id_1"].astype(str)
    + "_"
    + labels["step_pct"].astype(str)
)
labels["helmet_id_2"] = (
    labels["game_play"].astype(str)
    + "_"
    + labels["nfl_player_id_2"].astype(str)
    + "_"
    + labels["step_pct"].astype(str)
)



## === cell 3
train_helmets = (
    train_helmets.groupby(["helmet_id", "view"])[["left", "width", "top", "height"]]
    .mean()
    .reset_index()
)



## === cell 4
train_helmets["x"] = train_helmets["left"] + (train_helmets["width"] / 2)
train_helmets["y"] = train_helmets["top"] - (train_helmets["height"] / 2)
train_helmets.drop(columns=["left", "width", "top", "height"], inplace=True)



## === cell 5
sideline = train_helmets[train_helmets["view"] == "Sideline"].drop(columns="view")
endzone = train_helmets[train_helmets["view"] == "Endzone"].drop(columns="view")

sideline_player1 = sideline.rename(
    columns={"helmet_id": "helmet_id_1", "x": "sl_x_1", "y": "sl_y_1"}
)
sideline_player2 = sideline.rename(
    columns={"helmet_id": "helmet_id_2", "x": "sl_x_2", "y": "sl_y_2"}
)
endzone_player1 = endzone.rename(
    columns={"helmet_id": "helmet_id_1", "x": "ez_x_1", "y": "ez_y_1"}
)
endzone_player2 = endzone.rename(
    columns={"helmet_id": "helmet_id_2", "x": "ez_x_2", "y": "ez_y_2"}
)



## === cell 6
basetable = pd.merge(labels, sideline_player1, on="helmet_id_1", how="left")
basetable = pd.merge(basetable, sideline_player2, on="helmet_id_2", how="left")
basetable = pd.merge(basetable, endzone_player1, on="helmet_id_1", how="left")
basetable = pd.merge(basetable, endzone_player2, on="helmet_id_2", how="left")



## === cell 7
basetable["sideline_player_distance"] = np.sqrt(
    (basetable["sl_x_1"] - basetable["sl_x_2"]) ** 2
    + (basetable["sl_y_1"] - basetable["sl_y_2"]) ** 2
)
basetable["endzone_player_distance"] = np.sqrt(
    (basetable["ez_x_1"] - basetable["ez_x_2"]) ** 2
    + (basetable["ez_y_1"] - basetable["ez_y_2"]) ** 2
)



## === cell 8
MAX_THR = 300

valid = basetable.dropna(subset=["sideline_player_distance"])
dist = valid["sideline_player_distance"].values
contact = valid["contact"].values

order = np.argsort(dist)
dist_sorted = dist[order]
contact_sorted = contact[order]

total_pos = contact.sum()
total_neg = len(contact) - total_pos

cumulative_tp = np.cumsum(contact_sorted, dtype=np.int64)
cumulative_fp = np.arange(1, len(contact_sorted) + 1, dtype=np.int64) - cumulative_tp

best_thr_sideline = None
best_mcc_sideline = -np.inf

for thr in range(1, MAX_THR + 1):
    idx = np.searchsorted(dist_sorted, thr, side="right")
    if idx == 0:
        tp = fp = 0
    else:
        tp = cumulative_tp[idx - 1]
        fp = cumulative_fp[idx - 1]
    fn = total_pos - tp
    tn = total_neg - fp

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom != 0 else -1.0

    if mcc > best_mcc_sideline:
        best_mcc_sideline = mcc
        best_thr_sideline = thr

valid_ez = basetable.dropna(subset=["endzone_player_distance"])
dist_ez = valid_ez["endzone_player_distance"].values
contact_ez = valid_ez["contact"].values

order_ez = np.argsort(dist_ez)
dist_sorted_ez = dist_ez[order_ez]
contact_sorted_ez = contact_ez[order_ez]

cumulative_tp_ez = np.cumsum(contact_sorted_ez, dtype=np.int64)
cumulative_fp_ez = (
    np.arange(1, len(contact_sorted_ez) + 1, dtype=np.int64) - cumulative_tp_ez
)

best_thr_endzone = None
best_mcc_endzone = -np.inf

for thr in range(1, MAX_THR + 1):
    idx = np.searchsorted(dist_sorted_ez, thr, side="right")
    if idx == 0:
        tp = fp = 0
    else:
        tp = cumulative_tp_ez[idx - 1]
        fp = cumulative_fp_ez[idx - 1]
    fn = total_pos - tp
    tn = total_neg - fp

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom != 0 else -1.0

    if mcc > best_mcc_endzone:
        best_mcc_endzone = mcc
        best_thr_endzone = thr

combined_dist = np.minimum(
    basetable["sideline_player_distance"].fillna(1e9),
    basetable["endzone_player_distance"].fillna(1e9),
).values
contact_comb = basetable["contact"].values

order_c = np.argsort(combined_dist)
dist_sorted_c = combined_dist[order_c]
contact_sorted_c = contact_comb[order_c]

cumulative_tp_c = np.cumsum(contact_sorted_c, dtype=np.int64)
cumulative_fp_c = (
    np.arange(1, len(contact_sorted_c) + 1, dtype=np.int64) - cumulative_tp_c
)

best_thr_combined = None
best_mcc_combined = -np.inf

for thr in range(1, MAX_THR + 1):
    idx = np.searchsorted(dist_sorted_c, thr, side="right")
    if idx == 0:
        tp = fp = 0
    else:
        tp = cumulative_tp_c[idx - 1]
        fp = cumulative_fp_c[idx - 1]
    fn = total_pos - tp
    tn = total_neg - fp

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom != 0 else -1.0

    if mcc > best_mcc_combined:
        best_mcc_combined = mcc
        best_thr_combined = thr



## === cell 9
sub = pd.read_csv("/kaggle/input/nfl-player-contact-detection/sample_submission.csv")
sub["id_split"] = sub["contact_id"].str.split("_")
sub["game_play"] = sub["id_split"].str[0] + "_" + sub["id_split"].str[1]
sub["step"] = sub["id_split"].str[2].astype(int)
sub["nfl_player_id_1"] = sub["id_split"].str[3]
sub["nfl_player_id_2"] = sub["id_split"].str[4]
sub.drop(columns="id_split", inplace=True)

test_helmets = pd.read_csv(
    "/kaggle/input/nfl-player-contact-detection/test_baseline_helmets.csv",
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
        "game_play": cat_dtype,
        "view": cat_dtype,
        "frame": "int32",
        "nfl_player_id": cat_dtype,
        "left": "int8",
        "width": "int8",
        "top": "int8",
        "height": "int8",
    },
).rename(columns={"frame": "step"})

test_helmets["step"] = test_helmets["step"].astype(int)
sub["step"] = sub["step"].astype(int)



## === cell 10
test_helmets = add_step_pct(test_helmets)
sub = add_step_pct(sub)

test_helmets["helmet_id"] = (
    test_helmets["game_play"].astype(str)
    + "_"
    + test_helmets["nfl_player_id"].astype(str)
    + "_"
    + test_helmets["step_pct"].astype(str)
)

sub["helmet_id_1"] = (
    sub["game_play"].astype(str)
    + "_"
    + sub["nfl_player_id_1"].astype(str)
    + "_"
    + sub["step_pct"].astype(str)
)
sub["helmet_id_2"] = (
    sub["game_play"].astype(str)
    + "_"
    + sub["nfl_player_id_2"].astype(str)
    + "_"
    + sub["step_pct"].astype(str)
)



## === cell 11
test_helmets = (
    test_helmets.groupby(["helmet_id", "view"])[["left", "width", "top", "height"]]
    .mean()
    .reset_index()
)

test_helmets["x"] = test_helmets["left"] + (test_helmets["width"] / 2)
test_helmets["y"] = test_helmets["top"] - (test_helmets["height"] / 2)
test_helmets.drop(columns=["left", "width", "top", "height"], inplace=True)



## === cell 12
sideline = test_helmets[test_helmets["view"] == "Sideline"].drop(columns="view")
endzone = test_helmets[test_helmets["view"] == "Endzone"].drop(columns="view")

sideline_player1 = sideline.rename(
    columns={"helmet_id": "helmet_id_1", "x": "sl_x_1", "y": "sl_y_1"}
)
sideline_player2 = sideline.rename(
    columns={"helmet_id": "helmet_id_2", "x": "sl_x_2", "y": "sl_y_2"}
)
endzone_player1 = endzone.rename(
    columns={"helmet_id": "helmet_id_1", "x": "ez_x_1", "y": "ez_y_1"}
)
endzone_player2 = endzone.rename(
    columns={"helmet_id": "helmet_id_2", "x": "ez_x_2", "y": "ez_y_2"}
)



## === cell 13
test_basetable = pd.merge(sub, sideline_player1, on="helmet_id_1", how="left")
test_basetable = pd.merge(
    test_basetable, sideline_player2, on="helmet_id_2", how="left"
)
test_basetable = pd.merge(test_basetable, endzone_player1, on="helmet_id_1", how="left")
test_basetable = pd.merge(test_basetable, endzone_player2, on="helmet_id_2", how="left")

test_basetable["sideline_player_distance"] = np.sqrt(
    (test_basetable["sl_x_1"] - test_basetable["sl_x_2"]) ** 2
    + (test_basetable["sl_y_1"] - test_basetable["sl_y_2"]) ** 2
)
test_basetable["endzone_player_distance"] = np.sqrt(
    (test_basetable["ez_x_1"] - test_basetable["ez_x_2"]) ** 2
    + (test_basetable["ez_y_1"] - test_basetable["ez_y_2"]) ** 2
)

combined_distance = np.minimum(
    test_basetable["sideline_player_distance"].fillna(1e9),
    test_basetable["endzone_player_distance"].fillna(1e9),
)
contact_pred = (combined_distance <= best_thr_combined).astype(int)

sub["contact"] = contact_pred
sub = sub[["contact_id", "contact"]]
sub.to_csv("submission.csv", index=False)
