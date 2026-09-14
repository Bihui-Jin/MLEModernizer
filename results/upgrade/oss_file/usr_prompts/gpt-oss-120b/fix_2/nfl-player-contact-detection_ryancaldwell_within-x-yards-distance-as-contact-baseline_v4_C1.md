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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
pd.set_option("display.max_columns", None)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3670957086.py in <cell line: 0>()
----> 1 pd.set_option("display.max_columns", None)
      2 

NameError: name 'pd' is not defined

## === cell 1
train_helmets = pd.read_csv(
    "../input/nfl-player-contact-detection/train_baseline_helmets.csv",
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
        "frame": "str",
        "nfl_player_id": "str",
        "left": "int8",
        "width": "int8",
        "top": "int8",
        "height": "int8",
    },
).rename(columns={"frame": "step"})

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



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/471807783.py in <cell line: 0>()
----> 1 train_helmets = pd.read_csv(
      2     "../input/nfl-player-contact-detection/train_baseline_helmets.csv",
      3     usecols=[
      4         "game_play",
      5         "view",

NameError: name 'pd' is not defined

## === cell 2
train_helmets["step_int"] = train_helmets["step"].astype(int)
labels["step_int"] = labels["step"].astype(int)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/617958396.py in <cell line: 0>()
----> 1 train_helmets["step_int"] = train_helmets["step"].astype(int)
      2 labels["step_int"] = labels["step"].astype(int)
      3 
      4 

NameError: name 'train_helmets' is not defined

## === cell 3
def add_step_pct(df):
    df["step_pct"] = (
        100
        * (df["step_int"] - min(df["step_int"]))
        / (max(df["step_int"]) - min(df["step_int"]))
    )
    df["step_pct"] = df["step_pct"].apply(np.ceil).astype(np.int32)
    return df




## === cell 4
train_helmets = train_helmets.groupby("game_play").apply(add_step_pct)
labels = labels.groupby("game_play").apply(add_step_pct)

train_helmets["helmet_id"] = (
    train_helmets["game_play"]
    + "_"
    + train_helmets["nfl_player_id"].astype(str)
    + "_"
    + train_helmets["step_pct"].astype(str)
)
labels["helmet_id_1"] = (
    labels["game_play"]
    + "_"
    + labels["nfl_player_id_1"].astype(str)
    + "_"
    + labels["step_pct"].astype(str)
)
labels["helmet_id_2"] = (
    labels["game_play"]
    + "_"
    + labels["nfl_player_id_2"].astype(str)
    + "_"
    + labels["step_pct"].astype(str)
)

train_helmets.drop("step_int", axis=1, inplace=True)
labels.drop("step_int", axis=1, inplace=True)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1182455410.py in <cell line: 0>()
----> 1 train_helmets = train_helmets.groupby("game_play").apply(add_step_pct)
      2 labels = labels.groupby("game_play").apply(add_step_pct)
      3 
      4 train_helmets["helmet_id"] = (
      5     train_helmets["game_play"]

NameError: name 'train_helmets' is not defined

## === cell 5
train_helmets = (
    train_helmets.groupby(["helmet_id", "view"])[["left", "width", "top", "height"]]
    .mean()
    .reset_index()
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3220661306.py in <cell line: 0>()
      1 train_helmets = (
----> 2     train_helmets.groupby(["helmet_id", "view"])[["left", "width", "top", "height"]]
      3     .mean()
      4     .reset_index()
      5 )

NameError: name 'train_helmets' is not defined

## === cell 6
train_helmets["x"] = train_helmets["left"] + (train_helmets["width"] / 2)
train_helmets["y"] = train_helmets["top"] - (train_helmets["height"] / 2)

train_helmets.drop(["left", "width", "top", "height"], axis=1, inplace=True)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/787649551.py in <cell line: 0>()
----> 1 train_helmets["x"] = train_helmets["left"] + (train_helmets["width"] / 2)
      2 train_helmets["y"] = train_helmets["top"] - (train_helmets["height"] / 2)
      3 
      4 train_helmets.drop(["left", "width", "top", "height"], axis=1, inplace=True)
      5 

NameError: name 'train_helmets' is not defined

## === cell 7
train_helmets.head()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2290008575.py in <cell line: 0>()
----> 1 train_helmets.head()
      2 

NameError: name 'train_helmets' is not defined

## === cell 8
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

sideline_player1.columns = ["helmet_id_1", "sl_x_1", "sl_y_1"]
sideline_player2.columns = ["helmet_id_2", "sl_x_2", "sl_y_2"]

endzone_player1.columns = ["helmet_id_1", "ez_x_1", "ez_y_1"]
endzone_player2.columns = ["helmet_id_2", "ez_x_2", "ez_y_2"]



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1297394217.py in <cell line: 0>()
      1 sideline_player1 = (
----> 2     train_helmets[train_helmets["view"] == "Sideline"].drop("view", axis=1).copy()
      3 )
      4 sideline_player2 = (
      5     train_helmets[train_helmets["view"] == "Sideline"].drop("view", axis=1).copy()

NameError: name 'train_helmets' is not defined

## === cell 9
basetable = pd.merge(labels, sideline_player1, on=["helmet_id_1"], how="left")
basetable = pd.merge(basetable, sideline_player2, on=["helmet_id_2"], how="left")
basetable = pd.merge(basetable, endzone_player1, on=["helmet_id_1"], how="left")
basetable = pd.merge(basetable, endzone_player2, on=["helmet_id_2"], how="left")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2197542343.py in <cell line: 0>()
----> 1 basetable = pd.merge(labels, sideline_player1, on=["helmet_id_1"], how="left")
      2 basetable = pd.merge(basetable, sideline_player2, on=["helmet_id_2"], how="left")
      3 basetable = pd.merge(basetable, endzone_player1, on=["helmet_id_1"], how="left")
      4 basetable = pd.merge(basetable, endzone_player2, on=["helmet_id_2"], how="left")
      5 

NameError: name 'pd' is not defined

## === cell 10
basetable["sideline_player_distance"] = np.sqrt(
    (basetable["sl_x_1"] - basetable["sl_x_2"]) ** 2
    + (basetable["sl_y_1"] - basetable["sl_y_2"]) ** 2
)
basetable["endzone_player_distance"] = np.sqrt(
    (basetable["ez_x_1"] - basetable["ez_x_2"]) ** 2
    + (basetable["ez_y_1"] - basetable["ez_y_2"]) ** 2
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3610094660.py in <cell line: 0>()
----> 1 basetable["sideline_player_distance"] = np.sqrt(
      2     (basetable["sl_x_1"] - basetable["sl_x_2"]) ** 2
      3     + (basetable["sl_y_1"] - basetable["sl_y_2"]) ** 2
      4 )
      5 basetable["endzone_player_distance"] = np.sqrt(

NameError: name 'np' is not defined

## === cell 11
basetable.groupby("contact")[
    ["sideline_player_distance", "endzone_player_distance"]
].describe()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3705737497.py in <cell line: 0>()
----> 1 basetable.groupby("contact")[
      2     ["sideline_player_distance", "endzone_player_distance"]
      3 ].describe()
      4 

NameError: name 'basetable' is not defined

## === cell 12
distances = [14, 22, 40, 50, 60, 70, 75]
sideline_corrs = []
for dist in distances:
    sideline_labels = [
        1 if x <= dist else 0 for x in basetable.sideline_player_distance
    ]
    sideline_corr = np.round(
        matthews_corrcoef(y_true=basetable.contact, y_pred=sideline_labels), 5
    )
    sideline_corrs.append(sideline_corr)

plt.figure(figsize=(8, 5))
plt.plot(distances, sideline_corrs)
plt.title("Matthews Correlation across Player Distance Thresholds")
plt.xlabel("Player Distance (Yards) Threshold")
plt.ylabel("Correlation")
plt.show()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/667203055.py in <cell line: 0>()
      3 for dist in distances:
      4     sideline_labels = [
----> 5         1 if x <= dist else 0 for x in basetable.sideline_player_distance
      6     ]
      7     sideline_corr = np.round(

NameError: name 'basetable' is not defined

## === cell 13
distances = [10, 20, 27, 35, 44, 50, 60, 75, 84]
endzone_corrs = []
for dist in distances:
    endzone_labels = [1 if x <= dist else 0 for x in basetable.endzone_player_distance]
    endzone_corr = np.round(
        matthews_corrcoef(y_true=basetable.contact, y_pred=endzone_labels), 5
    )
    endzone_corrs.append(endzone_corr)

plt.figure(figsize=(8, 5))
plt.plot(distances, endzone_corrs)
plt.title("Matthews Correlation across Player Distance Thresholds")
plt.xlabel("Player Distance (Yards) Threshold")
plt.ylabel("Correlation")
plt.show()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3072288543.py in <cell line: 0>()
      2 endzone_corrs = []
      3 for dist in distances:
----> 4     endzone_labels = [1 if x <= dist else 0 for x in basetable.endzone_player_distance]
      5     endzone_corr = np.round(
      6         matthews_corrcoef(y_true=basetable.contact, y_pred=endzone_labels), 5

NameError: name 'basetable' is not defined

## === cell 14
sub = pd.read_csv("../input/nfl-player-contact-detection/sample_submission.csv")
sub["id_split"] = sub["contact_id"].apply(lambda x: x.split("_"))
sub["game_play"] = sub["id_split"].apply(lambda x: x[0] + "_" + x[1])
sub["step"] = sub["id_split"].apply(lambda x: x[2])
sub["nfl_player_id_1"] = sub["id_split"].apply(lambda x: x[3])
sub["nfl_player_id_2"] = sub["id_split"].apply(lambda x: x[4])



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4026434783.py in <cell line: 0>()
----> 1 sub = pd.read_csv("../input/nfl-player-contact-detection/sample_submission.csv")
      2 sub["id_split"] = sub["contact_id"].apply(lambda x: x.split("_"))
      3 sub["game_play"] = sub["id_split"].apply(lambda x: x[0] + "_" + x[1])
      4 sub["step"] = sub["id_split"].apply(lambda x: x[2])
      5 sub["nfl_player_id_1"] = sub["id_split"].apply(lambda x: x[3])

NameError: name 'pd' is not defined

## === cell 15
test_helmets = pd.read_csv(
    "../input/nfl-player-contact-detection/test_baseline_helmets.csv",
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
        "frame": "str",
        "nfl_player_id": "str",
        "left": "int8",
        "width": "int8",
        "top": "int8",
        "height": "int8",
    },
).rename(columns={"frame": "step"})



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3623977688.py in <cell line: 0>()
----> 1 test_helmets = pd.read_csv(
      2     "../input/nfl-player-contact-detection/test_baseline_helmets.csv",
      3     usecols=[
      4         "game_play",
      5         "view",

NameError: name 'pd' is not defined

## === cell 16
test_helmets["step_int"] = test_helmets["step"].astype(int)
sub["step_int"] = sub["step"].astype(int)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2286776202.py in <cell line: 0>()
----> 1 test_helmets["step_int"] = test_helmets["step"].astype(int)
      2 sub["step_int"] = sub["step"].astype(int)
      3 

NameError: name 'test_helmets' is not defined

## === cell 17
test_helmets = test_helmets.groupby("game_play").apply(add_step_pct)
sub = sub.groupby("game_play").apply(add_step_pct)

test_helmets["helmet_id"] = (
    test_helmets["game_play"]
    + "_"
    + test_helmets["nfl_player_id"].astype(str)
    + "_"
    + test_helmets["step_pct"].astype(str)
)
sub["helmet_id_1"] = (
    sub["game_play"]
    + "_"
    + sub["nfl_player_id_1"].astype(str)
    + "_"
    + sub["step_pct"].astype(str)
)
sub["helmet_id_2"] = (
    sub["game_play"]
    + "_"
    + sub["nfl_player_id_2"].astype(str)
    + "_"
    + sub["step_pct"].astype(str)
)

test_helmets.drop("step_int", axis=1, inplace=True)
sub.drop("step_int", axis=1, inplace=True)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2028539773.py in <cell line: 0>()
----> 1 test_helmets = test_helmets.groupby("game_play").apply(add_step_pct)
      2 sub = sub.groupby("game_play").apply(add_step_pct)
      3 
      4 test_helmets["helmet_id"] = (
      5     test_helmets["game_play"]

NameError: name 'test_helmets' is not defined

## === cell 18
test_helmets = (
    test_helmets.groupby(["helmet_id", "view"])[["left", "width", "top", "height"]]
    .mean()
    .reset_index()
)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3779133892.py in <cell line: 0>()
      1 test_helmets = (
----> 2     test_helmets.groupby(["helmet_id", "view"])[["left", "width", "top", "height"]]
      3     .mean()
      4     .reset_index()
      5 )

NameError: name 'test_helmets' is not defined

## === cell 19
test_helmets["x"] = test_helmets["left"] + (test_helmets["width"] / 2)
test_helmets["y"] = test_helmets["top"] - (test_helmets["height"] / 2)

test_helmets.drop(["left", "width", "top", "height"], axis=1, inplace=True)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/778184844.py in <cell line: 0>()
----> 1 test_helmets["x"] = test_helmets["left"] + (test_helmets["width"] / 2)
      2 test_helmets["y"] = test_helmets["top"] - (test_helmets["height"] / 2)
      3 
      4 test_helmets.drop(["left", "width", "top", "height"], axis=1, inplace=True)
      5 

NameError: name 'test_helmets' is not defined

## === cell 20
sideline_player1 = (
    test_helmets[test_helmets["view"] == "Sideline"].drop("view", axis=1).copy()
)
sideline_player2 = (
    test_helmets[test_helmets["view"] == "Sideline"].drop("view", axis=1).copy()
)

endzone_player1 = (
    test_helmets[test_helmets["view"] == "Endzone"].drop("view", axis=1).copy()
)
endzone_player2 = (
    test_helmets[test_helmets["view"] == "Endzone"].drop("view", axis=1).copy()
)

sideline_player1.columns = ["helmet_id_1", "sl_x_1", "sl_y_1"]
sideline_player2.columns = ["helmet_id_2", "sl_x_2", "sl_y_2"]

endzone_player1.columns = ["helmet_id_1", "ez_x_1", "ez_y_1"]
endzone_player2.columns = ["helmet_id_2", "ez_x_2", "ez_y_2"]



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2430393706.py in <cell line: 0>()
      1 sideline_player1 = (
----> 2     test_helmets[test_helmets["view"] == "Sideline"].drop("view", axis=1).copy()
      3 )
      4 sideline_player2 = (
      5     test_helmets[test_helmets["view"] == "Sideline"].drop("view", axis=1).copy()

NameError: name 'test_helmets' is not defined

## === cell 21
test_basetable = pd.merge(sub, sideline_player1, on=["helmet_id_1"], how="left")
test_basetable = pd.merge(
    test_basetable, sideline_player2, on=["helmet_id_2"], how="left"
)
test_basetable = pd.merge(
    test_basetable, endzone_player1, on=["helmet_id_1"], how="left"
)
test_basetable = pd.merge(
    test_basetable, endzone_player2, on=["helmet_id_2"], how="left"
)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/398148414.py in <cell line: 0>()
----> 1 test_basetable = pd.merge(sub, sideline_player1, on=["helmet_id_1"], how="left")
      2 test_basetable = pd.merge(
      3     test_basetable, sideline_player2, on=["helmet_id_2"], how="left"
      4 )
      5 test_basetable = pd.merge(

NameError: name 'pd' is not defined

## === cell 22
test_basetable["sideline_player_distance"] = np.sqrt(
    (test_basetable["sl_x_1"] - test_basetable["sl_x_2"]) ** 2
    + (test_basetable["sl_y_1"] - test_basetable["sl_y_2"]) ** 2
)
test_basetable["endzone_player_distance"] = np.sqrt(
    (test_basetable["ez_x_1"] - test_basetable["ez_x_2"]) ** 2
    + (test_basetable["ez_y_1"] - test_basetable["ez_y_2"]) ** 2
)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3234023486.py in <cell line: 0>()
----> 1 test_basetable["sideline_player_distance"] = np.sqrt(
      2     (test_basetable["sl_x_1"] - test_basetable["sl_x_2"]) ** 2
      3     + (test_basetable["sl_y_1"] - test_basetable["sl_y_2"]) ** 2
      4 )
      5 test_basetable["endzone_player_distance"] = np.sqrt(

NameError: name 'np' is not defined

## === cell 23
best_thr = None
best_mcc = -np.inf
for thr in range(1, 101):
    preds = (basetable["sideline_player_distance"] <= thr).astype(int)
    mcc = matthews_corrcoef(basetable["contact"], preds)
    if mcc > best_mcc:
        best_mcc = mcc
        best_thr = thr

test_basetable["sideline_contact"] = (
    test_basetable["sideline_player_distance"] <= best_thr
).astype(int)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1343513670.py in <cell line: 0>()
      1 # Find the optimal sideline distance threshold on the training data
      2 best_thr = None
----> 3 best_mcc = -np.inf
      4 for thr in range(1, 101):
      5     preds = (basetable["sideline_player_distance"] <= thr).astype(int)

NameError: name 'np' is not defined

## === cell 24
sub["contact"] = test_basetable["sideline_contact"]
sub = sub[["contact_id", "contact"]]
sub.to_csv("submission.csv", header=True, index=False)

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1724289144.py in <cell line: 0>()
----> 1 sub["contact"] = test_basetable["sideline_contact"]
      2 sub = sub[["contact_id", "contact"]]
      3 sub.to_csv("submission.csv", header=True, index=False)

NameError: name 'test_basetable' is not defined
