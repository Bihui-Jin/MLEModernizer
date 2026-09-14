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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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

0.53231

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import LogisticRegression

BASE_DIR = "../input/nfl-player-contact-detection"

labels = pd.read_csv(f"{BASE_DIR}/train_labels.csv", parse_dates=["datetime"])
ss = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")

tr_tracking = pd.read_csv(
    f"{BASE_DIR}/train_player_tracking.csv", parse_dates=["datetime"]
)
te_tracking = pd.read_csv(
    f"{BASE_DIR}/test_player_tracking.csv", parse_dates=["datetime"]
)

tr_helmets = pd.read_csv(f"{BASE_DIR}/train_baseline_helmets.csv")
te_helmets = pd.read_csv(f"{BASE_DIR}/test_baseline_helmets.csv")

tr_video_metadata = pd.read_csv(
    "../input/nfl-player-contact-detection/train_video_metadata.csv",
    parse_dates=["start_time", "end_time", "snap_time"],
)



## === cell 1
import os
import cv2
import subprocess
from IPython.display import Video, display
import pandas as pd


def video_with_helmets(
    video_path: str, baseline_boxes: pd.DataFrame, verbose=True
) -> str:
    """
    Annotates a video with baseline model boxes and labels.
    """
    VIDEO_CODEC = "MP4V"
    HELMET_COLOR = (0, 0, 0)  # Black
    video_name = os.path.basename(video_path)
    if verbose:
        print(f"Running for {video_name}")
    baseline_boxes = baseline_boxes.copy()

    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = "labeled_" + video_name
    tmp_output_path = "tmp_" + output_path
    output_video = cv2.VideoWriter(
        tmp_output_path, cv2.VideoWriter_fourcc(*VIDEO_CODEC), fps, (width, height)
    )

    frame = 0
    while True:
        it_worked, img = vidcap.read()
        if not it_worked:
            break
        frame += 1

        img_name = video_name.replace(".mp4", "")
        cv2.putText(
            img,
            img_name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            HELMET_COLOR,
            thickness=1,
        )

        cv2.putText(
            img,
            str(frame),
            (1280 - 90, 720 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            HELMET_COLOR,
            thickness=1,
        )

        boxes = baseline_boxes.query("video == @video_name and frame == @frame")
        for box in boxes.itertuples(index=False):
            cv2.rectangle(
                img,
                (box.left, box.top),
                (box.left + box.width, box.top + box.height),
                HELMET_COLOR,
                thickness=1,
            )
            cv2.putText(
                img,
                box.player_label,
                (box.left + 1, max(0, box.top - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                HELMET_COLOR,
                thickness=1,
            )

        output_video.write(img)
    output_video.release()
    if os.path.exists(output_path):
        os.remove(output_path)
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            tmp_output_path,
            "-crf",
            "18",
            "-preset",
            "veryfast",
            "-hide_banner",
            "-loglevel",
            "error",
            "-vcodec",
            "libx264",
            output_path,
        ]
    )
    os.remove(tmp_output_path)

    return output_path




## === cell 2
example_video = "../input/nfl-player-contact-detection/train/58168_003392_Sideline.mp4"
output_video = video_with_helmets(example_video, tr_helmets)

frac = 0.65  # scaling factor for display
display(
    Video(data=output_video, embed=True, height=int(720 * frac), width=int(1280 * frac))
)




## === cell 3
def _parse_contact_id(cid: str) -> pd.Series:
    parts = cid.split("_")
    game_play = "_".join(parts[:2])
    step = int(parts[2])
    player1 = parts[3]
    player2 = parts[4] if len(parts) > 4 else None
    return pd.Series(
        {"game_play": game_play, "step": step, "player1": player1, "player2": player2}
    )


train_df = ss.copy()
train_df[["game_play", "step", "player1", "player2"]] = train_df["contact_id"].apply(
    _parse_contact_id
)

train_df = train_df.merge(
    labels[["contact_id", "contact"]], on="contact_id", how="left"
)

train_df["player1"] = pd.to_numeric(train_df["player1"], errors="coerce")
train_df["player2"] = train_df["player2"].astype(str)

track = tr_tracking.rename(columns={"nfl_player_id": "player_id"})

train_df = train_df.merge(
    track[["game_play", "step", "player_id", "x_position", "y_position", "speed"]],
    left_on=["game_play", "step", "player1"],
    right_on=["game_play", "step", "player_id"],
    how="left",
)
train_df = train_df.rename(
    columns={
        "x_position": "x1",
        "y_position": "y1",
        "speed": "speed1",
    }
)

train_df = train_df.merge(
    track[["game_play", "step", "player_id", "x_position", "y_position", "speed"]],
    left_on=["game_play", "step", "player2"],
    right_on=["game_play", "step", "player_id"],
    how="left",
)
train_df = train_df.rename(
    columns={
        "x_position": "x2",
        "y_position": "y2",
        "speed": "speed2",
    }
)

train_df["is_ground"] = (train_df["player2"] == "G").astype(int)

train_df["dx"] = train_df["x1"] - train_df["x2"]
train_df["dy"] = train_df["y1"] - train_df["y2"]
train_df["distance"] = np.sqrt(train_df["dx"] ** 2 + train_df["dy"] ** 2)
train_df["distance"] = train_df["distance"].fillna(0.0)

train_df["speed2"] = train_df["speed2"].fillna(0.0)

feature_cols = ["distance", "speed1", "speed2", "is_ground"]
X_train = train_df[feature_cols]
y_train = train_df["contact"]

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/191343495.py in <cell line: 0>()
     51 
     52 # Player‑2 positions & speed (will be NaN for ground 'G')
---> 53 train_df = train_df.merge(
     54     track[["game_play", "step", "player_id", "x_position", "y_position", "speed"]],
     55     left_on=["game_play", "step", "player2"],

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in merge(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
  10830         from pandas.core.reshape.merge import merge
  10831 
> 10832         return merge(
  10833             self,
  10834             right,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    168         )
    169     else:
--> 170         op = _MergeOperation(
    171             left_df,
    172             right_df,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in __init__(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)
    805         # validate the merge keys dtypes. We may need to coerce
    806         # to avoid incompatible dtypes
--> 807         self._maybe_coerce_merge_keys()
    808 
    809         # If argument passed to validate,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _maybe_coerce_merge_keys(self)
   1506                     inferred_right in string_types and inferred_left not in string_types
   1507                 ):
-> 1508                     raise ValueError(msg)
   1509 
   1510             # datetimelikes must match exactly

ValueError: You are trying to merge on object and int64 columns for key 'player2'. If you wish to proceed you should use pd.concat

## === cell 4
test_df = ss.copy()
test_df[["game_play", "step", "player1", "player2"]] = test_df["contact_id"].apply(
    _parse_contact_id
)

test_df["player1"] = pd.to_numeric(test_df["player1"], errors="coerce")
test_df["player2"] = test_df["player2"].astype(str)

test_df = test_df.merge(
    track[["game_play", "step", "player_id", "x_position", "y_position", "speed"]],
    left_on=["game_play", "step", "player1"],
    right_on=["game_play", "step", "player_id"],
    how="left",
)
test_df = test_df.rename(
    columns={
        "x_position": "x1",
        "y_position": "y1",
        "speed": "speed1",
    }
)

test_df = test_df.merge(
    track[["game_play", "step", "player_id", "x_position", "y_position", "speed"]],
    left_on=["game_play", "step", "player2"],
    right_on=["game_play", "step", "player_id"],
    how="left",
)
test_df = test_df.rename(
    columns={
        "x_position": "x2",
        "y_position": "y2",
        "speed": "speed2",
    }
)

test_df["is_ground"] = (test_df["player2"] == "G").astype(int)
test_df["dx"] = test_df["x1"] - test_df["x2"]
test_df["dy"] = test_df["y1"] - test_df["y2"]
test_df["distance"] = np.sqrt(test_df["dx"] ** 2 + test_df["dy"] ** 2)
test_df["distance"] = test_df["distance"].fillna(0.0)
test_df["speed2"] = test_df["speed2"].fillna(0.0)

X_test = test_df[feature_cols]

test_df["contact_pred"] = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)

submission_path = "submission.csv"
ss["contact"] = test_df["contact_pred"]
ss.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3992240253.py in <cell line: 0>()
     25 )
     26 
---> 27 test_df = test_df.merge(
     28     track[["game_play", "step", "player_id", "x_position", "y_position", "speed"]],
     29     left_on=["game_play", "step", "player2"],

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in merge(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
  10830         from pandas.core.reshape.merge import merge
  10831 
> 10832         return merge(
  10833             self,
  10834             right,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    168         )
    169     else:
--> 170         op = _MergeOperation(
    171             left_df,
    172             right_df,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in __init__(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)
    805         # validate the merge keys dtypes. We may need to coerce
    806         # to avoid incompatible dtypes
--> 807         self._maybe_coerce_merge_keys()
    808 
    809         # If argument passed to validate,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _maybe_coerce_merge_keys(self)
   1506                     inferred_right in string_types and inferred_left not in string_types
   1507                 ):
-> 1508                     raise ValueError(msg)
   1509 
   1510             # datetimelikes must match exactly

ValueError: You are trying to merge on object and int64 columns for key 'player2'. If you wish to proceed you should use pd.concat
