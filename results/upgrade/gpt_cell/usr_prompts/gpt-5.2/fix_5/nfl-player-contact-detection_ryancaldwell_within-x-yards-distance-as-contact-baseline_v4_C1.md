# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.11

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

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



## --- ERROR in cell 11, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mMergeError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2697774185.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     32[0m     [0mtest_basetable[0m[0;34m,[0m [0msideline_player2_t[0m[0;34m,[0m [0mon[0m[0;34m=[0m[0;34m[[0m[0;34m"helmet_id_2"[0m[0;34m][0m[0;34m,[0m [0mhow[0m[0;34m=[0m[0;34m"left"[0m[0;34m[0m[0;34m[0m[0m
[1;32m     33[0m )
[0;32m---> 34[0;31m test_basetable = pd.merge(
[0m[1;32m     35[0m     [0mtest_basetable[0m[0;34m,[0m [0mendzone_player1_t[0m[0;34m,[0m [0mon[0m[0;34m=[0m[0;34m[[0m[0;34m"helmet_id_1"[0m[0;34m][0m[0;34m,[0m [0mhow[0m[0;34m=[0m[0;34m"left"[0m[0;34m[0m[0;34m[0m[0m
[1;32m     36[0m )

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py[0m in [0;36mmerge[0;34m(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)[0m
[1;32m    182[0m             [0mvalidate[0m[0;34m=[0m[0mvalidate[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    183[0m         )
[0;32m--> 184[0;31m         [0;32mreturn[0m [0mop[0m[0;34m.[0m[0mget_result[0m[0;34m([0m[0mcopy[0m[0;34m=[0m[0mcopy[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    185[0m [0;34m[0m[0m
[1;32m    186[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py[0m in [0;36mget_result[0;34m(self, copy)[0m
[1;32m    886[0m         [0mjoin_index[0m[0;34m,[0m [0mleft_indexer[0m[0;34m,[0m [0mright_indexer[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_get_join_info[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    887[0m [0;34m[0m[0m
[0;32m--> 888[0;31m         result = self._reindex_and_concat(
[0m[1;32m    889[0m             [0mjoin_index[0m[0;34m,[0m [0mleft_indexer[0m[0;34m,[0m [0mright_indexer[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0mcopy[0m[0;34m[0m[0;34m[0m[0m
[1;32m    890[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py[0m in [0;36m_reindex_and_concat[0;34m(self, join_index, left_indexer, right_indexer, copy)[0m
[1;32m    838[0m         [0mright[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mright[0m[0;34m[[0m[0;34m:[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m    839[0m [0;34m[0m[0m
[0;32m--> 840[0;31m         llabels, rlabels = _items_overlap_with_suffix(
[0m[1;32m    841[0m             [0mself[0m[0;34m.[0m[0mleft[0m[0;34m.[0m[0m_info_axis[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mright[0m[0;34m.[0m[0m_info_axis[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0msuffixes[0m[0;34m[0m[0;34m[0m[0m
[1;32m    842[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py[0m in [0;36m_items_overlap_with_suffix[0;34m(left, right, suffixes)[0m
[1;32m   2755[0m         [0mdups[0m[0;34m.[0m[0mextend[0m[0;34m([0m[0mrlabels[0m[0;34m[[0m[0;34m([0m[0mrlabels[0m[0;34m.[0m[0mduplicated[0m[0;34m([0m[0;34m)[0m[0;34m)[0m [0;34m&[0m [0;34m([0m[0;34m~[0m[0mright[0m[0;34m.[0m[0mduplicated[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m][0m[0;34m.[0m[0mtolist[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2756[0m     [0;32mif[0m [0mdups[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2757[0;31m         raise MergeError(
[0m[1;32m   2758[0m             [0;34mf"Passing 'suffixes' which cause duplicate columns {set(dups)} is "[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2759[0m             [0;34mf"not allowed."[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;31mMergeError[0m: Passing 'suffixes' which cause duplicate columns {'game_play_x'} is not allowed.

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
