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
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef


## === cell 1
pd.set_option("display.max_columns", None)


## === cell 2
train = pd.read_csv('../input/nfl-player-contact-detection/train_player_tracking.csv',
                    usecols=['game_play','game_key','play_id','step','nfl_player_id','x_position','y_position'],
                    dtype={'game_play':'str',
                           'game_key':'str',
                           'play_id':'str',
                           'step':'str',
                           'nfl_player_id':'str',
                           'x_position':'float32',
                           'y_position':'float32'})

labels = pd.read_csv('../input/nfl-player-contact-detection/train_labels.csv',
                     usecols=['game_play','step','nfl_player_id_1','nfl_player_id_2','contact'],
                     dtype={'game_play':'str',
                            'step':'str',
                            'nfl_player_id_1':'str',
                            'nfl_player_id_2':'str',
                            'contact':'int'})


## === cell 3
player1 = train.copy()
player2 = train.drop(['game_key','play_id'], axis=1).copy()

player1.columns = ['game_play','game_key','play_id','nfl_player_id_1','step','x_position_1','y_position_1']
player2.columns = ['game_play','nfl_player_id_2','step','x_position_2','y_position_2']


## === cell 4
print(player1.shape, player2.shape, labels.shape)


## === cell 5
def add_step_pct(df):
    df['step_pct'] = 100 * (df['step_int']-min(df['step_int']))/(max(df['step_int'])-min(df['step_int']))
    df['step_pct'] = df['step_pct'].apply(np.ceil).astype(np.int16)
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
    basetable = pd.merge(pairs, tracking_1, on=['game_play','step','nfl_player_id_1'], how='inner')
    basetable = pd.merge(basetable, tracking_2, on=['game_play','step','nfl_player_id_2'], how='left')
    
    basetable['player_distances'] = np.sqrt((basetable['x_position_1'] - basetable['x_position_2'])**2 + (basetable['y_position_1'] - basetable['y_position_2'])**2)
    
    basetable['step_int'] = basetable['step'].astype('int16')
    basetable = basetable.groupby('game_play').apply(add_step_pct)
    
    return basetable


## === cell 6
%%time
train_basetable = feature_prep(labels, player1, player2)


## === cell 7
X = train_basetable.drop(['game_play','step','nfl_player_id_1','nfl_player_id_2','play_id','x_position_1',
                          'y_position_1','x_position_2','y_position_2','step_int'], axis=1)


## === cell 8
games = list(set(X.game_key))
N_train = int(0.8*len(games))
train_games = [x for x in np.random.choice(games, N_train, replace=False)]
val_games = list(set(games)-set(train_games))

train_games = pd.DataFrame({'game_key':train_games})
val_games = pd.DataFrame({'game_key':val_games})
print(len(games), train_games.shape[0], val_games.shape[0])


## === cell 9
X_train = pd.merge(X, train_games, on='game_key', how='inner')
X_val = pd.merge(X, val_games, on='game_key', how='inner')

y_train = X_train.contact
y_val = X_val.contact

X_train.drop(['game_key','contact'], axis=1, inplace=True)
X_val.drop(['game_key','contact'], axis=1, inplace=True)


## === cell 10
xgb_train = xgb.DMatrix(X_train, y_train)
xgb_val = xgb.DMatrix(X_val, y_val)


## === cell 11
Diagnosis: Cell 11 crashes because `tree_method='gpu_hist'` forces XGBoost to use a GPU, but the runtime has no available GPU device (`ctx_->gpu_id >= 0 ... Must have at least one device`). This is an environment incompatibility rather than a data/model bug. The minimal deterministic fix is to switch to the CPU histogram algorithm while keeping all other parameters, training loop, and evaluation semantics unchanged.

Patch summary: In cell 11 only, change `tree_method` from `'gpu_hist'` to `'hist'` so training runs on CPU without requiring a GPU. No other logic (objective, metrics, rounds, early stopping, prediction, threshold search) is modified.

Updated cells: (cell 11 only)

Compatibility notes for cell k+1: `model` remains an `xgboost.Booster` trained via `xgb.train`, so `xgb.plot_importance(model)` in cell 12 continues to work unchanged.

Assumptions: The environment has no CUDA-capable GPU accessible to XGBoost; CPU training is acceptable and preserves the intended algorithm aside from compute device.

```python
%%time

xgb_params = {'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'learning_rate':0.01,
              'max_depth':5,
              'tree_method':'hist'}

evals = [(xgb_train,'train'),(xgb_val,'val')]

model = xgb.train(xgb_params, 
                  xgb_train,
                  num_boost_round=1000,
                  early_stopping_rounds=20, 
                  evals=evals, 
                  verbose_eval=100)

train_preds = [x for x in model.predict(xgb_train)]
val_preds = [x for x in model.predict(xgb_val)]

thresholds = [x/100. for x in range(20, 31)]
train_corrs = []
val_corrs = []
val_pred_rate = []
for thresh in thresholds:
    train_labels = [1 if x>=thresh else 0 for x in train_preds]
    val_labels = [1 if x>=thresh else 0 for x in val_preds]

    train_corr = matthews_corrcoef(y_true=y_train, y_pred=train_labels)
    val_corr = matthews_corrcoef(y_true=y_val, y_pred=val_labels)

    train_corrs.append(train_corr)
    val_corrs.append(val_corr)
    val_pred_rate.append(np.mean(val_labels))

thresh_results = pd.DataFrame({'thresholds':thresholds,
                               'train_corrs':train_corrs,
                               'val_corrs':val_corrs,
                               'val_pred_rate':val_pred_rate})

print(thresh_results)
```

## --- ERROR in cell 11, traceback:
[0;36m  File [0;32m"/tmp/ipykernel_11/1609582508.py"[0;36m, line [0;32m1[0m
[0;31m    Diagnosis: Cell 11 crashes because `tree_method='gpu_hist'` forces XGBoost to use a GPU, but the runtime has no available GPU device (`ctx_->gpu_id >= 0 ... Must have at least one device`). This is an environment incompatibility rather than a data/model bug. The minimal deterministic fix is to switch to the CPU histogram algorithm while keeping all other parameters, training loop, and evaluation semantics unchanged.[0m
[0m                    ^[0m
[0;31mSyntaxError[0m[0;31m:[0m invalid syntax


## === cell 12
xgb.plot_importance(model)
