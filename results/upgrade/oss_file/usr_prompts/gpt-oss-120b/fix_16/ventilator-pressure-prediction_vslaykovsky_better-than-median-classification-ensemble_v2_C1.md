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
Given time series of breaths, predict the airway pressure in the respiratory circuit during the breath, given the time series of control inputs.

The best submissions will take lung attributes compliance and resistance into account.

## Metric
Mean absolute error between the predicted and actual pressures during the inspiratory phase of each breath. The expiratory phase is not scored.

## Submission Format
For each `id` in the test set, you must predict a value for the `pressure` variable. The file should contain a header and have the following format:

```
id,pressure
1,20
2,23
3,24
etc.
```

## Dataset
The ventilator data used in this competition was produced using a modified [open-source ventilator](https://pvp.readthedocs.io/) connected to an [artificial bellows test lung](https://www.ingmarmed.com/product/quicklung/) via a respiratory circuit. The diagram below illustrates the setup, with the two control inputs highlighted in green and the state variable (airway pressure) to predict in blue. The first control input is a continuous variable from 0 to 100 representing the percentage the inspiratory solenoid valve is open to let air into the lung (i.e., 0 is completely closed and no air is let in and 100 is completely open). The second control input is a binary variable representing whether the exploratory valve is open (1) or closed (0) to let air out.

![Ventilator diagram](https://raw.githubusercontent.com/google/deluca-lung/main/assets/2020-10-02%20Ventilator%20diagram.svg)

Each time series represents an approximately 3-second breath. The files are organized such that each row is a time step in a breath and gives the two control signals, the resulting airway pressure, and relevant attributes of the lung, described below.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - globally-unique time step identifier across an entire file
- `breath_id` - globally-unique time step for breaths
- `R` - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change `R` by changing the diameter of the straw, with higher `R` being harder to blow.
- `C` - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change `C` by changing the thickness of the balloon’s latex, with higher `C` having thinner latex and easier to blow.
- `time_step` - the actual time stamp.
- `u_in` - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.
- `u_out` - the control input for the exploratory solenoid valve. Either 0 or 1.
- `pressure` - the airway pressure measured in the respiratory circuit, measured in cmH2O.

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
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
transformers==4.53.3
wandb==0.21.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        input/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        working/
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
```

-> data/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/ventilator-pressure-prediction/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> input/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> (stopped after 10 files for performance)

# 5. Target score

0.1580959161450108

# 6. Current score

4.4522

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 11.37665) has done: 'Implemented robust handling for cases where no pretrained models are found. The `test_loop_pred` function now gracefully falls back to using class probabilities alone (or zeros if unavailable) instead of assuming at least one model, preventing the IndexError. This ensures a valid prediction array is generated and the submission CSV is written correctly.'
- What this solution (achieved 8.50184) has done: 'We define the missing `find_nearest` function (using the sorted pressure values already computed) and remove the undefined call error by placing this helper before the submission generation. No other logic is changed, preserving the original model and pipeline while ensuring a valid `.csv` is written.'
- What this solution (achieved 8.6291) has done: 'I add class‑weighting to the loss (to help the model learn rare pressure values), train a bit longer, and smooth the raw predictions per breath before mapping them to the nearest allowed pressure. These minimal tweaks keep the original architecture and training pipeline while aiming to lower the MAE toward the target.'
- What this solution (achieved 8.67926) has done: 'I remove the rolling‑mean smoothing that can blur the predictions and map the raw model outputs directly to the nearest pressure value seen in the training data. This small change keeps the model architecture and training unchanged while expected to lower the MAE, moving the score toward the target.'
- What this solution (achieved 10.86355) has done: 'The fix updates the data directory path handling so the script correctly finds the CSV files (whether they reside in `input` or `data`). It adds a small utility that selects the first existing folder and uses it for all reads, then proceeds with the baseline prediction and writes a valid `submission.csv`.'
- What this solution (achieved 5.21688) has done: 'I add a lightweight per‑group regression model (Polynomial + Linear) that uses the continuous inputs (`u_in`, `u_out`, `time_step`) to predict pressure for each `(R, C)` combination. This replaces the simple median baseline, keeping the overall script structure unchanged while providing much more accurate predictions and thus moving the MAE toward the target score. The submission file is then written as before.'
- What this solution (achieved 4.4522) has done: 'The script’s main slowdown is the full‑data fit of `GradientBoostingRegressor` (5.4 M rows × 300 trees). We keep the same feature set and target, but replace the classic GB‑R with its histogram‑based equivalent (`HistGradientBoostingRegressor`), which implements the same gradient‑boosting logic much faster on large numeric data. We also ensure the training matrix is a contiguous `float32` array to avoid unnecessary casting. All other steps (data loading, prediction, submission) stay unchanged, so the result semantics are preserved while fitting comfortably within the 600 s limit.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
)  # histogram‑based GBM, faster


class config:
    INPUT = None
    N_FOLD = 5
    BS = 1024


_possible_input_dirs = ["input", "data", "../input", "../data"]
for _dir in _possible_input_dirs:
    if os.path.isdir(_dir):
        config.INPUT = _dir
        break
if config.INPUT is None:
    raise FileNotFoundError(
        "No input directory found among: " + ", ".join(_possible_input_dirs)
    )

usecols = ["u_in", "u_out", "time_step", "R", "C", "pressure"]
dtype = {
    "u_in": np.float32,
    "u_out": np.int8,
    "time_step": np.float32,
    "R": np.int16,
    "C": np.int16,
    "pressure": np.float32,
}
train_df = pd.read_csv(
    os.path.join(config.INPUT, "train.csv"), usecols=usecols, dtype=dtype
)

test_usecols = ["u_in", "u_out", "time_step", "R", "C"]
test_dtype = {
    "u_in": np.float32,
    "u_out": np.int8,
    "time_step": np.float32,
    "R": np.int16,
    "C": np.int16,
}
test_df = pd.read_csv(
    os.path.join(config.INPUT, "test.csv"), usecols=test_usecols, dtype=test_dtype
)

features = ["u_in", "u_out", "time_step", "R", "C"]
target = "pressure"

X = train_df[features].to_numpy(dtype=np.float32, copy=False)
y = train_df[target].to_numpy(dtype=np.float32, copy=False)

gbr = HistGradientBoostingRegressor(
    max_iter=300,  # equivalent to n_estimators
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
)
gbr.fit(X, y)

test_df["pressure"] = gbr.predict(
    test_df[features].to_numpy(dtype=np.float32, copy=False)
)



## === cell 1
submission_path = os.path.join(config.INPUT, "sample_submission.csv")
df_submission = pd.read_csv(submission_path)
df_submission["pressure"] = test_df["pressure"]
df_submission.to_csv("submission.csv", index=False)
