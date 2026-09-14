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
sklearn-pandas==2.2.0

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

0.1433856476677093

# 6. Current score

10.86378

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 10.86378) has done: 'I fix the IndexError in `wc` by handling cases with fewer than two scores, and make `g` robust when the target directory does not contain any qualifying high‑score files. If no files are found, the function now creates a baseline submission using the median pressure from the training set. The script always write a valid `submission.csv` file ready for Kaggle.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import gc
import numpy as np
import pandas as pd


def set_seed(seed=2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def find_nearest(prediction, sorted_vals):
    """Return the nearest value in `sorted_vals` to `prediction`."""
    insert_idx = np.searchsorted(sorted_vals, prediction)
    if insert_idx == len(sorted_vals):
        return sorted_vals[-1]
    if insert_idx == 0:
        return sorted_vals[0]
    lower = sorted_vals[insert_idx - 1]
    upper = sorted_vals[insert_idx]
    return lower if abs(lower - prediction) < abs(upper - prediction) else upper


def wc(input_list):
    """
    Blend predictions from a list of files that passed the score filter.
    Handles 0, 1, or 2+ inputs safely.
    """
    l = []
    allow = [1348, 1359, 1758]  # allowed public leaderboard scores
    for i in range(len(input_list)):
        public_lb_score = int(input_list[i].split("/")[-1].split(".")[1].split(" ")[0])
        if public_lb_score in allow:
            l.append(public_lb_score)
            input_list[i] = pd.read_csv(input_list[i])["pressure"].values
        else:
            continue

    if len(input_list) == 0:
        return np.zeros_like(
            pd.read_csv(
                "../input/ventilator-pressure-prediction/sample_submission.csv"
            )["pressure"].values
        )

    if len(input_list) == 1:
        return input_list[0]

    l_sum = sum(l)
    if l_sum == 0:
        weight1 = 0.5
    else:
        weight1 = (l[1] / l_sum) + 0.1 if len(l) > 1 else 0.5
    weight2 = 1.0 - weight1
    output = input_list[0] * weight1 + input_list[1] * weight2
    return output


def blend_predictions(dp):
    """
    Main blending routine.
    If the directory contains no qualifying files, fall back to a simple median baseline.
    The final CSV is written as `submission.csv` in the current working directory.
    """
    allowed_scores = {1348, 1359, 1758}
    candidate_files = []
    for path in glob.iglob(f"{dp}/*"):
        try:
            score_part = path.split("/")[-1].split(".")[1].split(" ")[0]
            score = int(score_part)
            if score in allowed_scores:
                candidate_files.append(path)
        except Exception:
            continue

    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    submission = pd.read_csv(sample_path)

    if not candidate_files:
        train_path = "../input/ventilator-pressure-prediction/train.csv"
        train_df = pd.read_csv(train_path)
        median_pressure = train_df["pressure"].median()
        submission["pressure"] = median_pressure
        submission.to_csv("submission.csv", index=False)
        print("No high‑score files found – baseline submission created.")
        return

    splits = 2
    candidate_files.sort()
    groups = []
    for i in range(splits):
        start = i * round(len(candidate_files) / splits)
        end = (
            None if i == splits - 1 else (i + 1) * round(len(candidate_files) / splits)
        )
        groups.append(candidate_files[start:end])

    group_preds = [wc(group) for group in groups]

    loop_time = 125
    pred_list = []
    for loop_idx in range(loop_time):
        set_seed(loop_idx)
        raw_weights = [random.random() for _ in group_preds]
        weight_sum = sum(raw_weights)
        weights = [w / weight_sum for w in raw_weights]
        weighted_pred = sum(
            p * w for p, w in zip(group_preds, sorted(weights, reverse=True))
        )
        pred_list.append(weighted_pred)

    median_pred = np.median(np.vstack(pred_list), axis=0)
    train_pressures = np.sort(
        pd.read_csv("../input/ventilator-pressure-prediction/train.csv")[
            "pressure"
        ].unique()
    )
    submission["pressure"] = [find_nearest(p, train_pressures) for p in median_pred]
    submission.to_csv("submission.csv", index=False)
    print("Blended submission written to submission.csv")




## === cell 1
blend_predictions("../input/ventilator-pressure-high-score-submissions")
