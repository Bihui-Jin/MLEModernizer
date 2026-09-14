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

0.1423689560412636

# 6. Current score

5.99565

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.03413) has done: 'I added the missing `glob` import, implemented a lightweight linear‑regression baseline (using NumPy lstsq) to generate predictions when no external files are present, and combined these model predictions with the existing median‑blending output. The script now reliably writes a correctly formatted `submission.csv` even when the blending directory is empty, fixing the NameError and ensuring a valid submission file is produced.'
- What this solution (achieved 8.14882) has done: 'I correct the length mismatch by aligning all prediction arrays to the number of rows in the test/sample‑submission file instead of the training set size. This fixes the ValueError when assigning predictions to the submission DataFrame while keeping the original model logic unchanged.'
- What this solution (achieved 5.99565) has done: 'I add a stronger gradient‑boosting model (still a linear‑type ensemble, not a completely new architecture) and give it the largest weight in the final blend. This keeps the original group‑mean and linear baselines for safety while markedly improving predictive power, moving the MAE much closer to the target. I also tidy up imports and ensure the submission file is always written correctly.'

# 9. Code solution

## === cell 0
import os
import gc
import random
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
from random import random as rd
from sklearn.ensemble import GradientBoostingRegressor


def find_nearest(prediction):
    """Map a continuous prediction to the nearest pressure value seen in training."""
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return sorted_pressures[-1]
    if insert_idx == 0:
        return sorted_pressures[0]
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return (
        lower_val
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else upper_val
    )


def set_seed(seed=2021):
    """Fix random seeds for reproducibility."""
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """Weighted blend of up to two prediction files."""
    scores = []
    arrays = []
    for f in input_list:
        try:
            score = int(f.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            score = 0
        scores.append(score)
        try:
            arr = pd.read_csv(f).pressure.values.ravel()
        except Exception:
            arr = np.zeros(test_len)  # use test length as fallback
        arrays.append(arr)

    if not arrays:
        return np.zeros(test_len)

    if len(arrays) == 1:
        return arrays[0]

    l_sum = sum(scores)
    weight1 = (scores[1] / l_sum) + 0.15 if l_sum != 0 else 0.5
    weight2 = 1 - weight1
    return arrays[0] * weight1 + arrays[1] * weight2


def blend(a, b):
    """Blend two existing submission files equally."""
    a_df = pd.read_csv(a)
    b_df = pd.read_csv(b)
    a_df.pressure = a_df.pressure * 0.5 + b_df.pressure * 0.5
    a_df["pressure"] = a_df["pressure"].apply(find_nearest)
    a_df.to_csv("blend.csv", index=False)
    return a_df




## === cell 1
df_train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)

group_mean_pressure = (
    df_train.groupby(["R", "C"])["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "group_mean"})
)


def simple_linear_model():
    """Fit a plain linear regression (with intercept) using NumPy."""
    feats = df_train[["R", "C", "time_step", "u_in", "u_out"]].values.astype(np.float64)
    X = np.column_stack([np.ones(feats.shape[0]), feats])  # add intercept
    y = df_train["pressure"].values.astype(np.float64)
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs


lin_coeffs = simple_linear_model()


def predict_with_linear_model(df):
    """Generate predictions for a dataframe using the pre‑computed linear model."""
    feats = df[["R", "C", "time_step", "u_in", "u_out"]].values.astype(np.float64)
    X = np.column_stack([np.ones(feats.shape[0]), feats])
    return X @ lin_coeffs


def predict_group_mean(df):
    """Return the mean‑pressure per (R, C) pair; fallback to global median."""
    merged = df.merge(group_mean_pressure, on=["R", "C"], how="left")
    median_pressure = np.median(df_train["pressure"])
    merged["group_mean"].fillna(median_pressure, inplace=True)
    return merged["group_mean"].values


def train_gbr():
    """Train a GradientBoostingRegressor on a subsample of the training data."""
    sample_df = df_train.sample(frac=0.10, random_state=2021)
    X = sample_df[["R", "C", "time_step", "u_in", "u_out"]].values
    y = sample_df["pressure"].values
    gbr = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=2021,
        loss="huber",
    )
    gbr.fit(X, y)
    return gbr


gbr_model = train_gbr()


def predict_with_gbr(df):
    """Predict using the trained GradientBoostingRegressor."""
    X = df[["R", "C", "time_step", "u_in", "u_out"]].values
    return gbr_model.predict(X)




## === cell 2
def g(dp):
    """Generate a submission by blending existing predictions, a linear model,
    the group‑mean baseline, and a GradientBoostingRegressor."""
    sample_sub_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    output_template = pd.read_csv(sample_sub_path)
    global test_len
    test_len = len(output_template)

    all_files = sorted([p for p in glob.iglob(f"{dp}/*") if os.path.isfile(p)])
    if not all_files:
        test_df = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
        pred_linear = predict_with_linear_model(test_df)
        pred_gbr = predict_with_gbr(test_df)
        baseline = np.full(pred_linear.shape[0], df_train.pressure.median())
        pred_median = baseline
    else:
        splits = len(all_files) // 2 or 1
        chunk_size = max(1, round(len(all_files) / splits))
        flist = []
        for i in range(splits):
            start = i * chunk_size
            end = None if i == splits - 1 else (i + 1) * chunk_size
            group = all_files[start:end]
            flist.append(wc(group))
        pred_median = np.median(np.column_stack(flist), axis=1)

        test_df = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
        pred_linear = predict_with_linear_model(test_df)
        pred_gbr = predict_with_gbr(test_df)

    test_df = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
    pred_group = predict_group_mean(test_df)

    def align(vec):
        if vec.shape[0] != test_len:
            if vec.shape[0] > test_len:
                return vec[:test_len]
            else:
                pad = np.full(test_len - vec.shape[0], np.median(vec))
                return np.concatenate([vec, pad])
        return vec

    pred_median = align(pred_median)
    pred_linear = align(pred_linear)
    pred_group = align(pred_group)
    pred_gbr = align(pred_gbr)

    final_pred = (
        0.2 * pred_median + 0.2 * pred_linear + 0.2 * pred_group + 0.4 * pred_gbr
    )

    output = output_template.copy()
    output.pressure = final_pred
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv("submission.csv", index=False)
    print("Submission written to submission.csv")


g("../input/gb-data-blending-recover")
