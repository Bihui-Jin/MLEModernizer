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

0.1366266368106691

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.42284) has done: 'I fixed the crash that occurred when the blending directory contained no prediction files by adding safety checks and a fallback baseline that predicts the mean pressure for each (R, C) combination from the training data. The `g` function now returns a zero‑filled submission when no files are found, and the main cell calls `g` only if the directory exists; otherwise it builds the simple baseline submission. This ensures a valid `submission.csv` of the correct shape is always written, allowing the notebook to finish without errors and produce a usable score.'
- What this solution (achieved 7.55219) has done: 'I add a lightweight linear‑regression model trained on a sampled subset of the training data and use its predictions (rounded to the nearest observed pressure) as the baseline when no blending files are found. This replaces the simple per‑(R,C) mean, which was giving a very high MAE (≈8.4), with a model that captures the relationship of pressure to R, C, u_in, u_out and time_step, moving the score much closer to the target.'
- What this solution (achieved 5.73365) has done: 'The changes replace the simple linear‑regression baseline with a polynomial‑feature ridge model trained on a larger random subset of the data, which captures interactions between the control signals and lung attributes and therefore yields predictions much closer to the true pressures, moving the MAE toward the target value. The rest of the pipeline (blending fallback, CSV handling, rounding to the nearest observed pressure) remains unchanged.'
- What this solution (achieved 4.19553) has done: 'I replace the simple ridge‑polynomial baseline with a stronger tree‑based model (HistGradientBoostingRegressor) trained on a random subset of the full training data. This keeps the overall pipeline unchanged while providing a much more accurate prediction that is still rounded to the nearest observed pressure, moving the MAE dramatically closer to the target. The rest of the code (blending logic, CSV handling, seeding) remains the same.'
- What this solution (achieved 4.15213) has done: 'I train the tree model on the full training set (instead of a 20 % random slice) and include the `breath_id` feature, which provides useful information about each breath without changing the overall pipeline. Using all data and an extra informative feature should noticeably lower the MAE, moving the score toward the target while keeping the original architecture and logic unchanged.'
- What this solution (achieved 4.04284) has done: 'I add two small, targeted changes: (1) train the tree on polynomial‑expanded features (degree 2) to give the model richer nonlinear information, and (2) switch the HistGradientBoostingRegressor to the `absolute_error` loss which directly optimises MAE. These tweaks keep the overall pipeline and model type unchanged but should move the validation MAE much closer to the target.'
- What this solution (achieved 4.05097) has done: 'I add the `breath_id` feature to the model’s input and slightly strengthen the HistGradientBoostingRegressor (more depth, more iterations, lower learning‑rate) so it captures more of the underlying dynamics. These minimal tweaks keep the overall pipeline and model type unchanged while giving the regressor richer information, which should lower the MAE toward the target.'
- What this solution (achieved 3.98486) has done: 'I add feature scaling before the polynomial expansion and slightly strengthen the HistGradientBoostingRegressor (deeper trees, a bit more boosting rounds and a lower learning rate). Scaling makes the tree learner see features on comparable ranges, which usually improves MAE, and the modest model tweaks give it more capacity without breaking the original pipeline.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
import random
from random import random as rd
import gc
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

DTYPES = {
    "R": np.int16,
    "C": np.int16,
    "breath_id": np.int32,
    "id": np.int16,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.int8,
    "pressure": np.float32,
}
df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv", dtype=DTYPES
)

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction):
    """Map a single continuous prediction to the nearest pressure value seen in training."""
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return sorted_pressures[-1]
    elif insert_idx == 0:
        return sorted_pressures[0]
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return (
        lower_val
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else upper_val
    )


def map_to_nearest(arr):
    """Vectorized version of `find_nearest` for an array of predictions."""
    idx = np.searchsorted(sorted_pressures, arr, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)

    lower_idx = np.maximum(idx - 1, 0)
    lower = sorted_pressures[lower_idx]
    upper = sorted_pressures[idx]

    use_lower = (arr - lower) <= (upper - arr)
    return np.where(use_lower, lower, upper)


def set_seed(seed=2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def wc(input_list):
    """Weighted combination of two prediction files based on public LB scores embedded in filenames."""
    l = []
    for i in range(len(input_list)):
        public_lb_score = int(input_list[i].split("/")[-1].split(".")[1].split(" ")[0])
        l.append(public_lb_score)
        input_list[i] = pd.read_csv(input_list[i]).pressure.values.ravel()
    output = 0
    l_sum = sum(l)
    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def add_features(df):
    """Engineer simple intra‑breath dynamics."""
    df = df.copy()
    df["cum_u_in"] = df.groupby("breath_id")["u_in"].cumsum()
    df["diff_u_in"] = df.groupby("breath_id")["u_in"].diff().fillna(0)
    df["time_step_sq"] = df["time_step"] ** 2
    return df


def g(dp):
    """Blend predictions from CSV files inside *dp*.
    Returns a DataFrame with columns id, pressure.
    If no files are found, creates a zero‑filled submission.
    """
    file_paths = [p for p in glob.iglob(f"{dp}/*") if p.lower().endswith(".csv")]
    file_count = len(file_paths)

    if file_count == 0:
        output = pd.read_csv(
            "../input/ventilator-pressure-prediction/sample_submission.csv",
            dtype={"id": np.int16, "pressure": np.float32},
        )
        output["pressure"] = 0.0
        output.to_csv("submission.csv", index=False)
        return output

    preds = []
    for p in file_paths:
        preds.append(
            pd.read_csv(p, dtype={"pressure": np.float32}).pressure.values.ravel()
        )
    mean_pred = np.mean(np.vstack(preds), axis=0)

    mean_pred = map_to_nearest(mean_pred)

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv",
        dtype={"id": np.int16, "pressure": np.float32},
    )
    output.pressure = mean_pred
    output.to_csv("submission.csv", index=False)
    return output


def train_tree_model(sample_frac=0.2):
    """Train a HistGradientBoostingRegressor on a sampled subset of the data.
    Uses early stopping and a modest increase in capacity to improve MAE while keeping runtime reasonable.
    """
    set_seed(42)
    df_sample = df_train.sample(frac=sample_frac, random_state=42)
    df_feat = add_features(df_sample)

    base_features = [
        "R",
        "C",
        "u_in",
        "u_out",
        "time_step",
        "time_step_sq",
        "breath_id",
        "cum_u_in",
        "diff_u_in",
    ]
    X_base = df_feat[base_features]
    y = df_feat["pressure"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_base)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X = poly.fit_transform(X_scaled)

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_depth=12,
        learning_rate=0.01,
        max_iter=1500,
        early_stopping=True,
        random_state=42,
        l2_regularization=0.1,
    )
    model.fit(X, y)
    return model, poly, scaler


def baseline_submission():
    """Generate baseline predictions using the enhanced tree model and output raw values."""
    model, poly, scaler = train_tree_model()
    test_path = "../input/ventilator-pressure-prediction/test.csv"
    df_test = pd.read_csv(test_path, dtype=DTYPES)

    df_test = add_features(df_test)

    base_features = [
        "R",
        "C",
        "u_in",
        "u_out",
        "time_step",
        "time_step_sq",
        "breath_id",
        "cum_u_in",
        "diff_u_in",
    ]
    X_test_scaled = scaler.transform(df_test[base_features])
    X_test = poly.transform(X_test_scaled)

    preds = model.predict(X_test)

    preds = map_to_nearest(preds)

    df_test["pressure"] = preds

    submission = df_test[["id", "pressure"]].copy()
    submission.to_csv("submission.csv", index=False)
    return submission




## === cell 1
blending_dir = "../input/gb-data-blending-recover"
if os.path.isdir(blending_dir):
    g(blending_dir)
else:
    baseline_submission()
