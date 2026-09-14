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

0.143744859069948

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.73432) has done: 'The script now safely handles the missing prediction directory by falling back to a simple baseline model: it computes the mean pressure for each combination of `R`, `C`, and rounded `u_in` from the training data and uses these averages to predict the test set. The predictions are rounded to the nearest existing pressure value and saved as `submission.csv`, ensuring a valid Kaggle submission file even when no external prediction files are present. This fix resolves the runtime error and provides a reasonable score without altering the core ensemble logic.'
- What this solution (achieved 7.54837) has done: 'I replace the simple mean‑based baseline with a lightweight linear regression that uses the numeric features (`u_in`, `u_out`, `R`, `C`, `time_step`). This small change keeps the overall pipeline unchanged but should dramatically lower the MAE, moving the score toward the target. The rest of the script (blending external predictions, fallback handling, CSV writing) remains intact.'
- What this solution (achieved 4.04821) has done: 'The script is optimized by replacing the slow `np.vectorize` nearest‑value lookup with a fully vectorized implementation, freeing memory after training, and adding brief comments explaining the speed gains while keeping all original logic unchanged.'
- What this solution (achieved 3.93214) has done: 'The update focuses on the heavy fallback path `baseline_prediction`. It reads only the necessary columns, uses the smallest appropriate dtypes, and reduces the ExtraTrees model to a still‑accurate but faster configuration (300 → 200 trees, modest max_depth). All feature engineering stays vectorised, and the rest of the script—including the blending logic—remains unchanged.'

# 9. Code solution

## === cell 0
import os
import gc
import glob
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def set_seed(seed=2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed()



## === cell 1
df_train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction):
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


def find_nearest_arr(preds):
    """Map an array of predictions to the nearest pressure values efficiently."""
    idx = np.searchsorted(sorted_pressures, preds)
    res = np.empty_like(preds, dtype=sorted_pressures.dtype)

    mask_low = idx == 0
    mask_high = idx == total_pressures_len
    res[mask_low] = sorted_pressures[0]
    res[mask_high] = sorted_pressures[-1]

    mask_mid = ~mask_low & ~mask_high
    idx_mid = idx[mask_mid]
    lower = sorted_pressures[idx_mid - 1]
    upper = sorted_pressures[idx_mid]
    pred_mid = preds[mask_mid]
    choose_lower = np.abs(lower - pred_mid) < np.abs(upper - pred_mid)
    res[mask_mid] = np.where(choose_lower, lower, upper)

    return res


def wc(input_list):
    l = []
    for i in range(len(input_list)):
        public_lb_score = int(input_list[i].split("/")[-1].split(".")[1].split(" ")[0])
        l.append(public_lb_score)
        input_list[i] = (pd.read_csv(input_list[i]).pressure).ravel()
    output = 0
    l_sum = sum(l)
    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = (l[1] / l_sum) + 0.15
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    """
    Optimised blending:
    * Simplified split into two halves using np.array_split.
    * Vectorised the 154 weight‑generation / blending loops.
    * Applied the fast NumPy nearest‑pressure mapping.
    """
    files = sorted(glob.glob(f"{dp}/*"))
    if not files:
        raise FileNotFoundError(f"No prediction files found in {dp}")

    flist = [wc(part.tolist()) for part in np.array_split(files, 2)]

    loop_time = 154
    n_models = len(flist)

    pred_stack = np.stack(flist, axis=0)  # (2, 603600)

    rng = np.random.RandomState()
    weight_matrix = np.empty((loop_time, n_models), dtype=np.float64)
    for itr in range(loop_time):
        rng.seed(itr)
        w = rng.rand(n_models)
        w /= w.sum()
        weight_matrix[itr] = np.sort(w)[::-1]

    blended = weight_matrix @ pred_stack  # broadcasting matrix multiplication

    median_preds = np.median(blended, axis=0)

    median_preds = find_nearest_arr(median_preds)

    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    output = pd.read_csv(sample_path)
    output["pressure"] = median_preds
    output.to_csv("submission.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a


def baseline_prediction():
    """Improved baseline with bias correction and higher‑capacity ExtraTrees."""
    dtype_train = {
        "id": np.int32,
        "breath_id": np.int32,
        "R": np.int8,
        "C": np.int8,
        "time_step": np.float32,
        "u_in": np.float32,
        "u_out": np.int8,
        "pressure": np.float32,
    }
    dtype_test = {
        "id": np.int32,
        "breath_id": np.int32,
        "R": np.int8,
        "C": np.int8,
        "time_step": np.float32,
        "u_in": np.float32,
        "u_out": np.int8,
    }

    train_df = pd.read_csv(
        "../input/ventilator-pressure-prediction/train.csv",
        dtype=dtype_train,
        usecols=list(dtype_train.keys()),
    )
    test_df = pd.read_csv(
        "../input/ventilator-pressure-prediction/test.csv",
        dtype=dtype_test,
        usecols=list(dtype_test.keys()),
    )

    base_features = ["u_in", "u_out", "R", "C", "time_step"]

    for df in (train_df, test_df):
        df["u_in_sq"] = df["u_in"] ** 2
        df["time_step_sq"] = df["time_step"] ** 2
        df["u_in_R"] = df["u_in"] * df["R"]
        df["u_in_C"] = df["u_in"] / (df["C"] + 1e-6)
        df["u_in_time"] = df["u_in"] * df["time_step"]

    feature_cols = base_features + [
        "u_in_sq",
        "time_step_sq",
        "u_in_R",
        "u_in_C",
        "u_in_time",
    ]

    X = train_df[feature_cols].values.astype(np.float32, copy=False)
    y = train_df["pressure"].values.astype(np.float32, copy=False)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.10, random_state=42)

    model = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    val_preds = model.predict(X_val)
    bias = np.mean(val_preds - y_val)

    del train_df, X, y, X_tr, y_tr
    gc.collect()

    X_test = test_df[feature_cols].values.astype(np.float32, copy=False)
    test_preds = model.predict(X_test) - bias  # bias‑corrected predictions

    test_preds = find_nearest_arr(test_preds)

    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    submission = pd.read_csv(sample_path)
    submission["pressure"] = test_preds
    submission.to_csv("submission.csv", index=False)




## === cell 2
try:
    g("../input/gb-data-blending-recover")
except FileNotFoundError:
    baseline_prediction()
