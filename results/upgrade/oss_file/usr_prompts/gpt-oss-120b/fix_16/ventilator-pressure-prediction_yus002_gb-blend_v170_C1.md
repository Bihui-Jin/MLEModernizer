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

0.1358631054497741

# 6. Current score

3.77573

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.42284) has done: 'I fixed the runtime error that occurred when no blending prediction files were found. The `g` function now checks for an empty file list and falls back to a simple baseline model that predicts pressure as the mean pressure for each `(R, C)` lung‑attribute pair (using the training data). This guarantees a correctly‑sized array and produces a valid `submission.csv` file with the required columns, ensuring the notebook runs end‑to‑end without errors.'
- What this solution (achieved 6.89239) has done: 'The baseline is upgraded from a simple (R, C) mean to a finer‑grained grouping that also bins the inspiratory input `u_in`.  
By averaging the pressure for each `(R, C, u_in_bin)` combination we capture more of the underlying dynamics, then map the predictions back to the nearest valid pressure value. This change keeps the original workflow (still a deterministic, non‑learned baseline) but dramatically reduces the MAE, moving the score much closer to the target.'
- What this solution (achieved 4.09794) has done: 'I make the baseline deterministic grouping much finer‑grained: round `u_in` to one decimal place and bucket `time_step` into centisecond bins, then compute the mean pressure for each `(R, C, u_in_bin, u_out, time_bin)` combination. Missing groups fall back to the mean for `(R, C, u_in_bin, u_out)` and finally to the overall mean. This keeps the original non‑learned approach while dramatically improving prediction accuracy, moving the MAE much closer to the target.'
- What this solution (achieved 4.02441) has done: 'I replace the deterministic grouping baseline with a fast RandomForest model that learns from the raw features (R, C, u_in, u_out, time_step). This learned baseline is still called `baseline_prediction()` so the existing workflow stays intact, but it should dramatically lower MAE toward the target. I also add a small random‑sample fallback to keep training time reasonable on the large dataset.'
- What this solution (achieved 3.77573) has done: 'The changes keep the same overall approach (random‑sampled training, feature engineering, RandomForest, and nearest‑pressure mapping) but reduce work‑time by sampling a smaller fraction of rows and by using a slightly smaller bootstrap sample per tree. Converting DataFrames to NumPy with `to_numpy` avoids an unnecessary copy, and the seed is set once before sampling. These tweaks preserve correctness while staying well under the 600 s limit.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import random
import glob
import gc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def find_nearest(prediction):
    """Fallback scalar version kept for compatibility."""
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


def vectorized_nearest(preds):
    """Map an array of predictions to the nearest valid pressure efficiently."""
    idx = np.searchsorted(sorted_pressures, preds, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)

    left_idx = np.maximum(idx - 1, 0)
    right_idx = idx

    left_vals = sorted_pressures[left_idx]
    right_vals = sorted_pressures[right_idx]

    use_left = np.abs(preds - left_vals) <= np.abs(right_vals - preds)
    return np.where(use_left, left_vals, right_vals)


def set_seed(seed=2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def wc(input_list):
    l = []
    for i in range(len(input_list)):
        public_lb_score = int(input_list[i].split("/")[-1].split(".")[1].split(" ")[0])
        l.append(public_lb_score)
        df = pd.read_csv(
            input_list[i],
            usecols=["pressure"],
            dtype={"pressure": "float32"},
            memory_map=True,
        )
        input_list[i] = df.pressure.values.ravel()
    if len(input_list) == 1:
        return input_list[0]
    weight1 = (l[1] / sum(l)) + 0.1
    weight2 = 1 - weight1
    return input_list[0] * weight1 + input_list[1] * weight2


def g(dp):
    """
    Blending logic – abort early when many prediction files are present.
    This avoids the previous O(loops × files) cost that caused time‑outs.
    """
    files = sorted(glob.iglob(f"{dp}/*"))
    if not files or len(files) > 5:  # tighter guard for speed
        return None

    file_count = len(files)
    loop_time = 154
    splits = file_count // 2
    flist = []

    for i in range(splits):
        start = i * round(len(files) / splits)
        end = (i + 1) * round(len(files) / splits) if i < splits - 1 else len(files)
        flist.append(wc(files[start:end]))

    flist_arr = np.vstack(flist)  # shape: (splits, n_predictions)

    pred_list = []
    for seed in range(loop_time):
        rng = np.random.RandomState(seed)
        weight = rng.rand(splits)
        weight_sum = weight.sum()
        weight = weight / weight_sum
        weight = -np.sort(-weight)
        pred = np.dot(weight, flist_arr)
        pred_list.append(pred)

    pred_stack = np.vstack(pred_list)  # shape: (loop_time, n_predictions)
    median_pred = np.median(pred_stack, axis=0)
    mean_pred = np.mean(pred_stack, axis=0)
    blended = 0.8 * median_pred + 0.2 * mean_pred
    blended = vectorized_nearest(blended)
    return blended


def blend(a, b):
    a = pd.read_csv(
        a, usecols=["pressure"], dtype={"pressure": "float32"}, memory_map=True
    )
    b = pd.read_csv(
        b, usecols=["pressure"], dtype={"pressure": "float32"}, memory_map=True
    )
    a.pressure = a.pressure * 0.7 + b.pressure * 0.3
    a.pressure = vectorized_nearest(a.pressure.values)
    a.to_csv("blend.csv", index=False)
    return a


def avg(dp):
    input_list = [
        pd.read_csv(
            p,
            usecols=["pressure"],
            dtype={"pressure": "float32"},
            memory_map=True,
        ).pressure.values.ravel()
        for p in glob.iglob(f"{dp}/*")
    ]
    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv",
        memory_map=True,
    )
    output.pressure = np.median(np.vstack(input_list), axis=0)
    output.pressure = vectorized_nearest(output.pressure.values)
    output.to_csv("avg.csv", index=False)
    return output


df_train_pressures = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    usecols=["pressure"],
    dtype={"pressure": "float32"},
    memory_map=True,
)
sorted_pressures = np.sort(df_train_pressures["pressure"].unique())
total_pressures_len = len(sorted_pressures)
del df_train_pressures
gc.collect()




## === cell 1
def _chunked_random_sample(
    csv_path, usecols, dtypes, frac=0.5, seed=42, chunksize=500_000
):
    """Read CSV in chunks and keep each row with probability `frac` (deterministic)."""
    rng = np.random.RandomState(seed)
    sampled_chunks = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype=dtypes,
        memory_map=True,
        chunksize=chunksize,
    ):
        mask = rng.rand(len(chunk)) < frac
        if mask.any():
            sampled_chunks.append(chunk[mask])
    if sampled_chunks:
        return pd.concat(sampled_chunks, ignore_index=True)
    else:
        return pd.DataFrame(columns=usecols)


def baseline_prediction():
    """
    Learned baseline using an enhanced RandomForestRegressor.
    - Features: original + simple interaction/squared terms.
    - Uses 30 % of the training data (reduced from 50 %) for a safer speed/accuracy trade‑off.
    - Predictions are still mapped to the nearest valid pressure value.
    """
    dtypes_train = {
        "R": "int8",
        "C": "int8",
        "u_in": "float32",
        "u_out": "int8",
        "time_step": "float32",
        "pressure": "float32",
        "breath_id": "int32",
        "id": "int32",
    }
    dtypes_test = {
        "R": "int8",
        "C": "int8",
        "u_in": "float32",
        "u_out": "int8",
        "time_step": "float32",
        "breath_id": "int32",
        "id": "int32",
    }
    usecols_train = ["R", "C", "u_in", "u_out", "time_step", "pressure", "id"]
    usecols_test = ["R", "C", "u_in", "u_out", "time_step", "id"]

    train_path = "../input/ventilator-pressure-prediction/train.csv"
    test_path = "../input/ventilator-pressure-prediction/test.csv"

    train = _chunked_random_sample(
        train_path,
        usecols=usecols_train,
        dtypes=dtypes_train,
        frac=0.3,
        seed=42,
        chunksize=500_000,
    )
    test = pd.read_csv(
        test_path,
        usecols=usecols_test,
        dtype=dtypes_test,
        memory_map=True,
    )

    train["u_in_time"] = train["u_in"] * train["time_step"]
    test["u_in_time"] = test["u_in"] * test["time_step"]
    train["u_in_sq"] = train["u_in"] ** 2
    test["u_in_sq"] = test["u_in"] ** 2
    train["time_sq"] = train["time_step"] ** 2
    test["time_sq"] = test["time_step"] ** 2

    feature_cols = [
        "R",
        "C",
        "u_in",
        "u_out",
        "time_step",
        "u_in_time",
        "u_in_sq",
        "time_sq",
    ]

    X_train = train[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_train = train["pressure"].to_numpy(dtype=np.float32, copy=False)
    X_test = test[feature_cols].to_numpy(dtype=np.float32, copy=False)

    set_seed(42)
    model = RandomForestRegressor(
        n_estimators=150,  # unchanged
        max_depth=None,
        min_samples_leaf=1,
        max_samples=0.5,
        bootstrap=True,
        n_jobs=5,
        random_state=42,
        verbose=0,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    preds = vectorized_nearest(preds)

    submission = pd.DataFrame({"id": test["id"], "pressure": preds})
    submission.to_csv("submission.csv", index=False)
    print(
        "Learned baseline submission written to submission.csv (rows:",
        len(submission),
        ")",
    )

    del train, test, X_train, y_train, X_test, model, preds
    gc.collect()
    return submission


def main():
    blended = g("../input/gb-data-blending-recover")
    if blended is not None:
        sample = pd.read_csv(
            "../input/ventilator-pressure-prediction/sample_submission.csv",
            memory_map=True,
        )
        sample["pressure"] = blended
        sample["pressure"] = vectorized_nearest(sample["pressure"].values)
        sample.to_csv("submission.csv", index=False)
        print("Blended submission written to submission.csv")
    else:
        baseline_prediction()


if __name__ == "__main__":
    main()
