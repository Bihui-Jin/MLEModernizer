# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import os
import random
import gc




## === cell 1
def set_seed(seed: int = 2021):
    """
    Set all relevant random seeds for reproducibility.
    """
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state




## === cell 2
def train_and_predict():
    """
    Train a histogram‑based GradientBoostingRegressor on the full training data,
    predict pressures for the test set, and write a correctly‑formatted
    submission file.
    """
    train_path = "../input/ventilator-pressure-prediction/train.csv"
    test_path = "../input/ventilator-pressure-prediction/test.csv"
    sample_sub_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    output_path = "submission.csv"

    base_features = ["R", "C", "time_step", "u_in", "u_out"]
    target_col = "pressure"

    dtype_map = {
        "R": np.int16,
        "C": np.int16,
        "time_step": np.float32,
        "u_in": np.float32,
        "u_out": np.int8,
        "pressure": np.float32,
        "breath_id": np.int32,
    }

    train_df = pd.read_csv(
        train_path,
        usecols=base_features + [target_col, "breath_id"],
        dtype={k: dtype_map[k] for k in base_features + [target_col, "breath_id"]},
    )
    test_df = pd.read_csv(
        test_path,
        usecols=base_features + ["breath_id"],
        dtype={k: dtype_map[k] for k in base_features + ["breath_id"]},
    )

    agg_cols = {
        "breath_u_in_sum": ("u_in", "sum"),
        "breath_u_in_mean": ("u_in", "mean"),
        "breath_u_in_max": ("u_in", "max"),
        "breath_u_in_min": ("u_in", "min"),
        "breath_time_step_max": ("time_step", "max"),
        "breath_time_step_min": ("time_step", "min"),
        "breath_time_step_mean": ("time_step", "mean"),
        "breath_u_out_sum": ("u_out", "sum"),
        "breath_len": ("time_step", "size"),
    }

    train_agg = train_df.groupby("breath_id").agg(**agg_cols).reset_index()
    test_agg = test_df.groupby("breath_id").agg(**agg_cols).reset_index()

    train_df = train_df.merge(train_agg, on="breath_id", how="left")
    test_df = test_df.merge(test_agg, on="breath_id", how="left")

    for df in (train_df, test_df):
        df["R_mul_C"] = (df["R"].astype(np.float32) * df["C"]).astype(np.float32)
        df["R_sq"] = (df["R"].astype(np.float32) ** 2).astype(np.float32)
        df["C_sq"] = (df["C"].astype(np.float32) ** 2).astype(np.float32)
        df["u_in_mul_u_out"] = (df["u_in"] * df["u_out"]).astype(np.float32)
        df["time_step_mul_u_in"] = (df["time_step"] * df["u_in"]).astype(np.float32)
        df["time_step_mul_u_out"] = (df["time_step"] * df["u_out"]).astype(np.float32)
        df["time_step_R"] = (df["time_step"] * df["R"]).astype(np.float32)
        df["time_step_C"] = (df["time_step"] * df["C"]).astype(np.float32)
        df["u_in_sq"] = (df["u_in"] ** 2).astype(np.float32)

        df["u_in_cum"] = df.groupby("breath_id")["u_in"].cumsum().astype(np.float32)
        df["u_out_cum"] = df.groupby("breath_id")["u_out"].cumsum().astype(np.float32)
        df["u_in_lag"] = (
            df.groupby("breath_id")["u_in"].shift(1).fillna(0).astype(np.float32)
        )
        df["u_in_diff"] = (df["u_in"] - df["u_in_lag"]).astype(np.float32)

        df["u_in_cum_norm"] = (df["u_in_cum"] / df["breath_u_in_sum"]).astype(
            np.float32
        )
        df["u_in_cum_per_len"] = (df["u_in_cum"] / df["breath_len"]).astype(np.float32)
        df["time_step_norm"] = (df["time_step"] / df["breath_len"]).astype(np.float32)

        df["u_out_lag"] = (
            df.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int8)
        )
        df["u_out_diff"] = (df["u_out"] - df["u_out_lag"]).astype(np.int8)
        df["u_out_sq"] = (df["u_out"] ** 2).astype(np.int8)

    feature_cols = base_features + [
        "R_mul_C",
        "R_sq",
        "C_sq",
        "u_in_mul_u_out",
        "time_step_mul_u_in",
        "time_step_mul_u_out",
        "time_step_R",
        "time_step_C",
        "u_in_sq",
        "u_in_cum",
        "u_out_cum",
        "u_in_lag",
        "u_in_diff",
        "u_in_cum_norm",
        "u_in_cum_per_len",
        "time_step_norm",
        "u_out_lag",
        "u_out_diff",
        "u_out_sq",
        "breath_u_in_sum",
        "breath_u_in_mean",
        "breath_u_in_max",
        "breath_u_in_min",
        "breath_time_step_max",
        "breath_time_step_min",
        "breath_time_step_mean",
        "breath_u_out_sum",
        "breath_len",
    ]

    X = train_df[feature_cols].values.astype(np.float32)
    y = train_df[target_col].values.astype(np.float32)

    pressure_min = y.min()
    pressure_max = y.max()

    del train_df
    gc.collect()

    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        max_iter=3500,
        learning_rate=0.015,
        max_depth=12,
        random_state=42,
        verbose=0,
    )
    model.fit(X, y)

    test_features = test_df[feature_cols].values.astype(np.float32)
    del test_df
    gc.collect()

    raw_preds = model.predict(test_features)
    preds = np.clip(raw_preds, pressure_min, pressure_max)

    submission = pd.read_csv(sample_sub_path)
    submission["pressure"] = preds.astype(float)
    submission.to_csv(output_path, index=False)
    print(f"Submission written to {output_path}")




## === cell 3
train_and_predict()
