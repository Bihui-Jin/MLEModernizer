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
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from joblib import Parallel, delayed  # parallelize group model training

TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"
SUBMISSION_PATH = "submission.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)


def add_features(df):
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)
    df["cum_u_in"] = (df["u_in"] * df["dt"]).groupby(df["breath_id"]).cumsum()
    df["u_in_R"] = df["u_in"] * df["R"]
    df["u_in_C"] = df["u_in"] * df["C"]
    df["cum_u_in_R"] = df["cum_u_in"] * df["R"]
    df["cum_u_in_C"] = df["cum_u_in"] * df["C"]
    df["flow"] = df["u_in"] * df["dt"]
    df["vol_over_C"] = df["cum_u_in"] / df["C"]
    df["u_in_over_R"] = df["u_in"] / df["R"]
    df["breath_total_volume"] = df.groupby("breath_id")["cum_u_in"].transform("max")
    df["breath_len"] = df.groupby("breath_id")["time_step"].transform("max")
    df["u_in_sq"] = df["u_in"] ** 2
    df["time_step_sq"] = df["time_step"] ** 2
    df["dt_sq"] = df["dt"] ** 2
    df["cum_u_in_sq"] = df["cum_u_in"] ** 2
    df["flow_sq"] = df["flow"] ** 2
    df["frac_cum_u_in"] = df["cum_u_in"] / df["breath_total_volume"].replace(0, np.nan)
    df["time_frac"] = df["time_step"] / df["breath_len"].replace(0, np.nan)
    df["frac_cum_u_in"] = df["frac_cum_u_in"].fillna(0)
    df["time_frac"] = df["time_frac"].fillna(0)
    df["u_in_u_out"] = df["u_in"] * df["u_out"]
    df["dt_u_out"] = df["dt"] * df["u_out"]
    df["u_in_dt"] = df["u_in"] * df["dt"]
    df["dt_R"] = df["dt"] * df["R"]
    df["dt_C"] = df["dt"] * df["C"]
    df["u_in_R_C"] = df["u_in"] * df["R"] * df["C"]
    df["cum_u_in_R_C"] = df["cum_u_in"] * df["R"] * df["C"]
    df["flow_R"] = df["flow"] * df["R"]
    df["flow_C"] = df["flow"] * df["C"]
    return df


train_df = add_features(train_df)
test_df = add_features(test_df)

features = [
    "u_in",
    "u_out",
    "time_step",
    "dt",
    "cum_u_in",
    "u_in_R",
    "u_in_C",
    "cum_u_in_R",
    "cum_u_in_C",
    "flow",
    "vol_over_C",
    "u_in_over_R",
    "breath_total_volume",
    "breath_len",
    "u_in_sq",
    "time_step_sq",
    "dt_sq",
    "cum_u_in_sq",
    "flow_sq",
    "frac_cum_u_in",
    "time_frac",
    "u_in_u_out",
    "dt_u_out",
    "u_in_dt",
    "dt_R",
    "dt_C",
    "u_in_R_C",
    "cum_u_in_R_C",
    "flow_R",
    "flow_C",
]


def fit_ridge(X, y, alpha=0.01):
    """Fit a Ridge regression on standardized features."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_scaled, y)
    return scaler, model


def predict_ridge(scaler, model, X):
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


def fit_gbr(X, y):
    """Fit a lightweight GradientBoostingRegressor (no scaling needed)."""
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    model.fit(X, y)
    return None, model  # No scaler


def predict_gbr(_, model, X):
    return model.predict(X)


X_global = train_df[features].values
y_global = train_df["pressure"].values
global_scaler, global_model = fit_ridge(X_global, y_global, alpha=0.01)

grouped = train_df.groupby(["R", "C", "u_out"])


def train_group(key_group):
    key, group = key_group
    X = group[features].values
    y = group["pressure"].values
    scaler, model = fit_gbr(X, y)  # use GradientBoosting for each (R,C,u_out) group
    return key, (scaler, model)


model_items = Parallel(n_jobs=5, backend="loky")(
    delayed(train_group)((key, group)) for key, group in grouped
)

model_dict = dict(model_items)

predictions = np.full(len(test_df), np.nan, dtype=np.float64)

for (R_val, C_val, u_out_val), (scaler, model) in model_dict.items():
    mask = (
        (test_df["R"] == R_val)
        & (test_df["C"] == C_val)
        & (test_df["u_out"] == u_out_val)
    )
    if not mask.any():
        continue
    X_test = test_df.loc[mask, features].values
    predictions[mask] = predict_gbr(scaler, model, X_test)

fallback_mask = np.isnan(predictions)
if fallback_mask.any():
    X_fallback = test_df.loc[fallback_mask, features].values
    predictions[fallback_mask] = predict_ridge(global_scaler, global_model, X_fallback)



## === cell 1
assert len(predictions) == len(sample_sub), "Prediction length mismatch."



## === cell 2
submission = pd.DataFrame({"id": sample_sub["id"], "pressure": predictions})
submission.to_csv(SUBMISSION_PATH, index=False)

submission.head()
