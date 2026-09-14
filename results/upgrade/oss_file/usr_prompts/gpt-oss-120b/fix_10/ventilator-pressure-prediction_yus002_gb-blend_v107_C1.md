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

0.1510712250966403

# 6. Current score

1.49771

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.54837) has done: 'I guard the blending step against missing files so it no longer crashes, and then add a straightforward linear‑regression model (using NumPy) that learns from the training data and produces a valid `submission.csv` with the required `id,pressure` columns. The prediction is passed through the existing `find_nearest` function to map it to the closest observed pressure, matching the competition’s evaluation style.'
- What this solution (achieved 5.7343) has done: 'I keep the overall linear‑regression pipeline but enrich the feature set with second‑order interaction terms (squares and pairwise products). This gives the same least‑squares solution while providing a much more expressive model, which should sharply lower the MAE toward the target. The rest of the script (seed handling, nearest‑pressure mapping, CSV output) stays unchanged.'
- What this solution (achieved 4.42829) has done: 'The changes limit feature construction to the sampled subset (avoiding building the full 5.4 M‑row matrix) and replace the slow `np.vectorize` nearest‑pressure mapping with a fully vectorized `np.searchsorted` implementation.  This cuts memory use and CPU time while preserving the exact feature set, model, and post‑processing logic, so the predictions remain identical apart from negligible floating‑point differences.'
- What this solution (achieved 3.94324) has done: 'The changes keep the same feature engineering and overall model type (gradient‑boosted trees) but replace the classic `GradientBoostingRegressor` with the much faster `HistGradientBoostingRegressor`, which uses histogram binning and native OpenMP parallelism while preserving the loss function and tree depth. Feature matrices are built as `float32` to cut memory traffic and speed up training without altering numeric results appreciably. All other steps, paths, and output formats stay unchanged.'
- What this solution (achieved 1.5515) has done: 'I added informative time‑series features (cumulative sums and differences of the control signals) to give the gradient‑boosting model more context about each breath, and I implemented a fast vectorized nearest‑pressure mapping that replaces the missing `find_nearest` call. These changes keep the same model type while providing richer input data and aligning predictions with the discrete pressure values, which should move the MAE much closer to the target.'
- What this solution (achieved 1.49771) has done: 'I keep the same data handling and feature engineering but slightly improve the gradient‑boosting model by increasing tree depth (to capture more non‑linear patterns) and using a smaller learning rate with a modestly larger number of iterations. I also remove the unnecessary nearest‑pressure mapping – clipping the raw predictions to the observed range is sufficient and usually yields a lower MAE. These minimal tweaks keep the original pipeline intact while moving the validation score closer to the target.'

# 9. Code solution

## === cell 0
import os
import random
import gc
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
from random import random as rd

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


seed = 2021
rng = set_seed(seed)




## === cell 1
def blend(a_path, b_path):
    if not (os.path.exists(a_path) and os.path.exists(b_path)):
        print("Blend files not found – skipping blending step.")
        return None
    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    if "find_nearest" in globals():
        a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a


a_path = "../input/gb-data-blending-recover/0.149 blend.csv"
b_path = "../input/gb-data-blending-recover/0.149 blend2.csv"
blend(a_path, b_path)  # will run only if files are present




## === cell 2
train_cols = ["u_in", "u_out", "R", "C", "time_step", "pressure", "breath_id", "id"]
test_cols = ["u_in", "u_out", "R", "C", "time_step", "breath_id", "id"]

dtype_spec = {
    "u_in": "float32",
    "u_out": "int8",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "pressure": "float32",
    "breath_id": "int32",
    "id": "int32",
}

df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    usecols=train_cols,
    dtype=dtype_spec,
)

df_test = pd.read_csv(
    "../input/ventilator-pressure-prediction/test.csv",
    usecols=test_cols,
    dtype=dtype_spec,
)

df_train["cum_u_in"] = df_train.groupby("breath_id")["u_in"].cumsum()
df_train["cum_u_out"] = df_train.groupby("breath_id")["u_out"].cumsum()
df_test["cum_u_in"] = df_test.groupby("breath_id")["u_in"].cumsum()
df_test["cum_u_out"] = df_test.groupby("breath_id")["u_out"].cumsum()

df_train["diff_u_in"] = df_train.groupby("breath_id")["u_in"].diff().fillna(0)
df_train["diff_u_out"] = df_train.groupby("breath_id")["u_out"].diff().fillna(0)
df_test["diff_u_in"] = df_test.groupby("breath_id")["u_in"].diff().fillna(0)
df_test["diff_u_out"] = df_test.groupby("breath_id")["u_out"].diff().fillna(0)

sorted_pressures = np.sort(df_train["pressure"].unique())
total_pressures_len = len(sorted_pressures)


def find_nearest(val):
    """
    Vectorized nearest‑pressure lookup.
    """
    idx = np.searchsorted(sorted_pressures, val, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)
    left = np.maximum(idx - 1, 0)
    right = idx
    left_diff = np.abs(sorted_pressures[left] - val)
    right_diff = np.abs(sorted_pressures[right] - val)
    choose = np.where(left_diff <= right_diff, left, right)
    return sorted_pressures[choose]




## === cell 3
base_feature_cols = [
    "u_in",
    "u_out",
    "R",
    "C",
    "time_step",
    "cum_u_in",
    "cum_u_out",
    "diff_u_in",
    "diff_u_out",
]


def build_features(df):
    """
    Create second‑order polynomial & interaction features together with the
    extra cumulative/difference columns.
    """
    base = df[base_feature_cols].to_numpy(dtype=np.float32, copy=False)
    u_in, u_out, R, C, t, cum_u_in, cum_u_out, diff_u_in, diff_u_out = (
        base[:, 0],
        base[:, 1],
        base[:, 2],
        base[:, 3],
        base[:, 4],
        base[:, 5],
        base[:, 6],
        base[:, 7],
        base[:, 8],
    )
    features = np.column_stack(
        (
            u_in,
            u_out,
            R,
            C,
            t,
            cum_u_in,
            cum_u_out,
            diff_u_in,
            diff_u_out,
            u_in * u_out,
            u_in * R,
            u_in * C,
            u_in * t,
            u_out * R,
            u_out * C,
            u_out * t,
            R * C,
            R * t,
            C * t,
            cum_u_in * u_in,
            cum_u_out * u_out,
            diff_u_in * u_in,
            diff_u_out * u_out,
            u_in**2,
            u_out**2,
            R**2,
            C**2,
            t**2,
        )
    )
    return features.astype(np.float64)  # HistGradientBoosting expects float64


X_train = build_features(df_train)
y_train = df_train["pressure"].to_numpy(dtype=np.float64, copy=False)

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor

gbr = HistGradientBoostingRegressor(
    max_iter=1500,  # increased number of trees
    learning_rate=0.03,  # finer step size
    max_depth=7,  # deeper trees for richer interactions
    random_state=seed,
    verbose=0,
)

gbr.fit(X_train, y_train)




## === cell 4
X_test_raw = build_features(df_test)
pred_raw = gbr.predict(X_test_raw)

pred_clipped = np.clip(pred_raw, sorted_pressures.min(), sorted_pressures.max())

submission = pd.DataFrame({"id": df_test["id"], "pressure": pred_clipped})
submission = submission[["id", "pressure"]]
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
