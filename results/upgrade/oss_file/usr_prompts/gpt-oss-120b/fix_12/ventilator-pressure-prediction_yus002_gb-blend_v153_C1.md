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

0.1372865105501195

# 6. Current score

4.14189

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.19027) has done: 'I add the missing sklearn imports, keep the existing helper functions, and replace the failing blend call with a safe fallback: if the specified prediction files are absent, the code train a fast HistGradientBoostingRegressor on a sample of the training data, generate predictions for the test set, round them to the nearest training pressure value, and write a proper submission.csv file. This fixes the FileNotFoundError and ensures a valid submission is produced.'
- What this solution (achieved 4.10365) has done: 'I keep the overall workflow but replace the fallback‑model section with a stronger predictor: use the full training set (instead of a 20 % sample), add a simple interaction feature (`u_in * time_step`), and train the HistGradientBoostingRegressor with deeper trees and more iterations. These changes stay within the original modelling approach while expected to lower the MAE toward the target score.'
- What this solution (achieved 4.05394) has done: 'I add a few lightweight features ( `R/C`, `C/R`, `u_out*time_step` ) and stop rounding the model’s output to the nearest training pressure, because that rounding adds unnecessary error. I also enable a small validation split with early‑stopping to keep the model from over‑fitting while preserving the original HistGradientBoostingRegressor‑based fallback. These changes keep the core workflow unchanged but are expected to move the MAE much closer to the target.'
- What this solution (achieved 3.85699) has done: 'I add a few cheap interaction features (quadratic terms via PolynomialFeatures) and slightly tune the HistGradientBoostingRegressor’s depth, learning‑rate and iterations. These changes stay within the original modelling pipeline, keep the same validation split, and are expected to lower the validation MAE, moving the score closer to the target.'
- What this solution (achieved 4.14179) has done: 'We move the heavy CSV load inside the fallback branch so it runs only when needed, and we lower the number of boosting iterations (max_iter) to 1000, which keeps the same model type and other hyper‑parameters while cutting the training workload enough to finish within the 600 s limit. The rest of the logic – feature engineering, polynomial expansion, validation split, early stopping, and post‑processing with the nearest‑pressure mapping – stays unchanged.'
- What this solution (achieved 4.14189) has done: 'I keep the overall workflow and model unchanged but remove the post‑processing step that snaps each prediction to the nearest pressure observed in the training set. That rounding introduced large bias and was the main reason the validation MAE was far above the target. By using the raw model outputs for the submission, the predictions stay continuous and the MAE moves much closer to the desired score. The rest of the code (feature engineering, polynomial expansion, validation split, early‑stopping) is left intact.'

# 9. Code solution

## === cell 0
import os
import gc
import random
import glob
import sys
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures  # interaction terms
from sklearn.utils import check_random_state




## === cell 1
usecols_train = [
    "R",
    "C",
    "u_in",
    "u_out",
    "time_step",
    "pressure",
]  # 'breath_id' and 'id' are not needed for model training


def find_nearest(prediction):
    """Find nearest pressure for a single scalar prediction."""
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


def find_nearest_array(preds):
    """Vectorized version that returns an array of nearest pressures."""
    idx = np.searchsorted(sorted_pressures, preds, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)
    lower_idx = np.maximum(idx - 1, 0)
    lower = sorted_pressures[lower_idx]
    upper = sorted_pressures[idx]
    use_lower = np.abs(lower - preds) < np.abs(upper - preds)
    return np.where(use_lower, lower, upper)


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


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


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
a_path = "../input/gb-data-blending-recover/0.1348.csv"
b_path = "../input/gb-data-blending-recover/0.1357.csv"

if os.path.exists(a_path) and os.path.exists(b_path):
    blend(a_path, b_path)
    sys.exit(0)  # Prevent further execution

df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    usecols=usecols_train,
    dtype={
        "R": np.int16,
        "C": np.int16,
        "u_in": np.float32,
        "u_out": np.int8,
        "time_step": np.float32,
        "pressure": np.float32,
    },
)

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures).astype(np.float32)
total_pressures_len = len(sorted_pressures)

df_train["u_in_time"] = (df_train["u_in"] * df_train["time_step"]).astype(np.float32)
df_train["R_div_C"] = (df_train["R"] / df_train["C"]).astype(np.float32)
df_train["C_div_R"] = (df_train["C"] / df_train["R"]).astype(np.float32)
df_train["u_out_time"] = (df_train["u_out"] * df_train["time_step"]).astype(np.float32)

usecols_test = [
    "R",
    "C",
    "u_in",
    "u_out",
    "time_step",
    "breath_id",  # kept to match original read signature
    "id",
]
df_test = pd.read_csv(
    "../input/ventilator-pressure-prediction/test.csv",
    usecols=usecols_test,
    dtype={
        "R": np.int16,
        "C": np.int16,
        "u_in": np.float32,
        "u_out": np.int8,
        "time_step": np.float32,
        "breath_id": np.int32,
        "id": np.int16,
    },
)

df_test["u_in_time"] = (df_test["u_in"] * df_test["time_step"]).astype(np.float32)
df_test["R_div_C"] = (df_test["R"] / df_test["C"]).astype(np.float32)
df_test["C_div_R"] = (df_test["C"] / df_test["R"]).astype(np.float32)
df_test["u_out_time"] = (df_test["u_out"] * df_test["time_step"]).astype(np.float32)

base_features = [
    "R",
    "C",
    "u_in",
    "u_out",
    "time_step",
    "u_in_time",
    "R_div_C",
    "C_div_R",
    "u_out_time",
]

poly = PolynomialFeatures(degree=2, include_bias=False)

X_base = df_train[base_features].to_numpy(dtype=np.float32, copy=False)
X = poly.fit_transform(X_base).astype(np.float32, copy=False)
y = df_train["pressure"].to_numpy(dtype=np.float32, copy=False)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, shuffle=True
)

model = HistGradientBoostingRegressor(
    loss="absolute_error",
    max_depth=12,
    learning_rate=0.01,
    max_iter=1000,  # decreased from 3000 for runtime safety
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    l2_regularization=0.0,
    max_bins=127,
)
model.fit(X_train, y_train)

val_pred = model.predict(X_val)
val_mae = mean_absolute_error(y_val, val_pred)
print(f"Validation MAE (raw predictions): {val_mae:.5f}")

X_test_base = df_test[base_features].to_numpy(dtype=np.float32, copy=False)
X_test = poly.transform(X_test_base).astype(np.float32, copy=False)
preds = model.predict(X_test).astype(np.float32)

submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)
submission["pressure"] = preds
submission.to_csv("submission.csv", index=False)
print("Fallback submission saved as submission.csv")
