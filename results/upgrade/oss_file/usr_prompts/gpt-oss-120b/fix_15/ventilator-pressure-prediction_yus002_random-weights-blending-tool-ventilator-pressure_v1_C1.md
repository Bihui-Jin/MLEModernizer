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

1.773636033397321

# 6. Current score

5.58362

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.43315) has done: 'I make the ensemble function robust by handling the case when no prediction files are found. If the directory is empty, it fall back to a simple baseline that predicts the overall mean pressure from the training set (mapped to the nearest recorded pressure). This prevents the `np.vstack` error and guarantees a valid CSV submission is written.'
- What this solution (achieved 7.55219) has done: 'I replace the simple mean‑pressure fallback with a lightweight regression that learns from the training data (using R, C, u_in, u_out, time_step) and then maps its predictions to the nearest recorded pressure value. This keeps the original ensemble logic unchanged while providing a far more accurate baseline when no prediction files are present, moving the MAE from ~8.4 toward the target 1.77.'
- What this solution (achieved 5.7343) has done: 'I improve the fallback model that is used when no ensemble files are present. Instead of training on a 20 % random subset with plain linear regression, the code now trains on the full training set and adds simple second‑order interaction features (via `PolynomialFeatures`). This keeps the same linear‑regression core but gives it richer information, which should substantially lower the MAE and move the score toward the target while leaving the rest of the pipeline unchanged.'
- What this solution (achieved 5.73444) has done: 'I remove the unnecessary rounding‑to‑nearest‑recorded‑pressure step, which adds extra error, and output the raw regression predictions directly both in the fallback case and after ensembling. This small change keeps the model and feature engineering unchanged while reducing MAE, moving the score closer to the target.'
- What this solution (achieved 6.5046) has done: 'I keep the overall pipeline unchanged but replace the plain LinearRegression in the fallback with a regularized Ridge regression (RidgeCV) and clip the predictions to the observed pressure range. This small regularization tweak usually reduces over‑fitting on the polynomial features, bringing the MAE closer to the target without altering the ensemble logic.'
- What this solution (achieved 23.92066) has done: 'Implemented a higher‑order polynomial fallback model to boost predictive power while keeping the original pipeline untouched.  
- Switched `PolynomialFeatures` degree from 2 to 3, giving richer interaction terms.  
- Expanded the RidgeCV α grid for better regularisation.  
- Added a brief comment explaining the change and its expected effect on MAE.'
- What this solution (achieved 14.84511) has done: 'Implemented a modest regression tweak: lowered polynomial degree from 3 to 2 (reducing over‑fit risk) while retaining the RidgeCV regularisation. This small change preserves the original fallback‑regression design but yields far more realistic pressure estimates, moving the MAE markedly closer to the target score. No other pipeline logic was altered.'
- What this solution (achieved 5.58362) has done: 'The fix limits the expensive fallback model by drastically cutting the GradientBoostingRegressor size (fewer trees, larger learning rate) while keeping the same model class and feature engineering, and adds a quick‑exit when many ensemble files are present. This reduces training time well below the 600 s limit without altering the overall algorithmic flow or prediction semantics.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
import random
from random import random as rd

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

CSV_DTYPE = {
    "R": np.int8,
    "C": np.int8,
    "u_in": np.float32,
    "u_out": np.int8,
    "time_step": np.float32,
    "pressure": np.float32,
}

df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    usecols=["R", "C", "u_in", "u_out", "time_step", "pressure"],
    dtype=CSV_DTYPE,
    memory_map=True,
)
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
        input_list[i] = pd.read_csv(
            input_list[i], usecols=["pressure"]
        ).pressure.values.ravel()
    output = 0
    l_sum = sum(l)
    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = (1 - weight1) - 0.1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def train_regression_fallback():
    """
    Train a Gradient Boosting Regressor on degree‑2 polynomial features.
    Reduced n_estimators from 40 to 20 and increased learning_rate to 0.1
    to keep training time low while preserving the same model class
    and feature engineering.
    """
    X_raw = df_train[["R", "C", "u_in", "u_out", "time_step"]].values
    y = df_train["pressure"].values

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X = poly.fit_transform(X_raw).astype(np.float32)

    model = GradientBoostingRegressor(
        n_estimators=20,  # fewer trees for speed
        learning_rate=0.1,  # compensate reduced depth
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X, y)
    return model, poly


def fallback_prediction():
    """
    Produce predictions for the test set using the regression model
    and clip them to the training pressure bounds.
    """
    model, poly = train_regression_fallback()

    test_path = "../input/ventilator-pressure-prediction/test.csv"
    df_test = pd.read_csv(
        test_path,
        usecols=["R", "C", "u_in", "u_out", "time_step"],
        dtype=CSV_DTYPE,
        memory_map=True,
    )

    X_test_raw = df_test[["R", "C", "u_in", "u_out", "time_step"]].values
    X_test = poly.transform(X_test_raw).astype(np.float32)

    raw_pred = model.predict(X_test)

    pressure_min = df_train["pressure"].min()
    pressure_max = df_train["pressure"].max()
    raw_pred = np.clip(raw_pred, pressure_min, pressure_max)

    submission = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    submission["pressure"] = raw_pred
    submission.to_csv("submission.csv", index=False)
    print("No prediction files found – created regression fallback submission.csv")


def g(dp):
    """Ensemble predictions from files in *dp*.
    If no files are found, fall back to the regression‑based baseline.
    If the number of files is large, skip the expensive O(N²) weighting
    and use the fallback model to stay within the time limit.
    """
    file_paths = [p for p in glob.iglob(f"{dp}/*") if os.path.isfile(p)]
    file_count = len(file_paths)

    MAX_FILES_FOR_ENSEMBLE = 5
    if file_count == 0:
        fallback_prediction()
        return
    if file_count > MAX_FILES_FOR_ENSEMBLE:
        print(
            f"Found {file_count} prediction files – exceeding threshold; using fallback."
        )
        fallback_prediction()
        return

    splits = file_count // 2
    file_paths.sort()
    flist = []
    for i in range(splits):
        if i == splits - 1:
            flist.append(file_paths[i * round(len(file_paths) / splits) :])
        else:
            flist.append(
                file_paths[
                    i
                    * round(len(file_paths) / splits) : (i + 1)
                    * round(len(file_paths) / splits)
                ]
            )
    for i in range(len(flist)):
        flist[i] = wc(flist[i])

    loop_time = file_count**2
    pred_stack = np.vstack(flist)  # shape: (file_count, n_steps)

    rng = np.random.RandomState(0)  # deterministic seed
    weights = rng.rand(loop_time, file_count)  # shape: (loop_time, file_count)

    weight_sum = weights.sum(axis=1, keepdims=True)
    weights = weights / weight_sum
    weights = np.sort(weights, axis=1)[:, ::-1]  # descending order

    weighted_preds = weights @ pred_stack  # shape: (loop_time, n_steps)

    median_pred = np.median(weighted_preds, axis=0)

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = median_pred
    output.to_csv("submission.csv", index=False)
    print(f"Ensembled {file_count} files – created submission.csv")




## === cell 1
g("../input/gb-pred-files")
