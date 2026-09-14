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

0.1366625097300452

# 6. Current score

3.877

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'I make the averaging helper robust so it no longer crashes when the expected directory is empty, and I add a lightweight fallback model that trains a simple linear regression on the available training data and creates a valid `submission.csv`. This ensures the notebook runs end‑to‑end, produces a correctly‑formatted submission, and gives a reasonable MAE without altering the original blending logic.'
- What this solution (achieved 10.86378) has done: 'I keep the overall workflow unchanged but improve the fallback LinearRegression model by adding a few simple interaction and polynomial features and switching to a regularized Ridge regression, which remains a linear model. These extra features give the model more expressive power while preserving the core logic, and the regularization helps avoid over‑fitting, moving the MAE closer to the target. I also adjust the feature list accordingly.'
- What this solution (achieved 10.86378) has done: 'I replace the simple Ridge regression with a more expressive tree‑based model (HistGradientBoostingRegressor) and keep the same feature engineering and post‑processing. This change respects the “large gap” rule, uses only the existing libraries, and should substantially lower the MAE while still producing a correctly‑formatted `submission.csv`.'
- What this solution (achieved 10.86378) has done: 'I add a few more interaction‑polynomial features in `add_features` and train the same HistGradientBoostingRegressor with a deeper tree, lower learning rate and more iterations (turning off early stopping). These tweaks keep the original model type and workflow while giving the learner more expressive power, which should appreciably lower the MAE toward the target value.'
- What this solution (achieved 10.86378) has done: 'I add a simple median‑lookup baseline that groups the training data by rounded control features (C, R, u_in, u_out, time_step) and uses the median pressure of each group as a fast “lookup” prediction. The final prediction is a 50/50 blend of the HistGradientBoosting model and this lookup; this minor change keeps the original workflow but gives the model extra concrete information, which should lower the MAE and move the score closer to the target.'
- What this solution (achieved 3.877) has done: 'The fix replaces the invalid loss name `'least_absolute_deviation'` with the correct Scikit‑learn option `'absolute_error'` for the HistGradientBoostingRegressor, allowing the model to train without error and generate a proper `submission.csv`. No other logic is altered, preserving the original workflow and keeping the score‑related behavior unchanged.'

# 9. Code solution

## === cell 0
import numpy as np, pandas as pd, os, random, gc, glob
from random import random as rd


def set_seed(seed=2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return np.random.RandomState(seed)




## === cell 1
df_train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
unique_pressures = np.sort(df_train["pressure"].unique())
total_pressures_len = len(unique_pressures)


def find_nearest(prediction):
    """Round a prediction to the nearest pressure value seen in the training set."""
    idx = np.searchsorted(unique_pressures, prediction)
    if idx == total_pressures_len:
        return unique_pressures[-1]
    if idx == 0:
        return unique_pressures[0]
    low, high = unique_pressures[idx - 1], unique_pressures[idx]
    return low if abs(low - prediction) < abs(high - prediction) else high


def wc(input_list):
    """Weighted combine two prediction files (used in original blending script)."""
    scores = []
    for i, path in enumerate(input_list):
        try:
            score = int(path.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            score = 0
        scores.append(score)
        input_list[i] = pd.read_csv(path).pressure.values.ravel()
    if len(input_list) == 1:
        return input_list[0]
    w1 = (scores[1] / sum(scores)) + 0.1
    w2 = 1 - w1
    return input_list[0] * w1 + input_list[1] * w2


def g(dp):
    """Original blending routine – kept unchanged."""
    files = sorted(glob.glob(f"{dp}/*"))
    if not files:
        raise FileNotFoundError(f"No files found in {dp}")
    splits = len(files) // 2 or 1
    groups = [files[i::splits] for i in range(splits)]
    preds = [wc(group) for group in groups]
    loop_time = 154
    pred_list = []
    for it in range(loop_time):
        set_seed(it)
        weights = np.random.dirichlet(np.ones(len(preds)), size=1).ravel()
        combined = sum(p * w for p, w in zip(preds, weights))
        pred_list.append(combined)
    out = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")
    out.pressure = np.median(np.vstack(pred_list), axis=0)
    out["pressure"] = out["pressure"].apply(find_nearest)
    out.to_csv(f"rwb_{loop_time}_loops.csv", index=False)


def blend(a_path, b_path):
    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)
    a.pressure = a.pressure * 0.7 + b.pressure * 0.3
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a


def avg(dp):
    """Average predictions from all CSVs in *dp*.
    If the folder is empty, raise an error so the fallback model is used.
    """
    file_paths = list(glob.glob(f"{dp}/*"))
    if not file_paths:
        raise FileNotFoundError(f"No prediction files in {dp}")
    preds = [pd.read_csv(p).pressure.values.ravel() for p in file_paths]
    out = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")
    out.pressure = np.median(np.vstack(preds), axis=0)
    out.to_csv("avg.csv", index=False)
    return out


def add_features(df):
    """Create simple interaction / polynomial features to enrich the linear model."""
    df = df.copy()
    df["C_R"] = df["C"] * df["R"]
    df["u_in_sq"] = df["u_in"] ** 2
    df["time_step_sq"] = df["time_step"] ** 2
    df["u_in_times_u_out"] = df["u_in"] * df["u_out"]
    df["C_u_in"] = df["C"] * df["u_in"]
    df["R_u_in"] = df["R"] * df["u_in"]
    df["time_step_u_in"] = df["time_step"] * df["u_in"]
    df["C_sq"] = df["C"] ** 2
    df["R_sq"] = df["R"] ** 2
    return df




## === cell 2
def train_and_predict():
    from sklearn.ensemble import HistGradientBoostingRegressor

    base_feat_cols = ["C", "R", "u_in", "u_out", "time_step"]
    X_base = df_train[base_feat_cols]
    X = add_features(X_base)
    y = df_train["pressure"]

    model = HistGradientBoostingRegressor(
        max_depth=8,
        learning_rate=0.05,
        max_iter=500,
        random_state=2021,
        loss="absolute_error",  # corrected loss name
        early_stopping=False,
    )
    model.fit(X, y)

    df_train_rounded = df_train.copy()
    df_train_rounded["u_in_r"] = df_train_rounded["u_in"].round(1)
    df_train_rounded["time_step_r"] = df_train_rounded["time_step"].round(2)
    lookup = (
        df_train_rounded.groupby(["C", "R", "u_in_r", "u_out", "time_step_r"])[
            "pressure"
        ]
        .median()
        .reset_index()
        .rename(columns={"pressure": "med_pressure"})
    )

    df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
    test_base = df_test[base_feat_cols]
    test_feats = add_features(test_base)
    model_preds = model.predict(test_feats)

    df_test_rounded = df_test.copy()
    df_test_rounded["u_in_r"] = df_test_rounded["u_in"].round(1)
    df_test_rounded["time_step_r"] = df_test_rounded["time_step"].round(2)
    df_merged = df_test_rounded.merge(
        lookup, on=["C", "R", "u_in_r", "u_out", "time_step_r"], how="left"
    )
    lookup_preds = df_merged["med_pressure"].values

    combined_preds = np.where(
        np.isnan(lookup_preds),
        model_preds,
        0.5 * model_preds + 0.5 * lookup_preds,
    )

    final_preds = np.vectorize(find_nearest)(combined_preds)

    submission = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    submission["pressure"] = final_preds
    submission.to_csv("submission.csv", index=False)
    print(
        "Generated submission.csv using HistGradientBoostingRegressor blended with a median‑lookup baseline."
    )
    return submission




## === cell 3
try:
    avg("../input/gb-data-blending-recover")
except Exception as e:
    print(f"Avg blending failed ({e}); using tree‑based fallback.")
    train_and_predict()
