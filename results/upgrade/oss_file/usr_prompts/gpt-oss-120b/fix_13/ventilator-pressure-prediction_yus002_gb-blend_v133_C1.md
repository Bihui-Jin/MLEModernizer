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

0.1363000140389698

# 6. Current score

4.01841

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.43315) has done: 'I add a safeguard to the blending routine so it no longer crashes when the expected directory contains no prediction files. If no files are found, the function now falls back to a simple baseline that predicts the overall mean pressure from the training set (rounded to the nearest valid pressure value). This guarantees a correctly‑sized “pressure” column and produces a valid `submission.csv` file, allowing the notebook to finish without errors while keeping the original logic untouched for cases where data is present.'
- What this solution (achieved 3.78084) has done: 'I add a lightweight “lookup‑model” that learns the average pressure for each unique combination of the main features (R, C, rounded u_in, u_out, rounded time_step) from the training set.  
If the blending directory is empty, the script now uses this lookup (falling back to the global mean when a combination is unseen) to generate predictions instead of the naïve global‑mean baseline. This modest change keeps the original blending logic untouched while dramatically reducing the MAE, moving the score toward the target.'
- What this solution (achieved 3.72731) has done: 'I keep the existing blending logic unchanged and only improve the fallback prediction used when no blending files are found. Instead of relying solely on a coarse group‑by mean lookup, I train a tiny linear regression on the full training set (features R, C, u_in, u_out, time_step) and use its prediction whenever the exact lookup key is missing. This adds essentially no extra complexity, preserves the original workflow, and is expected to lower the MAE toward the target score.'
- What this solution (achieved 5.24673) has done: 'I adjust the fallback prediction logic so that when a lookup key exists we combine its mean pressure with the linear‑regression estimate (averaging the two). This keeps the original model components but gives a slightly more calibrated prediction, which should reduce the MAE and move the score nearer the target. No other parts of the pipeline are changed.'
- What this solution (achieved 5.43807) has done: 'I tighten the fallback‑prediction logic by using a finer‑grained lookup table (u_in rounded to one decimal and time_step rounded to two decimals) and only fall back to the linear‑regression estimate when the exact key is missing. This keeps the overall workflow unchanged while giving a more accurate baseline, which should lower the MAE toward the target.'
- What this solution (achieved 4.11889) has done: 'I added a finer‑grained group‑by mean lookup based on the key (`R`, `C`, `u_out`) and combined it with the existing linear‑regression estimate. This provides a much stronger baseline when no blending files are present, reducing the MAE dramatically while keeping the overall workflow unchanged.'
- What this solution (achieved 4.03616) has done: 'I keep the overall workflow and fallback‑model logic but replace the simple linear‑regression estimate with a lightweight gradient‑boosting regressor (HistGradientBoostingRegressor). This still uses a linear‑type feature set and preserves the lookup‑based and group‑mean components, while providing a more accurate baseline that should lower the MAE toward the target. I also adjust the prediction function to use the new model and keep the nearest‑pressure rounding unchanged.'
- What this solution (achieved 4.00785) has done: 'The fix replaces the buggy vectorized lookup in **predict_pressure_vectorized** with a pandas‑based mapping that correctly creates 1‑dimensional arrays for the fine and coarse look‑ups, eliminating the “too many indices” error. The rest of the logic (model training, blending, and submission writing) is left untouched, preserving the original workflow while ensuring a valid `submission.csv` is produced.'
- What this solution (achieved 4.01139) has done: 'I add two simple interaction features (`RC` and `u_in_time`) to the training data and include them in the model’s feature set, increase the GBDT depth and iteration count for a stronger fit, and slightly rebalance the blend between the fine‑grained lookup and the model prediction (weighting them equally). These adjustments keep the overall workflow unchanged while giving the model more expressive power, which should lower the MAE toward the target.'
- What this solution (achieved 4.01841) has done: 'I slightly strengthen the model and give the fine‑grained lookup a higher influence when blending predictions. Keeping the overall workflow untouched, I increase the HistGradientBoostingRegressor iteration count for a better fit and adjust the blending weights inside `predict_pressure_vectorized` (fine lookup 70 %, model 30 % and coarse lookup 70 % with the model when fine is missing). These minimal changes are expected to lower the MAE and move the score closer to the target while still producing a valid submission.csv​.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
import random
import gc
from sklearn.ensemble import HistGradientBoostingRegressor

df_train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)

df_train["RC"] = df_train["R"] * df_train["C"]
df_train["u_in_time"] = df_train["u_in"] * df_train["time_step"]


def find_nearest_vectorized(preds):
    """Vectorized rounding of predictions to the nearest pressure seen in training."""
    idx = np.searchsorted(sorted_pressures, preds, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)
    lower = sorted_pressures[np.maximum(idx - 1, 0)]
    upper = sorted_pressures[idx]
    choose_lower = np.abs(lower - preds) < np.abs(upper - preds)
    return np.where(choose_lower, lower, upper)


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """Read prediction CSVs and extract their public leaderboard scores for weighting."""
    l = []
    for i in range(len(input_list)):
        public_lb_score = int(input_list[i].split("/")[-1].split(".")[1].split(" ")[0])
        l.append(public_lb_score)
        input_list[i] = (pd.read_csv(input_list[i]).pressure).values.astype(np.float64)
    output = 0
    l_sum = sum(l)
    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


feature_cols = ["R", "C", "u_in", "u_out", "time_step", "RC", "u_in_time"]
X = df_train[feature_cols].values.astype(np.float64)
y = df_train["pressure"].values.astype(np.float64)

gbdt_model = HistGradientBoostingRegressor(
    max_depth=7,
    learning_rate=0.05,
    max_iter=800,  # increased from 500
    random_state=2021,
)
gbdt_model.fit(X, y)

group_mean = df_train.groupby(["R", "C", "u_out"])["pressure"].mean().to_dict()

df_train["u_in_round10"] = (df_train["u_in"] * 10).round().astype(int)
df_train["time_step_round2"] = df_train["time_step"].round(2)

lookup = (
    df_train.groupby(["R", "C", "u_in_round10", "u_out", "time_step_round2"])[
        "pressure"
    ]
    .mean()
    .to_dict()
)

global_mean_pressure = df_train["pressure"].mean()


def predict_pressure_vectorized(df):
    """
    Vectorized prediction using the trained GBDT model blended with fine‑grained
    and coarse lookup tables. The fine lookup now receives a larger weight
    (70 %) to better exploit the locally observed averages.
    """
    df["RC"] = df["R"] * df["C"]
    df["u_in_time"] = df["u_in"] * df["time_step"]

    xb = df[feature_cols].values.astype(np.float64)
    gbdt_pred = gbdt_model.predict(xb)

    fine_keys = list(
        zip(
            df["R"].astype(int),
            df["C"].astype(int),
            (df["u_in"] * 10).round().astype(int),
            df["u_out"].astype(int),
            df["time_step"].round(2),
        )
    )
    fine_lookup = pd.Series(fine_keys).map(lookup).values.astype(float)

    coarse_keys = list(
        zip(df["R"].astype(int), df["C"].astype(int), df["u_out"].astype(int))
    )
    coarse_lookup = pd.Series(coarse_keys).map(group_mean).values.astype(float)

    result = np.empty_like(gbdt_pred)

    mask_fine = ~np.isnan(fine_lookup)
    result[mask_fine] = 0.7 * fine_lookup[mask_fine] + 0.3 * gbdt_pred[mask_fine]

    mask_coarse = ~mask_fine & ~np.isnan(coarse_lookup)
    result[mask_coarse] = (
        0.7 * coarse_lookup[mask_coarse] + 0.3 * gbdt_pred[mask_coarse]
    )

    result[~mask_fine & ~mask_coarse] = gbdt_pred[~mask_fine & ~mask_coarse]

    return result


def g(dp):
    """Blend predictions from CSV files under ``dp``.
    If the directory is empty, fall back to the enhanced lookup‑model.
    The resulting CSV is written to ``submission.csv`` with correct length."""
    file_paths = list(glob.iglob(f"{dp}/*"))
    file_count = len(file_paths)

    splits = file_count // 2
    file_paths.sort()
    flist = []
    for i in range(splits):
        start = int(i * round(len(file_paths) / splits))
        end = (
            int((i + 1) * round(len(file_paths) / splits)) if i != splits - 1 else None
        )
        flist.append(wc(file_paths[start:end]))

    if len(flist) == 0:
        test_path = "../input/ventilator-pressure-prediction/test.csv"
        df_test = pd.read_csv(test_path)
        preds = predict_pressure_vectorized(df_test)
        preds = find_nearest_vectorized(preds)

        sub = pd.read_csv(
            "../input/ventilator-pressure-prediction/sample_submission.csv"
        )
        sub["pressure"] = preds
        sub.to_csv("submission.csv", index=False)
        print("No blending files found – wrote fallback‑model submission.")
        return

    loop_time = 154
    pred_list = []
    for i in range(loop_time):
        set_seed(i)
        weights = np.random.rand(len(flist))
        weights /= weights.sum()
        weights = np.sort(weights)[::-1]  # descending as original
        blended = sum(arr * w for arr, w in zip(flist, weights))
        pred_list.append(blended)

        del blended
        gc.collect()

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    stacked = np.vstack(pred_list)
    output["pressure"] = np.median(stacked, axis=0)
    output["pressure"] = find_nearest_vectorized(output["pressure"].values)
    output.to_csv("submission.csv", index=False)
    print("Blended submission written to submission.csv.")


def blend(a, b):
    """Simple weighted blending of two submission CSVs."""
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.55 + b.pressure * 0.45
    a["pressure"] = find_nearest_vectorized(a["pressure"].values)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 1
g("../input/gb-data-blending-recover")
