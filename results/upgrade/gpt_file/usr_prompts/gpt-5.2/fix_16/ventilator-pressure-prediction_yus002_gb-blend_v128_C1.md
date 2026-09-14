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

0.1431657204406646

# 6. Current score

3.49027

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.99691) has done: 'The error comes from trying to read external Kaggle datasets (`gb-data-blending-recover`) that are not available in your environment, so the pipeline never produces a submission. I keep your core logic (pressure snapping via `find_nearest`) but replace the missing-file blend with a self-contained baseline that trains on the provided `train.csv` and predicts for `test.csv`. To preserve evaluation semantics (only inspiratory phase is scored), the training uses only rows with `u_out==0` and fits a simple, fast linear model on the same provided features. Finally, it writes a valid `submission.csv` with `id,pressure` to the working directory.'
- What this solution (achieved 7.74169) has done: 'Your current score is far worse than the target (lower is better), so we should improve the predictive signal while keeping the same “simple linear model + snap-to-known-pressures” core. The biggest issue is that the model ignores the breath-wise time-series structure; adding a few standard, leak-free engineered features (cumulative/lagged u_in within each breath) usually yields a large MAE drop without changing the training loop or model class. We keep LinearRegression, keep training only on inspiratory rows (u_out==0), keep the same snapping via `find_nearest`, and ensure the submission aligns by `id`. These minimal feature additions should move the score much closer to your target band.'
- What this solution (achieved 7.8322) has done: 'We keep your current core approach (LinearRegression on engineered, leak-free per-breath features + snapping predictions to the known discrete pressure grid) and make one minimal, high-impact correction: avoid training only on `u_out==0` rows because it creates a distribution mismatch (the model never learns what to output during expiration even though you must submit values for all timesteps). Instead, we train on all rows but apply a higher sample weight to inspiratory timesteps (`u_out==0`) to stay aligned with the metric while still learning reasonable expiratory outputs. This is a small, safe change that typically drops MAE a lot versus the current 7.74 and should move you substantially closer to the 0.143 target band. The submission writing and `id` alignment are kept unchanged.'
- What this solution (achieved 7.50172) has done: 'Your current score (7.8322, lower is better) is far worse than the target (0.1432), so we should improve predictive signal without changing your core approach (LinearRegression + engineered leak-free features + snapping to the known pressure grid). The main minimal fix is to add a few standard, breath-wise “state” features (area under u_in, lagged u_out, time delta, and simple interaction terms with R/C) that preserve semantics but better capture the underlying dynamics. I keep the same pipeline, still train on all rows with inspiratory upweighting, and keep the exact same `find_nearest` snapping and submission writing. These feature additions typically reduce MAE substantially while staying lightweight and within Kaggle constraints.'
- What this solution (achieved 7.61563) has done: 'Your current MAE (7.50, lower is better) is far above the target (0.143), so we need a real signal gain without changing the overall approach (fast linear model + leak-free feature engineering + snapping to the discrete pressure grid). The biggest missing piece is that your model doesn’t explicitly encode the “inspiratory-only scoring” rule; we can keep training on all rows but add a feature indicating inspiratory phase and, more importantly, force expiration predictions (u_out==1) to a stable baseline (e.g., near PEEP) learned from the training data—this reduces error on the many expiratory timesteps even though they aren’t scored, and typically improves generalization as well. We also add two very small, standard per-breath features (normalized time index within breath and a cumulative sum of u_in when u_out==0) that keep the same semantics but better reflect dynamics. Everything else (LinearRegression pipeline, preprocessing, snapping, submission writing) stays the same.'
- What this solution (achieved 7.51539) has done: 'Your current MAE is far above the target (lower is better), so the smallest meaningful improvement while preserving your core logic (LinearRegression + engineered features + snap-to-pressure-grid) is to (1) fix the `id` alignment bug (your provided `id` range indicates it’s not globally unique, so overwriting `sub["id"]` can corrupt row mapping), and (2) stop forcing all expiratory (`u_out==1`) predictions to a constant baseline, which can indirectly harm the learned mapping and also isn’t needed because expiration isn’t scored. I keep the same feature set, same model, same sample-weighting idea, and the same `find_nearest` snapping, but I ensure the submission uses the sample submission’s `id` ordering and only applies snapping after prediction. These minimal changes are directly tied to producing valid, correctly aligned predictions and should move the score substantially closer to your target.'
- What this solution (achieved 8.05876) has done: 'Your current MAE is much worse than the target (lower is better), so we should improve predictive signal without changing the overall approach (LinearRegression on engineered features + snap-to-known-pressure-grid). The biggest minimal win here is to respect the competition’s scoring rule by training primarily on inspiratory timesteps (u_out==0), and then set expiratory predictions (u_out==1) to a stable per-(R,C) baseline learned from the training inspiratory data (expiration isn’t scored, but this prevents out-of-distribution nonsense outputs). I also add two tiny, leak-free per-breath state features (lagged pressure proxy via cumulative u_in and a “time index within breath”) while keeping the same pipeline and loss semantics. Finally, the submission be written using the sample submission’s id order to guarantee correct alignment.'
- What this solution (achieved 3.98216) has done: 'We keep your exact core approach (LinearRegression on your engineered features + snap-to-discrete pressure grid), but fix two score-hurting issues with minimal changes: (1) train on all rows using sample weights to emphasize inspiratory timesteps (matches metric) instead of training only on `u_out==0` (distribution mismatch), and (2) remove the forced expiratory constant baseline overwrite, since expiratory rows are not scored and overriding can distort predictions around phase transitions. We also ensure deterministic ordering/alignment by sorting test features to match `sample_submission`’s `id` order before predicting. These changes should materially reduce MAE from ~8 toward your target without changing the model class, loss, or prediction snapping semantics.'
- What this solution (achieved 3.54139) has done: 'I fix the runtime KeyError inside `add_features` by correcting the `groupby(...)[u_out_change]` misuse (it is treating the 0/1 values as column labels) and computing the segment id with a proper grouped cumulative sum. I also keep your existing core logic (LinearRegression on engineered features + snapping to discrete pressure grid) unchanged otherwise. Finally, I make the read paths robust to both `/kaggle/input/...` and your `../input/...` layout so it runs end-to-end in this environment and always writes `submission.csv` with the required `id,pressure` columns.'
- What this solution (achieved 4.03749) has done: 'Your current score (3.54139, lower is better) is still far above the target (0.1432), so we should improve signal with the smallest changes that preserve your core approach (LinearRegression + your feature engineering + snap-to-pressure-grid). The biggest issue is you train only on inspiratory rows (`u_out==0`), but you must predict for all rows; this distribution mismatch typically hurts generalization even though expiration isn’t scored. I keep the same model and features, but train on all rows with sample weights to emphasize inspiratory timesteps (aligns with the metric without changing the algorithm). I also make the prediction alignment more robust by predicting in the exact `sample_submission` id order (via an `id`-indexed reindex), avoiding any subtle ordering mismatch.'
- What this solution (achieved 3.54139) has done: 'Your current MAE (4.037, lower is better) is still far above the target (0.143), so we need a real-but-minimal signal gain without changing the core approach (LinearRegression on engineered, leak-free features + snapping to the discrete pressure grid). The highest-impact small fix here is to stop training on expiratory rows at all (they are not scored) while still producing sane expiratory predictions by copying the nearest inspiratory prediction within the same breath; this typically improves the inspiratory fit because the linear model isn’t “compromising” to explain expiration dynamics. We keep the exact same feature set and model class, but switch fitting to inspiratory-only with no sample weights and apply a simple, leak-free per-breath forward-fill for `u_out==1` predictions at inference time. We also ensure alignment strictly follows `sample_submission` id order and still snap to the known pressure grid before writing `submission.csv`.'
- What this solution (achieved 3.49173) has done: 'Your current MAE (3.541) is still far above the target (0.143), so we need a modest but meaningful signal gain while keeping your exact core approach (LinearRegression on engineered, leak-free features + snap-to-known pressure grid + inspiratory-only training + per-breath forward-fill for u_out==1 at inference). The smallest high-impact change here is to add a few standard, leak-free “state” features that a linear model can exploit: per-breath rolling means of `u_in`, cumulative counts of inspiratory steps, and simple interaction terms between `(R,C)` and inputs/time. These additions preserve evaluation semantics and don’t change the model class, loss, or training loop—just provide better linear separability. I also keep the strict `id` alignment via `sample_submission` and continue snapping predictions to the discrete pressure grid before writing `submission.csv`.'
- What this solution (achieved 3.49027) has done: 'Your current MAE (3.49173, lower is better) is still far above the target (0.1432), so we should cautiously improve predictive signal without changing your core approach (LinearRegression + your leak-free features + snapping to the discrete pressure grid + inspiratory-only training + per-breath ffill for expiration). The smallest high-impact change here is to add a *minimal* amount of regularization via `Ridge` (still a linear regression model, same semantics) to reduce overfitting/noise amplification from many correlated engineered features, which often yields a noticeable MAE drop on this competition. I also add two very standard, leak-free per-breath features (`u_in_roll10_mean` and `u_in_cum_mean`) that a linear model can exploit for smoother dynamics, while keeping everything else (training mask, alignment, snapping, and submission writing) identical. These changes are directly aimed at lowering MAE toward your target and should run within the time limit.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
import os
import copy
import glob
import random
from random import random as rd
import gc




## === cell 1
def _resolve_path(rel_path: str) -> str:
    candidates = [
        rel_path,
        rel_path.replace("../input/", "/kaggle/input/"),
        rel_path.replace("../input/", "/kaggle/data/"),
        rel_path.replace("../input/", "/kaggle/input/ventilator-pressure-prediction/"),
        rel_path.replace("../input/", "/kaggle/data/ventilator-pressure-prediction/"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return rel_path  # let pandas raise a clear error if nothing matches


train_path = _resolve_path("../input/ventilator-pressure-prediction/train.csv")
df_train = pd.read_csv(train_path)

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
        input_list[i] = (pd.read_csv(input_list[i]).pressure).ravel()
    output = 0
    l_sum = sum(l)
    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = (l[1] / l_sum) + 0.05
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)
    file_count = len(l)
    loop_time = 154
    splits = file_count // 2
    l.sort()
    flist = []
    for i in range(splits):
        if i == splits - 1:
            flist.append(l[i * round(len(l) / splits) :])
        else:
            flist.append(
                l[i * round(len(l) / splits) : (i + 1) * round(len(l) / splits)]
            )
    for i in range(len(flist)):
        flist[i] = wc(flist[i])
    pred_list = []
    for i in range(loop_time):
        weight = []
        set_seed(i)
        for i in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for i in range(len(weight)):
            weight[i] /= weight_sum
        weight.sort(reverse=True)
        temp = 0
        for i in range(len(flist)):
            temp += flist[i] * weight[i]
        pred_list.append(temp)
        del temp
        gc.collect()
    output = pd.read_csv(
        _resolve_path("../input/ventilator-pressure-prediction/sample_submission.csv")
    )
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.62 + b.pressure * 0.38
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import Ridge

test_path = _resolve_path("../input/ventilator-pressure-prediction/test.csv")
sample_path = _resolve_path(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)

df_test = pd.read_csv(test_path)
sub = pd.read_csv(sample_path)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.sort_values(["breath_id", "time_step"], inplace=True)

    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2).fillna(0.0)

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_diff2"] = df["u_in_lag1"] - df["u_in_lag2"]

    df["u_in_cum"] = df.groupby("breath_id")["u_in"].cumsum()
    df["u_in_x_time"] = df["u_in"] * df["time_step"]

    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0.0)

    df["u_out_lag1"] = df.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(int)
    df["u_out_cum"] = df.groupby("breath_id")["u_out"].cumsum().astype(float)

    df["u_in_area"] = (df["u_in"] * df["dt"]).groupby(df["breath_id"]).cumsum()

    df["u_in_over_R"] = df["u_in"] / df["R"].astype(float)
    df["u_in_over_C"] = df["u_in"] / df["C"].astype(float)
    df["time_over_R"] = df["time_step"] / df["R"].astype(float)
    df["time_over_C"] = df["time_step"] / df["C"].astype(float)

    df["is_insp"] = (df["u_out"] == 0).astype(int)

    tmin = df.groupby("breath_id")["time_step"].transform("min")
    tmax = df.groupby("breath_id")["time_step"].transform("max")
    denom = (tmax - tmin).replace(0, 1.0)
    df["t_norm"] = (df["time_step"] - tmin) / denom

    df["u_in_insp"] = df["u_in"] * (df["u_out"] == 0).astype(float)
    df["u_in_cum_insp"] = df.groupby("breath_id")["u_in_insp"].cumsum()

    df["t_idx"] = df.groupby("breath_id").cumcount().astype(np.int16)
    df["u_in_cum_sq"] = (df["u_in_cum"] ** 2).astype(float)

    u_out_change = (df["u_out"] != df["u_out_lag1"]).astype(np.int8)
    seg_id = u_out_change.groupby(df["breath_id"]).cumsum().astype(np.int16)

    first_time_in_seg = df.groupby(["breath_id", seg_id])["time_step"].transform(
        "first"
    )
    df["time_since_u_out_change"] = df["time_step"] - first_time_in_seg

    g = df.groupby("breath_id")

    df["insp_step"] = df["is_insp"].astype(np.int16)
    df["insp_count"] = g["insp_step"].cumsum().astype(np.int16)

    df["u_in_roll3_mean"] = (
        g["u_in"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["u_in_diff1_roll3_mean"] = (
        g["u_in_diff1"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["u_in_x_R"] = df["u_in"] * df["R"].astype(float)
    df["u_in_x_C"] = df["u_in"] * df["C"].astype(float)
    df["time_x_R"] = df["time_step"] * df["R"].astype(float)
    df["time_x_C"] = df["time_step"] * df["C"].astype(float)

    df["u_in_roll10_mean"] = (
        g["u_in"]
        .rolling(window=10, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["u_in_cum_mean"] = df["u_in_cum"] / (df["t_idx"].astype(float) + 1.0)

    return df


df_train_fe = add_features(df_train)
df_test_fe = add_features(df_test)

features_num = [
    "time_step",
    "u_in",
    "u_out",
    "u_in_lag1",
    "u_in_lag2",
    "u_in_diff1",
    "u_in_diff2",
    "u_in_cum",
    "u_in_x_time",
    "dt",
    "u_out_lag1",
    "u_out_cum",
    "u_in_area",
    "u_in_over_R",
    "u_in_over_C",
    "time_over_R",
    "time_over_C",
    "is_insp",
    "t_norm",
    "u_in_cum_insp",
    "t_idx",
    "u_in_cum_sq",
    "time_since_u_out_change",
    "insp_count",
    "u_in_roll3_mean",
    "u_in_diff1_roll3_mean",
    "u_in_x_R",
    "u_in_x_C",
    "time_x_R",
    "time_x_C",
    "u_in_roll10_mean",
    "u_in_cum_mean",
]
features_cat = ["R", "C"]

train_insp_mask = df_train_fe["u_out"].values == 0
X_train = df_train_fe.loc[train_insp_mask, features_num + features_cat]
y_train = df_train_fe.loc[train_insp_mask, "pressure"].astype(float)

df_test_fe_id = df_test_fe.set_index("id", drop=False)
sub_id = sub.set_index("id", drop=False)
df_test_fe_aligned = df_test_fe_id.reindex(sub_id.index)
if df_test_fe_aligned.isnull().any().any():
    raise ValueError(
        "Alignment by id introduced NaNs; check id uniqueness/mismatch between test and sample_submission."
    )

X_test = df_test_fe_aligned[features_num + features_cat]

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
            features_num,
        ),
        ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
    ],
    remainder="drop",
)

model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("reg", Ridge(alpha=1.0, random_state=0)),
    ]
)

model.fit(X_train, y_train)

pred = model.predict(X_test).astype(float)

pred_df = df_test_fe_aligned[["breath_id", "time_step", "u_out"]].copy()
pred_df["pred"] = pred
pred_df.sort_values(["breath_id", "time_step"], inplace=True)
pred_df["pred_insp_only"] = pred_df["pred"].where(pred_df["u_out"].values == 0)
pred_df["pred_filled"] = pred_df.groupby("breath_id")["pred_insp_only"].ffill()
pred_df["pred_filled"] = pred_df["pred_filled"].fillna(pred_df["pred"])
pred_df.sort_index(inplace=True)  # back to aligned-by-id row order

pred_final = pred_df["pred_filled"].to_numpy(dtype=float)
pred_final = np.array([find_nearest(p) for p in pred_final], dtype=float)

sub_out = sub_id.copy()
sub_out["pressure"] = pred_final
sub_out.reset_index(drop=True)[["id", "pressure"]].to_csv("submission.csv", index=False)

print(sub_out.reset_index(drop=True)[["id", "pressure"]].head())
print(
    "Wrote submission.csv with shape:",
    sub_out.reset_index(drop=True)[["id", "pressure"]].shape,
)
