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

10.86378

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'The crash happens because `../input/gb-data-blending-recover` doesn’t exist in your environment, so `glob` finds zero files, `pred_list` becomes a list of scalars, and `np.median(np.vstack(pred_list), axis=0)` returns a length-1 array that can’t be assigned to 603600 rows. I fix `g()` to (1) search for the blending folder in the actual provided paths and (2) validate that found prediction files have the expected length. If no external prediction files are available, the script fall back to a simple deterministic baseline (median pressure from train) so it always produces a valid `submission.csv`. These changes are only to make the pipeline run end-to-end and generate a correct-format submission.'
- What this solution (achieved 10.86378) has done: 'Your current score is extremely far from the target (MAE 10.86 vs 0.1437), and the reason is that the code is effectively producing a near-constant “median pressure” fallback because no valid external prediction files exist in this environment. To move the score sharply toward the target while preserving your core “blend multiple CSV submissions then snap to nearest allowed pressure” logic, I generate those missing per-row prediction CSVs from the provided `train.csv/test.csv` using a deterministic, lightweight lookup baseline keyed by `(R, C, time_step, u_in, u_out)` (and a safe fallback for unseen keys). Then your existing `g()` blender read and combine them as intended, producing a much more accurate submission without changing the blending approach. I also ensure the generated prediction files have the exact required length/alignment with `sample_submission.csv` so the output is always valid.'
- What this solution (achieved 10.86378) has done: 'Your current MAE (10.86) is far from the target (0.1437) because the script ends up blending only locally-generated constant/weak lookup predictions; the biggest win while preserving the same “blend CSVs then snap to allowed pressures” core is to make those locally-generated blend inputs much stronger. I keep your blending loop intact and only improve `_make_local_blend_inputs()` by (1) using `breath_id` + `time_step` keyed medians (much more informative than global time_step medians) and (2) creating two diverse-but-reasonable variants so the blender has meaningful signals to combine. I also ensure perfect row alignment with `sample_submission` by building predictions in `id` order and writing the generated CSVs into the same folder your `g()` already reads.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import copy
import glob
import random
from random import random as rd
import gc



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


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """
    Original intent: read multiple submission files and combine them with a heuristic weight based on
    a score encoded in the filename. This is kept, but made robust to unexpected filenames.
    """
    l = []
    arrs = []
    for i in range(len(input_list)):
        fn = input_list[i]
        base = fn.split("/")[-1]
        public_lb_score = None
        try:
            public_lb_score = int(base.split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)

        df = pd.read_csv(fn)
        if "pressure" not in df.columns:
            raise ValueError(f"File {fn} does not contain 'pressure' column.")
        arrs.append(df["pressure"].to_numpy().ravel())

    l_sum = sum(l) if sum(l) != 0 else 1

    if len(arrs) == 1:
        return arrs[0]
    else:
        weight1 = (l[1] / l_sum) + 0.15
        weight2 = 1 - weight1
        return arrs[0] * weight1 + arrs[1] * weight2


def _resolve_blend_dir(dp):
    """
    Fix: the originally referenced ../input/gb-data-blending-recover may not exist.
    Try a few likely locations; otherwise return None.
    """
    candidates = [
        dp,
        "../input/gb-data-blending-recover",
        "../kaggle/input/gb-data-blending-recover",
        "../input/ventilator-pressure-prediction",
        "../kaggle/input/ventilator-pressure-prediction",
    ]
    for c in candidates:
        if c is not None and os.path.isdir(c):
            return c
    return None


def _make_local_blend_inputs():
    """
    Change (score-relevant): create stronger per-row prediction CSVs from train->test lookup so g()
    blends informative signals (instead of weak global medians).

    Core blender remains unchanged; we only supply better input CSVs derived from the provided data.
    """
    base_dir = "../input/gb-data-blending-recover"
    os.makedirs(base_dir, exist_ok=True)

    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    test_path = "../input/ventilator-pressure-prediction/test.csv"

    sub = pd.read_csv(sample_path)
    test = pd.read_csv(
        test_path, usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
    )

    test = test.merge(
        sub[["id"]], on="id", how="inner", validate="one_to_one"
    ).sort_values("id")

    tr = df_train[
        ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    ].copy()

    for col, dt in [
        ("R", np.int16),
        ("C", np.int16),
        ("u_out", np.int8),
        ("breath_id", np.int32),
    ]:
        tr[col] = tr[col].astype(dt)
        test[col] = test[col].astype(dt)

    global_median = float(tr["pressure"].median())

    u_in_bins = np.array(
        [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.float64
    )
    tr["u_in_bin"] = np.digitize(tr["u_in"].to_numpy(), u_in_bins, right=True).astype(
        np.int16
    )
    test["u_in_bin"] = np.digitize(
        test["u_in"].to_numpy(), u_in_bins, right=True
    ).astype(np.int16)

    map_a = (
        tr.groupby(["R", "C", "time_step", "u_out", "u_in_bin"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_a"})
    )
    p_a = test.merge(
        map_a, on=["R", "C", "time_step", "u_out", "u_in_bin"], how="left"
    )["p_a"]

    map_b = (
        tr.groupby(["R", "C", "time_step", "u_out"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_b"})
    )
    p_b = test.merge(map_b, on=["R", "C", "time_step", "u_out"], how="left")["p_b"]

    map_c = (
        tr.groupby(["R", "C", "time_step"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_c"})
    )
    p_c = test.merge(map_c, on=["R", "C", "time_step"], how="left")["p_c"]

    base_pred = (
        p_a.fillna(p_b).fillna(p_c).fillna(global_median).to_numpy(dtype=np.float64)
    )

    b_pred = 0.90 * base_pred + 0.10 * p_b.fillna(global_median).to_numpy(
        dtype=np.float64
    )

    c_pred = 0.97 * base_pred + 0.03 * global_median

    def _snap(arr):
        return np.array([find_nearest(x) for x in arr], dtype=np.float64)

    out_a = sub.copy()
    out_a["pressure"] = _snap(base_pred)
    out_a.to_csv(os.path.join(base_dir, "local.1 a.csv"), index=False)

    out_b = sub.copy()
    out_b["pressure"] = _snap(b_pred)
    out_b.to_csv(os.path.join(base_dir, "local.2 b.csv"), index=False)

    out_c = sub.copy()
    out_c["pressure"] = _snap(c_pred)
    out_c.to_csv(os.path.join(base_dir, "local.3 c.csv"), index=False)


def g(dp):
    """
    Fix: robustly handle missing/empty blending directory and ensure final prediction length matches
    sample_submission length so a valid .csv is always written.
    """
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    output = pd.read_csv(sample_path)
    n = len(output)

    dp_resolved = _resolve_blend_dir(dp)
    if dp_resolved is None:
        dp_resolved = dp  # keep for message context

    files = []
    if dp_resolved is not None and os.path.isdir(dp_resolved):
        for f in glob.iglob(f"{dp_resolved}/*"):
            if f.lower().endswith(".csv"):
                files.append(f)
    files.sort()

    if len(files) == 0:
        _make_local_blend_inputs()
        dp_resolved = _resolve_blend_dir(dp) or dp
        files = []
        if dp_resolved is not None and os.path.isdir(dp_resolved):
            for f in glob.iglob(f"{dp_resolved}/*"):
                if f.lower().endswith(".csv"):
                    files.append(f)
        files.sort()

    if len(files) == 0:
        baseline = float(df_train["pressure"].median())
        output["pressure"] = baseline
        output["pressure"] = output["pressure"].apply(find_nearest)
        output.to_csv("submission.csv", index=False)
        return

    file_count = len(files)
    loop_time = 154
    splits = max(1, file_count // 2)

    flist = []
    for i in range(splits):
        start = i * round(len(files) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(files) / splits)
        chunk = files[start:end]
        if len(chunk) == 0:
            continue
        flist.append(chunk)

    preds = []
    for chunk in flist:
        try:
            arr = wc(chunk)
            if len(arr) != n:
                continue
            preds.append(arr)
        except Exception:
            continue

    if len(preds) == 0:
        baseline = float(df_train["pressure"].median())
        output["pressure"] = baseline
        output["pressure"] = output["pressure"].apply(find_nearest)
        output.to_csv("submission.csv", index=False)
        return

    pred_list = []
    for t in range(loop_time):
        set_seed(t)
        weight = [rd() for _ in range(len(preds))]
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        weight = [w / weight_sum for w in weight]
        weight.sort(reverse=True)

        temp = np.zeros(n, dtype=np.float64)
        for i in range(len(preds)):
            temp += preds[i] * weight[i]
        pred_list.append(temp)
        del temp
        gc.collect()

    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv("submission.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
g("../input/gb-data-blending-recover")
print("Wrote submission.csv")
