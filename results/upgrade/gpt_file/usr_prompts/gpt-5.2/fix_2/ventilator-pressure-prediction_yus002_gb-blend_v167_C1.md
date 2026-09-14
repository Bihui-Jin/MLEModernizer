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

0.1358446937940982

# 6. Current score

17.65244

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 17.65244) has done: 'I fix the crash by removing hard-coded references to missing external blending files and instead blending only whatever prediction CSVs are actually present in the provided input directory. To keep the original core logic (random-weight blending + median/mean combination + snapping to nearest allowed pressure), I preserve those steps and just make them robust to an arbitrary number of files (including the case of only one). I also ensure the script always writes a valid Kaggle submission named `submission.csv` with the required `id,pressure` columns and correct row order/length. Finally, I update the paths to use the existing `/kaggle/input/...` structure so it runs end-to-end in the Kaggle environment.'

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
BASE_COMP_PATH = "/kaggle/input/ventilator-pressure-prediction"
TRAIN_PATH = f"{BASE_COMP_PATH}/train.csv"
TEST_PATH = f"{BASE_COMP_PATH}/test.csv"
SAMPLE_SUB_PATH = f"{BASE_COMP_PATH}/sample_submission.csv"

df_train = pd.read_csv(TRAIN_PATH)
unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction: float) -> float:
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return float(sorted_pressures[-1])
    if insert_idx == 0:
        return float(sorted_pressures[0])
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return float(
        lower_val
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else upper_val
    )


def set_seed(seed: int = 2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """
    Original intent: parse LB score from filename and use as weight.
    Bug fix: filenames in this environment won't follow that convention.
    Keep minimal behavior: just average the provided prediction vectors if 2 files,
    or return the single vector if 1 file.
    """
    preds = []
    for p in input_list:
        df = pd.read_csv(p)
        if "pressure" not in df.columns:
            raise ValueError(f"File {p} does not contain 'pressure' column.")
        preds.append(df["pressure"].to_numpy(dtype=np.float32))
    if len(preds) == 1:
        return preds[0]
    return 0.5 * preds[0] + 0.5 * preds[1]


def g(dp):
    """
    Robust version of original blender:
    - Uses all CSVs found in dp (must have 'pressure' column).
    - Preserves original approach: split -> wc -> random-weight ensembles repeated loop_time ->
      final blend = 0.7*median + 0.3*mean (keeps same spirit as original; removed missing externals).
    - Writes 'submission.csv' for Kaggle.
    """
    files = sorted([p for p in glob.iglob(f"{dp}/*") if p.lower().endswith(".csv")])
    if len(files) == 0:
        raise FileNotFoundError(
            f"No .csv prediction files found in '{dp}'. "
            f"Provide a directory under /kaggle/input containing prediction csvs with columns [id, pressure] (or at least pressure)."
        )

    output = pd.read_csv(SAMPLE_SUB_PATH)
    n = len(output)

    all_preds = []
    for p in files:
        df = pd.read_csv(p)
        if "pressure" not in df.columns:
            continue
        pred = df["pressure"].to_numpy(dtype=np.float32)
        if len(pred) != n:
            continue
        all_preds.append(pred)

    if len(all_preds) == 0:
        raise ValueError(
            f"All csvs in '{dp}' were missing 'pressure' column or had wrong length != {n}."
        )

    splits = 2
    if len(all_preds) < 2:
        splits = 1

    if splits == 1:
        flist = [all_preds[0]]
    else:
        usable_files = [p for p in files if p.lower().endswith(".csv")]
        filtered_files = []
        for p in usable_files:
            df = pd.read_csv(p, nrows=5)
            if "pressure" not in df.columns:
                continue
            pred_len = pd.read_csv(p, usecols=["pressure"]).shape[0]
            if pred_len != n:
                continue
            filtered_files.append(p)

        if len(filtered_files) == 0:
            raise ValueError(
                f"No compatible prediction files in '{dp}' after filtering."
            )
        filtered_files = sorted(filtered_files)

        chunk_size = int(np.ceil(len(filtered_files) / splits))
        chunks = [
            filtered_files[i * chunk_size : (i + 1) * chunk_size] for i in range(splits)
        ]
        flist = [wc(chunk) for chunk in chunks if len(chunk) > 0]

    loop_time = 154
    pred_list = []
    for seed in range(loop_time):
        set_seed(seed)
        weights = [rd() for _ in range(len(flist))]
        wsum = sum(weights)
        weights = [w / wsum for w in weights]
        weights.sort(reverse=True)

        temp = np.zeros(n, dtype=np.float32)
        for i in range(len(flist)):
            temp += flist[i] * weights[i]
        pred_list.append(temp)
        del temp
        if seed % 20 == 0:
            gc.collect()

    stack = np.vstack(pred_list)
    median_pred = np.median(stack, axis=0)
    mean_pred = stack.mean(axis=0)

    final_pred = 0.7 * median_pred + 0.3 * mean_pred

    output["pressure"] = pd.Series(final_pred).apply(find_nearest).astype(np.float32)
    output.to_csv("submission.csv", index=False)
    return output


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a["pressure"] = (a["pressure"] * 0.7 + b["pressure"] * 0.3).apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a


def avg(dp):
    input_list = []
    for i in glob.iglob(f"{dp}/*"):
        if i.lower().endswith(".csv"):
            input_list.append(i)
    if len(input_list) == 0:
        raise FileNotFoundError(f"No .csv files found in '{dp}'.")
    preds = []
    for p in input_list:
        df = pd.read_csv(p)
        if "pressure" not in df.columns:
            continue
        preds.append(df["pressure"].to_numpy(dtype=np.float32))
    if len(preds) == 0:
        raise ValueError(f"No usable prediction csvs with 'pressure' in '{dp}'.")
    output = pd.read_csv(SAMPLE_SUB_PATH)
    output["pressure"] = (
        pd.Series(np.median(np.vstack(preds), axis=0))
        .apply(find_nearest)
        .astype(np.float32)
    )
    output.to_csv("avg.csv", index=False)
    return output




## === cell 2
preferred_dp = "/kaggle/input/gb-data-blending-recover"
if os.path.isdir(preferred_dp):
    g(preferred_dp)
else:
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    n = len(sample)
    candidate = None
    for sub in sorted(glob.glob("/kaggle/input/*")):
        if not os.path.isdir(sub):
            continue
        csvs = glob.glob(os.path.join(sub, "*.csv"))
        ok = False
        for p in csvs[:50]:  # limit checks for speed
            try:
                df = pd.read_csv(p, nrows=5)
                if "pressure" not in df.columns:
                    continue
                if pd.read_csv(p, usecols=["pressure"]).shape[0] != n:
                    continue
                ok = True
                break
            except Exception:
                continue
        if ok:
            candidate = sub
            break
    if candidate is None:
        raise FileNotFoundError(
            "Could not find any /kaggle/input/* directory containing compatible prediction CSVs to blend. "
            "Please add at least one dataset with a submission-like csv (with 'pressure' and correct row count)."
        )
    g(candidate)
