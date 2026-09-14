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

0.1533508918615785

# 6. Current score

17.65486

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.445) has done: 'Your notebook fails because it tries to read four external Kaggle Dataset submissions that are not present in your environment, so `sub_1..sub_4` never load and the blend crashes. To keep the core “blend several submissions with fixed weights” logic while making it runnable end-to-end, I (1) load whatever listed files actually exist, (2) if none exist, fall back to a simple, deterministic baseline that predicts the mean training pressure (valid but not competitive), and (3) always write a correctly formatted `submission.csv`. This guarantees a valid submission file is produced with minimal, stability-focused edits and no dependency on missing inputs.'
- What this solution (achieved 17.65486) has done: 'Your current score (8.445, lower-is-better) is far worse than the target (~0.153), and the main reason is that the intended blended submissions are missing so you’re falling back to a mean-pressure baseline. To move sharply toward the target without changing the “blend fixed-weight submissions” core idea, I (1) load any valid submission-like CSVs found under `/kaggle/input` automatically (not just the four hardcoded missing ones), (2) prefer the strongest single candidate if blending would be unreliable, and (3) keep strict `id` alignment and always write a valid `submission.csv`. This keeps the same overall approach (use other submissions and blend) but makes it actually pick up available competitive predictions in your environment instead of collapsing to the weak baseline.'
- What this solution (achieved 17.65486) has done: 'Your current score is far above (worse than) the target, so we need a small change that legitimately improves predictions without changing the “load-and-blend existing submission CSVs” core approach. The main issue is that your auto-discovery assigns the same tiny weight (0.05) to every found submission, which can dilute a strong candidate with many weak/duplicate ones and hurt MAE. I keep the same blending logic but (1) de-duplicate near-identical submissions, (2) rank candidates by agreement with others and keep only the most consistent top-K, and (3) use a weighted median (more robust than mean) for the final blend. This stays within the same semantics (blending external predictions) while typically moving the score much closer to the target if any good submission(s) exist in the environment.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np



## === cell 1
BASE_INPUT = "/kaggle/input"
BASE_DATA = "/kaggle/data"  # fallback if needed


def first_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


sample_path = first_existing_path(
    [
        f"{BASE_INPUT}/ventilator-pressure-prediction/sample_submission.csv",
        f"{BASE_DATA}/sample_submission.csv",
        f"{BASE_DATA}/ventilator-pressure-prediction/sample_submission.csv",
    ]
)
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations."
    )

sub = pd.read_csv(sample_path)


def is_submission_like_csv(path):
    if not path.lower().endswith(".csv"):
        return False
    name = os.path.basename(path).lower()
    if "submission" in name or "sub" == name or name.startswith("sub"):
        return True
    return False


def load_submission_csv(path, ref_ids):
    try:
        df = pd.read_csv(path, usecols=["id", "pressure"])
    except Exception:
        return None
    if len(df) != len(ref_ids):
        return None
    if not df["id"].equals(ref_ids):
        return None
    df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")
    if df["pressure"].isna().all():
        return None
    df["pressure"] = df["pressure"].astype(np.float64).fillna(0.0)
    return df


blend_sources = [
    (
        0.6,
        f"{BASE_INPUT}/random-weights-blending-tool-ventilator-pressure/rwb 121 loops.csv",
    ),
    (
        0.1,
        f"{BASE_INPUT}/ventilator-pressure-prediction-lstm-gpu-infer/submission_median_round.csv",
    ),
    (
        0.1,
        f"{BASE_INPUT}/a-dummy-approach-to-improve-your-score-postprocess/submission.csv",
    ),
    (
        0.2,
        f"{BASE_INPUT}/ensemble-folds-with-median-0-153/submission_median_round_LB153.csv",
    ),
]

loaded = []
seen_paths = set()

for w, p in blend_sources:
    if os.path.exists(p):
        df = load_submission_csv(p, sub["id"])
        if df is not None:
            loaded.append((float(w), df, p))
            seen_paths.add(os.path.abspath(p))

if len(loaded) < 2:
    for root, _, files in os.walk(BASE_INPUT):
        for fn in files:
            if not fn.lower().endswith(".csv"):
                continue
            full = os.path.join(root, fn)
            ab = os.path.abspath(full)
            if ab in seen_paths:
                continue
            if not is_submission_like_csv(full):
                continue
            df = load_submission_csv(full, sub["id"])
            if df is None:
                continue
            loaded.append((0.05, df, full))
            seen_paths.add(ab)

loaded_ids_ok = True
for _, df, _ in loaded:
    if not df["id"].equals(sub["id"]):
        loaded_ids_ok = False
        break

loaded_ids_ok, len(loaded)




## === cell 2
def weighted_median(values_2d, weights):
    """
    values_2d: shape (n_models, n_rows)
    weights: shape (n_models,)
    """
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.sum(w)
    order = np.argsort(values_2d, axis=0)
    sorted_vals = np.take_along_axis(values_2d, order, axis=0)
    sorted_w = w[order]
    cdf = np.cumsum(sorted_w, axis=0)
    idx = (cdf >= 0.5).argmax(axis=0)
    return sorted_vals[idx, np.arange(values_2d.shape[1])]


def mae(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.mean(np.abs(a - b))


if len(loaded) > 0 and loaded_ids_ok:
    preds_list = []
    meta_list = []
    for w, df, p in loaded:
        pr = df["pressure"].to_numpy(dtype=np.float64)
        preds_list.append(pr)
        meta_list.append((w, p))

    uniq = []
    uniq_meta = []
    seen = set()
    for (w, p), pr in zip(meta_list, preds_list):
        key = hash(np.round(pr, 6).tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(pr)
        uniq_meta.append((w, p))

    preds_list = uniq
    meta_list = uniq_meta

    if len(preds_list) == 1:
        sub["pressure"] = preds_list[0]
    else:
        P = np.vstack(preds_list)  # (n_models, n_rows)

        n_models = P.shape[0]
        idx = np.linspace(0, P.shape[1] - 1, num=min(20000, P.shape[1]), dtype=int)
        P_small = P[:, idx]

        pair_mae = np.zeros((n_models, n_models), dtype=np.float64)
        for i in range(n_models):
            for j in range(i + 1, n_models):
                d = mae(P_small[i], P_small[j])
                pair_mae[i, j] = d
                pair_mae[j, i] = d

        consensus = pair_mae.mean(axis=1)  # lower is better
        order = np.argsort(consensus)

        K = int(min(8, n_models))
        keep = order[:K]
        Pk = P[keep]

        wk = np.array([meta_list[i][0] for i in keep], dtype=np.float64)
        if not np.isfinite(wk).all() or wk.sum() <= 0:
            wk = np.ones_like(wk, dtype=np.float64)

        sub["pressure"] = weighted_median(Pk, wk)
else:
    train_path = first_existing_path(
        [
            f"{BASE_INPUT}/ventilator-pressure-prediction/train.csv",
            f"{BASE_DATA}/train.csv",
            f"{BASE_DATA}/ventilator-pressure-prediction/train.csv",
        ]
    )
    if train_path is None:
        sub["pressure"] = 0.0
    else:
        train = pd.read_csv(train_path, usecols=["pressure"])
        sub["pressure"] = float(train["pressure"].mean())

sub["pressure"] = (
    pd.to_numeric(sub["pressure"], errors="coerce").astype(float).fillna(0.0)
)

sub[["id", "pressure"]].to_csv("submission.csv", index=False)

sub.head()
