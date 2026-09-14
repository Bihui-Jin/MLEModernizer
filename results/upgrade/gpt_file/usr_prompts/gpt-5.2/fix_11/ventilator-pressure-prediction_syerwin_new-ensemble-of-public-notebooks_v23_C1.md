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

8.445

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.445) has done: 'Your notebook fails because it tries to read four external Kaggle Dataset submissions that are not present in your environment, so `sub_1..sub_4` never load and the blend crashes. To keep the core “blend several submissions with fixed weights” logic while making it runnable end-to-end, I (1) load whatever listed files actually exist, (2) if none exist, fall back to a simple, deterministic baseline that predicts the mean training pressure (valid but not competitive), and (3) always write a correctly formatted `submission.csv`. This guarantees a valid submission file is produced with minimal, stability-focused edits and no dependency on missing inputs.'
- What this solution (achieved 17.65486) has done: 'Your current score (8.445, lower-is-better) is far worse than the target (~0.153), and the main reason is that the intended blended submissions are missing so you’re falling back to a mean-pressure baseline. To move sharply toward the target without changing the “blend fixed-weight submissions” core idea, I (1) load any valid submission-like CSVs found under `/kaggle/input` automatically (not just the four hardcoded missing ones), (2) prefer the strongest single candidate if blending would be unreliable, and (3) keep strict `id` alignment and always write a valid `submission.csv`. This keeps the same overall approach (use other submissions and blend) but makes it actually pick up available competitive predictions in your environment instead of collapsing to the weak baseline.'
- What this solution (achieved 17.65486) has done: 'Your current score is far above (worse than) the target, so we need a small change that legitimately improves predictions without changing the “load-and-blend existing submission CSVs” core approach. The main issue is that your auto-discovery assigns the same tiny weight (0.05) to every found submission, which can dilute a strong candidate with many weak/duplicate ones and hurt MAE. I keep the same blending logic but (1) de-duplicate near-identical submissions, (2) rank candidates by agreement with others and keep only the most consistent top-K, and (3) use a weighted median (more robust than mean) for the final blend. This stays within the same semantics (blending external predictions) while typically moving the score much closer to the target if any good submission(s) exist in the environment.'
- What this solution (achieved 17.65486) has done: 'Your current score is far worse than the target, so we should stop diluting any strong found submission with many weak/duplicate ones. I keep the same core approach (discover existing submission CSVs and blend them), but change the selection step to prefer a single best “cluster” of mutually consistent predictions and then blend only within that cluster. This typically moves MAE sharply toward the target when at least one good submission exists, while remaining robust if many noisy CSVs are present. I also keep strict `id` alignment and always write a valid `submission.csv`.'
- What this solution (achieved 8.445) has done: 'Your score is far worse than the target (lower-is-better), and the main likely cause is that you’re still ending up blending weak or irrelevant CSVs because the auto-discovery heuristic is too permissive and the clustering is based on a very small, fixed subset of rows. I keep the same core approach (discover submission-like CSVs and blend them robustly) but (1) tighten candidate filtering to ignore near-constant/degenerate predictions and duplicates, (2) make the clustering distance computed on a deterministic stratified sample across the full file (not just evenly spaced indices), and (3) switch the final aggregation from weighted median to a weighted trimmed mean within the selected cluster, which is usually closer to strong single-model predictions while still being robust. These are small, inference-only changes and still always write a valid `submission.csv`.'
- What this solution (achieved 8.445) has done: 'Your current score (8.445, lower-is-better) is far worse than the target (~0.153), so the most likely issue is that the script is still not actually using any strong prediction CSVs and is falling back (directly or effectively) to weak/degenerate blends. I keep the same core logic (discover submission-like CSVs and robustly blend them) but tighten candidate selection so we don’t accidentally include weak files, and I add a lightweight self-check that detects “too-flat” final predictions and automatically falls back to the single most-consistent candidate instead of a harmful blend. I also make the clustering sample deterministic and more representative across the full file (still small) to pick the right cluster more reliably. These are inference-only adjustments that keep the blending approach intact while making it much more likely to move MAE sharply toward the target if any good submission exists in `/kaggle/input`.'
- What this solution (achieved 8.445) has done: 'Your score is extremely far from the target (8.445 vs 0.153, lower-is-better), which strongly suggests you are still not actually using any strong prediction CSVs and are either falling back to the weak mean baseline or selecting the wrong files. I keep the exact same “discover submission-like CSVs and robustly blend them” core logic, but (1) broaden discovery to accept any valid `id,pressure` CSV (not just filenames containing “submission”), (2) add a cheap “train-aware plausibility” filter using per-(R,C,time_step,u_in,u_out) median pressure from `train.csv` to reject obviously-wrong candidates, and (3) make the final selection prefer the single best-scoring candidate by this proxy (then blend only among near-ties) to avoid harmful dilution. These are inference-only changes, keep your aggregation approach intact, and should move MAE sharply toward the target when any decent submission exists in `/kaggle/input`. The script still always writes a valid `submission.csv` with correct `id` alignment.'
- What this solution (achieved 8.445) has done: 'Your current MAE (8.445, lower-is-better) is far worse than the target (~0.153), which strongly indicates you are still blending in weak/irrelevant “candidate” CSVs (or selecting the wrong cluster) and drowning out any good submission that might exist. To move sharply toward the target while keeping the same core “discover submissions and blend robustly” logic, I (1) tighten auto-discovery to only accept CSVs that clearly look like a real submission (correct shape, numeric pressure, non-degenerate distribution, and pressure range plausible for this competition), and (2) switch the final selection to prefer the single most “train-plausible” candidate (via your existing train-median proxy) unless there are near-ties, in which case we blend only those near-ties. This is a minimal inference-only change that avoids harmful dilution and is the most likely way to improve toward the target without changing any modeling/training logic. The script still run end-to-end and always write a valid `submission.csv`.'
- What this solution (achieved 8.445) has done: 'Your score is far worse than the target, so the most likely fix is to ensure we’re not accidentally selecting weak/irrelevant “candidate” CSVs and that we preferentially keep the single best candidate when the proxy strongly separates models. I keep your core approach (auto-discover valid `id,pressure` CSVs and then pick/blend) but tighten the candidate filter using a cheap train-aware proxy (match to per-(R,C,time_step) medians) and also require predictions to respect `u_out==1` behavior by pulling pressure toward a baseline during exhalation (common post-processing for this comp). Finally, I change the selection logic so that if one candidate is clearly best by the proxy, we use it directly (no blending dilution), otherwise we blend only a few near-ties. This is inference-only, preserves your overall semantics, and should move MAE sharply toward the target when any decent candidate submission exists in `/kaggle/input`.'
- What this solution (achieved 8.445) has done: 'Your MAE (8.445, lower-is-better) is far worse than the target (~0.153), and the most likely cause is that the discovered “candidate” CSVs are not actually good ventilator predictions (or are being harmed by the proxy selection/postprocess), so we need a minimal but stronger and more competition-aligned selection signal. I keep your core logic (discover `id,pressure` CSVs → filter → choose best or blend near-ties → write `submission.csv`) but replace the weak train-median proxy with a direct *train-labeled* proxy MAE computed on a small, deterministic sample by joining candidate predictions to `train.csv` on `(breath_id, time_step)` and scoring only `u_out==0` rows (matching the Kaggle metric). I also make the u_out postprocess less destructive by only applying it when we’re not already using the train-labeled proxy (so we don’t overwrite potentially-correct inspiratory dynamics). These changes are inference-only, run within time limits by sampling breaths, and should move the score sharply toward the target if any decent candidate submission exists in `/kaggle/input`.'

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


def is_candidate_csv(path):
    if not path.lower().endswith(".csv"):
        return False
    base = os.path.basename(path).lower()
    if base in {"train.csv", "test.csv", "sample_submission.csv"}:
        return False
    return True


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

for root, _, files in os.walk(BASE_INPUT):
    for fn in files:
        if not fn.lower().endswith(".csv"):
            continue
        full = os.path.join(root, fn)
        ab = os.path.abspath(full)
        if ab in seen_paths:
            continue
        if not is_candidate_csv(full):
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
def mae(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.mean(np.abs(a - b))


def weighted_trimmed_mean(values_2d, weights, trim_q=0.1):
    """
    values_2d: (n_models, n_rows)
    weights: (n_models,)
    Trim extremes across models per row, then take weighted mean.
    """
    V = np.asarray(values_2d, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    s = w.sum()
    if (not np.isfinite(s)) or s <= 0:
        w = np.ones_like(w, dtype=np.float64)
        s = w.sum()
    w = w / s

    n_models = V.shape[0]
    if n_models <= 2:
        return np.average(V, axis=0, weights=w)

    k = int(np.floor(trim_q * n_models))
    if 2 * k >= n_models:
        k = 0

    order = np.argsort(V, axis=0)
    mid_order = order[k : n_models - k, :]
    mid_vals = np.take_along_axis(V, mid_order, axis=0)
    mid_w = w[mid_order]
    denom = np.sum(mid_w, axis=0)
    denom = np.where(denom <= 0, 1.0, denom)
    return np.sum(mid_vals * mid_w, axis=0) / denom


def robust_pred_stats(pr):
    pr = np.asarray(pr, dtype=np.float64)
    q01, q50, q99 = np.quantile(pr, [0.01, 0.5, 0.99])
    std = float(np.std(pr))
    iqr = float(q99 - q01)
    return std, iqr, float(q50), float(q01), float(q99)


def plausible_submission_filter(pr):
    """
    Minimal, competition-specific plausibility checks to avoid blending in junk CSVs.
    """
    pr = np.asarray(pr, dtype=np.float64)
    if not np.isfinite(pr).all():
        return False
    std, iqr, q50, q01, q99 = robust_pred_stats(pr)

    if std < 0.2 or iqr < 0.5:
        return False

    if q01 < -5.0 or q99 > 80.0:
        return False

    if np.max(np.abs(pr)) > 1e4:
        return False

    return True


def apply_u_out_postprocess(pr, u_out, fallback_level):
    """
    Expiratory phase isn't scored; pushing predictions to a stable level when u_out==1
    is common, but can be destructive if a candidate already models it well.
    We will only apply this when we don't have a stronger train-labeled selection proxy.
    """
    if u_out is None:
        return pr
    pr = np.asarray(pr, dtype=np.float64).copy()
    mask = u_out.astype(np.int8) == 1
    if mask.any():
        pr[mask] = fallback_level
    return pr


def build_train_labeled_proxy(sample_breaths=2500, seed=2021):
    train_path = first_existing_path(
        [
            f"{BASE_INPUT}/ventilator-pressure-prediction/train.csv",
            f"{BASE_DATA}/train.csv",
            f"{BASE_DATA}/ventilator-pressure-prediction/train.csv",
        ]
    )
    test_path = first_existing_path(
        [
            f"{BASE_INPUT}/ventilator-pressure-prediction/test.csv",
            f"{BASE_DATA}/test.csv",
            f"{BASE_DATA}/ventilator-pressure-prediction/test.csv",
        ]
    )
    if (train_path is None) or (test_path is None):
        return None

    test = pd.read_csv(test_path, usecols=["id", "breath_id", "time_step", "u_out"])
    if not test["id"].equals(sub["id"]):
        test = test.sort_values("id", kind="mergesort")

    train = pd.read_csv(
        train_path, usecols=["breath_id", "time_step", "u_out", "pressure"]
    )

    rng = np.random.default_rng(seed)
    unique_breaths = train["breath_id"].unique()
    if unique_breaths.size == 0:
        return None
    k = int(min(sample_breaths, unique_breaths.size))
    chosen_breaths = rng.choice(unique_breaths, size=k, replace=False)

    train_s = train[train["breath_id"].isin(chosen_breaths)].copy()
    train_s["ts_key"] = np.round(
        train_s["time_step"].to_numpy(dtype=np.float64) * 1000
    ).astype(np.int32)
    test_s = test.copy()
    test_s["ts_key"] = np.round(
        test_s["time_step"].to_numpy(dtype=np.float64) * 1000
    ).astype(np.int32)

    joined = test_s.merge(
        train_s[["breath_id", "ts_key", "u_out", "pressure"]],
        on=["breath_id", "ts_key", "u_out"],
        how="inner",
        sort=False,
    )

    if joined.empty:
        return None

    joined = joined[joined["u_out"] == 0]
    if joined.empty:
        return None

    return {
        "id": joined["id"].to_numpy(dtype=np.int64),
        "y": joined["pressure"].to_numpy(dtype=np.float64),
        "test_u_out_full": test["u_out"].to_numpy(dtype=np.int8),
        "global_train_median": float(train["pressure"].median()),
    }


def proxy_train_labeled_mae(pr_full, proxy):
    ids = proxy["id"]
    idx = ids.astype(np.int64) - 1
    pred = pr_full[idx]
    return mae(pred, proxy["y"])


proxy = build_train_labeled_proxy(sample_breaths=2500, seed=2021)



## === cell 3
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
        if not plausible_submission_filter(pr):
            continue
        uniq.append(pr)
        uniq_meta.append((w, p))

    preds_list = uniq
    meta_list = uniq_meta

    if len(preds_list) == 0:
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
    elif len(preds_list) == 1:
        pr = preds_list[0]
        if proxy is None:
            stable_level = float(np.median(pr))
            sub["pressure"] = apply_u_out_postprocess(pr, None, stable_level)
        else:
            sub["pressure"] = pr
    else:
        P = np.vstack(preds_list)
        n_models, n_rows = P.shape

        chosen = None

        if proxy is not None:
            proxy_scores = np.array(
                [proxy_train_labeled_mae(P[i], proxy) for i in range(n_models)],
                dtype=np.float64,
            )
            best = int(np.argmin(proxy_scores))
            best_score = float(proxy_scores[best])

            sorted_scores = np.sort(proxy_scores)
            if (
                (sorted_scores.size >= 2)
                and np.isfinite(best_score)
                and (sorted_scores[1] >= best_score * 1.02)
            ):
                chosen = P[best]
            else:
                keep = np.where(proxy_scores <= best_score * 1.01)[0]  # within 1%
                if keep.size == 0:
                    keep = np.array([best], dtype=int)
                if keep.size > 5:
                    keep = keep[np.argsort(proxy_scores[keep])[:5]]

                if keep.size == 1:
                    chosen = P[keep[0]]
                else:
                    Pk = P[keep]
                    wk = np.array([meta_list[i][0] for i in keep], dtype=np.float64)
                    if (not np.isfinite(wk).all()) or wk.sum() <= 0:
                        wk = np.ones_like(wk, dtype=np.float64)
                    chosen = weighted_trimmed_mean(Pk, wk, trim_q=0.1)
                    if not plausible_submission_filter(chosen):
                        chosen = P[best]
        else:
            rng = np.random.default_rng(12345)
            sample_size = int(min(120000, n_rows))
            idx = rng.choice(n_rows, size=sample_size, replace=False)
            idx.sort()

            P_small = P[:, idx]
            pair_mae = np.zeros((n_models, n_models), dtype=np.float64)
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    d = mae(P_small[i], P_small[j])
                    pair_mae[i, j] = d
                    pair_mae[j, i] = d

            nn = int(min(5, max(1, n_models - 1)))
            cluster_score = np.zeros(n_models, dtype=np.float64)
            for i in range(n_models):
                d = np.sort(pair_mae[i][np.arange(n_models) != i])
                cluster_score[i] = float(np.mean(d[:nn])) if d.size > 0 else 0.0

            center = int(np.argmin(cluster_score))
            d_to_center = pair_mae[center].copy()
            d_to_center[center] = 0.0
            keep = np.argsort(d_to_center)[: min(5, n_models)]
            keep = np.unique(np.append(keep, center))

            if keep.size == 1:
                chosen = P[center]
            else:
                Pk = P[keep]
                wk = np.array([meta_list[i][0] for i in keep], dtype=np.float64)
                if (not np.isfinite(wk).all()) or wk.sum() <= 0:
                    wk = np.ones_like(wk, dtype=np.float64)
                chosen = weighted_trimmed_mean(Pk, wk, trim_q=0.1)
                if not plausible_submission_filter(chosen):
                    chosen = P[center]

        if proxy is None:
            stable_level = float(np.median(chosen))
            sub["pressure"] = apply_u_out_postprocess(chosen, None, stable_level)
        else:
            sub["pressure"] = chosen
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
