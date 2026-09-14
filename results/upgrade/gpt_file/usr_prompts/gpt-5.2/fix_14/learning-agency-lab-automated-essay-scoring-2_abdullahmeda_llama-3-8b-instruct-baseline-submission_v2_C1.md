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
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Target score

0.8030170953994958

# 6. Current score

0.60074

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.60074) has done: 'I fix the heuristic fallback crash by ensuring the bin-mapped predictions are converted off the Categorical dtype before calling `fillna`, which currently raises the “new category” TypeError. I also make the inference loop robust to environments where `transformers/peft` (or their protobuf stack) fail to import by keeping the try/except fallback intact, so a valid `submission.csv` is always produced. Finally, I keep the core modeling logic unchanged and only adjust the postprocessing/parsing and fallback pipeline so it runs end-to-end and writes the required `essay_id,score` CSV.'
- What this solution (achieved 0.60074) has done: 'I fix the crash that prevents the LLM inference path from running by forcing the pure-Python protobuf implementation before importing `transformers/peft`, which avoids the `MessageFactory.GetPrototype` AttributeError in many Kaggle/Python 3.12 images. To move the score up toward your target while keeping the same overall “dependency-free fallback” concept, I minimally upgrade the heuristic fallback from length-binning to a small TF‑IDF + Ridge regression model (still just scikit‑learn; no deep learning or external downloads), and clip/round to the required 1–6 integer scores. I also make the fallback automatically choose the best available model based on local CV quadratic weighted kappa (QWK) using an internal QWK implementation, so it stays deterministic and metric-aligned without changing the LLM core logic. Finally, I ensure a valid `submission.csv` is always written with correct columns and row count.'
- What this solution (achieved 0.60074) has done: 'I fix the protobuf-related `MessageFactory.GetPrototype` crash by setting the pure-Python protobuf implementation *before* any possible `transformers/peft` import and by running the LLM inference in a clean subprocess with that environment enforced. I also remove a current runtime bug in the notebook driver cell where `heuristic_fallback_submission` is referenced but not defined in that scope, ensuring a valid `submission.csv` is always produced. To move the score toward your target while preserving the existing core approach, I keep the exact TF‑IDF+Ridge fallback but add a minimal QWK-aligned calibration step (fit a single scalar `a,b` mapping on out-of-fold predictions, then round/clip to 1–6), which usually improves QWK without changing the model class. All paths and submission format remain unchanged.'
- What this solution (achieved 0.60074) has done: 'I fix the runtime error by ensuring the protobuf “python” implementation is forced before any `transformers/peft` import and by cleanly isolating the LLM inference into a subprocess (so the parent kernel’s already-imported protobuf can’t break it). I also fix the notebook-driver bug where `heuristic_fallback_submission` is referenced but not defined in the outer scope by duplicating that function in the driver cell (minimal change, required for robustness). The TF‑IDF + Ridge fallback (which is your current scoring path) be kept identical in logic, and I only add small stability guards to always write a valid `submission.csv` with correct columns/row-count and integer scores in [1,6]. This should run end-to-end and, when the LLM path fails, preserve your current ~0.60 score behavior while remaining eligible to improve if the LLM path succeeds.'
- What this solution (achieved 0.60074) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by enforcing the pure‑Python protobuf implementation in a subprocess (as you already started doing) and also forcing the corresponding `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`, which is the most reliable combo on Kaggle Python 3.12 images. I make the driver always produce a valid `submission.csv` by treating “infer ran but didn’t write the file” as a failure and immediately running the existing TF‑IDF+Ridge fallback. To move your score up toward the 0.803 target without changing the core model class, I replace the current (a,b) least-squares calibration with a tiny QWK-aligned “optimal rounding thresholds” fit on OOF predictions (same Ridge outputs, only post-processing changes). Finally, I keep paths and required `essay_id,score` submission format unchanged and add safety checks for row alignment and integer clipping.'
- What this solution (achieved 0.60074) has done: 'You’re currently far below the target (0.60074 vs 0.8030), so we should make a small, metric-aligned improvement without changing the core TF‑IDF+Ridge approach. The biggest safe gain here is to use stratified CV (so each fold matches the 1–6 label distribution) and to fit the QWK-optimal rounding thresholds on *out-of-fold predictions from the selected alpha* (rather than using non-stratified folds), which typically improves QWK with the same model. I also make the alpha selection metric consistent with the thresholded post-processing you actually use at test time (still same Ridge; only post-processing/selection). All paths and the LLM branch remain intact, and the script still always produce a valid `submission.csv`.'
- What this solution (achieved 0.60074) has done: 'We’re currently well below the target (0.6007 vs 0.8030), so we should make a small, metric-aligned improvement without changing the TF‑IDF+Ridge core. The biggest likely gain is to tune the TF‑IDF representation minimally (still TF‑IDF, same Ridge) by enabling `analyzer="char_wb"` with a small char n-gram range, which is a common robustness boost for essay scoring while keeping the same modeling approach. We also make the threshold search slightly more expressive (same thresholding logic) by expanding candidate quantiles a bit and iterating one extra pass, which typically improves QWK calibration with negligible runtime impact. All LLM/subprocess logic, paths, and submission schema remain unchanged, and the fallback still guarantees a valid `submission.csv`.'
- What this solution (achieved 0.60074) has done: 'Your current score (0.60074) is far below the target (0.8030), so we should make a small, metric-aligned improvement while keeping the same core TF‑IDF + Ridge fallback logic (since that’s what’s actually scoring when the LLM path fails). The minimal high-impact change is to use a *mixed* TF‑IDF representation (word n-grams + char_wb n-grams) via `FeatureUnion` (still TF‑IDF features, still Ridge), which is a common robustness boost for essay scoring without changing the learning algorithm. To better match QWK, we keep your same threshold-based discretization but fit thresholds on *out-of-fold predictions averaged across folds* and then refit a final model on all data with the selected alpha and learned thresholds (same approach, just slightly more consistent). All paths, subprocess/LLM logic, and submission writing remain intact, and the script still always produces a valid `submission.csv`.'
- What this solution (achieved 0.60074) has done: 'We’re far below the target (0.6007 vs 0.8030, higher-is-better), so the smallest likely QWK gain without changing the core “TF‑IDF + Ridge + thresholding” fallback is to make the post-processing more metric-aligned and stable. I keep the exact same model family (mixed TF‑IDF via `FeatureUnion` + `Ridge`) and training loop structure, but (1) increase CV folds from 3→5 for a more reliable alpha/threshold fit, and (2) fit thresholds using a slightly stronger QWK threshold search (same thresholding semantics; just a denser candidate grid + one extra refinement pass). I apply the exact same changes in both the in-notebook fallback and the subprocess `infer.py` fallback so behavior is consistent regardless of whether LLM inference fails. All paths remain unchanged and the script still always writes a valid `submission.csv` with `essay_id,score` and integer scores clipped to [1,6].'
- What this solution (achieved 0.60074) has done: 'Your current score (0.60074) is well below the target (0.8030), so we should make a small, metric-aligned improvement without changing the core “TF‑IDF (word+char) + Ridge + thresholding” approach. The biggest low-risk gain is to make the Ridge fit slightly more robust by using `RidgeCV` over the same alpha grid with proper CV (still Ridge, just selecting alpha more reliably), and to widen the alpha grid a bit so the model can land closer to its best regularization on this dataset. Then, keep your exact thresholding semantics, but fit the QWK-optimal thresholds on the final OOF predictions coming from the chosen alpha (rather than refitting thresholds per-alpha with different noise), which typically improves QWK with minimal code change. I apply the exact same fallback upgrades both in-notebook and inside `infer.py` so behavior is consistent regardless of whether the LLM branch succeeds, and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 0.60074) has done: 'We’re far below the target (0.6007 vs 0.8030), so we should make a small, metric-aligned improvement while keeping the same TF‑IDF + Ridge + threshold-discretization core. The biggest low-risk issue is that `RidgeCV(scoring=None)` selects alpha by MSE, which is misaligned with QWK; instead, we select alpha by out-of-fold QWK after applying the same thresholding you use at inference (same model class, same loop structure, just a metric-consistent selection). To keep changes minimal, we reuse your existing `StratifiedKFold`, threshold fitter, and feature pipeline, and only adjust: (1) alpha selection to use QWK on OOF preds, and (2) slightly widen the alpha grid a bit for a better chance of landing near the target. The exact same fallback logic is applied both in-notebook and inside `infer.py` so you don’t get inconsistent behavior depending on whether the LLM branch fails.'

# 9. Code solution

## === cell 0
from pathlib import Path
import subprocess, shlex
import os
import pandas as pd


def heuristic_fallback_submission(data_dir: str, sub_pth: str) -> None:
    """
    Dependency-light fallback that creates a valid submission.

    Core model stays TF-IDF + Ridge + threshold discretization.

    CHANGE (score toward target, minimal + metric-aligned):
    - Select Ridge alpha by OOF QWK *after applying the same thresholding postprocess* (instead of MSE via RidgeCV),
      because Kaggle evaluates QWK on discrete 1-6 scores.
    - Fit QWK-optimal thresholds once on OOF preds from the chosen alpha (same discretization semantics).
    """
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    sample_path = os.path.join(data_dir, "sample_submission.csv")

    train = pd.read_csv(train_path, usecols=["essay_id", "full_text", "score"])
    test = pd.read_csv(test_path, usecols=["essay_id", "full_text"])
    sub = pd.read_csv(sample_path, usecols=["essay_id", "score"])

    train["full_text"] = train["full_text"].fillna("").astype(str)
    test["full_text"] = test["full_text"].fillna("").astype(str)
    y = train["score"].astype(int).to_numpy()

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import FeatureUnion
        import numpy as np

        def qwk(y_true, y_pred, min_rating=1, max_rating=6):
            y_true = np.asarray(y_true, dtype=int)
            y_pred = np.asarray(y_pred, dtype=int)
            y_true = np.clip(y_true, min_rating, max_rating)
            y_pred = np.clip(y_pred, min_rating, max_rating)

            n = max_rating - min_rating + 1
            O = np.zeros((n, n), dtype=np.float64)
            for a_, b_ in zip(y_true, y_pred):
                O[a_ - min_rating, b_ - min_rating] += 1.0

            act_hist = np.bincount(y_true - min_rating, minlength=n).astype(np.float64)
            pred_hist = np.bincount(y_pred - min_rating, minlength=n).astype(np.float64)
            E = np.outer(act_hist, pred_hist)
            if E.sum() > 0:
                E *= O.sum() / E.sum()

            W = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(n):
                    W[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)

            denom = (W * E).sum()
            if denom == 0:
                return 1.0
            return 1.0 - (W * O).sum() / denom

        def apply_thresholds(pred_cont, thr):
            pred_cont = np.asarray(pred_cont, dtype=np.float64)
            thr = np.asarray(thr, dtype=np.float64)
            out = np.ones_like(pred_cont, dtype=np.int32)
            out[pred_cont >= thr[0]] = 2
            out[pred_cont >= thr[1]] = 3
            out[pred_cont >= thr[2]] = 4
            out[pred_cont >= thr[3]] = 5
            out[pred_cont >= thr[4]] = 6
            return out

        def fit_thresholds_qwk(pred_cont, y_true, init_thr=None, n_iter=4):
            pred_cont = np.asarray(pred_cont, dtype=np.float64)
            y_true = np.asarray(y_true, dtype=np.int32)

            if init_thr is None:
                means = []
                for c in range(1, 7):
                    m = pred_cont[y_true == c].mean() if np.any(y_true == c) else np.nan
                    means.append(m)
                g = np.nanmean(means)
                means = np.array(
                    [g if np.isnan(v) else v for v in means], dtype=np.float64
                )
                means.sort()
                init_thr = [(means[i] + means[i + 1]) / 2.0 for i in range(5)]
            thr = np.array(init_thr, dtype=np.float64)

            for i in range(1, 5):
                if thr[i] <= thr[i - 1]:
                    thr[i] = thr[i - 1] + 1e-3

            best_thr = thr.copy()
            best_score = qwk(y_true, apply_thresholds(pred_cont, best_thr))

            qs = np.linspace(0.01, 0.99, 161)
            cand = np.unique(np.quantile(pred_cont, qs))

            for _ in range(n_iter):
                for k in range(5):
                    lo = -np.inf if k == 0 else best_thr[k - 1] + 1e-6
                    hi = np.inf if k == 4 else best_thr[k + 1] - 1e-6
                    feasible = cand[(cand > lo) & (cand < hi)]
                    if feasible.size == 0:
                        continue
                    local_best_t = best_thr[k]
                    local_best = best_score
                    for t in feasible:
                        tmp = best_thr.copy()
                        tmp[k] = float(t)
                        s = qwk(y_true, apply_thresholds(pred_cont, tmp))
                        if s > local_best + 1e-12:
                            local_best = s
                            local_best_t = float(t)
                    best_thr[k] = local_best_t
                    best_score = local_best

            return best_thr, float(best_score)

        X_text = train["full_text"].values
        X_test_text = test["full_text"].values

        word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=120000,
            strip_accents="unicode",
            lowercase=True,
            sublinear_tf=True,
        )
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=120000,
            strip_accents="unicode",
            lowercase=True,
            sublinear_tf=True,
        )
        vectorizer = FeatureUnion([("word", word_vec), ("char", char_vec)])

        X = vectorizer.fit_transform(X_text)
        X_test = vectorizer.transform(X_test_text)

        alphas = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        best_alpha = None
        best_thr = None
        best_cv = -1e9
        best_oof = None

        for a in alphas:
            oof = np.zeros(len(y), dtype=np.float64)
            for tr_idx, va_idx in skf.split(X, y):
                model = Ridge(alpha=float(a), random_state=42)
                model.fit(X[tr_idx], y[tr_idx])
                oof[va_idx] = model.predict(X[va_idx])

            thr_a, score_a = fit_thresholds_qwk(oof, y, init_thr=None, n_iter=4)
            if score_a > best_cv + 1e-12:
                best_cv = score_a
                best_alpha = float(a)
                best_thr = thr_a
                best_oof = oof

        best_thr, _ = fit_thresholds_qwk(best_oof, y, init_thr=best_thr, n_iter=4)

        final_model = Ridge(alpha=best_alpha, random_state=42)
        final_model.fit(X, y)
        te_pred_cont = final_model.predict(X_test)

        te_pred = apply_thresholds(te_pred_cont, best_thr).astype(int)
        te_pred = np.clip(te_pred, 1, 6)

        pred_df = pd.DataFrame({"essay_id": test["essay_id"].values, "score": te_pred})
        sub = sub[["essay_id"]].merge(pred_df, on="essay_id", how="left")
        sub["score"] = (
            sub["score"]
            .fillna(int(round(float(train["score"].mean()))))
            .astype(int)
            .clip(1, 6)
        )
        sub.to_csv(sub_pth, index=False)
        return

    except Exception as e:
        print(
            f"[WARN] sklearn TF-IDF fallback failed ({type(e).__name__}: {e}); using length-binning fallback."
        )

    tr_len = train["full_text"].astype(str).str.len()
    te_len = test["full_text"].astype(str).str.len()

    qs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    edges = tr_len.quantile(qs).to_numpy()
    edges = sorted(set(int(x) for x in edges))

    if len(edges) < 3:
        global_pred = int(round(float(train["score"].mean())))
        pred = pd.Series([global_pred] * len(test), index=test.index)
    else:
        edges[0] = max(edges[0], 0)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1

        tr_bins = pd.cut(tr_len, bins=edges, include_lowest=True, duplicates="drop")
        bin_mean = train.groupby(tr_bins, observed=False)["score"].mean()

        te_bins = pd.cut(te_len, bins=edges, include_lowest=True, duplicates="drop")
        pred = te_bins.map(bin_mean)

        pred = pd.Series(pred, index=test.index, dtype="float64")
        pred = pred.fillna(float(train["score"].mean()))
        pred = pred.round().astype(int)

    pred = pred.clip(1, 6)
    pred_df = pd.DataFrame({"essay_id": test["essay_id"].values, "score": pred.values})
    sub = sub[["essay_id"]].merge(pred_df, on="essay_id", how="left")
    sub["score"] = sub["score"].fillna(3).astype(int).clip(1, 6)
    sub.to_csv(sub_pth, index=False)


infer_py = r"""import os, sys, re, gc, argparse
import pandas as pd

# Ensure pure-Python protobuf before importing peft/transformers to avoid MessageFactory.GetPrototype crash
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

def heuristic_fallback_submission(data_dir: str, sub_pth: str) -> None:
    train_path = os.path.join(data_dir, "train.csv")
    test_path  = os.path.join(data_dir, "test.csv")
    sample_path = os.path.join(data_dir, "sample_submission.csv")

    train = pd.read_csv(train_path, usecols=["essay_id","full_text", "score"])
    test  = pd.read_csv(test_path, usecols=["essay_id", "full_text"])
    sub   = pd.read_csv(sample_path, usecols=["essay_id", "score"])

    train["full_text"] = train["full_text"].fillna("").astype(str)
    test["full_text"]  = test["full_text"].fillna("").astype(str)
    y = train["score"].astype(int).to_numpy()

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import FeatureUnion
        import numpy as np

        def qwk(y_true, y_pred, min_rating=1, max_rating=6):
            y_true = np.asarray(y_true, dtype=int)
            y_pred = np.asarray(y_pred, dtype=int)
            y_true = np.clip(y_true, min_rating, max_rating)
            y_pred = np.clip(y_pred, min_rating, max_rating)

            n = max_rating - min_rating + 1
            O = np.zeros((n, n), dtype=np.float64)
            for a_, b_ in zip(y_true, y_pred):
                O[a_ - min_rating, b_ - min_rating] += 1.0

            act_hist = np.bincount(y_true - min_rating, minlength=n).astype(np.float64)
            pred_hist = np.bincount(y_pred - min_rating, minlength=n).astype(np.float64)
            E = np.outer(act_hist, pred_hist)
            if E.sum() > 0:
                E *= O.sum() / E.sum()

            W = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(n):
                    W[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)

            denom = (W * E).sum()
            if denom == 0:
                return 1.0
            return 1.0 - (W * O).sum() / denom

        def apply_thresholds(pred_cont, thr):
            pred_cont = np.asarray(pred_cont, dtype=np.float64)
            thr = np.asarray(thr, dtype=np.float64)
            out = np.ones_like(pred_cont, dtype=np.int32)
            out[pred_cont >= thr[0]] = 2
            out[pred_cont >= thr[1]] = 3
            out[pred_cont >= thr[2]] = 4
            out[pred_cont >= thr[3]] = 5
            out[pred_cont >= thr[4]] = 6
            return out

        def fit_thresholds_qwk(pred_cont, y_true, init_thr=None, n_iter=4):
            pred_cont = np.asarray(pred_cont, dtype=np.float64)
            y_true = np.asarray(y_true, dtype=np.int32)

            if init_thr is None:
                means = []
                for c in range(1, 7):
                    m = pred_cont[y_true == c].mean() if (y_true == c).any() else float("nan")
                    means.append(m)
                g = np.nanmean(means)
                means = np.array([g if (m != m) else m for m in means], dtype=np.float64)
                means.sort()
                init_thr = [(means[i] + means[i+1]) / 2.0 for i in range(5)]
            thr = np.array(init_thr, dtype=np.float64)
            for i in range(1, 5):
                if thr[i] <= thr[i-1]:
                    thr[i] = thr[i-1] + 1e-3

            best_thr = thr.copy()
            best_score = qwk(y_true, apply_thresholds(pred_cont, best_thr))

            qs = np.linspace(0.01, 0.99, 161)
            cand = np.unique(np.quantile(pred_cont, qs))

            for _ in range(n_iter):
                for k in range(5):
                    lo = -1e18 if k == 0 else best_thr[k-1] + 1e-6
                    hi =  1e18 if k == 4 else best_thr[k+1] - 1e-6
                    feasible = cand[(cand > lo) & (cand < hi)]
                    if feasible.size == 0:
                        continue
                    local_best_t = best_thr[k]
                    local_best = best_score
                    for t in feasible:
                        tmp = best_thr.copy()
                        tmp[k] = float(t)
                        s = qwk(y_true, apply_thresholds(pred_cont, tmp))
                        if s > local_best + 1e-12:
                            local_best = s
                            local_best_t = float(t)
                    best_thr[k] = local_best_t
                    best_score = local_best
            return best_thr, float(best_score)

        X_text = train["full_text"].values
        X_test_text = test["full_text"].values

        # mixed TF-IDF (word + char_wb), still TF-IDF + Ridge.
        word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=120000,
            strip_accents="unicode",
            lowercase=True,
            sublinear_tf=True,
        )
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=120000,
            strip_accents="unicode",
            lowercase=True,
            sublinear_tf=True,
        )
        vectorizer = FeatureUnion([("word", word_vec), ("char", char_vec)])

        X = vectorizer.fit_transform(X_text)
        X_test = vectorizer.transform(X_test_text)

        # CHANGE (score toward target): slightly wider alpha grid, still Ridge.
        alphas = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # CHANGE (score toward target): choose alpha by OOF QWK using same thresholding post-process.
        best_alpha = None
        best_thr = None
        best_cv = -1e9
        best_oof = None

        for a in alphas:
            oof = np.zeros(len(y), dtype=np.float64)
            for tr_idx, va_idx in skf.split(X, y):
                model = Ridge(alpha=float(a), random_state=42)
                model.fit(X[tr_idx], y[tr_idx])
                oof[va_idx] = model.predict(X[va_idx])

            thr_a, score_a = fit_thresholds_qwk(oof, y, init_thr=None, n_iter=4)
            if score_a > best_cv + 1e-12:
                best_cv = score_a
                best_alpha = float(a)
                best_thr = thr_a
                best_oof = oof

        best_thr, _ = fit_thresholds_qwk(best_oof, y, init_thr=best_thr, n_iter=4)

        final_model = Ridge(alpha=best_alpha, random_state=42)
        final_model.fit(X, y)

        te_pred_cont = final_model.predict(X_test)
        te_pred = apply_thresholds(te_pred_cont, best_thr).astype(int)
        te_pred = np.clip(te_pred, 1, 6)

        pred_df = pd.DataFrame({"essay_id": test["essay_id"].values, "score": te_pred})
        sub = sub[["essay_id"]].merge(pred_df, on="essay_id", how="left")
        sub["score"] = sub["score"].fillna(int(round(float(train["score"].mean())))).astype(int).clip(1, 6)
        sub.to_csv(sub_pth, index=False)
        return
    except Exception as e:
        print(f"[WARN] sklearn TF-IDF fallback failed ({type(e).__name__}: {e}); using length-binning fallback.")

    tr_len = train["full_text"].astype(str).str.len()
    te_len = test["full_text"].astype(str).str.len()

    qs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    edges = tr_len.quantile(qs).to_numpy()
    edges = sorted(set(int(x) for x in edges))

    if len(edges) < 3:
        global_pred = int(round(float(train["score"].mean())))
        pred = pd.Series([global_pred] * len(test), index=test.index)
    else:
        edges[0] = max(edges[0], 0)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i-1]:
                edges[i] = edges[i-1] + 1

        tr_bins = pd.cut(tr_len, bins=edges, include_lowest=True, duplicates="drop")
        bin_mean = train.groupby(tr_bins, observed=False)["score"].mean()

        te_bins = pd.cut(te_len, bins=edges, include_lowest=True, duplicates="drop")
        pred = te_bins.map(bin_mean)

        pred = pd.Series(pred, index=test.index, dtype="float64")
        pred = pred.fillna(float(train["score"].mean()))
        pred = pred.round().astype(int)

    pred = pred.clip(1, 6)
    pred_df = pd.DataFrame({"essay_id": test["essay_id"].values, "score": pred.values})
    sub = sub[["essay_id"]].merge(pred_df, on="essay_id", how="left")
    sub["score"] = sub["score"].fillna(3).astype(int).clip(1, 6)
    sub.to_csv(sub_pth, index=False)


def main(args):
    data_dir = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"

    try:
        import torch
        from tqdm import tqdm
        from peft import PeftModel
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)

        model = AutoModelForCausalLM.from_pretrained(
            args.model_pth,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, args.lora_pth)
        tokenizer = AutoTokenizer.from_pretrained(args.model_pth, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token

        def preprocess(sample, text=False, infer_mode=False, max_seq=args.max_length, return_tensors=None):
            sys_prompt = (
                "Please read the following essay and assign a score of 1,2,3,4,5,6 where 6 is the best. "
                "Output only a single number with no explanation.\n\n"
            )
            prompt = sample["full_text"]
            answer = "" if infer_mode else str(sample["score"])

            messages = [
                {"role": "user", "content": sys_prompt + prompt},
                {"role": "assistant", "content": f"\n\nThe score is: " + answer},
            ]
            formatted_sample = tokenizer.apply_chat_template(messages, tokenize=False)
            if infer_mode:
                formatted_sample = formatted_sample.replace("<|eot_id|>", "")

            tokenized_sample = tokenizer(
                formatted_sample,
                padding=True,
                return_tensors=return_tensors,
                truncation=True,
                add_special_tokens=False,
                max_length=max_seq,
            )

            if return_tensors == "pt":
                tokenized_sample["labels"] = tokenized_sample["input_ids"].clone()
            else:
                tokenized_sample["labels"] = tokenized_sample["input_ids"].copy()

            return formatted_sample if text else tokenized_sample

        df_test = pd.read_csv(f"{data_dir}/test.csv")
        sub = pd.read_csv(f"{data_dir}/sample_submission.csv")

        test_preds = []
        for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
            tokenized_sample = preprocess(row, infer_mode=True, max_seq=args.max_length, return_tensors="pt")
            device = next(model.parameters()).device
            tokenized_sample = {k: v.to(device) for k, v in tokenized_sample.items()}

            generated_ids = model.generate(
                **tokenized_sample,
                max_new_tokens=2,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            pred = 3
            try:
                if "The score is:" in decoded:
                    tail = decoded.rsplit("The score is:", 1)[1]
                elif "The score is: " in decoded:
                    tail = decoded.rsplit("The score is: ", 1)[1]
                else:
                    tail = decoded

                m = re.search(r"\b([1-6])\b", tail)
                if m:
                    pred = int(m.group(1))
                else:
                    m2 = re.search(r"(\d)", tail)
                    if m2:
                        pred = int(m2.group(1))
            except Exception:
                pred = 3

            pred = max(1, min(6, int(pred)))
            test_preds.append(pred)

        if len(test_preds) != len(sub):
            raise RuntimeError(f"LLM inference produced {len(test_preds)} preds for {len(sub)} rows")

        sub["score"] = pd.Series(test_preds).astype(int).clip(1, 6)
        sub.to_csv(args.sub_pth, index=False)

        del model, tokenizer
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    except Exception as e:
        print(f"[WARN] Falling back to heuristic submission due to: {type(e).__name__}: {e}")
        heuristic_fallback_submission(data_dir=data_dir, sub_pth=args.sub_pth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pth", type=str, required=True)
    parser.add_argument("--lora_pth", type=str, required=True)
    parser.add_argument("--sub_pth", type=str, required=True)
    parser.add_argument("--max_length", type=int, required=True)
    args = parser.parse_args()
    main(args)
"""
Path("infer.py").write_text(infer_py)



## === cell 1
cmd = """python infer.py \
    --max_length 2048 \
    --sub_pth submission.csv \
    --model_pth /kaggle/input/llama-3-8b-instruct/Meta-Llama-3-8B-Instruct \
    --lora_pth /kaggle/input/llama-3-8b-lora-fine-tuned-exp-1/Meta-Llama-3-8B-Instruct-max-len-1024-fold-1-exp-1-ckpt
"""

env = os.environ.copy()
env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

try:
    out = subprocess.check_output(
        shlex.split(cmd), stderr=subprocess.STDOUT, env=env, timeout=600
    ).decode("utf-8", errors="ignore")
    print(out)
except subprocess.TimeoutExpired as e:
    print("[WARN] infer.py timed out; captured output so far:\n")
    try:
        print((e.output or b"").decode("utf-8", errors="ignore"))
    except Exception:
        pass
except subprocess.CalledProcessError as e:
    print("[WARN] infer.py failed; captured output:\n")
    print(e.output.decode("utf-8", errors="ignore"))

need_fallback = False
if not os.path.exists("submission.csv"):
    need_fallback = True
else:
    try:
        tmp = pd.read_csv("submission.csv")
        if list(tmp.columns) != ["essay_id", "score"]:
            need_fallback = True
    except Exception:
        need_fallback = True

if need_fallback:
    print(
        "[WARN] submission.csv missing/invalid; writing heuristic fallback submission."
    )
    heuristic_fallback_submission(
        data_dir="/kaggle/input/learning-agency-lab-automated-essay-scoring-2",
        sub_pth="submission.csv",
    )

sub = pd.read_csv("submission.csv")
print(sub.head())
print(sub.shape)
print(sub.columns.tolist())
print(sub["score"].min(), sub["score"].max())

assert list(sub.columns) == ["essay_id", "score"]
assert (
    sub.shape[0]
    == pd.read_csv(
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
    ).shape[0]
)

sub["score"] = (
    pd.to_numeric(sub["score"], errors="coerce")
    .fillna(3)
    .round()
    .astype(int)
    .clip(1, 6)
)
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
