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

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
transformers==4.53.3

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

0.7756038481842737

# 6. Current score

0.68135

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime blocker by making the Hugging Face loader treat the provided `/kaggle/input/...` paths as local directories (the current error happens because the hub validators interpret the absolute path as an invalid repo id). Then I add a safe fallback: if that local model directory doesn’t exist in your environment, the script still run end-to-end by producing a valid submission using a simple constant prediction (so you always get a `.csv`). Finally, I ensure a `score` column is always created, cast to int, clipped to the valid 1–6 range, and written with the exact required columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from a modeling bug: you’re loading a causal LM and then applying a `Linear(1,1)` head to the last-token logits, which can’t produce a 6-class prediction (so it either errors or degenerates to constant/invalid behavior). To move your score toward the 0.775 target with minimal semantic change, I keep your overall “load local HF artifacts if available, else constant fallback” structure, but fix the head to be a proper 6-class classifier applied to a scalar feature derived from the model outputs (mean logit over vocab at last token). I also vectorize inference with a DataLoader-style batch loop so it finishes within the time limit and avoid per-row GPU cache clearing (which slows and can destabilize). The fallback constant submission remains intact so the notebook always produces a valid `submission.csv`.'
- What this solution (achieved 0.67285) has done: 'Your current 0.0 score is consistent with producing essentially uninformative/invalid predictions (constant `3` fallback or a head/model mismatch), so the smallest safe move toward the 0.7756 target is to make the script always produce a non-constant, label-aligned prediction while keeping your “local HF artifacts if available, else fallback” structure. I keep your existing causal-LM + simple head approach, but fix two likely score-killers: (1) ensure tokenization includes special tokens (so the model sees a sensible input format) and (2) replace the constant fallback with a deterministic, data-driven baseline (length-based mapping) that is legitimate and usually scores well above 0.0 on QWK. I also enforce that predictions match the required 1–6 integer scale and that the submission rows exactly align to `essay_id`. These are minimal changes that should increase score substantially without changing your core loading/inference design.'
- What this solution (achieved 0.67062) has done: 'To move your QWK from 0.67285 toward the 0.7756 target (higher is better), I keep your exact overall structure (local HF model+head if available, else length-based baseline), but make the fallback baseline slightly more label-aligned by using word counts (more robust than raw character counts) and by calibrating the binning with a train-fitted monotonic mapping from the continuous proxy to the 1–6 classes. These are small, legitimate changes that usually improve ranking/ordinal agreement without changing your “no training / no fine-tuning” approach. I also make the model-path branch more robust by ensuring a pad token is set (common for causal LMs) to avoid padding-related weirdness, which can silently hurt predictions if the local model exists. Submission writing, columns, clipping to 1–6, and file path remain unchanged.'
- What this solution (achieved 0.63658) has done: 'Your current score (0.67062) is below the target (0.77560), so we should make a small, legitimate improvement that better aligns the ordinal predictions with QWK without changing your overall “use local HF model if present, else proxy baseline” design. The safest lever is the proxy baseline: keep word-count as the single feature, but replace the median/quantile threshold heuristic with train-fitted ordinal thresholds that explicitly maximize Quadratic Weighted Kappa on the training distribution. This preserves your core logic (proxy → thresholds → digitize into 1–6) while directly optimizing for the competition metric, and it stays fast. The model branch is left intact except for keeping output handling unchanged.'
- What this solution (achieved 0.67902) has done: 'We keep your proxy→thresholds→digitize core logic intact, but fix a metric-alignment bug: the threshold fitter currently maximizes per-class “correct bin counts”, not Quadratic Weighted Kappa, so it can easily overfit the wrong objective and hurt leaderboard QWK. With minimal change, we instead fit the 5 word-count thresholds by directly maximizing QWK on a held-out validation split (deterministic), using a small coordinate-wise local search around quantile-initialized thresholds. This preserves the same single-feature baseline and same prediction semantics (word count + digitize to 1–6), while making the thresholds explicitly optimized for the competition metric. The model branch and submission writing stay the same, and we still always output a valid `submission.csv` with `essay_id,score`.'
- What this solution (achieved 0.0) has done: 'Your current score (0.67902) is below the target (0.77560), so we should make a small, legitimate improvement that better matches QWK without changing your core “single proxy feature (word count) → 5 thresholds → digitize into 1–6” approach. The minimal change is to fit thresholds on *all* training data (instead of a fixed 80/20 split) while still optimizing QWK, using deterministic cross-validated QWK as the objective; this usually improves generalization for a 1D ordinal mapping. I’m also fixing the validation split hash to be stable across Python runs (Python’s built-in `hash()` is randomized per process), which can otherwise make thresholds noisy and hurt LB score. Everything else (proxy feature, digitize prediction semantics, clipping 1–6, and submission format/path) stays the same.'
- What this solution (achieved 0.68135) has done: 'I fix the runtime blocker in the fold assignment by removing the unsupported NumPy `astype("string")` call and replacing it with a safe conversion that works under Python 3.12 + NumPy. This allow the threshold fitting cell to run, which in turn defines `proxy_thresholds` so the prediction cell no longer fails with `NameError`. I keep your proxy feature, threshold optimization logic, and submission formatting unchanged, only making minimal type-handling fixes to restore end-to-end execution and produce a valid `submission.csv`. These changes are score-positive versus the current 0.0 because they enable the intended (non-constant) proxy-threshold predictions.'
- What this solution (achieved 0.68135) has done: 'Your current score (0.68135) is below the target (0.77560), so we should make a small, legitimate improvement that keeps your exact “single proxy feature (word count) → 5 thresholds → digitize into 1–6” core logic. The most likely score-killer is that your cross-validation objective is computed using thresholds fitted on *all* data (including each fold’s validation), which can select thresholds that don’t generalize well to the test distribution. I change the CV scoring to be leakage-free (fit thresholds on each fold’s training split, score on that fold’s validation), then refit a final threshold set on all training data using the same coordinate search. This is still the same proxy+threshold approach (no new features, no new model), but should move QWK upward toward the target.'
- What this solution (achieved 0.57343) has done: 'We keep your single-proxy “word count → 5 thresholds → digitize to 1–6” approach and the same coordinate-search fitter, but fix a key bug: your CV objective currently evaluates one shared threshold set without refitting per fold, which makes the “CV” score noisy/misleading and can select thresholds that don’t generalize to test. Concretely, we (a) change the CV scorer used inside the final search to be truly leakage-free by fitting thresholds on each fold’s training split before scoring that fold’s validation, and (b) then refit one final threshold set on all training data using the same search procedure. This is a minimal, metric-aligned change that should move QWK upward toward the 0.7756 target without altering your core model/baseline semantics. Submission writing/format and the optional local-model branch remain unchanged.'
- What this solution (achieved 0.57343) has done: 'The timeout is almost certainly caused by the proxy-threshold fitting in cell 6: it repeatedly recomputes QWK via Python loops inside a nested coordinate search with per-fold refits (very high call count). I keep the exact same threshold-search logic and CV semantics, but replace the loop-based QWK and confusion-matrix construction with a vectorized, precomputed version (including a precomputed weight matrix), and avoid repeated allocations in the scoring function. I also speed up fold assignment by hashing essay_ids in a vectorized way (CRC32) while keeping it stable/deterministic. These changes are mathematically equivalent (same QWK definition and same optimization objective), but drastically reduce Python overhead so the script fits within 600s.'
- What this solution (achieved 0.68135) has done: 'Your current QWK (0.57343) is far below the 0.7756 target, so we need a small but meaningful improvement without changing your core “single proxy feature (word count) → 5 thresholds → digitize to 1–6” approach. The biggest likely issue is that the threshold optimizer is effectively trying to maximize a very expensive CV-refit objective, which tends to be noisy and can settle on worse generalization thresholds; we keep the same coordinate-search but fit thresholds directly on out-of-fold predictions from a leakage-free CV, then derive thresholds by optimizing QWK on those OOF pairs (still the same proxy→threshold mapping). This keeps the exact same inference semantics (word-count digitization), but makes the thresholds better match what generalizes. I also vectorize fold assignment (CRC32) and ensure determinism; submission format/path remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch



## === cell 1
print("CUDA available:", torch.cuda.is_available())
print("CUDA device_count:", torch.cuda.device_count())



## === cell 2
DATA_DIR = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
test_path = os.path.join(DATA_DIR, "test.csv")
train_path = os.path.join(DATA_DIR, "train.csv")

test = pd.read_csv(test_path, usecols=["essay_id", "full_text"])
train = pd.read_csv(train_path, usecols=["essay_id", "full_text", "score"])

print("Loaded test:", test.shape, test.columns.tolist())
print("Loaded train:", train.shape, train.columns.tolist())



## === cell 3
reverse_score_mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}



## === cell 4
from transformers import AutoModelForCausalLM, AutoTokenizer

local_model_path = "/kaggle/input/hf-proxy-roberta"
local_tokenizer_path = "/kaggle/input/hf-proxy-roberta"

have_local_model = os.path.isdir(local_model_path) and os.path.isdir(
    local_tokenizer_path
)
print("Local model dir exists:", have_local_model, "| path:", local_model_path)

tokenizer = None
model = None
head = None
use_model = False
device = "cuda" if torch.cuda.is_available() else "cpu"

if have_local_model:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            local_tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.sep_token_id is not None:
                tokenizer.pad_token = tokenizer.sep_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        model = (
            AutoModelForCausalLM.from_pretrained(
                local_model_path,
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True,
                local_files_only=True,
            )
            .to(device)
            .eval()
        )

        head_weights_path = os.path.join(local_model_path, "classification_head.pth")

        if os.path.isfile(head_weights_path):
            head_weights = torch.load(head_weights_path, map_location=device)

            head = torch.nn.Linear(1, 6, bias=True).to(device).eval()

            if isinstance(head_weights, dict):
                if "weight" in head_weights and "bias" in head_weights:
                    head.weight.data = head_weights["weight"].to(device)
                    head.bias.data = head_weights["bias"].to(device)
                elif "state_dict" in head_weights:
                    head.load_state_dict(head_weights["state_dict"])
                else:
                    try:
                        head.load_state_dict(head_weights)
                    except Exception:
                        raise ValueError(
                            f"Unrecognized head weight dict keys: {list(head_weights.keys())[:10]}"
                        )
            elif torch.is_tensor(head_weights):
                w = head_weights.to(device)
                if w.numel() == 6:
                    head.weight.data = w.view(6, 1)
                    head.bias.data.zero_()
                elif w.shape == head.weight.data.shape:
                    head.weight.data = w
                    head.bias.data.zero_()
                else:
                    raise ValueError(
                        f"Unexpected tensor shape for head weights: {tuple(w.shape)}"
                    )
            else:
                raise ValueError(
                    f"Unexpected type for head weights: {type(head_weights)}"
                )

            use_model = True
            print("Hooray! Model/tokenizer/head loaded.")
        else:
            print(
                "WARNING: classification_head.pth not found; will fall back to baseline predictions."
            )
    except Exception as e:
        print(
            "WARNING: Failed to load local model/tokenizer/head; will fall back to baseline predictions."
        )
        print("Load error:", repr(e))
else:
    print(
        "WARNING: Local model directory not available; will fall back to baseline predictions."
    )



## === cell 5
if "score" not in test.columns:
    test["score"] = np.nan



## === cell 6
import zlib


def _word_count(series: pd.Series) -> np.ndarray:
    s = series.astype("string").fillna("")
    return s.str.split().str.len().to_numpy(dtype=np.int32)


def _qwk_fast_from_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    W: np.ndarray,
    min_rating: int = 1,
    max_rating: int = 6,
) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    y_true = np.clip(y_true, min_rating, max_rating)
    y_pred = np.clip(y_pred, min_rating, max_rating)

    n = max_rating - min_rating + 1
    a = y_true - min_rating
    b = y_pred - min_rating
    idx = a * n + b
    O = np.bincount(idx, minlength=n * n).reshape(n, n).astype(np.float64)

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    Osum = O.sum()
    Esum = E.sum()
    if Esum > 0:
        E = E / Esum * Osum

    num = float((W * O).sum())
    den = float((W * E).sum())
    if den == 0.0:
        return 0.0
    return 1.0 - (num / den)


def qwk(
    y_true: np.ndarray, y_pred: np.ndarray, min_rating: int = 1, max_rating: int = 6
) -> float:
    n = max_rating - min_rating + 1
    ii = np.arange(n, dtype=np.float64)
    W = ((ii[:, None] - ii[None, :]) ** 2) / ((n - 1) ** 2)
    return _qwk_fast_from_labels(y_true, y_pred, W, min_rating, max_rating)


def predict_from_proxy(texts, thresholds: np.ndarray) -> np.ndarray:
    n_words = _word_count(pd.Series(texts, dtype="string"))
    return (
        np.digitize(n_words.astype(np.float32), thresholds, right=False) + 1
    ).astype(np.int32)


def _apply_thresholds(
    x: np.ndarray, thr: np.ndarray, min_rating: int = 1
) -> np.ndarray:
    return (
        np.digitize(x.astype(np.float32), thr.astype(np.float32), right=False)
        + min_rating
    ).astype(np.int32)


def _stable_fold_id_from_essay_id(
    essay_ids: np.ndarray, n_folds: int = 5
) -> np.ndarray:
    ids = np.asarray(essay_ids, dtype=str)
    out = np.empty(ids.shape[0], dtype=np.int32)
    for i, s in enumerate(ids):
        out[i] = zlib.crc32(s.encode("utf-8")) % n_folds
    return out


def _enforce_monotone(t: np.ndarray) -> np.ndarray:
    t = t.astype(np.float32).copy()
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + 1.0
    for i in range(len(t) - 2, -1, -1):
        if t[i] >= t[i + 1]:
            t[i] = t[i + 1] - 1.0
    return t


def _coordinate_search_thresholds(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    score_fn,
    min_rating: int = 1,
    max_rating: int = 6,
    rounds: int = 3,
    max_unique_grid: int = 5000,
) -> tuple[np.ndarray, float]:
    K = max_rating - min_rating + 1  # 6 => 5 thresholds
    qs = [c / K for c in range(1, K)]
    thr = np.quantile(x_fit, qs).astype(np.float32)
    thr = _enforce_monotone(thr)

    ux = np.unique(x_fit)
    if len(ux) > max_unique_grid:
        stride = int(np.ceil(len(ux) / max_unique_grid))
        ux = ux[::stride]

    best_q = float(score_fn(thr))

    for _round in range(rounds):
        improved = False
        for i in range(len(thr)):
            cur = float(thr[i])
            pos = int(np.searchsorted(ux, cur))

            offsets = np.array(
                [-80, -40, -20, -10, -5, -2, 0, 2, 5, 10, 20, 40, 80], dtype=int
            )
            idxs = np.clip(pos + offsets, 0, len(ux) - 1)

            cands = []
            for j in np.unique(idxs):
                if j < len(ux) - 1:
                    cands.append(0.5 * (float(ux[j]) + float(ux[j + 1])))
                else:
                    cands.append(float(ux[j]))
            cands = sorted(set(cands))

            best_local_t = cur
            best_local_q = best_q
            for tval in cands:
                t2 = thr.copy()
                t2[i] = float(tval)
                t2 = _enforce_monotone(t2)
                q2 = float(score_fn(t2))
                if q2 > best_local_q:
                    best_local_q = q2
                    best_local_t = float(tval)

            if best_local_q > best_q:
                thr[i] = best_local_t
                thr = _enforce_monotone(thr)
                best_q = best_local_q
                improved = True
        if not improved:
            break

    return thr.astype(np.float32), float(best_q)


def fit_qwk_optimal_thresholds_from_train(
    train_df: pd.DataFrame,
    text_col: str = "full_text",
    y_col: str = "score",
    min_rating: int = 1,
    max_rating: int = 6,
) -> np.ndarray:
    tmp = train_df[[text_col, y_col, "essay_id"]].copy()
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce").astype("Int64")
    tmp = tmp.dropna(subset=[y_col])
    tmp[text_col] = tmp[text_col].astype("string").fillna("")

    x_all = _word_count(tmp[text_col]).astype(np.float32)
    y_all = tmp[y_col].astype(int).to_numpy(dtype=np.int32)

    n = max_rating - min_rating + 1
    ii = np.arange(n, dtype=np.float64)
    W = ((ii[:, None] - ii[None, :]) ** 2) / ((n - 1) ** 2)

    n_folds = 5
    fold_id = _stable_fold_id_from_essay_id(
        tmp["essay_id"].astype(str).to_numpy(), n_folds=n_folds
    )

    oof_pred = np.empty_like(y_all, dtype=np.int32)
    fold_scores = []

    for f in range(n_folds):
        tr = fold_id != f
        va = fold_id == f
        x_tr, y_tr = x_all[tr], y_all[tr]
        x_va, y_va = x_all[va], y_all[va]

        def _fold_train_score_fn(thr: np.ndarray) -> float:
            thr = _enforce_monotone(thr)
            pred_tr = _apply_thresholds(x_tr, thr, min_rating=min_rating)
            return _qwk_fast_from_labels(y_tr, pred_tr, W, min_rating, max_rating)

        thr_f, _best_train = _coordinate_search_thresholds(
            x_fit=x_tr,
            y_fit=y_tr,
            score_fn=_fold_train_score_fn,
            min_rating=min_rating,
            max_rating=max_rating,
            rounds=3,
        )

        pred_va = _apply_thresholds(x_va, thr_f, min_rating=min_rating)
        oof_pred[va] = pred_va
        fold_scores.append(
            float(_qwk_fast_from_labels(y_va, pred_va, W, min_rating, max_rating))
        )

    print(
        f"{n_folds}-fold (leakage-free) CV proxy-threshold QWK (diagnostic):",
        float(np.mean(fold_scores)),
    )
    print(
        "OOF QWK (diagnostic):",
        float(_qwk_fast_from_labels(y_all, oof_pred, W, min_rating, max_rating)),
    )

    def _oof_score_fn(thr: np.ndarray) -> float:
        thr = _enforce_monotone(thr)
        pred = _apply_thresholds(x_all, thr, min_rating=min_rating)
        return _qwk_fast_from_labels(y_all, pred, W, min_rating, max_rating)

    thr_oof, best_oof = _coordinate_search_thresholds(
        x_fit=x_all,
        y_fit=y_all,
        score_fn=_oof_score_fn,
        min_rating=min_rating,
        max_rating=max_rating,
        rounds=3,
    )

    train_pred = _apply_thresholds(x_all, thr_oof, min_rating=min_rating)
    print(
        "Train proxy-threshold QWK (diagnostic):",
        float(_qwk_fast_from_labels(y_all, train_pred, W, min_rating, max_rating)),
    )
    print("Optimized-on-train QWK objective value (diagnostic):", float(best_oof))

    return thr_oof.astype(np.float32)


proxy_thresholds = fit_qwk_optimal_thresholds_from_train(train, "full_text", "score")
print("Proxy thresholds (word counts) between classes:", proxy_thresholds.tolist())



## === cell 7
max_length = 512

if use_model:
    batch_size = 8 if device == "cuda" else 2

    texts = test["full_text"].astype(str).to_numpy()
    pred_scores = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size].tolist()

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            out = model(**inputs).logits  # [B, T, V]
            last_token_logits = out[:, -1, :]  # [B, V]
            feat = last_token_logits.mean(dim=1, keepdim=True)  # [B, 1]

            cls_logits = head(feat)  # [B, 6]
            pred_class = torch.argmax(cls_logits, dim=1).detach().cpu().numpy()  # 0..5

            for c in pred_class.tolist():
                pred_scores.append(reverse_score_mapping.get(int(c), 3))

    test["score"] = pred_scores
else:
    test["score"] = predict_from_proxy(
        test["full_text"].astype(str).to_numpy(), proxy_thresholds
    )



## === cell 8
test["score"] = (
    pd.to_numeric(test["score"], errors="coerce").fillna(3).round().astype(int)
)
test["score"] = test["score"].clip(1, 6)

print(test[["essay_id", "score"]].head())



## === cell 9
sub = test[["essay_id", "score"]].copy()
sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print("Saved:", sub_path, "| shape:", sub.shape)
print(sub.head())
print("Score distribution:", sub["score"].value_counts().sort_index().to_dict())
