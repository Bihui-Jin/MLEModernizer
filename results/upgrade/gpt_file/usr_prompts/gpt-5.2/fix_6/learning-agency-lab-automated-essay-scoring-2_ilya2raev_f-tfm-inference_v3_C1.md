# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

datasets==4.4.1
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
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
vega-datasets==0.9.0

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

0.8029422989347521

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.03578) has done: 'I fix the crash by making the model path robust: if the provided `/kaggle/input/f-tfm-small/funnel-small-ft_ver4` folder is missing, the code automatically fall back to a common Kaggle-available transformer (`distilbert-base-uncased`) and run inference end-to-end. I also make the score decoding consistent and safe by handling both multi-class logits and single-regression-logit cases, mapping predictions into the required integer 1–6 range. Finally, I ensure the submission is aligned to `sample_submission.csv` and always writes a valid `submission.csv` with the correct columns and row count.'
- What this solution (achieved 0.71466) has done: 'The crash happens in the TF‑IDF+Ridge fallback: your scikit‑learn Ridge defaults to a solver that calls `scipy.sparse.linalg.cg(tol=...)`, but the installed SciPy version’s `cg` signature no longer accepts `tol`, causing the `TypeError` and preventing `df_sub` from being created. I fix this by forcing a Ridge solver that doesn’t use `cg` (e.g., `lsqr`), which preserves the same modeling approach while restoring compatibility. I also keep the transformer path behavior unchanged, but make the submission creation robust so `df_sub` is always defined before writing `submission.csv`. These changes are score-neutral for transformer mode and should yield a reasonable baseline score in fallback mode rather than crashing.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import torch

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score



## === cell 1
MODEL_NAME = "/kaggle/input/f-tfm-small/funnel-small-ft_ver4"
MAX_LENGTH = 3072
BATCH_SIZE = 2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 2
df_train = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)
df_test = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
sample_submission = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)

assert {"essay_id", "full_text"}.issubset(df_test.columns)
assert {"essay_id", "full_text", "score"}.issubset(df_train.columns)
assert list(sample_submission.columns) == ["essay_id", "score"]




## === cell 3
def resolve_model_dir(path: str) -> str:
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
        return path
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            if "config.json" in files:
                return root
    return path


MODEL_DIR = resolve_model_dir(MODEL_NAME)

use_transformer = os.path.isdir(MODEL_DIR) and os.path.exists(
    os.path.join(MODEL_DIR, "config.json")
)

tokenizer = None
model = None
num_labels = None

if use_transformer:
    try:
        LOAD_ID = MODEL_DIR
        local_only = True
        tokenizer = AutoTokenizer.from_pretrained(LOAD_ID, local_files_only=local_only)
        model = AutoModelForSequenceClassification.from_pretrained(
            LOAD_ID, local_files_only=local_only
        )
        model.to(device)
        model.eval()
        num_labels = getattr(model.config, "num_labels", None)
        print(
            "Loaded model:",
            LOAD_ID,
            "| num_labels:",
            num_labels,
            "| MAX_LENGTH:",
            MAX_LENGTH,
        )
    except Exception as e:
        print(
            "Transformer load failed, falling back to TF-IDF + Ridge. Error:", repr(e)
        )
        use_transformer = False
else:
    print("Local fine-tuned model not found; falling back to TF-IDF + Ridge.")



## === cell 4
if use_transformer:
    texts = df_test["full_text"].astype(str).tolist()

    all_logits = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start : start + BATCH_SIZE]
            enc = tokenizer(
                batch_texts,
                max_length=MAX_LENGTH,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            logits = out.logits.detach().float().cpu().numpy()
            all_logits.append(logits)

    logits = np.concatenate(all_logits, axis=0)
    print("Logits shape:", logits.shape)
else:
    logits = None




## === cell 5
def apply_thresholds(x: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """
    Map continuous predictions to ordinal classes 1..6 using 5 thresholds.
    """
    x = x.astype(np.float64)
    thr = np.asarray(thr, dtype=np.float64)
    return (np.digitize(x, thr, right=False) + 1).astype(np.int32)


def fit_thresholds_bruteforce(oof_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Minimal, metric-aligned post-processing for QWK:
    choose thresholds from data quantiles to maximize QWK on OOF predictions.
    This does not change model training; it only calibrates rounding to the metric.
    """
    oof_pred = oof_pred.astype(np.float64)
    y_true = y_true.astype(np.int32)

    qs = np.linspace(0.05, 0.95, 19)
    cand = np.unique(np.quantile(oof_pred, qs))
    cand = cand[np.isfinite(cand)]
    if cand.size < 10:
        cand = np.linspace(np.min(oof_pred), np.max(oof_pred), 25)

    base = []
    for k in range(1, 6):
        left = (
            np.mean(oof_pred[y_true <= k]) if np.any(y_true <= k) else np.min(oof_pred)
        )
        right = (
            np.mean(oof_pred[y_true > k]) if np.any(y_true > k) else np.max(oof_pred)
        )
        base.append((left + right) / 2.0)
    thr = np.sort(np.array(base, dtype=np.float64))

    def score_for(thresholds):
        pred_cls = apply_thresholds(oof_pred, thresholds)
        return cohen_kappa_score(y_true, pred_cls, weights="quadratic")

    best_thr = thr.copy()
    best = score_for(best_thr)

    for _ in range(2):  # two passes are enough for small gains without heavy compute
        for i in range(5):
            lo = best_thr[i - 1] + 1e-6 if i > 0 else -np.inf
            hi = best_thr[i + 1] - 1e-6 if i < 4 else np.inf
            feasible = cand[(cand > lo) & (cand < hi)]
            if feasible.size == 0:
                continue
            local_best = best
            local_thr = best_thr[i]
            for v in feasible:
                trial = best_thr.copy()
                trial[i] = v
                s = score_for(trial)
                if s > local_best:
                    local_best = s
                    local_thr = v
            best_thr[i] = local_thr
            best = local_best

    return np.sort(best_thr)


if use_transformer:
    if logits.ndim == 2 and logits.shape[1] >= 2:
        pred_class = logits.argmax(axis=1).astype(np.int32)

        if logits.shape[1] == 6:
            pred_score = pred_class + 1
        else:
            if logits.shape[1] == 2:
                pred_score = (pred_class + 3).astype(np.int32)  # 0->3, 1->4
            else:
                pred_score = (
                    np.clip(
                        np.rint((pred_class / (logits.shape[1] - 1)) * 5), 0, 5
                    ).astype(np.int32)
                    + 1
                )
    else:
        raw = logits.reshape(-1)
        pred_score = np.clip(np.rint(raw), 0, 5).astype(np.int32) + 1

    pred_score = np.clip(pred_score, 1, 6).astype(np.int32)

else:
    X_train_text = df_train["full_text"].astype(str).values
    y_train = df_train["score"].astype(np.int32).values
    X_test_text = df_test["full_text"].astype(str).values

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        sublinear_tf=True,
        stop_words="english",
    )
    Xtr = vectorizer.fit_transform(X_train_text)
    Xte = vectorizer.transform(X_test_text)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(Xtr.shape[0], dtype=np.float64)

    for tr_idx, va_idx in skf.split(Xtr, y_train):
        clf = RidgeClassifier(alpha=1.0, random_state=SEED, solver="lsqr")
        clf.fit(Xtr[tr_idx], y_train[tr_idx])
        oof[va_idx] = clf.decision_function(Xtr[va_idx]).astype(np.float64)

    thresholds = fit_thresholds_bruteforce(oof, y_train)

    clf_final = RidgeClassifier(alpha=1.0, random_state=SEED, solver="lsqr")
    clf_final.fit(Xtr, y_train)
    test_margin = clf_final.decision_function(Xte).astype(np.float64)

    pred_score = apply_thresholds(test_margin, thresholds)
    pred_score = np.clip(pred_score, 1, 6).astype(np.int32)

df_sub = pd.DataFrame({"essay_id": df_test["essay_id"].values, "score": pred_score})
df_sub["score"] = df_sub["score"].astype("int32")

df_sub = sample_submission[["essay_id"]].merge(df_sub, on="essay_id", how="left")
if df_sub["score"].isna().any():
    df_sub["score"] = df_sub["score"].fillna(3).astype("int32")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3529391374.py in <cell line: 0>()
    117         clf.fit(Xtr[tr_idx], y_train[tr_idx])
    118         # decision_function gives a continuous margin we can threshold into 1..6
--> 119         oof[va_idx] = clf.decision_function(Xtr[va_idx]).astype(np.float64)
    120 
    121     thresholds = fit_thresholds_bruteforce(oof, y_train)

ValueError: shape mismatch: value array of shape (3116,6) could not be broadcast to indexing result of shape (3116,)

## === cell 6
df_sub.to_csv("submission.csv", index=False)

assert os.path.exists("submission.csv")
assert list(df_sub.columns) == ["essay_id", "score"]
assert df_sub.shape[0] == sample_submission.shape[0]
assert df_sub["score"].between(1, 6).all()

print(df_sub.head())
print("Saved submission.csv with shape:", df_sub.shape)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2816395620.py in <cell line: 0>()
----> 1 df_sub.to_csv("submission.csv", index=False)
      2 
      3 assert os.path.exists("submission.csv")
      4 assert list(df_sub.columns) == ["essay_id", "score"]
      5 assert df_sub.shape[0] == sample_submission.shape[0]

NameError: name 'df_sub' is not defined
