# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import sys
import importlib

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf") or m.startswith("google._upb"):
        del sys.modules[m]
importlib.invalidate_caches()
sys.modules["google._upb"] = None

try:
    import google.protobuf.message_factory as _mf  # noqa: E402

    if hasattr(_mf, "MessageFactory") and not hasattr(
        _mf.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(descriptor)
            return _mf.GetMessageClass(descriptor)

        _mf.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.manual_seed(42)
np.random.seed(42)
try:
    torch.use_deterministic_algorithms(False)
except Exception:
    pass

TEST_DATA_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
if not os.path.exists(TEST_DATA_PATH):
    TEST_DATA_PATH = "/kaggle/input/test.csv"

TRAIN_DATA_PATH = (
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)
if not os.path.exists(TRAIN_DATA_PATH):
    TRAIN_DATA_PATH = "/kaggle/input/train.csv"

MAX_LENGTH = 1024
EVAL_BATCH_SIZE = 8 if torch.cuda.is_available() else 2


def _is_hf_model_dir(p: str) -> bool:
    if not os.path.isdir(p):
        return False
    has_config = os.path.exists(os.path.join(p, "config.json"))
    has_tok = os.path.exists(os.path.join(p, "tokenizer.json")) or os.path.exists(
        os.path.join(p, "tokenizer_config.json")
    )
    has_weights = (
        os.path.exists(os.path.join(p, "pytorch_model.bin"))
        or os.path.exists(os.path.join(p, "model.safetensors"))
        or os.path.exists(os.path.join(p, "pytorch_model.bin.index.json"))
        or os.path.exists(os.path.join(p, "model.safetensors.index.json"))
    )
    return has_config and has_weights and has_tok


def find_best_local_model():
    explicit = [
        "/kaggle/input/deberta-v3-large-aes",
        "/kaggle/input/aes-deberta-v3-large",
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/deberta-v3-large-aes",
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/model",
        "/kaggle/input/model",
        "/kaggle/input/deberta-v3-large",
        "/kaggle/input/deberta",
    ]
    for p in explicit:
        if _is_hf_model_dir(p):
            return p
    return None


MODEL_PATH = find_best_local_model() or "microsoft/deberta-v3-large"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df_test = pd.read_csv(TEST_DATA_PATH)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
except (AttributeError, ImportError) as e:
    if "GetPrototype" in str(e) or "google._upb" in str(e):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    else:
        raise


def _tokenize_texts(texts, tokenizer, max_length, batch_size=256):
    input_ids = []
    attention_mask = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i : i + batch_size],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids.append(enc["input_ids"])
        attention_mask.append(enc["attention_mask"])
    return {
        "input_ids": torch.cat(input_ids, dim=0),
        "attention_mask": torch.cat(attention_mask, dim=0),
    }


class TokenizedEssayDataset(Dataset):
    def __init__(self, encodings: dict):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self):
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def _stack_collate(features):
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features], dim=0),
        "attention_mask": torch.stack([f["attention_mask"] for f in features], dim=0),
    }


test_texts = df_test["full_text"].astype(str).tolist()
test_enc = _tokenize_texts(test_texts, tokenizer, MAX_LENGTH, batch_size=256)
ds = TokenizedEssayDataset(test_enc)

_num_workers = min(2, (os.cpu_count() or 2))
dl = DataLoader(
    ds,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    collate_fn=_stack_collate,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=2 if _num_workers > 0 else None,
)

print("Loaded test:", df_test.shape, "Model:", MODEL_PATH, "Device:", device)




## === cell 1
def load_model_safely():
    try:
        return AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    except (AttributeError, ImportError) as e:
        if "GetPrototype" in str(e) or "google._upb" in str(e):
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

            for m in list(sys.modules.keys()):
                if m.startswith("google.protobuf") or m.startswith("google._upb"):
                    del sys.modules[m]
            importlib.invalidate_caches()
            sys.modules["google._upb"] = None

            try:
                import google.protobuf.message_factory as _mf  # noqa: F811

                if hasattr(_mf, "MessageFactory") and not hasattr(
                    _mf.MessageFactory, "GetPrototype"
                ):

                    def _GetPrototype(self, descriptor):
                        if hasattr(self, "GetMessageClass"):
                            return self.GetMessageClass(descriptor)
                        return _mf.GetMessageClass(descriptor)

                    _mf.MessageFactory.GetPrototype = _GetPrototype
            except Exception:
                pass

            return AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        raise


model = load_model_safely()
model.to(device)
model.eval()

n_test = len(ds)
with torch.inference_mode():
    sample_batch = next(iter(dl))
    if torch.cuda.is_available():
        sample_batch = {
            k: v.to(device, non_blocking=True) for k, v in sample_batch.items()
        }
    else:
        sample_batch = {k: v.to(device) for k, v in sample_batch.items()}
    sample_out = model(**sample_batch).logits
    num_labels = int(sample_out.shape[1])

logits = np.empty((n_test, num_labels), dtype=np.float32)
write_pos = 0
with torch.inference_mode():
    for batch in dl:
        bs = int(batch["input_ids"].shape[0])
        if torch.cuda.is_available():
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        else:
            batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch).logits.detach().cpu().numpy()
        logits[write_pos : write_pos + bs] = out
        write_pos += bs

print("Logits shape:", logits.shape, "num_labels:", num_labels)




## === cell 2
def _qwk(y_true, y_pred, min_rating=1, max_rating=6):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_true = np.clip(y_true, min_rating, max_rating)
    y_pred = np.clip(y_pred, min_rating, max_rating)
    n = max_rating - min_rating + 1

    O = np.zeros((n, n), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        O[a - min_rating, b - min_rating] += 1.0

    act_hist = np.bincount(y_true - min_rating, minlength=n).astype(np.float64)
    pred_hist = np.bincount(y_pred - min_rating, minlength=n).astype(np.float64)

    E = np.outer(act_hist, pred_hist)
    E = E / E.sum() * O.sum()

    W = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            W[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)

    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    return 1.0 - (W * O).sum() / denom


def _logits_to_continuous_score(logits_arr: np.ndarray) -> np.ndarray:
    if logits_arr.ndim != 2:
        raise ValueError("Expected 2D logits")
    if logits_arr.shape[1] == 1:
        return logits_arr[:, 0].astype(np.float64)
    x = logits_arr.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    p = np.exp(x)
    p = p / p.sum(axis=1, keepdims=True)
    classes = np.arange(1, x.shape[1] + 1, dtype=np.float64)[None, :]
    return (p * classes).sum(axis=1)


def _apply_cutpoints(s, cuts):
    s = np.asarray(s, dtype=np.float64)
    c1, c2, c3, c4, c5 = cuts
    out = np.ones_like(s, dtype=int)
    out[s > c1] = 2
    out[s > c2] = 3
    out[s > c3] = 4
    out[s > c4] = 5
    out[s > c5] = 6
    return out


def _fit_cutpoints(s_fit, y_fit, s_eval, y_eval):
    init = np.quantile(s_fit, [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]).astype(np.float64)

    best_cuts = init.copy()
    best = _qwk(y_eval, _apply_cutpoints(s_eval, best_cuts))

    p5, p95 = np.percentile(s_fit, [5, 95])
    span = p95 - p5
    step = max(span / 30.0, 1e-3)

    for _ in range(10):
        improved = False
        for i in range(5):
            base = best_cuts[i]
            for delta in (-step, -step / 2, 0.0, step / 2, step):
                cand = best_cuts.copy()
                cand[i] = base + delta
                cand = np.sort(cand)
                score = _qwk(y_eval, _apply_cutpoints(s_eval, cand))
                if score > best + 1e-9:
                    best = score
                    best_cuts = cand
                    improved = True
        if not improved:
            step *= 0.5
        if step < 1e-4:
            break
    return best_cuts, best


preds_argmax = (
    logits.argmax(axis=-1) + 1
    if logits.shape[1] > 1
    else np.rint(logits[:, 0]).astype(int)
)
preds_argmax = np.clip(preds_argmax, 1, 6).astype(int)

use_calibration = os.path.exists(TRAIN_DATA_PATH)

if use_calibration:
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    y_all = df_train["score"].astype(int).values
    n = len(df_train)
    rng = np.random.default_rng(42)
    idx = np.arange(n)

    cal_idx = []
    for cls in range(1, 7):
        cls_idx = idx[y_all == cls]
        if len(cls_idx) == 0:
            continue
        k = max(3000, int(0.10 * len(cls_idx)))
        k = min(k, len(cls_idx))
        cal_idx.append(rng.choice(cls_idx, size=k, replace=False))
    cal_idx = (
        np.unique(np.concatenate(cal_idx))
        if len(cal_idx)
        else rng.choice(idx, size=min(20000, n), replace=False)
    )

    rng.shuffle(cal_idx)
    mid = len(cal_idx) // 2
    fit_idx = cal_idx[:mid]
    eval_idx = cal_idx[mid:]

    cal_texts = df_train.iloc[cal_idx]["full_text"].astype(str).tolist()
    cal_enc = _tokenize_texts(cal_texts, tokenizer, MAX_LENGTH, batch_size=256)
    ds_cal = TokenizedEssayDataset(cal_enc)

    dl_cal = DataLoader(
        ds_cal,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=_stack_collate,
        num_workers=_num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(_num_workers > 0),
        prefetch_factor=2 if _num_workers > 0 else None,
    )

    n_cal = len(ds_cal)
    cal_logits = np.empty((n_cal, num_labels), dtype=np.float32)
    write_pos = 0
    with torch.inference_mode():
        for b in dl_cal:
            bs = int(b["input_ids"].shape[0])
            if torch.cuda.is_available():
                b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            else:
                b = {k: v.to(device) for k, v in b.items()}
            o = model(**b).logits.detach().cpu().numpy()
            cal_logits[write_pos : write_pos + bs] = o
            write_pos += bs

    pos = np.arange(len(cal_idx))
    fit_pos = pos[:mid]
    eval_pos = pos[mid:]

    fit_logits = cal_logits[fit_pos]
    eval_logits = cal_logits[eval_pos]

    s_fit = _logits_to_continuous_score(fit_logits)
    y_fit = df_train.iloc[fit_idx]["score"].astype(int).values
    s_eval = _logits_to_continuous_score(eval_logits)
    y_eval = df_train.iloc[eval_idx]["score"].astype(int).values

    s_test = _logits_to_continuous_score(logits)

    cuts, qwk_eval = _fit_cutpoints(s_fit, y_fit, s_eval, y_eval)
    preds = _apply_cutpoints(s_test, cuts)
else:
    preds = preds_argmax

preds = np.clip(preds, 1, 6).astype(int)

df_sub = pd.DataFrame({"essay_id": df_test["essay_id"].values, "score": preds})
df_sub.to_csv("submission.csv", index=False)

del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

print(df_sub.head())
print("Wrote submission.csv with shape:", df_sub.shape)
print("Submission columns:", df_sub.columns.tolist())
print("Model used:", MODEL_PATH)
print("Used calibration:", bool(use_calibration))
if use_calibration:
    print("Calibration cutpoints:", cuts.tolist())
    print("Calibration QWK on held-out calibration-eval split:", float(qwk_eval))
