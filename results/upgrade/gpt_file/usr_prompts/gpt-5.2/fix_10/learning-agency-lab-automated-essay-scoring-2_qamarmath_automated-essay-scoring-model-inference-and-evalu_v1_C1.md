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

# 5. Target score

0.7577191694774467

# 6. Current score

-0.00149

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.15209) has done: 'I fix the two root causes preventing end-to-end execution: (1) the `datasets` mapping crash caused by an incompatible protobuf runtime, and (2) the model path failing HF validation because the referenced Kaggle input model directory doesn’t exist in your provided file tree. To keep the core inference logic (tokenize → DeBERTa sequence classification → argmax+1) intact, I load a compatible public DeBERTa-v3 model from the local Transformers cache when available, otherwise download it (no training added). I also avoid `datasets` entirely (use a plain PyTorch `Dataset` + DataLoader) to sidestep the protobuf issue while keeping identical tokenization and padding semantics. Finally, I write `submission.csv` with the required `essay_id,score` columns and correct row alignment.'
- What this solution (achieved 0.00791) has done: 'I fix the protobuf-related crash (`MessageFactory.GetPrototype`) that happens when importing/using `transformers` in this Kaggle image by forcing a compatible pure-Python protobuf implementation before `transformers` is imported. This is a minimal environment fix and keeps your core logic identical (tokenize → DeBERTa sequence classification → argmax+1 → write submission). I also add safe fallbacks for the test file path and ensure the submission is always written with the required `essay_id,score` columns. No training, architecture, or inference semantics are changed—this should both run end-to-end and substantially improve score vs. the broken/degenerate run.'
- What this solution (achieved 0.24088) has done: 'We fix the protobuf runtime error by forcing the pure-Python protobuf implementation *and* ensuring it takes effect before any `protobuf/transformers` internals load, plus set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` which is the most compatible mode for this Kaggle image. This is an execution-stability fix that preserves your exact inference semantics (same model, same tokenization, same argmax+1 postprocess) and should restore the intended score (your current 0.00791 is consistent with a broken runtime). I also add a small safety fallback to CPU if CUDA is present but unstable, without changing outputs otherwise, and keep the submission format unchanged. No training, architecture, or thresholding changes are introduced.'
- What this solution (achieved 0.06337) has done: 'The crash is coming from an incompatible compiled `protobuf` backend being used at runtime despite your env vars; `transformers` ends up importing `google.protobuf` before the pure-Python implementation is enforced. I fix this by forcing the env vars *and* reloading the protobuf modules before importing anything from `transformers`, which is the minimal stability change that keeps your inference logic identical. I also add a small fallback to disable fast tokenizers only if the same protobuf error persists (score-neutral but makes the notebook robust). The rest (model choice, max_length, argmax+1 postprocess, submission format) stays unchanged.'
- What this solution (achieved 0.34869) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by enforcing the pure-Python protobuf implementation *before* any `google.protobuf` / `transformers` imports and by hard-blocking the compiled `google._upb` module if it was loaded. This is an execution-stability fix that keeps your core inference logic unchanged (same tokenizer/model family, same argmax+1 postprocess) and should restore a non-degenerate score. I also make the exception handling catch both `AttributeError` and `ImportError` for this specific protobuf failure mode so the fallback actually triggers. Finally, I keep paths unchanged and ensure `submission.csv` is always written with the required `essay_id,score` columns.'
- What this solution (achieved 0.11274) has done: 'We fix the remaining protobuf `MessageFactory.GetPrototype` crash by enforcing the pure-Python protobuf implementation even more strictly before `transformers` ever imports it, and by patching the missing `GetPrototype` method to call `GetMessageClass` when needed (a minimal compatibility shim). This keeps your core logic identical (same DeBERTa model family, same tokenization, same argmax+1 postprocess) while allowing the model to load and run end-to-end. To move the score toward the target, we also load the competition-relevant fine-tuned checkpoint if it exists locally in `/kaggle/input` (otherwise fall back to your current base model), which is a minimal change in model weights without changing architecture or inference semantics. Finally, we ensure `submission.csv` is always written with correct columns and row alignment.'
- What this solution (achieved 0.0) has done: 'Your current score (0.11274) is far below the target (0.7577), so we should improve performance while keeping the same inference core (tokenize → DeBERTa sequence classification → argmax+1). The biggest likely issue is that you are loading a base DeBERTa model (`microsoft/deberta-v3-large`) that is not fine-tuned for this competition, which yields near-random class predictions. With minimal change, we prefer a locally available competition fine-tuned DeBERTa checkpoint (common in Kaggle datasets) by scanning `/kaggle/input` recursively for a folder that contains model weights/config, and only fall back to the base model if none is found. Additionally, we increase eval batch size to safely speed up inference (no semantic change) and ensure `problem_type="single_label_classification"` so logits shape/behavior is consistent across checkpoints.'
- What this solution (achieved 0.0) has done: 'Your current score (0.0) is far below the target, so the most likely issue is that you’re producing a valid CSV but with systematically bad predictions from a non–competition-tuned checkpoint and/or a label mapping mismatch. To move the score upward with minimal logic change, I (1) load the model without forcing `num_labels=6` when a fine-tuned head exists (this avoids silently reshaping/reinitializing heads and preserves the checkpoint’s intended label space), and (2) add a tiny, metric-aligned postprocess: optimize 5 cutpoints on a small stratified validation split to map a single continuous score to 1–6 (this keeps the same DeBERTa forward pass and just calibrates outputs for QWK). If no training data is available for calibration, it fall back to your original `argmax+1` behavior. The script still runs end-to-end and writes `submission.csv` with `essay_id,score`.'
- What this solution (achieved -0.00149) has done: 'You’re currently far below the target (0.0 vs 0.7577, higher-is-better), so we should improve predictions while keeping your core pipeline (tokenize → DeBERTa seq-cls → logits → postprocess → CSV) intact. The biggest correctness/performance issue is that your “calibration” is fitting cutpoints on the same validation data it evaluates on (in-sample), which tends to overfit and can hurt generalization/QWK on the test set. I make a minimal change to fit cutpoints on a train split and select them by QWK on a separate held-out split, without adding training or changing the model architecture. I also ensure logits-to-score uses the correct class index mapping when `num_labels==6` (scores 1–6), preserving your original behavior but avoiding off-by-one if a checkpoint uses 0–5 internally.'

# 9. Code solution

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

TEST_DATA_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
if not os.path.exists(TEST_DATA_PATH):
    TEST_DATA_PATH = "/kaggle/input/test.csv"

TRAIN_DATA_PATH = (
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)
if not os.path.exists(TRAIN_DATA_PATH):
    TRAIN_DATA_PATH = "/kaggle/input/train.csv"

MAX_LENGTH = 1024
EVAL_BATCH_SIZE = 4


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
    ]
    for p in explicit:
        if _is_hf_model_dir(p):
            return p

    best = None
    best_score = -1

    for root, dirs, files in os.walk("/kaggle/input"):
        rlow = root.lower()
        if ("deberta" not in rlow) and ("aes" not in rlow) and ("essay" not in rlow):
            continue

        if _is_hf_model_dir(root):
            score = 0
            if os.path.exists(os.path.join(root, "model.safetensors")):
                score += 3
            if os.path.exists(os.path.join(root, "pytorch_model.bin")):
                score += 2
            if "large" in rlow:
                score += 2
            if "v3" in rlow:
                score += 1
            if "finetune" in rlow or "finetuned" in rlow or "aes" in rlow:
                score += 2

            if score > best_score:
                best_score = score
                best = root

    return best


MODEL_PATH = find_best_local_model()
if MODEL_PATH is None:
    MODEL_PATH = "microsoft/deberta-v3-large"

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    device = torch.device("cpu")

df_test = pd.read_csv(TEST_DATA_PATH)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
except (AttributeError, ImportError) as e:
    if "GetPrototype" in str(e) or "google._upb" in str(e):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    else:
        raise


class EssayDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts = df["full_text"].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"full_text": self.texts[idx]}


class DataCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        texts = [f["full_text"] for f in features]
        batch = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return batch


ds = EssayDataset(df_test, tokenizer, MAX_LENGTH)
collator = DataCollator(tokenizer, MAX_LENGTH)
dl = DataLoader(ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collator)




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

all_logits = []
with torch.no_grad():
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        all_logits.append(out.logits.detach().cpu())

logits = torch.cat(all_logits, dim=0).numpy()

del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()




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


def _fit_cutpoints(s_fit, y_fit, s_eval, y_eval, seed=42):
    init = np.quantile(s_fit, [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]).astype(np.float64)

    best_cuts = init.copy()
    best = _qwk(y_eval, _apply_cutpoints(s_eval, best_cuts))

    span = np.percentile(s_fit, 95) - np.percentile(s_fit, 5)
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
        k = max(
            1500, int(0.06 * len(cls_idx))
        )  # small but a bit larger than before for stability
        k = min(k, len(cls_idx))
        cal_idx.append(rng.choice(cls_idx, size=k, replace=False))
    cal_idx = (
        np.unique(np.concatenate(cal_idx))
        if len(cal_idx)
        else rng.choice(idx, size=min(12000, n), replace=False)
    )

    rng.shuffle(cal_idx)
    mid = len(cal_idx) // 2
    fit_idx = cal_idx[:mid]
    eval_idx = cal_idx[mid:]

    def _infer_logits(df_slice: pd.DataFrame) -> np.ndarray:
        ds_part = EssayDataset(df_slice.reset_index(drop=True), tokenizer, MAX_LENGTH)
        dl_part = DataLoader(
            ds_part, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collator
        )

        m = load_model_safely()
        m.to(device)
        m.eval()
        outs = []
        with torch.no_grad():
            for b in dl_part:
                b = {k: v.to(device) for k, v in b.items()}
                o = m(**b)
                outs.append(o.logits.detach().cpu())
        out_logits = torch.cat(outs, dim=0).numpy()

        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return out_logits

    fit_logits = _infer_logits(df_train.iloc[fit_idx])
    eval_logits = _infer_logits(df_train.iloc[eval_idx])

    s_fit = _logits_to_continuous_score(fit_logits)
    y_fit = df_train.iloc[fit_idx]["score"].astype(int).values
    s_eval = _logits_to_continuous_score(eval_logits)
    y_eval = df_train.iloc[eval_idx]["score"].astype(int).values

    s_test = _logits_to_continuous_score(logits)

    cuts, qwk_eval = _fit_cutpoints(s_fit, y_fit, s_eval, y_eval, seed=42)
    preds = _apply_cutpoints(s_test, cuts)
else:
    preds = preds_argmax

preds = np.clip(preds, 1, 6).astype(int)

df_sub = pd.DataFrame({"essay_id": df_test["essay_id"].values, "score": preds})
df_sub.to_csv("submission.csv", index=False)

print(df_sub.head())
print("Wrote submission.csv with shape:", df_sub.shape)
print("Submission columns:", df_sub.columns.tolist())
print("Model used:", MODEL_PATH)
print("Used calibration:", bool(use_calibration))
if use_calibration:
    print("Calibration cutpoints:", cuts.tolist())
    print("Calibration QWK on held-out calibration-eval split:", float(qwk_eval))
