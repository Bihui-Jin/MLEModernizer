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

0.8021729294158836

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.01486) has done: 'I fix the protobuf-related import crash by forcing the pure-Python protobuf runtime before importing `transformers/datasets`, which avoids the `MessageFactory.GetPrototype` error in this environment. Next, I prevent the fallback DistilBERT path from crashing by capping `MAX_LENGTH` to the tokenizer/model’s maximum supported length (512 for DistilBERT), eliminating the tensor size mismatch. I also make inference deterministic/stable and memory-safe by enabling `torch.inference_mode()` and `fp16` only when on CUDA (score-neutral). Finally, I ensure `preds` is always defined and that a valid `submission.csv` is written with the required `essay_id,score` columns.'
- What this solution (achieved -0.01867) has done: 'I fix the protobuf import crash by forcing the pure-Python protobuf runtime *and* ensuring it is applied early enough, plus applying a small compatibility patch for the `MessageFactory.GetPrototype` attribute that `transformers/datasets` can hit in some Kaggle images. I also make the fallback model path use the correct head (6 labels) so it doesn’t output a 2-class classification head by default, which is the main reason your QWK score is extremely low. Finally, I keep your core inference/ensembling logic intact while ensuring the submission is generated deterministically and with valid 1–6 integer scores.'
- What this solution (achieved -0.03141) has done: 'I fix the protobuf compatibility patch so it doesn’t crash on import in this Kaggle image by applying the `MessageFactory.GetPrototype` shim at the *instance* level as well (some builds expose it only per-instance), and by doing it before importing `datasets/transformers`. Then I keep your model/inference logic intact, but correct the post-processing for the 6-class classification head: instead of `argmax` over logits, convert logits to an expected score via softmax-weighted mean (a minimal calibration-aligned change that typically improves QWK for ordinal labels). Finally, I keep the same submission merge/alignment logic and ensure `submission.csv` is always written with valid 1–6 integer scores.'
- What this solution (achieved 0.0) has done: 'I fix the early protobuf shim that currently triggers the `MessageFactory.GetPrototype` AttributeError by removing the self-referential call and only defining `GetPrototype` when it’s actually missing. Then I make the model-loading logic robust in fully-offline Kaggle by selecting an available local HF model directory if your intended `/kaggle/input/f-tfm-small` artifact isn’t attached, and (if none exists) falling back to a deterministic, score-safe baseline using the train-score distribution so we always produce a valid `submission.csv`. Finally, I ensure `tokenize` and `preds` are always defined so later cells can’t crash, and keep the same post-processing to integer scores 1–6 and the required submission format.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from google.protobuf.message_factory import MessageFactory  # type: ignore

    if not hasattr(MessageFactory, "GetPrototype") and hasattr(
        MessageFactory, "GetMessageClass"
    ):

        def _shim_getprototype(self, descriptor):  # type: ignore
            return self.GetMessageClass(descriptor)  # type: ignore[attr-defined]

        MessageFactory.GetPrototype = _shim_getprototype  # type: ignore[attr-defined]

    try:
        mf = MessageFactory()  # type: ignore[call-arg]
        if not hasattr(mf, "GetPrototype") and hasattr(mf, "GetMessageClass"):
            mf.GetPrototype = MessageFactory.GetPrototype.__get__(mf, MessageFactory)  # type: ignore[attr-defined]
    except Exception:
        pass

except Exception:
    pass

import random
import numpy as np
import pandas as pd

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
MODEL_ROOT = "/kaggle/input/f-tfm-small"
MAX_LENGTH = 3072
BATCH_SIZE = 2
N_FOLDS = 5

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 2
def _resolve_competition_dir() -> str:
    candidates = [
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2",
        "/kaggle/data/learning-agency-lab-automated-essay-scoring-2",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for base in candidates:
        if (
            os.path.isdir(base)
            and os.path.exists(os.path.join(base, "train.csv"))
            and os.path.exists(os.path.join(base, "test.csv"))
        ):
            return base

    for base in ["/kaggle/input", "/kaggle/data"]:
        if os.path.isdir(base):
            for name in os.listdir(base):
                p = os.path.join(base, name)
                if (
                    os.path.isdir(p)
                    and os.path.exists(os.path.join(p, "train.csv"))
                    and os.path.exists(os.path.join(p, "test.csv"))
                    and os.path.exists(os.path.join(p, "sample_submission.csv"))
                ):
                    return p

    raise FileNotFoundError(
        "Could not locate competition dataset directory with train.csv and test.csv under /kaggle/input or /kaggle/data."
    )


COMP_DIR = _resolve_competition_dir()

df_train = pd.read_csv(os.path.join(COMP_DIR, "train.csv"))
df_test = pd.read_csv(os.path.join(COMP_DIR, "test.csv"))
sample_submission = pd.read_csv(os.path.join(COMP_DIR, "sample_submission.csv"))

assert {"essay_id", "full_text"}.issubset(df_test.columns)
assert {"essay_id", "full_text", "score"}.issubset(df_train.columns)

print("COMP_DIR:", COMP_DIR)
print(
    "train:", df_train.shape, "test:", df_test.shape, "sample:", sample_submission.shape
)




## === cell 3
def _resolve_model_root(model_root: str) -> str:
    if os.path.isdir(model_root):
        return model_root

    base = "/kaggle/input"
    if os.path.isdir(base):
        for name in os.listdir(base):
            p = os.path.join(base, name)
            if os.path.isdir(p) and os.path.basename(p) == os.path.basename(model_root):
                return p
    return model_root  # keep original; downstream will handle fallback


MODEL_ROOT = _resolve_model_root(MODEL_ROOT)


def _find_fold_dir(fold: int) -> str:
    """
    Prefer the originally intended folder pattern.
    If not found, fall back to scanning MODEL_ROOT for any directory containing the fold id.
    """
    direct = os.path.join(MODEL_ROOT, f"f-tfm-small_AES2_fold_{fold}")
    if os.path.isdir(direct):
        return direct

    candidates = []
    if os.path.isdir(MODEL_ROOT):
        for name in os.listdir(MODEL_ROOT):
            p = os.path.join(MODEL_ROOT, name)
            if os.path.isdir(p) and (f"fold_{fold}" in name or f"fold{fold}" in name):
                candidates.append(p)
    candidates = sorted(candidates)
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not locate model directory for fold={fold}. "
        f"Tried: {direct} and scanning under {MODEL_ROOT}. "
        f"MODEL_ROOT exists={os.path.isdir(MODEL_ROOT)}"
    )


def _looks_like_hf_model_dir(p: str) -> bool:
    if not os.path.isdir(p):
        return False
    files = set(os.listdir(p))
    if "config.json" not in files:
        return False
    has_weights = any(
        f in files
        for f in [
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.bin.index.json",
            "model.safetensors.index.json",
        ]
    )
    has_tokenizer = any(
        f in files
        for f in [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "merges.txt",
            "spiece.model",
        ]
    )
    return has_weights and has_tokenizer


def _find_any_local_model_dir() -> str | None:
    """
    Fix: in offline Kaggle, if MODEL_ROOT isn't mounted, we must still find a local model under /kaggle/input.
    This keeps the same inference approach (AutoTokenizer/AutoModel) while preventing hard crash.
    """
    base = "/kaggle/input"
    if not os.path.isdir(base):
        return None
    candidates = []
    for name in os.listdir(base):
        p = os.path.join(base, name)
        if _looks_like_hf_model_dir(p):
            candidates.append(p)
        elif os.path.isdir(p):
            try:
                for sub in os.listdir(p):
                    sp = os.path.join(p, sub)
                    if _looks_like_hf_model_dir(sp):
                        candidates.append(sp)
            except Exception:
                pass
    candidates = sorted(set(candidates))
    return candidates[0] if candidates else None


FALLBACK_MODEL = "distilbert-base-uncased"

use_fallback = False
tokenizer_dir = None
tokenizer = None
EVAL_MAX_LENGTH = None

local_model_dir = None
try:
    tokenizer_dir = _find_fold_dir(0)
    local_model_dir = tokenizer_dir
except FileNotFoundError as e:
    print("WARNING:", str(e))
    alt = _find_any_local_model_dir()
    if alt is not None:
        print(f"Falling back to locally-available model dir under /kaggle/input: {alt}")
        tokenizer_dir = alt
        local_model_dir = alt
    else:
        tokenizer_dir = None
        local_model_dir = None

if local_model_dir is not None:
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_dir, local_files_only=True, use_fast=True
    )
    tok_max = getattr(tokenizer, "model_max_length", None)
    if tok_max is None or tok_max > 100000:
        tok_max = 512
    EVAL_MAX_LENGTH = int(min(MAX_LENGTH, tok_max))

    def tokenize(batch):
        return tokenizer(
            batch["full_text"], max_length=EVAL_MAX_LENGTH, truncation=True
        )

else:
    def tokenize(batch):
        return {}


print("MODEL_ROOT:", MODEL_ROOT)
print("tokenizer_dir:", tokenizer_dir)
print("local_model_dir:", local_model_dir)
print("has_tokenizer:", tokenizer is not None)
print("MAX_LENGTH:", MAX_LENGTH, "EVAL_MAX_LENGTH:", EVAL_MAX_LENGTH)



## === cell 4
preds = None

if local_model_dir is None:
    train_mean = float(df_train["score"].mean())
    baseline_score = int(np.clip(int(np.rint(train_mean)), 1, 6))
    scores = np.full(len(df_test), baseline_score, dtype=np.int32)

    sub = pd.DataFrame({"essay_id": df_test["essay_id"].values, "score": scores})
    sub = sample_submission[["essay_id"]].merge(sub, on="essay_id", how="left")
    assert (
        sub["score"].isna().sum() == 0
    ), "Missing predictions for some essay_id after merge."
    sub["score"] = sub["score"].astype(np.int32)
    sub.to_csv("submission.csv", index=False)

    print(sub.head())
    print(
        f"Wrote submission.csv with shape={sub.shape} to {os.path.abspath('submission.csv')}"
    )
    print("Baseline score used:", baseline_score)
    print("Score distribution:\n", sub["score"].value_counts().sort_index())
else:
    test_ds = Dataset.from_pandas(
        df_test[["essay_id", "full_text"]], preserve_index=False
    )
    test_ds = test_ds.map(tokenize, batched=True, desc="Tokenizing test")
    test_ds = test_ds.remove_columns(["essay_id", "full_text"])

    use_cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir="./_tmp_pred",
        per_device_eval_batch_size=BATCH_SIZE,
        report_to="none",
        dataloader_num_workers=0,
        fp16=use_cuda,  # inference only
    )

    predictions = []
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    with torch.inference_mode():
        fold_dirs = []
        for fold in range(N_FOLDS):
            try:
                fold_dirs.append(_find_fold_dir(fold))
            except FileNotFoundError:
                fold_dirs = []
                break

        if fold_dirs:
            for fold, model_dir in enumerate(fold_dirs):
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir, local_files_only=True
                )
                trainer = Trainer(
                    model=model,
                    args=args,
                    data_collator=data_collator,
                    tokenizer=tokenizer,
                )
                fold_logits = trainer.predict(test_ds).predictions
                predictions.append(fold_logits)
                print(f"Fold {fold}: logits shape={fold_logits.shape} from {model_dir}")
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                local_model_dir, local_files_only=True
            )
            trainer = Trainer(
                model=model, args=args, data_collator=data_collator, tokenizer=tokenizer
            )
            fold_logits = trainer.predict(test_ds).predictions
            predictions.append(fold_logits)
            print(
                f"Single local model: logits shape={fold_logits.shape} from {local_model_dir}"
            )

    preds = np.mean(np.stack(predictions, axis=0), axis=0)
    print("Ensembled preds shape:", preds.shape)

    if preds.ndim == 2 and preds.shape[1] == 1:
        raw = preds[:, 0]
        scores = (np.clip(raw, 0, 5).round(0) + 1).astype(np.int32)
    else:
        if preds.ndim != 2:
            cls = np.argmax(preds, axis=1)
            scores = (cls + 1).astype(np.int32)
            scores = np.clip(scores, 1, 6).astype(np.int32)
        elif preds.shape[1] != 6:
            cls = np.argmax(preds, axis=1)
            scores = (cls + 1).astype(np.int32)
            scores = np.clip(scores, 1, 6).astype(np.int32)
        else:
            x = preds - preds.max(axis=1, keepdims=True)
            p = np.exp(x)
            p = p / p.sum(axis=1, keepdims=True)

            classes = np.arange(1, 7, dtype=np.float32)[None, :]
            exp_score = (p * classes).sum(axis=1)

            scores = np.rint(exp_score).astype(np.int32)
            scores = np.clip(scores, 1, 6).astype(np.int32)

    sub = pd.DataFrame({"essay_id": df_test["essay_id"].values, "score": scores})
    sub = sample_submission[["essay_id"]].merge(sub, on="essay_id", how="left")
    assert (
        sub["score"].isna().sum() == 0
    ), "Missing predictions for some essay_id after merge."
    sub["score"] = sub["score"].astype(np.int32)

    sub.to_csv("submission.csv", index=False)
    print(sub.head())
    print(
        f"Wrote submission.csv with shape={sub.shape} to {os.path.abspath('submission.csv')}"
    )
    print("Score distribution:\n", sub["score"].value_counts().sort_index())
