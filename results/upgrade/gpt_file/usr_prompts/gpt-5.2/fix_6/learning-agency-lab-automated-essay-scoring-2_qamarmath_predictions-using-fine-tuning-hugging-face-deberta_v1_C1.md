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

0.7964847912419482

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00616) has done: 'I fix the environment/runtime issues that prevent the notebook from running and writing `submission.csv`. First, I patch the common protobuf incompatibility that triggers `MessageFactory.GetPrototype` errors in Kaggle when importing `transformers/datasets`. Next, I make `from_pretrained()` correctly treat `MODEL_PATH` as a local directory (and fall back to a base DeBERTa checkpoint if the provided path doesn’t exist), while keeping the same inference-only core logic. Finally, I ensure tokenization truncates safely to `MAX_LENGTH` and that predictions are converted to valid integer scores 1–6 and saved in the required submission format.'
- What this solution (achieved 0.06847) has done: 'We fix the protobuf `MessageFactory.GetPrototype` crash by forcing a compatible pure-Python protobuf implementation *before* any `datasets/transformers` imports, and we add a safe fallback import to avoid the AttributeError in this Kaggle image. Next, we correct the scoring logic: this is an ordinal regression setup (scores 1–6), so using `argmax()+1` on a sequence classification head produces near-random labels (hence the very low QWK); instead we convert logits to a scalar score via the expected value over classes, then round+clip to 1–6 (calibration-only post-processing, no architecture change). Finally, we keep the same inference-only workflow and ensure the submission CSV is written with the required columns and row alignment.'
- What this solution (achieved 0.06847) has done: 'I fix the runtime crash happening before any model code runs by pinning protobuf to the pure-Python backend earlier and adding a robust fallback that patches the missing `MessageFactory.GetPrototype` attribute (a known incompatibility in some Kaggle images). This is a correctness/stability fix and should not change model behavior or scoring logic. I also keep your existing expected-value-from-logits post-processing (already a big improvement over argmax for ordinal labels) and ensure inference runs on GPU if available and always writes a valid `submission.csv` with the required columns and row alignment.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf crash by patching both `MessageFactory.GetPrototype` and the older `message_factory.GetPrototype` entry-point before importing `transformers/datasets` (the current patch only covered one code path, so the AttributeError can still occur). I also make the model load use `num_labels=6` if the checkpoint/config doesn’t already enforce it, which is a minimal, inference-only correctness fix that prevents mismatched heads from producing unusable logits. Finally, I keep your expected-value ordinal post-processing intact, but add a small safety branch to handle the case where the model outputs a single regression logit (shape `(N,1)`) so the script always produces valid 1–6 integer scores and writes `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by applying a more robust, version-agnostic monkey patch that targets the actual `MessageFactory` class used at runtime (including `google.protobuf.message_factory.MessageFactory`). This unblocks `transformers`/`datasets` imports so inference can run end-to-end and write `submission.csv`. I also keep your existing inference-only workflow and ordinal expected-value post-processing unchanged, only adding a small safety cast to ensure predictions are always a NumPy array of the right shape. No model/training logic is changed beyond making the environment import-stable and guaranteeing a valid submission file.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import google.protobuf.message_factory as _mf_mod

    def _ensure_getprototype_on_class(cls):
        if cls is None:
            return
        if not hasattr(cls, "GetPrototype") and hasattr(cls, "GetMessageClass"):
            try:
                setattr(cls, "GetPrototype", cls.GetMessageClass)
            except Exception:
                pass

    _ensure_getprototype_on_class(getattr(_mf_mod, "MessageFactory", None))

    from google.protobuf import message_factory as _mf_alias

    _ensure_getprototype_on_class(getattr(_mf_alias, "MessageFactory", None))

    if hasattr(_mf_alias, "MessageFactory"):
        try:
            _inst = _mf_alias.MessageFactory()
            if not hasattr(_inst, "GetPrototype") and hasattr(_inst, "GetMessageClass"):
                try:
                    _mf_alias.MessageFactory.GetPrototype = (
                        _mf_alias.MessageFactory.GetMessageClass
                    )
                except Exception:
                    pass
        except Exception:
            pass
except Exception:
    pass

import numpy as np
import pandas as pd
import gc
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

TEST_DATA_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
MAX_LENGTH = 1024
MODEL_PATH = "/kaggle/input/fine-tuning-hugging-face-deberta-v3/deberta-large-fold1/checkpoint-1700/"
EVAL_BATCH_SIZE = 1

if os.path.isdir(MODEL_PATH):
    PRETRAINED_PATH = MODEL_PATH
else:
    PRETRAINED_PATH = "microsoft/deberta-v3-large"

print("Using model path:", PRETRAINED_PATH)
print("Test path exists:", os.path.exists(TEST_DATA_PATH))

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
tokenizer = AutoTokenizer.from_pretrained(
    PRETRAINED_PATH, local_files_only=os.path.isdir(PRETRAINED_PATH)
)


def tokenize(sample):
    return tokenizer(sample["full_text"], max_length=MAX_LENGTH, truncation=True)


df_test = pd.read_csv(TEST_DATA_PATH)

ds = Dataset.from_pandas(df_test, preserve_index=False)
ds = ds.map(tokenize, batched=False)
ds = ds.remove_columns([c for c in ["essay_id", "full_text"] if c in ds.column_names])

print(ds)




## === cell 2
class DataCollator:
    def __call__(self, features):
        model_inputs = [
            {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]}
            for f in features
        ]
        batch = tokenizer.pad(
            model_inputs,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            pad_to_multiple_of=16,
        )
        return batch




## === cell 3
model = AutoModelForSequenceClassification.from_pretrained(
    PRETRAINED_PATH,
    local_files_only=os.path.isdir(PRETRAINED_PATH),
    num_labels=6,
    ignore_mismatched_sizes=True,
)

collator = DataCollator()

args = TrainingArguments(
    output_dir=".",
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    report_to="none",
    dataloader_num_workers=0,
)

trainer = Trainer(model=model, args=args, data_collator=collator, tokenizer=tokenizer)

pred_out = trainer.predict(ds)
predictions = (
    pred_out.predictions
)  # expected shape: (N, num_labels) or (N, 1) for regression

del model, trainer
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

predictions = np.asarray(predictions)
print("Predictions shape:", predictions.shape)



## === cell 4
pred_arr = np.asarray(predictions)

if pred_arr.ndim == 1:
    reg = pred_arr
    preds = np.rint(reg).astype(np.int64)
elif pred_arr.ndim == 2 and pred_arr.shape[1] == 1:
    reg = pred_arr[:, 0]
    preds = np.rint(reg).astype(np.int64)
else:
    logits = torch.tensor(pred_arr)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    num_labels = probs.shape[1]
    class_values = np.arange(1, num_labels + 1, dtype=np.float32)  # 1..num_labels
    expected_score = (probs * class_values[None, :]).sum(axis=1)
    preds = np.rint(expected_score).astype(np.int64)

preds = np.clip(preds, 1, 6)

df_sub = df_test[["essay_id"]].copy()
df_sub["score"] = preds

df_sub.to_csv("submission.csv", index=False)
print(df_sub.head())
print("Wrote submission.csv with shape:", df_sub.shape)
print("Score value counts:\n", df_sub["score"].value_counts().sort_index())
