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

3.13

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

0.7846490997226996

# 6. Current score

-0.11952

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.11952) has done: 'I fix the Transformers import/runtime crash (`MessageFactory ... GetPrototype`) by setting a compatible protobuf implementation before importing `transformers`, which is a common issue on newer Python/protobuf builds. Then I fix the BERT max sequence length mismatch by capping `MAX_LENGTH` to the loaded model’s `config.max_position_embeddings` (BERT is 512), which removes the 1024 vs 512 runtime error without changing the model itself. Finally, I ensure prediction runs on the correct device and that the submission is written as a valid `submission.csv` with `essay_id,score` aligned to the test order. These changes are execution/stability fixes; they don’t alter the core modeling approach (still argmax over 6-class classifier), and should yield a valid submission file.'
- What this solution (achieved -0.11952) has done: 'I fix the remaining `MessageFactory ... GetPrototype` crash by forcing the pure-Python protobuf runtime and (critically) importing `google.protobuf` before `transformers`, which prevents the incompatible C++ runtime from being used in this Kaggle/Python 3.13 setup. I also make the checkpoint resolution deterministic by preferring the explicitly provided `MODEL_DIR/CHECKPOINT_SUBDIR` (as intended) and only falling back to a search if it’s missing. Finally, I keep the core inference logic the same (argmax over 6 logits + 1) while ensuring the dataset includes all required model inputs and that `submission.csv` is always written in the correct format and test order.'
- What this solution (achieved -0.11952) has done: 'I fix the remaining `MessageFactory ... GetPrototype` crash by forcing the pure-Python protobuf runtime *before Python imports protobuf/transformers* and by explicitly setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`, which is the most reliable workaround on Python 3.13. I also set `TRANSFORMERS_NO_TF/NO_FLAX` to avoid any optional backend protobuf imports that can trigger the same issue. These are execution-stability fixes and do not change the model, inference logic (still argmax over 6 logits + 1), or submission formatting. The rest of the pipeline (checkpoint resolution, MAX_LENGTH capping, predict → submission.csv) is kept the same.'
- What this solution (achieved -0.11952) has done: 'I fix the protobuf/transformers crash by forcing the pure-Python protobuf implementation even earlier and (critically) ensuring the C++ `message_factory` can’t be used, which is what triggers `MessageFactory.GetPrototype` failures on Python 3.13 in this environment. Then I keep your checkpoint loading, tokenization, and argmax-over-6-classes inference logic unchanged, but make the Trainer instantiation more robust by providing a minimal tokenizer hook (score-neutral) and ensuring model is on the correct device. Finally, I keep the submission writing exactly as required (`submission.csv` with `essay_id,score` aligned to test order).'
- What this solution (achieved -0.11952) has done: 'I fix the protobuf/transformers crash by forcing the pure-Python protobuf runtime before any protobuf-dependent imports and by patching the missing `MessageFactory.GetPrototype` method (some libraries still call it even when only `GetMessageClass` exists). This is an execution-only fix that keeps your core model/inference logic identical (same checkpoint loading, tokenization, Trainer predict, argmax+1). I also keep the MAX_LENGTH capping and submission formatting, only adjusting cell numbering to start from 1 so the notebook/script runs cleanly end-to-end and writes a valid `submission.csv`. No score-tuning changes are introduced; the main goal is to unblock inference so you can reach the expected model-based score range instead of crashing/producing an invalid run.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import google.protobuf  # noqa: F401
from google.protobuf import message_factory as _message_factory  # noqa: E402

if not hasattr(_message_factory.MessageFactory, "GetPrototype") and hasattr(
    _message_factory.MessageFactory, "GetMessageClass"
):

    def _GetPrototype(self, descriptor):
        return self.GetMessageClass(descriptor)

    _message_factory.MessageFactory.GetPrototype = _GetPrototype  # type: ignore[attr-defined]

try:
    _message_factory._DEFAULT = _message_factory.MessageFactory()
except Exception:
    pass

import torch  # noqa: E402

print(f"PyTorch version: {torch.__version__}")

import transformers  # noqa: E402

print(f"Hugging Face Transformers version: {transformers.__version__}")

import datasets  # noqa: E402

print(f"Hugging Face Datasets version: {datasets.__version__}")



## === cell 1
import json
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_NAME = "bert-base-cased"
INPUT_DIR = "/kaggle/input/"
MODEL_DIR = "/kaggle/input/aes2-persuade-bertbase-5ep-results/"
CHECKPOINT_SUBDIR = "checkpoint-2500"

MAX_LENGTH = 1024
RANDOM_SEED = 42
EVAL_USE_PRETRAIN = 1
SUBMISSION = 1

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device:", device)
if torch.cuda.is_available():
    print("cuda current_device:", torch.cuda.current_device())
else:
    print("cuda not available; running on CPU")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)




## === cell 2
def _is_hf_checkpoint_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    if (p / "config.json").exists():
        return True
    return False


def _find_checkpoint_dir() -> Path | None:
    expected = Path(MODEL_DIR) / CHECKPOINT_SUBDIR
    if _is_hf_checkpoint_dir(expected):
        return expected

    model_dir = Path(MODEL_DIR)
    if _is_hf_checkpoint_dir(model_dir):
        return model_dir

    root = Path(INPUT_DIR)
    candidates = []
    for p in root.rglob("config.json"):
        parent = p.parent
        score = 0
        name = parent.name.lower()
        if name.startswith("checkpoint-"):
            score += 10
        if (parent / "model.safetensors").exists() or (
            parent / "pytorch_model.bin"
        ).exists():
            score += 5
        if "bert" in name:
            score += 1
        candidates.append((score, parent))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (-x[0], len(str(x[1]))))
    return candidates[0][1]


ckpt_path = _find_checkpoint_dir()
print(
    "Resolved checkpoint path:", ckpt_path if ckpt_path is not None else "(none found)"
)

if ckpt_path is not None:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(ckpt_path), local_files_only=True
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        str(ckpt_path), local_files_only=True
    )
    print("Loaded tokenizer/model from local checkpoint:", ckpt_path)
else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=6
    )
    print("Loaded tokenizer/model from base pretrained:", MODEL_NAME)

model_max_len = int(getattr(model.config, "max_position_embeddings", 512))
if MAX_LENGTH > model_max_len:
    print(
        f"Capping MAX_LENGTH from {MAX_LENGTH} to model max_position_embeddings={model_max_len}"
    )
MAX_LENGTH = min(MAX_LENGTH, model_max_len)

use_fp16 = bool(torch.cuda.is_available())

training_args = transformers.TrainingArguments(
    output_dir=".",
    per_device_eval_batch_size=32,
    report_to="none",
    fp16=use_fp16,
)

trainer = transformers.Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
)

trainer.model.to(device)
trainer.model.eval()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
test_path = (
    Path(INPUT_DIR) / "learning-agency-lab-automated-essay-scoring-2" / "test.csv"
)
sample_path = (
    Path(INPUT_DIR)
    / "learning-agency-lab-automated-essay-scoring-2"
    / "sample_submission.csv"
)

df = pd.read_csv(test_path)
texts = df["full_text"].astype(str).values.tolist()
print("Read test csv successfully:", df.shape)

encodings = tokenizer(
    texts,
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
)

dataset = datasets.Dataset.from_dict(encodings)

cols = ["input_ids", "attention_mask"]
if "token_type_ids" in dataset.column_names:
    cols.append("token_type_ids")

dataset = dataset.with_format("torch", columns=cols)
print("Dataset built:", dataset, "columns:", cols)

pred_out = trainer.predict(dataset)
submission_logits = pred_out.predictions

submission_preds = (
    torch.argmax(torch.tensor(submission_logits), dim=1).cpu().numpy() + 1
)

print(
    "Predictions generated:",
    submission_preds.shape,
    "min/max:",
    int(submission_preds.min()),
    int(submission_preds.max()),
)

submission_csv = pd.DataFrame(
    {"essay_id": df["essay_id"].values, "score": submission_preds.astype(int)}
)
submission_csv = submission_csv[["essay_id", "score"]]

submission_path = "submission.csv"
submission_csv.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission_csv.head())
