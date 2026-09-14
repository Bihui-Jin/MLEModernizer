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

0.7690453836220605

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix two blockers so the notebook runs end-to-end and writes a valid `submission.csv`: (1) avoid the `MessageFactory.GetPrototype` crash by disabling fast tokenizers (this is a known protobuf/tokenizers interaction in some Kaggle images), and (2) fix the BERT max length mismatch by capping `MAX_LENGTH` at the model’s supported `model_max_length` (BERT is 512). I also ensure the Trainer actually moves the model to GPU/CPU deterministically and uses `torch.no_grad()`-style inference via `Trainer.predict`. These changes preserve your core inference logic (load checkpoint → tokenize → Trainer.predict → argmax+1) while making it runnable.'
- What this solution (achieved 0.0) has done: 'I fix the `MessageFactory.GetPrototype` crash by forcing Python protobuf implementation (a common incompatibility in some Kaggle images with newer `protobuf` + `transformers/tokenizers`). I also keep `use_fast=False` and ensure the environment variables are set before importing `transformers` to make the fix take effect. The rest of the pipeline (load checkpoint → tokenize → `Trainer.predict` → argmax+1 → write `submission.csv`) is preserved to keep evaluation semantics unchanged while producing a valid submission.'
- What this solution (achieved 0.0) has done: 'I fix the `MessageFactory.GetPrototype` crash by setting the additional protobuf environment switch `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` *before* importing `transformers`/`datasets`, which is the minimal reliable workaround in Kaggle images where protobuf 5+ breaks older generated code paths. I also add a safe fallback to bypass protobuf-based `datasets` entirely by using a tiny Torch `Dataset` wrapper around tokenizer outputs, so prediction and submission writing always complete end-to-end. These changes do not alter your core logic (load checkpoint → tokenize → Trainer.predict → argmax+1) and are score-neutral compared to the intended pipeline, but prevent the 0.0 score caused by not producing a valid submission. The script still write `submission.csv` with columns `essay_id,score`.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf/transformers crash causing the `MessageFactory.GetPrototype` error by forcing the pure-Python protobuf runtime *and* preventing `transformers` from importing protobuf-dependent components (like `sentencepiece`) before the env vars take effect. I also make the import order strict (env vars first, then `transformers`) and add a safe fallback to load the tokenizer/model from the base competition directory if the checkpoint folder is not a valid HF directory. These changes are execution/stability fixes (score should jump from 0.0 to a real value because a valid submission be produced) and preserve your intended core logic: load checkpoint → tokenize → `Trainer.predict` → argmax+1 → write `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import re
import numpy as np
import pandas as pd
from pathlib import Path
import torch

import transformers

MODEL_NAME = "bert-base-cased"
INPUT_DIR = "/kaggle/input/"

MODEL_DIR = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"
CHECKPOINT_SUBDIR = "checkpoint-8000"

MAX_LENGTH = 1024
RANDOM_SEED = 42
EVAL_USE_PRETRAIN = 1
SUBMISSION = 1

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
if torch.cuda.is_available():
    print(torch.cuda.current_device())

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print(f"PyTorch version: {torch.__version__}")
print(f"Hugging Face Transformers version: {transformers.__version__}")



## === cell 1
base_path = Path(MODEL_DIR)
ckpt_path = base_path / CHECKPOINT_SUBDIR


def _is_hf_model_dir(p: Path) -> bool:
    if not p.exists() or not p.is_dir():
        return False
    cfg = p / "config.json"
    return cfg.exists() and cfg.is_file()


def _extract_step(p: Path) -> int:
    m = re.search(r"checkpoint-(\d+)$", p.name)
    return int(m.group(1)) if m else -1


def _find_best_hf_dir(root: Path):
    if not root.exists() or not root.is_dir():
        return None

    candidates = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        if _is_hf_model_dir(p):
            candidates.append(p)

    if not candidates:
        return None

    candidates.sort(key=lambda p: (_extract_step(p), str(p)))
    return candidates[-1]


resolved_path = None
if _is_hf_model_dir(ckpt_path):
    resolved_path = str(ckpt_path)
elif _is_hf_model_dir(base_path):
    resolved_path = str(base_path)
else:
    best = _find_best_hf_dir(base_path)
    resolved_path = (
        str(best) if best is not None else MODEL_NAME
    )  # fall back to HF hub/cache

print(f"Resolved model_path: {resolved_path}")

try:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        resolved_path, local_files_only=True, use_fast=False
    )
    print(
        f"Loaded tokenizer from: {resolved_path} (local_files_only=True, use_fast=False)"
    )
except Exception as e:
    print(f"Tokenizer local load failed: {type(e).__name__}: {e}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        resolved_path, local_files_only=False, use_fast=False
    )
    print(
        f"Loaded tokenizer from: {resolved_path} (local_files_only=False, use_fast=False)"
    )

try:
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        resolved_path, local_files_only=True
    )
    print(f"Loaded model from: {resolved_path} (local_files_only=True)")
except Exception as e:
    print(f"Model local load failed: {type(e).__name__}: {e}")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        resolved_path, local_files_only=False
    )
    print(f"Loaded model from: {resolved_path} (local_files_only=False)")

tokenizer_max = getattr(tokenizer, "model_max_length", None)
if tokenizer_max is None or tokenizer_max > 100000:  # sentinel values used by HF
    tokenizer_max = 512
model_max_pos = getattr(getattr(model, "config", None), "max_position_embeddings", None)
if model_max_pos is None:
    model_max_pos = tokenizer_max

MAX_LENGTH = int(min(MAX_LENGTH, tokenizer_max, model_max_pos))
print(f"Effective MAX_LENGTH set to: {MAX_LENGTH}")

training_args = transformers.TrainingArguments(
    output_dir=".",
    per_device_eval_batch_size=32,
    report_to="none",
    fp16=bool(torch.cuda.is_available()),
)

trainer = transformers.Trainer(model=model, args=training_args)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
test_path_1 = (
    Path(INPUT_DIR) / "learning-agency-lab-automated-essay-scoring-2" / "test.csv"
)
test_path_2 = Path(INPUT_DIR) / "test.csv"
if test_path_1.exists():
    test_path = str(test_path_1)
elif test_path_2.exists():
    test_path = str(test_path_2)
else:
    test_path = INPUT_DIR + "learning-agency-lab-automated-essay-scoring-2/test.csv"

df_test = pd.read_csv(test_path)
texts = df_test["full_text"].astype(str).tolist()
print("Read csv successfully:", test_path)

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
)


class EncodedTorchDataset(torch.utils.data.Dataset):
    def __init__(self, enc):
        self.enc = enc
        self.keys = list(enc.keys())
        self.length = len(enc[self.keys[0]]) if self.keys else 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = {}
        for k in self.keys:
            v = self.enc[k][idx]
            if not torch.is_tensor(v):
                v = torch.tensor(v, dtype=torch.long)
            item[k] = v
        return item


ds = EncodedTorchDataset(encodings)
print("Dataset prepared (torch dataset)")

pred = trainer.predict(ds).predictions
pred_labels = torch.argmax(torch.tensor(pred), dim=1).cpu().numpy() + 1  # 1..6
pred_labels = pred_labels.astype(int)
print("Predicted successfully")

submission = pd.DataFrame(
    {"essay_id": df_test["essay_id"].values, "score": pred_labels}
)
submission["score"] = submission["score"].clip(1, 6).astype(int)
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv")
