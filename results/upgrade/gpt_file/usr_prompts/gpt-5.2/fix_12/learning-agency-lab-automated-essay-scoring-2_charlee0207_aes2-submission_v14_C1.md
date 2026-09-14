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

0.781539931926555

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the environment/runtime error coming from an incompatible protobuf stack by forcing the pure-Python protobuf implementation before importing `transformers`/`datasets`. Then I fix the inference crash by aligning `MAX_LENGTH` with BERT’s maximum positional embeddings (512), which removes the 1024-vs-512 shape mismatch while keeping the same argmax+1 prediction logic. Finally, I make the dataset compatible with `Trainer.predict()` by setting torch format and moving the model to the detected device, ensuring a `submission.csv` with the required columns is always written.'
- What this solution (achieved 0.0) has done: 'We fix the crash in `transformers` caused by an incompatible protobuf runtime by force-uninstalling `protobuf` in-notebook and then installing a compatible 4.x version before importing `transformers/datasets` (the environment variable alone doesn’t prevent that specific `MessageFactory.GetPrototype` failure). This is a runtime-only fix and keeps your model/inference logic unchanged. We also keep `MAX_LENGTH=512`, ensure tensors are produced for `Trainer.predict()`, and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is almost certainly because the submission is malformed for this competition (the test set has 15335 essays, but you’re writing a file with only 1731 rows from `sample_submission.csv`, so Kaggle score it as invalid/0.0). I make the smallest fix: build the submission directly from `test.csv` so `essay_id` alignment and row count are correct, while keeping your existing model loading and argmax+1 prediction logic unchanged. I also add a strict sanity check that the written submission has exactly the expected columns, no missing IDs, and 15335 rows to prevent another 0.0. Everything else (model, tokenizer, MAX_LENGTH=512, Trainer.predict) stays the same.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with Kaggle marking the submission invalid (wrong row count) and/or producing constant/degenerate predictions due to a label-mapping mismatch in the saved checkpoint (common when fine-tuning with labels 1–6 but the head is configured for 0–5, or vice versa). I keep your exact inference approach (tokenize → `Trainer.predict()` → `argmax` → clip to 1–6), but add a minimal, safe post-processing step that respects `model.config.id2label/label2id` when present so logits are mapped back to the intended score values. I also add a hard assertion that the submission row count matches `test.csv` (15335) to prevent another invalid/0.0, while leaving all paths and core logic intact. These changes are directly aimed at turning the current 0.0 into a valid, non-degenerate submission that should move the QWK upward toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with Kaggle rejecting the submission because the predictions are not aligned to the test set order/IDs or because the checkpoint isn’t actually being used and the model outputs degenerate/unmapped labels. I keep your exact inference pipeline (tokenize → `Trainer.predict()` → argmax → map/clip to 1–6), but make the submission alignment deterministic by merging predictions back onto `sample_submission.csv` (the canonical required ordering) using `essay_id`. I also add strict checks that the checkpoint loads as expected and that the submission has exactly the same `essay_id` set and order as `sample_submission.csv`, preventing another invalid/0.0. These are minimal changes aimed at producing a valid, correctly-aligned submission so the score can move upward toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with Kaggle rejecting the submission as invalid due to the wrong row count: you’re currently forcing the submission to match `sample_submission.csv` (1731 rows) while the real `test.csv` has 15335 rows, so the file won’t validate. I make the minimal fix: build the submission directly from `test.csv` (one row per test `essay_id`) and add strict checks that the output has exactly 15335 rows and the required columns. I keep your core inference pipeline unchanged (same checkpoint loading, tokenization, `Trainer.predict()`, argmax-based mapping, clipping to 1–6). This should move the score upward from 0.0 toward the target by producing a valid, correctly-aligned submission.'
- What this solution (achieved 0.0) has done: 'Your 0.0 is most consistent with Kaggle marking the submission invalid, and the biggest red flag in your script is that you read the nested `learning-agency.../test.csv` (15335 rows) while `sample_submission.csv` in that same folder has 1731 rows, indicating the competition’s evaluated test set is likely the *root* `/kaggle/input/test.csv` with 1731 rows. I make the minimal change to always build predictions for the exact `essay_id` list in the root `sample_submission.csv` (canonical ordering), joining the corresponding `full_text` from the matching root `test.csv`, so the submission schema/row count matches what Kaggle expects. I keep your core inference pipeline unchanged (same checkpoint loading, tokenizer, `Trainer.predict()`, argmax mapping + clip), but add strict assertions that the output row count and essay_id order exactly match the sample submission to prevent another invalid/0.0. This should move the score upward toward your target by producing a valid, correctly-aligned submission.'
- What this solution (achieved 0.0) has done: 'Your 0.0 is still most consistent with Kaggle rejecting the submission as invalid due to a row-count/ID-mismatch: your code currently forces the 1731-row `sample_submission.csv` ordering, but in this dataset the evaluated `test.csv` is 15335 rows, so the correct submission must be built from `test.csv` (or at minimum match its `essay_id` set). I make the smallest change that fixes validity: generate predictions for every row in `/kaggle/input/test.csv` and write `submission.csv` with exactly 15335 rows (`essay_id,score`) in that same order. I keep your core inference intact (same tokenizer/model/Trainer.predict, argmax mapping + clip) and only adjust the alignment logic plus add strict assertions to prevent another invalid submission. This should move the score up from 0.0 toward your target by producing a valid, properly-aligned submission.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely coming from Kaggle rejecting the submission as invalid due to a row-count mismatch: your `sample_submission.csv` has 1731 rows while your `test.csv` has 15335 rows, and Kaggle typically expects the submission to match the sample submission exactly. I make the smallest alignment change: build the prediction set by taking the canonical `essay_id` order from `sample_submission.csv` and joining the corresponding `full_text` from `test.csv`, then predict only for those rows. I keep your exact model/tokenizer/`Trainer.predict()` pipeline and the same argmax(+label-mapping)+clip-to-1..6 postprocessing, but add strict checks to guarantee the written `submission.csv` matches the sample submission IDs and order. This should convert the 0.0 into a valid scored submission and move QWK upward toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 is still most consistent with Kaggle treating the submission as invalid because the row count/IDs don’t match what the competition expects: your code forces the 1731-row `sample_submission.csv`, but the provided `test.csv` has 15335 rows, and Kaggle scoring requires one prediction per `essay_id` in the evaluated test set. I make the minimal change to build the submission directly from `test.csv` (preserving test order), while keeping your exact inference pipeline (tokenize → `Trainer.predict()` → argmax-based label mapping → clip to 1–6) intact. I also add strict assertions that the output has exactly the test row count and the same essay_id order to prevent another 0.0. Everything else (model, tokenizer, MAX_LENGTH=512, fp16 behavior) remains unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")



## === cell 1
import sys
import subprocess


def _pip_install(req: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", req])


def _pip_uninstall(pkg: str):
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", pkg])


_pip_uninstall("protobuf")
_pip_install("protobuf>=4.21.0,<5")

import torch

print(f"PyTorch version: {torch.__version__}")

import transformers

print(f"Hugging Face Transformers version: {transformers.__version__}")

import datasets

print(f"Hugging Face Datasets version: {datasets.__version__}")



## === cell 2
import numpy as np
import pandas as pd

MODEL_NAME = "bert-base-cased"
INPUT_DIR = "/kaggle/input/"
MODEL_DIR = "/kaggle/input/aes2-bertbase-5ep-results/aes2_bertbase_5ep_results/"
CHECKPOINT_SUBDIR = "checkpoint-1000"

MAX_LENGTH = 512

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



## === cell 3
checkpoint_path = os.path.join(MODEL_DIR, CHECKPOINT_SUBDIR)
use_path = checkpoint_path if os.path.isdir(checkpoint_path) else MODEL_NAME

if os.path.isdir(checkpoint_path):
    print("Loading local checkpoint from:", checkpoint_path)
else:
    print("WARNING: checkpoint not found at:", checkpoint_path)
    print("Falling back to base model:", MODEL_NAME)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    use_path,
    local_files_only=os.path.isdir(checkpoint_path),
    use_fast=True,
)
print("Loaded tokenizer successfully")

model = transformers.AutoModelForSequenceClassification.from_pretrained(
    use_path,
    local_files_only=os.path.isdir(checkpoint_path),
)
model.to(device)
model.eval()
print("Loaded model successfully")

if hasattr(model, "config"):
    print("num_labels:", getattr(model.config, "num_labels", None))
    print(
        "id2label sample:",
        dict(list(getattr(model.config, "id2label", {}).items())[:5]),
    )

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
)




## === cell 4
def _logits_to_scores_with_label_mapping(logits: np.ndarray, model) -> np.ndarray:
    """
    Keep core inference semantics (argmax over logits), but map argmax indices to
    intended score values using model.config.id2label when available to avoid
    label-index mismatches that can collapse QWK.
    """
    pred_idx = np.argmax(logits, axis=1).astype(int)

    id2label = getattr(getattr(model, "config", None), "id2label", None)
    if isinstance(id2label, dict) and len(id2label) > 0:
        parsed = {}
        ok = True
        for k, v in id2label.items():
            try:
                parsed[int(k)] = v
            except Exception:
                ok = False
                break
        if ok:
            id2label = parsed

        if all(int(i) in id2label for i in range(int(np.max(pred_idx)) + 1)):
            mapped = []
            for i in pred_idx:
                lab = id2label.get(int(i))
                if isinstance(lab, str):
                    s = lab.strip().upper()
                    if s.startswith("LABEL_"):
                        try:
                            mapped.append(int(s.replace("LABEL_", "")) + 1)
                            continue
                        except Exception:
                            pass
                    try:
                        mapped.append(int(lab))
                        continue
                    except Exception:
                        pass
                mapped.append(int(i) + 1)
            return np.asarray(mapped, dtype=int)

    return (pred_idx + 1).astype(int)


test_path = os.path.join(INPUT_DIR, "test.csv")
sample_path = os.path.join(INPUT_DIR, "sample_submission.csv")

df_test = pd.read_csv(test_path)
df_sample = pd.read_csv(sample_path)

print("Read test.csv:", df_test.shape)
print("Read sample_submission.csv:", df_sample.shape)

if list(df_test.columns) != ["essay_id", "full_text"]:
    raise ValueError(f"Unexpected test columns: {df_test.columns.tolist()}")
if list(df_sample.columns) != ["essay_id", "score"]:
    raise ValueError(
        f"Unexpected sample_submission columns: {df_sample.columns.tolist()}"
    )

if df_test["essay_id"].isna().any() or df_test["full_text"].isna().any():
    raise ValueError("test.csv contains NaNs in required columns")
if df_test["essay_id"].duplicated().any():
    raise ValueError("test.csv contains duplicated essay_id values")

texts = df_test["full_text"].astype(str).values.tolist()

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
)
print("Encoded texts successfully; keys:", list(encodings.keys()))

dataset = datasets.Dataset.from_dict(encodings)
dataset = dataset.with_format("torch", columns=list(encodings.keys()))
print("Built Dataset successfully:", dataset)

submission_logits = trainer.predict(dataset).predictions
submission_logits = np.asarray(submission_logits)
print("Predicted successfully; logits shape:", submission_logits.shape)

pred_scores = _logits_to_scores_with_label_mapping(submission_logits, model).astype(int)
pred_scores = np.clip(pred_scores, 1, 6).astype(int)

print(
    "Post-processed predictions:",
    pred_scores.shape,
    "unique preds (up to 20):",
    np.unique(pred_scores)[:20],
)

submission = pd.DataFrame(
    {"essay_id": df_test["essay_id"].astype(str).values, "score": pred_scores}
)

if list(submission.columns) != ["essay_id", "score"]:
    raise ValueError(f"Bad submission columns: {submission.columns.tolist()}")
if len(submission) != len(df_test):
    raise ValueError(
        f"Submission row mismatch vs test.csv: {len(submission)} vs {len(df_test)}"
    )
if not submission["essay_id"].equals(df_test["essay_id"].astype(str)):
    raise ValueError("Submission essay_id order does not match test.csv exactly")
if submission["score"].isna().any():
    raise ValueError("Submission contains NaNs in score")
if submission["essay_id"].duplicated().any():
    raise ValueError("Submission has duplicated essay_id values")

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", submission.shape)
print(submission.head())
