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

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification



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
        for root, dirs, files in os.walk(path):
            if "config.json" in files:
                return root
    return path


MODEL_DIR = resolve_model_dir(MODEL_NAME)
if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(f"MODEL_DIR not found: {MODEL_DIR}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, local_files_only=True
)
model.to(device)
model.eval()

num_labels = getattr(model.config, "num_labels", None)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1451101473.py in <cell line: 0>()
     15 MODEL_DIR = resolve_model_dir(MODEL_NAME)
     16 if not os.path.isdir(MODEL_DIR):
---> 17     raise FileNotFoundError(f"MODEL_DIR not found: {MODEL_DIR}")
     18 
     19 tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

FileNotFoundError: MODEL_DIR not found: /kaggle/input/f-tfm-small/funnel-small-ft_ver4

## === cell 4
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



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/550828072.py in <cell line: 0>()
      7     for start in range(0, len(texts), BATCH_SIZE):
      8         batch_texts = texts[start : start + BATCH_SIZE]
----> 9         enc = tokenizer(
     10             batch_texts,
     11             max_length=MAX_LENGTH,

NameError: name 'tokenizer' is not defined

## === cell 5
if logits.ndim == 2 and logits.shape[1] >= 2:
    pred_class = logits.argmax(axis=1)
    pred_class = np.clip(pred_class, 0, 5)
    pred_score = (pred_class + 1).astype(np.int32)
else:
    raw = logits.reshape(-1)
    pred_score = np.clip(np.rint(raw), 0, 5).astype(np.int32) + 1

df_sub = pd.DataFrame({"essay_id": df_test["essay_id"].values, "score": pred_score})
df_sub["score"] = df_sub["score"].astype("int32")

df_sub = sample_submission[["essay_id"]].merge(df_sub, on="essay_id", how="left")
if df_sub["score"].isna().any():
    df_sub["score"] = df_sub["score"].fillna(3).astype("int32")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4271241893.py in <cell line: 0>()
      1 # BUGFIX/LOGIC: Original code clipped/rounded raw model outputs, which is incorrect for classification logits.
      2 # Use argmax over logits to get class index; map to score 1..6.
----> 3 if logits.ndim == 2 and logits.shape[1] >= 2:
      4     pred_class = logits.argmax(axis=1)
      5     # If model has exactly 6 labels, assume classes 0..5 map to scores 1..6.

NameError: name 'logits' is not defined

## === cell 6
df_sub.to_csv("submission.csv", index=False)

assert os.path.exists("submission.csv")
assert list(df_sub.columns) == ["essay_id", "score"]
assert df_sub["score"].between(1, 6).all()

print(df_sub.head())
print("Saved submission.csv with shape:", df_sub.shape)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3436570779.py in <cell line: 0>()
      1 # Write valid Kaggle submission
----> 2 df_sub.to_csv("submission.csv", index=False)
      3 
      4 # Lightweight sanity checks
      5 assert os.path.exists("submission.csv")

NameError: name 'df_sub' is not defined
