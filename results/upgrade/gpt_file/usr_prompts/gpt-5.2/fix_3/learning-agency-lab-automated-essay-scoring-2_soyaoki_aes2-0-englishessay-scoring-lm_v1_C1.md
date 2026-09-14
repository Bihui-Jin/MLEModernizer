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
tqdm==4.67.1
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

0.3194937486952924

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime blocker by correcting the Hugging Face model loading to point at a real local directory (or fall back to a standard public model if the Kaggle dataset isn’t present), so `tokenizer`/`model` are defined and inference can run. I also make the data path robust by using the existing `/kaggle/input/learning-agency-lab-automated-essay-scoring-2/` directory you listed, with a safe fallback to `/kaggle/input/.../` variants. Finally, I keep your core approach identical (LM logits as features + `LinearRegression` + clip/round to 1–6) and ensure `submission.csv` is written with the required columns.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.linear_model import LinearRegression

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
def resolve_model_path():
    candidates = [
        "/kaggle/input/englishessay-scoring-lm/models--rong4ivy--EnglishEssay_Scoring_LM",
        "/kaggle/input/englishessay-scoring-lm",
    ]
    for base in candidates:
        if not os.path.isdir(base):
            continue
        snapshots_dir = os.path.join(base, "snapshots")
        if os.path.isdir(snapshots_dir):
            snapshot_subdirs = [
                os.path.join(snapshots_dir, d)
                for d in os.listdir(snapshots_dir)
                if os.path.isdir(os.path.join(snapshots_dir, d))
            ]
            snapshot_subdirs.sort()
            if len(snapshot_subdirs) > 0:
                return snapshot_subdirs[-1]
        if os.path.isfile(os.path.join(base, "config.json")):
            return base

    return "distilbert-base-uncased"


MODEL_LOAD_PATH = resolve_model_path()
print("MODEL_LOAD_PATH:", MODEL_LOAD_PATH)

local_only = os.path.isdir(MODEL_LOAD_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_LOAD_PATH, local_files_only=local_only)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_LOAD_PATH, local_files_only=local_only
)

if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
model.to(device)
model.eval()




## === cell 3
@torch.inference_mode()
def inference_batch(text_list, max_length=256, batch_size=32):
    feats = []
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        logits = out.logits

        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if logits.shape[1] == 1:
            logits = logits.repeat(1, 6)

        feats.append(logits.detach().float().cpu().numpy())
    return np.vstack(feats)




## === cell 4
def resolve_data_path(fname: str) -> str:
    candidates = [
        f"/kaggle/input/learning-agency-lab-automated-essay-scoring-2/{fname}",
        f"/kaggle/data/learning-agency-lab-automated-essay-scoring-2/{fname}",
        f"/kaggle/input/{fname}",
        f"/kaggle/data/{fname}",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find {fname} in expected Kaggle input/data locations."
    )


train_path = resolve_data_path("train.csv")
test_path = resolve_data_path("test.csv")

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

print(df_train.shape, df_test.shape)
print(df_train.columns.tolist())




## === cell 5
MAX_TRAIN_ROWS = (
    12000  # chosen to be safe under the 600s timeout in Kaggle CPU/GPU environments
)

if len(df_train) > MAX_TRAIN_ROWS:
    df_train_sub = df_train.sample(n=MAX_TRAIN_ROWS, random_state=SEED).reset_index(
        drop=True
    )
else:
    df_train_sub = df_train.copy()

train_texts = df_train_sub["full_text"].astype(str).tolist()
y_true_train = df_train_sub["score"].astype(float).values

X_train = inference_batch(train_texts, max_length=256, batch_size=32)
print("X_train shape:", X_train.shape, "y_true_train shape:", y_true_train.shape)




## === cell 6
lr = LinearRegression()
lr.fit(X_train, y_true_train)

print("coefficient shape =", getattr(lr.coef_, "shape", None))
print("intercept =", lr.intercept_)




## === cell 7
y_pred_train_cont = lr.predict(X_train)
y_pred_train_1to6 = np.clip(y_pred_train_cont, 1, 6).round().astype("int8")
print(
    "Train preds min/max:", int(y_pred_train_1to6.min()), int(y_pred_train_1to6.max())
)




## === cell 8
test_texts = df_test["full_text"].astype(str).tolist()
X_test = inference_batch(test_texts, max_length=256, batch_size=32)
print("X_test shape:", X_test.shape)




## === cell 9
y_pred_regress_test = np.clip(lr.predict(X_test), 1, 6).round().astype("int8")

sub = pd.DataFrame(
    {"essay_id": df_test["essay_id"].values, "score": y_pred_regress_test}
)
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Saved submission.csv with shape:", sub.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
print("Unique predicted scores:", np.unique(sub["score"].values))
