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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
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

0.6623544245383161

# 6. Current score

0.46202

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.46202) has done: 'I remove the dependency on the missing external `/kaggle/input/multi-scale-bert-aes` script (which is causing the pipeline to crash) and replace it with an in-notebook predictor that works in the provided Kaggle environment. To keep core logic simple and stable, I use a sentence-transformer to embed `full_text` and then fit a lightweight regressor on train, rounding/clipping predictions to the required 1–6 integer scores. This also eliminate the non-numeric “pred file parsing” issue and guarantee a valid `submission.csv` is written. The submission be aligned to `essay_id` order from `test.csv` and use the exact required column names.'

# 9. Code solution

## === cell 0
import os
import sys
import glob
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np

print("Python:", sys.version)
print("Working dir:", os.getcwd())
print("Listing /kaggle/input (top-level):", os.listdir("/kaggle/input")[:10])



## === cell 1
TEST_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
test_df = pd.read_csv(TEST_PATH)

test_df["full_text"] = (
    test_df["full_text"].astype(str).str.replace("\n", " ", regex=False)
)

test_df.head()



## === cell 2
out_tsv = "/kaggle/working/p8_fold3_test.txt"
test_df_out = test_df.copy()

if "score" not in test_df_out.columns:
    test_df_out["score"] = np.random.randint(
        1, 7, size=len(test_df_out)
    )  # inclusive 1..6

test_df_out.to_csv(out_tsv, header=False, index=False, sep="\t")
print("Wrote:", out_tsv, "rows:", len(test_df_out))



## === cell 3
TRAIN_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
train_df = pd.read_csv(TRAIN_PATH)

train_df["full_text"] = (
    train_df["full_text"].astype(str).str.replace("\n", " ", regex=False)
)
train_df["score"] = pd.to_numeric(train_df["score"], errors="coerce").astype(int)

print("Train shape:", train_df.shape, "Test shape:", test_df.shape)
print(train_df[["essay_id", "score"]].head())



## === cell 4
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge

np.random.seed(42)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(model_name)

train_texts = train_df["full_text"].tolist()
test_texts = test_df["full_text"].tolist()

X_train = embedder.encode(
    train_texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
X_test = embedder.encode(
    test_texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

y_train = train_df["score"].values.astype(float)

reg = Ridge(alpha=1.0, random_state=42)
reg.fit(X_train, y_train)

pred_cont = reg.predict(X_test).astype(float)

df_pred = pd.DataFrame({"label": np.zeros(len(test_df), dtype=int), "pred": pred_cont})
df_pred.head()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
df_pred["pred"] = pd.to_numeric(df_pred["pred"], errors="coerce")

if df_pred["pred"].isna().any():
    fill_val = float(np.mean(y_train))
    df_pred["pred"] = df_pred["pred"].fillna(fill_val)

df_pred["pred_1to6"] = np.rint(df_pred["pred"]).astype(int).clip(1, 6)

df_pred[["pred", "pred_1to6"]].head()



## === cell 6
if len(df_pred) != len(test_df):
    raise ValueError(f"Row mismatch: test={len(test_df)} preds={len(df_pred)}")

submission = pd.DataFrame(
    {"essay_id": test_df["essay_id"].values, "score": df_pred["pred_1to6"].values}
)
submission["score"] = submission["score"].clip(1, 6).astype(int)

sub_path = "/kaggle/working/submission.csv"
submission.to_csv(sub_path, index=False)

print("Wrote submission:", sub_path)
print(submission.head())
print("Submission shape:", submission.shape)
print("Score value counts:\n", submission["score"].value_counts().sort_index())
