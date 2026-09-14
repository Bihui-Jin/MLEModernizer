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
Given questions and answers from various StackExchange properties, predict target values of 30 labels for each question-answer pair.

## Metric
Mean column-wise Spearman's correlation coefficient. The Spearman's rank correlation is computed for each target column, and the mean of these values is calculated for the submission score.

## Submission Format
For each qa_id in the test set, you must predict a probability for each target variable. The predictions should be in the range [0,1]. The file should contain a header and have the following format:

```
qa_id,question_asker_intent_understanding,...,answer_well_written
6,0.0,...,0.5
8,0.5,...,0.1
18,1.0,...,0.0
etc.
```

## Dataset
The list of 30 target labels are the same as the column names in the `sample_submission.csv` file. Target labels with the prefix `question_` relate to the `question_title` and/or `question_body` features in the data. Target labels with the prefix `answer_` relate to the `answer` feature.

Target labels are aggregated from multiple raters, and can have continuous values in the range `[0,1]`. Therefore, predictions must also be in that range.

- **train.csv** - the training data (target labels are the last 30 columns)
- **test.csv** - the test set (you must predict 30 labels for each test set row)
- **sample_submission.csv** - a sample submission file in the correct format; column names are the 30 target labels

# 2. Python version

3.8

# 3. Installed packages

cufflinks==0.17.3
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
plotly==5.24.1
plotly-express==0.4.1
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        input/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        working/
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
```

-> data/google-quest-challenge/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/google-quest-challenge/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/google-quest-challenge/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> data/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.11905

# 6. Current score

0.32058

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.32058) has done: 'Your notebook currently stops after imports and directory listing, so it never trains a model or writes `submission.csv`, which is why no score was yielded. I add the smallest end-to-end pipeline that matches the competition metric reasonably: TF‑IDF features on the text fields, a simple multi-output regression model, and then rank-based postprocessing so predictions are in `[0,1]` and better aligned with Spearman correlation. I keep it lightweight to fit the 600s limit by training on a subset of the training rows (core approach stays “TF‑IDF -> SVD -> regression”), and I ensure the submission matches `sample_submission.csv` columns and row order. The output be a valid `submission.csv` in the working directory.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_version
except Exception:
    _pb_version = None


def _major(ver):
    try:
        return int(str(ver).split(".")[0])
    except Exception:
        return None


if (
    _pb_version is not None
    and (_major(_pb_version) is not None)
    and _major(_pb_version) >= 6
):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )
    import importlib
    import google.protobuf as _gp

    importlib.reload(_gp)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette()
import plotly.offline as py

py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline

offline.init_notebook_mode()
import cufflinks as cf

cf.go_offline()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from matplotlib_venn import venn2
import re
import nltk
from nltk.corpus import stopwords
import string
import gc

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
DATA_DIR = "/kaggle/input/google-quest-challenge"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "/kaggle/input"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

target_cols = [c for c in sample_sub.columns if c != "qa_id"]
assert len(target_cols) == 30, f"Expected 30 target columns, got {len(target_cols)}"

print("train:", train_df.shape, "test:", test_df.shape, "sample:", sample_sub.shape)
print("targets:", target_cols[:5], "...", target_cols[-1])




## === cell 2
def _clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_text(df: pd.DataFrame) -> pd.Series:
    return (
        df["question_title"].map(_clean_text)
        + " [SEP] "
        + df["question_body"].map(_clean_text)
        + " [SEP] "
        + df["answer"].map(_clean_text)
        + " [CAT] "
        + df["category"].map(_clean_text)
        + " [HOST] "
        + df["host"].map(_clean_text)
    )


train_text = build_text(train_df)
test_text = build_text(test_df)

y = train_df[target_cols].astype(np.float32).values



## === cell 3
SEED = 42
rng = np.random.default_rng(SEED)

max_train_rows = 50000  # chosen to be fast yet decent; avoids timeout/OOM
if len(train_df) > max_train_rows:
    idx = rng.choice(len(train_df), size=max_train_rows, replace=False)
    idx.sort()
    train_text_fit = train_text.iloc[idx]
    y_fit = y[idx]
else:
    train_text_fit = train_text
    y_fit = y

print("Using train rows:", len(train_text_fit), "of", len(train_df))



## === cell 4
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    max_features=80000,
    strip_accents="unicode",
    lowercase=True,
)

X_tfidf_train = tfidf.fit_transform(train_text_fit)
X_tfidf_test = tfidf.transform(test_text)

svd = TruncatedSVD(n_components=300, random_state=SEED)
X_train = svd.fit_transform(X_tfidf_train).astype(np.float32)
X_test = svd.transform(X_tfidf_test).astype(np.float32)

print("X_train:", X_train.shape, "X_test:", X_test.shape)

del X_tfidf_train, X_tfidf_test
gc.collect()



## === cell 5
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

base = Ridge(alpha=1.0, random_state=SEED)
model = MultiOutputRegressor(base, n_jobs=-1)
model.fit(X_train, y_fit)

pred = model.predict(X_test).astype(np.float32)


def rank_scale_01(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(a), dtype=np.float32)
    if len(a) > 1:
        ranks /= len(a) - 1.0
    else:
        ranks[:] = 0.5
    return ranks


pred_ranked = np.zeros_like(pred, dtype=np.float32)
for j in range(pred.shape[1]):
    pred_ranked[:, j] = rank_scale_01(pred[:, j])

pred_ranked = np.clip(pred_ranked, 0.0, 1.0)



## === cell 6
submission = pd.DataFrame(pred_ranked, columns=target_cols)
submission.insert(0, "qa_id", test_df["qa_id"].values)

submission = submission[sample_sub.columns.tolist()]

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path, "shape:", submission.shape)
print(submission.head())
