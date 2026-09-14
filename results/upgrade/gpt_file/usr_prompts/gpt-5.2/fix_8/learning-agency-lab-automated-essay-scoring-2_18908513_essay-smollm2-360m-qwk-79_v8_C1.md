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
text-unidecode==1.3
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

0.786623184549083

# 6. Current score

0.69828

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00556) has done: 'I fix the immediate crash by making model loading robust to local Kaggle dataset folder structures (including nested `snapshots/` layouts) and by forcing Transformers to treat the path as a local directory rather than a Hub repo id. Then I unblock the downstream cells by ensuring `df`, `tokenizer`, and `model` are always defined (with a safe fallback model if the provided local model isn’t present), so the script runs end-to-end and writes `submission.csv` with the required columns. I also add missing tokenizer settings needed for some decoder-only tokenizers (pad token) to prevent runtime padding errors during batching. These changes keep the core inference logic (argmax over logits, +1, clip 1–6) the same while ensuring a valid submission is produced.'
- What this solution (achieved 0.69828) has done: 'I fix the crash by switching Ridge away from the `"sag"` solver (which is triggering an incompatibility between scikit-learn and the available SciPy `cg()` signature in this environment) to a solver that works reliably with sparse TF‑IDF matrices. I keep the same TF‑IDF + Ridge core approach and identical post-processing (round → clip 1–6) so the evaluation semantics stay the same while producing a valid `submission.csv`. I also add a couple of small robustness checks (ensure required columns exist; keep output column order) that are score-neutral but prevent silent format issues.'

# 9. Code solution

## === cell 0
import os
import re
import codecs
from typing import Tuple

import numpy as np
import pandas as pd
from text_unidecode import unidecode

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end


codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve encoding problems and normalize abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


def preprocess_essay_text(text: str) -> str:
    """
    Prepares essay text for scoring by cleaning non-essential issues without altering quality indicators.
    - Resolves encoding issues
    - Normalizes whitespace
    - Preserves original spelling, grammar, and casing
    """
    text = resolve_encodings_and_normalize(str(text))
    text = re.sub(r"\s+", " ", text.strip())  # Normalize whitespace
    text = re.sub(r'\s+([?.!,"])', r"\1", text)  # Remove spaces before punctuation
    text = re.sub(r",([^\s])", r", \1", text)  # Add space after commas
    return text


def _pick_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the candidate paths exist: {candidates}")


TRAIN_CANDIDATES = [
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv",
    "/kaggle/input/train.csv",
]
TEST_CANDIDATES = [
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv",
    "/kaggle/input/test.csv",
]

train_path = _pick_existing_path(TRAIN_CANDIDATES)
test_path = _pick_existing_path(TEST_CANDIDATES)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

required_train_cols = {"essay_id", "full_text", "score"}
required_test_cols = {"essay_id", "full_text"}
missing_train = required_train_cols - set(train_df.columns)
missing_test = required_test_cols - set(test_df.columns)
if missing_train:
    raise ValueError(f"train.csv missing columns: {missing_train}")
if missing_test:
    raise ValueError(f"test.csv missing columns: {missing_test}")

train_df["full_text"] = train_df["full_text"].astype(str).apply(preprocess_essay_text)
test_df["full_text"] = test_df["full_text"].astype(str).apply(preprocess_essay_text)

print("Loaded train from:", train_path, "rows:", len(train_df))
print("Loaded test  from:", test_path, "rows:", len(test_df))
print(train_df.head(2))




## === cell 1
X_train = train_df["full_text"].values
y_train = train_df["score"].astype(float).values
X_test = test_df["full_text"].values

model = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=False,
                strip_accents=None,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                max_features=200_000,
            ),
        ),
        ("ridge", Ridge(alpha=3.0, random_state=42, solver="lsqr")),
    ]
)

model.fit(X_train, y_train)
preds = model.predict(X_test)

preds = np.rint(preds).astype(int)
preds = np.clip(preds, 1, 6)

submission = pd.DataFrame({"essay_id": test_df["essay_id"].values, "score": preds})
submission = submission[["essay_id", "score"]]

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print(f"Wrote {out_path} with shape:", submission.shape)
print(submission.head())
print(submission["score"].value_counts().sort_index())
