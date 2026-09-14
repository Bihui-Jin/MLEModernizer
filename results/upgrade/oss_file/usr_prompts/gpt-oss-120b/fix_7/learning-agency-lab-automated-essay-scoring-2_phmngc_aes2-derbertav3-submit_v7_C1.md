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

3.14

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

0.8017657349561782

# 6. Current score

0.66954

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.67218) has done: 'I fixed the file‑search logic so the script can locate the CSVs regardless of the exact folder layout, defined the output path, and renumbered the cells to start at 1. I also set a more expressive TF‑IDF configuration (100 k features, 1‑3 grams) to give a better chance of reaching the target kappa while keeping the original model unchanged. The updated script now runs end‑to‑end and writes a proper `submission.csv` file.'
- What this solution (achieved 0.67821) has done: 'I keep the overall pipeline unchanged but improve the text representation and model regularization, which are the most effective low‑impact levers for quadratic weighted kappa. Both TF‑IDF vectorizers now use `sublinear_tf=True` (helps with long essays) and the retraining step also expands the feature limit to 200 k. The logistic regression classifier gets a slightly larger inverse‑regularisation (`C=2.0`) to capture more signal without altering the model class. These tweaks are minimal yet should raise validation kappa toward the target.'
- What this solution (achieved 0.66954) has done: 'I raise the TF‑IDF capacity and n‑gram range (to capture richer text patterns) and make the logistic regression a bit less regularised (C = 4.0). These small hyper‑parameter tweaks keep the original pipeline intact while giving the model more expressive power, which should lift the validation quadratic weighted kappa toward the target.'

# 9. Code solution

## === cell 0
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings("ignore", category=UserWarning)


def find_file(filename: str) -> str:
    """
    Search recursively from the current directory for the given filename.
    Returns the first match found.
    """
    matches = list(Path(".").rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not locate {filename} in the repository")
    return str(matches[0])


TRAIN_PATH = find_file("train.csv")
TEST_PATH = find_file("test.csv")
OUTPUT_PATH = "."  # write submission to current working directory




## === cell 1
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

assert {"essay_id", "full_text", "score"}.issubset(train_df.columns)
assert {"essay_id", "full_text"}.issubset(test_df.columns)

train_split, val_split = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["score"],
    random_state=42,
)

vectorizer = TfidfVectorizer(
    max_features=300_000,
    ngram_range=(1, 4),
    stop_words="english",
    dtype=np.float32,
    sublinear_tf=True,
    min_df=2,
)

X_train = vectorizer.fit_transform(train_split["full_text"])
y_train = train_split["score"].astype(int)

X_val = vectorizer.transform(val_split["full_text"])
y_val = val_split["score"].astype(int)

model = LogisticRegression(
    max_iter=1000,
    n_jobs=5,
    multi_class="multinomial",
    solver="lbfgs",
    class_weight="balanced",
    C=4.0,
)

model.fit(X_train, y_train)

val_pred = model.predict(X_val)
val_kappa = cohen_kappa_score(y_val, val_pred, weights="quadratic")
print(f"Validation Quadratic Weighted Kappa: {val_kappa:.6f}")

TARGET_KAPPA = 0.8017657349561782
if val_kappa < TARGET_KAPPA - 0.02:
    print("Kappa low – retraining with larger feature set...")
    vectorizer = TfidfVectorizer(
        max_features=400_000,
        ngram_range=(1, 4),
        stop_words="english",
        dtype=np.float32,
        sublinear_tf=True,
        min_df=2,
    )
    X_train = vectorizer.fit_transform(train_split["full_text"])
    X_val = vectorizer.transform(val_split["full_text"])
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_kappa = cohen_kappa_score(y_val, val_pred, weights="quadratic")
    print(f"New Validation Kappa: {val_kappa:.6f}")




## === cell 2
X_full = vectorizer.fit_transform(train_df["full_text"])
y_full = train_df["score"].astype(int)
model.fit(X_full, y_full)




## === cell 3
X_test = vectorizer.transform(test_df["full_text"])
test_pred = model.predict(X_test)
test_pred = np.clip(test_pred, 1, 6).astype(int)

submission_df = pd.DataFrame({"essay_id": test_df["essay_id"], "score": test_pred})

print("Submission preview:")
print(submission_df.head())




## === cell 4
submission_path = os.path.join(OUTPUT_PATH, "submission.csv")
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
