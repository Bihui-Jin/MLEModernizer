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

3.12

# 3. Installed packages

datasets==4.4.1
geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
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

0.77431

# 6. Current score

0.65291

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.66996) has done: 'The fix adds a small environment patch to avoid the protobuf import error, replaces the failing HuggingFace model loading with a lightweight TF‑IDF + LogisticRegression pipeline that trains on the provided training data and creates a valid `submission.csv`. This resolves both the import‑time crash and the invalid local‑path model error while keeping the overall modelling approach simple and deterministic, producing a proper submission file ready for evaluation.'
- What this solution (achieved 0.65291) has done: 'I keep the overall TF‑IDF + LogisticRegression pipeline but improve the prediction step by using the model’s class‑probabilities to compute an expected score and then round it, which usually yields a smoother estimate and a higher quadratic weighted kappa. Additionally, I increase the TF‑IDF feature limit to 100 000 to capture more information. These minimal changes preserve the core logic while aiming to raise the validation QWK toward the target score.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import cohen_kappa_score
import numpy as np




## === cell 1
def train_and_predict(
    train_path: str = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv",
    test_path: str = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Trains a TF‑IDF + LogisticRegression model on the training set and
    generates predictions for the test set. Returns a DataFrame ready for
    submission (columns: essay_id, score).
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train, X_val, y_train, y_val = train_test_split(
        train_df["full_text"],
        train_df["score"],
        test_size=0.1,
        random_state=random_state,
        stratify=train_df["score"],
    )

    pipeline = make_pipeline(
        TfidfVectorizer(
            max_features=100000,  # increased feature count for richer representation
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        ),
        LogisticRegression(
            max_iter=300,  # a few more iterations for convergence stability
            multi_class="multinomial",
            solver="lbfgs",
            n_jobs=-1,
            class_weight="balanced",
        ),
    )

    pipeline.fit(X_train, y_train)

    val_proba = pipeline.predict_proba(X_val)  # shape: (n_samples, n_classes)
    classes = pipeline[-1].classes_  # e.g., array([1,2,3,4,5,6])
    val_expected = np.dot(val_proba, classes)  # continuous score
    val_preds = np.rint(val_expected).astype(int)  # round to nearest integer
    val_preds = np.clip(val_preds, 1, 6)  # ensure valid range

    qwk = cohen_kappa_score(y_val, val_preds, weights="quadratic")
    print(f"Validation Quadratic Weighted Kappa (expected‑score rounding): {qwk:.5f}")

    test_proba = pipeline.predict_proba(test_df["full_text"])
    test_expected = np.dot(test_proba, classes)
    test_preds = np.rint(test_expected).astype(int)
    test_preds = np.clip(test_preds, 1, 6)

    submission = pd.DataFrame(
        {
            "essay_id": test_df["essay_id"],
            "score": test_preds,
        }
    )
    return submission




## === cell 2
if __name__ == "__main__":
    sub_df = train_and_predict()
    sub_df.to_csv("submission.csv", index=False)
    print("Submission file 'submission.csv' written successfully.")
