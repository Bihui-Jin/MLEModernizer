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

0.7984215469600132

# 6. Current score

0.66029

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.62124) has done: 'The fix removes the problematic protobuf environment setting, checks whether the provided DeBERTa checkpoint exists, and falls back to a lightweight TF‑IDF + LogisticRegression model trained on the available training data when the checkpoint is missing. This ensures the script runs end‑to‑end, produces a valid `submission.csv`, and gives a reasonable baseline score without altering the overall workflow.'
- What this solution (achieved 0.66029) has done: 'I keep the overall pipeline structure but replace the LogisticRegression classifier with a balanced LinearSVC and tweak the TF‑IDF vectorizer (more features and sub‑linear term frequency). These small adjustments usually raise Quadratic Weighted Kappa for text‑classification tasks while leaving the rest of the code untouched, moving the score upward toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

TRAIN_DATA_PATH = (
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)
TEST_DATA_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
MODEL_PATH = "/kaggle/input/training-fold0-aes-deberta-model-starter/deberta-small-fold0/checkpoint-8500"

use_transformer = os.path.isdir(MODEL_PATH)

if use_transformer:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
    )
    from datasets import Dataset

    MAX_LENGTH = 2048
    EVAL_BATCH_SIZE = 1

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    def tokenize(sample):
        return tokenizer(sample["full_text"], max_length=MAX_LENGTH, truncation=True)

    df_test = pd.read_csv(TEST_DATA_PATH)

    ds = (
        Dataset.from_pandas(df_test)
        .map(tokenize, batched=False)
        .remove_columns(["essay_id", "full_text"])
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, local_files_only=True
    )

    args = TrainingArguments(
        output_dir=".",  # required but not used for inference
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=DataCollatorWithPadding(tokenizer),
        tokenizer=tokenizer,
    )

    preds = trainer.predict(ds).predictions
    df_test["score"] = preds.argmax(axis=-1) + 1

else:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import make_pipeline

    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)

    pipeline = make_pipeline(
        TfidfVectorizer(
            max_features=50000,  # more features for richer representation
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,  # dampen impact of very frequent terms
        ),
        LinearSVC(
            C=0.8,  # mild regularisation
            class_weight="balanced",  # handle possible class imbalance
        ),
    )

    pipeline.fit(df_train["full_text"], df_train["score"])

    df_test["score"] = pipeline.predict(df_test["full_text"])




## === cell 1
submission_path = "submission.csv"
df_test[["essay_id", "score"]].to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
