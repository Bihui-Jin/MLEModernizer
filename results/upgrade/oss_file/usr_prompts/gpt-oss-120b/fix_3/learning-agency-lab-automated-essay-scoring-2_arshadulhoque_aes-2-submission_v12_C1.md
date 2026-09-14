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

0.0200793280808134

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I wrap the Hugging‑Face imports in a safe try/except block, fall back to a simple median‑score baseline when the tokenizer/model cannot be loaded (which avoids the protobuf error), and generate a valid `submission.csv` using that baseline. This keeps the original workflow but ensures the script runs end‑to‑end and writes the required file.'
- What this solution (achieved 0.0) has done: 'The fix removes the problematic Hugging Face import, adds a safe fallback that trains a lightweight TF‑IDF + Ridge regression model when the transformer model cannot be used, and ensures a valid `submission.csv` is always written. This resolves the import error, provides a simple but better-than‑median baseline (raising the score toward the target), and keeps the original workflow unchanged.'

# 9. Code solution

## === cell 0
import pandas as pd
import torch
import numpy as np
from datasets import Dataset

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )

    HF_AVAILABLE = True
except Exception as e:
    print(f"Transformers import failed ({e}); proceeding with fallback model.")
    HF_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    SKLEARN_AVAILABLE = True
except Exception as e:
    print(f"Sklearn import failed ({e}); fallback will use median score.")
    SKLEARN_AVAILABLE = False




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
test_data = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
test_data.head()




## === cell 2
train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
train_df = pd.read_csv(train_path)
median_score = int(round(train_df["score"].median()))
print(f"Fallback median score = {median_score}")




## === cell 3
if HF_AVAILABLE:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "/kaggle/input/trained-deberta-xsmall"
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            "/kaggle/input/trained-deberta-xsmall"
        )
    except Exception as e:
        print(f"Failed to load tokenizer/model ({e}); using fallback predictions.")
        tokenizer = None
        model = None
else:
    tokenizer = None
    model = None




## === cell 4
if tokenizer is not None:
    test_encodings = tokenizer(
        test_data["full_text"].tolist(),
        truncation=True,
        padding=True,
        max_length=1536,
    )
    test_dataset = Dataset.from_dict(test_encodings)
    test_dataset = test_dataset.add_column("essay_id", test_data["essay_id"].tolist())
else:
    test_dataset = None




## === cell 5
if model is not None and test_dataset is not None:
    predict_args = TrainingArguments(
        output_dir=".", per_device_eval_batch_size=4, report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=predict_args,
        tokenizer=tokenizer,
    )
    predictions = trainer.predict(test_dataset)
    predicted_labels = torch.argmax(
        torch.tensor(predictions.predictions), dim=-1
    ).numpy()
else:
    if SKLEARN_AVAILABLE:
        try:
            if "tfidf_vectorizer" not in globals():
                tfidf_vectorizer = TfidfVectorizer(
                    max_features=20000, ngram_range=(1, 2)
                )
                X_train = tfidf_vectorizer.fit_transform(
                    train_df["full_text"].astype(str)
                )
                y_train = train_df["score"].astype(float)
                simple_regressor = Ridge(alpha=1.0)
                simple_regressor.fit(X_train, y_train)
            X_test = tfidf_vectorizer.transform(test_data["full_text"].astype(str))
            preds = simple_regressor.predict(X_test)
            predicted_labels = np.clip(np.rint(preds).astype(int), 1, 6)
        except Exception as e:
            print(f"Simple sklearn model failed ({e}); using median fallback.")
            predicted_labels = np.full(len(test_data), median_score, dtype=np.int32)
    else:
        predicted_labels = np.full(len(test_data), median_score, dtype=np.int32)




## === cell 6
submission = pd.DataFrame(
    {"essay_id": test_data["essay_id"], "score": predicted_labels.astype("int32")}
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
