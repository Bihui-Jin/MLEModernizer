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
numpy==1.26.4
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

0.807150201953024

# 6. Current score

0.67517

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.6748) has done: 'I add a robust fallback that trains a lightweight TF‑IDF + LogisticRegression model when the pretrained BERT checkpoint cannot be loaded offline. The script now loads both train and test data, cleans the text, attempts to load the BERT model, and if that fails it fits the fallback model on the training set and uses it for predictions. The rest of the pipeline (batch handling, device setup, and CSV submission) remains unchanged, ensuring a valid `submission.csv` is produced.'
- What this solution (achieved 0.67517) has done: 'I replace the fallback LogisticRegression with a LinearSVC that uses balanced class weights, which usually yields better class‑wise performance on high‑dimensional TF‑IDF features while keeping the overall pipeline unchanged. This small change should raise the Quadratic Weighted Kappa toward the target score without altering any core BERT logic.'

# 9. Code solution

## === cell 0
from datasets import Dataset
import pandas as pd
import numpy as np
import re
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC  # use LinearSVC for a stronger fallback classifier



## === cell 1
model_path = "/kaggle/input/bert-baseline-train/output/bert-base"
MAX_LEN = 512  # BERT's max token length

train_df = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)
test_df = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)


## === cell 2
_whitespace_re = re.compile(r"\s+")
_non_alnum_re = re.compile(r"[^a-zA-Z0-9]")


def clean_text_series(series: pd.Series) -> pd.Series:
    series = series.str.replace(_whitespace_re, " ", regex=True)
    series = series.str.replace(_non_alnum_re, " ", regex=True)
    return series.str.strip()


train_df["full_text"] = clean_text_series(train_df["full_text"])
test_df["full_text"] = clean_text_series(test_df["full_text"])


## === cell 3
use_fallback = False
tokenizer = None
model = None

if os.path.isdir(model_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True
        )
    except Exception as e:
        print(f"Failed to load local BERT model: {e}. Will use fallback model.")
        use_fallback = True
else:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=6,
            problem_type="single_label_classification",
            local_files_only=True,
        )
    except Exception as e:
        print(f"Failed to load pretrained BERT: {e}. Will use fallback model.")
        use_fallback = True

if not use_fallback:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    torch.set_num_threads(os.cpu_count() or 8)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        model.half()
else:
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
    )
    X_train = vectorizer.fit_transform(train_df["full_text"])
    y_train = train_df["score"]  # scores are 1‑6
    clf = LinearSVC(class_weight="balanced")
    clf.fit(X_train, y_train)


## === cell 4
batch_size = 128
texts = test_df["full_text"].tolist()
scores_np = np.empty(len(texts), dtype=np.int32)

if not use_fallback:
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    input_ids_all = encodings["input_ids"]
    attention_mask_all = encodings["attention_mask"]

    with torch.inference_mode():
        for start in range(0, input_ids_all.size(0), batch_size):
            end = min(start + batch_size, input_ids_all.size(0))
            batch_input_ids = input_ids_all[start:end].to(device, non_blocking=True)
            batch_attention_mask = attention_mask_all[start:end].to(
                device, non_blocking=True
            )

            if device.type == "cuda":
                batch_input_ids = batch_input_ids.half()
                batch_attention_mask = batch_attention_mask.half()

            outputs = model(
                input_ids=batch_input_ids, attention_mask=batch_attention_mask
            )
            batch_scores = torch.argmax(outputs.logits, dim=1).cpu().numpy() + 1
            scores_np[start:end] = batch_scores
else:
    X_test = vectorizer.transform(texts)
    batch_predictions = clf.predict(X_test)
    scores_np = batch_predictions.astype(np.int32)


## === cell 5
sub = test_df[["essay_id"]].copy()
sub["score"] = scores_np
sub.to_csv("submission.csv", index=False)
display(sub.head())
