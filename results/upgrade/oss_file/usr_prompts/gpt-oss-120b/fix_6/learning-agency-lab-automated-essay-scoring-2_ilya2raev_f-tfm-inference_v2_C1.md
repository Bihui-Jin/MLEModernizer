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

0.8030895862887903

# 6. Current score

0.5662

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.34922) has done: 'I add a safe fallback that loads a HuggingFace model only if the local path is valid; otherwise it trains a lightweight TF‑IDF + LinearRegression model on the training data and uses it for inference. This fixes the import/name errors, ensures a tokenizer/model are defined, and guarantees a `submission.csv` with the correct columns is written. The fallback keeps the original workflow structure while providing a functional baseline that produce a reasonable score.'
- What this solution (achieved 0.60397) has done: 'We avoid the protobuf import error by moving all `transformers` imports inside the try‑except that loads the model, so the fallback TF‑IDF pipeline runs even when the transformer libraries break.  
In the fallback we replace the simple `LinearRegression` with a `LinearSVC` classifier, which better matches the 1‑6 ordinal labels and typically yields a much higher quadratic weighted kappa.  
Minor tweaks to the final post‑processing ensure scores stay in the required 1‑6 range and the submission CSV is correctly written.'
- What this solution (achieved 0.0) has done: 'I replace the fallback LinearSVC with a combination of LinearSVC and a Ridge regression model trained on TF‑IDF features, then average their predictions before clipping and rounding. This keeps the original pipeline while giving a stronger ordinal‑regression signal, which should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.64768) has done: 'Implemented a minimal fix to the fallback pipeline: removed the Ridge regression (which caused a SciPy cg signature error) and kept only a balanced LinearSVC model for TF‑IDF features. Added `class_weight='balanced'` to improve handling of imbalanced score classes. The rest of the workflow remains unchanged, and the script now runs end‑to‑end, writing a valid `submission.csv` with correct columns.'
- What this solution (achieved 0.5662) has done: 'Implemented a calibrated LinearSVC in the fallback pipeline to produce probabilistic predictions, then converted these to expected score values before averaging. This modest enhancement retains the original TF‑IDF + LinearSVC approach while providing smoother, ordinal‑aware predictions that typically raise the quadratic weighted kappa toward the target. No other logic or file handling was altered.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import cohen_kappa_score
from scipy.special import softmax

import torch




## === cell 1
df_train = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)
df_test = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
sample_submission = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)




## === cell 2
_fallback = False
MODEL_NAME = "/kaggle/input/f-tfm-small/funnel-small-ft_ver4"  # may be missing
MAX_LENGTH = 3072
BATCH_SIZE = 2

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        TrainingArguments,
        Trainer,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=6, local_files_only=True
    )
except Exception as e:
    print(f"Transformer load failed ({e}), switching to fallback model.")
    _fallback = True
    tokenizer = None
    model = None




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
if not _fallback:

    def tokenize(sample):
        return tokenizer(sample["full_text"], max_length=MAX_LENGTH, truncation=True)

else:

    def tokenize(sample):
        return {"full_text": sample["full_text"]}




## === cell 4
train_ds = Dataset.from_pandas(df_train)

if not _fallback:
    test_ds = (
        Dataset.from_pandas(df_test)
        .map(tokenize, batched=True)
        .remove_columns(["essay_id", "full_text"])
    )
else:
    test_ds = None




## === cell 5
predictions = []
if not _fallback:
    args = TrainingArguments(
        output_dir=".",  # required but not used for training
        per_device_eval_batch_size=BATCH_SIZE,
        report_to="none",
    )
    trainer = Trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    preds = trainer.predict(test_ds).predictions  # logits (n_samples, 6)
    preds_classes = np.argmax(preds, axis=1) + 1
    predictions.append(preds_classes)
else:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    vectorizer = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2), stop_words="english"
    )
    X = vectorizer.fit_transform(df_train["full_text"].astype(str))
    y = df_train["score"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    svc = LinearSVC(class_weight="balanced")
    svc.fit(X_train, y_train)

    calibrator = CalibratedClassifierCV(svc, cv="prefit", method="sigmoid")
    calibrator.fit(X_val, y_val)

    val_proba = calibrator.predict_proba(X_val)
    val_pred_classes = np.argmax(val_proba, axis=1) + 1
    val_kappa_svc = cohen_kappa_score(y_val, val_pred_classes, weights="quadratic")
    print(f"Fallback Calibrated LinearSVC validation QWK: {val_kappa_svc:.4f}")

    X_test = vectorizer.transform(df_test["full_text"].astype(str))
    test_proba = calibrator.predict_proba(X_test)

    class_labels = np.arange(1, 7)  # scores 1‑6
    test_pred_continuous = np.dot(test_proba, class_labels)

    predictions.append(test_pred_continuous)




## === cell 6
preds_array = np.mean(predictions, axis=0)




## === cell 7
df_test["score"] = np.round(np.clip(preds_array, 1, 6)).astype(int)

submission = df_test[["essay_id", "score"]]
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
