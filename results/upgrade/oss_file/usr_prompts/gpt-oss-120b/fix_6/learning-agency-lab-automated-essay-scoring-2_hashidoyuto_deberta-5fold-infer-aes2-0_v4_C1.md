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

0.8053301059475746

# 6. Current score

0.60472

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.60922) has done: 'I added a protobuf monkey‑patch before importing transformers to prevent the `MessageFactory` error, then replaced the missing‑model inference pipeline with a lightweight baseline that predicts scores based on essay length buckets derived from the training data. This eliminates the file‑not‑found issue, ensures a valid `submission.csv` is written, and keeps the core logic minimal while still producing reasonable predictions.'
- What this solution (achieved 0.60472) has done: 'I keep the existing length‑bucket baseline but add a tiny linear‑regression correction based on essay length. This uses the same features already computed, introduces only a few lines, and should raise the quadratic weighted kappa toward the target without overhauling the model.'
- What this solution (achieved 0.6008) has done: 'I added a lightweight linear‑regression model that uses a few easy text features (character length, average word length, and sentence count) and blended its prediction with the existing length‑bucket baseline. The regression coefficients are fitted on the training set with NumPy’s least‑squares, keeping the core logic unchanged while providing a richer signal that should raise the quadratic weighted kappa toward the target. The final prediction is still rounded and clipped to the 1‑6 limits, and the script writes a valid `submission.csv`.'
- What this solution (achieved 0.60472) has done: 'I add a quick validation split and a small grid search to find optimal blending weights for the three simple predictors (bucket mean, length‑linear, and feature‑linear). The best weights are stored in global variables and used in the `predict_score` function, replacing the fixed 0.4/0.3/0.3 blend. This keeps the core logic unchanged while calibrating the ensemble to raise the quadratic weighted kappa toward the target.'

# 9. Code solution

## === cell 0
import os, sys, warnings

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def _get_prototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        message_factory.MessageFactory.GetPrototype = _get_prototype
except Exception:
    pass  # If protobuf isn’t present, we’ll still proceed without transformers

warnings.simplefilter("ignore")
import numpy as np
import pandas as pd




## === cell 1
class PATHS:
    train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
    test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
    submission_path = "submission.csv"




## === cell 2
train_df = pd.read_csv(PATHS.train_path)
test_df = pd.read_csv(PATHS.test_path)

train_df["length"] = train_df["full_text"].str.len()
bucket_size = 50
train_df["len_bucket"] = (train_df["length"] // bucket_size) * bucket_size

bucket_means = (
    train_df.groupby("len_bucket")["score"]
    .mean()
    .reset_index()
    .rename(columns={"score": "mean_score"})
)

global_mean = train_df["score"].mean()


def compute_features(text_series):
    """
    Return a NumPy array of shape (n_samples, 3) containing:
    1) length (character count)
    2) average word length
    3) sentence count (based on '.' occurrences)
    """
    lengths = text_series.str.len().values.astype(float)

    words = text_series.str.split()
    total_chars = text_series.str.replace(r"\s+", "", regex=True).str.len()
    avg_word_len = (total_chars / words.str.len().replace(0, np.nan)).fillna(0).values

    sentence_cnt = text_series.str.count(r"\.").replace(0, 1).astype(float).values

    return np.column_stack([lengths, avg_word_len, sentence_cnt])


X_train = compute_features(train_df["full_text"])
X_train_aug = np.column_stack([np.ones(X_train.shape[0]), X_train])
y_train = train_df["score"].values.astype(float)

coeffs, *_ = np.linalg.lstsq(X_train_aug, y_train, rcond=None)

slope, intercept = np.polyfit(train_df["length"], train_df["score"], 1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

W_BUCKET, W_LEN, W_FEAT = 0.4, 0.3, 0.3

train_idx, val_idx = train_test_split(
    train_df.index,
    test_size=0.1,
    random_state=42,
    stratify=train_df["score"],
)
val_df = train_df.loc[val_idx].copy()


def _bucket_pred(length_series):
    buckets = (length_series // bucket_size) * bucket_size
    preds = []
    for b in buckets:
        row = bucket_means[bucket_means["len_bucket"] == b]
        pred = row["mean_score"].values[0] if not row.empty else global_mean
        preds.append(pred)
    return np.array(preds)


val_lengths = val_df["length"].values
val_bucket_pred = _bucket_pred(val_lengths)
val_len_pred = intercept + slope * val_lengths

val_feats = compute_features(val_df["full_text"])
val_feat_pred = np.column_stack([np.ones(val_feats.shape[0]), val_feats]) @ coeffs

best_kappa = -np.inf
best_weights = (W_BUCKET, W_LEN, W_FEAT)

weight_options = [0.2, 0.3, 0.4, 0.5, 0.6]
for w1 in weight_options:
    for w2 in weight_options:
        w3 = 1.0 - w1 - w2
        if w3 < 0 or w3 > 1:
            continue
        blended = w1 * val_bucket_pred + w2 * val_len_pred + w3 * val_feat_pred
        pred = np.rint(blended).astype(int)
        pred = np.clip(pred, 1, 6)
        kappa = cohen_kappa_score(val_df["score"], pred, weights="quadratic")
        if kappa > best_kappa:
            best_kappa = kappa
            best_weights = (w1, w2, w3)

W_BUCKET, W_LEN, W_FEAT = best_weights


def predict_score(text):
    ln = len(text)
    bucket = (ln // bucket_size) * bucket_size
    row = bucket_means[bucket_means["len_bucket"] == bucket]
    bucket_pred = row["mean_score"].values[0] if not row.empty else global_mean

    lin_pred_len = intercept + slope * ln

    feats = np.array(
        [
            1.0,
            ln,
            (len(text.replace(" ", "")) / max(1, len(text.split()))),
            max(1, text.count(".")),
        ]
    )
    lin_pred_feat = feats @ coeffs

    blended = W_BUCKET * bucket_pred + W_LEN * lin_pred_len + W_FEAT * lin_pred_feat
    pred = int(round(blended))
    pred = max(1, min(6, pred))
    return pred


test_df["pred_score"] = test_df["full_text"].apply(predict_score)




## === cell 3
submission = pd.DataFrame(
    {"essay_id": test_df["essay_id"], "score": test_df["pred_score"]}
)
submission.to_csv(PATHS.submission_path, index=False)




## === cell 4
print(submission.head())
