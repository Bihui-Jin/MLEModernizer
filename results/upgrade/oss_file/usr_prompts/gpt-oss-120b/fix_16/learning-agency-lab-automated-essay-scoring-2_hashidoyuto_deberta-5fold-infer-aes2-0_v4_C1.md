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

0.6818

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.60922) has done: 'I added a protobuf monkey‑patch before importing transformers to prevent the `MessageFactory` error, then replaced the missing‑model inference pipeline with a lightweight baseline that predicts scores based on essay length buckets derived from the training data. This eliminates the file‑not‑found issue, ensures a valid `submission.csv` is written, and keeps the core logic minimal while still producing reasonable predictions.'
- What this solution (achieved 0.60472) has done: 'I keep the existing length‑bucket baseline but add a tiny linear‑regression correction based on essay length. This uses the same features already computed, introduces only a few lines, and should raise the quadratic weighted kappa toward the target without overhauling the model.'
- What this solution (achieved 0.6008) has done: 'I added a lightweight linear‑regression model that uses a few easy text features (character length, average word length, and sentence count) and blended its prediction with the existing length‑bucket baseline. The regression coefficients are fitted on the training set with NumPy’s least‑squares, keeping the core logic unchanged while providing a richer signal that should raise the quadratic weighted kappa toward the target. The final prediction is still rounded and clipped to the 1‑6 limits, and the script writes a valid `submission.csv`.'
- What this solution (achieved 0.60472) has done: 'I add a quick validation split and a small grid search to find optimal blending weights for the three simple predictors (bucket mean, length‑linear, and feature‑linear). The best weights are stored in global variables and used in the `predict_score` function, replacing the fixed 0.4/0.3/0.3 blend. This keeps the core logic unchanged while calibrating the ensemble to raise the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.61023) has done: 'I fine‑tune the ensemble that blends the three simple predictors. First, I make the length‑bucket baseline a little more granular by reducing `bucket_size` from 50 to 30, which gives more accurate bucket means. Then, instead of the coarse grid search for blending weights, I fit a least‑squares linear model on the validation set to compute optimal non‑negative weights that sum to 1. This small calibration keeps the original feature set and model logic while expectedly raising the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.61023) has done: 'I add a tiny calibration step that finds an optimal additive shift for the blended continuous prediction on a held‑out validation split. The shift that maximizes quadratic weighted kappa on validation is then applied to every test prediction before rounding and clipping. This small tweak keeps the original feature set and model unchanged while nudging the score upward toward the target.'
- What this solution (achieved 0.62763) has done: 'I add a fourth simple text feature (word count) to the linear‑regression model and refit its coefficients, then broaden and fine‑tune the calibration‑shift search range (‑2 → 2 with 0.01 steps). These small, targeted tweaks keep the original bucket/length/feature ensemble untouched while giving it a richer feature set and a more precise additive calibration, which should raise the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.6185) has done: 'I reduce the bucket granularity from 30 to 15 to give the length‑bucket baseline finer resolution, and I make the calibration‑shift search finer (step 0.001) so the additive adjustment is more precisely tuned. These tiny hyper‑parameter tweaks keep the original logic intact while likely raising the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.59855) has done: 'I make two small, targeted tweaks that keep the original baseline logic while giving the model a richer linear‑regression feature set and a finer length‑bucket granularity.  
1. Reduce `bucket_size` from 15 to 5 so the bucket‑mean baseline can capture more detailed score trends.  
2. Extend the feature extractor with two extra numeric features (length² and log length) and recompute the linear‑regression coefficients accordingly. Both changes are incorporated into the validation‑based weight fitting and the final calibration‑shift, so they should raise the quadratic weighted kappa toward the target without altering the overall pipeline.'
- What this solution (achieved 0.61515) has done: 'Implemented fixes to resolve runtime errors:
- Forced `Ridge` to use the `'lsqr'` solver, avoiding the SciPy CG bug.
- Ensured the linear fit (`slope`, `intercept`) is defined before prediction.
- Adjusted cell ordering to start at 1 as required.
- Added a safety comment for clarity.

The script now runs end‑to‑end, creates a valid `submission.csv`, and retains the original modeling logic.'
- What this solution (achieved 0.64628) has done: 'Implemented a modest but targeted enhancement:
- Raised TF‑IDF dimensionality (`max_features=1000`) to capture richer text signals.
- Added a lightweight grid‑search over blending weights (bucket, length, TF‑IDF) on the validation set to directly maximize quadratic weighted kappa, while keeping the original least‑squares weights as a fallback.
- Retained the existing calibration‑shift search and all other pipeline steps, preserving the core logic and output format.'
- What this solution (achieved 0.64628) has done: 'I add a fourth predictor – a simple linear‑regression model built on the six hand‑crafted numeric features – and include it in the blending ensemble. The existing ridge‑TF‑IDF, length bucket, and length‑linear predictors stay unchanged; I only compute the new predictor for validation (to fit blending weights) and for each test essay (used in the final prediction). If the grid‑search still finds the best three‑weight blend, the new weight defaults to 0, otherwise the fallback least‑squares fit determines a non‑negative normalized weight for the linear‑feature model. This small extension keeps the original pipeline intact while giving it extra signal, which should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.6818) has done: 'I increase the TF‑IDF dimensionality to capture richer text information and replace the coarse grid‑search blending with a direct non‑negative least‑squares fit that includes all four predictors (bucket, length‑linear, TF‑IDF ridge, handcrafted‑feature linear). This keeps the overall pipeline unchanged while providing a more optimal weight calibration, which should raise the quadratic weighted kappa toward the target score.'

# 9. Code solution

## === cell 0
import os, warnings

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def _get_prototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        message_factory.MessageFactory.GetPrototype = _get_prototype
except Exception:
    pass  # Proceed without protobuf if unavailable

warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score




## === cell 1
class PATHS:
    train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
    test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
    submission_path = "submission.csv"


train_df = pd.read_csv(PATHS.train_path)
test_df = pd.read_csv(PATHS.test_path)

bucket_size = 5

train_df["length"] = train_df["full_text"].str.len()
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
    Return a NumPy array (n_samples, 6) with:
    0) character length
    1) average word length
    2) sentence count ('.' occurrences, min 1)
    3) word count
    4) length squared
    5) log(1 + length)
    """
    lengths = text_series.str.len().astype(float).values

    words = text_series.str.split()
    total_chars = text_series.str.replace(r"\s+", "", regex=True).str.len()
    avg_word_len = (
        (total_chars / words.str.len().replace(0, np.nan))
        .fillna(0)
        .astype(float)
        .values
    )

    sentence_cnt = text_series.str.count(r"\.").replace(0, 1).astype(float).values
    word_cnt = words.str.len().astype(float).values

    length_squared = np.square(lengths)
    log_length = np.log1p(lengths)

    return np.column_stack(
        [
            lengths,
            avg_word_len,
            sentence_cnt,
            word_cnt,
            length_squared,
            log_length,
        ]
    )


X_train = compute_features(train_df["full_text"])
X_train_aug = np.column_stack([np.ones(X_train.shape[0]), X_train])
y_train = train_df["score"].astype(float).values
coeffs, *_ = np.linalg.lstsq(X_train_aug, y_train, rcond=None)

vectorizer = TfidfVectorizer(
    max_features=5000, ngram_range=(1, 2), stop_words="english"
)
tfidf_train = vectorizer.fit_transform(train_df["full_text"])
ridge = Ridge(alpha=1.0, fit_intercept=True, random_state=42, solver="lsqr")
ridge.fit(tfidf_train, y_train)

slope, intercept = np.polyfit(train_df["length"], train_df["score"], 1)

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

tfidf_val = vectorizer.transform(val_df["full_text"])
val_feat_pred = ridge.predict(tfidf_val)

X_val = compute_features(val_df["full_text"])
X_val_aug = np.column_stack([np.ones(X_val.shape[0]), X_val])
val_lr_pred = X_val_aug @ coeffs

val_pred_matrix = np.column_stack(
    [val_bucket_pred, val_len_pred, val_feat_pred, val_lr_pred]
)

raw_weights, *_ = np.linalg.lstsq(val_pred_matrix, val_df["score"].values, rcond=None)
raw_weights = np.clip(raw_weights, 0, None)
if raw_weights.sum() == 0:
    raw_weights = np.ones_like(raw_weights) / len(raw_weights)
else:
    raw_weights = raw_weights / raw_weights.sum()
W_BUCKET, W_LEN, W_FEAT, W_LR = raw_weights.tolist()

val_blended = (
    W_BUCKET * val_bucket_pred
    + W_LEN * val_len_pred
    + W_FEAT * val_feat_pred
    + W_LR * val_lr_pred
)
best_shift = 0.0
best_kappa_shift = -np.inf
for shift in np.arange(-2.0, 2.001, 0.001):
    preds = np.clip(np.round(val_blended + shift), 1, 6).astype(int)
    kappa = cohen_kappa_score(val_df["score"], preds, weights="quadratic")
    if kappa > best_kappa_shift:
        best_kappa_shift = kappa
        best_shift = shift
CALIBRATION_SHIFT = best_shift




## === cell 2
def predict_score(text):
    """Predict a single essay score using the calibrated ensemble."""
    ln = len(text)
    bucket = (ln // bucket_size) * bucket_size
    row = bucket_means[bucket_means["len_bucket"] == bucket]
    bucket_pred = row["mean_score"].values[0] if not row.empty else global_mean

    lin_pred_len = intercept + slope * ln
    tfidf_vec = vectorizer.transform([text])
    lin_pred_feat = ridge.predict(tfidf_vec)[0]

    feats = compute_features(pd.Series([text]))[0]
    lin_pred_lr = np.dot(np.r_[1, feats], coeffs)

    blended = (
        W_BUCKET * bucket_pred
        + W_LEN * lin_pred_len
        + W_FEAT * lin_pred_feat
        + W_LR * lin_pred_lr
        + CALIBRATION_SHIFT
    )
    pred = int(round(blended))
    return max(1, min(6, pred))


test_df["pred_score"] = test_df["full_text"].apply(predict_score)



## === cell 3
submission = pd.DataFrame(
    {"essay_id": test_df["essay_id"], "score": test_df["pred_score"]}
)
submission.to_csv(PATHS.submission_path, index=False)



## === cell 4
print("Submission preview:")
print(submission.head())
