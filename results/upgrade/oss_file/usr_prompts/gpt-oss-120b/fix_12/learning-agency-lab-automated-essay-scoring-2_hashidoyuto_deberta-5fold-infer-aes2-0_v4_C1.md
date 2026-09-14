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

Not yielded

# 7. Whether higher score is better

Higher is better

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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1, 2), stop_words="english")
tfidf_train = vectorizer.fit_transform(train_df["full_text"])
ridge = Ridge(alpha=1.0, fit_intercept=True, random_state=42)
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

val_pred_matrix = np.column_stack([val_bucket_pred, val_len_pred, val_feat_pred])
raw_weights, *_ = np.linalg.lstsq(val_pred_matrix, val_df["score"].values, rcond=None)
raw_weights = np.clip(raw_weights, 0, None)
if raw_weights.sum() == 0:
    raw_weights = np.array([1 / 3, 1 / 3, 1 / 3])
else:
    raw_weights = raw_weights / raw_weights.sum()
W_BUCKET, W_LEN, W_FEAT = raw_weights.tolist()

val_blended = W_BUCKET * val_bucket_pred + W_LEN * val_len_pred + W_FEAT * val_feat_pred
best_shift = 0.0
best_kappa = -np.inf
for shift in np.arange(-2.0, 2.001, 0.001):
    preds = np.clip(np.round(val_blended + shift), 1, 6).astype(int)
    kappa = cohen_kappa_score(val_df["score"], preds, weights="quadratic")
    if kappa > best_kappa:
        best_kappa = kappa
        best_shift = shift
CALIBRATION_SHIFT = best_shift




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)
    128             try:
--> 129                 coefs[i], info = sp_linalg.cg(
    130                     C, y_column, maxiter=max_iter, tol=tol, atol="legacy"

TypeError: cg() got an unexpected keyword argument 'tol'

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3193727822.py in <cell line: 0>()
     70 tfidf_train = vectorizer.fit_transform(train_df["full_text"])
     71 ridge = Ridge(alpha=1.0, fit_intercept=True, random_state=42)
---> 72 ridge.fit(tfidf_train, y_train)
     73 
     74 # length‑vs‑score linear fit (unchanged)

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in fit(self, X, y, sample_weight)
   1132             y_numeric=True,
   1133         )
-> 1134         return super().fit(X, y, sample_weight=sample_weight)
   1135 
   1136 

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in fit(self, X, y, sample_weight)
    898                 params = {}
    899 
--> 900             self.coef_, self.n_iter_ = _ridge_regression(
    901                 X,
    902                 y,

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _ridge_regression(X, y, alpha, sample_weight, solver, max_iter, tol, verbose, positive, random_state, return_n_iter, return_intercept, X_scale, X_offset, check_input, fit_intercept)
    669     n_iter = None
    670     if solver == "sparse_cg":
--> 671         coef = _solve_sparse_cg(
    672             X,
    673             y,

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)
    132             except TypeError:
    133                 # old scipy
--> 134                 coefs[i], info = sp_linalg.cg(C, y_column, maxiter=max_iter, tol=tol)
    135 
    136         if info < 0:

TypeError: cg() got an unexpected keyword argument 'tol'

## === cell 3
def predict_score(text):
    ln = len(text)
    bucket = (ln // bucket_size) * bucket_size
    row = bucket_means[bucket_means["len_bucket"] == bucket]
    bucket_pred = row["mean_score"].values[0] if not row.empty else global_mean

    lin_pred_len = intercept + slope * ln

    tfidf_vec = vectorizer.transform([text])
    lin_pred_feat = ridge.predict(tfidf_vec)[0]

    blended = (
        W_BUCKET * bucket_pred
        + W_LEN * lin_pred_len
        + W_FEAT * lin_pred_feat
        + CALIBRATION_SHIFT
    )
    pred = int(round(blended))
    pred = max(1, min(6, pred))
    return pred


test_df["pred_score"] = test_df["full_text"].apply(predict_score)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1258006838.py in <cell line: 0>()
     22 
     23 
---> 24 test_df["pred_score"] = test_df["full_text"].apply(predict_score)
     25 
     26 

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in apply(self, func, convert_dtype, args, by_row, **kwargs)
   4922             args=args,
   4923             kwargs=kwargs,
-> 4924         ).apply()
   4925 
   4926     def _reindex_indexer(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
   1425 
   1426         # self.func is Callable
-> 1427         return self.apply_standard()
   1428 
   1429     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1505         #  Categorical (GH51645).
   1506         action = "ignore" if isinstance(obj.dtype, CategoricalDtype) else None
-> 1507         mapped = obj._map_values(
   1508             mapper=curried, na_action=action, convert=self.convert_dtype
   1509         )

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _map_values(self, mapper, na_action, convert)
    919             return arr.map(mapper, na_action=na_action)
    920 
--> 921         return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
    922 
    923     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/algorithms.py in map_array(arr, mapper, na_action, convert)
   1741     values = arr.astype(object, copy=False)
   1742     if na_action is None:
-> 1743         return lib.map_infer(values, mapper, convert=convert)
   1744     else:
   1745         return lib.map_infer_mask(

lib.pyx in pandas._libs.lib.map_infer()

/tmp/ipykernel_11/1258006838.py in predict_score(text)
      5     bucket_pred = row["mean_score"].values[0] if not row.empty else global_mean
      6 
----> 7     lin_pred_len = intercept + slope * ln
      8 
      9     # TF‑IDF ridge prediction for this single essay

NameError: name 'intercept' is not defined

## === cell 4
submission = pd.DataFrame(
    {"essay_id": test_df["essay_id"], "score": test_df["pred_score"]}
)
submission.to_csv(PATHS.submission_path, index=False)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'pred_score'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1528415314.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"essay_id": test_df["essay_id"], "score": test_df["pred_score"]}
      3 )
      4 submission.to_csv(PATHS.submission_path, index=False)
      5 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'pred_score'

## === cell 5
print(submission.head())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2388794795.py in <cell line: 0>()
----> 1 print(submission.head())

NameError: name 'submission' is not defined
