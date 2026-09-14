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
Predict the sentiment of phrases.

## Metric
Classification accuracy.

## Submission Format
For each phrase in the test set, predict a label for the sentiment. Your submission should have a header and look like the following:

```
PhraseId,Sentiment
156061,2
156062,2
156063,2
...
```

## Dataset
The dataset is comprised of tab-separated files with phrases. Each phrase has a PhraseId. Each sentence has a SentenceId.

The sentiment labels are:

0 - negative

1 - somewhat negative

2 - neutral

3 - somewhat positive

4 - positive

# 2. Python version

3.7

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        input/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        working/
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
```

-> data/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> data/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> (stopped after 10 files for performance)

# 5. Target score

0.64579

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random

import numpy as np
import pandas as pd

from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MaxAbsScaler

np.random.seed(0)
random.seed(0)

SENTIMENT_LABELS = [
    "negative",
    "somewhat negative",
    "neutral",
    "somewhat positive",
    "positive",
]

print("Environment ready (sklearn-based fallback: no TF/Hub required).")




## === cell 1
def add_readable_labels_column(df, sentiment_value_column):
    df["SentimentLabel"] = df[sentiment_value_column].replace(
        range(5), SENTIMENT_LABELS
    )


def get_data(validation_set_ratio=0.1):
    train_path = "/kaggle/input/movie-review-sentiment-analysis-kernels-only/train.tsv"
    test_path = "/kaggle/input/movie-review-sentiment-analysis-kernels-only/test.tsv"

    if not os.path.exists(train_path):
        train_path = "/kaggle/input/train.tsv"
    if not os.path.exists(test_path):
        test_path = "/kaggle/input/test.tsv"

    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")

    add_readable_labels_column(train_df, "Sentiment")

    train_indices, validation_indices = model_selection.train_test_split(
        np.unique(train_df["SentenceId"]),
        test_size=validation_set_ratio,
        random_state=0,
    )

    validation_df = train_df[train_df["SentenceId"].isin(validation_indices)].copy()
    train_df = train_df[train_df["SentenceId"].isin(train_indices)].copy()

    print(
        "Split the training data into %d training and %d validation examples."
        % (len(train_df), len(validation_df))
    )

    return train_df, validation_df, test_df


train_df, validation_df, test_df = get_data()
train_df.head()




## === cell 2
X_train = train_df["Phrase"].astype(str).fillna("")
y_train = train_df["Sentiment"].astype(int).values

X_val = validation_df["Phrase"].astype(str).fillna("")
y_val = validation_df["Sentiment"].astype(int).values

X_test = test_df["Phrase"].astype(str).fillna("")

model = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=50000,
                min_df=2,
                strip_accents="unicode",
                lowercase=True,
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=False,
                norm="l2",
                dtype=np.float32,
            ),
        ),
        ("scale", MaxAbsScaler(copy=False)),
        (
            "clf",
            MLPClassifier(
                hidden_layer_sizes=(250, 50),
                activation="relu",
                solver="adam",  # stable + fast; analogous to adaptive optimizers
                alpha=1e-4,  # mild regularization (dropout analog)
                batch_size=256,
                learning_rate_init=0.003,
                max_iter=20,  # fixed training budget; no early stopping
                shuffle=True,
                random_state=0,
                verbose=False,
                early_stopping=False,
                n_iter_no_change=10,  # kept explicit for determinism/clarity
                tol=1e-4,  # kept explicit for determinism/clarity
            ),
        ),
    ]
)

model.fit(X_train, y_train)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1135840344.py in <cell line: 0>()
     48 )
     49 
---> 50 model.fit(X_train, y_train)
     51 
     52 

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in fit(self, X, y, **fit_params)
    399         """
    400         fit_params_steps = self._check_fit_params(**fit_params)
--> 401         Xt = self._fit(X, y, **fit_params_steps)
    402         with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
    403             if self._final_estimator != "passthrough":

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit(self, X, y, **fit_params_steps)
    357                 cloned_transformer = clone(transformer)
    358             # Fit or load from cache the current transformer
--> 359             X, fitted_transformer = fit_transform_one_cached(
    360                 cloned_transformer,
    361                 X,

/usr/local/lib/python3.11/dist-packages/joblib/memory.py in __call__(self, *args, **kwargs)
    324 
    325     def __call__(self, *args, **kwargs):
--> 326         return self.func(*args, **kwargs)
    327 
    328     def call_and_shelve(self, *args, **kwargs):

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit_transform_one(transformer, X, y, weight, message_clsname, message, **fit_params)
    891     with _print_elapsed_time(message_clsname, message):
    892         if hasattr(transformer, "fit_transform"):
--> 893             res = transformer.fit_transform(X, y, **fit_params)
    894         else:
    895             res = transformer.fit(X, y, **fit_params).transform(X)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in fit_transform(self, X, y, **fit_params)
    879         else:
    880             # fit method of arity 2 (supervised transformation)
--> 881             return self.fit(X, y, **fit_params).transform(X)
    882 
    883 

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in fit(self, X, y)
   1167         # Reset internal state before fitting
   1168         self._reset()
-> 1169         return self.partial_fit(X, y)
   1170 
   1171     def partial_fit(self, X, y=None):

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in partial_fit(self, X, y)
   1202 
   1203         if sparse.issparse(X):
-> 1204             mins, maxs = min_max_axis(X, axis=0, ignore_nan=True)
   1205             max_abs = np.maximum(np.abs(mins), np.abs(maxs))
   1206         else:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py in min_max_axis(X, axis, ignore_nan)
    504     if isinstance(X, (sp.csr_matrix, sp.csc_matrix)):
    505         if ignore_nan:
--> 506             return _sparse_nan_min_max(X, axis=axis)
    507         else:
    508             return _sparse_min_max(X, axis=axis)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py in _sparse_nan_min_max(X, axis)
    472 
    473 def _sparse_nan_min_max(X, axis):
--> 474     return (_sparse_min_or_max(X, axis, np.fmin), _sparse_min_or_max(X, axis, np.fmax))
    475 
    476 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py in _sparse_min_or_max(X, axis, min_or_max)
    459         axis += 2
    460     if (axis == 0) or (axis == 1):
--> 461         return _min_or_max_axis(X, axis, min_or_max)
    462     else:
    463         raise ValueError("invalid axis, use 0 for rows, or 1 for columns")

/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py in _min_or_max_axis(X, axis, min_or_max)
    442             (value, (major_index, np.zeros(len(value)))), dtype=X.dtype, shape=(M, 1)
    443         )
--> 444     return res.A.ravel()
    445 
    446 

AttributeError: 'coo_matrix' object has no attribute 'A'

## === cell 3
train_acc = float(model.score(X_train, y_train))
val_acc = float(model.score(X_val, y_val))

print(f"Training set accuracy: {train_acc:.6f}")
print(f"Validation set accuracy: {val_acc:.6f}")




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/935459423.py in <cell line: 0>()
      1 # Evaluate on train/validation using the fitted pipeline (avoids manual transforms mismatch)
----> 2 train_acc = float(model.score(X_train, y_train))
      3 val_acc = float(model.score(X_val, y_val))
      4 
      5 print(f"Training set accuracy: {train_acc:.6f}")

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in score(self, X, y, sample_weight)
    716         Xt = X
    717         for _, name, transform in self._iter(with_final=False):
--> 718             Xt = transform.transform(Xt)
    719         score_params = {}
    720         if sample_weight is not None:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in transform(self, X)
   1241 
   1242         if sparse.issparse(X):
-> 1243             inplace_column_scale(X, 1.0 / self.scale_)
   1244         else:
   1245             X /= self.scale_

AttributeError: 'MaxAbsScaler' object has no attribute 'scale_'

## === cell 4
test_pred = model.predict(X_test).astype(int)

test_df["Sentiment"] = test_pred
submission = test_df[["PhraseId", "Sentiment"]].copy()
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Submission columns:", submission.columns.tolist())
print("Sentiment value counts:\n", submission["Sentiment"].value_counts().sort_index())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2749170566.py in <cell line: 0>()
----> 1 test_pred = model.predict(X_test).astype(int)
      2 
      3 test_df["Sentiment"] = test_pred
      4 submission = test_df[["PhraseId", "Sentiment"]].copy()
      5 submission.to_csv("submission.csv", index=False)

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in predict(self, X, **predict_params)
    478         Xt = X
    479         for _, name, transform in self._iter(with_final=False):
--> 480             Xt = transform.transform(Xt)
    481         return self.steps[-1][1].predict(Xt, **predict_params)
    482 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in transform(self, X)
   1241 
   1242         if sparse.issparse(X):
-> 1243             inplace_column_scale(X, 1.0 / self.scale_)
   1244         else:
   1245             X /= self.scale_

AttributeError: 'MaxAbsScaler' object has no attribute 'scale_'
