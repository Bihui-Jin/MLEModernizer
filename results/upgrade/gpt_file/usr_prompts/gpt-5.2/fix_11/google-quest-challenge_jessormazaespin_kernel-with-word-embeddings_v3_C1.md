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
Given questions and answers from various StackExchange properties, predict target values of 30 labels for each question-answer pair.

## Metric
Mean column-wise Spearman's correlation coefficient. The Spearman's rank correlation is computed for each target column, and the mean of these values is calculated for the submission score.

## Submission Format
For each qa_id in the test set, you must predict a probability for each target variable. The predictions should be in the range [0,1]. The file should contain a header and have the following format:

```
qa_id,question_asker_intent_understanding,...,answer_well_written
6,0.0,...,0.5
8,0.5,...,0.1
18,1.0,...,0.0
etc.
```

## Dataset
The list of 30 target labels are the same as the column names in the `sample_submission.csv` file. Target labels with the prefix `question_` relate to the `question_title` and/or `question_body` features in the data. Target labels with the prefix `answer_` relate to the `answer` feature.

Target labels are aggregated from multiple raters, and can have continuous values in the range `[0,1]`. Therefore, predictions must also be in that range.

- **train.csv** - the training data (target labels are the last 30 columns)
- **test.csv** - the test set (you must predict 30 labels for each test set row)
- **sample_submission.csv** - a sample submission file in the correct format; column names are the 30 target labels

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        input/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        working/
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
```

-> data/google-quest-challenge/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/google-quest-challenge/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/google-quest-challenge/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> data/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.2265280693542634

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.18739) has done: 'I fix the environment/runtime issues by removing the broken `tqdm_notebook` import and avoiding the unavailable external model/tokenizer files, which currently prevent any submission from being created. To preserve the core “tokenize text → pad sequences → neural net predicts 30 targets” logic, I train the same style Keras model directly from the provided `train.csv` and then run inference on `test.csv`. I also make padding length deterministic (based on training sequence lengths) so train/test shapes match, and I write the submission using `sample_submission.csv` to guarantee correct column names/order and a valid `.csv` file. This should run end-to-end within the Kaggle environment and yield a reasonable score toward the target.'
- What this solution (achieved 0.19136) has done: 'I fix the protobuf/TensorFlow crash that happens at import time by forcing TensorFlow to use the pure‑Python protobuf implementation (a common Kaggle notebook workaround for the `MessageFactory.GetPrototype` error). Then I keep your existing tokenization, padding, and multi-branch BiGRU model intact, but switch the loss from `binary_crossentropy` to `mean_squared_error`, which better matches the continuous [0,1] targets and typically improves Spearman correlation with minimal semantic change. Finally, I ensure the script still writes a correctly formatted `submission.csv` with the exact `sample_submission.csv` column order.'
- What this solution (achieved 0.17442) has done: 'The crash happens before any CSVs are read because the TensorFlow/protobuf combo in this environment is still trying to call a removed `MessageFactory.GetPrototype`. I fix this by forcing the pure‑Python protobuf implementation *and* disabling the C++ protobuf backend via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`, and by ensuring these env vars are set before TensorFlow (or anything that might import protobuf) loads. I also add a safe fallback that sets `TF_USE_LEGACY_KERAS=1` (harmless if unused) to improve TF/Keras import stability in Kaggle runtimes. No changes are made to tokenization, padding, model architecture, training loop, or submission formatting—this is runtime-stability only so you can generate a valid `submission.csv` and then iterate on score.'
- What this solution (achieved 0.17442) has done: 'I fix the TensorFlow/protobuf import crash by ensuring the environment variables are set before any protobuf/TensorFlow-related imports and by importing protobuf early to force the pure-Python implementation. Then I keep your tokenization, padding, model architecture, and training loop intact, only adding a deterministic Spearman-friendly post-processing step: rank-based normalization per target column (a monotonic transform) to better match the evaluation metric without changing the model itself. Finally, I keep submission formatting anchored to `sample_submission.csv` column order and write a valid `submission.csv`.'
- What this solution (achieved 0.17442) has done: 'You’re failing before any data is read due to a TensorFlow/protobuf incompatibility (`MessageFactory.GetPrototype`) that the current env-var workaround isn’t reliably preventing. I make the protobuf fallback more robust by forcing the pure-Python protobuf implementation **and** uninstalling/avoiding the C++ backend at runtime when possible, and I switch imports to `tf.keras` only after protobuf is guaranteed initialized. Then I keep your existing tokenization, padding, BiGRU architecture, and training loop intact, and keep the rank-normalization post-processing (monotonic, Spearman-friendly) to move the score upward toward the target. Finally, I ensure the submission is written with correct column order and `qa_id` alignment.'
- What this solution (achieved 0.17442) has done: 'The current failure happens before any data is loaded due to a TensorFlow/protobuf incompatibility (`MessageFactory.GetPrototype`) that is not reliably avoided by environment variables alone in this runtime. I fix this by ensuring the pure-Python protobuf implementation is forced *and* by pinning protobuf to the Python implementation before TensorFlow is imported (including an early import of `google.protobuf`), plus a safe fallback to import `tensorflow.compat.v1` first if needed. I keep your tokenization, padding, BiGRU architecture, training loop, loss, and rank-normalization post-processing unchanged to preserve semantics while restoring end-to-end execution. The script still write a valid `submission.csv` with the exact `sample_submission.csv` column order and correct `qa_id` alignment.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

PATH = "../input/google-quest-challenge/"
train_path = os.path.join(PATH, "train.csv")
test_path = os.path.join(PATH, "test.csv")
sample_path = os.path.join(PATH, "sample_submission.csv")

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

print(df_train.shape, df_test.shape, sample_sub.shape)
print("First columns:", df_train.columns[:12].tolist())



## === cell 1
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

text_cols = ["question_title", "question_body", "answer"]

target_cols = sample_sub.columns.tolist()
target_cols.remove("qa_id")
assert len(target_cols) == 30, f"Expected 30 targets, got {len(target_cols)}"

for c in text_cols:
    df_train[c] = df_train[c].fillna("").astype(str)
    df_test[c] = df_test[c].fillna("").astype(str)

y = df_train[target_cols].astype(np.float32).values

train_text = (
    "title: "
    + df_train["question_title"]
    + " body: "
    + df_train["question_body"]
    + " answer: "
    + df_train["answer"]
).tolist()

test_text = (
    "title: "
    + df_test["question_title"]
    + " body: "
    + df_test["question_body"]
    + " answer: "
    + df_test["answer"]
).tolist()

vectorizer = HashingVectorizer(
    n_features=2**18,
    alternate_sign=False,
    norm="l2",
    ngram_range=(1, 2),
    lowercase=True,
)

X_train = vectorizer.transform(train_text)
X_test = vectorizer.transform(test_text)

print("Vectorized shapes:", X_train.shape, X_test.shape)



## === cell 2
base = Ridge(alpha=2.0, random_state=SEED, solver="sag")

model = MultiOutputRegressor(base, n_jobs=1)

n = X_train.shape[0]
idx = np.arange(n)
np.random.shuffle(idx)
val_size = int(0.1 * n)
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

X_trn, y_trn = X_train[trn_idx], y[trn_idx]
X_val, y_val = X_train[val_idx], y[val_idx]

model.fit(X_trn, y_trn)

val_pred = model.predict(X_val).astype(np.float32)
val_pred = np.clip(val_pred, 0.0, 1.0)
print(
    "Val pred stats:",
    float(val_pred.min()),
    float(val_pred.max()),
    float(val_pred.mean()),
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)
    114             try:
--> 115                 coef, info = sp_linalg.cg(C, y_column, tol=tol, atol="legacy")
    116             except TypeError:

TypeError: cg() got an unexpected keyword argument 'tol'

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1889194261.py in <cell line: 0>()
     17 X_val, y_val = X_train[val_idx], y[val_idx]
     18 
---> 19 model.fit(X_trn, y_trn)
     20 
     21 val_pred = model.predict(X_val).astype(np.float32)

/usr/local/lib/python3.11/dist-packages/sklearn/multioutput.py in fit(self, X, y, sample_weight, **fit_params)
    214         fit_params_validated = _check_fit_params(X, fit_params)
    215 
--> 216         self.estimators_ = Parallel(n_jobs=self.n_jobs)(
    217             delayed(_fit_estimator)(
    218                 self.estimator, X, y[:, i], sample_weight, **fit_params_validated

/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py in __call__(self, iterable)
     61             for delayed_func, args, kwargs in iterable
     62         )
---> 63         return super().__call__(iterable_with_config)
     64 
     65 

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in __call__(self, iterable)
   1984             output = self._get_sequential_output(iterable)
   1985             next(output)
-> 1986             return output if self.return_generator else list(output)
   1987 
   1988         # Let's create an ID that uniquely identifies the current call. If the

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _get_sequential_output(self, iterable)
   1912                 self.n_dispatched_batches += 1
   1913                 self.n_dispatched_tasks += 1
-> 1914                 res = func(*args, **kwargs)
   1915                 self.n_completed_tasks += 1
   1916                 self.print_progress()

/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py in __call__(self, *args, **kwargs)
    121             config = {}
    122         with config_context(**config):
--> 123             return self.function(*args, **kwargs)

/usr/local/lib/python3.11/dist-packages/sklearn/multioutput.py in _fit_estimator(estimator, X, y, sample_weight, **fit_params)
     47         estimator.fit(X, y, sample_weight=sample_weight, **fit_params)
     48     else:
---> 49         estimator.fit(X, y, **fit_params)
     50     return estimator
     51 

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
    116             except TypeError:
    117                 # old scipy
--> 118                 coef, info = sp_linalg.cg(C, y_column, tol=tol)
    119             coefs[i] = X1.rmatvec(coef)
    120         else:

TypeError: cg() got an unexpected keyword argument 'tol'

## === cell 3
test_pred = model.predict(X_test).astype(np.float32)
test_pred = np.clip(test_pred, 0.0, 1.0)


def rank_normalize_cols(arr):
    arr = np.asarray(arr)
    n_rows, n_cols = arr.shape
    out = np.empty_like(arr, dtype=np.float32)
    denom = max(1, n_rows - 1)
    for j in range(n_cols):
        order = np.argsort(arr[:, j], kind="mergesort")
        ranks = np.empty(n_rows, dtype=np.int32)
        ranks[order] = np.arange(n_rows, dtype=np.int32)
        out[:, j] = ranks.astype(np.float32) / float(denom)
    return out


test_pred = rank_normalize_cols(test_pred)
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample_sub.copy()
submission["qa_id"] = df_test["qa_id"].values
for i, c in enumerate(target_cols):
    submission[c] = test_pred[:, i]

submission = submission[["qa_id"] + target_cols]

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Submission shape:", submission.shape)
assert submission.shape[0] == df_test.shape[0]
assert submission.shape[1] == 31
assert out_path.endswith(".csv")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/2476606245.py in <cell line: 0>()
----> 1 test_pred = model.predict(X_test).astype(np.float32)
      2 test_pred = np.clip(test_pred, 0.0, 1.0)
      3 
      4 
      5 def rank_normalize_cols(arr):

/usr/local/lib/python3.11/dist-packages/sklearn/multioutput.py in predict(self, X)
    242             Note: Separate models are generated for each predictor.
    243         """
--> 244         check_is_fitted(self)
    245         if not hasattr(self.estimators_[0], "predict"):
    246             raise ValueError("The base estimator should implement a predict method")

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This MultiOutputRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
