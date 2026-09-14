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

0.0003765054558701

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.00364) has done: 'The crash is caused by importing the standalone `keras` package in this Kaggle environment, which can trigger a protobuf incompatibility (`MessageFactory.GetPrototype`). The minimal fix is to use `tensorflow.keras` everywhere (same Keras API, same model/logic) and to set a couple of safe environment flags before TensorFlow import to avoid protobuf implementation issues. I also fixed a logic bug where the test set was tokenized with a *new* tokenizer instead of the one fit on training, which would make inference inconsistent and often fail/produce garbage; this is score-improving while preserving the exact intended pipeline. Finally, I ensured predictions are clipped to `[0,1]` (required by the competition) and that the submission aligns row-wise with `qa_id`.'
- What this solution (achieved nan) has done: 'I fix the TensorFlow/protobuf crash by forcing a compatible protobuf implementation *before* importing TensorFlow, and by avoiding any standalone `keras` imports (keeping everything under `tensorflow.keras`). I also fix a subtle but important logic issue: `train_cols`/`target_cols` were produced from `set()` and thus had nondeterministic order, which can scramble the 30 targets and tank Spearman; I derive `target_cols` in the exact `sample_submission.csv` column order and keep feature columns in stable order. Finally, I keep the existing model/training loop intact, ensure prediction shapes align to 30 labels, clip to `[0,1]`, and always write a valid `submission.csv`.'
- What this solution (achieved nan) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf runtime and disabling C++ protobuf before TensorFlow is imported, which avoids the `MessageFactory.GetPrototype` error in this environment. I keep your model/training logic intact, but make the text preprocessing safe by copying dataframes and avoiding in-place mutation side-effects across splits. I also make prediction handling robust to Keras returning a list/extra dimensions, and ensure the submission uses the exact target column order from `sample_submission.csv`, clips outputs to `[0,1]`, and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(69)



## === cell 1
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from scipy.sparse import hstack, csr_matrix


def cat_to_numeric(category):
    if category == "LIFE_ARTS":
        return 1
    if category == "CULTURE":
        return 2
    if category == "SCIENCE":
        return 3
    if category == "STACKOVERFLOW":
        return 4
    if category == "TECHNOLOGY":
        return 5
    return 0


def prepare_data(frame):
    to_drop = []
    for col in frame.columns:
        if (
            ("user_page" in col)
            or ("host" in col)
            or ("url" in col)
            or ("user_name" in col)
            or ("categ" in col)
        ):
            to_drop.append(col)
    data = frame.drop(to_drop, axis=1)
    return data


def get_vars_and_targets(train_data, test_data, sample_submission_path):
    sub = pd.read_csv(sample_submission_path)
    target_cols = list(sub.columns[1:])  # 30 targets in correct order
    train_cols = [
        c for c in train_data.columns if c not in target_cols and c != "qa_id"
    ]
    return train_cols, target_cols


def get_text_cols(frame):
    text_cols = []
    for col in frame.columns:
        if ("title" in col) or ("body" in col) or (col == "answer"):
            text_cols.append(col)
    return text_cols




## === cell 2
def build_hashed_features(df, text_cols, vectorizers=None):
    df = df.copy()
    for c in text_cols:
        df[c] = df[c].fillna("").astype(str)

    if vectorizers is None:
        vectorizers = {}
        for c in text_cols:
            vectorizers[c] = HashingVectorizer(
                n_features=2**18,
                alternate_sign=False,
                norm="l2",
                ngram_range=(1, 2),
                lowercase=True,
            )

    X_parts = []
    for c in text_cols:
        X_parts.append(vectorizers[c].transform(df[c]))

    if "category" in df.columns:
        cat_num = (
            df["category"]
            .fillna("")
            .map(cat_to_numeric)
            .astype(np.float32)
            .values.reshape(-1, 1)
        )
        X_parts.append(csr_matrix(cat_num))

    X = hstack(X_parts).tocsr()
    return X, vectorizers




## === cell 3
def main():
    train_path = "../input/google-quest-challenge/train.csv"
    test_path = "../input/google-quest-challenge/test.csv"
    sample_sub_path = "../input/google-quest-challenge/sample_submission.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    train_data = prepare_data(train_data)
    test_data = prepare_data(test_data)

    train_cols, target_cols = get_vars_and_targets(
        train_data, test_data, sample_sub_path
    )

    X_train_df, X_valid_df, y_train, y_valid = train_test_split(
        train_data.loc[:, ["qa_id"] + train_cols],
        train_data.loc[:, target_cols],
        test_size=0.25,
        random_state=69,
    )

    text_cols = get_text_cols(X_train_df)
    Xtr, vecs = build_hashed_features(X_train_df, text_cols, vectorizers=None)
    Xva, _ = build_hashed_features(X_valid_df, text_cols, vectorizers=vecs)

    base = Ridge(alpha=2.0, random_state=69, solver="sag")
    model = MultiOutputRegressor(base, n_jobs=-1)

    model.fit(Xtr, y_train.values)
    valid_pred = model.predict(Xva)
    valid_pred = np.clip(valid_pred, 0.0, 1.0)

    print("Validation preds shape:", valid_pred.shape)

    test_ids = test_data["qa_id"].values
    test_x = test_data.loc[:, train_cols].copy()
    test_text_cols = get_text_cols(test_x)  # should match training text cols subset
    Xte, _ = build_hashed_features(test_x, test_text_cols, vectorizers=vecs)

    predictions = model.predict(Xte)
    predictions = np.asarray(predictions)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    if predictions.shape[1] != 30:
        predictions = (
            predictions[:, :30]
            if predictions.shape[1] > 30
            else np.pad(
                predictions, ((0, 0), (0, 30 - predictions.shape[1])), mode="constant"
            )
        )

    predictions = np.clip(predictions, 0.0, 1.0)

    submission_data = pd.read_csv(sample_sub_path, encoding="utf-8")
    labels = list(submission_data.columns[1:].values)  # exact correct order
    submission_data[labels] = predictions
    submission_data["qa_id"] = test_ids

    print(submission_data.head())
    submission_data.to_csv("submission.csv", index=False)


main()

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py", line 115, in _solve_sparse_cg
    coef, info = sp_linalg.cg(C, y_column, tol=tol, atol="legacy")
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: cg() got an unexpected keyword argument 'tol'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/process_executor.py", line 490, in _process_worker
    r = call_item()
        ^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/process_executor.py", line 291, in __call__
    return self.fn(*self.args, **self.kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in __call__
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in <listcomp>
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
            ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py", line 123, in __call__
    return self.function(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/multioutput.py", line 49, in _fit_estimator
    estimator.fit(X, y, **fit_params)
  File "/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py", line 1134, in fit
    return super().fit(X, y, sample_weight=sample_weight)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py", line 900, in fit
    self.coef_, self.n_iter_ = _ridge_regression(
                               ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py", line 671, in _ridge_regression
    coef = _solve_sparse_cg(
           ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py", line 118, in _solve_sparse_cg
    coef, info = sp_linalg.cg(C, y_column, tol=tol)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: cg() got an unexpected keyword argument 'tol'
"""

The above exception was the direct cause of the following exception:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3781996018.py in <cell line: 0>()
     67 
     68 
---> 69 main()

/tmp/ipykernel_11/3781996018.py in main()
     30     model = MultiOutputRegressor(base, n_jobs=-1)
     31 
---> 32     model.fit(Xtr, y_train.values)
     33     valid_pred = model.predict(Xva)
     34     valid_pred = np.clip(valid_pred, 0.0, 1.0)

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
   2070         next(output)
   2071 
-> 2072         return output if self.return_generator else list(output)
   2073 
   2074     def __repr__(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _get_outputs(self, iterator, pre_dispatch)
   1680 
   1681             with self._backend.retrieval_context():
-> 1682                 yield from self._retrieve()
   1683 
   1684         except GeneratorExit:

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _retrieve(self)
   1782             # worker traceback.
   1783             if self._aborting:
-> 1784                 self._raise_error_fast()
   1785                 break
   1786 

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _raise_error_fast(self)
   1857         # called directly or if the generator is gc'ed.
   1858         if error_job is not None:
-> 1859             error_job.get_result(self.timeout)
   1860 
   1861     def _warn_exit_early(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in get_result(self, timeout)
    756             # callback thread, and is stored internally. It's just waiting to
    757             # be returned.
--> 758             return self._return_or_raise()
    759 
    760         # For other backends, the main thread needs to run the retrieval step.

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _return_or_raise(self)
    771         try:
    772             if self.status == TASK_ERROR:
--> 773                 raise self._result
    774             return self._result
    775         finally:

TypeError: cg() got an unexpected keyword argument 'tol'
