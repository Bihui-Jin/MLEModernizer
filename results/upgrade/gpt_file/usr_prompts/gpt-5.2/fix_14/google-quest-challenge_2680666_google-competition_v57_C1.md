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

-0.0045766777157599

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.00118) has done: 'The crash comes from importing standalone `keras` in this Kaggle environment, which can trigger a protobuf/TF-Keras incompatibility (`MessageFactory.GetPrototype`). I fix this by switching the imports to `tensorflow.keras` (same Keras APIs, same model code) so the notebook runs reliably. I also fix a logic bug where you tokenize train and test with different tokenizers (breaking vocabulary alignment); instead, fit on train and reuse it for test, which is score-positive without changing the overall approach. Finally, I ensure predictions are shaped to exactly match `sample_submission.csv` (608 rows) and are clipped to `[0,1]`, then write a valid `submission.csv`.'
- What this solution (achieved nan) has done: 'I fix the TensorFlow/Keras import crash by forcing the bundled `tf.keras` and using the legacy Keras API path that’s stable in Kaggle TF2, which resolves the `MessageFactory.GetPrototype` error. Then I correct the logic so the same tokenizer fitted on training text is reused for any other splits and for the real test set, preventing vocabulary mismatch. Finally, I replace the random predictions with model inference on the competition `test.csv` (keeping the same model architecture/training loop style) and ensure the submission has exactly the `sample_submission.csv` columns, in the same row order, with predictions clipped to `[0,1]`.'
- What this solution (achieved nan) has done: 'I fix the TensorFlow/Keras import crash causing `MessageFactory.GetPrototype` by avoiding the legacy-keras path and using a safe `tf.keras` import with a protobuf compatibility setting that works on Kaggle. I also fix a shape bug in inference: `model.predict()` returns a list for a single-output model, so the current `np.asarray(test_pred)` produces the wrong shape and leads to invalid submission values (NaNs/objects). Finally, I make the train/target column ordering deterministic (matching `sample_submission.csv`) to avoid label-column mismatches and ensure the written `submission.csv` is valid with predictions clipped to `[0,1]`.'
- What this solution (achieved nan) has done: 'I fix the runtime crash caused by the protobuf implementation switch being applied too late (after protobuf is already imported indirectly), by setting the required environment variables before any TensorFlow/Keras import. I also ensure the Keras import path is consistently `tf.keras` to avoid the standalone/legacy Keras incompatibility that triggers `MessageFactory.GetPrototype`. Finally, I keep the existing model/training logic intact while making the submission creation robust (column order/shape, clipping to [0,1], and guaranteed `.csv` output).'
- What this solution (achieved nan) has done: 'The crash happens before your environment variables take effect because TensorFlow imports protobuf early; the simplest reliable fix in Kaggle is to stop forcing the pure-Python protobuf runtime and just use the default C++ implementation (remove those env overrides). I keep your model/training/inference logic intact, but add a small safety fallback to load the CSVs from either `../input/google-quest-challenge/...` or `../input/...` depending on the Kaggle mount. Finally, I keep the existing submission-shape checks and ensure the written file is a proper `submission.csv` with the exact `sample_submission.csv` column order and predictions clipped to `[0,1]`.'
- What this solution (achieved nan) has done: 'I fix the runtime crash coming from an incompatible protobuf/TF import path by setting the necessary protobuf environment variables before importing TensorFlow, and by forcing TensorFlow to use the pure-Python protobuf implementation that avoids `MessageFactory.GetPrototype` issues in some Kaggle images. I also keep your existing model/training logic intact while making one small robustness fix: ensure any missing category mappings return a default numeric value (to avoid accidental NaNs if category is later used). Finally, I keep the submission creation logic the same but add a defensive check to guarantee we always write a valid `submission.csv` with the exact sample submission row order and columns, with predictions clipped to `[0,1]`.'
- What this solution (achieved 0.05278) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by removing the protobuf environment overrides that force the pure-Python protobuf runtime, and by ensuring TensorFlow is imported cleanly with the Kaggle-default protobuf implementation. Then I keep your model/training/inference logic intact, but make the input column selection deterministic (preserve DataFrame column order instead of using sets) to avoid subtle train/test feature misalignment that can hurt score or create NaNs. Finally, I keep your submission assembly the same while adding a small safety check to ensure the final CSV has exactly the sample submission’s row order and 30 target columns with finite `[0,1]` values.'
- What this solution (achieved nan) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf implementation environment variables *before* importing TensorFlow, which is required to avoid the `MessageFactory.GetPrototype` error in some Kaggle images. I keep your model/training logic intact, but also fix a submission-shape bug where the code mistakenly expects 31 target columns instead of the required 30 (the labels exclude `qa_id`). Finally, I add a tiny safety guard to ensure we always infer the correct text column count from the tokenized column names, and still write a valid `submission.csv` with the exact sample-submission row order and columns clipped to `[0,1]`.'
- What this solution (achieved nan) has done: 'I fix the immediate crash (`MessageFactory.GetPrototype`) by removing the environment overrides that force the pure-Python protobuf runtime, which is what commonly triggers this incompatibility in Kaggle’s TensorFlow build. I keep your model, training loop, and text/tokenization approach the same, but ensure TensorFlow is imported cleanly using the default protobuf implementation. I also keep the submission assembly logic intact while adding a small defensive check that `labels` are exactly the 30 sample-submission targets and that predictions are finite and correctly shaped before writing `submission.csv`.'
- What this solution (achieved nan) has done: 'I fix the crash happening at import time (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by ensuring TensorFlow uses the pure-Python protobuf implementation and by forcing the stable `tf.keras` API, with the environment variables set *before* importing TensorFlow. I also add a small, score-neutral safety fallback: if the protobuf env settings still don’t prevent the crash in this image, the script retry importing TensorFlow in a clean subprocess configured correctly and then continue normally. The model architecture, tokenization approach, training loop, and prediction logic remain unchanged; the goal is purely to make the pipeline run end-to-end and reliably write a valid `submission.csv`.'
- What this solution (achieved nan) has done: 'I fix the TensorFlow import crash by removing the protobuf environment overrides that are triggering the `MessageFactory.GetPrototype` incompatibility in this Kaggle image, and import TensorFlow normally (still using `tf.keras` exactly as before). This is the minimal unblocker so the script runs end-to-end and trains/inferes as intended. I keep the model architecture, tokenization, training loop, and submission assembly logic unchanged, but add a tiny safeguard to ensure `target_cols` exactly matches the 30 sample-submission labels (preventing accidental empty/misaligned targets that can yield NaNs). The output always be a valid `submission.csv` with correct columns/order and predictions clipped to `[0,1]`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge

np.random.seed(42)




## === cell 1
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
            "user_page" in col
            or "host" in col
            or "url" in col
            or "user_name" in col
            or "categ" in col
        ):
            to_drop.append(col)
    data = frame.drop(to_drop, axis=1)
    return data


def get_vars_and_targets(train_data, test_data):
    test_cols = set(test_data.columns)
    train_cols = [c for c in train_data.columns if c in test_cols]
    target_cols = [c for c in train_data.columns if c not in test_cols]
    return train_cols, target_cols


def get_text_cols(frame):
    text_cols = []
    for col in frame.columns:
        if "title" in col or "body" in col or col == "answer":
            text_cols.append(col)
    return text_cols




## === cell 2
def _resolve_path(primary: str, fallback: str) -> str:
    if Path(primary).exists():
        return primary
    if Path(fallback).exists():
        return fallback
    raise FileNotFoundError(f"Could not find file at '{primary}' or '{fallback}'")


def _build_text_series(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ["question_title", "question_body", "answer"] if c in df.columns]
    if not cols:
        raise ValueError(
            "No text columns found among: question_title, question_body, answer"
        )
    s = df[cols[0]].fillna("").astype(str)
    for c in cols[1:]:
        s = s + " " + df[c].fillna("").astype(str)
    return s


def main():
    train_path = _resolve_path(
        "../input/google-quest-challenge/train.csv", "../input/train.csv"
    )
    test_path = _resolve_path(
        "../input/google-quest-challenge/test.csv", "../input/test.csv"
    )
    sample_path = _resolve_path(
        "../input/google-quest-challenge/sample_submission.csv",
        "../input/sample_submission.csv",
    )

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    train_data = prepare_data(train_data)
    test_data = prepare_data(test_data)

    submission_data = pd.read_csv(sample_path, encoding="utf-8")
    labels = list(submission_data.columns[1:].values)  # 30 target columns
    if len(labels) != 30:
        raise ValueError(
            f"Expected 30 target columns from sample_submission, got {len(labels)}"
        )

    train_cols, target_cols = get_vars_and_targets(train_data, test_data)

    target_cols = [c for c in labels if c in target_cols]
    if len(target_cols) != 30:
        missing = [c for c in labels if c not in target_cols]
        raise ValueError(
            f"Training data does not contain all 30 target columns. Missing: {missing}"
        )

    X_train, X_valid, y_train, y_valid = train_test_split(
        train_data.loc[:, train_cols],
        train_data.loc[:, target_cols],
        test_size=0.25,
        random_state=42,
    )

    X_train_text = _build_text_series(X_train)
    X_valid_text = _build_text_series(X_valid)
    X_test_text = _build_text_series(test_data.loc[:, train_cols])

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        strip_accents="unicode",
        lowercase=True,
        dtype=np.float32,
    )
    Xtr = vectorizer.fit_transform(X_train_text)
    Xva = vectorizer.transform(X_valid_text)
    Xte = vectorizer.transform(X_test_text)

    base = Ridge(alpha=1.0, random_state=42, solver="sag")
    model = MultiOutputRegressor(base, n_jobs=1)

    ytr = y_train.to_numpy(dtype=np.float32, copy=False)
    model.fit(Xtr, ytr)

    _ = model.predict(Xva)

    test_pred = model.predict(Xte)
    test_pred = np.asarray(test_pred, dtype=np.float32)

    if test_pred.ndim != 2 or test_pred.shape[1] != len(labels):
        raise ValueError(
            f"Bad prediction shape {test_pred.shape}, expected (n_test, {len(labels)})"
        )

    test_pred = np.nan_to_num(test_pred, nan=0.5, posinf=1.0, neginf=0.0)
    test_pred = np.clip(test_pred, 0.0, 1.0)

    test_qa = test_data[["qa_id"]].copy()
    pred_df = pd.DataFrame(test_pred, columns=labels)
    pred_df.insert(0, "qa_id", test_qa["qa_id"].values)

    submission = submission_data[["qa_id"]].merge(pred_df, on="qa_id", how="left")
    submission[labels] = submission[labels].fillna(0.5).astype(np.float32)

    submission = submission[["qa_id"] + labels]
    submission.to_csv("submission.csv", index=False)

    print(submission.head())
    print("Wrote submission.csv with shape:", submission.shape)


if __name__ == "__main__":
    main()

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
/tmp/ipykernel_11/3478234693.py in <cell line: 0>()
    112 
    113 if __name__ == "__main__":
--> 114     main()

/tmp/ipykernel_11/3478234693.py in main()
     82 
     83     ytr = y_train.to_numpy(dtype=np.float32, copy=False)
---> 84     model.fit(Xtr, ytr)
     85 
     86     _ = model.predict(Xva)

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
