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

-0.0015711539026244

# 6. Current score

nan

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.03038) has done: 'I fix the runtime crash caused by an incompatible `keras`/`protobuf` combination by switching to `tensorflow.keras` everywhere (same layers/architecture, just a compatible import path). I also fix a major logic bug where you fit a new tokenizer separately on train/test (breaking word indices); instead, fit once on train and reuse it to tokenize/pad val/test. Finally, I ensure prediction shape matches the 30 targets, clip predictions into `[0,1]` (required by the competition), and write a valid `submission.csv` with the exact sample submission columns.'
- What this solution (achieved nan) has done: 'I fix the TensorFlow/protobuf crash by forcing a compatible pure-Python protobuf implementation before importing TensorFlow (this is a common Kaggle environment issue behind `MessageFactory.GetPrototype`). I also make the data-path handling robust by automatically selecting an existing `../input/...` or `/kaggle/input/...` location without changing the intended dataset files. To keep the score changes minimal (you’re already above the target), I remove the fallback that replaces non-30-column predictions with random numbers and instead deterministically coerce predictions to 30 columns (pad/trim) and clip to `[0,1]`. Finally, I ensure the submission columns exactly match `sample_submission.csv` and that `qa_id` alignment is correct.'
- What this solution (achieved nan) has done: 'The crash happens before your code runs because TensorFlow is importing an incompatible `protobuf` version (`MessageFactory.GetPrototype` missing). The most reliable minimal fix in Kaggle is to force the pure-Python protobuf implementation *and* disable the C++ fast-path before importing TensorFlow, and to clear any previously imported `google.protobuf` modules if present. I also keep your data/model logic unchanged, but make the test padding use the same columns as training (to avoid accidental column drops causing mismatch) and ensure the submission row count matches `test.csv` (19550) rather than `sample_submission.csv` (608). These changes are correctness/stability fixes and should eliminate the `nan`/no-submission issue without altering the core model/training loop.'
- What this solution (achieved nan) has done: 'You’re crashing before any training because TensorFlow is still importing an incompatible protobuf build, so I harden the protobuf/TensorFlow import sequence (force pure-Python protobuf, disable the C++ implementation, and clear any pre-imported protobuf modules) before importing `tensorflow`. I also move the TensorFlow import inside `main()` so the environment variables definitely take effect first. Everything else (data prep, tokenizer usage, model architecture, training loop, prediction coercion/clipping, and submission formatting) be kept the same so it runs end-to-end and produces a valid `submission.csv` with the correct 30 columns and 19550 rows.'
- What this solution (achieved nan) has done: 'The crash is happening during TensorFlow import because the protobuf C++ implementation is still being used despite the environment variables, leading to `MessageFactory.GetPrototype` missing. I harden the protobuf/TensorFlow import sequence by forcing the pure-Python protobuf implementation *before* any protobuf-related import, and I also pre-import `google.protobuf` once (pure-Python) to “lock in” the correct backend before TensorFlow loads it. This is a stability/runtime fix (score-neutral) and keeps your data prep, tokenizer usage, model architecture, training loop, and submission formatting unchanged. The script then run end-to-end and write a valid `submission.csv` with 19550 rows and the exact 30 label columns from `sample_submission.csv`.'
- What this solution (achieved nan) has done: 'I fix the TensorFlow/protobuf import crash by ensuring the pure-Python protobuf implementation is selected before any TensorFlow-related imports, and by also forcing the C++ implementation off via the supported environment variable. I do this at the very top of the script (before any other imports) and harden it inside `main()` as well, without changing your model/training/data logic. I also make the TensorFlow import happen only after that setup, and keep the submission generation identical so it reliably writes `submission.csv` with 19550 rows and the 30 label columns. These changes are runtime/stability fixes and should eliminate the current `nan`/no-submission situation.'
- What this solution (achieved nan) has done: 'I fix the TensorFlow import crash by hardening the protobuf setup sequence so TensorFlow cannot load the incompatible C++ protobuf backend (the root cause of `MessageFactory.GetPrototype` missing). This is done by forcing the pure-Python protobuf implementation at process start, clearing any already-imported protobuf modules, and (critically) setting `TF_USE_CXX11_ABI=0`, which is a common Kaggle workaround when TF/protobuf wheels are mismatched. I keep your model/data logic unchanged, but move all TensorFlow/Keras-dependent functions inside `main()` so they are defined only after the import succeeds. Finally, I keep the submission formatting identical and ensure the output is a valid `submission.csv` with 19550 rows and the 30 correct label columns clipped to `[0,1]`.'
- What this solution (achieved 0.04026) has done: 'The crash is happening during TensorFlow import due to an incompatible protobuf C++ backend being loaded (`MessageFactory.GetPrototype` missing). I make the protobuf backend selection deterministic by forcing the pure-Python implementation at process start and also setting the supported `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before any TensorFlow import, without altering your model/data logic. I also add a safe fallback: if TensorFlow still cannot import in this environment, the script still produce a valid `submission.csv` (using deterministic, constant predictions in `[0,1]`) so you don’t get `nan` from “no submission”. All existing core modeling code remains unchanged and be used whenever TensorFlow imports successfully.'
- What this solution (achieved 0.01446) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf backend *before any other imports* and by clearing any preloaded `google.protobuf` modules so TensorFlow can’t latch onto the incompatible C++ implementation. This is a runtime/stability change and keeps your core model, tokenizer usage, training loop, and submission formatting the same. I also ensure the submission always has exactly 19550 rows and the 30 target columns from `sample_submission.csv`, with predictions clipped to `[0,1]` (already required). Since your current score (0.04026) is already above the target, I won’t make any score-improving changes beyond making the TensorFlow path actually run again.'
- What this solution (achieved nan) has done: 'I fix the TensorFlow/protobuf crash by making the protobuf backend selection deterministic before *any* TensorFlow import: force the pure-Python protobuf, disable the C++ implementation, and ensure `google.protobuf` isn’t already loaded. This is a runtime/stability fix and keeps your model, tokenizer usage, training loop, and submission formatting the same. If TensorFlow still cannot import, the existing deterministic fallback still produce a valid `submission.csv` with correct columns/rows. I won’t make any score-improving changes because your current score is already above the target and the goal is mainly to run end-to-end reliably.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CXX"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

for k in list(sys.modules.keys()):
    if k.startswith("google.protobuf"):
        del sys.modules[k]

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(69)




## === cell 1
def prepare_data(frame: pd.DataFrame) -> pd.DataFrame:
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


def get_vars_and_targets(train_data: pd.DataFrame, test_data: pd.DataFrame):
    target_cols = list(set(train_data.columns).difference(set(test_data.columns)))
    train_cols = list(set(train_data.columns) - set(target_cols))
    return train_cols, target_cols


def get_text_cols(frame: pd.DataFrame):
    text_cols = []
    for col in frame.columns:
        if "title" in col or "body" in col or col == "answer":
            text_cols.append(col)
    return text_cols




## === cell 2
def _first_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the candidate paths exist: {paths}")


def _coerce_preds_to_30(preds: np.ndarray, n_rows: int) -> np.ndarray:
    """Stability fix: deterministically pad/trim to 30 columns and clip to [0,1]."""
    preds = np.asarray(preds)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)

    if preds.shape[0] != n_rows:
        if preds.shape[0] > n_rows:
            preds = preds[:n_rows, :]
        else:
            pad_rows = n_rows - preds.shape[0]
            last = (
                preds[-1:, :] if preds.shape[0] > 0 else np.zeros((1, preds.shape[1]))
            )
            preds = np.vstack([preds, np.repeat(last, pad_rows, axis=0)])

    if preds.shape[1] > 30:
        preds = preds[:, :30]
    elif preds.shape[1] < 30:
        pad = np.zeros((preds.shape[0], 30 - preds.shape[1]), dtype=preds.dtype)
        preds = np.hstack([preds, pad])

    return np.clip(preds, 0.0, 1.0)




## === cell 3
def main():
    train_path = _first_existing_path(
        [
            "../input/google-quest-challenge/train.csv",
            "/kaggle/input/google-quest-challenge/train.csv",
            "/kaggle/data/google-quest-challenge/train.csv",
            "data/google-quest-challenge/train.csv",
            "data/train.csv",
            "/kaggle/input/train.csv",
        ]
    )
    test_path = _first_existing_path(
        [
            "../input/google-quest-challenge/test.csv",
            "/kaggle/input/google-quest-challenge/test.csv",
            "/kaggle/data/google-quest-challenge/test.csv",
            "data/google-quest-challenge/test.csv",
            "data/test.csv",
            "/kaggle/input/test.csv",
        ]
    )
    sample_path = _first_existing_path(
        [
            "../input/google-quest-challenge/sample_submission.csv",
            "/kaggle/input/google-quest-challenge/sample_submission.csv",
            "/kaggle/data/google-quest-challenge/sample_submission.csv",
            "data/google-quest-challenge/sample_submission.csv",
            "data/sample_submission.csv",
            "/kaggle/input/sample_submission.csv",
        ]
    )

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    train_data = prepare_data(train_data)
    test_data = prepare_data(test_data)

    train_cols, _ = get_vars_and_targets(train_data, test_data)
    submission_template = pd.read_csv(sample_path, encoding="utf-8")
    target_cols = list(submission_template.columns[1:])
    labels = target_cols

    X = train_data.loc[:, train_cols].copy()
    y = train_data.loc[:, target_cols].copy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=69
    )

    text_cols = get_text_cols(X_train)

    try:
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CXX"] = "1"
        for k in list(sys.modules.keys()):
            if k.startswith("google.protobuf"):
                del sys.modules[k]

        import tensorflow as tf
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Dense,
            Dropout,
            Embedding,
            Input,
            Add,
            RNN,
            SimpleRNNCell,
        )
        from tensorflow.keras.initializers import RandomUniform

        tf.random.set_seed(69)

        def fit_tokenizer_on_frame(frame: pd.DataFrame, text_cols):
            tokenizer = Tokenizer()
            for col in text_cols:
                tokenizer.fit_on_texts(frame[col].astype(str).fillna(""))
            return tokenizer

        def texts_to_padded_arrays(
            frame: pd.DataFrame, text_cols, tokenizer, maxlen=200
        ):
            padded_list = []
            for col in text_cols:
                seqs = tokenizer.texts_to_sequences(frame[col].astype(str).fillna(""))
                padded = pad_sequences(
                    seqs, maxlen=maxlen, padding="pre", truncating="pre"
                )
                padded_list.append(padded)
            return padded_list

        def model_5_rnn(X_train_padded, tokenizer, maxlen=200):
            initializer = RandomUniform(seed=69)
            inputs = []
            features = []

            for _ in X_train_padded:
                input_layer = Input(shape=(maxlen,))
                emb_layer = Embedding(len(tokenizer.word_index) + 1, output_dim=512)(
                    input_layer
                )
                rnn_layer_1 = RNN(
                    SimpleRNNCell(
                        512,
                        recurrent_dropout=0.1,
                        activation="sigmoid",
                        bias_initializer=initializer,
                    ),
                    return_sequences=True,
                )(emb_layer)
                rnn_layer_2 = RNN(
                    SimpleRNNCell(
                        256, activation="sigmoid", bias_initializer=initializer
                    )
                )(rnn_layer_1)
                dense_layer = Dense(128, activation="relu")(rnn_layer_2)
                features.append(dense_layer)
                inputs.append(input_layer)

            merged_dense = Add()(features)
            droput = Dropout(0.2)(merged_dense)
            dense_1 = Dense(64, activation="relu")(droput)
            dense_2 = Dense(32, activation="relu")(dense_1)
            output = Dense(30)(dense_2)

            model = Model(inputs=inputs, outputs=output)
            model.compile(
                optimizer="Adam", loss="mean_squared_error", metrics=["accuracy"]
            )
            return model

        tokenizer = fit_tokenizer_on_frame(X_train, text_cols)
        X_train_padded = texts_to_padded_arrays(
            X_train, text_cols, tokenizer, maxlen=200
        )
        X_val_padded = texts_to_padded_arrays(X_val, text_cols, tokenizer, maxlen=200)

        model = model_5_rnn(X_train_padded, tokenizer, maxlen=200)

        model.fit(
            X_train_padded,
            y_train.values,
            batch_size=32,
            epochs=1,
            validation_split=0.2,
            verbose=2,
        )

        try:
            print(model.evaluate(X_val_padded, y_val.values, verbose=0))
        except Exception as e:
            print("Evaluation skipped due to:", repr(e))

        test_ids = pd.read_csv(test_path)["qa_id"].values
        test_frame = test_data.loc[:, train_cols].copy()
        test_padded = texts_to_padded_arrays(
            test_frame, text_cols, tokenizer, maxlen=200
        )

        preds = model.predict(test_padded, verbose=0)
        preds = _coerce_preds_to_30(preds, n_rows=len(test_frame))

    except Exception as e:
        print(
            "TensorFlow import/train failed; writing deterministic fallback submission. Error:",
            repr(e),
        )
        test_ids = pd.read_csv(test_path)["qa_id"].values
        n_rows = len(test_ids)
        preds = np.full((n_rows, 30), 0.5, dtype=np.float32)

    submission_data = pd.DataFrame({"qa_id": test_ids})
    submission_data[labels] = preds
    submission_data.to_csv("submission.csv", index=False)

    print(submission_data.head())
    print("Wrote submission.csv with shape:", submission_data.shape)


main()

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
