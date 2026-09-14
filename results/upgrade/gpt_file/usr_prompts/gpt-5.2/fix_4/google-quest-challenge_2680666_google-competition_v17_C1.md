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

0.0548585491919292

# 6. Current score

nan

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved nan) has done: 'I fix the runtime crash by switching from standalone `keras` imports to `tensorflow.keras`, which avoids the known protobuf/MessageFactory incompatibility in Kaggle environments. I also fix the biggest data/logic bugs that prevent correct end-to-end execution: using one shared tokenizer fitted on train text and applied to val/test (instead of refitting), replacing the extremely slow/buggy manual padding loop with `pad_sequences`, and ensuring targets are ordered exactly like `sample_submission.csv`. Finally, I make submission generation robust (correct row count, correct `qa_id` alignment with `test.csv`, clip predictions to `[0,1]`, and write `submission.csv`). These changes preserve the same core model/training approach (same LSTM model, MSE loss, 1 epoch), but make it runnable and produce a valid submission.'
- What this solution (achieved nan) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error in Kaggle. Then I fix a data bug where the model was trained on non-text columns (like `qa_id`) by explicitly selecting only the three text fields for tokenization and sequence building, keeping the same LSTM architecture and training loop. I also correct the train/validation usage so the explicit `X_val/y_val` are actually used as `validation_data` (not a second split of `X_train`). Finally, I make submission generation robust by ensuring the test features include exactly the expected text columns, aligning columns to `sample_submission.csv`, clipping to `[0,1]`, and writing `submission.csv`.'
- What this solution (achieved nan) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from running by removing the incompatible pure-Python protobuf forcing and instead pinning protobuf to the C++ implementation (the default) before importing TensorFlow, plus add a safe fallback if the env var is pre-set. I also correct a logic issue in `prepare_data` where it blindly expects a `category` column and can crash if missing, and ensure the engineered column name is spelled consistently. Finally, I keep the model and training loop identical, but make the I/O paths robust by trying both the competition folder and the flat `/kaggle/input/` copies so it always finds the CSVs and produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.models import Model

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
    return 0  # handle unexpected/missing categories safely


def prepare_data(frame: pd.DataFrame) -> pd.DataFrame:
    to_drop = [
        "question_user_page",
        "host",
        "url",
        "answer_user_page",
        "answer_user_name",
        "question_user_name",
        "question_user_page",
    ]
    data = frame.drop([c for c in to_drop if c in frame.columns], axis=1)

    if "category" in data.columns:
        data["category_normalized"] = data["category"].apply(cat_to_numeric)
        data.drop("category", axis=1, inplace=True)
    else:
        data["category_normalized"] = 0

    return data


def get_vars_and_targets(train_data: pd.DataFrame, test_data: pd.DataFrame):
    target_cols = sorted(
        list(set(train_data.columns).difference(set(test_data.columns)))
    )
    train_cols = [c for c in train_data.columns if c not in target_cols]
    return train_cols, target_cols


def fit_tokenizer(frame: pd.DataFrame, text_cols=None):
    if text_cols is None:
        text_cols = ["question_title", "question_body", "answer"]
    tok = Tokenizer()
    tok.fit_on_texts(
        pd.concat([frame[c].fillna("") for c in text_cols], axis=0).astype(str).values
    )
    return tok


def texts_to_padded_arrays(
    frame: pd.DataFrame, tokenizer: Tokenizer, maxlen=1000, text_cols=None
):
    """
    Return model input shape (n, 3, maxlen) consistent with build_model_2.
    """
    if text_cols is None:
        text_cols = ["question_title", "question_body", "answer"]

    seqs = []
    for c in text_cols:
        s = frame[c].fillna("").astype(str).values
        seq = tokenizer.texts_to_sequences(s)
        seq = pad_sequences(
            seq, maxlen=maxlen, padding="post", truncating="post", value=0
        )
        seqs.append(seq)

    X = np.stack(seqs, axis=1).astype(np.int32)
    return X




## === cell 2
def build_model_2(tokenizer, maxlen=1000):
    input_qt = Input(shape=(3, maxlen))
    lstm_qt1 = LSTM(
        1024, return_sequences=True, recurrent_dropout=0.25, activation="sigmoid"
    )(input_qt)
    lstm_qt2 = LSTM(
        512, return_sequences=True, recurrent_dropout=0.25, activation="sigmoid"
    )(lstm_qt1)
    lstm_qt3 = LSTM(512)(lstm_qt2)

    dense = Dense(512, activation="relu")(lstm_qt3)
    dropout = Dropout(0.2)(dense)
    dense2 = Dense(256, activation="relu")(dropout)
    output = Dense(30, name="outputs")(dense2)

    model = Model(inputs=[input_qt], outputs=[output])
    model.compile(optimizer="Adam", loss="mean_squared_error", metrics=["accuracy"])
    return model




## === cell 3
def _resolve_path(*candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the candidate paths exist: {candidates}")


def main():
    train_path = _resolve_path(
        "../input/google-quest-challenge/train.csv",
        "/kaggle/input/google-quest-challenge/train.csv",
        "/kaggle/input/train.csv",
    )
    test_path = _resolve_path(
        "../input/google-quest-challenge/test.csv",
        "/kaggle/input/google-quest-challenge/test.csv",
        "/kaggle/input/test.csv",
    )
    sample_path = _resolve_path(
        "../input/google-quest-challenge/sample_submission.csv",
        "/kaggle/input/google-quest-challenge/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
    )

    train_data = pd.read_csv(train_path, encoding="utf-8")
    test_data = pd.read_csv(test_path, encoding="utf-8")
    sample_sub = pd.read_csv(sample_path, encoding="utf-8")

    train_data = prepare_data(train_data)
    test_data = prepare_data(test_data)

    target_cols = [c for c in sample_sub.columns if c != "qa_id"]
    text_cols = ["question_title", "question_body", "answer"]

    y = train_data.loc[:, target_cols].copy()

    X_train, X_val, y_train, y_val = train_test_split(
        train_data, y, test_size=0.25, random_state=SEED
    )

    tokenizer = fit_tokenizer(X_train, text_cols=text_cols)

    maxlen = 1000
    X_train_arr = texts_to_padded_arrays(
        X_train, tokenizer, maxlen=maxlen, text_cols=text_cols
    )
    X_val_arr = texts_to_padded_arrays(
        X_val, tokenizer, maxlen=maxlen, text_cols=text_cols
    )

    model = build_model_2(tokenizer, maxlen=maxlen)

    model.fit(
        X_train_arr,
        y_train.values,
        batch_size=16,
        epochs=1,
        validation_data=(X_val_arr, y_val.values),
        verbose=2,
    )

    _ = model.evaluate(X_val_arr, y_val.values, verbose=0)

    X_test_arr = texts_to_padded_arrays(
        test_data, tokenizer, maxlen=maxlen, text_cols=text_cols
    )

    predictions = model.predict(X_test_arr, batch_size=32, verbose=0)
    predictions = np.clip(predictions, 0.0, 1.0)

    submission = pd.DataFrame(predictions, columns=target_cols)
    submission.insert(0, "qa_id", test_data["qa_id"].values)

    submission = submission[sample_sub.columns]
    assert (
        submission.shape[0] == test_data.shape[0]
    ), "Submission row count mismatch vs test.csv"
    assert list(submission.columns) == list(
        sample_sub.columns
    ), "Submission columns mismatch vs sample_submission.csv"

    submission.to_csv("submission.csv", index=False)
    print("done! wrote submission.csv")
    print(submission.head())


main()
