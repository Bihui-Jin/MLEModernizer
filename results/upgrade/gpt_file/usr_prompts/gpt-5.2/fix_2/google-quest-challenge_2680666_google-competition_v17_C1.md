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

# 8. Previous improvement plan

- What this solution (achieved nan) has done: 'I fix the runtime crash by switching from standalone `keras` imports to `tensorflow.keras`, which avoids the known protobuf/MessageFactory incompatibility in Kaggle environments. I also fix the biggest data/logic bugs that prevent correct end-to-end execution: using one shared tokenizer fitted on train text and applied to val/test (instead of refitting), replacing the extremely slow/buggy manual padding loop with `pad_sequences`, and ensuring targets are ordered exactly like `sample_submission.csv`. Finally, I make submission generation robust (correct row count, correct `qa_id` alignment with `test.csv`, clip predictions to `[0,1]`, and write `submission.csv`). These changes preserve the same core model/training approach (same LSTM model, MSE loss, 1 epoch), but make it runnable and produce a valid submission.'

# 9. Code solution

## === cell 0
import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.models import Model

from tqdm import tqdm

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
    return 0  # Fix: handle unexpected/missing categories safely


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
    data["categoty_normalized"] = data["category"].apply(cat_to_numeric)
    data.drop("category", axis=1, inplace=True)
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
    Fix: replace extremely slow/buggy manual padding with pad_sequences,
    and return model input shape (n, 3, maxlen) consistent with build_model_2.
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
def main():
    train_path = "../input/google-quest-challenge/train.csv"
    test_path = "../input/google-quest-challenge/test.csv"
    sample_path = "../input/google-quest-challenge/sample_submission.csv"

    train_data = pd.read_csv(train_path, encoding="utf-8")
    test_data = pd.read_csv(test_path, encoding="utf-8")
    sample_sub = pd.read_csv(sample_path, encoding="utf-8")

    train_data = prepare_data(train_data)
    test_data = prepare_data(test_data)

    train_cols, target_cols_any = get_vars_and_targets(train_data, test_data)

    target_cols = [c for c in sample_sub.columns if c != "qa_id"]

    X = train_data.loc[:, train_cols].copy()
    y = train_data.loc[:, target_cols].copy()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=SEED
    )

    tokenizer = fit_tokenizer(
        X_train, text_cols=["question_title", "question_body", "answer"]
    )

    maxlen = 1000
    X_train_arr = texts_to_padded_arrays(X_train, tokenizer, maxlen=maxlen)
    X_val_arr = texts_to_padded_arrays(X_val, tokenizer, maxlen=maxlen)

    model = build_model_2(tokenizer, maxlen=maxlen)
    model.fit(
        X_train_arr,
        y_train.values,
        batch_size=16,
        epochs=1,
        validation_split=0.2,
        verbose=2,
    )

    _ = model.evaluate(X_val_arr, y_val.values, verbose=0)

    test_features = test_data.drop(["qa_id", "categoty_normalized"], axis=1).copy()
    X_test_arr = texts_to_padded_arrays(test_features, tokenizer, maxlen=maxlen)

    predictions = model.predict(X_test_arr, batch_size=32, verbose=0)

    predictions = np.clip(predictions, 0.0, 1.0)

    submission = pd.DataFrame(predictions, columns=target_cols)
    submission.insert(0, "qa_id", test_data["qa_id"].values)

    submission = submission[sample_sub.columns]

    submission.to_csv("submission.csv", index=False)
    print("done! wrote submission.csv")
    print(submission.head())


main()
