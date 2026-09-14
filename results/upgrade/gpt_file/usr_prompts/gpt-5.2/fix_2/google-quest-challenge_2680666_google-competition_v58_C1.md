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

0.0042779852675161

# 6. Current score

0.00288

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.00288) has done: 'The crash is caused by importing standalone `keras` in this Kaggle environment, which can trigger a protobuf/Keras incompatibility (`MessageFactory.GetPrototype`). I switch all Keras imports to `tensorflow.keras` (same APIs/core logic) to fix runtime. I also fix a major logic bug where you were fitting a separate tokenizer on `X_test`/`test_data` (causing mismatched vocab indices and hurting score); instead, I fit the tokenizer on `X_train` once and reuse it for validation and test. Finally, I ensure predictions are clipped to `[0,1]` and that the submission always has the correct shape `(len(test), 30)` and columns.'

# 9. Code solution

## === cell 0
import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

np.random.seed(42)
tf.random.set_seed(42)


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
    target_cols = set(train_data.columns).difference(set(test_data.columns))
    train_cols = set(train_data.columns) - target_cols
    target_cols = list(target_cols)
    train_cols = list(train_cols)
    return train_cols, target_cols


def get_text_cols(frame):
    text_cols = []
    for col in frame.columns:
        if "title" in col or "body" in col or col == "answer":
            text_cols.append(col)
    return text_cols




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1


def fit_tokenizer_on_frame(frame):
    text_cols = get_text_cols(frame)
    tokenizer = Tokenizer()
    for col in text_cols:
        tokenizer.fit_on_texts(frame[col].fillna("").astype(str))
    return tokenizer, text_cols


def apply_tokenizer(frame, tokenizer, text_cols):
    frame = frame.copy()
    renamed_cols = []
    for col in text_cols:
        new_col = col + "_tokenized"
        renamed_cols.append(new_col)
        frame[new_col] = tokenizer.texts_to_sequences(frame[col].fillna("").astype(str))
        frame.drop([col], inplace=True, axis=1)
    return frame, renamed_cols


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
            SimpleRNNCell(256, activation="sigmoid", bias_initializer=initializer)
        )(rnn_layer_1)
        dense_layer = Dense(128, activation="relu")(rnn_layer_2)
        features.append(dense_layer)
        inputs.append(input_layer)

    merged_dense = Add()(features)
    droput = Dropout(0.2)(merged_dense)
    dense_1 = Dense(64, activation="relu")(droput)
    dense_2 = Dense(32, activation="relu")(dense_1)
    output = Dense(30)(dense_2)

    model = Model(inputs=inputs, outputs=[output])
    model.compile(optimizer="Adam", loss="mean_squared_error", metrics=["accuracy"])
    return model




## === cell 2
def main():
    train_path = "../input/google-quest-challenge/train.csv"
    test_path = "../input/google-quest-challenge/test.csv"
    sample_path = "../input/google-quest-challenge/sample_submission.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    train_data = prepare_data(train_data)
    test_data_prepared = prepare_data(test_data)

    train_cols, target_cols = get_vars_and_targets(train_data, test_data_prepared)

    X = train_data.loc[:, train_cols].copy()
    y = train_data.loc[:, target_cols].copy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    tokenizer, base_text_cols = fit_tokenizer_on_frame(X_train)

    X_train_tok, text_cols = apply_tokenizer(X_train, tokenizer, base_text_cols)
    X_val_tok, _ = apply_tokenizer(X_val, tokenizer, base_text_cols)

    X_train_padded = []
    X_val_padded = []
    for col in text_cols:
        X_train_padded.append(
            pad_sequences(X_train_tok[col], maxlen=200, padding="pre")
        )
        X_val_padded.append(pad_sequences(X_val_tok[col], maxlen=200, padding="pre"))

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
        print("Validation evaluate failed:", repr(e))

    test_ids = test_data["qa_id"].values
    test_features = test_data_prepared.drop(["qa_id"], axis=1)

    test_tok, _ = apply_tokenizer(test_features, tokenizer, base_text_cols)
    test_padded = []
    for col in text_cols:
        test_padded.append(pad_sequences(test_tok[col], maxlen=200, padding="pre"))

    try:
        predictions = model.predict(test_padded, verbose=0)
        if isinstance(predictions, list):
            predictions = predictions[0]
        predictions = np.asarray(predictions)
        if predictions.shape[1] != 30:
            predictions = np.random.rand(len(test_ids), 30)
    except Exception as e:
        print("Prediction failed, using random fallback:", repr(e))
        predictions = np.random.rand(len(test_ids), 30)

    predictions = np.clip(np.abs(predictions), 0.0, 1.0)

    submission_data = pd.read_csv(sample_path, encoding="utf-8")
    labels = list(submission_data.columns[1:].values)

    submission = pd.DataFrame({"qa_id": test_ids})
    for i, lab in enumerate(labels):
        submission[lab] = predictions[:, i]

    submission.to_csv("submission.csv", index=False)
    print(submission.head())
    print("Wrote submission.csv with shape:", submission.shape)


main()
