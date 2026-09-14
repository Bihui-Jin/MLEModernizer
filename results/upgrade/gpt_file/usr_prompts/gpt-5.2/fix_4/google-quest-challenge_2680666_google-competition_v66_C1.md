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

nan

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.00364) has done: 'The crash is caused by importing the standalone `keras` package in this Kaggle environment, which can trigger a protobuf incompatibility (`MessageFactory.GetPrototype`). The minimal fix is to use `tensorflow.keras` everywhere (same Keras API, same model/logic) and to set a couple of safe environment flags before TensorFlow import to avoid protobuf implementation issues. I also fixed a logic bug where the test set was tokenized with a *new* tokenizer instead of the one fit on training, which would make inference inconsistent and often fail/produce garbage; this is score-improving while preserving the exact intended pipeline. Finally, I ensured predictions are clipped to `[0,1]` (required by the competition) and that the submission aligns row-wise with `qa_id`.'
- What this solution (achieved nan) has done: 'I fix the TensorFlow/protobuf crash by forcing a compatible protobuf implementation *before* importing TensorFlow, and by avoiding any standalone `keras` imports (keeping everything under `tensorflow.keras`). I also fix a subtle but important logic issue: `train_cols`/`target_cols` were produced from `set()` and thus had nondeterministic order, which can scramble the 30 targets and tank Spearman; I derive `target_cols` in the exact `sample_submission.csv` column order and keep feature columns in stable order. Finally, I keep the existing model/training loop intact, ensure prediction shapes align to 30 labels, clip to `[0,1]`, and always write a valid `submission.csv`.'
- What this solution (achieved nan) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf runtime and disabling C++ protobuf before TensorFlow is imported, which avoids the `MessageFactory.GetPrototype` error in this environment. I keep your model/training logic intact, but make the text preprocessing safe by copying dataframes and avoiding in-place mutation side-effects across splits. I also make prediction handling robust to Keras returning a list/extra dimensions, and ensure the submission uses the exact target column order from `sample_submission.csv`, clips outputs to `[0,1]`, and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_DISABLE_C_IMPLEMENTATION", "1")

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

np.random.seed(69)
tf.random.set_seed(69)




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


def transform_texts(frame, tokenizer=None, test=False):
    frame = frame.copy()

    text_cols = get_text_cols(frame)

    for col in text_cols:
        frame[col] = frame[col].fillna("").astype(str)

    if test is False:
        tokenizer = Tokenizer(oov_token="OOV")
        for col in text_cols:
            tokenizer.fit_on_texts(frame[col])
    else:
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when test=True")

    renamed_cols = []
    for col in text_cols:
        renamed_cols.append(col + "_tokenized")
        frame[col + "_tokenized"] = tokenizer.texts_to_sequences(frame[col])
        frame.drop([col], inplace=True, axis=1)

    return [frame, renamed_cols, tokenizer]


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
    sample_sub_path = "../input/google-quest-challenge/sample_submission.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    train_data = prepare_data(train_data)
    test_data = prepare_data(test_data)

    train_cols, target_cols = get_vars_and_targets(
        train_data, test_data, sample_sub_path
    )

    X_train, X_test, y_train, y_test = train_test_split(
        train_data.loc[:, train_cols],
        train_data.loc[:, target_cols],
        test_size=0.25,
        random_state=69,
    )

    X_train, text_cols, tokenizer = transform_texts(X_train, test=False)
    X_test, text_cols, _ = transform_texts(X_test, tokenizer, test=True)

    X_train_padded = []
    X_test_padded = []
    for i in range(len(text_cols)):
        X_train_padded.append(
            pad_sequences(X_train[text_cols[i]], maxlen=200, padding="pre")
        )
        X_test_padded.append(
            pad_sequences(X_test[text_cols[i]], maxlen=200, padding="pre")
        )

    model = model_5_rnn(X_train_padded, tokenizer)

    model.fit(
        X_train_padded,
        y_train.values,
        batch_size=32,
        epochs=1,
        validation_split=0.2,
        verbose=2,
    )

    print(model.evaluate(X_test_padded, y_test.values, verbose=0))

    test_ids = test_data["qa_id"].values
    test_x = test_data.drop(["qa_id"], axis=1).copy()
    test_x, test_text_cols, _ = transform_texts(test_x, tokenizer, test=True)

    test_padded = []
    for col in test_text_cols:
        test_padded.append(pad_sequences(test_x[col], maxlen=200, padding="pre"))

    try:
        predictions = model.predict(test_padded, verbose=0)
    except Exception:
        predictions = np.random.rand(len(test_x), 30)

    if isinstance(predictions, (list, tuple)):
        predictions = predictions[0]

    predictions = np.asarray(predictions)
    if predictions.ndim == 3:
        predictions = predictions.squeeze()
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
