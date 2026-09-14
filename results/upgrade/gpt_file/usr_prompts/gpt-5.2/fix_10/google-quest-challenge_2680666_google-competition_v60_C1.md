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

-0.0002288444451554

# 6. Current score

nan

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved nan) has done: 'The crash comes from importing standalone `keras` in this Kaggle runtime, which can trigger a protobuf/Keras incompatibility (`MessageFactory.GetPrototype`). I switch the Keras imports to `tensorflow.keras` (same APIs, same model/loss, minimal logic change) so the notebook runs end-to-end. I also fix a major inference logic bug: you were fitting a new tokenizer separately on train/test/test_data, causing inconsistent token IDs and often forcing the random-prediction fallback; instead, fit the tokenizer once on training text and reuse it for validation/test. Finally, I ensure predictions are shaped `(n_test, 30)` and clipped to `[0,1]` to match submission requirements, and write `submission.csv`.'
- What this solution (achieved nan) has done: 'I fix the runtime crash by ensuring TensorFlow/Keras imports happen safely in the Kaggle environment (the protobuf `MessageFactory.GetPrototype` issue commonly occurs with incompatible standalone `keras`/protobuf combos). Then I make the inference path consistent by fitting the tokenizer once on the training split and reusing it for validation/test, avoiding mismatched token IDs that can lead to invalid/NaN outputs. Finally, I ensure predictions are a numeric `(n_test, 30)` array, clipped to `[0,1]`, aligned to `qa_id` order from `test.csv`, and written to a proper `submission.csv`.'
- What this solution (achieved nan) has done: 'I fix the TensorFlow/protobuf crash that prevents the script from running by removing the forced pure-Python protobuf setting and making sure we only use `tensorflow.keras` (not standalone `keras`). Then I make target/feature column selection deterministic and in the exact order of `sample_submission.csv` to avoid accidental misalignment that can yield invalid/NaN scoring behavior. Finally, I ensure the model output is converted to a strict `(n_test, 30)` float array, clipped to `[0,1]`, and written to a correctly formatted `submission.csv` aligned to `test.csv` `qa_id` order.'
- What this solution (achieved nan) has done: 'I fix the runtime crash caused by the protobuf/Keras incompatibility by forcing TensorFlow to use the pure-Python protobuf implementation *before* importing TensorFlow (this is the direct cause of the `MessageFactory.GetPrototype` error in your first cell). I also keep the model and training logic intact but make the target column selection deterministic by taking the exact 30 label columns from `sample_submission.csv` (prevents column-order mismatches that can create invalid/NaN scores). Finally, I ensure the written `submission.csv` has the required columns, correct row order aligned to `test.csv` `qa_id`, and strictly finite clipped predictions in `[0,1]` so Kaggle accepts it and yields a numeric score.'
- What this solution (achieved nan) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced pure-Python protobuf setting (it’s the direct trigger for the `MessageFactory.GetPrototype` AttributeError in this environment) while keeping the TensorFlow/Keras API usage unchanged. I also make the data paths robust to both `../input/...` and the provided `/kaggle/input/...` layout so the script reliably finds the CSVs. Finally, I keep the same model/training logic but ensure the submission is always finite, clipped to `[0,1]`, aligned to `test.csv` `qa_id` order, and written as `submission.csv` with the exact sample-submission column order to avoid NaN/invalid scoring.'
- What this solution (achieved nan) has done: 'I fix the TensorFlow/protobuf crash that happens at import time by forcing TensorFlow to use the pure-Python protobuf implementation *before* importing TensorFlow (this directly addresses the `MessageFactory.GetPrototype` error). I also make target column selection deterministic by always using the exact 30 label columns from `sample_submission.csv`, preventing any column-order mismatch that can lead to invalid/NaN scoring. Finally, I harden the inference path to always use the same tokenizer (fit on training only), guarantee predictions are finite `(n_test, 30)` floats in `[0,1]`, and write a valid `submission.csv`.'
- What this solution (achieved nan) has done: 'I fix the runtime crash at TensorFlow import by removing the forced pure-Python protobuf implementation, which is what triggers the `MessageFactory.GetPrototype` AttributeError in this Kaggle environment. I also make text-column selection consistent between train and test by deriving `text_base_cols` from the actual training feature columns (not after dropping/altering), preventing missing-column issues during test preprocessing. Finally, I keep your model/training logic intact but harden submission creation by guaranteeing finite `(n_test, 30)` predictions in `[0,1]` and writing `submission.csv` with the exact `sample_submission.csv` column order.'
- What this solution (achieved nan) has done: 'I fix the TensorFlow import crash causing the `MessageFactory.GetPrototype` error by setting a protobuf compatibility environment variable before importing TensorFlow, without changing the model/training logic. I also make target column selection deterministic using `sample_submission.csv` (so the 30 outputs always align), and ensure the same tokenizer (fit on train split only) is reused for validation/test to avoid inconsistent token IDs. Finally, I harden prediction shaping/clipping so the pipeline always writes a valid `submission.csv` with finite values in `[0,1]` and the exact required column order.'
- What this solution (achieved nan) has done: 'I fix the immediate runtime crash by removing the protobuf pure-Python override that’s triggering the `MessageFactory.GetPrototype` error when importing TensorFlow in this environment. I keep your model/training logic intact, but make label/feature column selection deterministic by always using the 30 targets in `sample_submission.csv` order (avoids subtle misalignment that can yield invalid/NaN scoring). Finally, I harden inference to guarantee the tokenizer is fit once on the train split and reused everywhere, and ensure predictions are finite `(n_test, 30)` floats clipped to `[0,1]` before writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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


def get_vars_and_targets(train_data, test_data):
    target_cols = sorted(
        list(set(train_data.columns).difference(set(test_data.columns)))
    )
    train_cols = sorted(list(set(train_data.columns) - set(target_cols)))
    return train_cols, target_cols


def get_text_cols(frame):
    text_cols = []
    for col in frame.columns:
        if ("title" in col) or ("body" in col) or (col == "answer"):
            text_cols.append(col)
    return text_cols


def _resolve_path(*candidates):
    """Return the first existing path among candidates; raise if none exist."""
    for p in candidates:
        if p is None:
            continue
        pp = Path(p)
        if pp.exists():
            return str(pp)
    raise FileNotFoundError(f"None of the candidate paths exist: {candidates}")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def fit_tokenizer_on_frame(frame, text_cols):
    tokenizer = Tokenizer()
    for col in text_cols:
        tokenizer.fit_on_texts(frame[col].fillna("").astype(str).values)
    return tokenizer


def texts_to_sequences_inplace(frame, tokenizer, text_cols):
    frame = frame.copy()
    tokenized_cols = []
    for col in text_cols:
        tok_col = col + "_tokenized"
        tokenized_cols.append(tok_col)
        frame[tok_col] = tokenizer.texts_to_sequences(
            frame[col].fillna("").astype(str).values
        )
        frame.drop([col], inplace=True, axis=1)
    return frame, tokenized_cols


def build_padded_inputs(frame, tokenized_cols, maxlen=200):
    padded = []
    for col in tokenized_cols:
        padded.append(pad_sequences(frame[col], maxlen=maxlen, padding="pre"))
    return padded


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
    train_path = _resolve_path(
        "../input/google-quest-challenge/train.csv",
        "/kaggle/input/google-quest-challenge/train.csv",
        "/kaggle/input/train.csv",
        "/kaggle/data/google-quest-challenge/train.csv",
        "/kaggle/data/train.csv",
    )
    test_path = _resolve_path(
        "../input/google-quest-challenge/test.csv",
        "/kaggle/input/google-quest-challenge/test.csv",
        "/kaggle/input/test.csv",
        "/kaggle/data/google-quest-challenge/test.csv",
        "/kaggle/data/test.csv",
    )
    sample_path = _resolve_path(
        "../input/google-quest-challenge/sample_submission.csv",
        "/kaggle/input/google-quest-challenge/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/google-quest-challenge/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    )

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    train_data = prepare_data(train_data)
    test_data = prepare_data(test_data)

    sample_sub = pd.read_csv(sample_path, encoding="utf-8")
    labels = list(sample_sub.columns[1:].values)
    target_cols = labels

    train_cols, _ = get_vars_and_targets(train_data, test_data)

    np.random.seed(42)
    tf.random.set_seed(42)

    text_base_cols = get_text_cols(train_data.loc[:, train_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        train_data.loc[:, train_cols],
        train_data.loc[:, target_cols],
        test_size=0.25,
        random_state=42,
    )

    tokenizer = fit_tokenizer_on_frame(X_train, text_base_cols)

    X_train_tok, tokenized_cols = texts_to_sequences_inplace(
        X_train, tokenizer, text_base_cols
    )
    X_test_tok, _ = texts_to_sequences_inplace(X_test, tokenizer, text_base_cols)

    X_train_padded = build_padded_inputs(X_train_tok, tokenized_cols, maxlen=200)
    X_test_padded = build_padded_inputs(X_test_tok, tokenized_cols, maxlen=200)

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
        eval_res = model.evaluate(X_test_padded, y_test.values, verbose=0)
        print("Eval:", eval_res)
    except Exception as e:
        print("Evaluation failed:", repr(e))

    test_ids = test_data["qa_id"].values
    test_feats = test_data.drop(["qa_id"], axis=1)

    test_feats_tok, _ = texts_to_sequences_inplace(
        test_feats, tokenizer, text_base_cols
    )
    test_padded = build_padded_inputs(test_feats_tok, tokenized_cols, maxlen=200)

    predictions = model.predict(test_padded, verbose=0)
    if isinstance(predictions, list):
        predictions = predictions[0]
    predictions = np.asarray(predictions, dtype=np.float32)

    n_test = len(test_ids)

    if predictions.ndim == 1:
        predictions = predictions.reshape((n_test, 30))
    elif (
        predictions.ndim == 2
        and predictions.shape[0] == n_test
        and predictions.shape[1] == 30
    ):
        pass
    else:
        try:
            predictions = predictions.reshape((n_test, 30))
        except Exception:
            predictions = np.resize(predictions, (n_test, 30))

    predictions = np.nan_to_num(predictions, nan=0.5, posinf=1.0, neginf=0.0)
    predictions = np.clip(predictions, 0.0, 1.0)

    submission_data = pd.DataFrame({"qa_id": test_ids})
    for i, col in enumerate(labels):
        submission_data[col] = predictions[:, i]

    submission_data = submission_data[["qa_id"] + labels]
    submission_data.to_csv("submission.csv", index=False)

    print(submission_data.head())
    print("Wrote submission.csv with shape:", submission_data.shape)


if __name__ == "__main__":
    main()
