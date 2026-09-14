# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

-0.0032656002085286

# 6. Current score

nan

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.00229) has done: 'The runtime error comes from importing standalone `keras` in an environment where it conflicts with the installed protobuf/tensorflow stack, so I switch all Keras imports to `tensorflow.keras` (same API/core logic, but stable on Kaggle). I also fix a logic bug where train/test were tokenized with different tokenizers (silently harming predictions) by fitting the tokenizer on train only and reusing it everywhere. Finally, I ensure predictions have the right shape `(n_rows, 30)` and are clipped to `[0,1]` before writing `submission.csv` with the exact sample-submission columns so a valid file is always produced.'
- What this solution (achieved -0.01075) has done: 'I fix the TensorFlow/protobuf runtime crash by forcing the pure-Python protobuf implementation before importing TensorFlow (a common Kaggle fix for `MessageFactory.GetPrototype` errors). I also ensure the training/test feature columns always align by excluding `qa_id` from model inputs deterministically, which avoids subtle column-set issues. Finally, I keep the model and training loop unchanged, but make submission creation robust by always building the submission with `test.csv`’s `qa_id` ordering and `sample_submission.csv` label columns, clipping predictions to `[0,1]`.'
- What this solution (achieved 0.02942) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf implementation environment variables before importing TensorFlow, and add a safe fallback to run without TensorFlow by generating deterministic baseline predictions (so you always get a valid `submission.csv`). I also fix a core logic bug in `model_5_rnn` where it mistakenly creates one input per *training row* instead of one input per *text field* (this currently makes the model construction invalid and/or impossible to train). Finally, I make the target column order deterministic by reading it from `sample_submission.csv` (instead of using a set-derived order) so training and submission columns align, and I remove the broad try/except that was silently replacing real predictions with random values (which is driving your score down). These changes keep the same overall approach (RNN over tokenized title/body/answer with MSE) while making it run end-to-end and improving score stability toward the target.'
- What this solution (achieved nan) has done: 'I fix the protobuf/TensorFlow crash that prevents the RNN from running by forcing the pure-Python protobuf implementation *and* disabling the C++ implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`, and I also add a robust fallback to keep the script running if TF still can’t import. I keep your model architecture/training loop intact, but make the tokenizer fitting deterministic (fit once on the concatenation of all text fields from train) so train/valid/test use the same vocabulary consistently, which should nudge score upward (not a modeling change). Finally, I ensure the submission uses the exact `sample_submission.csv` column order and always writes a valid `submission.csv` with predictions clipped to `[0,1]`.'
- What this solution (achieved nan) has done: 'I fix the TensorFlow/protobuf import crash that currently prevents the script from running by forcing a compatible protobuf version before importing TensorFlow; if TensorFlow still can’t load, the existing baseline fallback still produce a valid `submission.csv`. I also correct a remaining logic bug in `model_5_rnn`: it currently creates `len(X_train_padded)` inputs (i.e., number of samples) instead of one input per text field (3), which break model construction/training; the fix keeps the same RNN architecture per field but iterates over the number of padded feature arrays. Finally, I make the target column selection deterministic by using the `sample_submission.csv` label order (already mostly done) and ensure predictions always end up as a `(n_test, 30)` array clipped to `[0,1]` before writing the submission.'
- What this solution (achieved nan) has done: 'I fix the runtime crash caused by an incompatible protobuf implementation by forcing the C++ protobuf backend (the Python backend is what triggers the `MessageFactory.GetPrototype` error in this environment) before importing TensorFlow. I also correct a small but critical logic bug in `model_5_rnn`: it currently creates one input per *sample* instead of per *text field*, which can make model construction invalid; the fix preserves the same architecture but sets `n_inputs = len(X_train_padded)`. Finally, I make submission generation robust by ensuring the tokenizer is fit once and reused, predictions always end up as an `(n_test, 30)` array, clipped to `[0,1]`, and written to `submission.csv` with the exact `sample_submission.csv` column order.'
- What this solution (achieved nan) has done: 'The crash happens before your try/except because TensorFlow is being imported with an incompatible protobuf backend; setting the protobuf environment variables *before* importing TensorFlow fixes this in Kaggle and keeps your model code unchanged. I also fix a real logic bug in `model_5_rnn`: it currently creates one input per padded array row count confusion risk; it should create one input per text field (3) by iterating over the number of padded feature arrays, preserving the same Embedding→RNN→Dense blocks and Add-merge. Finally, I make `transform_texts` non-destructive (work on a copy) so you don’t accidentally mutate shared frames, and ensure submission column order exactly matches `sample_submission.csv` with predictions clipped to `[0,1]` and shaped `(n_test, 30)`.'
- What this solution (achieved nan) has done: 'I fix the runtime crash that happens on importing TensorFlow by switching the protobuf environment setting to use the C++ implementation (the current `python` setting is what commonly triggers `MessageFactory.GetPrototype` errors on Kaggle). I also fix a critical logic bug in `model_5_rnn` where it incorrectly creates one model input per training sample instead of one per text field (3 inputs: title/body/answer), which otherwise makes model construction infeasible. Finally, I make target/feature column alignment deterministic using `sample_submission.csv` for label order and ensure the submission is always written as a valid `submission.csv` with predictions shaped `(n_test, 30)` and clipped to `[0,1]`.'
- What this solution (achieved nan) has done: 'Your current score is `nan`, which strongly suggests the submission file was invalid (wrong row count or NaNs), not that the model is “bad.” The biggest issue is that you build the submission using `test.csv` (19,550 rows) but Kaggle’s Google QUEST test set is 607 rows (matching `sample_submission.csv`’s 608 lines including header), so the submission shape won’t match and can score as invalid/NaN. I minimally fix data-path ambiguity by reading train/test/sample_submission from the same `../input/google-quest-challenge/` directory and then *filtering and ordering* the test rows to exactly match `sample_submission.qa_id`. I also guard against NaNs in predictions (replace with column means / 0.5) so the CSV is always numerically valid and within [0,1], keeping your model and training loop unchanged.'
- What this solution (achieved nan) has done: 'Your `nan` score is most consistent with an invalid submission (most commonly: wrong number/order of rows vs `sample_submission`, wrong/NaN values, or mismatched column order). I keep your model/training logic intact, but make submission construction strictly follow `sample_submission` row order and column order, and add hard guards to ensure predictions are finite float values in `[0,1]` with exact shape `(len(sample_submission), 30)`. I also fix a subtle preprocessing issue: `prepare_data()` currently drops the `category` column (because it matches `"categ"`), which can change which columns are considered “text columns” and can lead to inconsistent feature sets; we stop dropping `category` and keep it (even if unused) to reduce accidental schema drift. Finally, I ensure that `test_frame` passed into `transform_texts` contains only the same feature columns as training (so it can’t pick up extra non-text columns and silently alter behavior).'
- What this solution (achieved nan) has done: 'Your `nan` score is almost certainly from an invalid submission shape: you currently force `test.csv` to match `sample_submission.qa_id`, but in this dataset `test.csv` has 19,550 rows while `sample_submission.csv` has 608 rows, so that alignment fail or produce the wrong row count for Kaggle. I make the submission follow the competition’s actual test set (`test.csv`) row order and use `sample_submission.csv` only for the *label column order*, which produces a valid `(19550, 31)` submission every time. I also make target-column extraction deterministic using the known label list (instead of a set-difference), and add a small safety fix to `cat_to_numeric` to avoid introducing NaNs if it’s ever used. Core model/training/feature logic stays the same; these are correctness fixes that should move you from `nan` to a real score (likely near your previous non-NaN runs), which is the biggest step toward the target.'
- What this solution (achieved nan) has done: 'Your `nan` score strongly suggests Kaggle is treating the submission as invalid, and the most likely cause here is a row-count mismatch: you are writing predictions for `test.csv` (19,550 rows) but the competition’s `sample_submission.csv` indicates the expected test size is 607/608 rows (depending on header counting). I minimally fix this by aligning the test rows to exactly the `qa_id` order in `sample_submission.csv` before predicting and writing the submission, while keeping your model, tokenization, and training loop intact. I also add a small safety guard to ensure `test_frame` contains all required text columns (filled with empty strings if missing) so transform/padding can’t silently break and produce NaNs. These changes should convert the `nan` into a real score and move it toward your target band without changing core modeling behavior.'
- What this solution (achieved nan) has done: 'Your `nan` score is almost certainly because the script is reindexing `test.csv` to `sample_submission.qa_id`, producing mostly-missing rows and then predicting on 608 rows instead of the required 19,550 test rows (so Kaggle marks it invalid). I make the submission row order/length follow `test.csv` (19,550 rows) and use `sample_submission.csv` only for the label column order, which should immediately convert `nan` into a real score. To keep your core model/training logic intact but improve stability, I also make `get_vars_and_targets()` deterministic (no set-order randomness) and ensure the tokenizer is fit on the three training text fields (not on concatenated text) so it matches how you later tokenize each field. Finally, I keep the existing clipping/finite guards so the CSV is always valid and within `[0,1]`.'
- What this solution (achieved nan) has done: 'Your `nan` score is most consistent with an invalid submission (typically NaNs/inf in the CSV), and your current model can easily emit NaNs because the RNN uses `sigmoid` activations with no output constraint and can numerically blow up. I make the smallest change that preserves your core architecture/training loop but stabilizes outputs by setting the final layer activation to `sigmoid` (keeps outputs in `[0,1]` and improves Spearman stability). I also harden the prediction sanitization to replace any remaining non-finite values with per-label training means before clipping. These changes should convert `nan` into a real score and move it toward your (slightly negative) target band without changing the overall approach.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TF_AVAILABLE = True
TF_IMPORT_ERROR = None
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Dropout, Embedding, Input
    from tensorflow.keras.layers import SimpleRNNCell, Add, RNN
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.initializers import RandomUniform
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e)


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
        if "user_page" in col or "host" in col or "url" in col or "user_name" in col:
            to_drop.append(col)
    data = frame.drop(to_drop, axis=1)
    return data


def get_vars_and_targets(train_data, test_data):
    target_cols = [c for c in train_data.columns if c not in test_data.columns]
    train_cols = [c for c in train_data.columns if c in test_data.columns]
    return train_cols, target_cols


def get_text_cols(frame):
    text_cols = []
    for col in frame.columns:
        if "title" in col or "body" in col or col == "answer":
            text_cols.append(col)
    return text_cols




## === cell 1
def transform_texts(frame, tokenizer=None, fit=False):
    frame = frame.copy()

    text_cols = get_text_cols(frame)

    if tokenizer is None:
        tokenizer = Tokenizer()

    if fit:
        for col in text_cols:
            tokenizer.fit_on_texts(frame[col].fillna("").astype(str))

    renamed_cols = []
    for col in text_cols:
        renamed_cols.append(col + "_tokenized")
        frame[col + "_tokenized"] = tokenizer.texts_to_sequences(
            frame[col].fillna("").astype(str)
        )
        frame.drop([col], inplace=True, axis=1)

    return [frame, renamed_cols, tokenizer]


def convert_to_arrays(frame):
    arrays = []
    for index in frame.index:
        merged = []
        for each in [
            "question_title_tokenized_padded",
            "question_body_tokenized_padded",
            "answer_tokenized_padded",
        ]:
            arr = np.array(frame.loc[index, each])
            merged.append(arr)
        arrays.append(np.array(merged))
    return np.array(arrays)


def model_5_rnn(X_train_padded, tokenizer, maxlen=200):
    """
    Create one input per text field (len(X_train_padded) == 3).
    Core Embedding->RNN->Dense blocks preserved.
    """
    initializer = RandomUniform(seed=69)

    inputs = []
    features = []

    n_inputs = len(X_train_padded)

    for _ in range(n_inputs):
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

    output = Dense(30, activation="sigmoid")(dense_2)

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
    sample_sub = pd.read_csv(sample_path, encoding="utf-8")

    labels = list(sample_sub.columns[1:].values)

    train_data = prepare_data(train_data)
    test_data = prepare_data(test_data)

    target_cols = [c for c in labels if c in train_data.columns]

    train_cols = [
        c for c in train_data.columns if c not in target_cols and c != "qa_id"
    ]
    train_cols = sorted(train_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        train_data.loc[:, train_cols],
        train_data.loc[:, target_cols],
        test_size=0.25,
        random_state=42,
    )

    test_ids = test_data["qa_id"].values
    test_frame = test_data.loc[:, ["qa_id"] + train_cols].drop(["qa_id"], axis=1)

    for required_col in ["question_title", "question_body", "answer"]:
        if required_col not in test_frame.columns:
            test_frame[required_col] = ""

    if TF_AVAILABLE:
        text_base_cols = get_text_cols(X_train)
        tokenizer = Tokenizer()

        for col in text_base_cols:
            tokenizer.fit_on_texts(X_train[col].fillna("").astype(str).values)

        X_train_tx, text_cols, tokenizer = transform_texts(
            X_train, tokenizer=tokenizer, fit=False
        )
        X_test_tx, _, _ = transform_texts(X_test, tokenizer=tokenizer, fit=False)

        X_train_padded = []
        X_test_padded = []
        for i in range(len(text_cols)):
            X_train_padded.append(
                pad_sequences(X_train_tx[text_cols[i]], maxlen=200, padding="pre")
            )
            X_test_padded.append(
                pad_sequences(X_test_tx[text_cols[i]], maxlen=200, padding="pre")
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

        test_frame_tx, _, _ = transform_texts(
            test_frame, tokenizer=tokenizer, fit=False
        )

        test_padded = []
        for i in range(len(text_cols)):
            test_padded.append(
                pad_sequences(test_frame_tx[text_cols[i]], maxlen=200, padding="pre")
            )

        predictions = model.predict(test_padded, verbose=0)

        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        predictions = np.asarray(predictions)
    else:
        print(
            "WARNING: TensorFlow import failed; writing baseline submission. Error:",
            TF_IMPORT_ERROR,
        )
        col_means = (
            train_data[target_cols].mean(axis=0).reindex(labels).fillna(0.5).values
        )
        predictions = np.tile(col_means, (len(test_frame), 1))

    predictions = np.asarray(predictions, dtype=np.float32)

    n_rows = len(test_frame)
    n_labels = len(labels)

    if predictions.ndim != 2 or predictions.shape != (n_rows, n_labels):
        if predictions.size == n_rows * n_labels:
            predictions = predictions.reshape(n_rows, n_labels)
        else:
            col_means = (
                y_train.reindex(columns=labels)
                .mean(axis=0)
                .fillna(0.5)
                .values.astype(np.float32)
            )
            predictions = np.tile(col_means, (n_rows, 1)).astype(np.float32)

    col_means = (
        y_train.reindex(columns=labels)
        .mean(axis=0)
        .fillna(0.5)
        .values.astype(np.float32)
    )
    if not np.isfinite(predictions).all():
        predictions = np.where(
            np.isfinite(predictions), predictions, col_means[None, :]
        )

    predictions = np.clip(predictions, 0.0, 1.0)

    submission = pd.DataFrame({"qa_id": test_ids})
    submission[labels] = predictions
    submission.to_csv("submission.csv", index=False)

    print(submission.head())
    print("Wrote submission.csv with shape:", submission.shape)


main()
