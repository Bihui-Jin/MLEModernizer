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

nan

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00288) has done: 'The crash is caused by importing standalone `keras` in this Kaggle environment, which can trigger a protobuf/Keras incompatibility (`MessageFactory.GetPrototype`). I switch all Keras imports to `tensorflow.keras` (same APIs/core logic) to fix runtime. I also fix a major logic bug where you were fitting a separate tokenizer on `X_test`/`test_data` (causing mismatched vocab indices and hurting score); instead, I fit the tokenizer on `X_train` once and reuse it for validation and test. Finally, I ensure predictions are clipped to `[0,1]` and that the submission always has the correct shape `(len(test), 30)` and columns.'
- What this solution (achieved nan) has done: 'I fix the runtime crash by forcing TensorFlow’s bundled Keras and a compatible protobuf implementation before importing TensorFlow/Keras, which avoids the `MessageFactory.GetPrototype` error in this environment. Then I make one score-improving but core-logic-preserving change: actually use the (already computed) `category` column by mapping it to a numeric feature and feeding it through a tiny dense branch that’s added to the existing merged text features (no change to the text RNN architecture or training loop). I also make the test-time feature preparation consistent with train/val (so the category mapping happens for test too) and keep the submission format/shape robust and clipped to `[0,1]`. These changes are minimal, should run end-to-end, and should nudge the score upward toward the target without altering the overall modeling approach.'
- What this solution (achieved 0.05394) has done: 'I fix the runtime crash caused by the protobuf/TensorFlow incompatibility that triggers `MessageFactory.GetPrototype` by forcing the pure-Python protobuf implementation and importing TensorFlow only after that environment variable is set. I also fix a core logic bug where `prepare_data()` was accidentally dropping the `category` column (because it matched `"categ"`), which made the added category branch effectively useless and could lead to inconsistent feature columns. Finally, I make the target column selection deterministic by using `sample_submission.csv` for label order (instead of a set-difference that can scramble targets), ensuring train labels align with the 30 outputs and the submission columns match exactly.'
- What this solution (achieved nan) has done: 'I fix the protobuf/TensorFlow import crash by setting the protobuf env vars *before* any TensorFlow/Keras import and forcing the pure-Python protobuf implementation in a way that consistently works in Kaggle runtimes. I also remove the random-prediction fallback (it can destroy score and is unnecessary once the model runs) and instead fail loudly if shapes are wrong, while still keeping output clipped to `[0,1]` and formatted exactly like `sample_submission.csv`. Core model architecture/training logic stays the same; only runtime stability and submission correctness are addressed. This should run end-to-end and keep your score from being artificially harmed by random fallbacks.'
- What this solution (achieved nan) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory.GetPrototype`) by setting the required protobuf environment variables **before** any TensorFlow import and by ensuring we don’t import any standalone `keras`. I also make the input file paths robust to both `/kaggle/input/...` and the relative `../input/...` layouts so the script runs end-to-end in this environment without manual path edits. Finally, I keep your model/training logic the same, but I add a small safety check to ensure the tokenizer always sees the expected text columns (even if a column is missing) and that the submission is written as a valid `submission.csv` with the exact `sample_submission.csv` column order.'
- What this solution (achieved nan) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf-related environment variables before any TensorFlow/Keras import and by additionally forcing the pure-Python protobuf module via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, which avoids the `MessageFactory.GetPrototype` issue in Kaggle. I also remove the unused standalone `keras` risk by keeping all Keras usage under `tensorflow.keras` (already the intent) and delaying TF import until after env vars are set. Finally, I keep your model/training/inference logic the same, but add one small robustness fix: ensure `category_numeric` is not accidentally fed into the tokenizer (it’s non-text), preventing silent tokenization inconsistencies and stabilizing predictions/submission generation.'
- What this solution (achieved -0.00938) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf environment variables before any TensorFlow import and (critically) importing `google.protobuf` right after setting them, which prevents the `MessageFactory.GetPrototype` error in this Kaggle runtime. I also keep all Keras usage strictly under `tensorflow.keras` (no standalone `keras`) and add a small, score-neutral robustness tweak: use `Tokenizer(oov_token=...)` so unseen words at test time map consistently instead of being dropped. Finally, I keep the existing model/training logic intact and ensure the submission is always written as `submission.csv` with the exact `sample_submission.csv` column order and `[0,1]` clipping.'
- What this solution (achieved nan) has done: 'I fix the immediate crash by ensuring protobuf is forced to the pure-Python implementation *before* anything can import TensorFlow (including via transitive imports), and by clearing any preloaded `google.protobuf`/`tensorflow` modules defensively in-notebook. Then I keep your model/training logic intact but correct one score-hurting input issue: the current text tokenizer accidentally tries to “tokenize” non-text columns like `category_numeric` (it contains “_” but not title/body/answer; however after `apply_tokenizer` you still keep numeric cols alongside tokenized ones—so we explicitly restrict tokenizer fitting/applying to only the intended raw text columns from the original schema). Finally, I keep the same submission formatting while ensuring deterministic target column order from `sample_submission.csv` and clipping to `[0,1]`.'
- What this solution (achieved -0.02433) has done: 'I fix the runtime crash happening before training by avoiding the protobuf implementation mismatch that triggers `MessageFactory.GetPrototype` in this Kaggle environment. Concretely, I stop forcing the pure-Python protobuf runtime and instead force the compiled protobuf implementation (the one TensorFlow is built/tested against here), and I keep all Keras imports under `tensorflow.keras` as you already do. This is a stability-only change (score-neutral) that allows the notebook to run end-to-end and write a valid `submission.csv` with the correct columns and clipped `[0,1]` predictions. I also keep the rest of your data prep/tokenizer/model/training logic unchanged.'
- What this solution (achieved nan) has done: 'I fix the TensorFlow/protobuf import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *before* any TensorFlow/Keras import and by importing `google.protobuf` immediately after setting the environment variables. This is a runtime-stability fix that should restore end-to-end execution in Kaggle without changing your model/training logic. I also keep all Keras usage under `tensorflow.keras` (already the intent) and preserve your tokenizer/model/data pipeline exactly, only adding the minimal import-order adjustments needed. The script still write a correctly formatted `submission.csv` with `[0,1]` clipped predictions and the exact `sample_submission.csv` target column order.'
- What this solution (achieved nan) has done: 'I fix the immediate runtime crash (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by avoiding the pure-Python protobuf override and by not importing/clearing protobuf modules in a way that breaks TensorFlow’s expected compiled protobuf runtime in Kaggle. This is a stability-only change that preserves your core model/data pipeline and should allow the script to run end-to-end. I also keep all Keras usage under `tensorflow.keras` (already true) and ensure the submission is always written as `submission.csv` with the exact `sample_submission.csv` column order and predictions clipped to `[0,1]`. No model architecture/training loop/feature logic is changed beyond the import/runtime fix.'

# 9. Code solution

## === cell 0
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "").lower() == "python":
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)


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
    return 0


def prepare_data(frame):
    to_drop = []
    for col in frame.columns:
        if "user_page" in col or "host" in col or "url" in col or "user_name" in col:
            to_drop.append(col)
    data = frame.drop(to_drop, axis=1)
    return data


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
def fit_tokenizer_on_frame(frame, text_cols):
    tokenizer = Tokenizer(oov_token="__OOV__")
    for col in text_cols:
        if col in frame.columns:
            tokenizer.fit_on_texts(frame[col].fillna("").astype(str))
    return tokenizer


def apply_tokenizer(frame, tokenizer, text_cols):
    frame = frame.copy()
    renamed_cols = []
    for col in text_cols:
        new_col = col + "_tokenized"
        renamed_cols.append(new_col)
        if col in frame.columns:
            frame[new_col] = tokenizer.texts_to_sequences(
                frame[col].fillna("").astype(str)
            )
            frame.drop([col], inplace=True, axis=1)
        else:
            frame[new_col] = [[] for _ in range(len(frame))]
    return frame, renamed_cols


def model_5_rnn(X_train_padded, tokenizer, maxlen=200, use_category=True):
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

    if use_category:
        cat_in = Input(shape=(1,), name="category_numeric")
        cat_feat = Dense(128, activation="relu")(cat_in)
        features.append(cat_feat)
        inputs.append(cat_in)

    merged_dense = Add()(features)
    droput = Dropout(0.2)(merged_dense)
    dense_1 = Dense(64, activation="relu")(droput)
    dense_2 = Dense(32, activation="relu")(dense_1)
    output = Dense(30)(dense_2)

    model = Model(inputs=inputs, outputs=[output])
    model.compile(optimizer="Adam", loss="mean_squared_error", metrics=["accuracy"])
    return model




## === cell 2
def _resolve_input_path(rel_path_under_input: str) -> str:
    p1 = Path("../input") / rel_path_under_input
    p2 = Path("/kaggle/input") / rel_path_under_input
    if p1.exists():
        return str(p1)
    if p2.exists():
        return str(p2)
    return str(p1)


def main():
    train_path = _resolve_input_path("google-quest-challenge/train.csv")
    test_path = _resolve_input_path("google-quest-challenge/test.csv")
    sample_path = _resolve_input_path("google-quest-challenge/sample_submission.csv")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    sample_sub = pd.read_csv(sample_path, encoding="utf-8")
    target_cols = list(sample_sub.columns[1:].values)

    train_data = prepare_data(train_data)
    test_data_prepared = prepare_data(test_data)

    train_cols = [c for c in train_data.columns if c not in target_cols]

    X = train_data.loc[:, train_cols].copy()
    y = train_data.loc[:, target_cols].copy()

    if "category" in X.columns:
        X["category_numeric"] = (
            X["category"].fillna("").astype(str).map(cat_to_numeric).astype(np.float32)
        )
    else:
        X["category_numeric"] = 0.0

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    base_text_cols = get_text_cols(train_data)

    tokenizer = fit_tokenizer_on_frame(X_train, base_text_cols)

    X_train_tok, text_cols = apply_tokenizer(X_train, tokenizer, base_text_cols)
    X_val_tok, _ = apply_tokenizer(X_val, tokenizer, base_text_cols)

    X_train_padded = []
    X_val_padded = []
    for col in text_cols:
        X_train_padded.append(
            pad_sequences(X_train_tok[col], maxlen=200, padding="pre")
        )
        X_val_padded.append(pad_sequences(X_val_tok[col], maxlen=200, padding="pre"))

    X_train_cat = (
        X_train_tok["category_numeric"]
        .fillna(0.0)
        .astype(np.float32)
        .values.reshape(-1, 1)
        if "category_numeric" in X_train_tok.columns
        else np.zeros((len(X_train_tok), 1), dtype=np.float32)
    )
    X_val_cat = (
        X_val_tok["category_numeric"]
        .fillna(0.0)
        .astype(np.float32)
        .values.reshape(-1, 1)
        if "category_numeric" in X_val_tok.columns
        else np.zeros((len(X_val_tok), 1), dtype=np.float32)
    )

    model = model_5_rnn(X_train_padded, tokenizer, maxlen=200, use_category=True)

    model.fit(
        X_train_padded + [X_train_cat],
        y_train.values,
        batch_size=32,
        epochs=1,
        validation_split=0.2,
        verbose=2,
    )

    try:
        print(model.evaluate(X_val_padded + [X_val_cat], y_val.values, verbose=0))
    except Exception as e:
        print("Validation evaluate failed:", repr(e))

    test_ids = test_data["qa_id"].values
    test_features = test_data_prepared.drop(["qa_id"], axis=1).copy()

    if "category" in test_features.columns:
        test_features["category_numeric"] = (
            test_features["category"]
            .fillna("")
            .astype(str)
            .map(cat_to_numeric)
            .astype(np.float32)
        )
    else:
        test_features["category_numeric"] = 0.0

    test_tok, _ = apply_tokenizer(test_features, tokenizer, base_text_cols)

    test_padded = []
    for col in text_cols:
        test_padded.append(pad_sequences(test_tok[col], maxlen=200, padding="pre"))

    test_cat = (
        test_tok["category_numeric"]
        .fillna(0.0)
        .astype(np.float32)
        .values.reshape(-1, 1)
        if "category_numeric" in test_tok.columns
        else np.zeros((len(test_tok), 1), dtype=np.float32)
    )

    predictions = model.predict(test_padded + [test_cat], verbose=0)
    if isinstance(predictions, list):
        predictions = predictions[0]
    predictions = np.asarray(predictions)

    if (
        predictions.ndim != 2
        or predictions.shape[1] != 30
        or predictions.shape[0] != len(test_ids)
    ):
        raise ValueError(
            f"Bad predictions shape: {predictions.shape}, expected ({len(test_ids)}, 30)"
        )

    predictions = np.clip(predictions, 0.0, 1.0)

    submission = pd.DataFrame({"qa_id": test_ids})
    for i, lab in enumerate(target_cols):
        submission[lab] = predictions[:, i]

    submission = submission[["qa_id"] + target_cols]
    submission.to_csv("submission.csv", index=False)

    print(submission.head())
    print("Wrote submission.csv with shape:", submission.shape)


main()
