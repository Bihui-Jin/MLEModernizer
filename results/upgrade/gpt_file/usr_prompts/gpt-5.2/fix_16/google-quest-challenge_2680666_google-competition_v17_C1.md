# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Reshape
from tensorflow.keras.models import Model

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.config.run_functions_eagerly(False)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass




## === cell 1
_CAT_MAP = {
    "LIFE_ARTS": 1,
    "CULTURE": 2,
    "SCIENCE": 3,
    "STACKOVERFLOW": 4,
    "TECHNOLOGY": 5,
}


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
    drop_cols = [c for c in to_drop if c in frame.columns]
    if drop_cols:
        frame = frame.drop(drop_cols, axis=1)

    if "category" in frame.columns:
        cat_norm = frame["category"].map(_CAT_MAP).fillna(0).astype(np.int16)
        frame = frame.drop(["category"], axis=1)
        frame["category_normalized"] = cat_norm
    else:
        frame["category_normalized"] = np.int16(0)

    return frame


def get_vars_and_targets(train_data: pd.DataFrame, test_data: pd.DataFrame):
    target_cols = sorted(
        list(set(train_data.columns).difference(set(test_data.columns)))
    )
    train_cols = [c for c in train_data.columns if c not in target_cols]
    return train_cols, target_cols


def fit_tokenizer(frame: pd.DataFrame, text_cols=None, num_words=50000):
    if text_cols is None:
        text_cols = ["question_title", "question_body", "answer"]
    tok = Tokenizer(num_words=num_words, oov_token="<OOV>")

    texts_concat = np.concatenate(
        [frame[c].fillna("").astype(str).to_numpy() for c in text_cols], axis=0
    )
    tok.fit_on_texts(texts_concat)
    return tok


def texts_to_padded_arrays(
    frame: pd.DataFrame, tokenizer: Tokenizer, maxlen=1000, text_cols=None
):
    """
    Return model input shape (n, 3, maxlen).
    """
    if text_cols is None:
        text_cols = ["question_title", "question_body", "answer"]

    n = len(frame)
    maxlen = int(maxlen)
    X = np.empty((n, len(text_cols), maxlen), dtype=np.int32)

    for j, c in enumerate(text_cols):
        texts = frame[c].fillna("").astype(str).to_numpy()

        seqs = tokenizer.texts_to_sequences(texts)

        X[:, j, :] = pad_sequences(
            seqs,
            maxlen=maxlen,
            padding="post",
            truncating="post",
            value=0,
        ).astype(np.int32, copy=False)

    return np.ascontiguousarray(X, dtype=np.int32)




## === cell 2
def build_model_2(tokenizer, maxlen=1000):
    input_qt = Input(shape=(3, maxlen))
    x = Reshape((3 * maxlen, 1))(input_qt)

    lstm_qt1 = LSTM(
        1024,
        return_sequences=True,
        recurrent_dropout=0.25,
        activation="sigmoid",
        unroll=False,
        implementation=2,
    )(x)
    lstm_qt2 = LSTM(
        512,
        return_sequences=True,
        recurrent_dropout=0.25,
        activation="sigmoid",
        unroll=False,
        implementation=2,
    )(lstm_qt1)
    lstm_qt3 = LSTM(
        512,
        recurrent_dropout=0.0,
        activation="tanh",
        unroll=False,
        implementation=2,
    )(lstm_qt2)

    dense = Dense(512, activation="relu")(lstm_qt3)
    dropout = Dropout(0.2)(dense)
    dense2 = Dense(256, activation="relu")(dropout)
    output = Dense(30, name="outputs")(dense2)

    model = Model(inputs=[input_qt], outputs=[output])

    model.compile(
        optimizer="Adam",
        loss="mean_squared_error",
        metrics=["accuracy"],
        run_eagerly=False,
    )
    return model




## === cell 3
def _resolve_path(*candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the candidate paths exist: {candidates}")


def _make_ds(X, y=None, batch_size=16, training=False):
    opt = tf.data.Options()
    opt.experimental_deterministic = True

    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(X)
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, y))

    ds = ds.with_options(opt)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


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

    sample_sub = pd.read_csv(sample_path, encoding="utf-8")
    target_cols = [c for c in sample_sub.columns if c != "qa_id"]
    text_cols = ["question_title", "question_body", "answer"]

    train_usecols = [
        "qa_id",
        "question_title",
        "question_body",
        "answer",
        "category",
    ] + target_cols
    test_usecols = [
        "qa_id",
        "question_title",
        "question_body",
        "answer",
        "category",
    ]

    train_dtypes = {
        c: "string" for c in ["question_title", "question_body", "answer", "category"]
    }
    test_dtypes = dict(train_dtypes)
    train_dtypes.update({c: "float32" for c in target_cols})

    train_data = pd.read_csv(
        train_path,
        encoding="utf-8",
        usecols=train_usecols,
        dtype=train_dtypes,
    )
    test_data = pd.read_csv(
        test_path,
        encoding="utf-8",
        usecols=test_usecols,
        dtype=test_dtypes,
    )

    train_data = prepare_data(train_data)
    test_data = prepare_data(test_data)

    y_all = train_data.loc[:, target_cols].to_numpy(dtype=np.float32, copy=False)

    X_train_df, X_val_df, y_train, y_val = train_test_split(
        train_data, y_all, test_size=0.25, random_state=SEED
    )

    tokenizer = fit_tokenizer(X_train_df, text_cols=text_cols, num_words=50000)

    maxlen = 200

    X_train_arr = texts_to_padded_arrays(
        X_train_df, tokenizer, maxlen=maxlen, text_cols=text_cols
    )
    X_val_arr = texts_to_padded_arrays(
        X_val_df, tokenizer, maxlen=maxlen, text_cols=text_cols
    )

    del train_data
    del X_train_df, X_val_df

    model = build_model_2(tokenizer, maxlen=maxlen)

    train_ds = _make_ds(X_train_arr, y_train, batch_size=16, training=True)
    val_ds = _make_ds(X_val_arr, y_val, batch_size=16, training=False)

    model.fit(
        train_ds,
        epochs=1,
        validation_data=val_ds,
        verbose=2,
    )

    _ = model.evaluate(val_ds, verbose=0)

    X_test_arr = texts_to_padded_arrays(
        test_data, tokenizer, maxlen=maxlen, text_cols=text_cols
    )
    test_ds = _make_ds(X_test_arr, y=None, batch_size=32, training=False)

    predictions = model.predict(test_ds, verbose=0)
    predictions = np.clip(predictions, 0.0, 1.0)

    submission = pd.DataFrame(predictions, columns=target_cols)
    submission.insert(0, "qa_id", test_data["qa_id"].to_numpy())

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
