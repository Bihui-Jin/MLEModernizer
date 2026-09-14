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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import random
import html
import warnings

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model

from tqdm.auto import tqdm
from sklearn.model_selection import KFold

from transformers import (
    AutoTokenizer,
    TFAutoModel,
    AutoConfig,
)

warnings.filterwarnings("ignore")

print("Imports OK")




## === cell 1
seed = 13
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass




## === cell 2
def _resolve_competition_path():
    """
    Kaggle can mount the dataset under different roots depending on environment.
    Try common Kaggle paths first, then fall back to the user-provided /kaggle/data layout.
    """
    candidates = [
        "../input/google-quest-challenge/",
        "../input/google-quest-challenge/google-quest-challenge/",
        "/kaggle/input/google-quest-challenge/",
        "/kaggle/input/google-quest-challenge/google-quest-challenge/",
        "/kaggle/data/google-quest-challenge/",
        "/kaggle/data/google-quest-challenge/google-quest-challenge/",
        "/kaggle/data/input/google-quest-challenge/",
        "/kaggle/data/input/google-quest-challenge/google-quest-challenge/",
        "/kaggle/data/input/google-quest-challenge/",
        "/kaggle/data/input/google-quest-challenge/google-quest-challenge/",
    ]
    for p in candidates:
        if os.path.exists(os.path.join(p, "train.csv")):
            return p
    raise FileNotFoundError(
        "Could not locate google-quest-challenge train.csv under known paths."
    )


def get_data():
    print("getting test and train data...")
    path = _resolve_competition_path()
    train = pd.read_csv(os.path.join(path, "train.csv"))
    test = pd.read_csv(os.path.join(path, "test.csv"))
    submission = pd.read_csv(os.path.join(path, "sample_submission.csv"))

    y = train[train.columns[11:]]
    X = train[["question_title", "question_body", "answer"]].copy()
    X_test = test[["question_title", "question_body", "answer"]].copy()

    for col in ["question_body", "question_title", "answer"]:
        X[col] = X[col].astype(str).str.replace("\r", "\n", regex=False)
        X_test[col] = X_test[col].astype(str).str.replace("\r", "\n", regex=False)
        X[col] = X[col].astype(str).map(html.unescape)
        X_test[col] = X_test[col].astype(str).map(html.unescape)

    return X, X_test, y, train, test, submission




## === cell 3
_TOKENIZER_CACHE = {}


def get_tokenizer(model_name):
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]
    print(f"getting tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    _TOKENIZER_CACHE[model_name] = tokenizer
    return tokenizer




## === cell 4
def fix_length(
    tokens,
    max_sequence_length=512,
    q_max_len=254,
    a_max_len=254,
    model_type="questions",
):
    if model_type == "questions":
        length = len(tokens)
        if length > max_sequence_length:
            tokens = tokens[: max_sequence_length - 1]
        return tokens
    else:
        question_tokens, answer_tokens = tokens
        q_len = len(question_tokens)
        a_len = len(answer_tokens)
        if q_len + a_len + 3 > max_sequence_length:
            if a_max_len <= a_len and q_max_len <= q_len:
                q_new_len_head = q_max_len // 2
                question_tokens = (
                    question_tokens[:q_new_len_head] + question_tokens[-q_new_len_head:]
                )
                a_new_len_head = a_max_len // 2
                answer_tokens = (
                    answer_tokens[:a_new_len_head] + answer_tokens[-a_new_len_head:]
                )
            elif q_len <= a_len and q_len < q_max_len:
                a_max_len = a_max_len + (q_max_len - q_len - 1)
                a_new_len_head = a_max_len // 2
                answer_tokens = (
                    answer_tokens[:a_new_len_head] + answer_tokens[-a_new_len_head:]
                )
            elif a_len < q_len:
                q_max_len = q_max_len + (a_max_len - a_len - 1)
                q_new_len_head = q_max_len // 2
                question_tokens = (
                    question_tokens[:q_new_len_head] + question_tokens[-q_new_len_head:]
                )

        return question_tokens, answer_tokens




## === cell 5
def transformer_inputs(
    title, question, answer, tokenizer, model_type="questions", MAX_SEQUENCE_LENGTH=512
):
    cls_tok = tokenizer.cls_token if tokenizer.cls_token is not None else "[CLS]"
    sep_tok = tokenizer.sep_token if tokenizer.sep_token is not None else "[SEP]"
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    if model_type == "questions":
        question = f"{title} {sep_tok} {question}"
        question_tokens = tokenizer.tokenize(question)
        question_tokens = fix_length(question_tokens, model_type=model_type)

        ids_q = tokenizer.convert_tokens_to_ids([cls_tok] + question_tokens)
        padded_ids = (ids_q + [pad_id] * (MAX_SEQUENCE_LENGTH - len(ids_q)))[
            :MAX_SEQUENCE_LENGTH
        ]

        token_type_ids = ([0] * MAX_SEQUENCE_LENGTH)[:MAX_SEQUENCE_LENGTH]
        attention_mask = ([1] * len(ids_q) + [0] * (MAX_SEQUENCE_LENGTH - len(ids_q)))[
            :MAX_SEQUENCE_LENGTH
        ]
        return padded_ids, token_type_ids, attention_mask
    else:
        question = f"{title} {sep_tok} {question}"
        question_tokens = tokenizer.tokenize(question)
        answer_tokens = tokenizer.tokenize(answer)
        question_tokens, answer_tokens = fix_length(
            tokens=(question_tokens, answer_tokens), model_type=model_type
        )

        ids = tokenizer.convert_tokens_to_ids(
            [cls_tok] + question_tokens + [sep_tok] + answer_tokens + [sep_tok]
        )
        padded_ids = (ids + [pad_id] * (MAX_SEQUENCE_LENGTH - len(ids)))[
            :MAX_SEQUENCE_LENGTH
        ]

        token_type_ids = (
            [0] * (1 + len(question_tokens) + 1)
            + [1] * (len(answer_tokens) + 1)
            + [0] * (MAX_SEQUENCE_LENGTH - len(ids))
        )[:MAX_SEQUENCE_LENGTH]

        attention_mask = ([1] * len(ids) + [0] * (MAX_SEQUENCE_LENGTH - len(ids)))[
            :MAX_SEQUENCE_LENGTH
        ]
        return padded_ids, token_type_ids, attention_mask




## === cell 6
_FULL_INPUT_CACHE = (
    {}
)  # (tokenizer_name, model_type, split_name) -> (input_ids, attn, type_ids)


def _batch_encode_questions(
    df, tokenizer, max_len=512, batch_size=4096, show_pbar=True
):
    sep_tok = tokenizer.sep_token if tokenizer.sep_token is not None else "[SEP]"
    texts = (
        df["question_title"].astype(str)
        + " "
        + sep_tok
        + " "
        + df["question_body"].astype(str)
    ).tolist()

    n = len(texts)
    input_ids = np.empty((n, max_len), dtype=np.int32)
    attn = np.empty((n, max_len), dtype=np.int32)

    type_ids = np.zeros((n, max_len), dtype=np.int32)

    rng = range(0, n, batch_size)
    if show_pbar:
        rng = tqdm(rng, total=(n + batch_size - 1) // batch_size)

    for start in rng:
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        end = start + len(batch)
        input_ids[start:end] = np.asarray(enc["input_ids"], dtype=np.int32)
        attn[start:end] = np.asarray(enc["attention_mask"], dtype=np.int32)
        if "token_type_ids" in enc:
            type_ids[start:end] = np.asarray(enc["token_type_ids"], dtype=np.int32)

    return input_ids, attn, type_ids


def _apply_fixlen_ids_pair(q_ids, a_ids, max_len=512, q_max_len=254, a_max_len=254):
    q_len = len(q_ids)
    a_len = len(a_ids)

    if q_len + a_len + 3 > max_len:
        if a_max_len <= a_len and q_max_len <= q_len:
            q_new_len_head = q_max_len // 2
            q_ids = q_ids[:q_new_len_head] + q_ids[-q_new_len_head:]
            a_new_len_head = a_max_len // 2
            a_ids = a_ids[:a_new_len_head] + a_ids[-a_new_len_head:]
        elif q_len <= a_len and q_len < q_max_len:
            a_max_len = a_max_len + (q_max_len - q_len - 1)
            a_new_len_head = a_max_len // 2
            a_ids = a_ids[:a_new_len_head] + a_ids[-a_new_len_head:]
        elif a_len < q_len:
            q_max_len = q_max_len + (a_max_len - a_len - 1)
            q_new_len_head = q_max_len // 2
            q_ids = q_ids[:q_new_len_head] + q_ids[-q_new_len_head:]
    return q_ids, a_ids


def _batch_encode_qa_preserve_fixlen(
    df, tokenizer, max_len=512, batch_size=2048, show_pbar=True
):
    sep_tok = tokenizer.sep_token if tokenizer.sep_token is not None else "[SEP]"
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    cls_id = (
        int(tokenizer.cls_token_id)
        if tokenizer.cls_token_id is not None
        else int(tokenizer.convert_tokens_to_ids("[CLS]"))
    )
    sep_id = (
        int(tokenizer.sep_token_id)
        if tokenizer.sep_token_id is not None
        else int(tokenizer.convert_tokens_to_ids("[SEP]"))
    )

    titles = df["question_title"].astype(str).values
    bodies = df["question_body"].astype(str).values
    answers = df["answer"].astype(str).values

    n = len(df)
    input_ids = np.empty((n, max_len), dtype=np.int32)
    attn = np.empty((n, max_len), dtype=np.int32)
    type_ids = np.empty((n, max_len), dtype=np.int32)

    q_texts = [f"{t} {sep_tok} {b}" for t, b in zip(titles, bodies)]
    a_texts = answers.tolist()

    rng = range(0, n, batch_size)
    if show_pbar:
        rng = tqdm(rng, total=(n + batch_size - 1) // batch_size)

    for start in rng:
        end = min(n, start + batch_size)

        q_tok = tokenizer(
            q_texts[start:end],
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        a_tok = tokenizer(
            a_texts[start:end],
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]

        for i, (q_ids, a_ids) in enumerate(zip(q_tok, a_tok)):
            q_ids, a_ids = _apply_fixlen_ids_pair(q_ids, a_ids, max_len=max_len)

            ids = [cls_id] + q_ids + [sep_id] + a_ids + [sep_id]
            ids_len = len(ids)

            if ids_len < max_len:
                ids_padded = ids + [pad_id] * (max_len - ids_len)
            else:
                ids_padded = ids[:max_len]
                ids_len = max_len

            seg = (
                [0] * (1 + len(q_ids) + 1)
                + [1] * (len(a_ids) + 1)
                + [0] * (max_len - len(ids))
            )[:max_len]
            mask = (
                [1] * min(len(ids), max_len) + [0] * (max_len - min(len(ids), max_len))
            )[:max_len]

            row = start + i
            input_ids[row] = np.asarray(ids_padded, dtype=np.int32)
            type_ids[row] = np.asarray(seg, dtype=np.int32)
            attn[row] = np.asarray(mask, dtype=np.int32)

    return input_ids, attn, type_ids


def input_data(df, tokenizer, model_type="questions", show_pbar=True):
    print(f"generating {model_type} input for transformer...")
    if model_type == "questions":
        return _batch_encode_questions(
            df, tokenizer, max_len=512, batch_size=4096, show_pbar=show_pbar
        )
    else:
        return _batch_encode_qa_preserve_fixlen(
            df, tokenizer, max_len=512, batch_size=2048, show_pbar=show_pbar
        )




## === cell 7
_MODEL_CACHE = {}


def get_model(name):
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]
    config = AutoConfig.from_pretrained(name, output_hidden_states=True)
    model = TFAutoModel.from_pretrained(name, config=config)
    _MODEL_CACHE[name] = model
    return model




## === cell 8
def create_model(name="xlnet-base-cased", model_type="questions"):
    print(f"creating model {name} ({model_type})...")
    K.clear_session()
    max_seq_length = 512

    input_tokens = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name="input_tokens"
    )
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name="input_mask"
    )
    input_segment = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name="input_segment"
    )

    base = get_model(name)

    def _forward(inputs):
        tok, msk, seg = inputs

        if name == "xlnet-base-cased":
            out = base.call(input_ids=tok, attention_mask=msk, training=False)
        elif name == "roberta-base" and model_type != "questions":
            out = base.call(input_ids=tok, attention_mask=msk, training=False)
        else:
            out = base.call(
                input_ids=tok,
                attention_mask=msk,
                token_type_ids=seg,
                training=False,
            )
        return out.hidden_states

    hidden_states = tf.keras.layers.Lambda(
        _forward,
        name=f"{name}_forward",
        output_shape=[(max_seq_length, 768)] * 13,
    )([input_tokens, input_mask, input_segment])

    h12 = tf.keras.layers.Reshape((1, 768))(hidden_states[-1][:, 0, :])
    h11 = tf.keras.layers.Reshape((1, 768))(hidden_states[-2][:, 0, :])
    h10 = tf.keras.layers.Reshape((1, 768))(hidden_states[-3][:, 0, :])
    h09 = tf.keras.layers.Reshape((1, 768))(hidden_states[-4][:, 0, :])
    concat_hidden = tf.keras.layers.Concatenate(axis=2)([h12, h11, h10, h09])

    x = GlobalAveragePooling1D()(concat_hidden)
    x = Dropout(0.2)(x)

    if model_type == "answers":
        output = Dense(9, activation="sigmoid")(x)
    elif model_type == "questions":
        output = Dense(21, activation="sigmoid")(x)
    else:
        output = Dense(30, activation="sigmoid")(x)

    model = Model(inputs=[input_tokens, input_mask, input_segment], outputs=output)
    return model




## === cell 9
class data_generator:
    def __init__(self, X, X_test, tokenizer, type_):
        tok_name = getattr(tokenizer, "name_or_path", str(id(tokenizer)))

        key_test = (tok_name, type_, "test")
        if key_test in _FULL_INPUT_CACHE:
            tokens, masks, segments = _FULL_INPUT_CACHE[key_test]
        else:
            tokens, masks, segments = input_data(
                X_test, tokenizer, type_, show_pbar=False
            )
            _FULL_INPUT_CACHE[key_test] = (tokens, masks, segments)

        self.test_data_full = {
            "input_tokens": tokens,
            "input_mask": masks,
            "input_segment": segments,
        }

        key_train = (tok_name, type_, "train")
        if key_train in _FULL_INPUT_CACHE:
            tr_tokens, tr_masks, tr_segments = _FULL_INPUT_CACHE[key_train]
        else:
            tr_tokens, tr_masks, tr_segments = input_data(
                X, tokenizer, type_, show_pbar=False
            )
            _FULL_INPUT_CACHE[key_train] = (tr_tokens, tr_masks, tr_segments)

        self.train_full = {
            "input_tokens": tr_tokens,
            "input_mask": tr_masks,
            "input_segment": tr_segments,
        }

        self.X = X
        self.tokenizer = tokenizer
        self.type_ = type_

    def generate_data(self, tr, cv, y, name="xlnet-base-cased", model_type="questions"):
        tr_tokens = self.train_full["input_tokens"][tr]
        tr_masks = self.train_full["input_mask"][tr]
        tr_segments = self.train_full["input_segment"][tr]

        cv_tokens = self.train_full["input_tokens"][cv]
        cv_masks = self.train_full["input_mask"][cv]
        cv_segments = self.train_full["input_segment"][cv]

        train_data = {
            "input_tokens": tr_tokens,
            "input_mask": tr_masks,
            "input_segment": tr_segments,
        }
        cv_data = {
            "input_tokens": cv_tokens,
            "input_mask": cv_masks,
            "input_segment": cv_segments,
        }

        if model_type == "questions":
            y_tr = y.values[tr, :21]
            y_cv = y.values[cv, :21]
        elif model_type == "answers":
            y_tr = y.values[tr, 21:]
            y_cv = y.values[cv, 21:]
        else:
            y_tr = y.values[tr]
            y_cv = y.values[cv]

        return train_data, cv_data, y_tr, y_cv

    def get_test_data(self, model_name, model_type):
        return self.test_data_full




## === cell 10
def optimize_ranks(preds, unique_labels):
    print("optimizing the predicted values...")
    preds = np.asarray(preds, dtype=np.float32)
    bins = np.asarray(unique_labels)

    out = np.empty_like(preds, dtype=np.float32)
    for i in range(preds.shape[1]):
        idx = np.digitize(preds[:, i], bins=bins, right=False)
        if np.unique(idx).size == 1:
            out[:, i] = preds[:, i]
        else:
            idx = np.clip(idx, 0, len(bins) - 1)
            out[:, i] = bins[idx].astype(np.float32)
    return np.clip(out, 0.0, 1.0)




## === cell 11
def get_exp_labels(train):
    X = train.iloc[:, 11:]
    unique_labels = np.unique(X.values)
    denominator = 60
    q = np.arange(0, 101, 100 / denominator)
    exp_labels = np.percentile(unique_labels, q)  # Generating the 60 bins.
    return exp_labels




## === cell 12
def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    n = x.size
    if n == 0:
        return x.astype(np.float64)

    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0  # 1-based average rank
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def _spearmanr_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 2:
        return np.nan

    ra = _rankdata_average_ties(a)
    rb = _rankdata_average_ties(b)

    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt(np.sum(ra * ra) * np.sum(rb * rb))
    if denom == 0.0:
        return np.nan
    return float(np.sum(ra * rb) / denom)


def compute_spearmanr_ignore_nan(trues, preds):
    trues = np.asarray(trues)
    preds = np.asarray(preds)
    rhos_list = []
    for tcol, pcol in zip(np.transpose(trues), np.transpose(preds)):
        rhos_list.append(_spearmanr_np(tcol, pcol))
    return np.nanmean(rhos_list)


class SpearmanR(tf.keras.metrics.Metric):
    def __init__(self, name="spearmanr", **kwargs):
        super().__init__(name=name, **kwargs)
        self.rho = self.add_weight(name="rho", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        val = tf.py_function(
            func=compute_spearmanr_ignore_nan,
            inp=[y_true, y_pred],
            Tout=tf.float32,
        )
        val = tf.reshape(val, ())  # ensure scalar
        self.rho.assign(val)

    def result(self):
        return self.rho

    def reset_state(self):
        self.rho.assign(0.0)




## === cell 13
def _resolve_optional_path(rel_paths):
    for p in rel_paths:
        if os.path.exists(p):
            return p
    return None


def _make_ds(features, labels=None, batch_size=4, training=False):
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(features)
    else:
        ds = tf.data.Dataset.from_tensor_slices((features, labels))
    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)
    if training:
        pass
    ds = ds.batch(batch_size, drop_remainder=False).cache().prefetch(tf.data.AUTOTUNE)
    return ds


def fit_model(
    model, model_name, model_type, data_gen, file_path, train, y, use_saved_weights=True
):
    weights_root = _resolve_optional_path(
        [
            "../input/google-qna-predicted-data/",
            "/kaggle/input/google-qna-predicted-data/",
            "/kaggle/data/google-qna-predicted-data/",
            "/kaggle/data/input/google-qna-predicted-data/",
        ]
    )

    weights_path = os.path.join(weights_root, file_path) if weights_root else None
    can_load = (
        use_saved_weights
        and (weights_path is not None)
        and os.path.exists(weights_path)
    )

    if can_load:
        print(f"getting saved weights for {model_name} from {weights_path}...")
        model.load_weights(weights_path)
    else:
        print(f"fitting data on {model_name} (no saved weights found)...")
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.00002)
        model.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=[SpearmanR()]
        )
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        for tr, cv in kf.split(np.arange(train.shape[0])):
            tr_data, cv_data, y_tr, y_cv = data_gen.generate_data(
                tr, cv, y, model_name, model_type
            )

            bs = 4
            tr_ds = _make_ds(tr_data, y_tr, batch_size=bs, training=True)
            cv_ds = _make_ds(cv_data, y_cv, batch_size=bs, training=False)

            model.fit(tr_ds, epochs=1, validation_data=cv_ds, verbose=1)

        try:
            model.save_weights(file_path)
        except Exception:
            pass

    return model




## === cell 14
def get_weighted_avg(model_predictions):
    xlnet_q, xlnet_a, roberta_q, roberta_a, roberta_qa, bert_q, bert_a, bert_qa = (
        model_predictions
    )
    xlnet_concat = np.concatenate((xlnet_q, xlnet_a), axis=1)
    bert_concat = np.concatenate((bert_q, bert_a), axis=1)
    roberta_concat = np.concatenate((roberta_q, roberta_a), axis=1)
    predict = (roberta_qa + bert_qa + xlnet_concat + bert_concat + roberta_concat) / 5.0
    return predict




## === cell 15
def get_predictions(predictions_present=True, model_saved_weights_present=True):
    X, X_test, y, train, test, sample_submission = get_data()

    pred_root = _resolve_optional_path(
        [
            "../input/google-qna-predicted-data/",
            "/kaggle/input/google-qna-predicted-data/",
            "/kaggle/data/google-qna-predicted-data/",
            "/kaggle/data/input/google-qna-predicted-data/",
        ]
    )

    model_names = ["xlnet-base-cased", "roberta-base", "bert-base-uncased"]
    model_types = ["questions", "answers", "questions_answers"]
    saved_weights_names = [
        "xlnet_q.h5",
        "xlnet_a.h5",
        "roberta_q.h5",
        "roberta_a.h5",
        "roberta_qa.h5",
        "bert_q.h5",
        "bert_a.h5",
        "bert_qa.h5",
    ]

    saved_model_predictions = [
        "xlnet_q.csv",
        "xlnet_a.csv",
        "roberta_q.csv",
        "roberta_a.csv",
        "roberta_qa.csv",
        "bert_q.csv",
        "bert_a.csv",
        "bert_qa.csv",
    ]
    if pred_root:
        saved_model_predictions = [
            os.path.join(pred_root, f) for f in saved_model_predictions
        ]

    model_predictions = []

    can_use_cached = (
        predictions_present
        and pred_root is not None
        and all(os.path.exists(f) for f in saved_model_predictions)
    )

    target_cols = [c for c in sample_submission.columns if c != "qa_id"]

    if can_use_cached:
        print("loading cached model predictions...")
        for file_name in saved_model_predictions:
            dfp = pd.read_csv(file_name)
            if set(target_cols).issubset(dfp.columns):
                arr = dfp[target_cols].values
            else:
                arr = dfp.iloc[:, 1:].values if dfp.shape[1] >= 31 else dfp.values
            model_predictions.append(arr.astype(np.float32))
    else:
        print(
            "cached predictions not found; generating predictions by running models..."
        )
        i = 0
        for name_ in model_names:
            tokenizer = get_tokenizer(name_)  # reused across types for same model
            for type_ in model_types:
                if name_ == "xlnet-base-cased" and type_ == "questions_answers":
                    continue
                print("-" * 100)
                model = create_model(name_, type_)

                data_gen = data_generator(X, X_test, tokenizer, type_)
                model = fit_model(
                    model,
                    name_,
                    type_,
                    data_gen,
                    saved_weights_names[i],
                    train=train,
                    y=y,
                    use_saved_weights=model_saved_weights_present,
                )
                print(f"getting target predictions from {name_} ({type_})...")
                test_data = data_gen.get_test_data(name_, type_)

                bs_pred = 64
                test_ds = _make_ds(
                    test_data, labels=None, batch_size=bs_pred, training=False
                )
                preds = model.predict(test_ds, verbose=0)
                model_predictions.append(preds.astype(np.float32, copy=False))

                K.clear_session()
                i += 1

    predicted_labels = get_weighted_avg(model_predictions)
    exp_labels = get_exp_labels(train)
    optimized_predicted_labels = optimize_ranks(predicted_labels, exp_labels)

    sub = pd.DataFrame(optimized_predicted_labels, columns=target_cols)
    sub.insert(0, "qa_id", test["qa_id"].values)
    sub = sub[["qa_id"] + target_cols]

    print("done...!")
    return sub


submission = get_predictions(predictions_present=True, model_saved_weights_present=True)

target_cols = [c for c in submission.columns if c != "qa_id"]
submission[target_cols] = (
    submission[target_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
)
submission[target_cols] = submission[target_cols].fillna(0.5).clip(0.0, 1.0)

submission = submission.drop_duplicates(subset=["qa_id"], keep="first")

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
