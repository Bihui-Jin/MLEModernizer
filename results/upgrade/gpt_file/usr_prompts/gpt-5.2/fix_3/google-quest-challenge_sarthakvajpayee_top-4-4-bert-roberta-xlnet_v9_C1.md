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

-0.0015378042945261

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
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
    BertTokenizer,
    BertConfig,
    TFBertModel,
    RobertaTokenizer,
    RobertaConfig,
    TFRobertaModel,
    XLNetTokenizer,
    XLNetConfig,
    TFXLNetModel,
)

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("Imports OK")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
seed = 13
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)




## === cell 2
def _resolve_competition_path():
    """
    Kaggle can mount the dataset under different roots depending on environment.
    We try common Kaggle paths first, then fall back to the user-provided /kaggle/data layout.
    """
    candidates = [
        "../input/google-quest-challenge/",
        "/kaggle/input/google-quest-challenge/",
        "/kaggle/data/google-quest-challenge/",
        "/kaggle/data/input/google-quest-challenge/",
        "/kaggle/input/google-quest-challenge/google-quest-challenge/",
        "/kaggle/data/google-quest-challenge/google-quest-challenge/",
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

    y = train[train.columns[11:]]  # storing the target values in y
    X = train[["question_title", "question_body", "answer"]].copy()
    X_test = test[["question_title", "question_body", "answer"]].copy()

    for col in ["question_body", "question_title", "answer"]:
        X[col] = X[col].apply(html.unescape)
        X_test[col] = X_test[col].apply(html.unescape)

    return X, X_test, y, train, test, submission




## === cell 3
def get_tokenizer(model_name):
    print(f"getting tokenizer for {model_name}...")
    if model_name == "xlnet-base-cased":
        tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    elif model_name == "roberta-base":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    elif model_name == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
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
def input_data(df, tokenizer, model_type="questions"):
    print(f"generating {model_type} input for transformer...")
    input_ids, input_token_type_ids, input_attention_masks = [], [], []
    for title, body, answer in tqdm(
        zip(
            df["question_title"].values, df["question_body"].values, df["answer"].values
        ),
        total=len(df),
    ):
        ids, type_ids, mask = transformer_inputs(
            title, body, answer, tokenizer, model_type=model_type
        )
        input_ids.append(ids)
        input_token_type_ids.append(type_ids)
        input_attention_masks.append(mask)

    return (
        np.asarray(input_ids, dtype=np.int32),
        np.asarray(input_attention_masks, dtype=np.int32),
        np.asarray(input_token_type_ids, dtype=np.int32),
    )




## === cell 7
def get_model(name):
    if name == "xlnet-base-cased":
        config = XLNetConfig.from_pretrained(
            "xlnet-base-cased", output_hidden_states=True
        )
        model = TFXLNetModel.from_pretrained("xlnet-base-cased", config=config)
    elif name == "roberta-base":
        config = RobertaConfig.from_pretrained(
            "roberta-base", output_hidden_states=True
        )
        model = TFRobertaModel.from_pretrained("roberta-base", config=config)
    elif name == "bert-base-uncased":
        config = BertConfig.from_pretrained(
            "bert-base-uncased", output_hidden_states=True
        )
        model = TFBertModel.from_pretrained("bert-base-uncased", config=config)
    else:
        raise ValueError(f"Unknown model name: {name}")
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

    if name == "xlnet-base-cased":
        outputs = base(
            input_ids=input_tokens, attention_mask=input_mask, training=False
        )
        hidden_states = outputs.hidden_states
    elif name == "roberta-base" and model_type != "questions":
        outputs = base(
            input_ids=input_tokens, attention_mask=input_mask, training=False
        )
        hidden_states = outputs.hidden_states
    else:
        outputs = base(
            input_ids=input_tokens,
            attention_mask=input_mask,
            token_type_ids=input_segment,
            training=False,
        )
        hidden_states = outputs.hidden_states

    h12 = tf.reshape(hidden_states[-1][:, 0], (-1, 1, 768))
    h11 = tf.reshape(hidden_states[-2][:, 0], (-1, 1, 768))
    h10 = tf.reshape(hidden_states[-3][:, 0], (-1, 1, 768))
    h09 = tf.reshape(hidden_states[-4][:, 0], (-1, 1, 768))
    concat_hidden = tf.keras.layers.Concatenate(axis=2)([h12, h11, h10, h09])

    x = GlobalAveragePooling1D()(concat_hidden)
    x = Dropout(0.2)(x)

    if model_type == "answers":
        output = Dense(9, activation="sigmoid")(x)
    elif model_type == "questions":
        output = Dense(21, activation="sigmoid")(x)
    else:
        output = Dense(30, activation="sigmoid")(x)

    if (name == "xlnet-base-cased") or (
        name == "roberta-base" and model_type != "questions"
    ):
        model = Model(inputs=[input_tokens, input_mask], outputs=output)
    else:
        model = Model(inputs=[input_tokens, input_mask, input_segment], outputs=output)

    return model




## === cell 9
class data_generator:
    def __init__(self, X, X_test, tokenizer, type_):
        tokens, masks, segments = input_data(X_test, tokenizer, type_)
        self.test_data = {
            "input_tokens": tokens,
            "input_mask": masks,
            "input_segment": segments,
        }
        self.tokens, self.masks, self.segments = input_data(X, tokenizer, type_)

    def generate_data(self, tr, cv, y, name="xlnet-base-cased", model_type="questions"):
        if name != "xlnet-base-cased":
            train_data = {
                "input_tokens": self.tokens[tr],
                "input_mask": self.masks[tr],
                "input_segment": self.segments[tr],
            }
            cv_data = {
                "input_tokens": self.tokens[cv],
                "input_mask": self.masks[cv],
                "input_segment": self.segments[cv],
            }
        else:
            train_data = {"input_tokens": self.tokens[tr], "input_mask": self.masks[tr]}
            cv_data = {"input_tokens": self.tokens[cv], "input_mask": self.masks[cv]}

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




## === cell 10
def optimize_ranks(preds, unique_labels):
    print("optimizing the predicted values...")
    new_preds = np.zeros(preds.shape, dtype=np.float32)
    for i in range(preds.shape[1]):
        interpolate_bins = np.digitize(preds[:, i], bins=unique_labels, right=False)
        if len(np.unique(interpolate_bins)) == 1:
            new_preds[:, i] = preds[:, i]
        else:
            interpolate_bins = np.clip(interpolate_bins, 0, len(unique_labels) - 1)
            new_preds[:, i] = unique_labels[interpolate_bins]
    return np.clip(new_preds, 0.0, 1.0)




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


def rhos(y, y_pred):
    return tf.py_function(compute_spearmanr_ignore_nan, (y, y_pred), tf.double)




## === cell 13
def _resolve_optional_path(rel_paths):
    for p in rel_paths:
        if os.path.exists(p):
            return p
    return None


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
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[rhos])
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        for tr, cv in kf.split(np.arange(train.shape[0])):
            tr_data, cv_data, y_tr, y_cv = data_gen.generate_data(
                tr, cv, y, model_name, model_type
            )
            model.fit(
                tr_data, y_tr, epochs=1, batch_size=4, validation_data=(cv_data, y_cv)
            )
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

    if can_use_cached:
        print("loading cached model predictions...")
        model_predictions = [
            pd.read_csv(file_name).values for file_name in saved_model_predictions
        ]
    else:
        print(
            "cached predictions not found; generating predictions by running models..."
        )
        i = 0
        for name_ in model_names:
            for type_ in model_types:
                if name_ == "xlnet-base-cased" and type_ == "questions_answers":
                    continue
                print("-" * 100)
                model = create_model(name_, type_)
                tokenizer = get_tokenizer(name_)
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
                preds = model.predict(data_gen.test_data, batch_size=4, verbose=0)
                model_predictions.append(preds)
                i += 1

    predicted_labels = get_weighted_avg(model_predictions)
    exp_labels = get_exp_labels(train)
    optimized_predicted_labels = optimize_ranks(predicted_labels, exp_labels)

    target_cols = [c for c in sample_submission.columns if c != "qa_id"]
    sub = pd.DataFrame(optimized_predicted_labels, columns=train.columns[11:])
    sub.insert(0, "qa_id", test["qa_id"].values)

    sub = sub[["qa_id"] + target_cols]

    print("done...!")
    return sub




## === cell 16
submission = get_predictions(predictions_present=True, model_saved_weights_present=True)

target_cols = [c for c in submission.columns if c != "qa_id"]
submission[target_cols] = (
    submission[target_cols].astype(np.float32).fillna(0.5).clip(0.0, 1.0)
)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1300314393.py in <cell line: 0>()
----> 1 submission = get_predictions(predictions_present=True, model_saved_weights_present=True)
      2 
      3 target_cols = [c for c in submission.columns if c != "qa_id"]
      4 submission[target_cols] = (
      5     submission[target_cols].astype(np.float32).fillna(0.5).clip(0.0, 1.0)

/tmp/ipykernel_11/1664114519.py in get_predictions(predictions_present, model_saved_weights_present)
     62                     continue
     63                 print("-" * 100)
---> 64                 model = create_model(name_, type_)
     65                 tokenizer = get_tokenizer(name_)
     66                 data_gen = data_generator(X, X_test, tokenizer, type_)

/tmp/ipykernel_11/1074999980.py in create_model(name, model_type)
     19 
     20     if name == "xlnet-base-cased":
---> 21         outputs = base(
     22             input_ids=input_tokens, attention_mask=input_mask, training=False
     23         )

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/transformers/modeling_tf_utils.py in run_call_with_unpacked_inputs(self, *args, **kwargs)
    434             config = self.config
    435 
--> 436         unpacked_inputs = input_processing(func, config, **fn_args_and_kwargs)
    437         return func(self, **unpacked_inputs)
    438 

/usr/local/lib/python3.11/dist-packages/transformers/modeling_tf_utils.py in input_processing(func, config, **kwargs)
    511             output[k] = v
    512         else:
--> 513             raise ValueError(f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")
    514 
    515     if isinstance(main_input, (tuple, list)):

ValueError: Exception encountered when calling layer 'tfxl_net_model' (type TFXLNetModel).

Data of type <class 'keras.src.backend.common.keras_tensor.KerasTensor'> is not allowed only (<class 'tensorflow.python.framework.tensor.Tensor'>, <class 'bool'>, <class 'int'>, <class 'transformers.utils.generic.ModelOutput'>, <class 'tuple'>, <class 'list'>, <class 'dict'>, <class 'numpy.ndarray'>) is accepted for attention_mask.

Call arguments received by layer 'tfxl_net_model' (type TFXLNetModel):
  • input_ids=<KerasTensor shape=(None, 512), dtype=int32, sparse=False, name=input_tokens>
  • attention_mask=<KerasTensor shape=(None, 512), dtype=int32, sparse=False, name=input_mask>
  • mems=None
  • perm_mask=None
  • target_mapping=None
  • token_type_ids=None
  • input_mask=None
  • head_mask=None
  • inputs_embeds=None
  • use_mems=None
  • output_attentions=None
  • output_hidden_states=None
  • return_dict=None
  • training=False
