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

0.2336372618481733

# 6. Current score

0.16445

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.16445) has done: 'The notebook currently fails before producing predictions because TensorFlow is crashing on import (protobuf incompatibility) and because it expects external model/tokenizer files that don’t exist in this Kaggle environment. To make it run end-to-end while preserving the overall “tokenize text → model predicts 30 targets → write submission.csv” logic, I replace the missing pre-trained Keras model with a lightweight in-notebook Keras model trained on the provided `train.csv` (same multi-input structure: title/body/answer/category). I also replace the missing pickled tokenizer with a fitted Keras `Tokenizer` trained on the training texts. Finally, I ensure the submission columns match `sample_submission.csv` exactly and clip predictions to `[0,1]` to satisfy submission requirements.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Model

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PATH = "/kaggle/input/google-quest-challenge/"
df_train = pd.read_csv(os.path.join(PATH, "train.csv"))
df_test = pd.read_csv(os.path.join(PATH, "test.csv"))
sample_sub = pd.read_csv(os.path.join(PATH, "sample_submission.csv"))

print(df_train.shape, df_test.shape, sample_sub.shape)
print("Train columns:", df_train.columns[:12].tolist(), "...")
print("Test columns:", df_test.columns.tolist())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
target_cols = [c for c in sample_sub.columns if c != "qa_id"]
assert len(target_cols) == 30, f"Expected 30 targets, got {len(target_cols)}"

for col in ["question_title", "question_body", "answer"]:
    df_train[col] = df_train[col].fillna("")
    df_test[col] = df_test[col].fillna("")

le = LabelEncoder()
all_cats = (
    pd.concat([df_train["category"], df_test["category"]], axis=0)
    .astype(str)
    .fillna("")
)
le.fit(all_cats)

train_cat_int = le.transform(df_train["category"].astype(str).fillna(""))
test_cat_int = le.transform(df_test["category"].astype(str).fillna(""))

num_classes = int(train_cat_int.max() + 1)
train_categoria = tf.keras.utils.to_categorical(train_cat_int, num_classes=num_classes)
test_categoria = tf.keras.utils.to_categorical(test_cat_int, num_classes=num_classes)

texts_for_tok = (
    pd.concat(
        [df_train["question_title"], df_train["question_body"], df_train["answer"]],
        axis=0,
    )
    .astype(str)
    .tolist()
)

max_words = 50000
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts_for_tok)

train_title_seq = tokenizer.texts_to_sequences(
    df_train["question_title"].astype(str).tolist()
)
train_body_seq = tokenizer.texts_to_sequences(
    df_train["question_body"].astype(str).tolist()
)
train_ans_seq = tokenizer.texts_to_sequences(df_train["answer"].astype(str).tolist())

test_title_seq = tokenizer.texts_to_sequences(
    df_test["question_title"].astype(str).tolist()
)
test_body_seq = tokenizer.texts_to_sequences(
    df_test["question_body"].astype(str).tolist()
)
test_ans_seq = tokenizer.texts_to_sequences(df_test["answer"].astype(str).tolist())

MAXLEN_TITLE = 30
MAXLEN_BODY = 200
MAXLEN_ANS = 200

Xtr_title = pad_sequences(
    train_title_seq, maxlen=MAXLEN_TITLE, padding="post", truncating="post"
)
Xtr_body = pad_sequences(
    train_body_seq, maxlen=MAXLEN_BODY, padding="post", truncating="post"
)
Xtr_ans = pad_sequences(
    train_ans_seq, maxlen=MAXLEN_ANS, padding="post", truncating="post"
)

Xte_title = pad_sequences(
    test_title_seq, maxlen=MAXLEN_TITLE, padding="post", truncating="post"
)
Xte_body = pad_sequences(
    test_body_seq, maxlen=MAXLEN_BODY, padding="post", truncating="post"
)
Xte_ans = pad_sequences(
    test_ans_seq, maxlen=MAXLEN_ANS, padding="post", truncating="post"
)

y = df_train[target_cols].astype(np.float32).values

print(
    "Shapes:",
    Xtr_title.shape,
    Xtr_body.shape,
    Xtr_ans.shape,
    train_categoria.shape,
    y.shape,
)



## === cell 2
vocab_size = min(max_words, len(tokenizer.word_index) + 1)
embed_dim = 64


def build_text_branch(inp, name_prefix):
    x = layers.Embedding(vocab_size, embed_dim, name=f"{name_prefix}_emb")(inp)
    x = layers.GlobalAveragePooling1D(name=f"{name_prefix}_gap")(x)
    x = layers.Dense(64, activation="relu", name=f"{name_prefix}_d1")(x)
    return x


inp_title = layers.Input(shape=(MAXLEN_TITLE,), dtype="int32", name="title")
inp_body = layers.Input(shape=(MAXLEN_BODY,), dtype="int32", name="body")
inp_ans = layers.Input(shape=(MAXLEN_ANS,), dtype="int32", name="answer")
inp_cat = layers.Input(shape=(num_classes,), dtype="float32", name="category")

b_title = build_text_branch(inp_title, "title")
b_body = build_text_branch(inp_body, "body")
b_ans = build_text_branch(inp_ans, "answer")
b_cat = layers.Dense(32, activation="relu", name="cat_d1")(inp_cat)

x = layers.Concatenate(name="concat")([b_title, b_body, b_ans, b_cat])
x = layers.Dense(128, activation="relu", name="dense_1")(x)
x = layers.Dropout(0.2, name="drop_1")(x)
out = layers.Dense(len(target_cols), activation="sigmoid", name="targets")(x)

model = Model(inputs=[inp_title, inp_body, inp_ans, inp_cat], outputs=out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3), loss="binary_crossentropy"
)

model.summary()

history = model.fit(
    [Xtr_title, Xtr_body, Xtr_ans, train_categoria],
    y,
    batch_size=256,
    epochs=2,
    validation_split=0.05,
    verbose=2,
)



## === cell 3
test_pred = model.predict(
    [Xte_title, Xte_body, Xte_ans, test_categoria], batch_size=512, verbose=1
)
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = pd.DataFrame(test_pred, columns=target_cols)
submission.insert(0, "qa_id", df_test["qa_id"].values)

submission = submission[sample_sub.columns]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Saved submission.csv with shape:", submission.shape)
print("Columns OK:", list(submission.columns) == list(sample_sub.columns))
