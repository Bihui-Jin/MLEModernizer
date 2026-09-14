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

0.1475958055409032

# 6. Current score

0.03267

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.03267) has done: 'The script now builds its own tokenizer and a lightweight Keras model (three GRU branches for title, body, and answer) to train on the available data, then generates predictions for the test set and writes a correctly‑formatted `submission.csv`. This removes the missing external files, fixes the undefined variables, and ensures a valid submission is produced.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, concatenate

tf.random.set_seed(42)
np.random.seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = "/kaggle/input/google-quest-challenge/train.csv"
test_path = "/kaggle/input/google-quest-challenge/test.csv"
sample_sub_path = "/kaggle/input/google-quest-challenge/sample_submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
submission = pd.read_csv(sample_sub_path)

print("train shape:", train.shape)
print("test shape:", test.shape)



## === cell 2
exclude_cols = [
    "qa_id",
    "question_title",
    "question_body",
    "question_user_name",
    "question_user_page",
    "answer",
    "answer_user_name",
    "answer_user_page",
    "url",
    "category",
    "host",
]
target_cols = [c for c in train.columns if c not in exclude_cols]
print("target columns:", target_cols)

all_text = pd.concat(
    [
        train["question_title"].astype(str),
        train["question_body"].astype(str),
        train["answer"].astype(str),
        test["question_title"].astype(str),
        test["question_body"].astype(str),
        test["answer"].astype(str),
    ]
)
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(all_text.tolist())




## === cell 3
def compute_sequences(cols, tokenizer, maxlens):
    sequences = []
    for texts, maxlen in zip(cols, maxlens):
        seq = tokenizer.texts_to_sequences(texts.astype(str).tolist())
        seq = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
        sequences.append(seq)
    return sequences


train_seqs = compute_sequences(
    [train.question_title, train.question_body, train.answer], tokenizer, [30, 300, 300]
)

test_seqs = compute_sequences(
    [test.question_title, test.question_body, test.answer], tokenizer, [30, 300, 300]
)

train_title_seq, train_body_seq, train_answer_seq = train_seqs
test_title_seq, test_body_seq, test_answer_seq = test_seqs



## === cell 4
vocab_size = min(20000, len(tokenizer.word_index) + 1)
embed_dim = 128

inp_title = Input(shape=(30,), name="title_input")
inp_body = Input(shape=(300,), name="body_input")
inp_answer = Input(shape=(300,), name="answer_input")

embedding_layer = Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)

x_title = embedding_layer(inp_title)
x_body = embedding_layer(inp_body)
x_answer = embedding_layer(inp_answer)

x_title = GRU(64)(x_title)
x_body = GRU(64)(x_body)
x_answer = GRU(64)(x_answer)

x = concatenate([x_title, x_body, x_answer])
x = Dense(128, activation="relu")(x)
out = Dense(len(target_cols), activation="sigmoid")(x)

model = Model(inputs=[inp_title, inp_body, inp_answer], outputs=out)
model.compile(optimizer="adam", loss="binary_crossentropy")
model.summary()



## === cell 5
y_train = train[target_cols].values

model.fit(
    x=[train_title_seq, train_body_seq, train_answer_seq],
    y=y_train,
    validation_split=0.1,
    epochs=2,
    batch_size=512,
    verbose=2,
)



## === cell 6
test_pred = model.predict(
    [test_title_seq, test_body_seq, test_answer_seq], batch_size=512, verbose=2
)

test_pred = np.clip(test_pred, 0, 1)

submission[target_cols] = test_pred
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
