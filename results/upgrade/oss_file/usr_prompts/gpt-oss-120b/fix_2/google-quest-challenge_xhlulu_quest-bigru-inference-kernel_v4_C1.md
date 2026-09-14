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

0.1873609852665528

# 6. Current score

0.04252

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.04252) has done: 'I replace the missing tokenizer/model loading with a lightweight end‑to‑end pipeline: fit a Keras Tokenizer on the text fields, convert them to padded sequences, build a small multi‑input neural network (embeddings + GRU + dense) and train it briefly on a train/validation split. After training the model predicts the 30 target columns for the test set and writes a correctly‑named `submission.csv`. This fixes the import/file errors, defines all needed variables, and yields a valid submission whose score should move toward the target range.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    GRU,
    Dense,
    Concatenate,
)
from tensorflow.keras.optimizers import Adam



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

print("train shape:", train.shape, "test shape:", test.shape)



## === cell 2
target_cols = train.columns[-30:].tolist()

text_fields = ["question_title", "question_body", "answer"]
all_text = pd.concat([train[text_fields], test[text_fields]], ignore_index=True).astype(
    str
)

tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(pd.concat([all_text[col] for col in text_fields]))


def compute_sequences(df, tokenizer, maxlens):
    sequences = []
    for col, maxlen in zip(text_fields, maxlens):
        seq = tokenizer.texts_to_sequences(df[col].astype(str).values)
        seq = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
        sequences.append(seq)
    return sequences


max_lengths = [30, 300, 300]

train_seqs = compute_sequences(train, tokenizer, max_lengths)
test_seqs = compute_sequences(test, tokenizer, max_lengths)

y = train[target_cols].values.astype(np.float32)



## === cell 3
np.random.seed(42)
indices = np.arange(len(train))
np.random.shuffle(indices)
split = int(0.9 * len(train))
train_idx, val_idx = indices[:split], indices[split:]

X_train = [seq[train_idx] for seq in train_seqs]
X_val = [seq[val_idx] for seq in train_seqs]
y_train, y_val = y[train_idx], y[val_idx]

vocab_size = min(20000, len(tokenizer.word_index) + 1)
embed_dim = 64

inp_title = Input(shape=(max_lengths[0],), name="title_input")
inp_body = Input(shape=(max_lengths[1],), name="body_input")
inp_answer = Input(shape=(max_lengths[2],), name="answer_input")


def encoder(seq_input, name):
    x = Embedding(
        input_dim=vocab_size, output_dim=embed_dim, mask_zero=True, name=f"{name}_emb"
    )(seq_input)
    x = Bidirectional(GRU(32, return_sequences=False), name=f"{name}_gru")(x)
    return x


enc_title = encoder(inp_title, "title")
enc_body = encoder(inp_body, "body")
enc_answer = encoder(inp_answer, "answer")

merged = Concatenate()([enc_title, enc_body, enc_answer])
dense = Dense(128, activation="relu")(merged)
output = Dense(len(target_cols), activation="sigmoid")(dense)

model = Model(inputs=[inp_title, inp_body, inp_answer], outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
model.summary()



## === cell 4
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=2,
    batch_size=512,
    verbose=2,
)



## === cell 5
test_pred = model.predict(test_seqs, batch_size=512, verbose=0)

submission.loc[:, target_cols] = test_pred
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {os.path.abspath(output_path)}")
