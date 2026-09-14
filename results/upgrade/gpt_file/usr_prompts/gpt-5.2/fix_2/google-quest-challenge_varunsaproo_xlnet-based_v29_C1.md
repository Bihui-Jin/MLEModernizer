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

0.2585599180325146

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

DIR = "/kaggle/input/google-quest-challenge"
BATCH_SIZE = 64

print("Listing /kaggle/input (truncated):")
shown = 0
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        if shown < 30:
            print(os.path.join(dirname, filename))
            shown += 1



## === cell 1
import re
import tensorflow as tf
import tensorflow_hub as hub
from scipy.stats import spearmanr

tf.random.set_seed(42)
np.random.seed(42)

embed = hub.load("/kaggle/input/universal-sentence-encoder")




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def bert_like_encoder_768(texts, batch_size=256):
    """Encode a list of strings into a (n, 768) float32 tensor using USE plus padding."""
    x = embed(texts).numpy().astype(np.float32)  # (n, 512)
    pad = np.zeros((x.shape[0], 256), dtype=np.float32)
    out = np.concatenate([x, pad], axis=1)  # (n, 768)
    return tf.convert_to_tensor(out, dtype=tf.float32)




## === cell 3
def func(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        s = ""
    s = str(s)
    s = re.sub(r"\n+", " ", s)
    s = re.sub(r"[?]", " . ", s)
    s = re.sub(r"[!\{\}]", " . ", s)
    s = re.sub(r"\.{2,}", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s




## === cell 4
def clean_data(df):
    df = df.copy()
    df["question_body"] = df["question_body"].apply(func)
    df["question_title"] = df["question_title"].apply(func)
    df["answer"] = df["answer"].apply(func)
    return df


def preprocess_data(df, offset=0):
    return None


def universal_encoding(df):
    return (
        (2 / 7) * embed(df["question_title"].tolist())
        + (1 / 7) * embed(df["question_body"].tolist())
        + (4 / 7) * embed(df["answer"].tolist())
    )




## === cell 5
train_df = pd.read_csv(DIR + "/train.csv")
test_df = pd.read_csv(DIR + "/test.csv")
sample_submission = pd.read_csv(DIR + "/sample_submission.csv")

train_df = clean_data(train_df)
test_df = clean_data(test_df)

labels = train_df.iloc[:, -30:]
columns = list(labels.columns)

print("Train:", train_df.shape, "Test:", test_df.shape, "Labels:", labels.shape)



## === cell 6
train_text = (
    train_df["question_title"]
    + " "
    + train_df["question_body"]
    + " "
    + train_df["answer"]
).tolist()
test_text = (
    test_df["question_title"] + " " + test_df["question_body"] + " " + test_df["answer"]
).tolist()

train_encoded_inputs_tensor = bert_like_encoder_768(train_text)
test_encoded_inputs_tensor = bert_like_encoder_768(test_text)

train_universal = tf.convert_to_tensor(universal_encoding(train_df), dtype=tf.float32)
test_universal = tf.convert_to_tensor(universal_encoding(test_df), dtype=tf.float32)

print("Encoded:", train_encoded_inputs_tensor.shape, test_encoded_inputs_tensor.shape)
print("Universal:", train_universal.shape, test_universal.shape)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3658916523.py in <cell line: 0>()
     12 
     13 # 768-d encoder features (BERT-like placeholder)
---> 14 train_encoded_inputs_tensor = bert_like_encoder_768(train_text)
     15 test_encoded_inputs_tensor = bert_like_encoder_768(test_text)
     16 

/tmp/ipykernel_11/2240068024.py in bert_like_encoder_768(texts, batch_size)
      3 def bert_like_encoder_768(texts, batch_size=256):
      4     """Encode a list of strings into a (n, 768) float32 tensor using USE plus padding."""
----> 5     x = embed(texts).numpy().astype(np.float32)  # (n, 512)
      6     pad = np.zeros((x.shape[0], 256), dtype=np.float32)
      7     out = np.concatenate([x, pad], axis=1)  # (n, 768)

NameError: name 'embed' is not defined

## === cell 7
def SpearmanCorrCoeff_2(A, B):
    overall_score = 0.0
    x1 = np.random.normal(loc=1e-8, scale=1e-12, size=A.shape[0])
    for index, col in enumerate(columns):
        overall_score += spearmanr(A[:, index] + x1, B[:, index]).correlation / 30.0
    return overall_score


def tf_SpearmanCorrCoeff(A, B):
    return tf.numpy_function(SpearmanCorrCoeff_2, [A, B], Tout=tf.double)




## === cell 8
encoded_inputs = tf.keras.layers.Input(shape=(768,), dtype=tf.float32)
universal_inputs = tf.keras.layers.Input(shape=(512,), dtype=tf.float32)

concat = tf.keras.layers.Concatenate(axis=-1)([encoded_inputs, universal_inputs])

dense_2 = tf.keras.layers.Dense(30, activation="sigmoid")(concat)
model_2 = tf.keras.Model(inputs=[encoded_inputs, universal_inputs], outputs=[dense_2])

model_2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
    loss=["binary_crossentropy"],
    metrics=[tf_SpearmanCorrCoeff],
)

model_2.fit(
    [train_encoded_inputs_tensor, train_universal],
    tf.convert_to_tensor(labels.values, dtype=tf.float32),
    batch_size=16,
    epochs=20,
    verbose=2,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3197825020.py in <cell line: 0>()
     18 # Train as originally configured (epochs=20, batch_size=16)
     19 model_2.fit(
---> 20     [train_encoded_inputs_tensor, train_universal],
     21     tf.convert_to_tensor(labels.values, dtype=tf.float32),
     22     batch_size=16,

NameError: name 'train_encoded_inputs_tensor' is not defined

## === cell 9
ans = model_2.predict(
    [test_encoded_inputs_tensor, test_universal], batch_size=64, verbose=1
)
ans = np.clip(ans, 0.0, 1.0)

print("Pred shape:", ans.shape)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/946372531.py in <cell line: 0>()
      1 ans = model_2.predict(
----> 2     [test_encoded_inputs_tensor, test_universal], batch_size=64, verbose=1
      3 )
      4 ans = np.clip(ans, 0.0, 1.0)
      5 

NameError: name 'test_encoded_inputs_tensor' is not defined

## === cell 10
sub = pd.DataFrame(ans, columns=sample_submission.columns[1:])
sub.insert(0, "qa_id", test_df["qa_id"].values)

sub = sub[sample_submission.columns]

assert sub.shape[0] == test_df.shape[0], "Submission rows must match test rows"
assert list(sub.columns) == list(
    sample_submission.columns
), "Submission columns must match sample_submission"

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", sub.shape)
print(sub.head())

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1889950526.py in <cell line: 0>()
      1 # Fix: submission must align with test qa_id, not sample_submission.qa_id (sample has 608 rows).
----> 2 sub = pd.DataFrame(ans, columns=sample_submission.columns[1:])
      3 sub.insert(0, "qa_id", test_df["qa_id"].values)
      4 
      5 # Ensure correct column order exactly as sample_submission

NameError: name 'ans' is not defined
