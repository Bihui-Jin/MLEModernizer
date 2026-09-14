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

0.240912663400819

# 6. Current score

0.01562

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.01562) has done: 'I replace the failing BERT‑based encoder with the Universal Sentence Encoder from TensorFlow Hub, fix the input shape specifications, and streamline the preprocessing so that tokenization is no longer required. The new pipeline cleans the text, creates embeddings for title‑body‑answer concatenations, trains a small dense network on these embeddings, and finally writes a correctly‑formatted `submission.csv`. These changes resolve the import and shape errors while keeping the overall model‑and‑training logic unchanged, allowing the script to run end‑to‑end and produce a valid submission.'

# 9. Code solution

## === cell 0
import os, re
import numpy as np, pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from scipy.stats import spearmanr



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DIR = "/kaggle/input/google-quest-challenge"
BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 1e-4



## === cell 2
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")




## === cell 3
def func(s):
    s = re.sub(r"\n+", " ", s)
    s = re.sub(r"[?]", " . ", s)
    s = re.sub(r"[!\{\}]", " . ", s)
    s = re.sub(r"\.{2,}", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def clean_data(df):
    for col in ["question_title", "question_body", "answer"]:
        df[col] = df[col].fillna("").apply(func)
    return df


def get_embeddings(df):
    texts = (
        df["question_title"] + " " + df["question_body"] + " " + df["answer"]
    ).tolist()
    return embed(texts).numpy()  # shape: (n_samples, 512)




## === cell 4
train_df = pd.read_csv(os.path.join(DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DIR, "test.csv"))

train_df = clean_data(train_df)
test_df = clean_data(test_df)

train_emb = get_embeddings(train_df)
test_emb = get_embeddings(test_df)

labels = train_df.iloc[:, -30:].values.astype(np.float32)
label_columns = list(train_df.columns[-30:])




## === cell 5
def spearman_metric(y_true, y_pred):
    def _spearman(y_t, y_p):
        score = 0.0
        eps = np.random.normal(loc=1e-8, scale=1e-12, size=y_t.shape)
        for i in range(y_t.shape[1]):
            score += spearmanr(y_t[:, i] + eps[:, 0], y_p[:, i] + eps[:, 0]).correlation
        return score / y_t.shape[1]

    return tf.numpy_function(_spearman, [y_true, y_pred], tf.double)




## === cell 6
input_emb = tf.keras.layers.Input(shape=(512,), dtype=tf.float32, name="embed")
x = tf.keras.layers.Dense(512, activation="elu", kernel_initializer="glorot_normal")(
    input_emb
)
output = tf.keras.layers.Dense(30, activation="sigmoid")(x)
model = tf.keras.Model(inputs=input_emb, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=[spearman_metric],
)



## === cell 7
model.fit(
    train_emb,
    labels,
    validation_split=0.1,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2,
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4032666834.py in <cell line: 0>()
      1 # Train / validate
----> 2 model.fit(
      3     train_emb,
      4     labels,
      5     validation_split=0.1,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/metrics/reduction_metrics.py in reduce_to_samplewise_values(values, sample_weight, reduce_fn, dtype)
     39             )
     40 
---> 41     values_ndim = len(values.shape)
     42     if values_ndim > 1:
     43         values = reduce_fn(values, axis=list(range(1, values_ndim)))

ValueError: Cannot take the length of shape with unknown rank.

## === cell 8
test_pred = model.predict(test_emb, batch_size=BATCH_SIZE)
test_pred = np.clip(test_pred, 0.0, 1.0)  # ensure [0,1] range



## === cell 9
sample_submission = pd.read_csv(os.path.join(DIR, "sample_submission.csv"))
submission = pd.DataFrame(test_pred, columns=sample_submission.columns[1:])
submission["qa_id"] = test_df["qa_id"].values
cols = ["qa_id"] + [c for c in submission.columns if c != "qa_id"]
submission = submission[cols]



## === cell 10
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
