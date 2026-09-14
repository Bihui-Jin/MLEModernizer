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

0.3475238241448931

# 6. Current score

0.09599

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.09599) has done: 'I fix the runtime crash coming from the `transformers`/protobuf incompatibility by removing the dependency on HuggingFace BERT downloads (which are unavailable offline here) and switching to a lightweight, fully-local text vectorization + dense neural network that still produces 30 sigmoid outputs in `[0,1]`. I also ensure the code reads the competition CSVs from the provided Kaggle paths, builds features from the same core text fields (title/body/answer), and writes a valid `submission.csv` with the exact sample submission columns. Finally, I add basic determinism and shape checks so the pipeline runs end-to-end reliably within the time limit.'

# 9. Code solution

## === cell 0
import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf

warnings.filterwarnings("ignore")

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("Python seed set to:", SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
max_sequence_length = 380  # no longer used by BERT; retained for compatibility
n_epoch = 3
learning_rate = 2e-5
n_fold = 5
batch_size = 8
dropout_rate = 0.1

DATA_PATH_CANDIDATES = [
    "/kaggle/input/google-quest-challenge",
    "/kaggle/input/google-quest-challenge/google-quest-challenge",
    "/kaggle/data/google-quest-challenge",
    "/kaggle/data/google-quest-challenge/google-quest-challenge",
    "../input/google-quest-challenge",
    "../input/google-quest-challenge/google-quest-challenge",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_PATH = next(
    (p for p in DATA_PATH_CANDIDATES if os.path.exists(os.path.join(p, "train.csv"))),
    None,
)
if DATA_PATH is None:
    raise FileNotFoundError(
        f"Could not find train.csv in any of: {DATA_PATH_CANDIDATES}"
    )

print("Using DATA_PATH:", DATA_PATH)



## === cell 2
df_train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
df_test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
df_sample = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))

input_columns = ["question_title", "question_body", "answer"]
output_labels = list(df_sample.columns[1:])

print("n_train:", len(df_train), "n_test:", len(df_test))
print("n_targets:", len(output_labels))
assert len(output_labels) == 30, f"Expected 30 targets, got {len(output_labels)}"

for c in input_columns:
    df_train[c] = df_train[c].fillna("").astype(str)
    df_test[c] = df_test[c].fillna("").astype(str)

y = df_train[output_labels].astype(np.float32).values

train_text = (
    df_train["question_title"]
    + " "
    + df_train["question_body"]
    + " "
    + df_train["answer"]
).values
test_text = (
    df_test["question_title"] + " " + df_test["question_body"] + " " + df_test["answer"]
).values

print("Example merged text length:", len(train_text[0]))



## === cell 3

VOCAB_SIZE = 50000
SEQ_LEN = 256

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LEN,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
)

vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_text).batch(256))

print("Vectorizer adapted. Vocab size:", len(vectorizer.get_vocabulary()))




## === cell 4
def create_model():
    inp = tf.keras.layers.Input(shape=(), dtype=tf.string)
    x = vectorizer(inp)
    x = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=128, mask_zero=True)(
        x
    )
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(1500, activation="relu")(x)
    x = tf.keras.layers.Dense(1500, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    out = tf.keras.layers.Dense(30, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model


model = create_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()



## === cell 5
n = len(train_text)
idx = np.arange(n)
np.random.shuffle(idx)

val_size = int(0.1 * n)
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

x_trn, y_trn = train_text[trn_idx], y[trn_idx]
x_val, y_val = train_text[val_idx], y[val_idx]

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_trn, y_trn))
    .shuffle(20000, seed=SEED)
    .batch(256)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
    .batch(256)
    .prefetch(tf.data.AUTOTUNE)
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=n_epoch,
    verbose=1,
)



## === cell 6
pred = model.predict(
    tf.data.Dataset.from_tensor_slices(test_text).batch(256), verbose=1
)
pred = np.clip(pred, 0.0, 1.0).astype(np.float32)

df_submission = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))
if pred.shape != (len(df_submission), len(output_labels)):
    raise ValueError(
        f"Prediction shape {pred.shape} does not match submission shape "
        f"({len(df_submission)}, {len(output_labels)})."
    )

df_submission.iloc[:, 1:] = pred
out_path = "submission.csv"
df_submission.to_csv(out_path, index=False)

print("Wrote submission.csv with shape:", df_submission.shape)
print(df_submission.head())
print("submission.csv saved at:", os.path.abspath(out_path))
