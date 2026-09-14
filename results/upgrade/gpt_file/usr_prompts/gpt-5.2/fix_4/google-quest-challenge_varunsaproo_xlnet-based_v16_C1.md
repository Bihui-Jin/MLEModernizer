# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tokenizers==0.21.2
transformers==4.53.3

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

-0.0066055213120586

# 6. Current score

-0.00808

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.00423) has done: 'I fix the environment-breaking `protobuf`/`transformers` import issue by forcing a compatible pure-Python protobuf implementation before importing transformers/TensorFlow. Then I fix the model/tokenizer loading: your referenced local model path doesn’t exist in this Kaggle dataset, so I load a standard `bert-base-uncased` model/tokenizer in offline-safe mode with a fallback to online if the cache is present. Finally, I correct a few API/key mismatches (`attention_mask` vs `attention_masks`, deprecated `pad_to_max_length`, and ensuring outputs are clipped to `[0,1]`) so training, inference, and `submission.csv` creation run end-to-end.'
- What this solution (achieved nan) has done: 'I fix the environment-breaking protobuf/transformers/TensorFlow import crash by pinning protobuf to a TF-compatible version at runtime (via pip) and then importing transformers only after that, which resolves the `MessageFactory.GetPrototype` error. I also make the text concatenation robust to missing values by using `fillna("")` (the current `x["col"] or ""` can fail when values are NaN). These changes are execution/stability fixes and should keep the same core model/training logic, while ensuring the notebook runs end-to-end and writes a valid `submission.csv`. No score-targeting changes are needed since your current score is already within ±10% of the target.'
- What this solution (achieved -0.00808) has done: 'Your current “nan” score strongly suggests the submission contained invalid values (NaN/inf), wrong row alignment/order, or wrong dtypes/columns. I make three minimal, score-relevant fixes: (1) guarantee predictions are finite (replace NaN/inf, then clip to [0,1]) so Spearman is well-defined; (2) enforce exact row order to match `sample_submission` by merging on `qa_id` (avoids any misalignment that can silently destroy score); and (3) make the model output bounded with a final sigmoid layer so predictions naturally stay in [0,1] and are more stable (this keeps the same core architecture: BERT CLS -> Dense(30), just adds the standard bounded activation for this competition). These changes are small, keep the same training loop and loss, and should turn the “nan” into a valid numeric score and nudge it toward the target band.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
DIR = "/kaggle/input/google-quest-challenge/"
BATCH_SIZE = 64
MAX_LEN = 128
MODEL_NAME = "bert-base-uncased"



## === cell 2
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
import subprocess


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver
    except Exception:
        pb_ver = None

    need_install = False
    if pb_ver is None:
        need_install = True
    else:
        try:
            major = int(str(pb_ver).split(".")[0])
            if major >= 5:
                need_install = True
        except Exception:
            need_install = True

    if need_install:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        import importlib

        if "google.protobuf" in sys.modules:
            importlib.reload(sys.modules["google.protobuf"])


_ensure_protobuf_compat()

import tensorflow as tf
import transformers
import tokenizers

print("TF:", tf.__version__)
print("Transformers:", transformers.__version__)
print("Tokenizers:", tokenizers.__version__)
from google.protobuf import __version__ as pb_ver

print("Protobuf:", pb_ver)



## === cell 3
from transformers import TFBertModel, BertTokenizerFast

try:
    encoder = TFBertModel.from_pretrained(MODEL_NAME, local_files_only=True)
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME, local_files_only=True)
    print("Loaded model/tokenizer from local cache (offline).")
except Exception as e:
    print("Offline load failed, trying regular from_pretrained. Error was:", repr(e))
    encoder = TFBertModel.from_pretrained(MODEL_NAME)
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)



## === cell 4
print(os.listdir("/kaggle/input"))



## === cell 5
input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.layers.Input(
    shape=(MAX_LEN,), dtype=tf.int32, name="attention_mask"
)

bert_outputs = encoder(
    {"input_ids": input_ids, "attention_mask": attention_mask}, training=False
)
cls_output = bert_outputs.last_hidden_state[:, 0, :]

dense_1 = tf.keras.layers.Dense(30, activation="sigmoid", name="targets")(cls_output)

model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=dense_1)
model.compile(optimizer="adam", loss="mse")
model.summary()



## === cell 6
df = pd.read_csv(DIR + "train.csv")

qt = df["question_title"].fillna("").astype(str)
qb = df["question_body"].fillna("").astype(str)
ans = df["answer"].fillna("").astype(str)
texts = (qt + " " + qb + " " + ans).values.tolist()



## === cell 7
t = tokenizer(
    texts,
    max_length=MAX_LEN,
    padding="max_length",
    truncation=True,
    return_tensors="tf",
    add_special_tokens=True,
)



## === cell 8
real_values = tf.convert_to_tensor(df.iloc[:, -30:].values, dtype=tf.float32)
print(
    "Train tokens:", {k: v.shape for k, v in t.items()}, "Targets:", real_values.shape
)



## === cell 9
history = model.fit(
    [t["input_ids"], t["attention_mask"]],
    real_values,
    batch_size=BATCH_SIZE,
    epochs=1,
    verbose=1,
)



## === cell 10
test_df = pd.read_csv(DIR + "test.csv")

tqt = test_df["question_title"].fillna("").astype(str)
tqb = test_df["question_body"].fillna("").astype(str)
tans = test_df["answer"].fillna("").astype(str)
test_texts = (tqt + " " + tqb + " " + tans).values.tolist()



## === cell 11
test_t = tokenizer(
    test_texts,
    max_length=MAX_LEN,
    padding="max_length",
    truncation=True,
    return_tensors="tf",
    add_special_tokens=True,
)



## === cell 12
pred = model.predict(
    [test_t["input_ids"], test_t["attention_mask"]], batch_size=BATCH_SIZE, verbose=1
)

pred = np.nan_to_num(pred, nan=0.5, posinf=1.0, neginf=0.0)
pred = np.clip(pred, 0.0, 1.0)
print("Pred shape:", pred.shape, "min/max:", float(pred.min()), float(pred.max()))



## === cell 13
sample_submission = pd.read_csv(DIR + "sample_submission.csv")
target_cols = sample_submission.columns[1:].tolist()
print("Expected targets:", len(target_cols))



## === cell 14
pred_df = pd.DataFrame(pred, columns=target_cols)
pred_df.insert(0, "qa_id", test_df["qa_id"].values)

sub = sample_submission[["qa_id"]].merge(pred_df, on="qa_id", how="left")

sub[target_cols] = sub[target_cols].apply(pd.to_numeric, errors="coerce").fillna(0.5)
sub[target_cols] = sub[target_cols].clip(0.0, 1.0)

sub = sub[["qa_id"] + target_cols]
print("Submission shape:", sub.shape, "NaNs:", int(sub.isna().sum().sum()))
sub.head()



## === cell 15
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)



## === cell 16
sub.head()
