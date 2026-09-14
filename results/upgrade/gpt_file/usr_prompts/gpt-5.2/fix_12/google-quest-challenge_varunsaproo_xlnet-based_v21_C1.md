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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scipy==1.15.3
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

0.2441508446366312

# 6. Current score

0.31023

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.31938) has done: 'We fix the shape mismatch causing BERT embedding extraction to fail by making the concatenated token length dynamic and using that same length in the Keras `Input(shape=(...))` layers, instead of the hard-coded 461. We also remove the protobuf downgrading attempt (it’s incompatible with your environment’s installed protobuf==6.x and can break imports), keeping the rest of the pipeline unchanged. Finally, we ensure the submission uses `test_df["qa_id"]` (19550 rows) and `sample_submission`’s 30 target columns, and always writes `submission.csv` successfully.'
- What this solution (achieved 0.31546) has done: 'I fix the runtime import error coming from a protobuf/TF/transformers incompatibility by forcing transformers to use the pure-Python protobuf implementation before importing TensorFlow/transformers. I also make the token concatenation length consistent between train and test by using the same offset in both calls, preventing any potential shape-related mismatches later. Because your current score (0.31938) is already better than the target (0.24415) and within the ±10% band, I won’t change any modeling/training logic that would intentionally move the score; the rest of the pipeline is kept identical. Finally, I keep the submission formatting robust and ensure `submission.csv` is always written with correct columns and row alignment.'
- What this solution (achieved 0.32345) has done: 'I fix the protobuf/TF/transformers crash by removing the forced pure-Python protobuf setting (it’s incompatible with protobuf 6.x and triggers the `MessageFactory.GetPrototype` error) and instead make Transformers avoid TF/Flax backends so it only uses the PyTorch-side code paths needed for tokenizer + TF model loading. I keep the exact model architecture/training/inference logic unchanged, only touching environment/import order to restore runtime stability. I also add a small safety check to ensure train/test sequence lengths match (and fall back to using the train length for both inputs if they don’t) to prevent downstream shape issues without changing semantics. Finally, the script always write a valid `submission.csv` with correct columns and row alignment.'
- What this solution (achieved 0.31407) has done: 'The crash happens before training because `transformers` is importing protobuf APIs that are incompatible with the installed `protobuf==6.x` (the `MessageFactory.GetPrototype` AttributeError). The minimal, stable fix in this Kaggle environment is to force protobuf to use the pure-Python implementation **before** importing TensorFlow/transformers, and to avoid TF/Flax auto-imports by transformers. I keep the model architecture/training/inference identical, only adjusting import order/env-vars and adding one defensive seed for determinism (score-neutral). The script still write `submission.csv` with the correct 19550 rows and 30 target columns.'
- What this solution (achieved 0.31615) has done: 'You’re crashing in the import cell due to forcing protobuf’s pure-Python implementation, which is incompatible with the installed `protobuf==6.x` and triggers the `MessageFactory.GetPrototype` error. I remove the protobuf override and instead prevent Transformers from importing unnecessary backends (PyTorch/Flax) while keeping TensorFlow enabled, which is the minimal stable setup for `TFBertModel`. I also keep everything else (tokenization lengths, model definitions, training loop, prediction, and submission formatting) unchanged to preserve your current score behavior while restoring end-to-end execution and a valid `submission.csv`.'
- What this solution (achieved 0.321) has done: 'We fix the crash in the import cell caused by an incompatibility between `transformers` and the installed `protobuf==6.x` by pinning protobuf to its pure-Python implementation *only if needed* and by preventing Transformers from importing TF/JAX backends during the initial `transformers` import. Then we re-enable TensorFlow usage and load `TFBertModel` as before, keeping the model/training/prediction logic unchanged (score-neutral). Finally, we keep the existing robust sequence-length handling and ensure `submission.csv` is always written with the correct 19550 rows and the 30 target columns.'
- What this solution (achieved 0.31117) has done: 'We fix the immediate crash caused by the `transformers` ↔ `protobuf==6.x` incompatibility by avoiding the protobuf “python” fallback (which triggers the `MessageFactory.GetPrototype` error) and instead importing `transformers` in a backend-minimal mode (no TF/Flax) so it doesn’t touch problematic protobuf code paths. Then we re-enable TensorFlow and proceed exactly as before to load `TFBertModel`, keeping the model architecture/training/inference unchanged (score-neutral, stability-focused since your current score is already above target). Finally, we keep the existing sequence-length safety and ensure `submission.csv` is written with correct rows/columns.'
- What this solution (achieved 0.31769) has done: 'I fix the crash in the import cell caused by the `transformers` + `protobuf==6.x` incompatibility by preventing Transformers from importing its TensorFlow/Flax/JAX integrations during the initial `transformers` import, then re-enabling TensorFlow usage for `TFBertModel` exactly as before. This is a stability-only change; the model architecture, tokenization, training loop, and prediction post-processing remain unchanged, so the score behavior should be essentially the same. I also keep the existing dynamic sequence-length handling to avoid any train/test shape mismatch. Finally, I ensure the submission is written as `submission.csv` with correct `qa_id` alignment and the 30 target columns from `sample_submission.csv`.'
- What this solution (achieved 0.31362) has done: 'I fix the import crash caused by the `transformers` ↔ `protobuf==6.x` incompatibility by (1) forcing Transformers to avoid importing TensorFlow/JAX/Flax integration modules at import time, and (2) loading the BERT TensorFlow model via `TFAutoModel.from_pretrained(..., from_pt=True)` which bypasses the problematic TF weight-loading/protobuf path. This keeps the same core pipeline (BERT [CLS] embedding → small dense head → sigmoid outputs) and should restore end-to-end execution while keeping score changes minimal. I also keep your existing dynamic sequence-length handling and ensure the submission is written as `submission.csv` with correct columns and row alignment.'
- What this solution (achieved 0.31023) has done: 'The crash happens during `import transformers` because in this environment `transformers==4.53.3` can hit an incompatible protobuf C++ API (`MessageFactory.GetPrototype`) with `protobuf==6.33.0`. The minimal, stable fix is to force protobuf to use the pure-Python runtime **before** importing Transformers/TensorFlow, which avoids that missing method path. I keep your model, tokenization lengths, dynamic sequence-length handling, training loop, and submission formatting unchanged (so score behavior should remain essentially the same) while restoring end-to-end execution. I also add a small safety check to ensure the sample submission columns are used exactly for the 30 targets.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
DIR = "/kaggle/input/google-quest-challenge/"
BATCH_SIZE = 64



## === cell 2
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import transformers
import tokenizers  # noqa: F401

os.environ.pop("TRANSFORMERS_NO_TF", None)

import tensorflow as tf
from scipy.stats import spearmanr

np.random.seed(42)
tf.random.set_seed(42)

print("Python:", sys.version)
print("TF:", tf.__version__)
print("Transformers:", transformers.__version__)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
MODEL_NAME = "bert-base-uncased"

tokenizer = transformers.BertTokenizerFast.from_pretrained(MODEL_NAME)

encoder = transformers.TFAutoModel.from_pretrained(MODEL_NAME, from_pt=True)




## === cell 4
def preprocess_data(df, offset=0):
    title = df["question_title"].fillna("").astype(str).tolist()
    body = df["question_body"].fillna("").astype(str).tolist()
    answer = df["answer"].fillna("").astype(str).tolist()

    title_tokens = tokenizer(
        title,
        return_tensors="tf",
        padding="max_length",
        truncation=True,
        max_length=30,  # keep small title chunk
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    body_tokens = tokenizer(
        body,
        return_tensors="tf",
        padding="max_length",
        truncation=True,
        max_length=171,
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    answer_tokens = tokenizer(
        answer,
        return_tensors="tf",
        padding="max_length",
        truncation=True,
        max_length=233,
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=False,
    )

    max_title_tokens = int(title_tokens["input_ids"].shape[1])
    max_body_tokens = int(body_tokens["input_ids"].shape[1]) - offset
    max_answer_tokens = int(answer_tokens["input_ids"].shape[1]) - offset

    t = {}
    t["input_ids"] = tf.concat(
        [
            title_tokens["input_ids"][:, :max_title_tokens],
            body_tokens["input_ids"][:, :max_body_tokens],
            answer_tokens["input_ids"][:, :max_answer_tokens],
        ],
        axis=-1,
    )
    t["attention_mask"] = tf.concat(
        [
            title_tokens["attention_mask"][:, :max_title_tokens],
            body_tokens["attention_mask"][:, :max_body_tokens],
            answer_tokens["attention_mask"][:, :max_answer_tokens],
        ],
        axis=-1,
    )
    return t




## === cell 5
train_df = pd.read_csv(DIR + "/train.csv")
test_df = pd.read_csv(DIR + "/test.csv")

train_t = preprocess_data(train_df, offset=4)
test_t = preprocess_data(test_df, offset=4)

print(
    "Train tokens shape:", train_t["input_ids"].shape, train_t["attention_mask"].shape
)
print("Test tokens shape:", test_t["input_ids"].shape, test_t["attention_mask"].shape)

SEQ_LEN_TRAIN = int(train_t["input_ids"].shape[1])
SEQ_LEN_TEST = int(test_t["input_ids"].shape[1])
print("Inferred SEQ_LEN_TRAIN:", SEQ_LEN_TRAIN, "SEQ_LEN_TEST:", SEQ_LEN_TEST)

if SEQ_LEN_TEST != SEQ_LEN_TRAIN:
    print(
        "Warning: SEQ_LEN_TEST != SEQ_LEN_TRAIN; using SEQ_LEN_TRAIN for both to avoid shape mismatch."
    )
    SEQ_LEN_TEST = SEQ_LEN_TRAIN
    test_t["input_ids"] = test_t["input_ids"][:, :SEQ_LEN_TEST]
    test_t["attention_mask"] = test_t["attention_mask"][:, :SEQ_LEN_TEST]



## === cell 6
labels = train_df.iloc[:, -30:]



## === cell 7
input_ids = tf.keras.layers.Input(shape=(SEQ_LEN_TRAIN,), dtype=tf.int32)
input_masks = tf.keras.layers.Input(shape=(SEQ_LEN_TRAIN,), dtype=tf.int32)

bert_output = encoder(
    {"input_ids": input_ids, "attention_mask": input_masks}, training=False
)[0][:, 0, :]
model_1 = tf.keras.Model(inputs=[input_ids, input_masks], outputs=[bert_output])



## === cell 8
train_encoded_inputs_tensor = model_1.predict(
    [train_t["input_ids"], train_t["attention_mask"]], batch_size=BATCH_SIZE, verbose=1
)

input_ids_t = tf.keras.layers.Input(shape=(SEQ_LEN_TEST,), dtype=tf.int32)
input_masks_t = tf.keras.layers.Input(shape=(SEQ_LEN_TEST,), dtype=tf.int32)
bert_output_t = encoder(
    {"input_ids": input_ids_t, "attention_mask": input_masks_t}, training=False
)[0][:, 0, :]
model_1_test = tf.keras.Model(
    inputs=[input_ids_t, input_masks_t], outputs=[bert_output_t]
)

test_encoded_inputs_tensor = model_1_test.predict(
    [test_t["input_ids"], test_t["attention_mask"]], batch_size=BATCH_SIZE, verbose=1
)

print("Encoded train shape:", train_encoded_inputs_tensor.shape)
print("Encoded test shape:", test_encoded_inputs_tensor.shape)



## === cell 9
columns = list(train_df.columns[-30:])



## === cell 10
encoded_inputs = tf.keras.layers.Input(shape=(768,))
dropout = tf.keras.layers.Dropout(rate=0.2)
drop_encoded_inputs = dropout(encoded_inputs)
dense_1 = tf.keras.layers.Dense(512, activation="elu")(drop_encoded_inputs)
dense_2 = tf.keras.layers.Dense(30, activation="sigmoid")(dense_1)
model_2 = tf.keras.Model(inputs=[encoded_inputs], outputs=[dense_2])




## === cell 11
def SpearmanCorrCoeff_2(A, B):
    overall_score = 0.0
    x = np.linspace(1e-12, 2e-12, A.shape[0], dtype=np.float64)
    for index, col in enumerate(columns):
        overall_score += spearmanr(A[:, index] + x, B[:, index]).correlation / 30.0
    return np.float64(overall_score)


def tf_SpearmanCorrCoeff(A, B):
    return tf.numpy_function(SpearmanCorrCoeff_2, [A, B], Tout=tf.float64)




## === cell 12
model_2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=["binary_crossentropy"],
    metrics=[tf_SpearmanCorrCoeff],
)

model_2.fit(
    tf.convert_to_tensor(train_encoded_inputs_tensor),
    tf.convert_to_tensor(labels.values, dtype=tf.float32),
    batch_size=16,
    epochs=20,
    verbose=1,
)



## === cell 13
ans = model_2.predict(
    tf.convert_to_tensor(test_encoded_inputs_tensor), batch_size=256, verbose=1
)



## === cell 14
sample_submission = pd.read_csv(DIR + "sample_submission.csv")

target_cols = list(sample_submission.columns[1:])

pred = np.clip(ans, 0.0, 1.0)
sub = pd.DataFrame(pred, columns=target_cols)
sub.insert(0, "qa_id", test_df["qa_id"].values)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
