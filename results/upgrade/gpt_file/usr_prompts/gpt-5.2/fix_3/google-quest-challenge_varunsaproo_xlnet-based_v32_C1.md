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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os




## === cell 1
DIR = "/kaggle/input/google-quest-challenge"
BATCH_SIZE = 64



## === cell 2
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from scipy.stats import spearmanr

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF:", tf.__version__)
print("Transformers:", transformers.__version__)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
MODEL_DIR_CANDIDATE = "/kaggle/input/model-1-quest/kaggle/working/models"
FALLBACK_MODEL = "bert-base-uncased"


def _is_valid_hf_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))


model_source = (
    MODEL_DIR_CANDIDATE if _is_valid_hf_dir(MODEL_DIR_CANDIDATE) else FALLBACK_MODEL
)
print("Loading model/tokenizer from:", model_source)

tokenizer = transformers.BertTokenizerFast.from_pretrained(model_source)
encoder = transformers.TFBertModel.from_pretrained(model_source)




## === cell 4
def func(s):
    if pd.isna(s):
        s = ""
    s = str(s)
    s = re.sub(r"\n+", " ", s)
    s = re.sub(r"[?]", " . ", s)
    s = re.sub(r"[!\{\}]", " . ", s)
    s = re.sub(r"\.{2,}", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s




## === cell 5
def clean_data(df):
    df = df.copy()
    df["question_body"] = df["question_body"].apply(func)
    df["question_title"] = df["question_title"].apply(func)
    df["answer"] = df["answer"].apply(func)
    return df


def preprocess_data(df, max_len=512, offset=0):
    title = df["question_title"].tolist()
    body = df["question_body"].tolist()
    answer = df["answer"].tolist()

    title_tokens = tokenizer(
        title,
        return_tensors="tf",
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
        max_length=65,
        return_attention_mask=True,
    )
    body_tokens = tokenizer(
        body,
        return_tensors="tf",
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
        max_length=200,  # was 512, but only [:, 1:201] was used
        return_attention_mask=True,
    )
    answer_tokens = tokenizer(
        answer,
        return_tensors="tf",
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
        max_length=200,  # was 512, but only [:, 1:201] was used
        return_attention_mask=True,
    )

    sep = tf.ones(shape=[df.shape[0], 1], dtype=tf.int32)
    t = {}
    t["input_ids"] = tf.concat(
        [
            title_tokens["input_ids"],
            sep * 102,
            body_tokens["input_ids"][:, 1:],
            sep * 102,
            answer_tokens["input_ids"][:, 1:],
        ],
        axis=-1,
    )
    t["attention_mask"] = tf.concat(
        [
            title_tokens["attention_mask"],
            sep,
            body_tokens["attention_mask"][:, 1:],
            sep,
            answer_tokens["attention_mask"][:, 1:],
        ],
        axis=-1,
    )
    return t




## === cell 6
train_df = pd.read_csv(DIR + "/train.csv")
test_df = pd.read_csv(DIR + "/test.csv")

train_df = clean_data(train_df)
test_df = clean_data(test_df)

train_encoded_inputs_tensor = preprocess_data(train_df)
test_encoded_inputs_tensor = preprocess_data(test_df)

print("Train input_ids shape:", train_encoded_inputs_tensor["input_ids"].shape)
print("Test input_ids shape:", test_encoded_inputs_tensor["input_ids"].shape)



## === cell 7
labels = tf.convert_to_tensor(train_df.iloc[:, -30:].values, dtype=tf.float32)
columns = list(train_df.columns[-30:])
print("Num labels:", labels.shape, "Num columns:", len(columns))




## === cell 8
def SpearmanCorrCoeff_2(A, B):
    overall_score = 0.0
    rng = np.random.RandomState(SEED)
    x1 = rng.normal(loc=1e-8, scale=1e-12, size=A.shape[0])
    x2 = rng.normal(loc=1e-8, scale=1e-12, size=B.shape[0])
    for index, col in enumerate(columns):
        overall_score += (
            spearmanr(A[:, index] + x1, B[:, index] + x2).correlation / 30.0
        )
    return np.array(overall_score, dtype=np.float64)


def tf_SpearmanCorrCoeff(A, B):
    return tf.numpy_function(SpearmanCorrCoeff_2, [A, B], Tout=tf.float64)




## === cell 9
tf.keras.backend.clear_session()

SEQ_LEN = int(train_encoded_inputs_tensor["input_ids"].shape[1])

input_id = tf.keras.layers.Input(shape=(SEQ_LEN,), dtype=tf.int32)
input_mask = tf.keras.layers.Input(shape=(SEQ_LEN,), dtype=tf.int32)

out = encoder({"input_ids": input_id, "attention_mask": input_mask})[0][:, 0, :]
model_1 = tf.keras.Model(inputs=[input_id, input_mask], outputs=out)

print("Embedding model output shape:", model_1.output_shape)




## === cell 10
def _make_ds(encoded, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(
        (encoded["input_ids"], encoded["attention_mask"])
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


_embed_bs = BATCH_SIZE
while True:
    try:
        train_ds = _make_ds(train_encoded_inputs_tensor, _embed_bs)
        test_ds = _make_ds(test_encoded_inputs_tensor, _embed_bs)

        data = model_1.predict(train_ds, verbose=1)
        test_data = model_1.predict(test_ds, verbose=1)
        break
    except tf.errors.ResourceExhaustedError:
        _embed_bs = max(4, _embed_bs // 2)
        if _embed_bs == 4:
            raise

print("Used embedding batch size:", _embed_bs)
print("Train embeddings:", data.shape, "Test embeddings:", test_data.shape)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2137942165.py in <cell line: 0>()
     17         test_ds = _make_ds(test_encoded_inputs_tensor, _embed_bs)
     18 
---> 19         data = model_1.predict(train_ds, verbose=1)
     20         test_data = model_1.predict(test_ds, verbose=1)
     21         break

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py in tf__predict_function(iterator)
     16                 except:
     17                     do_return = False
---> 18                     raise
     19                 return fscope.ret(retval_, do_return)
     20         return tf__predict_function

ValueError: in user code:

    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py", line 2436, in predict_function  *
        return step_function(self, iterator)
    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py", line 2421, in step_function  **
        outputs = model.distribute_strategy.run(run_step, args=(data,))
    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py", line 2409, in run_step  **
        outputs = model.predict_step(data)
    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py", line 2377, in predict_step
        return self(x, training=False)
    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py", line 70, in error_handler
        raise e.with_traceback(filtered_tb) from None
    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/input_spec.py", line 219, in assert_input_compatibility
        raise ValueError(

    ValueError: Layer "model" expects 2 input(s), but it received 1 input tensors. Inputs received: [<tf.Tensor 'IteratorGetNext:0' shape=(None, 465) dtype=int32>]


## === cell 11
tf.keras.backend.clear_session()

lr_sched = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-4 * (0.8 ** np.floor((epoch) / 2))
)

inp = tf.keras.layers.Input(shape=(data.shape[1],))
elu_dense = tf.keras.layers.Dense(
    512, activation="elu", kernel_initializer="glorot_normal"
)(inp)
dense = tf.keras.layers.Dense(30, activation="sigmoid")(elu_dense)
model = tf.keras.Model(inputs=[inp], outputs=[dense])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=["binary_crossentropy"],
    metrics=[tf_SpearmanCorrCoeff],
)

model.fit(
    [data],
    labels,
    validation_split=0.5,
    batch_size=16,
    epochs=10,
    callbacks=[lr_sched],
    verbose=1,
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/795163856.py in <cell line: 0>()
      5 )
      6 
----> 7 inp = tf.keras.layers.Input(shape=(data.shape[1],))
      8 elu_dense = tf.keras.layers.Dense(
      9     512, activation="elu", kernel_initializer="glorot_normal"

NameError: name 'data' is not defined

## === cell 12
ans = model.predict([test_data], batch_size=16, verbose=1)

ans = np.clip(ans, 0.0, 1.0)
print("Pred shape:", ans.shape, "min/max:", ans.min(), ans.max())



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1805513830.py in <cell line: 0>()
----> 1 ans = model.predict([test_data], batch_size=16, verbose=1)
      2 
      3 ans = np.clip(ans, 0.0, 1.0)
      4 print("Pred shape:", ans.shape, "min/max:", ans.min(), ans.max())
      5 

NameError: name 'model' is not defined

## === cell 13
sample_submission = pd.read_csv(DIR + "/sample_submission.csv")
target_cols = list(sample_submission.columns[1:])

sub = pd.DataFrame(ans, columns=target_cols)
sub.insert(0, "qa_id", test_df["qa_id"].values)

print(sub.shape)
print(sub.head())



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3287485233.py in <cell line: 0>()
      2 target_cols = list(sample_submission.columns[1:])
      3 
----> 4 sub = pd.DataFrame(ans, columns=target_cols)
      5 sub.insert(0, "qa_id", test_df["qa_id"].values)
      6 

NameError: name 'ans' is not defined

## === cell 14
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3046177952.py in <cell line: 0>()
----> 1 sub.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", sub.shape)
      3 

NameError: name 'sub' is not defined

## === cell 15
assert sub.shape[0] == test_df.shape[0]
assert sub.shape[1] == 31
assert (sub[target_cols].values >= 0).all() and (sub[target_cols].values <= 1).all()
sub.head()

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3918999613.py in <cell line: 0>()
----> 1 assert sub.shape[0] == test_df.shape[0]
      2 assert sub.shape[1] == 31
      3 assert (sub[target_cols].values >= 0).all() and (sub[target_cols].values <= 1).all()
      4 sub.head()

NameError: name 'sub' is not defined
