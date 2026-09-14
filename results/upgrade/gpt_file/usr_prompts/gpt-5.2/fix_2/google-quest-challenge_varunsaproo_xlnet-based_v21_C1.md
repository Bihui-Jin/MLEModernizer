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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

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
import sys, subprocess


def _ensure_protobuf_compatible():
    try:
        import google.protobuf  # noqa: F401
        import pkgutil
        import pkg_resources

        ver = pkg_resources.get_distribution("protobuf").version
        major = int(ver.split(".")[0])
        if major >= 5:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
            )
            for m in list(sys.modules.keys()):
                if m.startswith("google.protobuf"):
                    del sys.modules[m]
    except Exception:
        pass


_ensure_protobuf_compatible()

import transformers
import tokenizers  # noqa: F401
import tensorflow as tf
import functools  # noqa: F401
from scipy.stats import spearmanr

print("Python:", sys.version)
print("TF:", tf.__version__)
print("Transformers:", transformers.__version__)



## === cell 3
MODEL_NAME = "bert-base-uncased"

tokenizer = transformers.BertTokenizerFast.from_pretrained(MODEL_NAME)
encoder = transformers.TFBertModel.from_pretrained(MODEL_NAME)




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
        max_length=30,  # keep small title chunk; overall concat length matches later Input(shape=(461,))
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

train_t = preprocess_data(train_df, 4)
test_t = preprocess_data(test_df)

print(
    "Train tokens shape:", train_t["input_ids"].shape, train_t["attention_mask"].shape
)
print("Test tokens shape:", test_t["input_ids"].shape, test_t["attention_mask"].shape)



## === cell 6
labels = train_df.iloc[:, -30:]



## === cell 7
input_ids = tf.keras.layers.Input(shape=(461,), dtype=tf.int32)
input_masks = tf.keras.layers.Input(shape=(461,), dtype=tf.int32)

bert_output = encoder(
    {"input_ids": input_ids, "attention_mask": input_masks}, training=False
)[0][:, 0, :]
model_1 = tf.keras.Model(inputs=[input_ids, input_masks], outputs=[bert_output])



## === cell 8
train_encoded_inputs_tensor = model_1.predict(
    [train_t["input_ids"], train_t["attention_mask"]], batch_size=BATCH_SIZE, verbose=1
)
test_encoded_inputs_tensor = model_1.predict(
    [test_t["input_ids"], test_t["attention_mask"]], batch_size=BATCH_SIZE, verbose=1
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1149745232.py in <cell line: 0>()
----> 1 train_encoded_inputs_tensor = model_1.predict(
      2     [train_t["input_ids"], train_t["attention_mask"]], batch_size=BATCH_SIZE, verbose=1
      3 )
      4 test_encoded_inputs_tensor = model_1.predict(
      5     [test_t["input_ids"], test_t["attention_mask"]], batch_size=BATCH_SIZE, verbose=1

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
    File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/input_spec.py", line 298, in assert_input_compatibility
        raise ValueError(

    ValueError: Input 0 of layer "model" is incompatible with the layer: expected shape=(None, 461), found shape=(None, 426)


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



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1433880789.py in <cell line: 0>()
      7 
      8 model_2.fit(
----> 9     tf.convert_to_tensor(train_encoded_inputs_tensor),
     10     tf.convert_to_tensor(labels.values, dtype=tf.float32),
     11     batch_size=16,

NameError: name 'train_encoded_inputs_tensor' is not defined

## === cell 13
ans = model_2.predict(
    tf.convert_to_tensor(test_encoded_inputs_tensor), batch_size=256, verbose=1
)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/827176761.py in <cell line: 0>()
      1 ans = model_2.predict(
----> 2     tf.convert_to_tensor(test_encoded_inputs_tensor), batch_size=256, verbose=1
      3 )
      4 

NameError: name 'test_encoded_inputs_tensor' is not defined

## === cell 14
sample_submission = pd.read_csv(DIR + "sample_submission.csv")



## === cell 15
pred = np.clip(ans, 0.0, 1.0)
sub = pd.DataFrame(pred, columns=sample_submission.columns[1:])
sub.insert(0, "qa_id", test_df["qa_id"].values)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2444616879.py in <cell line: 0>()
      1 # Fix: ensure correct qa_id alignment with test.csv, not sample_submission's 608 ids.
      2 # Also clip predictions into [0,1] as required.
----> 3 pred = np.clip(ans, 0.0, 1.0)
      4 sub = pd.DataFrame(pred, columns=sample_submission.columns[1:])
      5 sub.insert(0, "qa_id", test_df["qa_id"].values)

NameError: name 'ans' is not defined

## === cell 16
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2737361063.py in <cell line: 0>()
----> 1 sub.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", sub.shape)
      3 print(sub.head())

NameError: name 'sub' is not defined
