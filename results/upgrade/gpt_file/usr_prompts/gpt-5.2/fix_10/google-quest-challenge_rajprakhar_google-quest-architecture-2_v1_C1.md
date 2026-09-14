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
tqdm==4.67.1

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

0.1205653379877138

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.22029) has done: 'I make the smallest changes needed to (1) ensure the notebook reliably runs in this environment without downgrading `protobuf` (your current cell likely break because `packaging` isn’t installed and TF 2.18 expects protobuf 5/6), and (2) improve score modestly toward your target by fixing a major generalization issue: you’re training on only the first 5000 rows without shuffling, which is a biased slice. I keep your exact preprocessing, embeddings, model architecture, loss, and training loop intact, but change the train sampling to a deterministic shuffled 5000-row subset and split validation from that shuffled subset. This should improve public score vs. the current non-yielding run, while remaining fast and within constraints, and still produces a valid `submission.csv`.'
- What this solution (achieved 0.21486) has done: 'I fix the TensorFlow import crash caused by an incompatibility between TF 2.18 and protobuf 6.x by setting the Python protobuf implementation before importing TensorFlow. I also remove the unused `packaging`/downgrade-style assumptions and keep your model, preprocessing, training loop, and submission formatting the same. These changes are score-neutral and only ensure the notebook runs end-to-end in this environment and produces a valid `submission.csv`.'
- What this solution (achieved 0.21092) has done: 'I fix the TensorFlow/protobuf crash by switching to a stable TF 2.18-compatible protobuf runtime choice (use the C++ implementation instead of the pure-Python fallback that triggers the `MessageFactory.GetPrototype` error). I keep your model, preprocessing, training loop, and submission formatting unchanged to avoid score-shifting edits, since your current score (0.21486) is already above the target band and we should not pursue improvements. I also make the environment setting deterministic and ensure the script runs end-to-end to write `submission.csv` with the exact `sample_submission.csv` columns. No changes are made to architecture, loss, epochs, batch size, or feature extraction.'
- What this solution (achieved 0.19049) has done: 'To fix the runtime crash, I set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and its version) before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` protobuf incompatibility seen with TF 2.18 + protobuf 6.x in some Kaggle images. I also remove the line that forcibly pops this environment variable (since it re-introduces the crash). No model/training/feature logic is changed, so the score behavior should remain essentially the same (and we avoid moving further away from your lower target). The script still run end-to-end and write a valid `submission.csv` with the exact `sample_submission.csv` columns.'
- What this solution (achieved 0.20967) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf runtime before importing TensorFlow (the current `cpp` setting is what triggers the `_message` import error with protobuf 6.x in this environment). I also fix the `tqdm` `NameError` by importing it in the preprocessing cell so your existing preprocessing loop runs. These are execution-blocking issues; once TensorFlow imports and preprocessing runs, the downstream `KeyError`/`NameError` cascades disappear because the missing columns/arrays get created. I keep your model, preprocessing, training loop, and submission formatting unchanged so the behavior stays consistent while reliably producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os, sys, subprocess, warnings, re

warnings.simplefilter("ignore")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

import tensorflow as tf

print("TensorFlow:", tf.__version__)

from scipy.stats import spearmanr

np.random.seed(42)
tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/3484346753.py in <cell line: 0>()
     12 import pandas as pd
     13 
---> 14 import tensorflow as tf
     15 
     16 print("TensorFlow:", tf.__version__)

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     47 _tf2.enable()
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__
     51 from tensorflow._api.v2 import audio

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator
     10 from tensorflow._api.v2.__internal__ import dispatch

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     19 import threading
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export
     23 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     15 """Utility module that contains APIs usable in the generated code."""
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors
     19 from tensorflow.python.autograph.utils.tensor_list import dynamic_list_append

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     17 import contextlib
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops
     21 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     31 
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2
     35 from tensorflow.core.framework import function_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      3 # source: tensorflow/core/framework/attr_value.proto
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor
      7 from google.protobuf import descriptor_pool as _descriptor_pool

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     16 
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message
     20 from google.protobuf import reflection as _reflection

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     36 import weakref
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod
     40 from google.protobuf import text_format

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     27   # TODO: Remove this import after fix api_implementation
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True
     31 

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

## === cell 1
PATH = "/kaggle/input/google-quest-challenge/"

df_train = pd.read_csv(os.path.join(PATH, "train.csv"))
df_test = pd.read_csv(os.path.join(PATH, "test.csv"))
df_sub = pd.read_csv(os.path.join(PATH, "sample_submission.csv"))

print("Train Shape =", df_train.shape)
print("Test Shape  =", df_test.shape)

output_categories = list(df_sub.columns[1:])  # the 30 targets
input_categories = ["question_title", "question_body", "answer"]
print("\n#targets:", len(output_categories))
print("#inputs:", input_categories)

for c in input_categories:
    df_train[c] = df_train[c].fillna("")
    df_test[c] = df_test[c].fillna("")

N_TRAIN = min(5000, len(df_train))
df_train_small = (
    df_train.sample(n=N_TRAIN, random_state=42).reset_index(drop=True).copy()
)

print("Using N_TRAIN for preprocessing/training =", N_TRAIN)



## === cell 2
from tqdm import tqdm

stopwords = [
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "you're",
    "you've",
    "you'll",
    "you'd",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "she's",
    "her",
    "hers",
    "herself",
    "it",
    "it's",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "that'll",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "don't",
    "should",
    "should've",
    "now",
    "d",
    "ll",
    "m",
    "o",
    "re",
    "ve",
    "y",
    "ain",
    "aren",
    "aren't",
    "couldn",
    "couldn't",
    "didn",
    "didn't",
    "doesn",
    "doesn't",
    "hadn",
    "hadn't",
    "hasn",
    "hasn't",
    "haven",
    "haven't",
    "isn",
    "isn't",
    "ma",
    "mightn",
    "mightn't",
    "mustn",
    "mustn't",
    "needn",
    "needn't",
    "shan",
    "shan't",
    "shouldn",
    "shouldn't",
    "wasn",
    "wasn't",
    "weren",
    "weren't",
    "won",
    "won't",
    "wouldn",
    "wouldn't",
]


def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def preprocess_text(text_data):
    preprocessed_text = []
    for sentance in tqdm(text_data, desc="preprocess", leave=False):
        sent = decontracted(str(sentance))
        sent = sent.replace("\\r", " ")
        sent = sent.replace("\\n", " ")
        sent = sent.replace('\\"', " ")
        sent = re.sub("[^A-Za-z0-9]+", " ", sent)
        sent = " ".join(e for e in sent.split() if e.lower() not in stopwords)
        preprocessed_text.append(sent.lower().strip())
    return preprocessed_text


def perform_preprocessing(text_array):
    lower_text_array = pd.Series(text_array).astype(str).str.lower()
    preprocessed_text_array = preprocess_text(lower_text_array)
    return pd.Series(preprocessed_text_array)


df_train_small["Preproc_Question_Title"] = perform_preprocessing(
    df_train_small["question_title"].values
)
df_train_small["Preproc_Question_Body"] = perform_preprocessing(
    df_train_small["question_body"].values
)
df_train_small["Preproc_Answer"] = perform_preprocessing(
    df_train_small["answer"].values
)

df_test["Preproc_Question_Title"] = perform_preprocessing(
    df_test["question_title"].values
)
df_test["Preproc_Question_Body"] = perform_preprocessing(
    df_test["question_body"].values
)
df_test["Preproc_Answer"] = perform_preprocessing(df_test["answer"].values)

print("=" * 40, "Example After Preprocessing", "=" * 40)
print("QT:", df_train_small["Preproc_Question_Title"].iloc[0][:200])
print("QB:", df_train_small["Preproc_Question_Body"].iloc[0][:200])
print("AN:", df_train_small["Preproc_Answer"].iloc[0][:200])




## === cell 3
def prepare_embedding(
    input_series_train, input_series_test, column_name, embed_dim=300
):
    print("=" * 40 + f" {column_name} " + "=" * 40)

    tokenizer_obj = tf.keras.preprocessing.text.Tokenizer()
    tokenizer_obj.fit_on_texts(input_series_train.values)

    word_index = tokenizer_obj.word_index
    print("Found %s unique tokens." % len(word_index))

    train_sequences = tokenizer_obj.texts_to_sequences(input_series_train.values)
    test_sequences = tokenizer_obj.texts_to_sequences(input_series_test.values)

    MAX_SEQUENCE_LENGTH = int(np.percentile(pd.Series(train_sequences).apply(len), 96))
    MAX_SEQUENCE_LENGTH = max(MAX_SEQUENCE_LENGTH, 10)
    print("96th percentile length =", MAX_SEQUENCE_LENGTH)

    vocab_size = len(word_index) + 1
    train_sequences_pad = tf.keras.preprocessing.sequence.pad_sequences(
        train_sequences, maxlen=MAX_SEQUENCE_LENGTH
    )
    test_sequences_pad = tf.keras.preprocessing.sequence.pad_sequences(
        test_sequences, maxlen=MAX_SEQUENCE_LENGTH
    )
    print(
        "Padded train:",
        train_sequences_pad.shape,
        "Padded test:",
        test_sequences_pad.shape,
    )

    rng = np.random.default_rng(42)
    embedding_matrix = rng.normal(0.0, 0.05, size=(vocab_size, embed_dim)).astype(
        "float32"
    )
    embedding_matrix[0] = 0.0

    return (
        vocab_size,
        embedding_matrix,
        MAX_SEQUENCE_LENGTH,
        train_sequences_pad,
        test_sequences_pad,
    )




## === cell 4
(
    vocab_size_question_title,
    embedding_matrix_question_title,
    MAX_SEQUENCE_LENGTH_question_title,
    train_sequences_pad_qt,
    test_sequences_pad_qt,
) = prepare_embedding(
    df_train_small["Preproc_Question_Title"],
    df_test["Preproc_Question_Title"],
    "Question Title",
)

(
    vocab_size_question_body,
    embedding_matrix_question_body,
    MAX_SEQUENCE_LENGTH_question_body,
    train_sequences_pad_qb,
    test_sequences_pad_qb,
) = prepare_embedding(
    df_train_small["Preproc_Question_Body"],
    df_test["Preproc_Question_Body"],
    "Question Body",
)

(
    vocab_size_answer,
    embedding_matrix_answer,
    MAX_SEQUENCE_LENGTH_answer,
    train_sequences_pad_ans,
    test_sequences_pad_ans,
) = prepare_embedding(
    df_train_small["Preproc_Answer"], df_test["Preproc_Answer"], "Answer"
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4251252270.py in <cell line: 0>()
      5     train_sequences_pad_qt,
      6     test_sequences_pad_qt,
----> 7 ) = prepare_embedding(
      8     df_train_small["Preproc_Question_Title"],
      9     df_test["Preproc_Question_Title"],

/tmp/ipykernel_55/3026399139.py in prepare_embedding(input_series_train, input_series_test, column_name, embed_dim)
      4     print("=" * 40 + f" {column_name} " + "=" * 40)
      5 
----> 6     tokenizer_obj = tf.keras.preprocessing.text.Tokenizer()
      7     tokenizer_obj.fit_on_texts(input_series_train.values)
      8 

NameError: name 'tf' is not defined

## === cell 5
VAL_FRAC = 0.2
n_val = max(1, int(N_TRAIN * VAL_FRAC))
n_train_eff = N_TRAIN - n_val

validation_sequences_pad_qt = train_sequences_pad_qt[n_train_eff:]
train_sequences_pad_qt = train_sequences_pad_qt[:n_train_eff]

validation_sequences_pad_qb = train_sequences_pad_qb[n_train_eff:]
train_sequences_pad_qb = train_sequences_pad_qb[:n_train_eff]

validation_sequences_pad_ans = train_sequences_pad_ans[n_train_eff:]
train_sequences_pad_ans = train_sequences_pad_ans[:n_train_eff]

print("Train size:", n_train_eff, "Valid size:", len(validation_sequences_pad_qt))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1316245412.py in <cell line: 0>()
      3 n_train_eff = N_TRAIN - n_val
      4 
----> 5 validation_sequences_pad_qt = train_sequences_pad_qt[n_train_eff:]
      6 train_sequences_pad_qt = train_sequences_pad_qt[:n_train_eff]
      7 

NameError: name 'train_sequences_pad_qt' is not defined

## === cell 6
embedding_layer_question_title = tf.keras.layers.Embedding(
    vocab_size_question_title,
    300,
    weights=[embedding_matrix_question_title],
    input_length=MAX_SEQUENCE_LENGTH_question_title,
    name="Question_Title",
    trainable=False,
)

embedding_layer_question_body = tf.keras.layers.Embedding(
    vocab_size_question_body,
    300,
    weights=[embedding_matrix_question_body],
    input_length=MAX_SEQUENCE_LENGTH_question_body,
    name="Question_Body",
    trainable=False,
)

embedding_layer_answer = tf.keras.layers.Embedding(
    vocab_size_answer,
    300,
    weights=[embedding_matrix_answer],
    input_length=MAX_SEQUENCE_LENGTH_answer,
    name="Answer",
    trainable=False,
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/140792804.py in <cell line: 0>()
----> 1 embedding_layer_question_title = tf.keras.layers.Embedding(
      2     vocab_size_question_title,
      3     300,
      4     weights=[embedding_matrix_question_title],
      5     input_length=MAX_SEQUENCE_LENGTH_question_title,

NameError: name 'tf' is not defined

## === cell 7
def create_model_2():
    input_question_title = tf.keras.layers.Input(
        shape=(MAX_SEQUENCE_LENGTH_question_title,), name="IP_Question_Title"
    )
    embedded_question_title = embedding_layer_question_title(input_question_title)

    input_question_body = tf.keras.layers.Input(
        shape=(MAX_SEQUENCE_LENGTH_question_body,), name="IP_Question_Body"
    )
    embedded_question_body = embedding_layer_question_body(input_question_body)

    input_answer = tf.keras.layers.Input(
        shape=(MAX_SEQUENCE_LENGTH_answer,), name="IP_Answer"
    )
    embedded_answer = embedding_layer_answer(input_answer)

    tower_1 = tf.keras.layers.Conv1D(64, 5, activation="relu")(embedded_question_title)
    tower_2 = tf.keras.layers.Conv1D(64, 5, activation="relu")(embedded_question_body)
    tower_3 = tf.keras.layers.Conv1D(64, 5, activation="relu")(embedded_answer)

    concat = tf.keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
    max_pool = tf.keras.layers.MaxPooling1D(9)(concat)

    tower_1a = tf.keras.layers.Conv1D(64, 5, activation="relu")(max_pool)
    tower_2b = tf.keras.layers.Conv1D(64, 7, activation="relu")(max_pool)
    tower_3c = tf.keras.layers.Conv1D(64, 9, activation="relu")(max_pool)

    concat2 = tf.keras.layers.concatenate([tower_1a, tower_2b, tower_3c], axis=1)
    max_pool2 = tf.keras.layers.MaxPooling1D(9)(concat2)

    convP = tf.keras.layers.Conv1D(64, 9, activation="relu")(max_pool2)
    flatten = tf.keras.layers.Flatten()(convP)
    dropout = tf.keras.layers.Dropout(0.7)(flatten)

    dense = tf.keras.layers.Dense(128, activation="relu")(dropout)
    preds = tf.keras.layers.Dense(30, activation="sigmoid", name="Output")(dense)

    model_created = tf.keras.models.Model(
        [input_question_title, input_question_body, input_answer],
        preds,
        name="Model_Google_QUEST",
    )
    return model_created


model_Google_QUEST = create_model_2()
model_Google_QUEST.summary()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/296653361.py in <cell line: 0>()
     44 
     45 
---> 46 model_Google_QUEST = create_model_2()
     47 model_Google_QUEST.summary()
     48 

/tmp/ipykernel_55/296653361.py in create_model_2()
      1 def create_model_2():
----> 2     input_question_title = tf.keras.layers.Input(
      3         shape=(MAX_SEQUENCE_LENGTH_question_title,), name="IP_Question_Title"
      4     )
      5     embedded_question_title = embedding_layer_question_title(input_question_title)

NameError: name 'tf' is not defined

## === cell 8
def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(
                col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])
            ).correlation
        )
    return np.nanmean(rhos)


class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_data = {
            "IP_Question_Title": train_sequences_pad_qt,
            "IP_Question_Body": train_sequences_pad_qb,
            "IP_Answer": train_sequences_pad_ans,
        }
        self.train_target = df_train_small[output_categories].values[:n_train_eff]

        self.validation_data = {
            "IP_Question_Title": validation_sequences_pad_qt,
            "IP_Question_Body": validation_sequences_pad_qb,
            "IP_Answer": validation_sequences_pad_ans,
        }
        self.validation_target = df_train_small[output_categories].values[n_train_eff:]

        self.valid_predictions = []

    def on_epoch_end(self, epoch, logs=None):
        if len(self.validation_target) == 0:
            return
        self.valid_predictions.append(
            self.model.predict(self.validation_data, verbose=0)
        )
        rho_val = compute_spearmanr(
            self.validation_target, np.average(self.valid_predictions, axis=0)
        )
        print("\nvalidation rho: %.4f" % rho_val)


custom_callback = CustomCallback()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/829792861.py in <cell line: 0>()
     10 
     11 
---> 12 class CustomCallback(tf.keras.callbacks.Callback):
     13     def on_train_begin(self, logs=None):
     14         self.train_data = {

NameError: name 'tf' is not defined

## === cell 9
train_data = {
    "IP_Question_Title": train_sequences_pad_qt,
    "IP_Question_Body": train_sequences_pad_qb,
    "IP_Answer": train_sequences_pad_ans,
}
train_target = df_train_small[output_categories].values[:n_train_eff]

valid_data = {
    "IP_Question_Title": validation_sequences_pad_qt,
    "IP_Question_Body": validation_sequences_pad_qb,
    "IP_Answer": validation_sequences_pad_ans,
}
valid_target = df_train_small[output_categories].values[n_train_eff:]

optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.01)
model_Google_QUEST.compile(loss="mean_squared_error", optimizer=optimizer_adam)

model_Google_QUEST.fit(
    train_data,
    train_target,
    validation_data=(valid_data, valid_target) if len(valid_target) > 0 else None,
    epochs=100,
    batch_size=64,
    verbose=1,
    callbacks=[custom_callback],
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1145923143.py in <cell line: 0>()
      1 train_data = {
----> 2     "IP_Question_Title": train_sequences_pad_qt,
      3     "IP_Question_Body": train_sequences_pad_qb,
      4     "IP_Answer": train_sequences_pad_ans,
      5 }

NameError: name 'train_sequences_pad_qt' is not defined

## === cell 10
train_prediction = model_Google_QUEST.predict(train_data, verbose=0)
print(
    "Train Spearman Rank Correlation:",
    compute_spearmanr(train_target, train_prediction),
)

if len(valid_target) > 0:
    validation_prediction = model_Google_QUEST.predict(valid_data, verbose=0)
    print(
        "Validation Spearman Rank Correlation:",
        compute_spearmanr(valid_target, validation_prediction),
    )



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2033098971.py in <cell line: 0>()
----> 1 train_prediction = model_Google_QUEST.predict(train_data, verbose=0)
      2 print(
      3     "Train Spearman Rank Correlation:",
      4     compute_spearmanr(train_target, train_prediction),
      5 )

NameError: name 'model_Google_QUEST' is not defined

## === cell 11
test_data = {
    "IP_Question_Title": test_sequences_pad_qt,
    "IP_Question_Body": test_sequences_pad_qb,
    "IP_Answer": test_sequences_pad_ans,
}
test_prediction = model_Google_QUEST.predict(test_data, verbose=0)

test_prediction = np.clip(test_prediction, 0.0, 1.0)

submission_df = pd.DataFrame({"qa_id": df_test["qa_id"].values})
pred_df = pd.DataFrame(test_prediction, columns=output_categories)
submission_df = pd.concat([submission_df, pred_df], axis=1)

submission_df = submission_df[df_sub.columns]

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "shape:", submission_df.shape)
print(submission_df.head())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1410949604.py in <cell line: 0>()
      1 test_data = {
----> 2     "IP_Question_Title": test_sequences_pad_qt,
      3     "IP_Question_Body": test_sequences_pad_qb,
      4     "IP_Answer": test_sequences_pad_ans,
      5 }

NameError: name 'test_sequences_pad_qt' is not defined
