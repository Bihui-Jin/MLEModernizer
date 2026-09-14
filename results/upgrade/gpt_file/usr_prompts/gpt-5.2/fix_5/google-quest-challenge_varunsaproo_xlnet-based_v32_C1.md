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
import tensorflow_hub as hub
from scipy.stats import spearmanr

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF:", tf.__version__)
print("TF Hub:", hub.__version__)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
BERT_ENCODER = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
encoder_layer = hub.KerasLayer(BERT_ENCODER, trainable=False, name="bert_encoder")




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




## === cell 6
def load_bert_vocab_from_hub_or_fallback():
    try:
        vocab_file = encoder_layer.resolved_object.vocab_file.asset_path.numpy().decode(
            "utf-8"
        )
        do_lower_case = bool(encoder_layer.resolved_object.do_lower_case.numpy())
        if os.path.exists(vocab_file):
            return vocab_file, do_lower_case
    except Exception:
        pass
    candidates = [
        "/kaggle/input/bert-base-uncased/vocab.txt",
        "/kaggle/input/google-bert/bert-base-uncased/vocab.txt",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c, True
    raise FileNotFoundError(
        "Could not locate a BERT vocab.txt. Encoder assets were not accessible and no fallback vocab found."
    )


VOCAB_FILE, DO_LOWER_CASE = load_bert_vocab_from_hub_or_fallback()
print("Using vocab:", VOCAB_FILE, "do_lower_case:", DO_LOWER_CASE)

tokenizer = tf.keras.layers.TextVectorization(
    standardize=None,  # we already cleaned text
    split="whitespace",  # fast; then we map token->id via vocab lookup below
    output_mode="int",
    output_sequence_length=256,  # fixed seq length
)

with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    bert_vocab = [line.strip() for line in f if line.strip()]

bert_vocab_table = tf.lookup.StaticVocabularyTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(bert_vocab),
        values=tf.constant(list(range(len(bert_vocab))), dtype=tf.int64),
    ),
    num_oov_buckets=1,
)

CLS_ID = int(bert_vocab.index("[CLS]")) if "[CLS]" in bert_vocab else 101
SEP_ID = int(bert_vocab.index("[SEP]")) if "[SEP]" in bert_vocab else 102
PAD_ID = int(bert_vocab.index("[PAD]")) if "[PAD]" in bert_vocab else 0
UNK_ID = int(bert_vocab.index("[UNK]")) if "[UNK]" in bert_vocab else 100

SEQ_LEN = 256


def _basic_tokenize(texts):
    if DO_LOWER_CASE:
        texts = tf.strings.lower(texts)
    tokens = tf.strings.split(texts)
    return tokens


def _tokens_to_ids(tokens_rt):
    ids = bert_vocab_table.lookup(tokens_rt)
    oov_id = tf.cast(len(bert_vocab), tf.int64)  # the oov bucket id
    ids = tf.where(ids == oov_id, tf.cast(UNK_ID, tf.int64), ids)
    return ids


def preprocess_data(df):
    title = tf.constant(df["question_title"].astype(str).tolist())
    body = tf.constant(df["question_body"].astype(str).tolist())
    answer = tf.constant(df["answer"].astype(str).tolist())

    t_tok = _basic_tokenize(title)
    b_tok = _basic_tokenize(body)
    a_tok = _basic_tokenize(answer)

    t_ids = _tokens_to_ids(t_tok)
    b_ids = _tokens_to_ids(b_tok)
    a_ids = _tokens_to_ids(a_tok)

    cls = tf.ragged.constant([[CLS_ID]], dtype=tf.int64)
    sep = tf.ragged.constant([[SEP_ID]], dtype=tf.int64)

    bs = tf.shape(title)[0]
    cls = tf.RaggedTensor.from_row_splits(
        tf.repeat([CLS_ID], bs), tf.range(bs + 1, dtype=tf.int64)
    )
    sep = tf.RaggedTensor.from_row_splits(
        tf.repeat([SEP_ID], bs), tf.range(bs + 1, dtype=tf.int64)
    )

    ids_rt = tf.concat([cls, t_ids, sep, b_ids, sep, a_ids, sep], axis=1)

    seg0_len = (
        1 + t_ids.row_lengths() + 1 + b_ids.row_lengths() + 1
    )  # CLS + title + SEP + body + SEP
    seg1_len = a_ids.row_lengths() + 1  # answer + SEP
    seg_rt = tf.concat(
        [
            tf.ragged.map_flat_values(
                lambda x: tf.zeros_like(x, dtype=tf.int64),
                tf.RaggedTensor.from_row_lengths(
                    tf.ones_like(seg0_len, dtype=tf.int64), seg0_len
                ),
            ),
            tf.ragged.map_flat_values(
                lambda x: tf.ones_like(x, dtype=tf.int64),
                tf.RaggedTensor.from_row_lengths(
                    tf.ones_like(seg1_len, dtype=tf.int64), seg1_len
                ),
            ),
        ],
        axis=1,
    )

    input_ids = ids_rt.to_tensor(default_value=PAD_ID, shape=[None, SEQ_LEN])
    segment_ids = seg_rt.to_tensor(default_value=0, shape=[None, SEQ_LEN])

    attention_mask = tf.cast(tf.not_equal(input_ids, PAD_ID), tf.int32)

    return {
        "input_ids": tf.cast(input_ids, tf.int32),
        "attention_mask": tf.cast(attention_mask, tf.int32),
        "segment_ids": tf.cast(segment_ids, tf.int32),
    }




## === cell 7
train_df = pd.read_csv(DIR + "/train.csv")
test_df = pd.read_csv(DIR + "/test.csv")

train_df = clean_data(train_df)
test_df = clean_data(test_df)

train_encoded_inputs_tensor = preprocess_data(train_df)
test_encoded_inputs_tensor = preprocess_data(test_df)

print("Train input_ids shape:", train_encoded_inputs_tensor["input_ids"].shape)
print("Test input_ids shape:", test_encoded_inputs_tensor["input_ids"].shape)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3620237307.py in <cell line: 0>()
      5 test_df = clean_data(test_df)
      6 
----> 7 train_encoded_inputs_tensor = preprocess_data(train_df)
      8 test_encoded_inputs_tensor = preprocess_data(test_df)
      9 

/tmp/ipykernel_11/698443743.py in preprocess_data(df)
    104     )
    105 
--> 106     ids_rt = tf.concat([cls, t_ids, sep, b_ids, sep, a_ids, sep], axis=1)
    107 
    108     # Segment ids: 0 for [CLS]+title+[SEP]+body+[SEP], 1 for answer+[SEP]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: cannot compute ConcatV2 as input #1(zero-based) was expected to be a int32 tensor but is a int64 tensor [Op:ConcatV2] name: concat

## === cell 8
labels = tf.convert_to_tensor(train_df.iloc[:, -30:].values, dtype=tf.float32)
columns = list(train_df.columns[-30:])
print("Num labels:", labels.shape, "Num columns:", len(columns))




## === cell 9
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




## === cell 10
tf.keras.backend.clear_session()

SEQ_LEN = int(train_encoded_inputs_tensor["input_ids"].shape[1])

input_id = tf.keras.layers.Input(shape=(SEQ_LEN,), dtype=tf.int32, name="input_ids")
input_mask = tf.keras.layers.Input(
    shape=(SEQ_LEN,), dtype=tf.int32, name="attention_mask"
)
input_type = tf.keras.layers.Input(shape=(SEQ_LEN,), dtype=tf.int32, name="segment_ids")

encoder_outputs = encoder_layer(
    {
        "input_word_ids": input_id,
        "input_mask": input_mask,
        "input_type_ids": input_type,
    }
)
out = encoder_outputs["pooled_output"]
model_1 = tf.keras.Model(inputs=[input_id, input_mask, input_type], outputs=out)

print("Embedding model output shape:", model_1.output_shape)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1092982541.py in <cell line: 0>()
      1 tf.keras.backend.clear_session()
      2 
----> 3 SEQ_LEN = int(train_encoded_inputs_tensor["input_ids"].shape[1])
      4 
      5 input_id = tf.keras.layers.Input(shape=(SEQ_LEN,), dtype=tf.int32, name="input_ids")

NameError: name 'train_encoded_inputs_tensor' is not defined

## === cell 11
def _make_ds(encoded, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(
        (encoded["input_ids"], encoded["attention_mask"], encoded["segment_ids"])
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.map(
        lambda ids, mask, seg: ((ids, mask, seg),), num_parallel_calls=tf.data.AUTOTUNE
    )
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



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2727033877.py in <cell line: 0>()
     14 while True:
     15     try:
---> 16         train_ds = _make_ds(train_encoded_inputs_tensor, _embed_bs)
     17         test_ds = _make_ds(test_encoded_inputs_tensor, _embed_bs)
     18 

NameError: name 'train_encoded_inputs_tensor' is not defined

## === cell 12
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



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/795163856.py in <cell line: 0>()
      5 )
      6 
----> 7 inp = tf.keras.layers.Input(shape=(data.shape[1],))
      8 elu_dense = tf.keras.layers.Dense(
      9     512, activation="elu", kernel_initializer="glorot_normal"

NameError: name 'data' is not defined

## === cell 13
ans = model.predict([test_data], batch_size=16, verbose=1)

ans = np.clip(ans, 0.0, 1.0)
print("Pred shape:", ans.shape, "min/max:", ans.min(), ans.max())



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1805513830.py in <cell line: 0>()
----> 1 ans = model.predict([test_data], batch_size=16, verbose=1)
      2 
      3 ans = np.clip(ans, 0.0, 1.0)
      4 print("Pred shape:", ans.shape, "min/max:", ans.min(), ans.max())
      5 

NameError: name 'model' is not defined

## === cell 14
sample_submission = pd.read_csv(DIR + "/sample_submission.csv")
target_cols = list(sample_submission.columns[1:])

sub = pd.DataFrame(ans, columns=target_cols)
sub.insert(0, "qa_id", test_df["qa_id"].values)

print(sub.shape)
print(sub.head())



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3287485233.py in <cell line: 0>()
      2 target_cols = list(sample_submission.columns[1:])
      3 
----> 4 sub = pd.DataFrame(ans, columns=target_cols)
      5 sub.insert(0, "qa_id", test_df["qa_id"].values)
      6 

NameError: name 'ans' is not defined

## === cell 15
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3046177952.py in <cell line: 0>()
----> 1 sub.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", sub.shape)
      3 

NameError: name 'sub' is not defined

## === cell 16
assert sub.shape[0] == test_df.shape[0]
assert sub.shape[1] == 31
assert (sub[target_cols].values >= 0).all() and (sub[target_cols].values <= 1).all()
sub.head()

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3918999613.py in <cell line: 0>()
----> 1 assert sub.shape[0] == test_df.shape[0]
      2 assert sub.shape[1] == 31
      3 assert (sub[target_cols].values >= 0).all() and (sub[target_cols].values <= 1).all()
      4 sub.head()

NameError: name 'sub' is not defined
