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

-0.01188

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved -0.01188) has done: 'I fix the TensorFlow import crash by removing the unused `spearmanr` import (it triggers a protobuf incompatibility in this environment) and rewriting the Spearman metric to use a pure-NumPy rank correlation implementation inside `tf.numpy_function`. I also fix the Keras Functional graph error by replacing `tf.identity(inp_vec)` with a Keras layer pass-through so `model_1` can be constructed and used for `predict`. After that, the rest of your pipeline (TF-IDF vectorization → Dense(512, elu) → Dense(30, sigmoid) with the same optimizer/loss/fit settings) run end-to-end and write a valid `submission.csv` with the correct columns and [0,1] bounds.'

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

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF:", tf.__version__)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
BERT_ENCODER = None
encoder_layer = None




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
VEC_TOKENS = 50000  # keep moderate for speed/memory; deterministic with fixed seed
NGRAMS = (1, 2)


def build_vectorizer(train_texts):
    vec = tf.keras.layers.TextVectorization(
        standardize=None,  # already cleaned
        split="whitespace",
        ngrams=NGRAMS,
        max_tokens=VEC_TOKENS,
        output_mode="tf_idf",
    )
    vec.adapt(train_texts)
    return vec


def preprocess_data(df, vectorizer):
    texts = (
        df["question_title"].astype(str)
        + " [SEP] "
        + df["question_body"].astype(str)
        + " [SEP] "
        + df["answer"].astype(str)
    ).tolist()
    texts = tf.constant(texts)
    features = vectorizer(texts)
    return features




## === cell 7
train_df = pd.read_csv(DIR + "/train.csv")
test_df = pd.read_csv(DIR + "/test.csv")

train_df = clean_data(train_df)
test_df = clean_data(test_df)

train_texts = tf.constant(
    (
        train_df["question_title"].astype(str)
        + " [SEP] "
        + train_df["question_body"].astype(str)
        + " [SEP] "
        + train_df["answer"].astype(str)
    ).tolist()
)

vectorizer = build_vectorizer(train_texts)

train_encoded_inputs_tensor = preprocess_data(train_df, vectorizer)
test_encoded_inputs_tensor = preprocess_data(test_df, vectorizer)

print("Train features shape:", train_encoded_inputs_tensor.shape)
print("Test features shape:", test_encoded_inputs_tensor.shape)



## === cell 8
labels = tf.convert_to_tensor(train_df.iloc[:, -30:].values, dtype=tf.float32)
columns = list(train_df.columns[-30:])
print("Num labels:", labels.shape, "Num columns:", len(columns))




## === cell 9
def _rankdata_average(x):
    """Average ranks for ties, like scipy.stats.rankdata(method='average')."""
    x = np.asarray(x)
    n = x.shape[0]
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)

    x_sorted = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x_sorted[j + 1] == x_sorted[i]:
            j += 1
        avg = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _spearman_corr_1d(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra = _rankdata_average(a)
    rb = _rankdata_average(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum()) * np.sqrt((rb * rb).sum())
    if denom == 0.0:
        return 0.0
    return float((ra * rb).sum() / denom)


def SpearmanCorrCoeff_2(A, B):
    overall_score = 0.0
    rng = np.random.RandomState(SEED)
    x1 = rng.normal(loc=1e-8, scale=1e-12, size=A.shape[0])
    x2 = rng.normal(loc=1e-8, scale=1e-12, size=B.shape[0])
    for index, col in enumerate(columns):
        overall_score += _spearman_corr_1d(A[:, index] + x1, B[:, index] + x2) / 30.0
    return np.array(overall_score, dtype=np.float64)


def tf_SpearmanCorrCoeff(A, B):
    return tf.numpy_function(SpearmanCorrCoeff_2, [A, B], Tout=tf.float64)




## === cell 10
tf.keras.backend.clear_session()

inp_vec = tf.keras.layers.Input(
    shape=(train_encoded_inputs_tensor.shape[1],),
    dtype=tf.float32,
    name="tfidf_features",
)

out_vec = tf.keras.layers.Lambda(lambda x: x, name="identity")(inp_vec)

model_1 = tf.keras.Model(inputs=[inp_vec], outputs=out_vec)
print("Embedding model output shape:", model_1.output_shape)




## === cell 11
def _make_ds_features(features, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(features)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


_embed_bs = BATCH_SIZE
while True:
    try:
        train_ds = _make_ds_features(train_encoded_inputs_tensor, _embed_bs)
        test_ds = _make_ds_features(test_encoded_inputs_tensor, _embed_bs)

        data = model_1.predict(train_ds, verbose=1)
        test_data = model_1.predict(test_ds, verbose=1)
        break
    except tf.errors.ResourceExhaustedError:
        _embed_bs = max(4, _embed_bs // 2)
        if _embed_bs == 4:
            raise

print("Used embedding batch size:", _embed_bs)
print("Train embeddings:", data.shape, "Test embeddings:", test_data.shape)



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
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/795163856.py in <cell line: 0>()
     18 )
     19 
---> 20 model.fit(
     21     [data],
     22     labels,

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

## === cell 13
ans = model.predict([test_data], batch_size=16, verbose=1)

ans = np.clip(ans, 0.0, 1.0)
print("Pred shape:", ans.shape, "min/max:", ans.min(), ans.max())



## === cell 14
sample_submission = pd.read_csv(DIR + "/sample_submission.csv")
target_cols = list(sample_submission.columns[1:])

sub = pd.DataFrame(ans, columns=target_cols)
sub.insert(0, "qa_id", test_df["qa_id"].values)

print(sub.shape)
print(sub.head())



## === cell 15
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)

assert sub.shape[0] == test_df.shape[0]
assert sub.shape[1] == 31
assert list(sub.columns) == ["qa_id"] + target_cols
vals = sub[target_cols].values
assert np.isfinite(vals).all()
assert (vals >= 0).all() and (vals <= 1).all()
sub.head()
