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
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
transformers==4.53.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Target score

0.98559

# 6. Current score

0.7212

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.72109) has done: 'The timeout is dominated by on-the-fly Python tokenization inside `tf.data` via `tf.numpy_function`, which forces per-batch GIL-bound work and prevents efficient pipelining, especially for 560k train + 550k test rows. To preserve the exact model and training semantics, I keep the same BERT, pooling, head, loss, optimizer, batch size, and 1 epoch, but pre-tokenize the full train/val/test once using the same tokenizer settings and feed pure TensorFlow tensors/datasets to training/prediction. This removes the expensive per-step Python callback overhead while keeping identical tokenization output and model behavior (negligible FP diffs only). I also ensure deterministic split is preserved and avoid extra copies by using NumPy int32 arrays directly.'
- What this solution (achieved 0.7212) has done: 'Main bottlenecks are per-example Python tokenization via `tf.py_function` (very slow for ~1.1M texts) and caching huge token datasets to disk (extra I/O and serialization). The core model and training stay identical, but we move tokenization to a single batched HuggingFace tokenizer call (fast Rust backend) and feed pre-tokenized `input_ids/attention_mask` directly to `tf.data` (pure-TF, no Python in the input pipeline). We also remove dataset caching (not beneficial for single-epoch training and single-pass inference at this scale) and keep determinism/seed behavior unchanged. These changes preserve the exact tokenization logic (same tokenizer, max_length, padding, truncation) and the same train/val split and training loop semantics.'
- What this solution (achieved 0.7212) has done: 'We fix the immediate crash caused by an incompatibility between `protobuf==6.x` and libraries used by TensorFlow/Transformers in this environment by forcing the pure-Python protobuf implementation before importing TensorFlow/Transformers. This is a minimal change (environment variable only) and should restore end-to-end execution without altering the model/training logic. The rest of the pipeline (data cleaning, tokenizer settings, model, optimizer, epoch count, and submission formatting) is kept identical to preserve evaluation semantics and keep score changes limited to negligible floating-point differences. Finally, we keep the submission validation and ensure `/kaggle/working/submission.csv` is always written.'
- What this solution (achieved 0.7212) has done: 'I fix the crash happening before any training by resolving the protobuf/Transformers incompatibility that triggers `MessageFactory.GetPrototype` errors in this environment. The most reliable minimal fix on Kaggle here is to pin protobuf to the 3.20.x runtime inside the notebook session (no model/loop changes) and only then import TensorFlow/Transformers. After that, the rest of your pipeline (cleaning, tokenizer settings, BERT forward pass, pooling/head, optimizer, 1 epoch, and submission formatting) stays identical to preserve semantics; this should also restore your ability to actually train and predict, which move the score upward toward the target from the current broken state. I also add a tiny safety fallback to use the alternative dataset path if the expected one isn’t present, without changing file contents.'

# 9. Code solution

## === cell 0
import os, sys, re, gc, subprocess
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from importlib.metadata import version

        v = version("protobuf")
        major = int(v.split(".", 1)[0])
        if major >= 5:
            raise RuntimeError(f"Incompatible protobuf version detected: {v}")
    except Exception as e:
        print("Protobuf compatibility fix:", repr(e))
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
        )
        import importlib

        importlib.invalidate_caches()


_ensure_protobuf_compat()

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import tensorflow as tf
from transformers import BertConfig, BertTokenizerFast, TFAutoModel
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

print("TF:", tf.__version__)

tf.keras.utils.set_random_seed(42)
try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print("GPU config warning:", repr(e))

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception as e:
    print("Thread config warning:", repr(e))



## === cell 1
BASE = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"
if not os.path.exists(BASE):
    BASE = "/kaggle/data/jigsaw-toxic-comment-classification-challenge"

train_path = f"{BASE}/train.csv"
test_path = f"{BASE}/test.csv"
sample_path = f"{BASE}/sample_submission.csv"

usecols_train = [
    "id",
    "comment_text",
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
df = pd.read_csv(train_path, usecols=usecols_train)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.head()



## === cell 2
_replacements = [
    (re.compile(r"what's"), "what is "),
    (re.compile(r"\'s"), " "),
    (re.compile(r"\'ve"), " have "),
    (re.compile(r"can't"), "cannot "),
    (re.compile(r"n't"), " not "),
    (re.compile(r"i'm"), "i am "),
    (re.compile(r"\'re"), " are "),
    (re.compile(r"\'d"), " would "),
    (re.compile(r"\'ll"), " will "),
    (re.compile(r"\'scuse"), " excuse "),
]
_nonword = re.compile(r"\W")
_multispace = re.compile(r"\s+")


def clean_text(text):
    text = str(text).lower()
    for pat, repl in _replacements:
        text = pat.sub(repl, text)
    text = _nonword.sub(" ", text)
    text = _multispace.sub(" ", text)
    text = text.strip(" ")
    return text




## === cell 3
s = df["comment_text"].astype("string").fillna("CVxTz").str.lower()
for pat, repl in _replacements:
    s = s.str.replace(pat, repl, regex=True)
s = s.str.replace(_nonword, " ", regex=True)
s = s.str.replace(_multispace, " ", regex=True).str.strip()
df["comment_text"] = s



## === cell 4
train_sentences = df["comment_text"].values  # already filled
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train_y = df[list_classes].values.astype("float32")

df.head()



## === cell 5
model_name = "bert-base-uncased"
max_length = 128

config = BertConfig.from_pretrained(model_name)
tokenizer = BertTokenizerFast.from_pretrained(
    pretrained_model_name_or_path=model_name, config=config
)

bert = TFAutoModel.from_pretrained(model_name)




## === cell 6
class BertLastHiddenState(tf.keras.layers.Layer):
    def __init__(self, bert_model, **kwargs):
        super().__init__(**kwargs)
        self.bert_model = bert_model

    def call(self, inputs):
        ids, mask = inputs
        out = self.bert_model(input_ids=ids, attention_mask=mask, training=False)
        return out.last_hidden_state

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0][0],
            input_shape[0][1],
            self.bert_model.config.hidden_size,
        )


input_ids = Input(shape=(max_length,), name="input_ids", dtype=tf.int32)
attention_mask = Input(shape=(max_length,), name="attention_mask", dtype=tf.int32)

x = BertLastHiddenState(bert, name="bert_last_hidden_state")(
    [input_ids, attention_mask]
)
x2 = GlobalAveragePooling1D()(x)
y = Dense(len(list_classes), activation="sigmoid", name="outputs")(x2)

model = Model(
    inputs={"input_ids": input_ids, "attention_mask": attention_mask}, outputs=y
)
model.summary()



## === cell 7
optimizer = Adam(learning_rate=1e-5)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])



## === cell 8
BATCH_SIZE = 32
VAL_FRAC = 0.1

n = len(train_sentences)
val_size = int(n * VAL_FRAC)
train_size = n - val_size

train_text = train_sentences[:train_size]
val_text = train_sentences[train_size:]
train_y_tr = train_y[:train_size]
val_y = train_y[train_size:]


def _encode_texts_batched(texts):
    if isinstance(texts, np.ndarray):
        texts_list = texts.astype("U").tolist()
    else:
        texts_list = list(texts)

    enc = tokenizer(
        texts_list,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors="np",
    )
    ids = enc["input_ids"].astype(np.int32, copy=False)
    mask = enc["attention_mask"].astype(np.int32, copy=False)
    return ids, mask


def make_ds_from_encoded(ids, mask, labels=None, batch_size=32):
    x = {"input_ids": ids, "attention_mask": mask}
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(x)
    else:
        ds = tf.data.Dataset.from_tensor_slices((x, labels))
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


train_ids, train_mask = _encode_texts_batched(train_text)
val_ids, val_mask = _encode_texts_batched(val_text)

train_ds = make_ds_from_encoded(
    train_ids, train_mask, labels=train_y_tr, batch_size=BATCH_SIZE
)
val_ds = make_ds_from_encoded(val_ids, val_mask, labels=val_y, batch_size=BATCH_SIZE)

del train_text, val_text, train_sentences
gc.collect()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    verbose=1,
)



## === cell 9
test_df = pd.read_csv(test_path, usecols=["id", "comment_text"])

ts = test_df["comment_text"].astype("string").fillna("CVxTz").str.lower()
for pat, repl in _replacements:
    ts = ts.str.replace(pat, repl, regex=True)
ts = ts.str.replace(_nonword, " ", regex=True)
ts = ts.str.replace(_multispace, " ", regex=True).str.strip()
test_df["comment_text"] = ts

test_sentences = test_df["comment_text"].values

del df
gc.collect()



## === cell 10
test_ids, test_mask = _encode_texts_batched(test_sentences)
test_ds = make_ds_from_encoded(test_ids, test_mask, labels=None, batch_size=BATCH_SIZE)

del test_sentences
gc.collect()

predictions = model.predict(
    test_ds,
    verbose=1,
)

print("Predictions shape:", predictions.shape)



## === cell 11
submission = pd.DataFrame(predictions, columns=list_classes)
submission.insert(0, "id", test_df["id"].values)
submission = submission[["id"] + list_classes]

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
submission.head()



## === cell 12
sample = pd.read_csv(sample_path, nrows=5)
print("Sample columns:", list(sample.columns))
print("Submission columns:", list(submission.columns))
print("Submission rows:", len(submission))
print("Test rows:", len(test_df))
assert list(submission.columns) == list(sample.columns)
assert len(submission) == len(test_df)
print("Submission validated.")
