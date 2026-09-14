# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os, sys
import re
import gc
import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

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



## === cell 1
BASE = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"
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


def _to_py_str_list(x):
    if tf.is_tensor(x):
        x = x.numpy()
    x = np.asarray(x)
    out = []
    for v in x.reshape(-1):
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="ignore"))
        else:
            out.append(str(v))
    return out


def encode_batch(text_batch):
    texts = _to_py_str_list(text_batch)
    enc = tokenizer(
        text=texts,  # must be list[str]
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors="np",
    )
    return enc["input_ids"].astype(np.int32), enc["attention_mask"].astype(np.int32)


def tf_encode_batch(text_batch, y_batch):
    input_ids_np, attn_np = tf.numpy_function(
        func=encode_batch,
        inp=[text_batch],
        Tout=[tf.int32, tf.int32],
    )
    input_ids_np.set_shape([None, max_length])
    attn_np.set_shape([None, max_length])
    return {"input_ids": input_ids_np, "attention_mask": attn_np}, y_batch


train_ds = tf.data.Dataset.from_tensor_slices((train_text.astype("U"), train_y_tr))
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
train_ds = train_ds.map(tf_encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_text.astype("U"), val_y))
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False)
val_ds = val_ds.map(tf_encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

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
def tf_encode_batch_x(text_batch):
    input_ids_np, attn_np = tf.numpy_function(
        func=encode_batch,
        inp=[text_batch],
        Tout=[tf.int32, tf.int32],
    )
    input_ids_np.set_shape([None, max_length])
    attn_np.set_shape([None, max_length])
    return {"input_ids": input_ids_np, "attention_mask": attn_np}


test_ds = tf.data.Dataset.from_tensor_slices(test_sentences.astype("U"))
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)
test_ds = test_ds.map(tf_encode_batch_x, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

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
