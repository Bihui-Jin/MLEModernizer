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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
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

df = pd.read_csv(train_path)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.head()




## === cell 2
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" ")
    return text




## === cell 3
df["comment_text"] = df["comment_text"].map(clean_text)



## === cell 4
train_sentences = df["comment_text"].fillna("CVxTz").values
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
x = tokenizer(
    text=list(train_sentences),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True,
    return_tensors="tf",
    return_token_type_ids=False,
    return_attention_mask=True,
)



## === cell 9
history = model.fit(
    x={"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]},
    y=train_y,
    validation_split=0.1,
    batch_size=32,
    epochs=1,
    verbose=1,
)



## === cell 10
test_df = pd.read_csv(test_path)
test_df["comment_text"] = test_df["comment_text"].map(clean_text)
test_sentences = test_df["comment_text"].fillna("CVxTz").values

test_x = tokenizer(
    text=list(test_sentences),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True,
    return_tensors="tf",
    return_token_type_ids=False,
    return_attention_mask=True,
)

del test_sentences
del df
del x
gc.collect()



## === cell 11
predictions = model.predict(
    x={"input_ids": test_x["input_ids"], "attention_mask": test_x["attention_mask"]},
    batch_size=32,
    verbose=1,
)

print("Predictions shape:", predictions.shape)



## === cell 12
submission = pd.DataFrame(predictions, columns=list_classes)
submission.insert(0, "id", test_df["id"].values)
submission = submission[["id"] + list_classes]

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
submission.head()



## === cell 13
sample = pd.read_csv(sample_path, nrows=5)
print("Sample columns:", list(sample.columns))
print("Submission columns:", list(submission.columns))
print("Submission rows:", len(submission))
print("Test rows:", len(test_df))
assert list(submission.columns) == list(sample.columns)
assert len(submission) == len(test_df)
print("Submission validated.")
