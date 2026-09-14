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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D, Lambda
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



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
input_ids = Input(shape=(max_length,), name="input_ids", dtype=tf.int32)
attention_mask = Input(shape=(max_length,), name="attention_mask", dtype=tf.int32)


def bert_last_hidden_state(inputs):
    ids, mask = inputs
    out = bert(input_ids=ids, attention_mask=mask, training=False)
    return out.last_hidden_state


x = Lambda(bert_last_hidden_state, name="bert_last_hidden_state")(
    [input_ids, attention_mask]
)
x2 = GlobalAveragePooling1D()(x)
y = Dense(len(list_classes), activation="sigmoid", name="outputs")(x2)

model = Model(
    inputs={"input_ids": input_ids, "attention_mask": attention_mask}, outputs=y
)
model.summary()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_11/770575414.py in <cell line: 0>()
     11 
     12 
---> 13 x = Lambda(bert_last_hidden_state, name="bert_last_hidden_state")(
     14     [input_ids, attention_mask]
     15 )

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/lambda_layer.py in compute_output_shape(self, input_shape)
     93                 return tree.map_structure(lambda x: x.shape, output_spec)
     94             except:
---> 95                 raise NotImplementedError(
     96                     "We could not automatically infer the shape of "
     97                     "the Lambda's output. Please specify the `output_shape` "

NotImplementedError: Exception encountered when calling Lambda.call().

We could not automatically infer the shape of the Lambda's output. Please specify the `output_shape` argument for this Lambda layer.

Arguments received by Lambda.call():
  • args=(['<KerasTensor shape=(None, 128), dtype=int32, sparse=False, name=input_ids>', '<KerasTensor shape=(None, 128), dtype=int32, sparse=False, name=attention_mask>'],)
  • kwargs={'mask': ['None', 'None']}

## === cell 7
optimizer = Adam(learning_rate=1e-5)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1874424425.py in <cell line: 0>()
      1 optimizer = Adam(learning_rate=1e-5)
----> 2 model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
      3 

NameError: name 'model' is not defined

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



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3385054172.py in <cell line: 0>()
----> 1 history = model.fit(
      2     x={"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]},
      3     y=train_y,
      4     validation_split=0.1,
      5     batch_size=32,

NameError: name 'model' is not defined

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

predictions.shape



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2989564521.py in <cell line: 0>()
----> 1 predictions = model.predict(
      2     x={"input_ids": test_x["input_ids"], "attention_mask": test_x["attention_mask"]},
      3     batch_size=32,
      4     verbose=1,
      5 )

NameError: name 'model' is not defined

## === cell 12
submission = pd.DataFrame(predictions, columns=list_classes)
submission.insert(0, "id", test_df["id"].values)
submission = submission[["id"] + list_classes]

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
submission.head()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2727938170.py in <cell line: 0>()
----> 1 submission = pd.DataFrame(predictions, columns=list_classes)
      2 submission.insert(0, "id", test_df["id"].values)
      3 submission = submission[["id"] + list_classes]
      4 
      5 out_path = "/kaggle/working/submission.csv"

NameError: name 'predictions' is not defined

## === cell 13
sample = pd.read_csv(sample_path, nrows=5)
print("Sample columns:", list(sample.columns))
print("Submission columns:", list(submission.columns))
print("Submission rows:", len(submission))
print("Test rows:", len(test_df))
assert list(submission.columns) == list(sample.columns)
assert len(submission) == len(test_df)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/115233687.py in <cell line: 0>()
      1 sample = pd.read_csv(sample_path, nrows=5)
      2 print("Sample columns:", list(sample.columns))
----> 3 print("Submission columns:", list(submission.columns))
      4 print("Submission rows:", len(submission))
      5 print("Test rows:", len(test_df))

NameError: name 'submission' is not defined
