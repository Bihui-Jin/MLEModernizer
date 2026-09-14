# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.9

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import sys
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver
except Exception:
    _pb_ver = None


def _major(ver):
    try:
        return int(str(ver).split(".")[0])
    except Exception:
        return None


if _pb_ver is None or (_major(_pb_ver) is not None and _major(_pb_ver) >= 4):
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
    )
    import importlib

    if "google.protobuf" in sys.modules:
        importlib.reload(sys.modules["google.protobuf"])

from transformers import TFBertModel, BertConfig, BertTokenizerFast, TFAutoModel

from tensorflow.keras.layers import Input, Dropout, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical

import pandas as pd
import tensorflow as tf
import re
import numpy as np
from sklearn.model_selection import train_test_split


## === cell 1
!unzip ../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip
!unzip ../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip


## === cell 2
df=pd.read_csv('train.csv')

df = df.sample(frac=1)


## === cell 3
df.head()


## === cell 4
def clean_text(text):
    text = text.lower()
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
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


## === cell 5
df['comment_text'] = df['comment_text'].map(lambda x : clean_text(x))


## === cell 6
train_sentences = df["comment_text"].fillna("CVxTz").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train_y = df[list_classes].values


## === cell 7
df.head()


## === cell 8
model_name = 'bert-base-uncased'

max_length = 128

config = BertConfig.from_pretrained(model_name)

tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
bert = TFAutoModel.from_pretrained(model_name)


## === cell 9
input_ids = Input(shape=(max_length,), name="input_ids", dtype="int32")
attention_mask = Input(shape=(max_length,), name="attention_mask", dtype="int32")
inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

x = tf.keras.layers.Lambda(
    lambda t: bert(input_ids=t[0], attention_mask=t[1]).last_hidden_state,
    output_shape=(max_length, bert.config.hidden_size),
    name="bert_wrapper",
)([input_ids, attention_mask])


## === cell 10
x


## === cell 11
x2 = GlobalAveragePooling1D()(x[0])
y =Dense(len(list_classes), activation='sigmoid', name='outputs')(x2)

model = Model(inputs=inputs, outputs=y)

model.summary()


## --- ERROR in cell 11, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/542367789.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;31m#x2 =Dense(512, activation='relu')(x[1])[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0mx2[0m [0;34m=[0m [0mGlobalAveragePooling1D[0m[0;34m([0m[0;34m)[0m[0;34m([0m[0mx[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0;31m#x3 = Dropout(0.5)(x2)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0my[0m [0;34m=[0m[0mDense[0m[0;34m([0m[0mlen[0m[0;34m([0m[0mlist_classes[0m[0;34m)[0m[0;34m,[0m [0mactivation[0m[0;34m=[0m[0;34m'sigmoid'[0m[0;34m,[0m [0mname[0m[0;34m=[0m[0;34m'outputs'[0m[0;34m)[0m[0;34m([0m[0mx2[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py[0m in [0;36merror_handler[0;34m(*args, **kwargs)[0m
[1;32m    120[0m             [0;31m# To get the full stack trace, call:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    121[0m             [0;31m# `keras.config.disable_traceback_filtering()`[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 122[0;31m             [0;32mraise[0m [0me[0m[0;34m.[0m[0mwith_traceback[0m[0;34m([0m[0mfiltered_tb[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    123[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    124[0m             [0;32mdel[0m [0mfiltered_tb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/layers/input_spec.py[0m in [0;36massert_input_compatibility[0;34m(input_spec, inputs, layer_name)[0m
[1;32m    184[0m         [0;32mif[0m [0mspec[0m[0;34m.[0m[0mndim[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m [0;32mand[0m [0;32mnot[0m [0mspec[0m[0;34m.[0m[0mallow_last_axis_squeeze[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    185[0m             [0;32mif[0m [0mndim[0m [0;34m!=[0m [0mspec[0m[0;34m.[0m[0mndim[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 186[0;31m                 raise ValueError(
[0m[1;32m    187[0m                     [0;34mf'Input {input_index} of layer "{layer_name}" '[0m[0;34m[0m[0;34m[0m[0m
[1;32m    188[0m                     [0;34m"is incompatible with the layer: "[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: Input 0 of layer "global_average_pooling1d" is incompatible with the layer: expected ndim=3, found ndim=2. Full shape received: (128, 768)

## === cell 12
optimizer = Adam(lr=1e-5, decay=1e-6)
model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
