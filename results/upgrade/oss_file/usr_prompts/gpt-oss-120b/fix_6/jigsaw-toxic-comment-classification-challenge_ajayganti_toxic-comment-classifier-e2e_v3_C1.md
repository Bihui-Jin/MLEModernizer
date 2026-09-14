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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
plotly==5.24.1
plotly-express==0.4.1
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    Input,
    LSTM,
    GlobalMaxPool1D,
    Dropout,
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_auc_score
import re, warnings, nltk, os

warnings.simplefilter(action="ignore")
nltk.download("punkt")
nltk.download("stopwords")

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass



## === cell 1
import subprocess, sys

subprocess.run(
    ["ls", "-R", "/kaggle/input/jigsaw-toxic-comment-classification-challenge/"],
    check=False,
)



## === cell 2
train_df = pd.read_csv(
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
)



## === cell 3
test_df = pd.read_csv(
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip"
)



## === cell 4
train_df["comment_text"].iloc[0]



## === cell 5
label_counts = train_df[
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
].sum()
pd.DataFrame(label_counts, columns=["Count"])



## === cell 6
temp = pd.DataFrame(label_counts, columns=["Count"])
sns.barplot(x=temp.index, y="Count", data=temp)
plt.xticks(rotation=45)
plt.show()



## === cell 7
from nltk import word_tokenize

train_df["tokenized_text"] = train_df["comment_text"].apply(lambda x: word_tokenize(x))
lengths = train_df["tokenized_text"].apply(len).tolist()
train_df["comment_text"].iloc[np.argmax(lengths)]



## === cell 8
import plotly.express as px

px.histogram(lengths)



## === cell 9
from nltk.corpus import stopwords


def process_text(data):
    stop = set(stopwords.words("english"))
    data["processed_text"] = data["comment_text"].str.replace("\n", " ", regex=False)
    data["processed_text"] = data["processed_text"].apply(
        lambda x: re.sub(r"http://\S+|https://\S+", "urls", x.lower())
    )
    data["processed_text"] = data["processed_text"].apply(
        lambda x: re.sub(r"[^a-zA-Z ]+", "", x.lower())
    )
    data["processed_text"] = data["processed_text"].apply(
        lambda x: " ".join([w for w in x.split() if w not in stop])
    )
    data["processed_text"] = data["processed_text"].apply(
        lambda x: re.sub(r" {2,}", " ", x).strip()
    )
    return data




## === cell 10
train = process_text(train_df.copy())
test = process_text(test_df.copy())



## === cell 11
train["processed_text"] = train.apply(
    lambda x: (
        x["comment_text"] if len(x["processed_text"]) == 0 else x["processed_text"]
    ),
    axis=1,
)
test["processed_text"] = test.apply(
    lambda x: (
        x["comment_text"] if len(x["processed_text"]) == 0 else x["processed_text"]
    ),
    axis=1,
)



## === cell 12
num_words = 30000
tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
tokenizer.fit_on_texts(train["processed_text"])
train_tokens = tokenizer.texts_to_sequences(train["processed_text"])
test_tokens = tokenizer.texts_to_sequences(test["processed_text"])
train_seq = pad_sequences(train_tokens, maxlen=300, padding="post", truncating="post")
test_seq = pad_sequences(test_tokens, maxlen=300, padding="post", truncating="post")



## === cell 13
print("Train sequence shape:", train_seq.shape)
print("Test sequence shape:", test_seq.shape)



## === cell 14
train_labels = train[
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
].values



## === cell 15
filepath = "/kaggle/working/best_model_{epoch:02d}.keras"
save_model_callback = ModelCheckpoint(
    filepath=filepath, monitor="val_auc", verbose=1, save_best_only=True, mode="max"
)



## === cell 16
earlystop = EarlyStopping(
    monitor="val_auc", min_delta=0.01, patience=2, verbose=1, restore_best_weights=True
)



## === cell 17
tf.keras.backend.clear_session()
input_layer = Input(shape=(300,))
x = Embedding(num_words, 200, mask_zero=True)(input_layer)
x = LSTM(60, return_sequences=True)(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
output_layer = Dense(6, activation="sigmoid")(x)
model = Model(inputs=input_layer, outputs=output_layer)
model.summary()



## === cell 18
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)
model.fit(
    train_seq,
    train_labels,
    batch_size=128,
    validation_split=0.2,
    epochs=5,
    callbacks=[save_model_callback, earlystop],
    verbose=2,
)



## === cell 19
y_pred = model.predict(test_seq, batch_size=128)



## === cell 20
submission = pd.DataFrame(
    y_pred,
    columns=[
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ],
)
submission.insert(0, "id", test["id"])
submission.head()



## === cell 21
submission_path = "/kaggle/working/sample_submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
