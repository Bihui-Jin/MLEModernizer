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

# 5. Target score

0.96862

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
I will remove the stray markdown text that caused a syntax error, reorder the imports so the protobuf‑implementation flag is set **before** TensorFlow is imported (fixing the `MessageFactory` error), and clean up the final cell that wrote the CSV. No core modeling logic is changed; the script now runs end‑to‑end and produces a valid `sample_submission.csv` file.

```python


## --- ERROR in cell 0, traceback:
  File "/tmp/ipykernel_55/2889319479.py", line 1
    I will remove the stray markdown text that caused a syntax error, reorder the imports so the protobuf‑implementation flag is set **before** TensorFlow is imported (fixing the `MessageFactory` error), and clean up the final cell that wrote the CSV. No core modeling logic is changed; the script now runs end‑to‑end and produces a valid `sample_submission.csv` file.
                                                                                                         ^
SyntaxError: invalid character '‑' (U+2011)


## === cell 1
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Input, LSTM, GlobalMaxPool1D, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_auc_score
import re, warnings, nltk
warnings.simplefilter(action="ignore")

nltk.download('punkt')
nltk.download('stopwords')




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
!ls /kaggle/input/jigsaw-toxic-comment-classification-challenge/




## === cell 3
train_df = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
train_df.head()




## === cell 4
test_df = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
test_df.head()




## === cell 5
train_df['comment_text'][0]




## === cell 6
pd.DataFrame(train_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].sum(),
             columns=['Count'])




## === cell 7
temp = pd.DataFrame(train_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].sum(),
                    columns=['Count'])
sns.barplot(x=temp.index, y='Count', data=temp)




## === cell 8
from nltk import word_tokenize
train_df['tokenized_text'] = train_df.apply(lambda row: word_tokenize(row['comment_text']), axis=1)
lengths = [len(line) for line in train_df["tokenized_text"]]
train_df['comment_text'].iloc[np.argmax(lengths)]




## === cell 9
import plotly.express as px
px.histogram(lengths)




## === cell 10
from nltk.corpus import stopwords
def process_text(data):
    stop = set(stopwords.words('english'))
    data['processed_text'] = data.apply(lambda row: row['comment_text'].replace("\n"," "), axis=1)
    data['processed_text'] = data.apply(lambda row: re.sub(r'http://\S+|https://\S+', 'urls', row['processed_text']).lower(), axis=1)
    data['processed_text'] = data.apply(lambda row: re.sub(r'[^A-Za-z ]+', '', row['processed_text']).lower(), axis=1)
    data['processed_text'] = data['processed_text'].apply(lambda x: ' '.join([w for w in x.split() if w not in stop]))
    data['processed_text'] = data.apply(lambda row: re.sub(r' {2,}', ' ', row['processed_text']).strip(), axis=1)
    return data




## === cell 11
train = process_text(train_df.copy())
test = process_text(test_df.copy())




## === cell 12
train["processed_text"] = train.apply(lambda x: x["comment_text"] if len(x["processed_text"])==0 else x['processed_text'], axis=1)
test["processed_text"] = test.apply(lambda x: x["comment_text"] if len(x["processed_text"])==0 else x['processed_text'], axis=1)




## === cell 13
num_words = 30000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(train['processed_text'])
train_tokens = tokenizer.texts_to_sequences(train['processed_text'])
test_tokens = tokenizer.texts_to_sequences(test['processed_text'])
train_seq = pad_sequences(train_tokens, maxlen=300)
test_seq = pad_sequences(test_tokens, maxlen=300)




## === cell 14
print(train_seq.shape, test_seq.shape)




## === cell 15
train_labels = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values




## === cell 16
filepath = '/kaggle/working/best_model_{epoch:02d}.keras'
save_model_callback = ModelCheckpoint(filepath=filepath,
                                      monitor='val_auc',
                                      verbose=1,
                                      save_best_only=True,
                                      mode='max')




## === cell 17
earlystop = EarlyStopping(monitor='val_auc', min_delta=0.01, patience=2, verbose=1)




## === cell 18
tf.keras.backend.clear_session()
input_layer = Input(shape=(300,))
x = Embedding(num_words, 200)(input_layer)
x = LSTM(60, return_sequences=True)(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
output_layer = Dense(6, activation="sigmoid")(x)
model = Model(inputs=input_layer, outputs=output_layer)
model.summary()




## === cell 19
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
model.fit(train_seq,
          train_labels,
          batch_size=128,
          validation_split=0.2,
          epochs=5,
          callbacks=[save_model_callback, earlystop])




## === cell 20
sample_submission = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")




## === cell 21
df_test = pd.merge(test, sample_submission, on="id")




## === cell 22
y_pred = model.predict(test_seq, batch_size=128)




## === cell 23
df_test[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]] = y_pred
df_test.head()




## === cell 24
df_test.drop(["comment_text", "processed_text"], axis=1, inplace=True)
df_test.to_csv("sample_submission.csv", index=False)
```

## --- ERROR in cell 24, traceback:
  File "/tmp/ipykernel_55/1230938641.py", line 3
    ```
    ^
SyntaxError: invalid syntax
