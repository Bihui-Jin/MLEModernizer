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

3.7

# 2. Installed packages

gensim==4.4.0
geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        input/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        working/
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
```

-> data/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/spooky-author-identification/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> data/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> input/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import re
import gensim 
from gensim.models import Word2Vec


## === cell 1
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
print(train_df.shape, test_df.shape)


## === cell 2
def clean_text(text):
    """
    Convert all to lowercase and remove punctuations
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # remove everything that isn't word or space
    text = re.sub(r'\_', '', text)      # remove underscore
    return text


## === cell 3
train_df['text'] = train_df['text'].map(lambda x: clean_text(x))
train_df['text'] = train_df['text'].map(lambda x: x.strip().split())
train_df.head()


## === cell 4
test_df['text'] = test_df['text'].map(lambda x: clean_text(x))
test_df['text'] = test_df['text'].map(lambda x: x.strip().split())
test_df.head()


## === cell 5
data = []  
for i in range(len(train_df)):
    data.append(train_df['text'][i])
for j in range(len(test_df)):
    data.append(test_df['text'][j])


## === cell 6
print(len(data))


## === cell 7
embedding = gensim.models.Word2Vec(data, vector_size=50, window=10, min_count=1, sg=0)


## === cell 8
print(embedding)


## === cell 9
embedding.train(data,total_examples=len(data),epochs=30)


## === cell 10
words = list(embedding.wv.index_to_key)
print(len(words))


## === cell 11
print(embedding.wv["capered"])


## === cell 12
embedding.wv.most_similar("dark", topn=5)


## === cell 13
embedding.wv.most_similar("shocked", topn=5)


## === cell 14
embedding.wv.most_similar("sprang", topn=5)


## === cell 15
embedding.wv.most_similar("pride", topn=5)


## === cell 16
train_df['author'] = pd.Categorical(train_df['author'])
df_Dummies = pd.get_dummies(train_df['author'], prefix='author')
train_df = pd.concat([train_df, df_Dummies], axis=1)
train_df.head()


## === cell 17
X = train_df['text'].str[:50]
Y = train_df[['author_EAP', 'author_HPL', 'author_MWS']].values
print(X.shape, X[0], Y.shape, Y[0])


## === cell 18
X_test = test_df['text'].str[:50]
print(X_test.shape, X_test[0])


## === cell 19
def text_to_avg(text):
    """
    Given a list of words, extract the respective GloVe representations
    and average the values into a single vector encoding the text meaning.
    """
    avg = np.zeros((50,))
    for w in text:
        avg += embedding[w]
    avg = avg/len(text)
    return avg


## === cell 20
def text_to_avg(text):
    """
    Given a list of words, extract the respective Word2Vec representations
    and average the values into a single vector encoding the text meaning.
    """
    avg = np.zeros((50,))
    for w in text:
        avg += embedding.wv[w]
    avg = avg / len(text)
    return avg


X_avg = np.zeros((X.shape[0], 50))  # initialize X_avg
for i in range(X.shape[0]):
    X_avg[i] = text_to_avg(X[i])


## === cell 21
print(X_avg.shape)
print(X_avg[0])


## === cell 22
X_test_avg = np.zeros((X_test.shape[0], 50)) # initialize X_test_avg
for i in range(X_test.shape[0]):
    X_test_avg[i] = text_to_avg(X_test[i])


## === cell 23
print(X_test_avg.shape)
print(X_test_avg[0])


## === cell 24
from sklearn.model_selection import train_test_split
X_train, X_dev, Y_train, Y_dev = train_test_split(X_avg, Y, test_size=0.2, random_state=123)
print(X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape)


## === cell 25
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

from tf_keras import models
from tf_keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(50,)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(3, activation="softmax"))

model.summary()


## --- ERROR in cell 25, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;31mAttributeError[0m: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 26
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
