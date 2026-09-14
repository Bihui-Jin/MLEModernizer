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
Predict the sentiment of phrases.

## Metric
Classification accuracy.

## Submission Format
For each phrase in the test set, predict a label for the sentiment. Your submission should have a header and look like the following:

```
PhraseId,Sentiment
156061,2
156062,2
156063,2
...
```

## Dataset
The dataset is comprised of tab-separated files with phrases. Each phrase has a PhraseId. Each sentence has a SentenceId.

The sentiment labels are:

0 - negative

1 - somewhat negative

2 - neutral

3 - somewhat positive

4 - positive

# 2. Python version

3.7

# 3. Installed packages

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
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0
tqdm==4.67.1
wordcloud==1.9.4

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        input/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        working/
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
```

-> data/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> data/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> (stopped after 10 files for performance)

# 5. Target score

0.6422

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator#
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, Flatten
from keras.optimizers import Adam
from keras.models import Model
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from nltk.corpus import stopwords
from keras.utils import to_categorical

import os
print(os.listdir("../input"))


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv("../input/train.tsv", sep="\t")
test_df= pd.read_csv("../input/test.tsv", sep="\t")


## === cell 2
stop_words = set(stopwords.words('english'))

for df in [train_df, test_df]:
    df['words_length'] = df['Phrase'].apply(lambda x: len(x))


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/63938435.py in <cell line: 0>()
----> 1 stop_words = set(stopwords.words('english'))
      2 
      3 for df in [train_df, test_df]:
      4     df['words_length'] = df['Phrase'].apply(lambda x: len(x))

NameError: name 'stopwords' is not defined

## === cell 3
train_df['Phrase'][(train_df['words_length']==1) & (train_df['Sentiment']==0) ] = "bad" 
train_df['Phrase'][(train_df['words_length']==1)& (train_df['Sentiment']==1 )] = "bad" 
train_df['Phrase'][(train_df['words_length']==1) & (train_df['Sentiment']==2) ] = "seem"
train_df['Phrase'][(train_df['words_length']==2) & (train_df['Sentiment'] >=2) ] = "seem"
test_df['Phrase'][(test_df['words_length']==1)]  = "seem"
test_df['Phrase'][(test_df['words_length']==2) & ((test_df['Phrase'] != "no") | (test_df['Phrase'] != "No"))] = "seem"
for df in [train_df, test_df]:
    df['words_length'] = df['Phrase'].apply(lambda x: len(x))


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'words_length'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1813118648.py in <cell line: 0>()
----> 1 train_df['Phrase'][(train_df['words_length']==1) & (train_df['Sentiment']==0) ] = "bad"
      2 train_df['Phrase'][(train_df['words_length']==1)& (train_df['Sentiment']==1 )] = "bad"
      3 train_df['Phrase'][(train_df['words_length']==1) & (train_df['Sentiment']==2) ] = "seem"
      4 train_df['Phrase'][(train_df['words_length']==2) & (train_df['Sentiment'] >=2) ] = "seem"
      5 test_df['Phrase'][(test_df['words_length']==1)]  = "seem"

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'words_length'

## === cell 4
train_df, val_df = train_test_split(train_df, test_size = 0.1, random_state= 144)
print(train_df.shape)
print(val_df.shape)


## === cell 5
embed_size = 100 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 50 # max number of words in a question to use

train_X = train_df["Phrase"].fillna("##").values
val_X = val_df["Phrase"].fillna("##").values
test_X = test_df['Phrase'].fillna("##").values
print("before tokenization")
print(train_X.shape)
print(val_X.shape)
print(test_X.shape)

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))

train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

print("after tokenization")
print(len(train_X))
print(len(val_X))
print(len(test_X))


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/236331963.py in <cell line: 0>()
     14 
     15 ## Tokenize the sentences
---> 16 tokenizer = Tokenizer(num_words=max_features)
     17 tokenizer.fit_on_texts(list(train_X))
     18 

NameError: name 'Tokenizer' is not defined

## === cell 6
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

print("after padding")
print(len(train_X))
print(len(val_X))
print(len(test_X))

train_y = train_df['Sentiment'].values
val_y = val_df['Sentiment'].values


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/570052104.py in <cell line: 0>()
      1 ## Pad the sentences
----> 2 train_X = pad_sequences(train_X, maxlen=maxlen)
      3 val_X = pad_sequences(val_X, maxlen=maxlen)
      4 test_X = pad_sequences(test_X, maxlen=maxlen)
      5 

NameError: name 'pad_sequences' is not defined

## === cell 7
np.random.seed(2018)
trn_idx = np.random.permutation(len(train_X))
val_idx = np.random.permutation(len(val_X))

train_y = train_df['Sentiment'].values
val_y = val_df['Sentiment'].values

train_X = train_X[trn_idx]
val_X = val_X[val_idx]
train_y = train_y[trn_idx]
val_y = val_y[val_idx]


## === cell 8
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)

x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
x = Flatten()(x)
x = Dense(64, activation="relu")(x)
x = Dense(5, activation="softmax")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2040320453.py in <cell line: 0>()
----> 1 inp = Input(shape=(maxlen,))
      2 x = Embedding(max_features, embed_size)(inp)
      3 
      4 x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
      5 x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)

NameError: name 'Input' is not defined

## === cell 9
model.summary()


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3035046171.py in <cell line: 0>()
----> 1 model.summary()

NameError: name 'model' is not defined

## === cell 10
train_y = to_categorical(train_y, num_classes=5)
val_y = to_categorical(val_y, num_classes=5)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4161609915.py in <cell line: 0>()
----> 1 train_y = to_categorical(train_y, num_classes=5)
      2 val_y = to_categorical(val_y, num_classes=5)

NameError: name 'to_categorical' is not defined

## === cell 11
model.fit(train_X, train_y, batch_size=512, epochs=4, validation_data=(val_X, val_y))


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2972768257.py in <cell line: 0>()
      1 # ## Train the model
----> 2 model.fit(train_X, train_y, batch_size=512, epochs=4, validation_data=(val_X, val_y))

NameError: name 'model' is not defined

## === cell 12
pred_glove_val_y = model.predict([test_X], batch_size=1024, verbose=1)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4233944847.py in <cell line: 0>()
----> 1 pred_glove_val_y = model.predict([test_X], batch_size=1024, verbose=1)

NameError: name 'model' is not defined

## === cell 13
predictions = []
for i in range(len(pred_glove_val_y)):
    predictions.append(np.argmax(pred_glove_val_y[i]))


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2071667880.py in <cell line: 0>()
      1 predictions = []
----> 2 for i in range(len(pred_glove_val_y)):
      3     predictions.append(np.argmax(pred_glove_val_y[i]))

NameError: name 'pred_glove_val_y' is not defined

## === cell 14
len(predictions)


## === cell 15
submission_df = pd.DataFrame()
submission_df['PhraseId'] = test_df['PhraseId']
submission_df['Sentiment'] = predictions 
submission_df.to_csv("submission.csv", index=False)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2227366101.py in <cell line: 0>()
      1 submission_df = pd.DataFrame()
      2 submission_df['PhraseId'] = test_df['PhraseId']
----> 3 submission_df['Sentiment'] = predictions
      4 submission_df.to_csv("submission.csv", index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5264 
   5265         if is_list_like(value):
-> 5266             com.require_length_match(value, self.index)
   5267         arr = sanitize_array(value, self.index, copy=True, allow_2d=True)
   5268         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/common.py in require_length_match(data, index)
    571     """
    572     if len(data) != len(index):
--> 573         raise ValueError(
    574             "Length of values "
    575             f"({len(data)}) "

ValueError: Length of values (0) does not match length of index (46818)

## === cell 16
submission_df.head()
