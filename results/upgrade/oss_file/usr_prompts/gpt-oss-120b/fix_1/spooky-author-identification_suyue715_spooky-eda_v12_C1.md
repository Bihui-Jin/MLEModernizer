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
Given some text, predict the author.

## Metric
Multi-class logarithmic loss. 

The submitted probabilities for a given sentences are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum).

In order to avoid the extremes of the log function, predicted probabilities are replaced with \\(max(min(p,1-10^{-15}),10^{-15})\\).

## Submission Format
You must submit a csv file with the id, and a probability for each of the three classes. The order of the rows does not matter. The file must have a header and should look like the following:

```
id,EAP,HPL,MWS
id07943,0.33,0.33,0.33
...
```

## Dataset 
### File descriptions
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Data fields
- **id** - a unique identifier for each sentence
- **text** - some text written by one of the authors
- **author** - the author of the sentence (EAP: Edgar Allan Poe, HPL: HP Lovecraft; MWS: Mary Wollstonecraft Shelley)

# 2. Python version

3.6

# 3. Installed packages

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
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
textblob==0.19.0
tf_keras==2.18.0
wordcloud==1.9.4

# 4. Data file paths

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

# 5. Target score

0.49321

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from textblob import TextBlob
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))



## === cell 1
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train = pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")


## === cell 3
train.head()


## === cell 4
train.author.unique()


## === cell 5
plt.hist(train.author)
plt.title("Frequency of Authors Occurence",fontsize=15)
plt.xticks(np.arange(3),(['Edgar Allen Poe', 'Mary Shelley', 'HP Lovecraft']))
x_coor = [0,1.8,1]
for idx, label in enumerate(train.author.value_counts().index):
    cnt = train.author.value_counts()[idx]
    plt.text(x_coor[idx], cnt, str(cnt), color='black', fontweight='bold')


## === cell 6
train_qs = pd.Series(train['text'].tolist()).astype(str)
dist_train = train_qs.apply(len)
plt.figure(figsize=(15, 10))
plt.hist(dist_train, bins=200, range=[0, 200], normed=True,alpha=.7)
plt.title('Normalized histogram of character count in text', fontsize=15)
plt.legend()
plt.xlabel('Number of characters', fontsize=15)
plt.ylabel('Probability', fontsize=15)


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1397883403.py in <cell line: 0>()
      2 dist_train = train_qs.apply(len)
      3 plt.figure(figsize=(15, 10))
----> 4 plt.hist(dist_train, bins=200, range=[0, 200], normed=True,alpha=.7)
      5 plt.title('Normalized histogram of character count in text', fontsize=15)
      6 plt.legend()

/usr/local/lib/python3.11/dist-packages/matplotlib/pyplot.py in hist(x, bins, range, density, weights, cumulative, bottom, histtype, align, orientation, rwidth, log, color, label, stacked, data, **kwargs)
   2643         orientation='vertical', rwidth=None, log=False, color=None,
   2644         label=None, stacked=False, *, data=None, **kwargs):
-> 2645     return gca().hist(
   2646         x, bins=bins, range=range, density=density, weights=weights,
   2647         cumulative=cumulative, bottom=bottom, histtype=histtype,

/usr/local/lib/python3.11/dist-packages/matplotlib/__init__.py in inner(ax, data, *args, **kwargs)
   1444     def inner(ax, *args, data=None, **kwargs):
   1445         if data is None:
-> 1446             return func(ax, *map(sanitize_sequence, args), **kwargs)
   1447 
   1448         bound = new_sig.bind(ax, *args, **kwargs)

/usr/local/lib/python3.11/dist-packages/matplotlib/axes/_axes.py in hist(self, x, bins, range, density, weights, cumulative, bottom, histtype, align, orientation, rwidth, log, color, label, stacked, **kwargs)
   6942             if patch:
   6943                 p = patch[0]
-> 6944                 p._internal_update(kwargs)
   6945                 if lbl is not None:
   6946                     p.set_label(lbl)

/usr/local/lib/python3.11/dist-packages/matplotlib/artist.py in _internal_update(self, kwargs)
   1221         The lack of prenormalization is to maintain backcompatibility.
   1222         """
-> 1223         return self._update_props(
   1224             kwargs, "{cls.__name__}.set() got an unexpected keyword argument "
   1225             "{prop_name!r}")

/usr/local/lib/python3.11/dist-packages/matplotlib/artist.py in _update_props(self, props, errfmt)
   1195                     func = getattr(self, f"set_{k}", None)
   1196                     if not callable(func):
-> 1197                         raise AttributeError(
   1198                             errfmt.format(cls=type(self), prop_name=k))
   1199                     ret.append(func(v))

AttributeError: Rectangle.set() got an unexpected keyword argument 'normed'

## === cell 7
print('mean num',dist_train.mean(),'\n','std num', dist_train.std(),'\n',
      'min num',dist_train.min(),'\n','max num', dist_train.max())


## === cell 8
dist_train = train_qs.apply(lambda x: len(x.split(' ')))
plt.figure(figsize=(15, 10))
plt.hist(dist_train, bins=50, range=[0, 50], normed=True,alpha=.7)
plt.title('Normalised histogram of word count in texts', fontsize=15)
plt.legend()
plt.xlabel('Number of words', fontsize=15)
plt.ylabel('Probability', fontsize=15)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1314291657.py in <cell line: 0>()
      1 dist_train = train_qs.apply(lambda x: len(x.split(' ')))
      2 plt.figure(figsize=(15, 10))
----> 3 plt.hist(dist_train, bins=50, range=[0, 50], normed=True,alpha=.7)
      4 plt.title('Normalised histogram of word count in texts', fontsize=15)
      5 plt.legend()

/usr/local/lib/python3.11/dist-packages/matplotlib/pyplot.py in hist(x, bins, range, density, weights, cumulative, bottom, histtype, align, orientation, rwidth, log, color, label, stacked, data, **kwargs)
   2643         orientation='vertical', rwidth=None, log=False, color=None,
   2644         label=None, stacked=False, *, data=None, **kwargs):
-> 2645     return gca().hist(
   2646         x, bins=bins, range=range, density=density, weights=weights,
   2647         cumulative=cumulative, bottom=bottom, histtype=histtype,

/usr/local/lib/python3.11/dist-packages/matplotlib/__init__.py in inner(ax, data, *args, **kwargs)
   1444     def inner(ax, *args, data=None, **kwargs):
   1445         if data is None:
-> 1446             return func(ax, *map(sanitize_sequence, args), **kwargs)
   1447 
   1448         bound = new_sig.bind(ax, *args, **kwargs)

/usr/local/lib/python3.11/dist-packages/matplotlib/axes/_axes.py in hist(self, x, bins, range, density, weights, cumulative, bottom, histtype, align, orientation, rwidth, log, color, label, stacked, **kwargs)
   6942             if patch:
   6943                 p = patch[0]
-> 6944                 p._internal_update(kwargs)
   6945                 if lbl is not None:
   6946                     p.set_label(lbl)

/usr/local/lib/python3.11/dist-packages/matplotlib/artist.py in _internal_update(self, kwargs)
   1221         The lack of prenormalization is to maintain backcompatibility.
   1222         """
-> 1223         return self._update_props(
   1224             kwargs, "{cls.__name__}.set() got an unexpected keyword argument "
   1225             "{prop_name!r}")

/usr/local/lib/python3.11/dist-packages/matplotlib/artist.py in _update_props(self, props, errfmt)
   1195                     func = getattr(self, f"set_{k}", None)
   1196                     if not callable(func):
-> 1197                         raise AttributeError(
   1198                             errfmt.format(cls=type(self), prop_name=k))
   1199                     ret.append(func(v))

AttributeError: Rectangle.set() got an unexpected keyword argument 'normed'

## === cell 9
print('mean num',dist_train.mean(),'\n','std num', dist_train.std(),'\n',
      'min num',dist_train.min(),'\n','max num', dist_train.max())


## === cell 10
from wordcloud import WordCloud
from stop_words import get_stop_words

stop_words = get_stop_words('en')
stop_words = get_stop_words('english')
cloud = WordCloud(width=1440, height=1080,stopwords=stop_words).generate(" ".join(train.text.astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/705393061.py in <cell line: 0>()
      1 from wordcloud import WordCloud
----> 2 from stop_words import get_stop_words
      3 
      4 stop_words = get_stop_words('en')
      5 stop_words = get_stop_words('english')

ModuleNotFoundError: No module named 'stop_words'

## === cell 12
eap = train[train.author=="EAP"]["text"].values
hpl = train[train.author=="HPL"]["text"].values
mws = train[train.author=="MWS"]["text"].values


## === cell 13
cloud = WordCloud(width=1440, height=1080,stopwords=stop_words).generate(" ".join(eap.astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1604179193.py in <cell line: 0>()
----> 1 cloud = WordCloud(width=1440, height=1080,stopwords=stop_words).generate(" ".join(eap.astype(str)))
      2 plt.figure(figsize=(20, 15))
      3 plt.imshow(cloud)
      4 plt.axis('off')

NameError: name 'stop_words' is not defined

## === cell 14
cloud = WordCloud(width=1440, height=1080,stopwords=stop_words).generate(" ".join(hpl.astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4108393347.py in <cell line: 0>()
----> 1 cloud = WordCloud(width=1440, height=1080,stopwords=stop_words).generate(" ".join(hpl.astype(str)))
      2 plt.figure(figsize=(20, 15))
      3 plt.imshow(cloud)
      4 plt.axis('off')

NameError: name 'stop_words' is not defined

## === cell 15
cloud = WordCloud(width=1440, height=1080,stopwords=stop_words).generate(" ".join(mws.astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1631560076.py in <cell line: 0>()
----> 1 cloud = WordCloud(width=1440, height=1080,stopwords=stop_words).generate(" ".join(mws.astype(str)))
      2 plt.figure(figsize=(20, 15))
      3 plt.imshow(cloud)
      4 plt.axis('off')

NameError: name 'stop_words' is not defined

## === cell 17
train['author_num']=train['author'].apply({'EAP':0,  'HPL':1,'MWS':2}.get)


## === cell 18
train.head()


## === cell 19
raw_text_train=train['text'].values
raw_text_test=test['text'].values
author_train=train['author_num'].values
num_labels = len(np.unique(author_train))


## === cell 20
num_labels


## === cell 21
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
stemmer = SnowballStemmer('english')


## === cell 22
print ("pre-processing train docs...")
processed_train = []
for doc in raw_text_train:
    tokens = word_tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    stemmed = [stemmer.stem(word) for word in filtered]
    processed_train.append(stemmed)


## === cell 23
print ("pre-processing test docs...")
processed_test = []
for doc in raw_text_test:
    tokens = word_tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    stemmed = [stemmer.stem(word) for word in filtered]
    processed_test.append(stemmed)


## === cell 24
processed_docs_all = np.concatenate((processed_train, processed_test), axis=0)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3558266745.py in <cell line: 0>()
----> 1 processed_docs_all = np.concatenate((processed_train, processed_test), axis=0)

ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (17621,) + inhomogeneous part.

## === cell 25
from gensim import corpora
dictionary = corpora.Dictionary(processed_docs_all)
dictionary_size = len(dictionary.keys())
print ("dictionary size: ", dictionary_size )


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1548303757.py in <cell line: 0>()
      1 from gensim import corpora
----> 2 dictionary = corpora.Dictionary(processed_docs_all)
      3 dictionary_size = len(dictionary.keys())
      4 print ("dictionary size: ", dictionary_size )
      5     #dictionary.save('dictionary.dict')

NameError: name 'processed_docs_all' is not defined

## === cell 26
print ("converting to token ids...")
word_id_train, word_id_len = [], []
for doc in processed_train:
    word_ids = [dictionary.token2id[word] for word in doc]
    word_id_train.append(word_ids)
    word_id_len.append(len(word_ids))


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1250860729.py in <cell line: 0>()
      2 word_id_train, word_id_len = [], []
      3 for doc in processed_train:
----> 4     word_ids = [dictionary.token2id[word] for word in doc]
      5     word_id_train.append(word_ids)
      6     word_id_len.append(len(word_ids))

/tmp/ipykernel_11/1250860729.py in <listcomp>(.0)
      2 word_id_train, word_id_len = [], []
      3 for doc in processed_train:
----> 4     word_ids = [dictionary.token2id[word] for word in doc]
      5     word_id_train.append(word_ids)
      6     word_id_len.append(len(word_ids))

NameError: name 'dictionary' is not defined

## === cell 27
word_id_test, word_ids = [], []
for doc in processed_test:
    word_ids = [dictionary.token2id[word] for word in doc]
    word_id_test.append(word_ids)
    word_id_len.append(len(word_ids))


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3468212011.py in <cell line: 0>()
      1 word_id_test, word_ids = [], []
      2 for doc in processed_test:
----> 3     word_ids = [dictionary.token2id[word] for word in doc]
      4     word_id_test.append(word_ids)
      5     word_id_len.append(len(word_ids))

/tmp/ipykernel_11/3468212011.py in <listcomp>(.0)
      1 word_id_test, word_ids = [], []
      2 for doc in processed_test:
----> 3     word_ids = [dictionary.token2id[word] for word in doc]
      4     word_id_test.append(word_ids)
      5     word_id_len.append(len(word_ids))

NameError: name 'dictionary' is not defined

## === cell 28
seq_len = np.round((np.mean(word_id_len) + 2*np.std(word_id_len))).astype(int)


## === cell 29
word_id_train = sequence.pad_sequences(np.array(word_id_train), maxlen=seq_len)
word_id_test = sequence.pad_sequences(np.array(word_id_test), maxlen=seq_len)
y_train_enc = np_utils.to_categorical(author_train,num_labels)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3029889192.py in <cell line: 0>()
      1 #pad sequences
----> 2 word_id_train = sequence.pad_sequences(np.array(word_id_train), maxlen=seq_len)
      3 word_id_test = sequence.pad_sequences(np.array(word_id_test), maxlen=seq_len)
      4 y_train_enc = np_utils.to_categorical(author_train,num_labels)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/sequence_utils.py in pad_sequences(sequences, maxlen, dtype, padding, truncating, value)
    111         )
    112 
--> 113     x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    114     for idx, s in enumerate(sequences):
    115         if not len(s):

/usr/local/lib/python3.11/dist-packages/numpy/core/numeric.py in full(shape, fill_value, dtype, order, like)
    327         fill_value = asarray(fill_value)
    328         dtype = fill_value.dtype
--> 329     a = empty(shape, dtype, order)
    330     multiarray.copyto(a, fill_value, casting='unsafe')
    331     return a

ValueError: negative dimensions are not allowed

## === cell 30
print ("fitting LSTM ...")
model = Sequential()
model.add(Embedding(dictionary_size, 128, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/475704217.py in <cell line: 0>()
      1 print ("fitting LSTM ...")
----> 2 model = Sequential()
      3 model.add(Embedding(dictionary_size, 128, dropout=0.2))
      4 model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
      5 model.add(Dense(num_labels))

NameError: name 'Sequential' is not defined

## === cell 31
model.fit(word_id_train, y_train_enc, nb_epoch=1, batch_size=250, verbose=1)
test_pred = model.predict_classes(word_id_test)


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/750012491.py in <cell line: 0>()
----> 1 model.fit(word_id_train, y_train_enc, nb_epoch=1, batch_size=250, verbose=1)
      2 test_pred = model.predict_classes(word_id_test)

NameError: name 'model' is not defined

## === cell 32
test_pred = model.predict_proba(word_id_test)


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1837895373.py in <cell line: 0>()
----> 1 test_pred = model.predict_proba(word_id_test)

NameError: name 'model' is not defined

## === cell 34
prob=pd.DataFrame(test_pred,columns=['EAP','HPL','MWS'])


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3556512767.py in <cell line: 0>()
----> 1 prob=pd.DataFrame(test_pred,columns=['EAP','HPL','MWS'])

NameError: name 'test_pred' is not defined

## === cell 35
prob.shape


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3415938670.py in <cell line: 0>()
----> 1 prob.shape

NameError: name 'prob' is not defined

## === cell 36
test.head()


## === cell 37
submit1=pd.concat([test, prob], axis=1)
del submit1['text']


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3627306297.py in <cell line: 0>()
----> 1 submit1=pd.concat([test, prob], axis=1)
      2 del submit1['text']

NameError: name 'prob' is not defined

## === cell 38
submit1.to_csv('./lstm_sentiment.csv', index=False, header=True)


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2997817384.py in <cell line: 0>()
----> 1 submit1.to_csv('./lstm_sentiment.csv', index=False, header=True)

NameError: name 'submit1' is not defined

## === cell 39
submit1.head()


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/182758741.py in <cell line: 0>()
----> 1 submit1.head()

NameError: name 'submit1' is not defined

## === cell 40
pd.read_csv("../input/sample_submission.csv").head()
