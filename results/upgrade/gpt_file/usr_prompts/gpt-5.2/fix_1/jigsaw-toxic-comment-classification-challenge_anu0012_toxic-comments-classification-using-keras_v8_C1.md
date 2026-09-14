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

3.6

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0
wordcloud==1.9.4
xgboost==2.0.3

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

0.08791

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
import nltk
from nltk import word_tokenize, ngrams
from nltk.classify import SklearnClassifier
from wordcloud import WordCloud,STOPWORDS
import xgboost as xgb
np.random.seed(25)


## === cell 1
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


## === cell 2
train.head()


## === cell 3
train['comment_text'][0]


## === cell 4
train['comment_text'][1]


## === cell 5
train.isnull().sum(axis=0)


## === cell 6
types = ['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate']

train[types].describe()


## === cell 7
count_list = []
for i in types:
    count_list.append(train[i].sum())
    
sns.barplot(x=types, y=count_list)


## === cell 8
types = ['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate']

sampled_train1 = train[(train['toxic'] ==0) & (train['severe_toxic'] ==0) & (train['obscene'] ==0)
                       & (train['threat'] ==0) & (train['insult'] ==0) & (train['identity_hate'] ==0)]

sampled_train2 = train[(train['toxic'] !=0) | (train['severe_toxic'] !=0) | (train['obscene'] !=0) 
                | (train['threat'] !=0) | (train['insult'] !=0) | (train['identity_hate'] !=0)]


## === cell 9
sampled_train2.head()


## === cell 10
sampled_train = sampled_train2.append(sampled_train1[:16223])
sampled_train = sampled_train.sample(frac=1)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1109986081.py in <cell line: 0>()
----> 1 sampled_train = sampled_train2.append(sampled_train1[:16223])
      2 sampled_train = sampled_train.sample(frac=1)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

## === cell 11
sampled_train.head()


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1927393271.py in <cell line: 0>()
----> 1 sampled_train.head()

NameError: name 'sampled_train' is not defined

## === cell 12
import string
import itertools 
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation

stop_words = set(stopwords.words('english'))

def cleanData(text, lowercase = False, remove_stops = False, stemming = False, lemmatization = False):
    txt = str(text)
    
    txt = txt.replace("isn't", "is not")
    txt = txt.replace("aren't", "are not")
    txt = txt.replace("ain't", "am not")
    txt = txt.replace("won't", "will not")
    txt = txt.replace("didn't", "did not")
    txt = txt.replace("shan't", "shall not")
    txt = txt.replace("haven't", "have not")
    txt = txt.replace("hadn't", "had not")
    txt = txt.replace("hasn't", "has not")
    txt = txt.replace("don't", "do not")
    txt = txt.replace("wasn't", "was not")
    txt = txt.replace("weren't", "were not")
    txt = txt.replace("doesn't", "does not")
    txt = txt.replace("'s", " is")
    txt = txt.replace("'re", " are")
    txt = txt.replace("'m", " am")
    txt = txt.replace("'d", " would")
    txt = txt.replace("'ll", " will")
    txt = txt.replace("--th", " ")
    
    txt = re.sub(r"alot", "a lot", txt)
    txt = re.sub(r"what's", "", txt)
    txt = re.sub(r"What's", "", txt)
    txt = re.sub(r"\'s", " ", txt)
    txt = txt.replace("pic", "picture")
    txt = re.sub(r"\'ve", " have ", txt)
    txt = re.sub(r"can't", "cannot ", txt)
    txt = re.sub(r"n't", " not ", txt)
    txt = re.sub(r"I'm", "I am", txt)
    txt = re.sub(r" m ", " am ", txt)
    txt = re.sub(r"\'re", " are ", txt)
    txt = re.sub(r"\'d", " would ", txt)
    txt = re.sub(r"\'ll", " will ", txt)
    txt = re.sub(r"60k", " 60000 ", txt)
    txt = re.sub(r" e g ", " eg ", txt)
    txt = re.sub(r" b g ", " bg ", txt)
    txt = re.sub(r"\0s", "0", txt)
    txt = re.sub(r" 9 11 ", "911", txt)
    txt = re.sub(r"e-mail", "email", txt)
    txt = re.sub(r"\s{2,}", " ", txt)
    txt = re.sub(r"quikly", "quickly", txt)
    txt = re.sub(r"imrovement", "improvement", txt)
    txt = re.sub(r"intially", "initially", txt)
    txt = re.sub(r"quora", "Quora", txt)
    txt = re.sub(r" dms ", "direct messages ", txt)  
    txt = re.sub(r"demonitization", "demonetization", txt) 
    txt = re.sub(r"actived", "active", txt)
    txt = re.sub(r"kms", " kilometers ", txt)
    txt = re.sub(r"KMs", " kilometers ", txt)
    txt = re.sub(r" cs ", " computer science ", txt) 
    txt = re.sub(r" upvotes ", " up votes ", txt)
    txt = re.sub(r" iPhone ", " phone ", txt)
    txt = re.sub(r"\0rs ", " rs ", txt) 
    txt = re.sub(r"calender", "calendar", txt)
    txt = re.sub(r"ios", "operating system", txt)
    txt = re.sub(r"gps", "GPS", txt)
    txt = re.sub(r"gst", "GST", txt)
    txt = re.sub(r"programing", "programming", txt)
    txt = re.sub(r"bestfriend", "best friend", txt)
    txt = re.sub(r"dna", "DNA", txt)
    txt = re.sub(r"III", "3", txt) 
    txt = re.sub(r"the US", "America", txt)
    txt = re.sub(r"Astrology", "astrology", txt)
    txt = re.sub(r"Method", "method", txt)
    txt = re.sub(r"Find", "find", txt) 
    txt = re.sub(r"banglore", "Banglore", txt)
    txt = re.sub(r" J K ", " JK ", txt)
    txt = re.sub(r"comfy", "comfortable", txt)
    txt = re.sub(r"colour", "color", txt)
    txt = re.sub(r"travellers", "travelers", txt)
    
    txt = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', txt, flags=re.MULTILINE)
    txt = re.sub(r'[\w\.-]+@[\w\.-]+', ' ', txt, flags=re.MULTILINE)
    
    txt = ''.join(''.join(s)[:2] for _, s in itertools.groupby(txt))
    
    txt = ''.join([c for c in text if c not in punctuation])
    
    txt = re.sub(r'[^A-Za-z\s]',r' ',txt)
    txt = re.sub(r'\n',r' ',txt)
    
    if lowercase:
        txt = " ".join([w.lower() for w in txt.split()])
        
    if remove_stops:
        txt = " ".join([w for w in txt.split() if w not in stop_words])
        
    if stemming:
        st = PorterStemmer()
        txt = " ".join([st.stem(w) for w in txt.split()])
    
    if lemmatization:
        wordnet_lemmatizer = WordNetLemmatizer()
        txt = " ".join([wordnet_lemmatizer.lemmatize(w, pos='v') for w in txt.split()])

    return txt


## === cell 13
sampled_train['comment_text'] = sampled_train['comment_text'].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, stemming=True, lemmatization = True))
test['comment_text'] = test['comment_text'].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, stemming=True, lemmatization = True))


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1492827422.py in <cell line: 0>()
      1 # clean comments
----> 2 sampled_train['comment_text'] = sampled_train['comment_text'].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, stemming=True, lemmatization = True))
      3 test['comment_text'] = test['comment_text'].map(lambda x: cleanData(x, lowercase=True, remove_stops=True, stemming=True, lemmatization = True))

NameError: name 'sampled_train' is not defined

## === cell 15
train.columns


## === cell 16
np.random.seed(25)
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, merge, LSTM, Lambda, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers import Convolution1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.layers.merge import concatenate
from keras.layers.core import Dense, Activation, Dropout
import codecs


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 17
MAX_SEQUENCE_LENGTH = 256
MAX_NB_WORDS = 200000 #200000


## === cell 18
tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(sampled_train['comment_text'])

sequences = tokenizer.texts_to_sequences(sampled_train['comment_text'])
test_sequences = tokenizer.texts_to_sequences(test['comment_text'])

train_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of train data tensor:', train_data.shape)

test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

nb_words = np.max(train_data) + 1


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3910870208.py in <cell line: 0>()
----> 1 tokenizer = Tokenizer(lower=False, filters='')
      2 tokenizer.fit_on_texts(sampled_train['comment_text'])
      3 
      4 sequences = tokenizer.texts_to_sequences(sampled_train['comment_text'])
      5 test_sequences = tokenizer.texts_to_sequences(test['comment_text'])

NameError: name 'Tokenizer' is not defined

## === cell 19
from keras.layers.recurrent import LSTM, GRU
model = Sequential()
model.add(Embedding(nb_words,20,input_length=MAX_SEQUENCE_LENGTH))
model.add(GlobalAveragePooling1D())
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2213588016.py in <cell line: 0>()
----> 1 from keras.layers.recurrent import LSTM, GRU
      2 model = Sequential()
      3 model.add(Embedding(nb_words,20,input_length=MAX_SEQUENCE_LENGTH))
      4 model.add(GlobalAveragePooling1D())
      5 model.add(Dense(2, activation='softmax'))

ModuleNotFoundError: No module named 'keras.layers.recurrent'

## === cell 20
li = []
labels = ['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate']
for label in labels:
    label = np.array(sampled_train[label])
    model.fit(train_data, to_categorical(label), validation_split=0.2, nb_epoch=2, batch_size=128)
    li.append(model.predict(test_data)[:,0])


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/190804819.py in <cell line: 0>()
      3        'insult', 'identity_hate']
      4 for label in labels:
----> 5     label = np.array(sampled_train[label])
      6     model.fit(train_data, to_categorical(label), validation_split=0.2, nb_epoch=2, batch_size=128)
      7     li.append(model.predict(test_data)[:,0])

NameError: name 'sampled_train' is not defined

## === cell 22
count=0
result = pd.DataFrame()
result['id'] = test['id']

for i in types:
    result[i] = li[count]
    count = count + 1

result.to_csv("result.csv", index=False)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/1749741978.py in <cell line: 0>()
      4 
      5 for i in types:
----> 6     result[i] = li[count]
      7     count = count + 1
      8 

IndexError: list index out of range

## === cell 23
result.head()
