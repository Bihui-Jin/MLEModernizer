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

3.9

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
sklearn-pandas==2.2.0
tf_keras==2.18.0

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

0.6234

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

import nltk
import os
import gc
from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D,Flatten,MaxPooling1D,GRU,SpatialDropout1D,Bidirectional
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_colwidth', -1)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv.zip',sep='\t', encoding='UTF')

print(train.shape)
train.head()


## === cell 2
test=pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv.zip',sep='\t')
print(test.shape)
test.head()


## === cell 3
test['Sentiment']=-999
test.head()


## === cell 4
sub=pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv')
sub.head()


## === cell 5
df=pd.concat([train,test],ignore_index=True)
print(df.shape)
df.tail()


## === cell 6
del train, test
gc.collect()


## === cell 7
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.stem import SnowballStemmer,WordNetLemmatizer
stemmer=SnowballStemmer('english')
lemma=WordNetLemmatizer()
from string import punctuation
import re


## === cell 8
def clean_review(review_col):
    review_corpus=[]
    for i in range(0,len(review_col)):
        review=str(review_col[i])
        review=re.sub('[^a-zA-Z]',' ',review)
        review=[lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review=' '.join(review)
        review_corpus.append(review)
    return review_corpus


## === cell 9
df['clean_review']=clean_review(df.Phrase.values)
df.head()


## === cell 10
df_train=df[df.Sentiment!=-999]
df_train.shape


## === cell 11
df_test=df[df.Sentiment==-999]
df_test.drop('Sentiment',axis=1,inplace=True)
print(df_test.shape)
df_test.head()


## === cell 12
del df
gc.collect()


## === cell 13
train_text=df_train.clean_review.values
test_text=df_test.clean_review.values
target=df_train.Sentiment.values
y=to_categorical(target)
print(train_text.shape,target.shape,y.shape)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3501408614.py in <cell line: 0>()
      2 test_text=df_test.clean_review.values
      3 target=df_train.Sentiment.values
----> 4 y=to_categorical(target)
      5 print(train_text.shape,target.shape,y.shape)

NameError: name 'to_categorical' is not defined

## === cell 14
X_train_text,X_val_text,y_train,y_val=train_test_split(train_text,y,test_size=0.2,stratify=y,random_state=123)
print(X_train_text.shape,y_train.shape)
print(X_val_text.shape,y_val.shape)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/665075301.py in <cell line: 0>()
----> 1 X_train_text,X_val_text,y_train,y_val=train_test_split(train_text,y,test_size=0.2,stratify=y,random_state=123)
      2 print(X_train_text.shape,y_train.shape)
      3 print(X_val_text.shape,y_val.shape)

NameError: name 'train_test_split' is not defined

## === cell 15
all_words=' '.join(X_train_text)
all_words=word_tokenize(all_words)
dist=FreqDist(all_words)
num_unique_word=len(dist)
num_unique_word


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1086686821.py in <cell line: 0>()
----> 1 all_words=' '.join(X_train_text)
      2 all_words=word_tokenize(all_words)
      3 dist=FreqDist(all_words)
      4 num_unique_word=len(dist)
      5 num_unique_word

NameError: name 'X_train_text' is not defined

## === cell 16
r_len=[]
for text in X_train_text:
    word=word_tokenize(text)
    l=len(word)
    r_len.append(l)
    
MAX_REVIEW_LEN=np.max(r_len)
MAX_REVIEW_LEN


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/739816846.py in <cell line: 0>()
      1 r_len=[]
----> 2 for text in X_train_text:
      3     word=word_tokenize(text)
      4     l=len(word)
      5     r_len.append(l)

NameError: name 'X_train_text' is not defined

## === cell 17
max_features = num_unique_word
max_words = MAX_REVIEW_LEN
batch_size = 128
epochs = 25
num_classes=5


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4062573349.py in <cell line: 0>()
----> 1 max_features = num_unique_word
      2 max_words = MAX_REVIEW_LEN
      3 batch_size = 128
      4 epochs = 25
      5 num_classes=5

NameError: name 'num_unique_word' is not defined

## === cell 18
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train_text))
X_train = tokenizer.texts_to_sequences(X_train_text)
X_val = tokenizer.texts_to_sequences(X_val_text)
X_test = tokenizer.texts_to_sequences(test_text)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2874337973.py in <cell line: 0>()
----> 1 tokenizer = Tokenizer(num_words=max_features)
      2 tokenizer.fit_on_texts(list(X_train_text))
      3 X_train = tokenizer.texts_to_sequences(X_train_text)
      4 X_val = tokenizer.texts_to_sequences(X_val_text)
      5 X_test = tokenizer.texts_to_sequences(test_text)

NameError: name 'Tokenizer' is not defined

## === cell 19
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_val = sequence.pad_sequences(X_val, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
print(X_train.shape,X_val.shape,X_test.shape)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2100988224.py in <cell line: 0>()
----> 1 X_train = sequence.pad_sequences(X_train, maxlen=max_words)
      2 X_val = sequence.pad_sequences(X_val, maxlen=max_words)
      3 X_test = sequence.pad_sequences(X_test, maxlen=max_words)
      4 print(X_train.shape,X_val.shape,X_test.shape)

NameError: name 'X_train' is not defined

## === cell 20
model= Sequential()
model.add(Embedding(max_features,100,input_length=max_words))
model.add(Dropout(0.2))

model.add(Conv1D(64,kernel_size=3,padding='same',activation='relu',strides=1))
model.add(GlobalMaxPooling1D())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes,activation='softmax'))


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/196725454.py in <cell line: 0>()
----> 1 model= Sequential()
      2 model.add(Embedding(max_features,100,input_length=max_words))
      3 model.add(Dropout(0.2))
      4 
      5 model.add(Conv1D(64,kernel_size=3,padding='same',activation='relu',strides=1))

NameError: name 'Sequential' is not defined

## === cell 21
%%time
history=model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=epochs, batch_size=batch_size, verbose=1)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'model' is not defined

## === cell 22
y_pred2=model.predict_classes(X_test, verbose=1)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3633648738.py in <cell line: 0>()
----> 1 y_pred2=model.predict_classes(X_test, verbose=1)

NameError: name 'model' is not defined

## === cell 23
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'validation')
plt.title('Model CNN Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3126789032.py in <cell line: 0>()
      1 # Summary for accuracy
----> 2 plt.plot(history.history['accuracy'], label='accuracy')
      3 plt.plot(history.history['val_accuracy'], label = 'validation')
      4 plt.title('Model CNN Accuracy')
      5 plt.xlabel('Epoch')

NameError: name 'plt' is not defined

## === cell 24
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model CNN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['training Loss', 'Validation Loss'], loc='upper right')


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1620935480.py in <cell line: 0>()
      1 # Summary for loss
----> 2 plt.plot(history.history['loss'])
      3 plt.plot(history.history['val_loss'])
      4 plt.title('Model CNN Loss')
      5 plt.xlabel('Epoch')

NameError: name 'plt' is not defined

## === cell 25
sub.Sentiment=y_pred2
sub.to_csv('submission.csv',index=False)
sub.head()


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2369554201.py in <cell line: 0>()
----> 1 sub.Sentiment=y_pred2
      2 sub.to_csv('submission.csv',index=False)
      3 sub.head()

NameError: name 'y_pred2' is not defined
