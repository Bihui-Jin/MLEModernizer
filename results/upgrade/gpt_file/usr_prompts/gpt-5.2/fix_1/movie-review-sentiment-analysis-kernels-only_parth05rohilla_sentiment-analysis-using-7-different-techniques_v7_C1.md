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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

0.64074

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


## === cell 1
PATH = '../input/'
os.listdir(PATH)


## === cell 2
train = pd.read_csv('../input/train.tsv',sep = '\t')
test = pd.read_csv('../input/test.tsv',sep = '\t')
sub = pd.read_csv('../input/sampleSubmission.csv' , sep = ',')


## === cell 3
train.head()


## === cell 4
test.head()


## === cell 5
class_count = train['Sentiment'].value_counts()
class_count


## === cell 6
x = np.array(class_count.index)
y = np.array(class_count.values)
plt.figure(figsize=(8,5))
sns.barplot(x,y)
plt.xlabel('Sentiment ')
plt.ylabel('Number of reviews ')


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1560277862.py in <cell line: 0>()
      2 y = np.array(class_count.values)
      3 plt.figure(figsize=(8,5))
----> 4 sns.barplot(x,y)
      5 plt.xlabel('Sentiment ')
      6 plt.ylabel('Number of reviews ')

TypeError: barplot() takes from 0 to 1 positional arguments but 2 were given

## === cell 7
print('Number of sentences in training set:',len(train['SentenceId'].unique()))
print('Number of sentences in test set:',len(test['SentenceId'].unique()))
print('Average words per sentence in train:',train.groupby('SentenceId')['Phrase'].count().mean())
print('Average words per sentence in test:',test.groupby('SentenceId')['Phrase'].count().mean())


## === cell 8
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
tokenizer = TweetTokenizer()


## === cell 9
vectorizer = TfidfVectorizer(ngram_range=(1, 3), tokenizer=tokenizer.tokenize)
full_text = list(train['Phrase'].values) + list(test['Phrase'].values)
vectorizer.fit(full_text)
train_vectorized = vectorizer.transform(train['Phrase'])
test_vectorized = vectorizer.transform(test['Phrase'])


## === cell 10
y = train['Sentiment']


## === cell 11
from sklearn.model_selection import train_test_split
x_train , x_val, y_train , y_val = train_test_split(train_vectorized,y,test_size = 0.2)


## === cell 12
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.multiclass import OneVsRestClassifier


## === cell 13
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


## === cell 14
lr = LogisticRegression()
ovr = OneVsRestClassifier(lr)
ovr.fit(x_train,y_train)
print(classification_report( ovr.predict(x_val) , y_val))
print(accuracy_score( ovr.predict(x_val) , y_val ))


## === cell 15
svm = LinearSVC()
svm.fit(x_train,y_train)
print(classification_report( svm.predict(x_val) , y_val))
print(accuracy_score( svm.predict(x_val) , y_val ))


## === cell 16
estimators = [ ('svm',svm) , ('ovr' , ovr) ]
clf = VotingClassifier(estimators , voting='hard')
clf.fit(x_train,y_train)
print(classification_report( clf.predict(x_val) , y_val))
print(accuracy_score( clf.predict(x_val) , y_val ))


## === cell 17
from keras.utils import to_categorical
target=train.Sentiment.values
y=to_categorical(target)
y


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 18
max_features = 13000
max_words = 50
batch_size = 128
epochs = 3
num_classes=5


## === cell 19
from sklearn.model_selection import train_test_split
X_train , X_val , Y_train , Y_val = train_test_split(train['Phrase'],y,test_size = 0.20)


## === cell 20
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,GRU,LSTM,Embedding
from keras.optimizers import Adam
from keras.layers import SpatialDropout1D,Dropout,Bidirectional,Conv1D,GlobalMaxPooling1D,MaxPooling1D,Flatten
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2406137976.py in <cell line: 0>()
----> 1 from keras.preprocessing.text import Tokenizer
      2 from keras.preprocessing.sequence import pad_sequences
      3 from keras.models import Sequential
      4 from keras.layers import Dense,GRU,LSTM,Embedding
      5 from keras.optimizers import Adam

ModuleNotFoundError: No module named 'keras.preprocessing.text'

## === cell 21
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2189424765.py in <cell line: 0>()
----> 1 tokenizer = Tokenizer(num_words=max_features)
      2 tokenizer.fit_on_texts(list(X_train))
      3 X_train = tokenizer.texts_to_sequences(X_train)
      4 X_val = tokenizer.texts_to_sequences(X_val)

NameError: name 'Tokenizer' is not defined

## === cell 22
X_test = tokenizer.texts_to_sequences(test['Phrase'])
X_test =pad_sequences(X_test, maxlen=max_words)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3405600963.py in <cell line: 0>()
----> 1 X_test = tokenizer.texts_to_sequences(test['Phrase'])
      2 X_test =pad_sequences(X_test, maxlen=max_words)

AttributeError: 'TweetTokenizer' object has no attribute 'texts_to_sequences'

## === cell 23
len(X_test)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1788158412.py in <cell line: 0>()
----> 1 len(X_test)

NameError: name 'X_test' is not defined

## === cell 24
X_train =pad_sequences(X_train, maxlen=max_words)
X_val = pad_sequences(X_val, maxlen=max_words)
X_test =pad_sequences(X_test, maxlen=max_words)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1229678293.py in <cell line: 0>()
----> 1 X_train =pad_sequences(X_train, maxlen=max_words)
      2 X_val = pad_sequences(X_val, maxlen=max_words)
      3 X_test =pad_sequences(X_test, maxlen=max_words)

NameError: name 'pad_sequences' is not defined

## === cell 25
model_GRU=Sequential()
model_GRU.add(Embedding(max_features,100,mask_zero=True))
model_GRU.add(GRU(64,dropout=0.4,return_sequences=True))
model_GRU.add(GRU(32,dropout=0.5,return_sequences=False))
model_GRU.add(Dense(num_classes,activation='softmax'))
model_GRU.compile(loss='categorical_crossentropy',optimizer=Adam(lr = 0.001),metrics=['accuracy'])
model_GRU.summary()


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/148828156.py in <cell line: 0>()
----> 1 model_GRU=Sequential()
      2 model_GRU.add(Embedding(max_features,100,mask_zero=True))
      3 model_GRU.add(GRU(64,dropout=0.4,return_sequences=True))
      4 model_GRU.add(GRU(32,dropout=0.5,return_sequences=False))
      5 model_GRU.add(Dense(num_classes,activation='softmax'))

NameError: name 'Sequential' is not defined

## === cell 26
model2_GRU=Sequential()
model2_GRU.add(Embedding(max_features,100,mask_zero=True))
model2_GRU.add(GRU(64,dropout=0.4,return_sequences=True))
model2_GRU.add(GRU(32,dropout=0.5,return_sequences=False))
model2_GRU.add(Dense(num_classes,activation='sigmoid'))
model2_GRU.compile(loss='binary_crossentropy',optimizer=Adam(lr = 0.001),metrics=['accuracy'])
model2_GRU.summary()


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1865322698.py in <cell line: 0>()
----> 1 model2_GRU=Sequential()
      2 model2_GRU.add(Embedding(max_features,100,mask_zero=True))
      3 model2_GRU.add(GRU(64,dropout=0.4,return_sequences=True))
      4 model2_GRU.add(GRU(32,dropout=0.5,return_sequences=False))
      5 model2_GRU.add(Dense(num_classes,activation='sigmoid'))

NameError: name 'Sequential' is not defined

## === cell 27
model3_LSTM=Sequential()
model3_LSTM.add(Embedding(max_features,100,mask_zero=True))
model3_LSTM.add(LSTM(64,dropout=0.4,return_sequences=True))
model3_LSTM.add(LSTM(32,dropout=0.5,return_sequences=False))
model3_LSTM.add(Dense(num_classes,activation='sigmoid'))
model3_LSTM.compile(loss='binary_crossentropy',optimizer=Adam(lr = 0.001),metrics=['accuracy'])
model3_LSTM.summary()


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3744110771.py in <cell line: 0>()
----> 1 model3_LSTM=Sequential()
      2 model3_LSTM.add(Embedding(max_features,100,mask_zero=True))
      3 model3_LSTM.add(LSTM(64,dropout=0.4,return_sequences=True))
      4 model3_LSTM.add(LSTM(32,dropout=0.5,return_sequences=False))
      5 model3_LSTM.add(Dense(num_classes,activation='sigmoid'))

NameError: name 'Sequential' is not defined

## === cell 28
model4_BGRU = Sequential()
model4_BGRU.add(Embedding(max_features, 100, input_length=max_words))
model4_BGRU.add(SpatialDropout1D(0.25))
model4_BGRU.add(Bidirectional(GRU(64,dropout=0.4,return_sequences = True)))
model4_BGRU.add(Bidirectional(GRU(32,dropout=0.5,return_sequences = False)))
model4_BGRU.add(Dense(5, activation='sigmoid'))
model4_BGRU.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model4_BGRU.summary()


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3630926673.py in <cell line: 0>()
----> 1 model4_BGRU = Sequential()
      2 model4_BGRU.add(Embedding(max_features, 100, input_length=max_words))
      3 model4_BGRU.add(SpatialDropout1D(0.25))
      4 model4_BGRU.add(Bidirectional(GRU(64,dropout=0.4,return_sequences = True)))
      5 model4_BGRU.add(Bidirectional(GRU(32,dropout=0.5,return_sequences = False)))

NameError: name 'Sequential' is not defined

## === cell 29
model5_CNN= Sequential()
model5_CNN.add(Embedding(max_features,100,input_length=max_words))
model5_CNN.add(Dropout(0.2))
model5_CNN.add(Conv1D(64,kernel_size=3,padding='same',activation='relu',strides=1))
model5_CNN.add(GlobalMaxPooling1D())
model5_CNN.add(Dense(128,activation='relu'))
model5_CNN.add(Dropout(0.2))
model5_CNN.add(Dense(num_classes,activation='sigmoid'))
model5_CNN.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model5_CNN.summary()


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4021346917.py in <cell line: 0>()
----> 1 model5_CNN= Sequential()
      2 model5_CNN.add(Embedding(max_features,100,input_length=max_words))
      3 model5_CNN.add(Dropout(0.2))
      4 model5_CNN.add(Conv1D(64,kernel_size=3,padding='same',activation='relu',strides=1))
      5 model5_CNN.add(GlobalMaxPooling1D())

NameError: name 'Sequential' is not defined

## === cell 30
%%time
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)

history5=model5_CNN.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=5, batch_size=batch_size, verbose=1,callbacks = [early_stop])


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'EarlyStopping' is not defined

## === cell 31
y_pred5=model5_CNN.predict_classes(X_test, verbose=1)
sub.Sentiment=y_pred5
sub.to_csv('sub5_CNN.csv',index=False)
sub.head()


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2843209290.py in <cell line: 0>()
----> 1 y_pred5=model5_CNN.predict_classes(X_test, verbose=1)
      2 sub.Sentiment=y_pred5
      3 sub.to_csv('sub5_CNN.csv',index=False)
      4 sub.head()

NameError: name 'model5_CNN' is not defined

## === cell 32
model6_CnnGRU= Sequential()
model6_CnnGRU.add(Embedding(max_features,100,input_length=max_words))
model6_CnnGRU.add(Conv1D(64,kernel_size=3,padding='same',activation='relu'))
model6_CnnGRU.add(MaxPooling1D(pool_size=2))
model6_CnnGRU.add(Dropout(0.25))
model6_CnnGRU.add(GRU(128,return_sequences=True))
model6_CnnGRU.add(Dropout(0.3))
model6_CnnGRU.add(Flatten())
model6_CnnGRU.add(Dense(128,activation='relu'))
model6_CnnGRU.add(Dropout(0.5))
model6_CnnGRU.add(Dense(5,activation='sigmoid'))
model6_CnnGRU.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model6_CnnGRU.summary()


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1214614237.py in <cell line: 0>()
----> 1 model6_CnnGRU= Sequential()
      2 model6_CnnGRU.add(Embedding(max_features,100,input_length=max_words))
      3 model6_CnnGRU.add(Conv1D(64,kernel_size=3,padding='same',activation='relu'))
      4 model6_CnnGRU.add(MaxPooling1D(pool_size=2))
      5 model6_CnnGRU.add(Dropout(0.25))

NameError: name 'Sequential' is not defined

## === cell 33
model7_GruCNN = Sequential()
model7_GruCNN.add(Embedding(max_features,100,input_length=max_words))
model7_GruCNN.add(Dropout(0.2))
model7_GruCNN.add(Bidirectional(GRU(units=128 , return_sequences=True)))
model7_GruCNN.add(Conv1D(32 , kernel_size=3 , padding='same' , activation='relu'))
model7_GruCNN.add(GlobalMaxPooling1D())
model7_GruCNN.add(Dense(units = 64 , activation='relu'))
model7_GruCNN.add(Dropout(0.5))
model7_GruCNN.add(Dense(units=5,activation='sigmoid'))
model7_GruCNN.compile(loss='binary_crossentropy' , optimizer = 'adam' , metrics=['accuracy'])
model7_GruCNN.summary()


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1877315919.py in <cell line: 0>()
----> 1 model7_GruCNN = Sequential()
      2 model7_GruCNN.add(Embedding(max_features,100,input_length=max_words))
      3 model7_GruCNN.add(Dropout(0.2))
      4 model7_GruCNN.add(Bidirectional(GRU(units=128 , return_sequences=True)))
      5 model7_GruCNN.add(Conv1D(32 , kernel_size=3 , padding='same' , activation='relu'))

NameError: name 'Sequential' is not defined
