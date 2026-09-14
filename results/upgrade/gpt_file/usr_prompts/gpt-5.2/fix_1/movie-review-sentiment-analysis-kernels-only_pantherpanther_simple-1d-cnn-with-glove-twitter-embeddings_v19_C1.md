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

No external packages required in the script and installed.

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

0.6389157062692331

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

import os
print(os.listdir("../input"))

path = '../input/movie-review-sentiment-analysis-kernels-only/train.tsv'
features = ['pid','sid','p','s']
sms = pd.read_csv(path,names=features,sep='\t',header=0)
print(sms.shape)
print(sms.head(10))


## === cell 2
import matplotlib.pyplot as plt
def plot_bars(auto_prices, cols):
    for col in cols:
        fig = plt.figure(figsize=(6,6)) # define plot area
        ax = fig.gca() # define axis    
        counts = auto_prices[col].value_counts() # find the counts for each unique category
        counts.plot.bar(ax = ax, color = 'blue') # Use the plot.bar method on the counts data frame
        ax.set_title('Number sentiments' + col) # Give the plot a main title
        ax.set_xlabel(col) # Set text for the x axis
        ax.set_ylabel('freq')# Set text for y axis
        plt.show()

plot_cols = ['s']
plot_bars(sms, plot_cols)  
print(sms.s.value_counts())

## === cell 3
X=sms.p
Y=sms.s

## === cell 5
import nltk
from nltk.stem import PorterStemmer
ps=PorterStemmer()
l2=[]
review=[]
s2=''
for row in X:
    for words in nltk.word_tokenize(row):
            l2.append(words.lower())
            l2.append(' ')
    s2=''.join(l2)
    review.append(s2)
    s2=''
    l2=[]
X=review
print(X[:1])

## === cell 7
from sklearn.cross_validation import train_test_split
X_train, X_inter, Y_train, Y_inter = train_test_split(X, Y,test_size=0.3,random_state=1)
X_val, X_test, Y_val, Y_test = train_test_split(X_inter, Y_inter,test_size=0.5,random_state=1)
print(len(X_train))
print(len(X_val))
print(len(X_test))

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3669013271.py in <cell line: 0>()
----> 1 from sklearn.cross_validation import train_test_split
      2 X_train, X_inter, Y_train, Y_inter = train_test_split(X, Y,test_size=0.3,random_state=1)
      3 X_val, X_test, Y_val, Y_test = train_test_split(X_inter, Y_inter,test_size=0.5,random_state=1)
      4 print(len(X_train))
      5 print(len(X_val))

ModuleNotFoundError: No module named 'sklearn.cross_validation'

## === cell 10
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

max_sentence=len(max(X,key=len))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 12
from numpy import asarray
embeddings_index = dict()
f = open('../input/glove-t/glove.twitter.27B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3164138884.py in <cell line: 0>()
      1 from numpy import asarray
      2 embeddings_index = dict()
----> 3 f = open('../input/glove-t/glove.twitter.27B.100d.txt')
      4 for line in f:
      5     values = line.split()

FileNotFoundError: [Errno 2] No such file or directory: '../input/glove-t/glove.twitter.27B.100d.txt'

## === cell 14
from numpy import zeros
vocab_size = len(tokenizer.word_index) + 1
embedding_matrix = zeros((vocab_size, 100))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4206886529.py in <cell line: 0>()
      1 from numpy import zeros
----> 2 vocab_size = len(tokenizer.word_index) + 1
      3 embedding_matrix = zeros((vocab_size, 100))
      4 for word, i in tokenizer.word_index.items():
      5     embedding_vector = embeddings_index.get(word)

NameError: name 'tokenizer' is not defined

## === cell 16
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

max_sentence=len(max(X,key=len))

encoded_docs = tokenizer.texts_to_sequences(X_train)
train_x = pad_sequences(encoded_docs, maxlen=max_sentence, padding='post')
print(train_x[0])    

encoded_docs=0
encoded_docs = tokenizer.texts_to_sequences(X_val)
val_x = pad_sequences(encoded_docs, maxlen=max_sentence, padding='post')
print(val_x[1])

encoded_docs=0
encoded_docs = tokenizer.texts_to_sequences(X_test)
test_x = pad_sequences(encoded_docs, maxlen=max_sentence, padding='post')
print(test_x[1])

encoder = LabelEncoder()
encoder.fit(Y_train)
encoded_Y_train = encoder.transform(Y_train)
dummy_y_train = np_utils.to_categorical(encoded_Y_train)
print(dummy_y_train[:3])

encoded_Y_val = encoder.transform(Y_val)
dummy_y_val = np_utils.to_categorical(encoded_Y_val)




vocab_size = len(tokenizer.word_index) + 1

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/341938927.py in <cell line: 0>()
----> 1 from keras.preprocessing.text import Tokenizer
      2 from keras.preprocessing.sequence import pad_sequences
      3 from sklearn.preprocessing import LabelEncoder
      4 from keras.utils import np_utils
      5 

ModuleNotFoundError: No module named 'keras.preprocessing.text'

## === cell 18
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding,CuDNNGRU,Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, GlobalMaxPooling1D

model=0
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_sentence, trainable=True, weights=[embedding_matrix] ))
model.add(Conv1D(128, 2,strides=1,padding='valid', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(5, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, dummy_y_train,  validation_data=(val_x, dummy_y_val), epochs=4,batch_size=128,verbose=1)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3696971723.py in <cell line: 0>()
      2 from keras.models import Sequential
      3 from keras.layers import Dense
----> 4 from keras.layers import Embedding,CuDNNGRU,Bidirectional
      5 from keras.layers.recurrent import LSTM
      6 from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, GlobalMaxPooling1D

ImportError: cannot import name 'CuDNNGRU' from 'keras.layers' (/usr/local/lib/python3.11/dist-packages/keras/api/layers/__init__.py)

## === cell 20
import sklearn.metrics as sklm
predictions=model.predict(test_x)
pred=[]
for idx,val in enumerate(predictions):
    pred.append(np.argmax(val))


print(len(Y_test))
print(len(pred))
print(set(Y_test))
print(set(pred))
metrics = sklm.precision_recall_fscore_support(Y_test, pred)


print('Accuracy  %0.2f' % sklm.accuracy_score(Y_test, pred))
print('           0     1     2     3     4')
print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1]+ '        %6.2f' % metrics[0][2]+ '        %6.2f' % metrics[0][3]+ '        %6.2f' % metrics[0][4])
print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1]+ '        %6.2f' % metrics[1][2]+ '        %6.2f' % metrics[1][3]+ '        %6.2f' % metrics[1][4])
print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1]+ '        %6.2f' % metrics[2][2]+ '        %6.2f' % metrics[2][3]+ '        %6.2f' % metrics[2][4])

Y_test=pd.Series(Y_test)
pred=pd.Series(pred)
pd.crosstab(Y_test, pred, rownames=['True'], colnames=['Predicted'], margins=True)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3643276819.py in <cell line: 0>()
      1 import sklearn.metrics as sklm
----> 2 predictions=model.predict(test_x)
      3 pred=[]
      4 for idx,val in enumerate(predictions):
      5     pred.append(np.argmax(val))

NameError: name 'model' is not defined

## === cell 22


path = '../input/movie-review-sentiment-analysis-kernels-only/test.tsv'
features = ['PhraseId','sid','p']
test_frame = pd.read_csv(path,names=features,sep='\t',header=0)



l2=[]
review=[]
s2=''
for row in test_frame['p']:
    for words in nltk.word_tokenize(row):
            l2.append(words.lower())
            l2.append(' ')
    s2=''.join(l2)
    review.append(s2)
    s2=''
    l2=[]
test_frame['p_stemmed']=review




encoded_docs=0
encoded_docs = tokenizer.texts_to_sequences(test_frame['p_stemmed'])
temp_test = pad_sequences(encoded_docs, maxlen=max_sentence, padding='post')

predictions=model.predict(temp_test)
pred=[]
for idx,val in enumerate(predictions):
    pred.append(np.argmax(val))

test_frame['Sentiment']=pred
test_frame.drop(['p','sid','p_stemmed'],axis=1,inplace=True)
print(test_frame.head(10))
test_frame.to_csv('output.csv',index=False)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1956191742.py in <cell line: 0>()
     24 
     25 encoded_docs=0
---> 26 encoded_docs = tokenizer.texts_to_sequences(test_frame['p_stemmed'])
     27 temp_test = pad_sequences(encoded_docs, maxlen=max_sentence, padding='post')
     28 

NameError: name 'tokenizer' is not defined
