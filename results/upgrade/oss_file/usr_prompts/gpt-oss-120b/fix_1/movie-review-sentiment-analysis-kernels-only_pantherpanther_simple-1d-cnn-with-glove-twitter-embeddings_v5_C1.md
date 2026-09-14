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

0.63488

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

path = '../input/train.tsv'
features = ['pid','sid','p','s']
sms = pd.read_csv(path,names=features,sep='\t',header=0)
print(sms.shape)
print(sms.head(10))


## === cell 1
len_checker=[]
for row in sms['p']:
    if(len(row)>3):
        len_checker.append(1)
    else:
        len_checker.append(0)
sms['dummy']=len_checker
sms=sms[sms['dummy']==1]
sms.head(10)


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


## === cell 4
import nltk
from nltk.stem import PorterStemmer
ps=PorterStemmer()
l2=[]
review=[]
s2=''
for row in X:
    for words in nltk.word_tokenize(row):
            l2.append(ps.stem(words.lower()))
            l2.append(' ')
    s2=''.join(l2)
    review.append(s2)
    s2=''
    l2=[]
X=review
print(X[:1])


## === cell 5
from sklearn.cross_validation import train_test_split
X_train, X_inter, Y_train, Y_inter = train_test_split(X, Y,test_size=0.3,random_state=1)
X_val, X_test, Y_val, Y_test = train_test_split(X_inter, Y_inter,test_size=0.5,random_state=1)
print(len(X_train))
print(len(X_val))
print(len(X_test))


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3659396103.py in <cell line: 0>()
----> 1 from sklearn.cross_validation import train_test_split
      2 X_train, X_inter, Y_train, Y_inter = train_test_split(X, Y,test_size=0.3,random_state=1)
      3 X_val, X_test, Y_val, Y_test = train_test_split(X_inter, Y_inter,test_size=0.5,random_state=1)
      4 print(len(X_train))
      5 print(len(X_val))

ModuleNotFoundError: No module named 'sklearn.cross_validation'

## === cell 6
import nltk
import gensim
data_words = [nltk.word_tokenize(x) for x in X_train] 
print(data_words[:10])
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=10) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=10)  

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


data_words_bigrams = make_trigrams(data_words)
print(data_words_bigrams[:5])
token=''
token2=[]
for ritem in data_words_bigrams:
    for eachword in ritem:
        token=token+' '+eachword
    token2.append(token)
    token=''
X_train=token2
print('X after trigrams : ')
print(X_train[:5])

data_words=0
data_words_bigrams=0
data_words = [nltk.word_tokenize(x) for x in X_val]
data_words_bigrams = make_trigrams(data_words)
token=''
token2=[]
for ritem in data_words_bigrams:
    for eachword in ritem:
        token=token+' '+eachword
    token2.append(token)
    token=''
X_val=token2


data_words=0
data_words_bigrams=0
data_words = [nltk.word_tokenize(x) for x in X_test]
data_words_bigrams = make_trigrams(data_words)
token=''
token2=[]
for ritem in data_words_bigrams:
    for eachword in ritem:
        token=token+' '+eachword
    token2.append(token)
    token=''
X_test=token2


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1674923791.py in <cell line: 0>()
      1 import nltk
      2 import gensim
----> 3 data_words = [nltk.word_tokenize(x) for x in X_train]
      4 print(data_words[:10])
      5 bigram = gensim.models.Phrases(data_words, min_count=5, threshold=10) # higher threshold fewer phrases.

NameError: name 'X_train' is not defined

## === cell 7
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

max_sentence=len(max(X,key=len))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
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


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 8
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, GlobalMaxPooling1D
model=0
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100,input_length=max_sentence, trainable=True))
model.add(Conv1D(256, 2,strides=1,padding='valid', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(5, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, dummy_y_train,validation_data=(val_x, dummy_y_val), epochs=3,batch_size=128,verbose=1)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1451242872.py in <cell line: 0>()
      2 from keras.layers import Dense
      3 from keras.layers import Embedding
----> 4 from keras.layers.recurrent import LSTM
      5 from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, GlobalMaxPooling1D
      6 #def baseline_model():

ModuleNotFoundError: No module named 'keras.layers.recurrent'

## === cell 9
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


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3643276819.py in <cell line: 0>()
      1 import sklearn.metrics as sklm
----> 2 predictions=model.predict(test_x)
      3 pred=[]
      4 for idx,val in enumerate(predictions):
      5     pred.append(np.argmax(val))

NameError: name 'model' is not defined

## === cell 10
path = '../input/test.tsv'
features = ['PhraseId','sid','p']
test_frame = pd.read_csv(path,names=features,sep='\t',header=0)



l2=[]
review=[]
s2=''
for row in test_frame['p']:
    for words in nltk.word_tokenize(row):
            l2.append(ps.stem(words.lower()))
            l2.append(' ')
    s2=''.join(l2)
    review.append(s2)
    s2=''
    l2=[]
test_frame['p_stemmed']=review


data_words=0
data_words_bigrams=0
data_words = [nltk.word_tokenize(x) for x in test_frame['p_stemmed']]
data_words_bigrams = make_trigrams(data_words)
token=''
token2=[]
for ritem in data_words_bigrams:
    for eachword in ritem:
        token=token+' '+eachword
    token2.append(token)
    token=''
test_frame['p_ngrams']=token2


encoded_docs=0
encoded_docs = tokenizer.texts_to_sequences(test_frame['p_ngrams'])
temp_test = pad_sequences(encoded_docs, maxlen=max_sentence, padding='post')

predictions=model.predict(temp_test)
pred=[]
for idx,val in enumerate(predictions):
    pred.append(np.argmax(val))

test_frame['Sentiment']=pred
test_frame.drop(['p_ngrams','p','sid','p_stemmed'],axis=1,inplace=True)
test_frame.head(10)
test_frame.to_csv('output.csv',index=False)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/202626452.py in <cell line: 0>()
     24 data_words_bigrams=0
     25 data_words = [nltk.word_tokenize(x) for x in test_frame['p_stemmed']]
---> 26 data_words_bigrams = make_trigrams(data_words)
     27 token=''
     28 token2=[]

NameError: name 'make_trigrams' is not defined

## === cell 11
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

path = '../input/train.tsv'
features = ['pid','sid','p','s']
sms = pd.read_csv(path,names=features,sep='\t',header=0)
print(sms.shape)
print(sms.head(10))


## === cell 12
len_checker=[]
for row in sms['p']:
    if(len(row)>3):
        len_checker.append(1)
    else:
        len_checker.append(0)
sms['dummy']=len_checker
sms=sms[sms['dummy']==1]
sms.head(10)


## === cell 13
import numpy.random as nr
Labels=np.array(sms.s)
Features=np.array(sms.p)
temp_Labels_4 = Labels[Labels == 4]  # Save these
temp_Features_4 = Features[Labels == 4] # Save these

temp_Labels_2 = Labels[Labels == 2]  # Undersample these
temp_Features_2 = Features[Labels == 2] # Undersample these
indx2 = nr.choice(temp_Features_2.shape[0], temp_Features_4.shape[0], replace=True)

temp_Labels_3 = Labels[Labels == 3]  # Undersample these
temp_Features_3 = Features[Labels == 3] # Undersample these
indx3 = nr.choice(temp_Features_3.shape[0], temp_Features_4.shape[0], replace=True)

temp_Labels_1 = Labels[Labels == 1]  # Undersample these
temp_Features_1 = Features[Labels == 1] # Undersample these
indx1 = nr.choice(temp_Features_1.shape[0], temp_Features_4.shape[0], replace=True)


l0=Labels[Labels==0]
f0=Features[Labels==0]

X = np.concatenate((f0, temp_Features_2[indx2,]), axis = 0)
Y = np.concatenate((l0, temp_Labels_2[indx2,]), axis = 0) 

X = np.concatenate((X, temp_Features_3[indx3,]), axis = 0)
Y = np.concatenate((Y, temp_Labels_3[indx3,]), axis = 0) 

X = np.concatenate((X, temp_Features_1[indx1,]), axis = 0)
Y = np.concatenate((Y, temp_Labels_1[indx1,]), axis = 0) 

X = np.concatenate((temp_Features_4,X), axis = 0)
Y = np.concatenate((temp_Labels_4,Y), axis = 0) 

sms=pd.DataFrame()
sms['p']=X
sms['s']=Y

print('done')


## === cell 14
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


## === cell 15
X=sms.p
Y=sms.s


## === cell 16
import nltk
from nltk.stem import PorterStemmer
ps=PorterStemmer()
l2=[]
review=[]
s2=''
for row in X:
    for words in nltk.word_tokenize(row):
            l2.append(ps.stem(words.lower()))
            l2.append(' ')
    s2=''.join(l2)
    review.append(s2)
    s2=''
    l2=[]
X=review
print(X[:1])


## === cell 17
from sklearn.cross_validation import train_test_split
X_train, X_inter, Y_train, Y_inter = train_test_split(X, Y,test_size=0.3,random_state=123)
X_val, X_test, Y_val, Y_test = train_test_split(X_inter, Y_inter,test_size=0.5,random_state=234)
print(len(X_train))
print(len(X_val))
print(len(X_test))


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1339433911.py in <cell line: 0>()
----> 1 from sklearn.cross_validation import train_test_split
      2 X_train, X_inter, Y_train, Y_inter = train_test_split(X, Y,test_size=0.3,random_state=123)
      3 X_val, X_test, Y_val, Y_test = train_test_split(X_inter, Y_inter,test_size=0.5,random_state=234)
      4 print(len(X_train))
      5 print(len(X_val))

ModuleNotFoundError: No module named 'sklearn.cross_validation'

## === cell 18
import nltk
import gensim
data_words = [nltk.word_tokenize(x) for x in X_train] 
print(data_words[:10])
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=10) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=10)  

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


data_words_bigrams = make_trigrams(data_words)
print(data_words_bigrams[:5])
token=''
token2=[]
for ritem in data_words_bigrams:
    for eachword in ritem:
        token=token+' '+eachword
    token2.append(token)
    token=''
X_train=token2
print('X after trigrams : ')
print(X_train[:5])

data_words=0
data_words_bigrams=0
data_words = [nltk.word_tokenize(x) for x in X_val]
data_words_bigrams = make_trigrams(data_words)
token=''
token2=[]
for ritem in data_words_bigrams:
    for eachword in ritem:
        token=token+' '+eachword
    token2.append(token)
    token=''
X_val=token2


data_words=0
data_words_bigrams=0
data_words = [nltk.word_tokenize(x) for x in X_test]
data_words_bigrams = make_trigrams(data_words)
token=''
token2=[]
for ritem in data_words_bigrams:
    for eachword in ritem:
        token=token+' '+eachword
    token2.append(token)
    token=''
X_test=token2


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1674923791.py in <cell line: 0>()
      1 import nltk
      2 import gensim
----> 3 data_words = [nltk.word_tokenize(x) for x in X_train]
      4 print(data_words[:10])
      5 bigram = gensim.models.Phrases(data_words, min_count=5, threshold=10) # higher threshold fewer phrases.

NameError: name 'X_train' is not defined

## === cell 19
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

max_sentence=len(max(X,key=len))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
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


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2074097162.py in <cell line: 0>()
----> 1 from keras.preprocessing.text import Tokenizer
      2 from keras.preprocessing.sequence import pad_sequences
      3 from sklearn.preprocessing import LabelEncoder
      4 from keras.utils import np_utils
      5 

ModuleNotFoundError: No module named 'keras.preprocessing.text'

## === cell 20
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, GlobalMaxPooling1D
model=0
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100,input_length=max_sentence, trainable=True))
model.add(Conv1D(128, 2,strides=1,padding='valid', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(5, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, dummy_y_train,validation_data=(val_x, dummy_y_val), epochs=2,batch_size=128,verbose=1)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3568763959.py in <cell line: 0>()
      2 from keras.layers import Dense, Dropout
      3 from keras.layers import Embedding
----> 4 from keras.layers.recurrent import LSTM
      5 from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, GlobalMaxPooling1D
      6 #def baseline_model():

ModuleNotFoundError: No module named 'keras.layers.recurrent'

## === cell 21
import sklearn.metrics as sklm
predictions=model.predict(test_x)
pred=[]
for idx,val in enumerate(predictions):
    pred.append(np.argmax(val))


print(len(Y_test))
print(len(pred))
print(set(Y_test))
print(set(pred))
print('\n\n\n')


conf = sklm.confusion_matrix(Y_test, pred)
metrics = sklm.precision_recall_fscore_support(Y_test, pred)


print('Accuracy  %0.2f' % sklm.accuracy_score(Y_test, pred))
print('                0        1        2        3        4')
print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1]+ '        %6.2f' % metrics[0][2]+ '        %6.2f' % metrics[0][3]+ '        %6.2f' % metrics[0][4])
print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1]+ '        %6.2f' % metrics[1][2]+ '        %6.2f' % metrics[1][3]+ '        %6.2f' % metrics[1][4])
print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1]+ '        %6.2f' % metrics[2][2]+ '        %6.2f' % metrics[2][3]+ '        %6.2f' % metrics[2][4])

Y_test=pd.Series(Y_test)
pred=pd.Series(pred)

print(conf)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/959558645.py in <cell line: 0>()
      1 import sklearn.metrics as sklm
----> 2 predictions=model.predict(test_x)
      3 pred=[]
      4 for idx,val in enumerate(predictions):
      5     pred.append(np.argmax(val))

NameError: name 'model' is not defined

## === cell 22
encoded_Y_val = encoder.transform(Y_val)
dummy_y_val = np_utils.to_categorical(encoded_Y_val)

model.fit(test_x,dummy_y_val,epochs=4,batch_size=128)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3073674622.py in <cell line: 0>()
----> 1 encoded_Y_val = encoder.transform(Y_val)
      2 # convert integers to dummy variables (i.e. one hot encoded)
      3 dummy_y_val = np_utils.to_categorical(encoded_Y_val)
      4 
      5 model.fit(test_x,dummy_y_val,epochs=4,batch_size=128)

NameError: name 'encoder' is not defined
