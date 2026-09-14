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

0.57952

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



## === cell 1
import numpy as np 
import pandas as pd 

train = pd.read_csv('../input/train.tsv',sep = '\t')
test = pd.read_csv('../input/test.tsv', sep = '\t')
print("Train set: {0}".format(train.shape))
print("Test set: {0}".format(test.shape))

df = pd.concat([train, test])
print("All df set: {0}".format(df.shape))

df.head()


## === cell 2
sub = pd.read_csv('../input/sampleSubmission.csv', sep = ',')
print("Submission: {0}".format(sub.shape))

sub.head()


## === cell 3
print("Training set distribution: ", train.groupby(['Sentiment']).size()/train.shape[0])


## === cell 4
import re
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


## === cell 5
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    review_stem=[]
    for word in text.split():
        word_stem = stemmer.stem(word)
        review_stem.append(word_stem)
    review_stem=' '.join(review_stem)
    return review_stem


## === cell 6
train['clean_phrase'] = train['Phrase'].apply(clean_text)
test['clean_phrase'] = test['Phrase'].apply(clean_text)
df['clean_phrase'] = df['Phrase'].apply(clean_text)


## === cell 7
train.head()


## === cell 8
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk import FreqDist


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 9
train_text=train.clean_phrase.values
test_text=test.clean_phrase.values
target=train.Sentiment.values
y=to_categorical(target)
print(train_text.shape,target.shape,y.shape)


## === cell 10
X_train_text,X_val_text,y_train,y_val=train_test_split(train_text,y,test_size=0.2,stratify=y,random_state=123)
print(X_train_text.shape,y_train.shape)
print(X_val_text.shape,y_val.shape)


## === cell 11
all_words = ' '.join(X_train_text)
word2count = {}
for word in all_words.split():
    if word not in word2count:
        word2count[word] = 1
    else:
        word2count[word] += 1
print("Number of unique words: ", len(word2count.keys()))


## === cell 12
df['length_review'] = df['clean_phrase'].apply(lambda x: len(x.split()))
print("Max phrase length: ", max(df['length_review']))


## === cell 13
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3073233065.py in <cell line: 0>()
----> 1 from keras.preprocessing.text import Tokenizer
      2 from keras.preprocessing.sequence import pad_sequences

ModuleNotFoundError: No module named 'keras.preprocessing.text'

## === cell 14
MAX_REVIEW_LENGTH = 49
FEATURE_LENGTH = 12011
BATCH_SIZE = 1000
EPOCHS = 100
NUM_CLASSES = 5


## === cell 15
tokenizer = Tokenizer(num_words = FEATURE_LENGTH)
tokenizer.fit_on_texts(list(np.concatenate((train_text, test_text), axis=0)))
X_train = tokenizer.texts_to_sequences(X_train_text)
X_val = tokenizer.texts_to_sequences(X_val_text)
X_test = tokenizer.texts_to_sequences(test_text)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3498490083.py in <cell line: 0>()
----> 1 tokenizer = Tokenizer(num_words = FEATURE_LENGTH)
      2 tokenizer.fit_on_texts(list(np.concatenate((train_text, test_text), axis=0)))
      3 X_train = tokenizer.texts_to_sequences(X_train_text)
      4 X_val = tokenizer.texts_to_sequences(X_val_text)
      5 X_test = tokenizer.texts_to_sequences(test_text)

NameError: name 'Tokenizer' is not defined

## === cell 16
X_train = pad_sequences(X_train, maxlen=MAX_REVIEW_LENGTH)
X_val = pad_sequences(X_val, maxlen=MAX_REVIEW_LENGTH)
X_test= pad_sequences(X_test, maxlen=MAX_REVIEW_LENGTH)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2072031703.py in <cell line: 0>()
----> 1 X_train = pad_sequences(X_train, maxlen=MAX_REVIEW_LENGTH)
      2 X_val = pad_sequences(X_val, maxlen=MAX_REVIEW_LENGTH)
      3 X_test= pad_sequences(X_test, maxlen=MAX_REVIEW_LENGTH)

NameError: name 'pad_sequences' is not defined

## === cell 17
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam


## === cell 23
from keras.layers import Flatten


## === cell 24
ann_model = Sequential()
ann_model.add(Embedding(FEATURE_LENGTH,250, input_length=MAX_REVIEW_LENGTH))
ann_model.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))
ann_model.add(Flatten())
ann_model.add(Dense(output_dim = 50, activation='tanh'))
ann_model.add(Dense(output_dim = 10, activation = 'relu'))
ann_model.add(Dense(NUM_CLASSES,activation='softmax'))
ann_model.compile(optimizer=Adam(lr=0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
ann_model.summary()


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1944202590.py in <cell line: 0>()
      1 ann_model = Sequential()
      2 ann_model.add(Embedding(FEATURE_LENGTH,250, input_length=MAX_REVIEW_LENGTH))
----> 3 ann_model.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))
      4 ann_model.add(Flatten())
      5 ann_model.add(Dense(output_dim = 50, activation='tanh'))

TypeError: Dense.__init__() missing 1 required positional argument: 'units'

## === cell 25
ann_history = ann_model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size = BATCH_SIZE, epochs = EPOCHS)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3317311458.py in <cell line: 0>()
----> 1 ann_history = ann_model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size = BATCH_SIZE, epochs = EPOCHS)

NameError: name 'X_train' is not defined

## === cell 26
y_pred = ann_model.predict_classes(X_test)


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1981625218.py in <cell line: 0>()
----> 1 y_pred = ann_model.predict_classes(X_test)

AttributeError: 'Sequential' object has no attribute 'predict_classes'

## === cell 27
test["Sentiment"] = y_pred
test[['PhraseId', 'Sentiment']].to_csv('submission.csv', index = False)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/469206064.py in <cell line: 0>()
----> 1 test["Sentiment"] = y_pred
      2 test[['PhraseId', 'Sentiment']].to_csv('submission.csv', index = False)

NameError: name 'y_pred' is not defined
