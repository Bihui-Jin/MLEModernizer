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

0.40443

# 6. Current score

0.45258

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import word2vec


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
print(train_df.shape, test_df.shape)


## === cell 2
train_df.head()


## === cell 3
train_df['author'].value_counts()


## === cell 4
train_df['text_length'] = train_df['text'].str.len()


## === cell 5
train_df.hist()
plt.show()


## === cell 6
test_df['text_length'] = test_df['text'].str.len()
test_df.hist()
plt.show()


## === cell 7
train_df['author'] = pd.Categorical(train_df['author'])
df_Dummies = pd.get_dummies(train_df['author'], prefix='author')
train_df = pd.concat([train_df, df_Dummies], axis=1)
train_df.head()


## === cell 8
def clean_text(text):
    text = text.lower()
    text = re.sub('\W', ' ', text)
    return text


## === cell 9
train_df['text'] = train_df['text'].map(lambda com : clean_text(com))


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3001263591.py in <cell line: 0>()
      1 # clean train_df['text']
----> 2 train_df['text'] = train_df['text'].map(lambda com : clean_text(com))

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in map(self, arg, na_action)
   4698         dtype: object
   4699         """
-> 4700         new_values = self._map_values(arg, na_action=na_action)
   4701         return self._constructor(new_values, index=self.index, copy=False).__finalize__(
   4702             self, method="map"

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _map_values(self, mapper, na_action, convert)
    919             return arr.map(mapper, na_action=na_action)
    920 
--> 921         return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
    922 
    923     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/algorithms.py in map_array(arr, mapper, na_action, convert)
   1741     values = arr.astype(object, copy=False)
   1742     if na_action is None:
-> 1743         return lib.map_infer(values, mapper, convert=convert)
   1744     else:
   1745         return lib.map_infer_mask(

lib.pyx in pandas._libs.lib.map_infer()

/tmp/ipykernel_11/3001263591.py in <lambda>(com)
      1 # clean train_df['text']
----> 2 train_df['text'] = train_df['text'].map(lambda com : clean_text(com))

/tmp/ipykernel_11/551245950.py in clean_text(text)
      2 def clean_text(text):
      3     text = text.lower()
----> 4     text = re.sub('\W', ' ', text)
      5     return text

NameError: name 're' is not defined

## === cell 10
test_df['text'] = test_df['text'].map(lambda com : clean_text(com))


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2859460325.py in <cell line: 0>()
      1 # clean test_df['text']
----> 2 test_df['text'] = test_df['text'].map(lambda com : clean_text(com))

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in map(self, arg, na_action)
   4698         dtype: object
   4699         """
-> 4700         new_values = self._map_values(arg, na_action=na_action)
   4701         return self._constructor(new_values, index=self.index, copy=False).__finalize__(
   4702             self, method="map"

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _map_values(self, mapper, na_action, convert)
    919             return arr.map(mapper, na_action=na_action)
    920 
--> 921         return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
    922 
    923     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/algorithms.py in map_array(arr, mapper, na_action, convert)
   1741     values = arr.astype(object, copy=False)
   1742     if na_action is None:
-> 1743         return lib.map_infer(values, mapper, convert=convert)
   1744     else:
   1745         return lib.map_infer_mask(

lib.pyx in pandas._libs.lib.map_infer()

/tmp/ipykernel_11/2859460325.py in <lambda>(com)
      1 # clean test_df['text']
----> 2 test_df['text'] = test_df['text'].map(lambda com : clean_text(com))

/tmp/ipykernel_11/551245950.py in clean_text(text)
      2 def clean_text(text):
      3     text = text.lower()
----> 4     text = re.sub('\W', ' ', text)
      5     return text

NameError: name 're' is not defined

## === cell 11
X = train_df['text']
y = train_df[['author_EAP', 'author_HPL', 'author_MWS']]


## === cell 12
from sklearn.model_selection import train_test_split
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=123)
print(X_train.shape, y_train.shape, X_dev.shape, y_dev.shape)


## === cell 13
print(y_train.sum(axis=0),'\n', y_dev.sum(axis=0))


## === cell 14
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vect = CountVectorizer()
vect


## === cell 15
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm = X_train_dtm.toarray()
X_train_dtm


## === cell 16
print(X_train_dtm.shape)


## === cell 17
X_dev_dtm = vect.transform(X_dev)
X_dev_dtm = X_dev_dtm.toarray()
X_dev_dtm


## === cell 18
print(X_train_dtm.shape, y_train.shape)
print(X_dev_dtm.shape, y_dev.shape)


## === cell 19
num_input_shape = X_train_dtm.shape[1]
num_class = y_train.shape[1]


## === cell 20
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(num_input_shape,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_class, activation='softmax'))


## === cell 21
model.summary()


## === cell 22
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
             metrics=['accuracy'])


## === cell 23
history = model.fit(X_train_dtm, y_train, epochs=20, batch_size=512,
                    validation_data=(X_dev_dtm, y_dev))


## === cell 24
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


## === cell 25
plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2859085711.py in <cell line: 0>()
      1 plt.clf()
----> 2 acc = history.history['acc']
      3 val_acc = history.history['val_acc']
      4 epochs = range(1, len(acc)+1)
      5 plt.plot(epochs, acc, 'bo', label='Training Accuracy')

KeyError: 'acc'

## === cell 26
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(num_input_shape,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_class, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
             metrics=['accuracy'])

model.fit(X_train_dtm, y_train, epochs=3, batch_size=512,
          validation_data=(X_dev_dtm, y_dev))


## === cell 27
results = model.evaluate(X_dev_dtm, y_dev)
print(results)


## === cell 28
test = test_df['text']
test_dtm = vect.transform(test)
test_dtm = test_dtm.toarray()
test_dtm


## === cell 29
print(test_dtm.shape)


## === cell 30
dnn_predictions = model.predict(test_dtm)
print(dnn_predictions.shape)


## === cell 31
print(dnn_predictions[:10])


## === cell 32
result = pd.DataFrame(dnn_predictions, columns=['EAP','HPL','MWS'])
result.insert(0, 'id', test_df['id'])
result.head()


## === cell 33
result.to_csv('rhodium_submission_17.csv', index=False, float_format='%.20f')
