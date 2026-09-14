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

0.5325

# 6. Current score

1.0847

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
embedding = gensim.models.Word2Vec(data, size=50, window=10, min_count=1, sg=0)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3799492301.py in <cell line: 0>()
      1 # Create Word2Vec model using CBOW (sg=0)
      2 # Set min_count to 1 so as to include all words
----> 3 embedding = gensim.models.Word2Vec(data, size=50, window=10, min_count=1, sg=0)

TypeError: Word2Vec.__init__() got an unexpected keyword argument 'size'

## === cell 8
print(embedding)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1625265400.py in <cell line: 0>()
----> 1 print(embedding)

NameError: name 'embedding' is not defined

## === cell 9
embedding.train(data,total_examples=len(data),epochs=30)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2188451566.py in <cell line: 0>()
----> 1 embedding.train(data,total_examples=len(data),epochs=30)

NameError: name 'embedding' is not defined

## === cell 10
words = list(embedding.wv.vocab)
print(len(words))


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2830714747.py in <cell line: 0>()
----> 1 words = list(embedding.wv.vocab)
      2 print(len(words))

NameError: name 'embedding' is not defined

## === cell 11
print(embedding['capered'])


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/196036746.py in <cell line: 0>()
----> 1 print(embedding['capered'])

NameError: name 'embedding' is not defined

## === cell 12
embedding.most_similar('dark', topn=5)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/417493562.py in <cell line: 0>()
----> 1 embedding.most_similar('dark', topn=5)

NameError: name 'embedding' is not defined

## === cell 13
embedding.most_similar('shocked', topn=5)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/880442372.py in <cell line: 0>()
----> 1 embedding.most_similar('shocked', topn=5)

NameError: name 'embedding' is not defined

## === cell 14
embedding.most_similar('sprang', topn=5)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2707755882.py in <cell line: 0>()
----> 1 embedding.most_similar('sprang', topn=5)

NameError: name 'embedding' is not defined

## === cell 15
embedding.most_similar('pride', topn=5)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/172385426.py in <cell line: 0>()
----> 1 embedding.most_similar('pride', topn=5)

NameError: name 'embedding' is not defined

## === cell 16
train_df['author'] = pd.Categorical(train_df['author'])
df_Dummies = pd.get_dummies(train_df['author'], prefix='author')
train_df = pd.concat([train_df, df_Dummies], axis=1)
train_df.head()


## === cell 17
X = train_df['text']
Y = train_df[['author_EAP', 'author_HPL', 'author_MWS']].values
print(X.shape, X[0], Y.shape, Y[0])


## === cell 18
X_test = test_df['text']
print(X_test.shape, X_test[0])


## === cell 19
def text_to_avg(text):
    """Given a list of words, extract the respective GloVe representations
    and average the values into a single vector encoding the text meaning."""
    avg = np.zeros((50,))
    for w in text:
        avg += embedding[w]
    avg = avg/len(text)
    return avg


## === cell 20
X_avg = np.zeros((X.shape[0], 50)) # initialize X_avg
for i in range(X.shape[0]):
    X_avg[i] = text_to_avg(X[i])


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3335996817.py in <cell line: 0>()
      1 X_avg = np.zeros((X.shape[0], 50)) # initialize X_avg
      2 for i in range(X.shape[0]):
----> 3     X_avg[i] = text_to_avg(X[i])

/tmp/ipykernel_11/3289460572.py in text_to_avg(text)
      6     # average the word vector by looping over the words in text
      7     for w in text:
----> 8         avg += embedding[w]
      9     avg = avg/len(text)
     10     return avg

NameError: name 'embedding' is not defined

## === cell 21
print(X_avg.shape)
print(X_avg[0])


## === cell 22
X_test_avg = np.zeros((X_test.shape[0], 50)) # initialize X_test_avg
for i in range(X_test.shape[0]):
    X_test_avg[i] = text_to_avg(X_test[i])


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1532690172.py in <cell line: 0>()
      1 X_test_avg = np.zeros((X_test.shape[0], 50)) # initialize X_test_avg
      2 for i in range(X_test.shape[0]):
----> 3     X_test_avg[i] = text_to_avg(X_test[i])

/tmp/ipykernel_11/3289460572.py in text_to_avg(text)
      6     # average the word vector by looping over the words in text
      7     for w in text:
----> 8         avg += embedding[w]
      9     avg = avg/len(text)
     10     return avg

NameError: name 'embedding' is not defined

## === cell 23
print(X_test_avg.shape)
print(X_test_avg[0])


## === cell 24
from sklearn.model_selection import train_test_split
X_train, X_dev, Y_train, Y_dev = train_test_split(X_avg, Y, test_size=0.2, random_state=123)
print(X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape)


## === cell 25
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(50, activation='relu', input_shape=(50,)))
model.add(layers.Dense(3, activation='softmax'))

model.summary()


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 26
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


## === cell 27
history = model.fit(X_train, Y_train, epochs=30, batch_size=128, validation_data=(X_dev, Y_dev))


## === cell 28
loss = history.history['loss']
dev_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, dev_loss, 'b', label='validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


## === cell 29
model = models.Sequential()
model.add(layers.Dense(50, activation='relu', input_shape=(50,)))
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


## === cell 30
model.fit(X_avg, Y, epochs=20, batch_size=128)


## === cell 31
preds = model.predict(X_test_avg)
print(preds.shape)
print(preds[7])


## === cell 32
pred_labels = []
for i in range(len(X_test_avg)):
    pred_label = np.argmax(preds[i])
    pred_labels.append(pred_label)


## === cell 33
print(pred_labels[7])


## === cell 34
result = pd.DataFrame(preds, columns=['EAP','HPL','MWS'])
result.insert(0, 'id', test_df['id'])
result.head()


## === cell 35
result.to_csv('submission.csv', index=False, float_format='%.20f')
