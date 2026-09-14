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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.60769

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import random
import pandas as pd
import os
print(os.listdir("../input"))


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_path = os.path.join("../input", 'train.tsv')
test_data_path = os.path.join("../input", 'test.tsv')
data = pd.read_csv(data_path, sep='\t')
test_data = pd.read_csv(test_data_path, sep='\t')
data.describe()


## === cell 2
data.head()


## === cell 3
from sklearn.model_selection import train_test_split
train_texts = data['Phrase']
train_labels = np.array(data['Sentiment'])


test_texts = test_data['Phrase']
X_train, X_test, y_train, y_test = train_test_split(train_texts, train_labels, test_size=0.33, random_state=42)
y_temp = y_train
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

data = ((X_train, np.array(y_train)),(X_test, np.array(y_test)))


## === cell 4
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

NGRAM_RANGE = (1,3)
TOP_K = 9000
TOKEN_MODE = 'word'
MIN_DOCUMENT_FREQUENCY = 5


## === cell 5
def ngram_vectorize(train_texts, train_labels, val_texts, test_texts):
    """Vectorizes texts as n-gram vectors.
    # Arguments
        train_texts: list, training text strings.
        train_labels: np.ndarray, training labels.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val: vectorized training and validation texts
    """
    
    stop_words_lst = text.ENGLISH_STOP_WORDS.union(["\'s"])
    kwargs = {
            'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,  # Split text into word tokens.
            'min_df': MIN_DOCUMENT_FREQUENCY,
            'stop_words' : stop_words_lst
    }
    vectorizer = TfidfVectorizer(**kwargs)

    x_train = vectorizer.fit_transform(train_texts)

    x_val = vectorizer.transform(val_texts)
    
    x_test = vectorizer.transform(test_texts)
    
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, y_temp)
    
    
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    x_test = selector.transform(x_test).astype('float32')
    return x_train, x_val, x_test


## === cell 6
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout


## === cell 7
def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    """Creates an instance of a multi-layer perceptron model.

    # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.

    # Returns
        An MLP model instance.
    """
    print('input shape : ', input_shape)
    model = models.Sequential()
    model.add(Dropout(dropout_rate, input_shape=input_shape))
    i=0
    for _ in range(layers-1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))
        pass
    
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
        
    model.add(Dense(units=num_classes, activation='softmax', name='d2'))
    return model


## === cell 8
def train_ngram_model(X_train, 
                      X_label,
                      X_validate,
                      val_label,
                      learning_rate=1e-3,
                      epochs=50,
                      batch_size=150,
                      layers=3,
                      units=128,
                      dropout_rate=0.2):
    """Trains n-gram model on the given dataset.

    # Arguments
        data: tuples of training and test texts and labels.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.
    """
    num_classes = 5
    
    x_train, x_val = X_train, X_validate
    val_labels = val_label
    model = mlp_model(layers=layers,
                      units=units,
                      dropout_rate=dropout_rate,
                      input_shape=x_train.shape[1:],
                      num_classes=num_classes)
    loss = 'categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
   
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=6)]
    history = model.fit(
            x_train,
            X_label,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(x_val, val_labels),
            verbose=2,  # Logs once per epoch.
            batch_size=batch_size)
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))
    
    return model


## === cell 9
(train_texts, train_labels), (val_texts, val_labels) = data
x_train, x_val, x_test = ngram_vectorize(train_texts, train_labels, val_texts, test_texts)
model = train_ngram_model(x_train, train_labels, x_val, val_labels)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
InvalidParameterError                     Traceback (most recent call last)
/tmp/ipykernel_11/334833132.py in <cell line: 0>()
      1 (train_texts, train_labels), (val_texts, val_labels) = data
----> 2 x_train, x_val, x_test = ngram_vectorize(train_texts, train_labels, val_texts, test_texts)
      3 model = train_ngram_model(x_train, train_labels, x_val, val_labels)

/tmp/ipykernel_11/2982474463.py in ngram_vectorize(train_texts, train_labels, val_texts, test_texts)
     25 
     26     # Learn vocabulary from training texts and vectorize training texts.
---> 27     x_train = vectorizer.fit_transform(train_texts)
     28 
     29     # Vectorize validation texts.

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in fit_transform(self, raw_documents, y)
   2131             sublinear_tf=self.sublinear_tf,
   2132         )
-> 2133         X = super().fit_transform(raw_documents)
   2134         self._tfidf.fit(X)
   2135         # X is already a transformed view of raw_documents so

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in fit_transform(self, raw_documents, y)
   1367             )
   1368 
-> 1369         self._validate_params()
   1370         self._validate_ngram_range()
   1371         self._warn_for_unused_params()

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_params(self)
    598         accepted constraints.
    599         """
--> 600         validate_parameter_constraints(
    601             self._parameter_constraints,
    602             self.get_params(deep=False),

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_param_validation.py in validate_parameter_constraints(parameter_constraints, params, caller_name)
     95                 )
     96 
---> 97             raise InvalidParameterError(
     98                 f"The {param_name!r} parameter of {caller_name} must be"
     99                 f" {constraints_str}. Got {param_val!r} instead."

InvalidParameterError: The 'stop_words' parameter of TfidfVectorizer must be a str among {'english'}, an instance of 'list' or None. Got frozenset({'seemed', 'do', 'moreover', 'ever', 'itself', 'least', 'such', 'or', 'anyway', 'afterwards', 'whose', 'too', 'each', 'am', 'down', 'also', 'i', 'under', 'toward', 'found', 'whereby', 'fifty', 'since', 'below', 'made', 'hence', 'ten', 'back', 'etc', 'whither', 'along', 'detail', 'through', 'him', 'might', 'put', 'else', 'they', 'nevertheless', 'them', 'themselves', 'whatever', 'alone', 'few', 'interest', 'to', 'another', 'its', 'can', 'yourselves', 'will', 'me', 'nine', 'throughout', 'same', 'still', 'latterly', 'the', 'besides', 'against', 'above', 'all', 'thru', 'thence', 'whereupon', 'see', 'become', 'third', 'sometimes', 'anyhow', 'until', 'into', 'nobody', 'latter', 'either', 'sixty', 'within', 'had', 'are', 'eight', 'again', 'you', 'whereas', 'never', 'anyone', 'elsewhere', 'yet', 'show', 'much', 'empty', 'take', 'hereupon', 'perhaps', 're', 'often', 'fire', 'whom', 'less', 'bottom', 'at', 'many', 'name', 'somehow', 'among', 'this', 'four', 'nowhere', 'upon', 'almost', 'next', 'now', 'amongst', 'namely', 'more', 'ours', 'herself', 'seems', 'from', 'hasnt', 'their', 'my', 'out', 'noone', 'very', 'sincere', 'former', 'amoungst', 'neither', 'became', 'even', 'beforehand', 'becoming', 'therefore', 'wherever', 'for', 'up', 'was', 'thereafter', 'others', 'his', 'hereby', 'where', 'top', 'further', 'not', 'six', 'wherein', 'were', 'fill', 'every', 'of', 'formerly', 'last', 'between', 'everyone', 'which', 'most', 'there', 'by', 'whoever', 'a', 'during', 'front', 'anything', 'as', 'whereafter', 'that', 'everywhere', 'but', 'mill', 'whole', 'give', 'meanwhile', 'someone', 'what', 'however', 'once', 'whether', 'rather', 'somewhere', 'without', 'find', 'seeming', 'only', 'mine', 'than', 'sometime', 'one', 'none', 'thereby', 'whenever', 'your', 'and', 'then', 'onto', 'across', 'before', 'herein', 'yours', 'yourself', 'may', "'s", 'towards', 'ie', 'around', 'amount', 'therein', 'over', 'an', 'de', 'after', 'nor', 'nothing', 'any', 'forty', 'us', 'so', 'hundred', 'these', 'myself', 'seem', 'already', 'always', 'have', 'we', 'although', 'five', 'fifteen', 'could', 'go', 'per', 'side', 'it', 'been', 'enough', 'if', 'full', 'well', 'be', 'ourselves', 'system', 'bill', 'who', 'twelve', 'con', 'beyond', 'she', 'call', 'no', 'three', 'cannot', 'should', 'done', 'would', 'has', 'why', 'other', 'anywhere', 'ltd', 'move', 'cry', 'is', 'when', 'thus', 'on', 'everything', 'hereafter', 'beside', 'mostly', 'here', 'otherwise', 'co', 'eleven', 'two', 'eg', 'in', 'part', 'off', 'those', 'both', 'together', 'please', 'thick', 'via', 'un', 'thin', 'due', 'own', 'cant', 'her', 'serious', 'get', 'indeed', 'twenty', 'some', 'must', 'being', 'whence', 'our', 'inc', 'with', 'behind', 'first', 'while', 'except', 'because', 'becomes', 'though', 'something', 'keep', 'he', 'about', 'thereupon', 'describe', 'hers', 'several', 'couldnt', 'himself', 'how'}) instead.

## === cell 10
y_test = model.predict(x_test)
y_class = np.argmax(y_test, axis=1)

my_submission = pd.DataFrame({'PhraseId': test_data.PhraseId, 'Sentiment': y_class})
my_submission.to_csv('submission.csv', index=False)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1826829633.py in <cell line: 0>()
      1 #make prediction
----> 2 y_test = model.predict(x_test)
      3 y_class = np.argmax(y_test, axis=1)
      4 
      5 #write output

NameError: name 'model' is not defined
