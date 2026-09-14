# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    from google.protobuf import message_factory as _message_factory
    from google.protobuf import symbol_database as _symbol_database

    _get_msg_cls = getattr(_message_factory, "GetMessageClass", None)

    if not hasattr(_message_factory.MessageFactory, "GetMessageClass") and callable(
        _get_msg_cls
    ):

        def _GetMessageClass(self, descriptor):
            return _get_msg_cls(descriptor)

        _message_factory.MessageFactory.GetMessageClass = _GetMessageClass

    if not hasattr(_message_factory.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(descriptor)
            if callable(_get_msg_cls):
                return _get_msg_cls(descriptor)
            raise AttributeError(
                "No GetMessageClass/GetPrototype available for protobuf MessageFactory"
            )

        _message_factory.MessageFactory.GetPrototype = _GetPrototype

    if not hasattr(_symbol_database.SymbolDatabase, "GetPrototype"):

        def _SDB_GetPrototype(self, descriptor):
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(descriptor)
            if callable(_get_msg_cls):
                return _get_msg_cls(descriptor)
            raise AttributeError(
                "No GetMessageClass/GetPrototype available for protobuf SymbolDatabase"
            )

        _symbol_database.SymbolDatabase.GetPrototype = _SDB_GetPrototype
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import pandas as pd
import os

print(os.listdir("../input"))


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
data = ((X_train, np.array(y_train)),(X_test, np.array(y_test)))


## === cell 4
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

NGRAM_RANGE = (1,3)
TOP_K = 5000
TOKEN_MODE = 'word'
MIN_DOCUMENT_FREQUENCY = 2


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
    
    kwargs = {
            'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,  # Split text into word tokens.
            'min_df': MIN_DOCUMENT_FREQUENCY,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    x_train = vectorizer.fit_transform(train_texts)

    x_val = vectorizer.transform(val_texts)
    
    x_test = vectorizer.transform(test_texts)
    
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
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
    op_units, op_activation = 5, 'sigmoid'
    model = models.Sequential()
    model.add(Dropout(dropout_rate, input_shape=input_shape))
    
    for _ in range(layers-1):
        model.add(Dense(units=units, activation='relu', name='d1'))
        model.add(Dropout(rate=dropout_rate))
        
    model.add(Dense(units=op_units, activation='softmax', name='d2'))
    return model


## === cell 8
def train_ngram_model(X_train, X_validate,
                      learning_rate=1e-3,
                      epochs=20,
                      batch_size=128,
                      layers=2,
                      units=64,
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
    model = mlp_model(layers=layers,
                      units=units,
                      dropout_rate=dropout_rate,
                      input_shape=x_train.shape[1:],
                      num_classes=num_classes)
    loss = 'sparse_categorical_crossentropy' #binary_crossentropy
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
   
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]
    history = model.fit(
            x_train,
            train_labels,
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
def train_ngram_model(
    X_train,
    X_validate,
    learning_rate=1e-3,
    epochs=20,
    batch_size=128,
    layers=2,
    units=64,
    dropout_rate=0.2,
):
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
    model = mlp_model(
        layers=layers,
        units=units,
        dropout_rate=dropout_rate,
        input_shape=x_train.shape[1:],
        num_classes=num_classes,
    )
    loss = "sparse_categorical_crossentropy"  # binary_crossentropy

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)]
    history = model.fit(
        x_train,
        train_labels,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(x_val, val_labels),
        verbose=2,  # Logs once per epoch.
        batch_size=batch_size,
    )
    history = history.history
    print(
        "Validation accuracy: {acc}, loss: {loss}".format(
            acc=history["val_acc"][-1], loss=history["val_loss"][-1]
        )
    )

    return model


## === cell 10
y_test = model.predict(x_test)
y_class = np.argmax(y_test, axis=1)


## --- ERROR in cell 10, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3934496306.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0my_test[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mx_test[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0my_class[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0margmax[0m[0;34m([0m[0my_test[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mNameError[0m: name 'model' is not defined

## === cell 11
my_submission = pd.DataFrame({'PhraseId': test_data.PhraseId, 'Sentiment': y_class})
my_submission.to_csv('submission.csv', index=False)
