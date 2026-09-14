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

import pandas as pd

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
import numpy as np

from sklearn.model_selection import train_test_split

train_texts = data["Phrase"]
train_labels = np.array(data["Sentiment"])


test_texts = test_data["Phrase"]
X_train, X_test, y_train, y_test = train_test_split(
    train_texts, train_labels, test_size=0.33, random_state=42
)
y_temp = y_train
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

data = ((X_train, np.array(y_train)), (X_test, np.array(y_test)))


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
import os
from importlib import reload

try:
    import google.protobuf as _gp

    _pb_major = int(_gp.__version__.split(".")[0])
except Exception:
    _pb_major = None

if _pb_major is not None and _pb_major >= 6:
    import sys
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==5.28.3"]
    )
    reload(_gp)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout


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
[0;31m---------------------------------------------------------------------------[0m
[0;31mInvalidParameterError[0m                     Traceback (most recent call last)
[0;32m/tmp/ipykernel_12/334833132.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;34m([0m[0mtrain_texts[0m[0;34m,[0m [0mtrain_labels[0m[0;34m)[0m[0;34m,[0m [0;34m([0m[0mval_texts[0m[0;34m,[0m [0mval_labels[0m[0;34m)[0m [0;34m=[0m [0mdata[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0mx_train[0m[0;34m,[0m [0mx_val[0m[0;34m,[0m [0mx_test[0m [0;34m=[0m [0mngram_vectorize[0m[0;34m([0m[0mtrain_texts[0m[0;34m,[0m [0mtrain_labels[0m[0;34m,[0m [0mval_texts[0m[0;34m,[0m [0mtest_texts[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0mmodel[0m [0;34m=[0m [0mtrain_ngram_model[0m[0;34m([0m[0mx_train[0m[0;34m,[0m [0mtrain_labels[0m[0;34m,[0m [0mx_val[0m[0;34m,[0m [0mval_labels[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_12/2982474463.py[0m in [0;36mngram_vectorize[0;34m(train_texts, train_labels, val_texts, test_texts)[0m
[1;32m     25[0m [0;34m[0m[0m
[1;32m     26[0m     [0;31m# Learn vocabulary from training texts and vectorize training texts.[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 27[0;31m     [0mx_train[0m [0;34m=[0m [0mvectorizer[0m[0;34m.[0m[0mfit_transform[0m[0;34m([0m[0mtrain_texts[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     28[0m [0;34m[0m[0m
[1;32m     29[0m     [0;31m# Vectorize validation texts.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py[0m in [0;36mfit_transform[0;34m(self, raw_documents, y)[0m
[1;32m   2131[0m             [0msublinear_tf[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0msublinear_tf[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2132[0m         )
[0;32m-> 2133[0;31m         [0mX[0m [0;34m=[0m [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mfit_transform[0m[0;34m([0m[0mraw_documents[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2134[0m         [0mself[0m[0;34m.[0m[0m_tfidf[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2135[0m         [0;31m# X is already a transformed view of raw_documents so[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py[0m in [0;36mfit_transform[0;34m(self, raw_documents, y)[0m
[1;32m   1367[0m             )
[1;32m   1368[0m [0;34m[0m[0m
[0;32m-> 1369[0;31m         [0mself[0m[0;34m.[0m[0m_validate_params[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1370[0m         [0mself[0m[0;34m.[0m[0m_validate_ngram_range[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1371[0m         [0mself[0m[0;34m.[0m[0m_warn_for_unused_params[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/base.py[0m in [0;36m_validate_params[0;34m(self)[0m
[1;32m    598[0m         [0maccepted[0m [0mconstraints[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    599[0m         """
[0;32m--> 600[0;31m         validate_parameter_constraints(
[0m[1;32m    601[0m             [0mself[0m[0;34m.[0m[0m_parameter_constraints[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    602[0m             [0mself[0m[0;34m.[0m[0mget_params[0m[0;34m([0m[0mdeep[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/_param_validation.py[0m in [0;36mvalidate_parameter_constraints[0;34m(parameter_constraints, params, caller_name)[0m
[1;32m     95[0m                 )
[1;32m     96[0m [0;34m[0m[0m
[0;32m---> 97[0;31m             raise InvalidParameterError(
[0m[1;32m     98[0m                 [0;34mf"The {param_name!r} parameter of {caller_name} must be"[0m[0;34m[0m[0;34m[0m[0m
[1;32m     99[0m                 [0;34mf" {constraints_str}. Got {param_val!r} instead."[0m[0;34m[0m[0;34m[0m[0m

[0;31mInvalidParameterError[0m: The 'stop_words' parameter of TfidfVectorizer must be a str among {'english'}, an instance of 'list' or None. Got frozenset({'everywhere', 'i', 'themselves', 'across', 'eg', 'nowhere', 'only', 'since', 'it', 'describe', 'seem', 'other', 'none', 'first', 'whose', 'because', 'onto', 'couldnt', 'everything', 'own', 'such', 'anyway', 'itself', 'eleven', 'yourselves', 'yet', 'within', 'some', 'mill', 'have', 'another', 'while', 'ever', 'whereupon', 'whereas', 'where', 'almost', 'among', 'seemed', 'sometimes', 'becomes', 'throughout', 'whence', 'hasnt', 'nobody', 'anything', 'whenever', 'somewhere', 'cry', 'get', 'yours', 'there', 'six', 'together', 'back', 'sixty', 'former', 'go', 'around', 'about', 'often', 'take', 'therefore', 'might', 'whom', 'wherein', 'with', 'has', 'full', 'three', 'as', 'though', 'both', 'least', 'beforehand', 'thereafter', 'fire', 'amount', 'hers', 'him', 'give', 'now', 'sincere', 'name', 'perhaps', 'alone', 'over', 'these', 'those', 'further', 'bill', 'five', 'what', 'nevertheless', 'except', 're', 'and', 'either', 'con', 'although', 'be', 'rather', 'ten', 'if', 'who', 'may', 'he', 'anywhere', 'can', 'down', 'we', 'must', 'next', 'through', 'well', 'interest', 'everyone', 'could', 'co', 'thus', 'had', 'our', 'beyond', 'nothing', 'system', 'thereby', 'every', 'are', 'at', 'whereby', 'this', 'so', 'un', 'found', 'however', 'a', 'how', 'becoming', 'them', 'in', 'most', 'put', 'someone', 'that', 'whither', 'nor', 'amoungst', 'please', 'made', 'below', 'his', 'thin', 'less', 'even', 'hereafter', 'side', 'herein', 'latter', 'seems', 'show', 'ie', 'few', 'moreover', 'to', 'latterly', 'against', 'again', 'due', 'via', 'somehow', 'whether', 'many', 'already', 'would', 'too', 'mine', 'noone', 'same', 'its', 'she', 'been', 'serious', 'were', 'herself', 'after', 'is', 'hereby', 'else', 'will', 'bottom', 'towards', 'whatever', 'us', 'amongst', 'being', 'more', 'the', 'something', 'from', 'any', "'s", 'namely', 'others', 'her', 'along', 'four', 'they', 'much', 'nine', 'hence', 'their', 'top', 'each', 'behind', 'etc', 'of', 'detail', 'anyone', 'hundred', 'himself', 'cant', 'upon', 'keep', 'therein', 'eight', 'find', 'fifteen', 'or', 'do', 'all', 'enough', 'inc', 'during', 'was', 'yourself', 'up', 'ourselves', 'never', 'done', 'but', 'front', 'twenty', 'toward', 'mostly', 'hereupon', 'whereafter', 'very', 'forty', 'here', 'then', 'one', 'which', 'an', 'no', 'per', 'fill', 'before', 'ltd', 'out', 'formerly', 'become', 'meanwhile', 'call', 'whoever', 'not', 'move', 'should', 'also', 'whole', 'otherwise', 'without', 'above', 'always', 'into', 'your', 'wherever', 'several', 'my', 'once', 'empty', 'myself', 'when', 'indeed', 'twelve', 'thru', 'why', 'thence', 'part', 'between', 'for', 'thereupon', 'cannot', 'by', 'third', 'under', 'until', 'de', 'neither', 'thick', 'see', 'fifty', 'than', 'became', 'last', 'me', 'am', 'elsewhere', 'ours', 'anyhow', 'off', 'you', 'beside', 'seeming', 'two', 'afterwards', 'on', 'besides', 'still', 'sometime'}) instead.

## === cell 10
y_test = model.predict(x_test)
y_class = np.argmax(y_test, axis=1)

my_submission = pd.DataFrame({'PhraseId': test_data.PhraseId, 'Sentiment': y_class})
my_submission.to_csv('submission.csv', index=False)
