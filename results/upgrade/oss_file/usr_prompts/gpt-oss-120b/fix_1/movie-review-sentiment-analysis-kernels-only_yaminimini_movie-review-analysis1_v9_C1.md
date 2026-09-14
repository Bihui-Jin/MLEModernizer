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

0.16143

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
import pandas as pd


## === cell 2
train = pd.read_csv("../input/train.tsv",sep = "\t")


## === cell 3
train.head()


## === cell 4
test = pd.read_csv("../input/test.tsv",sep = "\t")


## === cell 5
test.head()


## === cell 6
train["Sentiment"].unique()


## === cell 7
train.shape


## === cell 8
test.shape


## === cell 9
train.isnull().sum(axis=0)


## === cell 10
test.isnull().sum(axis = 0)


## === cell 11
train["SentenceId"].value_counts()[0:5]


## === cell 12
test["SentenceId"].value_counts()[0:5]


## === cell 13
len(train["SentenceId"].unique())+len(test["SentenceId"].unique())


## === cell 14
len(train["PhraseId"].unique())+len(test["PhraseId"].unique())


## === cell 15
from sklearn.feature_extraction.text import TfidfVectorizer


## === cell 16
tfidf = TfidfVectorizer(analyzer = "word", stop_words = 'english', min_df=0.01,max_df=0.9,ngram_range = (1,3))


## === cell 17
tfidf.fit(train["Phrase"])


## === cell 18
train_tfidf = tfidf.transform(train["Phrase"])


## === cell 19
X_train = train_tfidf.todense()


## === cell 20
Y_train = train["Sentiment"]


## === cell 21
from sklearn.naive_bayes import MultinomialNB


## === cell 22
NB = MultinomialNB()


## === cell 23
NB.fit(X_train,Y_train)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3740911007.py in <cell line: 0>()
----> 1 NB.fit(X_train,Y_train)

/usr/local/lib/python3.11/dist-packages/sklearn/naive_bayes.py in fit(self, X, y, sample_weight)
    747         """
    748         self._validate_params()
--> 749         X, y = self._check_X_y(X, y)
    750         _, n_features = X.shape
    751 

/usr/local/lib/python3.11/dist-packages/sklearn/naive_bayes.py in _check_X_y(self, X, y, reset)
    581     def _check_X_y(self, X, y, reset=True):
    582         """Validate X and y in fit methods."""
--> 583         return self._validate_data(X, y, accept_sparse="csr", reset=reset)
    584 
    585     def _update_class_log_prior(self, class_prior=None):

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    582                 y = check_array(y, input_name="y", **check_y_params)
    583             else:
--> 584                 X, y = check_X_y(X, y, **check_params)
    585             out = X, y
    586 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1104         )
   1105 
-> 1106     X = check_array(
   1107         X,
   1108         accept_sparse=accept_sparse,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    735     """
    736     if isinstance(array, np.matrix):
--> 737         raise TypeError(
    738             "np.matrix is not supported. Please convert to a numpy array with "
    739             "np.asarray. For more information see: "

TypeError: np.matrix is not supported. Please convert to a numpy array with np.asarray. For more information see: https://numpy.org/doc/stable/reference/generated/numpy.matrix.html

## === cell 24
test_tfidf = tfidf.transform(test["Phrase"])


## === cell 25
x_test = test_tfidf.todense()


## === cell 26
x_test.shape


## === cell 27
y_pred = NB.predict(x_test)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/1506995008.py in <cell line: 0>()
----> 1 y_pred = NB.predict(x_test)

/usr/local/lib/python3.11/dist-packages/sklearn/naive_bayes.py in predict(self, X)
    102             Predicted target values for X.
    103         """
--> 104         check_is_fitted(self)
    105         X = self._check_X(X)
    106         jll = self._joint_log_likelihood(X)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This MultinomialNB instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

## === cell 28
type(y_pred)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3438068662.py in <cell line: 0>()
----> 1 type(y_pred)

NameError: name 'y_pred' is not defined

## === cell 29
y_pred_df = pd.DataFrame(y_pred, columns = ["Sentiment"])


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4114412485.py in <cell line: 0>()
----> 1 y_pred_df = pd.DataFrame(y_pred, columns = ["Sentiment"])

NameError: name 'y_pred' is not defined

## === cell 30
y_pred_df.head()


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2435765971.py in <cell line: 0>()
----> 1 y_pred_df.head()

NameError: name 'y_pred_df' is not defined

## === cell 31
sub = pd.concat([test["PhraseId"],y_pred_df],axis = 1)


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1266575250.py in <cell line: 0>()
----> 1 sub = pd.concat([test["PhraseId"],y_pred_df],axis = 1)

NameError: name 'y_pred_df' is not defined

## === cell 32
sub.head()


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1894231914.py in <cell line: 0>()
----> 1 sub.head()

NameError: name 'sub' is not defined

## === cell 34
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer


## === cell 35
pattern = RegexpTokenizer(r'[a-zA-Z0-9]+')


## === cell 36
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = pattern.tokenize)


## === cell 37
cv.fit(train['Phrase'])


## === cell 38
train_cv = cv.transform(train["Phrase"])


## === cell 39
train_cv


## === cell 40
X_train2 = train_cv.todense()


## === cell 41
Y_train2 = train["Sentiment"]


## === cell 42
from sklearn.linear_model import SGDClassifier
sv = SGDClassifier()


## === cell 44
from sklearn.linear_model import SGDClassifier


## === cell 45
sv = SGDClassifier(max_iter = 200)


## === cell 46
sv.fit(X_train,Y_train)


## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3997544488.py in <cell line: 0>()
----> 1 sv.fit(X_train,Y_train)

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_stochastic_gradient.py in fit(self, X, y, coef_init, intercept_init, sample_weight)
    892         self._more_validate_params()
    893 
--> 894         return self._fit(
    895             X,
    896             y,

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_stochastic_gradient.py in _fit(self, X, y, alpha, C, loss, learning_rate, coef_init, intercept_init, sample_weight)
    681         self.t_ = 1.0
    682 
--> 683         self._partial_fit(
    684             X,
    685             y,

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_stochastic_gradient.py in _partial_fit(self, X, y, alpha, C, loss, learning_rate, max_iter, classes, sample_weight, coef_init, intercept_init)
    577     ):
    578         first_call = not hasattr(self, "classes_")
--> 579         X, y = self._validate_data(
    580             X,
    581             y,

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    582                 y = check_array(y, input_name="y", **check_y_params)
    583             else:
--> 584                 X, y = check_X_y(X, y, **check_params)
    585             out = X, y
    586 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1104         )
   1105 
-> 1106     X = check_array(
   1107         X,
   1108         accept_sparse=accept_sparse,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    735     """
    736     if isinstance(array, np.matrix):
--> 737         raise TypeError(
    738             "np.matrix is not supported. Please convert to a numpy array with "
    739             "np.asarray. For more information see: "

TypeError: np.matrix is not supported. Please convert to a numpy array with np.asarray. For more information see: https://numpy.org/doc/stable/reference/generated/numpy.matrix.html

## === cell 47
y_pred2 = sv.predict(x_test)


## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2495564130.py in <cell line: 0>()
----> 1 y_pred2 = sv.predict(x_test)

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in predict(self, X)
    417         """
    418         xp, _ = get_namespace(X)
--> 419         scores = self.decision_function(X)
    420         if len(scores.shape) == 1:
    421             indices = xp.astype(scores > 0, int)

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in decision_function(self, X)
    398         xp, _ = get_namespace(X)
    399 
--> 400         X = self._validate_data(X, accept_sparse="csr", reset=False)
    401         scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
    402         return xp.reshape(scores, -1) if scores.shape[1] == 1 else scores

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    563             raise ValueError("Validation should be done on X, y or both.")
    564         elif not no_val_X and no_val_y:
--> 565             X = check_array(X, input_name="X", **check_params)
    566             out = X
    567         elif no_val_X and not no_val_y:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    735     """
    736     if isinstance(array, np.matrix):
--> 737         raise TypeError(
    738             "np.matrix is not supported. Please convert to a numpy array with "
    739             "np.asarray. For more information see: "

TypeError: np.matrix is not supported. Please convert to a numpy array with np.asarray. For more information see: https://numpy.org/doc/stable/reference/generated/numpy.matrix.html

## === cell 48
y_pred2_df = pd.DataFrame(y_pred2, columns = ["Sentiment"])


## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2311093467.py in <cell line: 0>()
----> 1 y_pred2_df = pd.DataFrame(y_pred2, columns = ["Sentiment"])

NameError: name 'y_pred2' is not defined

## === cell 49
sub2 = pd.concat([test["PhraseId"],y_pred2_df],axis = 1)


## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/437748794.py in <cell line: 0>()
----> 1 sub2 = pd.concat([test["PhraseId"],y_pred2_df],axis = 1)

NameError: name 'y_pred2_df' is not defined

## === cell 50
sub2.head()


## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4239298052.py in <cell line: 0>()
----> 1 sub2.head()

NameError: name 'sub2' is not defined

## === cell 52
from keras.preprocessing.text import Tokenizer


## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 53
X_train = train['Phrase']


## === cell 54
train.dtypes


## === cell 55
from keras.utils import to_categorical


## === cell 56
Y_train = to_categorical(train['Sentiment'].values)


## === cell 57
Y_train.shape


## === cell 58
tz = Tokenizer(num_words = 10000, lower = True)


## --- ERROR in cell 58, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/857812209.py in <cell line: 0>()
----> 1 tz = Tokenizer(num_words = 10000, lower = True)

NameError: name 'Tokenizer' is not defined

## === cell 59
tz.fit_on_texts(list(X_train))


## --- ERROR in cell 59, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2594319882.py in <cell line: 0>()
----> 1 tz.fit_on_texts(list(X_train))

NameError: name 'tz' is not defined

## === cell 60
X_train2 = tz.texts_to_sequences(X_train)


## --- ERROR in cell 60, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3837596463.py in <cell line: 0>()
----> 1 X_train2 = tz.texts_to_sequences(X_train)

NameError: name 'tz' is not defined

## === cell 61
type(X_train2)


## === cell 62
len(X_train2)


## === cell 63
from keras.preprocessing.sequence import pad_sequences
X_train2 = pad_sequences(X_train2, maxlen=100)


## --- ERROR in cell 63, traceback:
---------------------------------------------------------------------------
MemoryError                               Traceback (most recent call last)
/tmp/ipykernel_11/1906347236.py in <cell line: 0>()
      1 from keras.preprocessing.sequence import pad_sequences
----> 2 X_train2 = pad_sequences(X_train2, maxlen=100)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/sequence_utils.py in pad_sequences(sequences, maxlen, dtype, padding, truncating, value)
    111         )
    112 
--> 113     x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    114     for idx, s in enumerate(sequences):
    115         if not len(s):

/usr/local/lib/python3.11/dist-packages/numpy/core/numeric.py in full(shape, fill_value, dtype, order, like)
    327         fill_value = asarray(fill_value)
    328         dtype = fill_value.dtype
--> 329     a = empty(shape, dtype, order)
    330     multiarray.copyto(a, fill_value, casting='unsafe')
    331     return a

MemoryError: Unable to allocate 609. GiB for an array with shape (109242, 100, 14961) and data type int32

## === cell 64
X.shape


## --- ERROR in cell 64, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/106882318.py in <cell line: 0>()
----> 1 X.shape

NameError: name 'X' is not defined

## === cell 65
X_test2 = tz.texts_to_sequences(test["Phrase"])


## --- ERROR in cell 65, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2194088211.py in <cell line: 0>()
----> 1 X_test2 = tz.texts_to_sequences(test["Phrase"])

NameError: name 'tz' is not defined

## === cell 66
X_test2 = pad_sequences(X_test2, maxlen=100)


## --- ERROR in cell 66, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2552397327.py in <cell line: 0>()
----> 1 X_test2 = pad_sequences(X_test2, maxlen=100)

NameError: name 'X_test2' is not defined

## === cell 67
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y_train, test_size=0.20, random_state=seed)


## --- ERROR in cell 67, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1950950017.py in <cell line: 0>()
      1 from sklearn.model_selection import train_test_split
----> 2 X_train, X_val, Y_train, Y_val = train_test_split(X, Y_train, test_size=0.20, random_state=seed)

NameError: name 'X' is not defined

## === cell 68
from keras.layers import Dense,Dropout,Embedding,LSTM
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Sequential


## === cell 69
model = Sequential()
model.add(Embedding(10000,100,mask_zero=True))
model.add(LSTM(64,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
model.add(LSTM(32,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
model.add(Dense(5,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])


## --- ERROR in cell 69, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1902512989.py in <cell line: 0>()
      4 model.add(LSTM(32,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
      5 model.add(Dense(5,activation='softmax'))
----> 6 model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/adam.py in __init__(self, learning_rate, beta_1, beta_2, epsilon, amsgrad, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)
     60         **kwargs,
     61     ):
---> 62         super().__init__(
     63             learning_rate=learning_rate,
     64             name=name,

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/optimizer.py in __init__(self, *args, **kwargs)
     19 class TFOptimizer(KerasAutoTrackable, base_optimizer.BaseOptimizer):
     20     def __init__(self, *args, **kwargs):
---> 21         super().__init__(*args, **kwargs)
     22         self._distribution_strategy = tf.distribute.get_strategy()
     23 

/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/base_optimizer.py in __init__(self, learning_rate, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)
     88             )
     89         if kwargs:
---> 90             raise ValueError(f"Argument(s) not recognized: {kwargs}")
     91 
     92         if name is None:

ValueError: Argument(s) not recognized: {'lr': 0.001}

## === cell 70
model.fit(X_train, Y_train,validation_data = (X_val,Y_val), epochs = 4, batch_size = 32)


## --- ERROR in cell 70, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3763922595.py in <cell line: 0>()
----> 1 model.fit(X_train, Y_train,validation_data = (X_val,Y_val), epochs = 4, batch_size = 32)

NameError: name 'X_val' is not defined

## === cell 71
sub2['Sentiment'] = model.predict_classes(X_test2, batch_size=32, verbose=1)


## --- ERROR in cell 71, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4233296893.py in <cell line: 0>()
----> 1 sub2['Sentiment'] = model.predict_classes(X_test2, batch_size=32, verbose=1)

AttributeError: 'Sequential' object has no attribute 'predict_classes'

## === cell 72
sub2.head()


## --- ERROR in cell 72, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4239298052.py in <cell line: 0>()
----> 1 sub2.head()

NameError: name 'sub2' is not defined

## === cell 73
sub2.to_csv('submission3.csv', index=False)


## --- ERROR in cell 73, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1874671330.py in <cell line: 0>()
----> 1 sub2.to_csv('submission3.csv', index=False)

NameError: name 'sub2' is not defined
