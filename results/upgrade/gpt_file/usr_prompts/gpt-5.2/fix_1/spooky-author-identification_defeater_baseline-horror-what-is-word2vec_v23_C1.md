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

3.6

# 3. Installed packages

gensim==4.4.0
geopandas==0.14.4
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
xgboost==2.0.3

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

1.09129

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd 
from subprocess import check_output
from gensim.models import Word2Vec

from nltk.tokenize import RegexpTokenizer
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline

import xgboost as xgb

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

alpha_tokenizer = RegexpTokenizer('[A-Za-z]\w+')
lemmatizer = WordNetLemmatizer()
stop = stopwords.words('english')


## === cell 1
from sklearn.model_selection import train_test_split

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

author_mapping = {'EAP':0, 'HPL':1, 'MWS':2}
y_train = train['author'].map(author_mapping).values


## === cell 2
data = [[lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(sent) if word.lower() not in stop] for sent in train.text.values]


## === cell 3
vectorizers = [ ('3-gram TF-IDF Vectorizer on words', TfidfVectorizer(ngram_range=(1, 3), analyzer='word', binary=False)),
                ('3-gram Count Vectorizer on words', CountVectorizer(ngram_range=(1, 3), analyzer='word', binary=False)),
                ('3-gram Hashing Vectorizer on words', HashingVectorizer(ngram_range=(1, 5), analyzer='word', binary=False)),
                ('TF-IDF + SVD', Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 3), analyzer='word', binary=False)),
                                 ('svd', TruncatedSVD(n_components=150)),
                                ])),
                ('TF-IDF + SVD + Normalizer', Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 3), analyzer='word', binary=False)),
                                 ('svd', TruncatedSVD(n_components=150)),
                                 ('norm', Normalizer()),
                                ]))
              ]


## === cell 4
estimators = [(KNeighborsClassifier(n_neighbors=3), 'K-Nearest Neighbors', 'yellow'),
              (SVC(C=1, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear', max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False), 'Support Vector Machine', 'red'),
              (LogisticRegression(tol=1e-8, penalty='l2', C=0.1), 'Logistic Regression', 'green'),
              (MultinomialNB(), 'Naive Bayes', 'magenta'),
              (RandomForestClassifier(n_estimators=10, criterion='gini'), 'Random Forest', 'gray'),
              (None, 'XGBoost', 'pink')
]


## === cell 5
params = {}
params['objective'] = 'multi:softprob'
params['eta'] = 0.1
params['max_depth'] = 3
params['silent'] = 1
params['num_class'] = 3
params['eval_metric'] = 'mlogloss'
params['min_child_weight'] = 1
params['subsample'] = 0.8
params['colsample_bytree'] = 0.3
params['seed'] = 0


## === cell 6
y_train, y_test = train_test_split(train, test_size=0.3)
y_train = y_train['author'].map(author_mapping).values
y_test = y_test['author'].map(author_mapping).values

def compare():
    for vectorizer in vectorizers:
        print(vectorizer[0] + '\n')
        X = vectorizer[1].fit_transform(train.text.values)
        X_train, X_test = train_test_split(X, test_size=0.3)
        for estimator in estimators:
            if estimator[1] == 'XGBoost': 
                xgtrain = xgb.DMatrix(X_train, y_train)
                xgtest = xgb.DMatrix(X_test)
                model = xgb.train(params=list(params.items()), dtrain=xgtrain,  num_boost_round=40)
                predictions = model.predict(xgtest, ntree_limit=model.best_ntree_limit).argmax(axis=1)
            else:
                estimator[0].fit(X_train, y_train)
                predictions = estimator[0].predict(X_test)
            print(accuracy_score(predictions, y_test), estimator[1])


## === cell 7
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

y_train = train['author'].map(author_mapping).values

X = vectorizers[4][1].fit_transform(np.hstack((train.text.values, test.text.values)))
X_train, X_test = train_test_split(X, test_size=8392)
xgtrain = xgb.DMatrix(X_train, y_train)
xgtest = xgb.DMatrix(X_test)
model = xgb.train(params=list(params.items()), dtrain=xgtrain,  num_boost_round=40)
probs = model.predict(xgtest, ntree_limit=model.best_ntree_limit)
author = pd.DataFrame(probs)
final = pd.DataFrame()
final['id'] = test.id
final['EAP'] = author[0]
final['HPL'] = author[1]
final['MWS'] = author[2]
final.to_csv('submission.csv', sep=',',index=False)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_11/2207396052.py in <cell line: 0>()
      6 X = vectorizers[4][1].fit_transform(np.hstack((train.text.values, test.text.values)))
      7 X_train, X_test = train_test_split(X, test_size=8392)
----> 8 xgtrain = xgb.DMatrix(X_train, y_train)
      9 xgtest = xgb.DMatrix(X_test)
     10 model = xgb.train(params=list(params.items()), dtrain=xgtrain,  num_boost_round=40)

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in __init__(self, data, label, weight, base_margin, missing, silent, feature_names, feature_types, nthread, group, qid, label_lower_bound, label_upper_bound, feature_weights, enable_categorical, data_split_mode)
    867         self.handle = handle
    868 
--> 869         self.set_info(
    870             label=label,
    871             weight=weight,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in set_info(self, label, weight, base_margin, group, qid, label_lower_bound, label_upper_bound, feature_names, feature_types, feature_weights)
    930 
    931         if label is not None:
--> 932             self.set_label(label)
    933         if weight is not None:
    934             self.set_weight(weight)

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in set_label(self, label)
   1068         from .data import dispatch_meta_backend
   1069 
-> 1070         dispatch_meta_backend(self, label, "label", "float")
   1071 
   1072     def set_weight(self, weight: ArrayLike) -> None:

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in dispatch_meta_backend(matrix, data, name, dtype)
   1216         return
   1217     if _is_np_array_like(data):
-> 1218         _meta_from_numpy(data, name, dtype, handle)
   1219         return
   1220     if _is_pandas_df(data):

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _meta_from_numpy(data, field, dtype, handle)
   1157         raise ValueError("Masked array is not supported.")
   1158     interface_str = _array_interface(data)
-> 1159     _check_call(_LIB.XGDMatrixSetInfoFromInterface(handle, c_str(field), interface_str))
   1160 
   1161 

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _check_call(ret)
    280     """
    281     if ret != 0:
--> 282         raise XGBoostError(py_str(_LIB.XGBGetLastError()))
    283 
    284 

XGBoostError: [01:51:45] /workspace/src/data/data.cc:501: Check failed: this->labels.Size() % this->num_row_ == 0 (6434 vs. 0) : Incorrect size for labels.
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3588ca) [0x7fff772fc8ca]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x389af7) [0x7fff7732daf7]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x38ab51) [0x7fff7732eb51]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGDMatrixSetInfoFromInterface+0xb0) [0x7fff771023a0]
  [bt] (4) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (5) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]
  [bt] (6) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7ffff63bc4d8]
  [bt] (7) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0x9c8e) [0x7ffff63bbc8e]
  [bt] (8) /usr/bin/python3(_PyObject_MakeTpCall+0x27c) [0x52f85c]



## === cell 8
NUM_FEATURES = 150

model = Word2Vec(data, min_count=3, size=NUM_FEATURES, window=5, sg=1, alpha=1e-4, workers=4)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3113105481.py in <cell line: 0>()
      1 NUM_FEATURES = 150
      2 
----> 3 model = Word2Vec(data, min_count=3, size=NUM_FEATURES, window=5, sg=1, alpha=1e-4, workers=4)

TypeError: Word2Vec.__init__() got an unexpected keyword argument 'size'

## === cell 9
len(model.wv.vocab)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3200786322.py in <cell line: 0>()
----> 1 len(model.wv.vocab)

NameError: name 'model' is not defined

## === cell 10
model.most_similar('raven')


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3681575831.py in <cell line: 0>()
----> 1 model.most_similar('raven')

NameError: name 'model' is not defined

## === cell 11
def get_feature_vec(tokens, num_features, model):
    featureVec = np.zeros(shape=(1, num_features), dtype='float32')
    missed = 0
    for word in tokens:
        try:
            featureVec = np.add(featureVec, model[word])
        except KeyError:
            missed += 1
            pass
    if len(tokens) - missed == 0:
        return np.zeros(shape=(num_features), dtype='float32')
    return np.divide(featureVec, len(tokens) - missed).squeeze()


## === cell 12
vectors = []
for i in train.text.values:
    vectors.append(get_feature_vec([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(i) if word.lower() not in stop], NUM_FEATURES, model))


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3066316661.py in <cell line: 0>()
      1 vectors = []
      2 for i in train.text.values:
----> 3     vectors.append(get_feature_vec([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(i) if word.lower() not in stop], NUM_FEATURES, model))

NameError: name 'model' is not defined

## === cell 13
estimator = LogisticRegression(C=1)
estimator.fit(np.array(vectors), y_train);


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2043673842.py in <cell line: 0>()
      1 estimator = LogisticRegression(C=1)
----> 2 estimator.fit(np.array(vectors), y_train);

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py in fit(self, X, y, sample_weight)
   1194             _dtype = [np.float64, np.float32]
   1195 
-> 1196         X, y = self._validate_data(
   1197             X,
   1198             y,

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
    900             # If input is 1D raise error
    901             if array.ndim == 1:
--> 902                 raise ValueError(
    903                     "Expected 2D array, got 1D array instead:\narray={}.\n"
    904                     "Reshape your data either using array.reshape(-1, 1) if "

ValueError: Expected 2D array, got 1D array instead:
array=[].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

## === cell 14
test_vectors = []
for i in test.text.values:
    test_vectors.append(get_feature_vec([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(i) if word.lower() not in stop], NUM_FEATURES, model))


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4239267019.py in <cell line: 0>()
      1 test_vectors = []
      2 for i in test.text.values:
----> 3     test_vectors.append(get_feature_vec([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(i) if word.lower() not in stop], NUM_FEATURES, model))

NameError: name 'model' is not defined

## === cell 15
probs = estimator.predict_proba(test_vectors)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/1198788033.py in <cell line: 0>()
----> 1 probs = estimator.predict_proba(test_vectors)

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py in predict_proba(self, X)
   1360             where classes are ordered as they are in ``self.classes_``.
   1361         """
-> 1362         check_is_fitted(self)
   1363 
   1364         ovr = self.multi_class in ["ovr", "warn"] or (

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This LogisticRegression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

## === cell 16
author = pd.DataFrame(probs)

final = pd.DataFrame()
final['id'] = test.id
final['EAP'] = author[0]
final['HPL'] = author[1]
final['MWS'] = author[2]


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3902640399.py in <cell line: 0>()
----> 1 author = pd.DataFrame(probs)
      2 
      3 final = pd.DataFrame()
      4 final['id'] = test.id
      5 final['EAP'] = author[0]

NameError: name 'probs' is not defined
