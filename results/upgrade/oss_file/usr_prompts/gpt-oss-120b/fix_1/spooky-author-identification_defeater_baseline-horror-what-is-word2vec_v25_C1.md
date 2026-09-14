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

1.05046

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
y_train_split, y_test_split = train_test_split(train, test_size=0.3)
y_train_split = y_train_split['author'].map(author_mapping).values
y_test_split = y_test_split['author'].map(author_mapping).values

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
NUM_FEATURES = 150

model = Word2Vec(data, min_count=3, size=NUM_FEATURES, window=5, sg=1, alpha=1e-4, workers=4)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3113105481.py in <cell line: 0>()
      1 NUM_FEATURES = 150
      2 
----> 3 model = Word2Vec(data, min_count=3, size=NUM_FEATURES, window=5, sg=1, alpha=1e-4, workers=4)

TypeError: Word2Vec.__init__() got an unexpected keyword argument 'size'

## === cell 8
len(model.wv.vocab)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3200786322.py in <cell line: 0>()
----> 1 len(model.wv.vocab)

NameError: name 'model' is not defined

## === cell 9
model.most_similar('raven')


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3681575831.py in <cell line: 0>()
----> 1 model.most_similar('raven')

NameError: name 'model' is not defined

## === cell 10
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


## === cell 11
train_vectors = []
for i in train.text.values:
    train_vectors.append(get_feature_vec([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(i) if word.lower() not in stop], NUM_FEATURES, model))


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/383605625.py in <cell line: 0>()
      1 train_vectors = []
      2 for i in train.text.values:
----> 3     train_vectors.append(get_feature_vec([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(i) if word.lower() not in stop], NUM_FEATURES, model))

NameError: name 'model' is not defined

## === cell 12
test_vectors = []
for i in test.text.values:
    test_vectors.append(get_feature_vec([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(i) if word.lower() not in stop], NUM_FEATURES, model))


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4239267019.py in <cell line: 0>()
      1 test_vectors = []
      2 for i in test.text.values:
----> 3     test_vectors.append(get_feature_vec([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(i) if word.lower() not in stop], NUM_FEATURES, model))

NameError: name 'model' is not defined

## === cell 13
xgtrain = xgb.DMatrix(train_vectors, y_train)
xgtest = xgb.DMatrix(test_vectors)
model = xgb.train(params=list(params.items()), dtrain=xgtrain,  num_boost_round=40)
probs = model.predict(xgtest, ntree_limit=model.best_ntree_limit)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2520347766.py in <cell line: 0>()
----> 1 xgtrain = xgb.DMatrix(train_vectors, y_train)
      2 xgtest = xgb.DMatrix(test_vectors)
      3 model = xgb.train(params=list(params.items()), dtrain=xgtrain,  num_boost_round=40)
      4 probs = model.predict(xgtest, ntree_limit=model.best_ntree_limit)

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in __init__(self, data, label, weight, base_margin, missing, silent, feature_names, feature_types, nthread, group, qid, label_lower_bound, label_upper_bound, feature_weights, enable_categorical, data_split_mode)
    855             return
    856 
--> 857         handle, feature_names, feature_types = dispatch_data_backend(
    858             data,
    859             missing=self.missing,

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in dispatch_data_backend(data, missing, threads, feature_names, feature_types, enable_categorical, data_split_mode)
   1079         return _from_uri(data, missing, feature_names, feature_types, data_split_mode)
   1080     if _is_list(data):
-> 1081         return _from_list(data, missing, threads, feature_names, feature_types)
   1082     if _is_tuple(data):
   1083         return _from_tuple(data, missing, threads, feature_names, feature_types)

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _from_list(data, missing, n_threads, feature_names, feature_types)
   1009     array = np.array(data)
   1010     _check_data_shape(data)
-> 1011     return _from_numpy_array(array, missing, n_threads, feature_names, feature_types)
   1012 
   1013 

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _from_numpy_array(data, missing, nthread, feature_names, feature_types, data_split_mode)
    202 ) -> DispatchedDataBackendReturnType:
    203     """Initialize data from a 2-D numpy matrix."""
--> 204     _check_data_shape(data)
    205     data, _ = _ensure_np_dtype(data, data.dtype)
    206     handle = ctypes.c_void_p()

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _check_data_shape(data)
     55 def _check_data_shape(data: DataType) -> None:
     56     if hasattr(data, "shape") and len(data.shape) != 2:
---> 57         raise ValueError("Please reshape the input data into 2-dimensional matrix.")
     58 
     59 

ValueError: Please reshape the input data into 2-dimensional matrix.

## === cell 14
author = pd.DataFrame(probs)

final = pd.DataFrame()
final['id'] = test.id
final['EAP'] = author[0]
final['HPL'] = author[1]
final['MWS'] = author[2]
final.to_csv('submission.csv', sep=',',index=False)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2011478898.py in <cell line: 0>()
----> 1 author = pd.DataFrame(probs)
      2 
      3 final = pd.DataFrame()
      4 final['id'] = test.id
      5 final['EAP'] = author[0]

NameError: name 'probs' is not defined
