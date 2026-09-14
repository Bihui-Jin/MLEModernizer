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

3.10

# 2. Installed packages

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
scipy==1.15.3
sklearn-pandas==2.2.0
tf_keras==2.18.0
xgboost==2.0.3

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
from nltk.stem import LancasterStemmer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import pickle

try:
    from keras_core import layers, models
except Exception:
    layers, models = None, None


## === cell 2
train = pd.read_csv('../input/us-patent-phrase-to-phrase-matching/train.csv')
test = pd.read_csv('../input/us-patent-phrase-to-phrase-matching/test.csv')
train.shape, test.shape


## === cell 3
stop_words = stopwords.words('english')
ls = LancasterStemmer()


## === cell 4
train['letter'] = train['context'].apply(lambda x: x[0])
train['anchor_token'] = train['anchor'].apply(lambda x: word_tokenize(x))
train['anchor_len'] = train['anchor_token'].apply(lambda x: len(x))
train['anchor_nostop'] = train['anchor_token'].apply(lambda x: [word for word in x if word not in stop_words])
train['anchor_stem'] = train['anchor_nostop'].apply(lambda x: [ls.stem(word) for word in x])
train['new_anchor'] = train['anchor_stem'].apply(lambda x: ' '.join([str(word) for word in x]))
train['target_token'] = train['target'].apply(lambda x: word_tokenize(x))
train['target_len'] = train['target_token'].apply(lambda x: len(x))
train['target_nostop'] = train['target_token'].apply(lambda x: [word for word in x if word not in stop_words])
train['target_stem'] = train['target_nostop'].apply(lambda x: [ls.stem(word) for word in x])
train['new_target'] = train['target_stem'].apply(lambda x: ' '.join([word for word in x]))
train['final'] = train['new_anchor'] + ' ' + train['new_target']
matchlist = []
for x, y in zip(train['anchor_stem'], train['target_stem']):
  matches = 0
  for word in x:
    if word in y:
      matches += 1
  matchlist.append(matches)
train['matches'] = matchlist
train_dummies = pd.get_dummies(train['letter'])
train = pd.concat([train.drop('letter', axis=1), train_dummies], axis=1)
train['score'] = np.log1p(train['score'])


## === cell 5
test['letter'] = test['context'].apply(lambda x: x[0])
test['anchor_token'] = test['anchor'].apply(lambda x: word_tokenize(x))
test['anchor_len'] = test['anchor_token'].apply(lambda x: len(x))
test['anchor_nostop'] = test['anchor_token'].apply(lambda x: [word for word in x if word not in stop_words])
test['anchor_stem'] = test['anchor_nostop'].apply(lambda x: [ls.stem(word) for word in x])
test['new_anchor'] = test['anchor_stem'].apply(lambda x: ' '.join([str(word) for word in x]))
test['target_token'] = test['target'].apply(lambda x: word_tokenize(x))
test['target_len'] = test['target_token'].apply(lambda x: len(x))
test['target_nostop'] = test['target_token'].apply(lambda x: [word for word in x if word not in stop_words])
test['target_stem'] = test['target_nostop'].apply(lambda x: [ls.stem(word) for word in x])
test['new_target'] = test['target_stem'].apply(lambda x: ' '.join([word for word in x]))
test['final'] = test['new_anchor'] + ' ' + test['new_target']
matchlist = []
for x, y in zip(test['anchor_stem'], test['target_stem']):
  matches = 0
  for word in x:
    if word in y:
      matches += 1
  matchlist.append(matches)
test['matches'] = matchlist
test_dummies = pd.get_dummies(test['letter'])
test = pd.concat([test.drop('letter', axis=1), test_dummies], axis=1)


## === cell 6
vectorizer = CountVectorizer(max_features=2000)
vectortrain = vectorizer.fit(train['final'])
pickle.dump(vectortrain, open("patent_vectorizer.pickle", "wb"))
matrixtrain = vectorizer.transform(train['final'])
trainvect = pd.DataFrame(matrixtrain.toarray(), columns=vectortrain.get_feature_names_out())
trainvect['score'] = train.reset_index().score
trainvect['anchor_len'] = train['anchor_len']
trainvect['target_len'] = train['target_len']
trainvect['matches'] = train['matches']
trainvect = pd.concat([trainvect, train_dummies], axis=1)
trainx, testx, trainy, testy = train_test_split(trainvect.drop(['score'], axis=1), trainvect['score'], test_size=.2, random_state=777)


## === cell 7
matrixtrain.shape


## === cell 8
'''
rfc = XGBRegressor()
random_grid = {'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 6, 10, None],
        'learning_rate':[0.03, 0.06, .1, .15, .2],
        'n_estimators':[100,200,250]}

rf_random = RandomizedSearchCV(estimator=rfc, param_distributions=random_grid,n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
rf_random.fit(trainx, trainy)

'''


## === cell 10
rfr1 = RandomForestRegressor()
parameters = {
    "n_estimators": [200,300,400],
    "max_features": ["auto", 10, 20],
    "min_samples_split": [6],
    "min_samples_leaf": [1],
    "bootstrap": [True, False],
}


## === cell 11
'''
rfr_grid = GridSearchCV(rfr1,
                        parameters,
                        cv=2,
                        n_jobs=-1,
                        verbose=True)

rfr_grid.fit(trainx,trainy)

print(rfr_grid.best_score_)
print(rfr_grid.best_params_)
'''


## === cell 12
xgb1 = XGBRegressor()
parameters = {'gamma': [0, 1, 2],
              'learning_rate': [.2, 0.25, .1],
              'max_depth': [6, 8, 10, None],
              'min_child_weight': [5],
              'subsample': [1.0],
              'colsample_bytree': [.6],
              'n_estimators': [400]}


## === cell 13
'''
testy = np.log1p(testy)
xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv=2,
                        n_jobs=-1,
                        verbose=True)

xgb_grid.fit(trainx, trainy)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)
'''


## === cell 14
testy = np.expm1(testy)


## === cell 15
xgbr = XGBRegressor(subsample= 1.0,n_estimators= 400,min_child_weight= 5,max_depth= 6,learning_rate= .2,gamma= 0,colsample_bytree= 0.6)
xgbr_fit = xgbr.fit(trainx, trainy, eval_set=[(testx, testy)])
pred1 = np.expm1(xgbr_fit.predict(testx))
print(pearsonr(testy, pred1)[0])
pickle.dump(xgbr_fit, open('patent_xgb.pickle', 'wb'))


## === cell 16
rfr = RandomForestRegressor(n_estimators=200, min_samples_leaf=1, min_samples_split=6, max_features=20, bootstrap=True, max_depth=150, max_samples=.6)
rfr_fit = rfr.fit(trainx, trainy)
pred = np.expm1(rfr_fit.predict(testx))
print(pearsonr(testy, pred)[0])
pickle.dump(rfr_fit, open('patent_rfr.pickle', 'wb'))


## === cell 17
pearsonr(testy, (pred1*.5+pred*.5))[0]


## === cell 18
pred


## === cell 19
pred1


## === cell 20
testy


## === cell 21
matrixtest = vectorizer.transform(test['final'])
dftest = pd.DataFrame(matrixtest.toarray(), columns=vectortrain.get_feature_names_out())
dftest['anchor_len'] = test['anchor_len']
dftest['target_len'] = test['target_len']
dftest['matches'] = test['matches']
dftest = pd.concat([dftest, test_dummies], axis=1)
prediction0 = xgbr_fit.predict(dftest)
prediction1 = rfr_fit.predict(dftest)
test['score'] = prediction0*0.5+prediction1*0.5
finaldf = test[['id', 'score']]


## === cell 22
test['score']


## === cell 23
if layers is None or models is None:
    from keras_core import layers, models

embedding_dim = 16
vocab_size = 3000
model = models.Sequential(
    [
        layers.Input(shape=(trainx.shape[1],)),
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ]
)


## --- ERROR in cell 23, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mImportError[0m                               Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3914674739.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0;31m# which fails due to protobuf C-extension incompatibility. Reuse `keras_core` instead.[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;32mif[0m [0mlayers[0m [0;32mis[0m [0;32mNone[0m [0;32mor[0m [0mmodels[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m     [0;32mfrom[0m [0mkeras_core[0m [0;32mimport[0m [0mlayers[0m[0;34m,[0m [0mmodels[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0;34m[0m[0m
[1;32m      6[0m [0membedding_dim[0m [0;34m=[0m [0;36m16[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras_core/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m
[0;32m----> 8[0;31m [0;32mfrom[0m [0mkeras_core[0m [0;32mimport[0m [0m_tf_keras[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;32mfrom[0m [0mkeras_core[0m [0;32mimport[0m [0mactivations[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m [0;32mfrom[0m [0mkeras_core[0m [0;32mimport[0m [0mapplications[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras_core/_tf_keras/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m
[0;32m----> 8[0;31m [0;32mfrom[0m [0mkeras_core[0m [0;32mimport[0m [0mactivations[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;32mfrom[0m [0mkeras_core[0m [0;32mimport[0m [0mapplications[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m [0;32mfrom[0m [0mkeras_core[0m [0;32mimport[0m [0mcallbacks[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras_core/activations/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m
[0;32m----> 8[0;31m [0;32mfrom[0m [0mkeras_core[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0mdeserialize[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;32mfrom[0m [0mkeras_core[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0mget[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m [0;32mfrom[0m [0mkeras_core[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0mserialize[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras_core/src/__init__.py[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mkeras_core[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mactivations[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;32mfrom[0m [0mkeras_core[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mapplications[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;32mfrom[0m [0mkeras_core[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mbackend[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mkeras_core[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mconstraints[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mkeras_core[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mdatasets[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras_core/src/activations/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0;32mimport[0m [0mtypes[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;34m[0m[0m
[0;32m----> 3[0;31m [0;32mfrom[0m [0mkeras_core[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0melu[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0;32mfrom[0m [0mkeras_core[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0mexponential[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mkeras_core[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mactivations[0m[0;34m.[0m[0mactivations[0m [0;32mimport[0m [0mgelu[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras_core/src/activations/activations.py[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mkeras_core[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mbackend[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;32mfrom[0m [0mkeras_core[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mops[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;32mfrom[0m [0mkeras_core[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mapi_export[0m [0;32mimport[0m [0mkeras_core_export[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;34m[0m[0m
[1;32m      5[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras_core/src/backend/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     32[0m [0;32mif[0m [0mbackend[0m[0;34m([0m[0;34m)[0m [0;34m==[0m [0;34m"tensorflow"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     33[0m     [0mprint_msg[0m[0;34m([0m[0;34m"Using TensorFlow backend"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 34[0;31m     [0;32mfrom[0m [0mkeras_core[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mtensorflow[0m [0;32mimport[0m [0;34m*[0m  [0;31m# noqa: F403[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     35[0m [0;34m[0m[0m
[1;32m     36[0m     [0mdistribution_lib[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras_core/src/backend/tensorflow/__init__.py[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mkeras_core[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mtensorflow[0m [0;32mimport[0m [0mcore[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;32mfrom[0m [0mkeras_core[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mtensorflow[0m [0;32mimport[0m [0mimage[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;32mfrom[0m [0mkeras_core[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mtensorflow[0m [0;32mimport[0m [0mmath[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mkeras_core[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mtensorflow[0m [0;32mimport[0m [0mnn[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mkeras_core[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m[0;34m.[0m[0mtensorflow[0m [0;32mimport[0m [0mnumpy[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras_core/src/backend/tensorflow/core.py[0m in [0;36m<module>[0;34m[0m
[1;32m      2[0m [0;34m[0m[0m
[1;32m      3[0m [0;32mimport[0m [0mnumpy[0m [0;32mas[0m [0mnp[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m [0;32mimport[0m [0mtensorflow[0m [0;32mas[0m [0mtf[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcompiler[0m[0;34m.[0m[0mtf2xla[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mxla[0m [0;32mimport[0m [0mdynamic_update_slice[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     47[0m [0m_tf2[0m[0;34m.[0m[0menable[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     48[0m [0;34m[0m[0m
[0;32m---> 49[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m [0;32mimport[0m [0m__internal__[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     50[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m [0;32mimport[0m [0m__operators__[0m[0;34m[0m[0;34m[0m[0m
[1;32m     51[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m [0;32mimport[0m [0maudio[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;32mimport[0m [0msys[0m [0;32mas[0m [0m_sys[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m
[0;32m----> 8[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m[0;34m.[0m[0m__internal__[0m [0;32mimport[0m [0mautograph[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m[0;34m.[0m[0m__internal__[0m [0;32mimport[0m [0mdecorator[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m[0;34m.[0m[0m__internal__[0m [0;32mimport[0m [0mdispatch[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;32mimport[0m [0msys[0m [0;32mas[0m [0m_sys[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m
[0;32m----> 8[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mag_ctx[0m [0;32mimport[0m [0mcontrol_status_ctx[0m [0;31m# line: 34[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mimpl[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mtf_convert[0m [0;31m# line: 493[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py[0m in [0;36m<module>[0;34m[0m
[1;32m     19[0m [0;32mimport[0m [0mthreading[0m[0;34m[0m[0;34m[0m[0m
[1;32m     20[0m [0;34m[0m[0m
[0;32m---> 21[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m [0;32mimport[0m [0mag_logging[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     22[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mutil[0m[0;34m.[0m[0mtf_export[0m [0;32mimport[0m [0mtf_export[0m[0;34m[0m[0;34m[0m[0m
[1;32m     23[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     15[0m [0;34m"""Utility module that contains APIs usable in the generated code."""[0m[0;34m[0m[0;34m[0m[0m
[1;32m     16[0m [0;34m[0m[0m
[0;32m---> 17[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mcontext_managers[0m [0;32mimport[0m [0mcontrol_dependency_on_returns[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     18[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mmisc[0m [0;32mimport[0m [0malias_tensors[0m[0;34m[0m[0;34m[0m[0m
[1;32m     19[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mtensor_list[0m [0;32mimport[0m [0mdynamic_list_append[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py[0m in [0;36m<module>[0;34m[0m
[1;32m     17[0m [0;32mimport[0m [0mcontextlib[0m[0;34m[0m[0;34m[0m[0m
[1;32m     18[0m [0;34m[0m[0m
[0;32m---> 19[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mops[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     20[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mops[0m [0;32mimport[0m [0mtensor_array_ops[0m[0;34m[0m[0;34m[0m[0m
[1;32m     21[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py[0m in [0;36m<module>[0;34m[0m
[1;32m     31[0m [0;34m[0m[0m
[1;32m     32[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mmessage[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 33[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mattr_value_pb2[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     34[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mfull_type_pb2[0m[0;34m[0m[0;34m[0m[0m
[1;32m     35[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mfunction_pb2[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py[0m in [0;36m<module>[0;34m[0m
[1;32m      3[0m [0;31m# source: tensorflow/core/framework/attr_value.proto[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;34m"""Generated protocol buffer code."""[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m[0;34m.[0m[0minternal[0m [0;32mimport[0m [0mbuilder[0m [0;32mas[0m [0m_builder[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mdescriptor[0m [0;32mas[0m [0m_descriptor[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mdescriptor_pool[0m [0;32mas[0m [0m_descriptor_pool[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py[0m in [0;36m<module>[0;34m[0m
[1;32m     16[0m [0;34m[0m[0m
[1;32m     17[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m[0;34m.[0m[0minternal[0m [0;32mimport[0m [0menum_type_wrapper[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 18[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m[0;34m.[0m[0minternal[0m [0;32mimport[0m [0mpython_message[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     19[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mmessage[0m [0;32mas[0m [0m_message[0m[0;34m[0m[0;34m[0m[0m
[1;32m     20[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mreflection[0m [0;32mas[0m [0m_reflection[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py[0m in [0;36m<module>[0;34m[0m
[1;32m     36[0m [0;32mimport[0m [0mweakref[0m[0;34m[0m[0;34m[0m[0m
[1;32m     37[0m [0;34m[0m[0m
[0;32m---> 38[0;31m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mdescriptor[0m [0;32mas[0m [0mdescriptor_mod[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     39[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mmessage[0m [0;32mas[0m [0mmessage_mod[0m[0;34m[0m[0;34m[0m[0m
[1;32m     40[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mtext_format[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py[0m in [0;36m<module>[0;34m[0m
[1;32m     27[0m   [0;31m# TODO: Remove this import after fix api_implementation[0m[0;34m[0m[0;34m[0m[0m
[1;32m     28[0m   [0;32mif[0m [0m_message[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 29[0;31m     [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m[0;34m.[0m[0mpyext[0m [0;32mimport[0m [0m_message[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     30[0m   [0m_USE_C_DESCRIPTORS[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m
[1;32m     31[0m [0;34m[0m[0m

[0;31mImportError[0m: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

## === cell 24
model.compile(optimizer='RMSprop',
              loss='mse',
              metrics=['RootMeanSquaredError'])
