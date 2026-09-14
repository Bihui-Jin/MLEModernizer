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
import os as _os

_os.environ.setdefault("KERAS_BACKEND", "numpy")

from keras import layers, models

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


## === cell 24
model.compile(optimizer='RMSprop',
              loss='mse',
              metrics=['RootMeanSquaredError'])


## === cell 25
model.summary()


## === cell 26
'''
model.fit(
    trainx, trainy,
    validation_data=(testx,testy),
    epochs=41)
pred = model.predict(testx)
pearsonr(np.expm1(testy), np.expm1(pred))
'''


## === cell 27
model.fit(
    trainvect.drop(['score'], axis=1), trainvect['score'],
    epochs=30)
pred = np.expm1(model.predict(dftest))
modelsub = test[['id']]
modelsub['score'] = pred
modelsub.to_csv('submission.csv', index=False)


## --- ERROR in cell 27, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNotImplementedError[0m                       Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/859649366.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m model.fit(
[0m[1;32m      2[0m     [0mtrainvect[0m[0;34m.[0m[0mdrop[0m[0;34m([0m[0;34m[[0m[0;34m'score'[0m[0;34m][0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;36m1[0m[0;34m)[0m[0;34m,[0m [0mtrainvect[0m[0;34m[[0m[0;34m'score'[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m     epochs=30)
[1;32m      4[0m [0mpred[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mexpm1[0m[0;34m([0m[0mmodel[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mdftest[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0mmodelsub[0m [0;34m=[0m [0mtest[0m[0;34m[[0m[0;34m[[0m[0;34m'id'[0m[0;34m][0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/backend/numpy/trainer.py[0m in [0;36mfit[0;34m(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq)[0m
[1;32m    167[0m         [0mvalidation_freq[0m[0;34m=[0m[0;36m1[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    168[0m     ):
[0;32m--> 169[0;31m         [0;32mraise[0m [0mNotImplementedError[0m[0;34m([0m[0;34m"fit not implemented for NumPy backend."[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    170[0m [0;34m[0m[0m
[1;32m    171[0m     [0;34m@[0m[0mtraceback_utils[0m[0;34m.[0m[0mfilter_traceback[0m[0;34m[0m[0;34m[0m[0m

[0;31mNotImplementedError[0m: fit not implemented for NumPy backend.
