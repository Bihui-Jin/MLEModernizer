# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.36146

# 6. Current score

0.4078

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.4078) has done: 'The crash happens because recent `xgboost` (2.0.3) no longer provides `model.best_ntree_limit` unless early-stopping is used, and this code doesn’t set early stopping. In this situation you should predict using all boosting rounds (the trained model already contains exactly `num_boost_round` trees). The minimal fix is to remove the `ntree_limit=model.best_ntree_limit` argument from `model.predict(...)` in cell 23. This preserves the core training/prediction logic and keeps `pred_full_test` as the same shape expected by cell 25.'

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
from sklearn.model_selection import KFold, train_test_split

from sklearn.metrics import f1_score, accuracy_score

import xgboost as xgb

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

alpha_tokenizer = RegexpTokenizer('[A-Za-z]\w+')
lemmatizer = WordNetLemmatizer()
stop = stopwords.words('english')


## === cell 1
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test_id = test['id'].values

author_mapping = {'EAP':0, 'HPL':1, 'MWS':2}
y_train = train['author'].map(author_mapping).values


## === cell 3
vectorizers = [ # ('3-gram TF-IDF Vectorizer on words', TfidfVectorizer(ngram_range=(1, 3), analyzer='word', binary=False)),
                ('TF-IDF + SVD', Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 3), analyzer='word', binary=False)),
                                 ('svd', TruncatedSVD(n_components=150)),
                                ])),
                ('TF-IDF + SVD + Normalizer', Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 3), analyzer='word', binary=False)),
                                 ('svd', TruncatedSVD(n_components=150)),
                                 ('norm', Normalizer()),
                                ]))
              ]


## === cell 4
estimators = [
              (KNeighborsClassifier(n_neighbors=3), 'K-Nearest Neighbors', 'yellow'),
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
def vectorize():
    
    test_size = 0.3

    train_split, test_split = train_test_split(train, test_size=test_size)

    y_train_split = train_split['author'].map(author_mapping).values
    y_test_split = test_split['author'].map(author_mapping).values
    
    for vectorizer in vectorizers:
        print(vectorizer[0] + '\n')
        X = vectorizer[1].fit_transform(train.text.values)
        X_train, X_test = train_test_split(X, test_size=test_size)
        for estimator in estimators:
            if estimator[1] == 'XGBoost': 
                xgtrain = xgb.DMatrix(X_train, y_train_split)
                xgtest = xgb.DMatrix(X_test)
                model = xgb.train(params=list(params.items()), dtrain=xgtrain,  num_boost_round=40)
                predictions = model.predict(xgtest, ntree_limit=model.best_ntree_limit).argmax(axis=1)
            else:
                estimator[0].fit(X_train, y_train_split)
                predictions = estimator[0].predict(X_test)
            print(accuracy_score(predictions, y_test_split), estimator[1])


## === cell 7
train['num_words'] = train.text.apply(lambda x: len(str(x).split()))
test['num_words'] = test.text.apply(lambda x: len(str(x).split()))

train['num_unique_words'] = train.text.apply(lambda x: len(set(str(x).split())))
test['num_unique_words'] = test.text.apply(lambda x: len(set(str(x).split())))

train['num_chars'] = train.text.apply(lambda x: len(str(x)))
test['num_chars'] = test.text.apply(lambda x: len(str(x)))

train['num_stopwords'] = train.text.apply(lambda x: len([w for w in str(x).lower().split() if w in stop]))
test['num_stopwords'] = test.text.apply(lambda x: len([w for w in str(x).lower().split() if w in stop]))

train['mean_word_len'] = train.apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test['mean_word_len'] = test.apply(lambda x: np.mean([len(w) for w in str(x).split()]))


## === cell 8
train_text = [' '.join([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(sent) if word.lower() not in stop]) for sent in train.text.values]
test_text = [' '.join([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(sent) if word.lower() not in stop]) for sent in test.text.values]


## === cell 9
vectorizer = CountVectorizer(ngram_range=(1,7), analyzer='char')

full = vectorizer.fit_transform(train_text + test_text)
X_train = vectorizer.transform(train_text)
X_test = vectorizer.transform(test_text)

pred_full_test = 0
pred_train = np.zeros([train.shape[0], 3])

for dev_index, val_index in KFold(n_splits=5, shuffle=True, random_state=42).split(train.drop(['id', 'author'], axis=1)):
    dev_X, val_X = X_train[dev_index], X_train[val_index]
    dev_y, val_y = y_train[dev_index], y_train[val_index]
    model = MultinomialNB()
    model.fit(dev_X, dev_y)
    pred_full_test = pred_full_test + model.predict_proba(X_test)
    pred_train[val_index,:] = model.predict_proba(val_X)

pred_full_test = pred_full_test / 5.

train['CH_EAP'] = pred_train[:,0]
train['CH_HPL'] = pred_train[:,1]
train['CH_MWS'] = pred_train[:,2]
test['CH_EAP'] = pred_full_test[:,0]
test['CH_HPL'] = pred_full_test[:,1]
test['CH_MWS'] = pred_full_test[:,2]


## === cell 10
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3))
full = vectorizer.fit_transform(train_text + test_text)
X_train = vectorizer.transform(train_text)
X_test = vectorizer.transform(test_text)

pred_full_test = 0
pred_train = np.zeros([train.shape[0], 3])

for dev_index, val_index in KFold(n_splits=5, shuffle=True, random_state=42).split(train.drop(['id', 'author'], axis=1)):
    dev_X, val_X = X_train[dev_index], X_train[val_index]
    dev_y, val_y = y_train[dev_index], y_train[val_index]
    model = MultinomialNB()
    model.fit(dev_X, dev_y)
    pred_full_test = pred_full_test + model.predict_proba(X_test)
    pred_train[val_index,:] = model.predict_proba(val_X)

pred_full_test = pred_full_test / 5.

train['C_EAP'] = pred_train[:,0]
train['C_HPL'] = pred_train[:,1]
train['C_MWS'] = pred_train[:,2]
test['C_EAP'] = pred_full_test[:,0]
test['C_HPL'] = pred_full_test[:,1]
test['C_MWS'] = pred_full_test[:,2]


## === cell 11
vectorizer = TfidfVectorizer(ngram_range=(1,5), analyzer='char')
full = vectorizer.fit_transform(train_text + test_text)
X_train = vectorizer.transform(train_text)
X_test = vectorizer.transform(test_text)

pred_full_test = 0
pred_train = np.zeros([train.shape[0], 3])

for dev_index, val_index in KFold(n_splits=5, shuffle=True, random_state=42).split(train.drop(['id', 'author'], axis=1)):
    dev_X, val_X = X_train[dev_index], X_train[val_index]
    dev_y, val_y = y_train[dev_index], y_train[val_index]
    model = MultinomialNB()
    model.fit(dev_X, dev_y)
    pred_full_test = pred_full_test + model.predict_proba(X_test)
    pred_train[val_index,:] = model.predict_proba(val_X)

pred_full_test = pred_full_test / 5.

train['T_EAP'] = pred_train[:,0]
train['T_HPL'] = pred_train[:,1]
train['T_MWS'] = pred_train[:,2]
test['T_EAP'] = pred_full_test[:,0]
test['T_HPL'] = pred_full_test[:,1]
test['T_MWS'] = pred_full_test[:,2]


## === cell 12
svd = TruncatedSVD(n_components=20, algorithm='arpack')
svd.fit(full)
train_svd = pd.DataFrame(svd.transform(X_train))
test_svd = pd.DataFrame(svd.transform(X_test))
    
train_svd.columns = ['SVD_' + str(i) for i in range(20)]
test_svd.columns = ['SVD_' + str(i) for i in range(20)]
train = pd.concat([train, train_svd], axis=1)
test = pd.concat([test, test_svd], axis=1)


## === cell 13
train = train.drop(['id', 'text', 'author'], axis=1)
test = test.drop(['id', 'text'], axis=1)


## === cell 15
NUM_FEATURES = 100

sentences = [s.split() for s in (train_text + test_text)]
model = Word2Vec(
    sentences,
    min_count=2,
    vector_size=NUM_FEATURES,
    window=4,
    sg=1,
    alpha=1e-4,
    workers=4,
)


## === cell 16
len(model.wv.key_to_index)


## === cell 17
model.wv.most_similar("raven")


## === cell 18
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


## === cell 19
train_vectors = []
for i in train_text:
    train_vectors.append(
        get_feature_vec(
            [
                lemmatizer.lemmatize(word.lower())
                for word in alpha_tokenizer.tokenize(i)
                if word.lower() not in stop
            ],
            NUM_FEATURES,
            model.wv,
        )
    )


## === cell 20
test_vectors = []
for i in test_text:
    test_vectors.append(
        get_feature_vec(
            [
                lemmatizer.lemmatize(word.lower())
                for word in alpha_tokenizer.tokenize(i)
                if word.lower() not in stop
            ],
            NUM_FEATURES,
            model.wv,
        )
    )


## === cell 21
full_vectors = []
for i in train_text + test_text:
    full_vectors.append(
        get_feature_vec(
            [
                lemmatizer.lemmatize(word.lower())
                for word in alpha_tokenizer.tokenize(i)
                if word.lower() not in stop
            ],
            NUM_FEATURES,
            model.wv,
        )
    )


## === cell 22
svd = TruncatedSVD(n_components=30, algorithm='arpack')

svd.fit(full_vectors)
train_svd = pd.DataFrame(svd.transform(np.array(train_vectors)))
test_svd = pd.DataFrame(svd.transform(np.array(test_vectors)))
    
train_svd.columns = ['W2V_' + str(i) for i in range(30)]
test_svd.columns = ['W2V_' + str(i) for i in range(30)]

train = pd.concat([train, train_svd], axis=1)
test = pd.concat([test, test_svd], axis=1)


## === cell 23
pred_full_test = 0
pred_train = np.zeros([train.shape[0], 3])
for dev_index, val_index in KFold(n_splits=5, shuffle=True, random_state=42).split(
    train
):
    dev_X, val_X = train.loc[dev_index], train.loc[val_index]
    dev_y, val_y = y_train[dev_index], y_train[val_index]
    xgtrain = xgb.DMatrix(dev_X, dev_y)
    xgtest = xgb.DMatrix(test)
    model = xgb.train(params=list(params.items()), dtrain=xgtrain, num_boost_round=1000)
    predictions = model.predict(xgtest)
    pred_full_test = pred_full_test + predictions
pred_full_test = pred_full_test / 5.0


## === cell 25
author = pd.DataFrame(pred_full_test)

final = pd.DataFrame()
final['id'] = test_id
final['EAP'] = author[0]
final['HPL'] = author[1]
final['MWS'] = author[2]

final.to_csv('submission.csv', sep=',',index=False)
