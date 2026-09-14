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
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Target score

0.71541

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score


## === cell 2
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        


train_df1 = pd.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv')


## === cell 3
train_df1['score'].dtypes


## === cell 4
train_df1.head(10)


## === cell 5
train_df1['score'].value_counts()


## === cell 6
submission = pd.read_csv("/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv")
submission


## === cell 8
train_df1['full_text'] = train_df1['full_text'].map(lambda x: re.sub('\n',' ',str(x)))

train_df1['full_text'] = train_df1['full_text'].map(lambda x: re.sub('[^\w\s]','',str(x)))  

train_df1['full_text'] = train_df1['full_text'].str.lower()  

train_df1['full_text'] = train_df1['full_text'].str.replace('\d+', '') 


## === cell 9
test_df1 = pd.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv')


## === cell 10
test_df1['full_text'] = test_df1['full_text'].map(lambda x: re.sub('\n',' ',str(x)))

test_df1['full_text'] = test_df1['full_text'].map(lambda x: re.sub('[^\w\s]','',str(x)))

test_df1['full_text'] = test_df1['full_text'].str.lower()

test_df1['full_text'] = test_df1['full_text'].str.replace('\d+', '') 


## === cell 11
essay_id_test_df1 = test_df1['essay_id']  


## === cell 12
type(essay_id_test_df1)


## === cell 13
essay_id_test_df1


## === cell 14
x = train_df1[['full_text', 'essay_id']]  
y = train_df1.iloc[:, 2:8] 


## === cell 15
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)  


## === cell 16
y_train['score'].value_counts()


## === cell 17
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


## === cell 18
text_vectorizer = TfidfVectorizer(
    stop_words='english',
    sublinear_tf=True,
    strip_accents='ascii',
    analyzer='word',
    token_pattern=r'\w{3,}',  
    ngram_range=(1,1),
    norm='l1', 
    use_idf=False, 
    smooth_idf=False,
    max_features=None,
    min_df=20)


## === cell 19
X_train_features = text_vectorizer.fit_transform(X_train['full_text'])


## === cell 20
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC

test_features = text_vectorizer.transform(X_test['full_text'])
    
clf = make_pipeline(MaxAbsScaler(), SVC(C=2.0, kernel='rbf', gamma='scale', decision_function_shape='ovr', random_state=123, tol=1e-5, shrinking=True, verbose=True, break_ties=True))
clf.fit(X_train_features, y_train['score'])  
y_pred = clf.predict(test_features)  


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3644881247.py in <cell line: 0>()
      7 
      8 clf = make_pipeline(MaxAbsScaler(), SVC(C=2.0, kernel='rbf', gamma='scale', decision_function_shape='ovr', random_state=123, tol=1e-5, shrinking=True, verbose=True, break_ties=True))
----> 9 clf.fit(X_train_features, y_train['score'])
     10 y_pred = clf.predict(test_features)

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in fit(self, X, y, **fit_params)
    399         """
    400         fit_params_steps = self._check_fit_params(**fit_params)
--> 401         Xt = self._fit(X, y, **fit_params_steps)
    402         with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
    403             if self._final_estimator != "passthrough":

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit(self, X, y, **fit_params_steps)
    357                 cloned_transformer = clone(transformer)
    358             # Fit or load from cache the current transformer
--> 359             X, fitted_transformer = fit_transform_one_cached(
    360                 cloned_transformer,
    361                 X,

/usr/local/lib/python3.11/dist-packages/joblib/memory.py in __call__(self, *args, **kwargs)
    324 
    325     def __call__(self, *args, **kwargs):
--> 326         return self.func(*args, **kwargs)
    327 
    328     def call_and_shelve(self, *args, **kwargs):

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit_transform_one(transformer, X, y, weight, message_clsname, message, **fit_params)
    891     with _print_elapsed_time(message_clsname, message):
    892         if hasattr(transformer, "fit_transform"):
--> 893             res = transformer.fit_transform(X, y, **fit_params)
    894         else:
    895             res = transformer.fit(X, y, **fit_params).transform(X)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in fit_transform(self, X, y, **fit_params)
    879         else:
    880             # fit method of arity 2 (supervised transformation)
--> 881             return self.fit(X, y, **fit_params).transform(X)
    882 
    883 

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in fit(self, X, y)
   1167         # Reset internal state before fitting
   1168         self._reset()
-> 1169         return self.partial_fit(X, y)
   1170 
   1171     def partial_fit(self, X, y=None):

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in partial_fit(self, X, y)
   1202 
   1203         if sparse.issparse(X):
-> 1204             mins, maxs = min_max_axis(X, axis=0, ignore_nan=True)
   1205             max_abs = np.maximum(np.abs(mins), np.abs(maxs))
   1206         else:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py in min_max_axis(X, axis, ignore_nan)
    504     if isinstance(X, (sp.csr_matrix, sp.csc_matrix)):
    505         if ignore_nan:
--> 506             return _sparse_nan_min_max(X, axis=axis)
    507         else:
    508             return _sparse_min_max(X, axis=axis)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py in _sparse_nan_min_max(X, axis)
    472 
    473 def _sparse_nan_min_max(X, axis):
--> 474     return (_sparse_min_or_max(X, axis, np.fmin), _sparse_min_or_max(X, axis, np.fmax))
    475 
    476 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py in _sparse_min_or_max(X, axis, min_or_max)
    459         axis += 2
    460     if (axis == 0) or (axis == 1):
--> 461         return _min_or_max_axis(X, axis, min_or_max)
    462     else:
    463         raise ValueError("invalid axis, use 0 for rows, or 1 for columns")

/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py in _min_or_max_axis(X, axis, min_or_max)
    442             (value, (major_index, np.zeros(len(value)))), dtype=X.dtype, shape=(M, 1)
    443         )
--> 444     return res.A.ravel()
    445 
    446 

AttributeError: 'coo_matrix' object has no attribute 'A'

## === cell 21
print(confusion_matrix(y_test['score'], y_pred))


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/223774424.py in <cell line: 0>()
----> 1 print(confusion_matrix(y_test['score'], y_pred))

NameError: name 'y_pred' is not defined

## === cell 22
print(classification_report(y_test['score'], y_pred, zero_division=1))


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1539733996.py in <cell line: 0>()
----> 1 print(classification_report(y_test['score'], y_pred, zero_division=1))

NameError: name 'y_pred' is not defined

## === cell 23
kappa = cohen_kappa_score(y_test['score'], y_pred, weights='quadratic')  
print('Cohen\'s kappa score: ', kappa)  


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3474485437.py in <cell line: 0>()
----> 1 kappa = cohen_kappa_score(y_test['score'], y_pred, weights='quadratic')
      2 print('Cohen\'s kappa score: ', kappa)

NameError: name 'y_pred' is not defined

## === cell 24



st_features = text_vectorizer.transform(test_df1['full_text'])



test_predictions = clf.predict(st_features)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1478775426.py in <cell line: 0>()
     11 # clf.fit(sr_word_features, selected_rows['score'])
     12 
---> 13 test_predictions = clf.predict(st_features)

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in predict(self, X, **predict_params)
    478         Xt = X
    479         for _, name, transform in self._iter(with_final=False):
--> 480             Xt = transform.transform(Xt)
    481         return self.steps[-1][1].predict(Xt, **predict_params)
    482 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in transform(self, X)
   1241 
   1242         if sparse.issparse(X):
-> 1243             inplace_column_scale(X, 1.0 / self.scale_)
   1244         else:
   1245             X /= self.scale_

AttributeError: 'MaxAbsScaler' object has no attribute 'scale_'

## === cell 25
submission = pd.read_csv("/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv")
submission['score'] = test_predictions
submission.to_csv("submission.csv", index=False)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1568801920.py in <cell line: 0>()
      1 submission = pd.read_csv("/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv")
----> 2 submission['score'] = test_predictions
      3 submission.to_csv("submission.csv", index=False)
      4 # display(submission.head())

NameError: name 'test_predictions' is not defined

## === cell 26
submission.dtypes
