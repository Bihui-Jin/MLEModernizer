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
plotly==5.24.1
plotly-express==0.4.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
shap==0.44.1
shapely==2.1.2
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

0.62054

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!pip install pandas==2.2.2
!pip install scikit-learn==1.2.2
!pip install numpy==1.26.4
!pip install matplotlib==3.7.5
!pip install seaborn==0.12.2
!pip install shap==0.44.1
!pip install plotly==5.18.0


## === cell 1
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.express as px


## === cell 2
def quadratic_weighted_kappa(y_true, y_pred):
    y_true = np.array(y_true)  # Ensure numpy array
    y_pred = np.array(y_pred)  # Ensure numpy array
    
    min_rating = np.min(y_true)
    max_rating = np.max(y_true)
    
    y_pred = np.clip(y_pred, min_rating, max_rating)
    
    hist_true = np.histogram(y_true, bins=max_rating-min_rating+1, range=(min_rating, max_rating+1))[0]
    hist_pred = np.histogram(y_pred, bins=max_rating-min_rating+1, range=(min_rating, max_rating+1))[0]

    O = np.zeros((max_rating-min_rating+1, max_rating-min_rating+1))
    for i in range(len(y_true)):
        O[y_true[i]-min_rating, y_pred[i]-min_rating] += 1

    E = np.outer(hist_true, hist_pred) / len(y_true)
    W = np.zeros((max_rating-min_rating+1, max_rating-min_rating+1))
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] = ((i - j) ** 2.0 / (max_rating - min_rating) ** 2.0)
    
    kappa = 1.0 - (np.sum(W * O) / np.sum(W * E))
    return kappa


## === cell 3
train_df = pd.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv')
test_df = pd.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv')


## === cell 4
plt.figure(figsize=(10, 6))
sns.histplot(train_df['score'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Scores in Training Data', fontsize=16)
plt.xlabel('Score', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True)
plt.show()


## === cell 5
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X = vectorizer.fit_transform(train_df['full_text'])
y = train_df['score']

feature_names = vectorizer.get_feature_names_out()
tfidf_scores = X.sum(axis=0).A1

df_tfidf = pd.DataFrame({'Feature': feature_names, 'TF-IDF Score': tfidf_scores})
df_tfidf = df_tfidf.sort_values(by='TF-IDF Score', ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(x='TF-IDF Score', y='Feature', data=df_tfidf, palette='viridis')
plt.title('Top 20 TF-IDF Features', fontsize=16)
plt.xlabel('TF-IDF Score', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.grid(True)
plt.show()


## === cell 6
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


## === cell 7
model = Ridge()
params = {'alpha': [0.1, 1.0, 10.0]}
scorer = make_scorer(quadratic_weighted_kappa)
grid = GridSearchCV(model, param_grid=params, scoring=scorer, cv=5)
grid.fit(X_train, y_train)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1311701215.py in <cell line: 0>()
      4 scorer = make_scorer(quadratic_weighted_kappa)
      5 grid = GridSearchCV(model, param_grid=params, scoring=scorer, cv=5)
----> 6 grid.fit(X_train, y_train)

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in fit(self, X, y, groups, **fit_params)
    872                 return results
    873 
--> 874             self._run_search(evaluate_candidates)
    875 
    876             # multimetric is determined here because in the case of a callable

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in _run_search(self, evaluate_candidates)
   1386     def _run_search(self, evaluate_candidates):
   1387         """Search all candidates in param_grid"""
-> 1388         evaluate_candidates(ParameterGrid(self.param_grid))
   1389 
   1390 

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in evaluate_candidates(candidate_params, cv, more_results)
    849                     )
    850 
--> 851                 _warn_or_raise_about_fit_failures(out, self.error_score)
    852 
    853                 # For callable self.scoring, the return type is only know after

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_validation.py in _warn_or_raise_about_fit_failures(results, error_score)
    365                 f"Below are more details about the failures:\n{fit_errors_summary}"
    366             )
--> 367             raise ValueError(all_fits_failed_message)
    368 
    369         else:

ValueError: 
All the 15 fits failed.
It is very likely that your model is misconfigured.
You can try to debug the error by setting error_score='raise'.

Below are more details about the failures:
--------------------------------------------------------------------------------
15 fits failed with the following error:
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py", line 115, in _solve_sparse_cg
    coef, info = sp_linalg.cg(C, y_column, tol=tol, atol="legacy")
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: cg() got an unexpected keyword argument 'tol'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_validation.py", line 686, in _fit_and_score
    estimator.fit(X_train, y_train, **fit_params)
  File "/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py", line 1134, in fit
    return super().fit(X, y, sample_weight=sample_weight)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py", line 900, in fit
    self.coef_, self.n_iter_ = _ridge_regression(
                               ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py", line 671, in _ridge_regression
    coef = _solve_sparse_cg(
           ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py", line 118, in _solve_sparse_cg
    coef, info = sp_linalg.cg(C, y_column, tol=tol)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: cg() got an unexpected keyword argument 'tol'


## === cell 8
y_pred = grid.predict(X_val).round().astype(int)
y_pred = np.clip(y_pred, np.min(y_val), np.max(y_val))  # Ensure valid range
kappa = quadratic_weighted_kappa(y_val, y_pred)
print(f'Quadratic Weighted Kappa: {kappa:.4f}')


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/1228192904.py in <cell line: 0>()
      1 # Evaluation
----> 2 y_pred = grid.predict(X_val).round().astype(int)
      3 y_pred = np.clip(y_pred, np.min(y_val), np.max(y_val))  # Ensure valid range
      4 kappa = quadratic_weighted_kappa(y_val, y_pred)
      5 print(f'Quadratic Weighted Kappa: {kappa:.4f}')

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in predict(self, X)
    496             the best found parameters.
    497         """
--> 498         check_is_fitted(self)
    499         return self.best_estimator_.predict(X)
    500 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This GridSearchCV instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

## === cell 9
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred, alpha=0.6, color='teal')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Scores', fontsize=16)
plt.xlabel('Actual Scores', fontsize=14)
plt.ylabel('Predicted Scores', fontsize=14)
plt.grid(True)
plt.show()


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2937005219.py in <cell line: 0>()
      1 # Actual vs Predicted Plot
      2 plt.figure(figsize=(10, 6))
----> 3 plt.scatter(y_val, y_pred, alpha=0.6, color='teal')
      4 plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
      5 plt.title('Actual vs Predicted Scores', fontsize=16)

NameError: name 'y_pred' is not defined

## === cell 10
X_test = vectorizer.transform(test_df['full_text'])
test_df['score'] = grid.predict(X_test).round().astype(int)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/3275748780.py in <cell line: 0>()
      1 # Predict Test Data
      2 X_test = vectorizer.transform(test_df['full_text'])
----> 3 test_df['score'] = grid.predict(X_test).round().astype(int)

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py in predict(self, X)
    496             the best found parameters.
    497         """
--> 498         check_is_fitted(self)
    499         return self.best_estimator_.predict(X)
    500 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This GridSearchCV instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

## === cell 11
test_df


## === cell 12
submit = pd.DataFrame(
    {
        'essay_id': test_df['essay_id'],
        'score': test_df['score'] # torch.argmax(outputs.logits, axis=1).to('cpu') + 1
    }
)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'score'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1979991674.py in <cell line: 0>()
      2     {
      3         'essay_id': test_df['essay_id'],
----> 4         'score': test_df['score'] # torch.argmax(outputs.logits, axis=1).to('cpu') + 1
      5     }
      6 )

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'score'

## === cell 13
submit


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3631037906.py in <cell line: 0>()
----> 1 submit

NameError: name 'submit' is not defined

## === cell 16
submit.to_csv('/kaggle/working/submission.csv', index=False)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1467593008.py in <cell line: 0>()
----> 1 submit.to_csv('/kaggle/working/submission.csv', index=False)

NameError: name 'submit' is not defined
