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

3.12

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
plotly==5.24.1
plotly-express==0.4.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
shap==0.44.1
shapely==2.1.2
sklearn-pandas==2.2.0

# 3. Data file paths

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

# 4. Code solution

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
model = Ridge(solver="sag")
params = {"alpha": [0.1, 1.0, 10.0]}
scorer = make_scorer(quadratic_weighted_kappa)
grid = GridSearchCV(model, param_grid=params, scoring=scorer, cv=5)
grid.fit(X_train, y_train)


## --- ERROR in cell 7, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2359474158.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      5[0m [0mscorer[0m [0;34m=[0m [0mmake_scorer[0m[0;34m([0m[0mquadratic_weighted_kappa[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0mgrid[0m [0;34m=[0m [0mGridSearchCV[0m[0;34m([0m[0mmodel[0m[0;34m,[0m [0mparam_grid[0m[0;34m=[0m[0mparams[0m[0;34m,[0m [0mscoring[0m[0;34m=[0m[0mscorer[0m[0;34m,[0m [0mcv[0m[0;34m=[0m[0;36m5[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 7[0;31m [0mgrid[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX_train[0m[0;34m,[0m [0my_train[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py[0m in [0;36mfit[0;34m(self, X, y, groups, **fit_params)[0m
[1;32m    872[0m                 [0;32mreturn[0m [0mresults[0m[0;34m[0m[0;34m[0m[0m
[1;32m    873[0m [0;34m[0m[0m
[0;32m--> 874[0;31m             [0mself[0m[0;34m.[0m[0m_run_search[0m[0;34m([0m[0mevaluate_candidates[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    875[0m [0;34m[0m[0m
[1;32m    876[0m             [0;31m# multimetric is determined here because in the case of a callable[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py[0m in [0;36m_run_search[0;34m(self, evaluate_candidates)[0m
[1;32m   1386[0m     [0;32mdef[0m [0m_run_search[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mevaluate_candidates[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1387[0m         [0;34m"""Search all candidates in param_grid"""[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1388[0;31m         [0mevaluate_candidates[0m[0;34m([0m[0mParameterGrid[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mparam_grid[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1389[0m [0;34m[0m[0m
[1;32m   1390[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_search.py[0m in [0;36mevaluate_candidates[0;34m(candidate_params, cv, more_results)[0m
[1;32m    849[0m                     )
[1;32m    850[0m [0;34m[0m[0m
[0;32m--> 851[0;31m                 [0m_warn_or_raise_about_fit_failures[0m[0;34m([0m[0mout[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0merror_score[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    852[0m [0;34m[0m[0m
[1;32m    853[0m                 [0;31m# For callable self.scoring, the return type is only know after[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_validation.py[0m in [0;36m_warn_or_raise_about_fit_failures[0;34m(results, error_score)[0m
[1;32m    365[0m                 [0;34mf"Below are more details about the failures:\n{fit_errors_summary}"[0m[0;34m[0m[0;34m[0m[0m
[1;32m    366[0m             )
[0;32m--> 367[0;31m             [0;32mraise[0m [0mValueError[0m[0;34m([0m[0mall_fits_failed_message[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    368[0m [0;34m[0m[0m
[1;32m    369[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: 
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
