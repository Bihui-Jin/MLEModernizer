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

3.5

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (70 lines)
            images.zip (22.0 MB)
            sample_submission.csv (100 lines)
            sample_submission.csv.zip (2.3 kB)
            test.csv (100 lines)
            test.csv.zip (39.3 kB)
            train.csv (892 lines)
            train.csv.zip (357.1 kB)
            images/
                42.jpg (32.6 kB)
                168.jpg (16.5 kB)
                ... and 988 other files
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
        input/
            description.md (70 lines)
            images.zip (22.0 MB)
            sample_submission.csv (100 lines)
            sample_submission.csv.zip (2.3 kB)
            test.csv (100 lines)
            test.csv.zip (39.3 kB)
            train.csv (892 lines)
            train.csv.zip (357.1 kB)
            images/
                42.jpg (32.6 kB)
                168.jpg (16.5 kB)
                ... and 988 other files
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
        working/
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
```

-> data/leaf-classification/sample_submission.csv has 99 rows and 100 columns.
The columns are: id, Acer_Capillipes, Acer_Circinatum, Acer_Mono, Acer_Opalus, Acer_Palmatum, Acer_Pictum, Acer_Platanoids, Acer_Rubrum, Acer_Rufinerve, Acer_Saccharinum, Alnus_Cordata, Alnus_Maximowiczii, Alnus_Rubra, Alnus_Sieboldiana... and 85 more columns

-> data/leaf-classification/test.csv has 99 rows and 193 columns.
The columns are: id, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13, margin14... and 178 more columns

-> data/leaf-classification/train.csv has 891 rows and 194 columns.
The columns are: id, species, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13... and 179 more columns

-> data/sample_submission.csv has 99 rows and 100 columns.
The columns are: id, Acer_Capillipes, Acer_Circinatum, Acer_Mono, Acer_Opalus, Acer_Palmatum, Acer_Pictum, Acer_Platanoids, Acer_Rubrum, Acer_Rufinerve, Acer_Saccharinum, Alnus_Cordata, Alnus_Maximowiczii, Alnus_Rubra, Alnus_Sieboldiana... and 85 more columns

-> data/test.csv has 99 rows and 193 columns.
The columns are: id, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13, margin14... and 178 more columns

-> data/train.csv has 891 rows and 194 columns.
The columns are: id, species, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13... and 179 more columns

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import time
import pandas as pd
import numpy as np
from pandas import DataFrame, Series

from sklearn import (
    linear_model,
    feature_selection,
    manifold,
    decomposition,
    random_projection,
)
from sklearn.model_selection import learning_curve, StratifiedKFold

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import (
    GradientBoostingClassifier,
    BaggingRegressor,
    RandomForestClassifier,
)
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import log_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib.pyplot as plt
import seaborn as sns

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass


## === cell 1
train_df = pd.read_csv('../input/train.csv')
train_df.fillna(0,inplace=True)
train_df


## === cell 2
le = LabelEncoder().fit(train_df.species)
labels = le.transform(train_df.species)
labels


## === cell 3
df = train_df.copy()
df.species = labels
df.species


## === cell 4
df.iloc[:, 2:] = MinMaxScaler().fit_transform(train_df.iloc[:, 2:])
df


## === cell 5
X = df.to_numpy()[:, 2:]
y = df.to_numpy()[:, 1]


## === cell 6
def plot_learning_curve(estimator, title, X, y, ylim=(0.4,1.1), cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    """
    start_time = time.time()
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=3, n_jobs=1, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    
    plt.annotate(test_scores_mean[-1],xy=(train_sizes[-1],test_scores_mean[-1]))

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid("on") 
    if ylim:
        plt.ylim(ylim)
    plt.title(title+'(time=%fs)'%(time.time()-start_time))
    plt.show()


## === cell 7
rf = RandomForestClassifier(n_estimators=5)
plot_learning_curve(rf,'RandomForestClassifier',X,y)


## === cell 8
lr = linear_model.LogisticRegression()
plot_learning_curve(lr,'LogisticRegression',X,y)


## === cell 9
lsvc = LinearSVC()
plot_learning_curve(lsvc,'LinearSVC',X,y)


## === cell 10
svc = SVC(probability=True,kernel='rbf',C=0.025)
plot_learning_curve(svc,'SVC',X,y)


## === cell 11
lda = LinearDiscriminantAnalysis()
plot_learning_curve(lda,'LinearDiscriminantAnalysis',X,y)


## === cell 12
def cal_log_loss(estimator):
    loss = []
    split = StratifiedKFold(y)
    for train_index, test_index in split:
        X_train,X_test = X[train_index],X[test_index]
        y_train,y_test = y[train_index],y[test_index]
        estimator.fit(X_train,y_train)
        y_prediction = estimator.predict_proba(X_test)
        loss.append(log_loss(y_test,y_prediction))
    return np.mean(loss)
    


## === cell 13
def cal_log_loss(estimator):
    loss = []
    split = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for train_index, test_index in split.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        estimator.fit(X_train, y_train)
        y_prediction = estimator.predict_proba(X_test)
        loss.append(log_loss(y_test, y_prediction))
    return np.mean(loss)


## === cell 14
estimator = lda
estimator.fit(X,y)


## === cell 15
test_data = pd.read_csv("../input/test.csv")

test_df = DataFrame(MinMaxScaler().fit_transform(test_data.iloc[:, 1:]))
test_df


## === cell 16
predict = estimator.predict_proba(test_df.to_numpy())
result = DataFrame(predict, columns=sorted(train_df.species.unique()))
result


## === cell 17
result.insert(0,'id',test_data.id)
result.to_csv('result.csv',index=False)


## === cell 19
import time
import pandas as pd
import numpy as np
from pandas import DataFrame, Series

from sklearn import (
    linear_model,
    feature_selection,
    manifold,
    decomposition,
    random_projection,
)
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import (
    GradientBoostingClassifier,
    BaggingRegressor,
    RandomForestClassifier,
)
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import log_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib.pyplot as plt
import seaborn as sns

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass


## === cell 20
train_df = pd.read_csv('../input/train.csv')
train_df.fillna(0,inplace=True)
train_df


## === cell 21
le = LabelEncoder().fit(train_df.species)
labels = le.transform(train_df.species)
labels


## === cell 22
df = train_df.copy()
df.species = labels
df.species


## === cell 23
df.iloc[:, 2:] = MinMaxScaler().fit_transform(train_df.iloc[:, 2:])
df


## === cell 24
arr = df.to_numpy()
X = arr[:, 2:]
y = arr[:, 1]


## === cell 25
def plot_learning_curve(estimator, title, X, y, ylim=(0.4,1.1), cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    """
    start_time = time.time()
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=3, n_jobs=1, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    
    plt.annotate(test_scores_mean[-1],xy=(train_sizes[-1],test_scores_mean[-1]))

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid("on") 
    if ylim:
        plt.ylim(ylim)
    plt.title(title+'(time=%fs)'%(time.time()-start_time))
    plt.show()


## === cell 26
rf = RandomForestClassifier(n_estimators=5)
plot_learning_curve(rf,'RandomForestClassifier',X,y)


## === cell 27
lr = linear_model.LogisticRegression()
plot_learning_curve(lr,'LogisticRegression',X,y)


## === cell 28
lsvc = LinearSVC()
plot_learning_curve(lsvc,'LinearSVC',X,y)


## === cell 29
svc = SVC(probability=True,kernel='rbf',C=0.025)
plot_learning_curve(svc,'SVC',X,y)


## === cell 30
lda = LinearDiscriminantAnalysis()
plot_learning_curve(lda,'LinearDiscriminantAnalysis',X,y)


## === cell 31
def cal_log_loss(estimator):
    loss = []
    split = StratifiedKFold(y)
    for train_index, test_index in split:
        X_train,X_test = X[train_index],X[test_index]
        y_train,y_test = y[train_index],y[test_index]
        estimator.fit(X_train,y_train)
        y_prediction = estimator.predict_proba(X_test)
        loss.append(log_loss(y_test,y_prediction))
    return np.mean(loss)
    


## === cell 32
cal_log_loss(lr)
cal_log_loss(rf)
cal_log_loss(lda)


## --- ERROR in cell 32, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4247190043.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mcal_log_loss[0m[0;34m([0m[0mlr[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mcal_log_loss[0m[0;34m([0m[0mrf[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0mcal_log_loss[0m[0;34m([0m[0mlda[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/660066412.py[0m in [0;36mcal_log_loss[0;34m(estimator)[0m
[1;32m      1[0m [0;32mdef[0m [0mcal_log_loss[0m[0;34m([0m[0mestimator[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m     [0mloss[0m [0;34m=[0m [0;34m[[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 3[0;31m     [0msplit[0m [0;34m=[0m [0mStratifiedKFold[0m[0;34m([0m[0my[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m     [0;32mfor[0m [0mtrain_index[0m[0;34m,[0m [0mtest_index[0m [0;32min[0m [0msplit[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m         [0mX_train[0m[0;34m,[0m[0mX_test[0m [0;34m=[0m [0mX[0m[0;34m[[0m[0mtrain_index[0m[0;34m][0m[0;34m,[0m[0mX[0m[0;34m[[0m[0mtest_index[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py[0m in [0;36m__init__[0;34m(self, n_splits, shuffle, random_state)[0m
[1;32m    666[0m [0;34m[0m[0m
[1;32m    667[0m     [0;32mdef[0m [0m__init__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mn_splits[0m[0;34m=[0m[0;36m5[0m[0;34m,[0m [0;34m*[0m[0;34m,[0m [0mshuffle[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m [0mrandom_state[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 668[0;31m         [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__init__[0m[0;34m([0m[0mn_splits[0m[0;34m=[0m[0mn_splits[0m[0;34m,[0m [0mshuffle[0m[0;34m=[0m[0mshuffle[0m[0;34m,[0m [0mrandom_state[0m[0;34m=[0m[0mrandom_state[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    669[0m [0;34m[0m[0m
[1;32m    670[0m     [0;32mdef[0m [0m_make_test_folds[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mX[0m[0;34m,[0m [0my[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py[0m in [0;36m__init__[0;34m(self, n_splits, shuffle, random_state)[0m
[1;32m    289[0m     [0;32mdef[0m [0m__init__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mn_splits[0m[0;34m,[0m [0;34m*[0m[0;34m,[0m [0mshuffle[0m[0;34m,[0m [0mrandom_state[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    290[0m         [0;32mif[0m [0;32mnot[0m [0misinstance[0m[0;34m([0m[0mn_splits[0m[0;34m,[0m [0mnumbers[0m[0;34m.[0m[0mIntegral[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 291[0;31m             raise ValueError(
[0m[1;32m    292[0m                 [0;34m"The number of folds must be of Integral type. "[0m[0;34m[0m[0;34m[0m[0m
[1;32m    293[0m                 [0;34m"%s of type %s was passed."[0m [0;34m%[0m [0;34m([0m[0mn_splits[0m[0;34m,[0m [0mtype[0m[0;34m([0m[0mn_splits[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: The number of folds must be of Integral type. [10. 93. 67.  6. 79. 28. 55. 98. 62. 40.  4.  3. 86. 37. 88. 23. 33. 53.
 66.  8. 22. 85.  2. 23. 50. 21. 33. 16. 53. 28. 66. 54. 35. 91. 21. 78.
 74.  6. 65. 83. 12. 19. 36. 10. 53. 63. 61. 69. 67. 49. 52. 16. 19. 26.
 51. 36. 94. 71. 13.  5. 71. 40. 35. 76. 23. 28. 42. 42.  9. 56. 38.  0.
 40. 83. 95. 96. 61. 28. 30. 37. 96. 40. 26. 26.  1. 63. 10. 71.  3. 17.
 90. 85. 19. 16. 82. 59. 39. 13.  4. 92. 25. 39. 97. 96. 65.  1. 84. 52.
  7. 27. 54. 26. 54. 54. 75. 57. 59. 24. 70. 51. 34. 11. 60. 76. 83. 59.
 17. 16. 33. 97. 79. 27. 65. 95.  2. 25. 30. 37. 11. 72. 28. 75. 65. 59.
 75. 19. 49. 20. 75. 15.  3.  7. 69. 95. 74. 26. 80. 61. 63. 73. 73. 83.
 64. 48. 57. 69. 42. 29. 24. 44. 41. 93. 33. 41. 33. 39. 60. 77. 93. 61.
 77. 90. 14.  9. 92. 96. 45. 40. 73. 98. 59.  4. 13. 58. 82. 36.  2. 65.
 87. 73. 89. 29. 83. 92. 79. 94. 24. 35.  9. 62. 46.  6. 94. 70. 52. 54.
 44. 31. 67. 62. 70. 79. 72. 64. 34. 25. 43. 42. 56. 30. 75. 39. 74. 23.
 55. 18. 30. 95.  6. 70.  5. 76. 33. 64. 32. 25. 10. 41.  4. 37. 25. 97.
 55. 90. 20. 17. 92. 39. 29. 57. 10. 93. 53. 81. 77. 43. 54. 35. 35. 17.
 97. 51. 62. 41. 84.  0. 60. 92. 38. 44. 28. 18. 84. 56. 49. 77. 70. 10.
 96. 31. 68. 68. 56. 41.  2. 67. 61.  8. 88. 88. 87. 35. 75. 28. 46. 84.
 72. 13. 63. 48. 10. 59.  3. 43. 63. 62. 91. 87. 20. 52. 66. 80. 63. 46.
 91. 22. 67. 32. 72. 82. 32. 39. 73. 92. 41. 41. 98. 76. 12. 64. 13. 53.
  3. 88.  9. 74. 79. 31. 85. 15. 11. 54.  4. 68.  8. 14. 73. 46. 67. 92.
 91. 80.  1. 47. 59. 21.  0.  6. 13. 66.  9. 28. 44. 76. 72. 55. 13. 15.
 77. 69. 96. 71. 54. 55. 38.  4.  4. 64. 11. 14. 74. 30. 89. 71. 36.  7.
 41. 87. 12. 22. 31. 89. 41. 78. 32. 63. 86. 67. 53. 20. 19. 54. 47. 75.
 71.  1. 35.  3. 32. 31. 91. 58. 76. 62. 98. 13. 36. 93. 65. 30.  5. 38.
 32. 85. 72. 97.  0. 46. 11. 73.  9. 64. 43. 62. 94. 56. 37. 20. 90. 64.
 23. 56. 87. 94. 43. 22. 18. 92. 32.  6. 22. 92. 24. 50. 35. 69. 45. 82.
  0. 49. 20. 17. 51. 78. 37. 47. 25. 33. 90. 50. 61. 66. 56. 12. 80. 22.
 61. 58. 58. 74. 73. 19.  9. 83. 88. 81.  1. 17. 64. 12. 34. 49. 37. 16.
 25. 20. 13. 38. 29.  2. 22.  7. 14. 62. 58. 48. 60. 81. 42. 24.  8. 58.
 84.  8. 23. 11. 89. 70. 96. 76. 58.  5. 74. 33. 47. 93. 55.  7. 32.  2.
 24. 82. 50. 49. 16. 88. 49. 87. 14. 92. 72. 64. 31. 61. 85. 74. 38.  2.
 84. 43.  5.  5. 65. 84. 48. 63.  2. 70. 68. 48.  1. 46.  5. 47. 95. 15.
 28. 55.  4. 77.  3. 52. 85. 86. 75.  6. 84. 86. 82. 44. 14. 80. 60. 12.
 31. 51. 81. 26. 94. 30. 44.  0. 89. 45. 67. 88. 68. 39. 21. 70. 43. 19.
 91. 51. 82. 30. 78.  1. 88. 33. 23. 45. 88. 86. 51. 61.  9. 10. 63. 60.
 14. 40. 83. 91. 89. 48. 94. 40. 26. 47. 87. 69. 57. 32. 84. 72. 75. 25.
 89. 68. 68. 52. 23. 80. 14. 69. 27. 19. 51. 32. 15. 10.  9. 18. 38. 27.
  0. 12. 71. 98. 68. 15. 34. 49. 48. 19. 85. 59. 18. 98. 24. 34. 57. 78.
 89. 69. 52. 36. 16.  6. 15. 50. 30.  8. 12. 46. 60. 79. 81. 39. 74. 66.
 96. 48. 93. 13. 43.  1. 44. 66.  2. 80. 18.  9. 25. 77. 66. 17. 80. 87.
 57.  7. 85. 23. 95. 81. 44. 62. 15. 81. 94. 51. 42. 95. 16. 58. 97. 34.
 21. 56. 81. 65. 14. 57. 56. 78. 83. 29. 94. 35. 91. 21. 29. 17.  1. 90.
 66. 29. 46.  0. 54. 86. 42. 29. 28. 26. 24. 80. 50. 79. 68. 21. 84. 40.
 57. 23. 56. 76. 57.  2. 47. 18. 38. 50.  3. 60. 90. 37. 27.  3. 35. 45.
 90. 52.  7. 45. 86.  8.  3. 80. 18. 64. 85.  1. 47. 26. 42. 62. 17. 77.
 21. 24. 38. 53. 53. 70. 69. 78. 95. 21. 53. 96. 36. 11. 95. 81. 73. 34.
 47. 73.  6. 46. 70. 63. 89. 67. 40. 87. 96. 82. 79. 18. 45. 46. 45. 34.
 31.  4. 81. 90.  5. 16. 34. 66. 88. 87. 98. 11. 51. 17. 93. 72. 68. 55.
 65. 79. 85. 67. 20. 36. 27. 37. 86.  8. 12. 60. 36. 43. 31.  7. 57.  7.
 89. 78. 50. 97. 82.  6. 29. 11. 71.] of type <class 'numpy.ndarray'> was passed.

## === cell 33
estimator = lda
estimator.fit(X,y)
