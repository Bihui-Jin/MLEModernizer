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
Build a model that recognizes toxicity and minimizes unintended bias with respect to mentions of identities.

## Metric
We combine several submetrics: An overall ROC-AUC for the full evaluation set, along with the ROC-AUCs on three specific subsets of the test set capturing different aspects of bias.

The final model score looks like:

$$
\text { score }=w_0 A U C_{\text {overall }}+\sum_{a=1}^A w_a M_p\left(m_{s, a}\right)
$$
where:
$A=$ number of submetrics $(3)$
$m_{s, a}=$ bias metric for identity subgroup $s$ using submetric $a$
$w_a=$ a weighting for the relative importance of each submetric; all four $w$ values set to 0.25

Overall AUC: This is the ROC-AUC for the full evaluation set.

### Bias AUCs
To measure unintended bias, we again calculate the ROC-AUC, this time on three specific subsets of the test set for each identity, each capturing a different aspect of unintended bias. 

**Subgroup AUC**: Here, we restrict the data set to only the examples that mention the specific identity subgroup. *A low value in this metric means the model does a poor job of distinguishing between toxic and non-toxic comments that mention the identity*.

**BPSN (Background Positive, Subgroup Negative) AUC**: Here, we restrict the test set to the non-toxic examples that mention the identity and the toxic examples that do not. *A low value in this metric means that the model confuses non-toxic examples that mention the identity with toxic examples that do not*, likely meaning that the model predicts higher toxicity scores than it should for non-toxic examples mentioning the identity.

**BNSP (Background Negative, Subgroup Positive) AUC**: Here, we restrict the test set to the toxic examples that mention the identity and the non-toxic examples that do not. *A low value here means that the model confuses toxic examples that mention the identity with non-toxic examples that do not*, likely meaning that the model predicts lower toxicity scores than it should for toxic examples mentioning the identity.

#### Generalized Mean of Bias AUCs
To combine the per-identity Bias AUCs into one overall measure, we calculate their generalized mean as defined below:

$$
M_p\left(m_s\right)=\left(\frac{1}{N} \sum_{s=1}^N m_s^p\right)^{\frac{1}{p}}
$$

where:
$M_p=$ the $p$ th power-mean function
$m_s=$ the bias metric $m$ calulated for subgroup $S$
$N=$ number of identity subgroups

For this competition, we use a $p$ value of -5 to encourage competitors to improve the model for the identity subgroups with the lowest model performance.

## Submission Format
```
id,prediction
7000000,0.0
7000001,0.0
etc.

```

## Dataset
The text of the individual comment is found in the `comment_text` column. Each comment in Train has a toxicity label (`target`), and models should predict the `target` toxicity for the Test data. This attribute (and all others) are fractional values which represent the fraction of human raters who believed the attribute applied to the given comment. For evaluation, test set examples with `target >= 0.5` will be considered to be in the positive class (toxic).

The data also has several additional toxicity subtype attributes. Models do not need to predict these attributes for the competition, they are included as an additional avenue for research. Subtype attributes are:

- severe_toxicity
- obscene
- threat
- insult
- identity_attack
- sexual_explicit

Additionally, a subset of comments have been labelled with a variety of identity attributes, representing the identities that are *mentioned* in the comment. The columns corresponding to identity attributes are listed below. Only identities shown below will be included in the evaluation calculation.

- **male**
- **female**
- **homosexual_gay_or_lesbian**
- **christian**
- **jewish**
- **muslim**
- **black**
- **white**
- **psychiatric_or_mental_illness**

### Files
- **train.csv** - the training set, which includes toxicity labels and subgroups
- **test.csv** - the test set, which does **not** include toxicity labels or subgroups
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

eli5==0.13.0
geopandas==0.14.4
imbalanced-learn==0.13.0
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
            description.md (286 lines)
            sample_submission.csv (97321 lines)
            sample_submission.csv.zip (230.8 kB)
            test.csv (205781 lines)
            test.csv.zip (12.5 MB)
            train.csv (3820210 lines)
            train.csv.zip (285.9 MB)
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
        input/
            description.md (286 lines)
            sample_submission.csv (97321 lines)
            sample_submission.csv.zip (230.8 kB)
            test.csv (205781 lines)
            test.csv.zip (12.5 MB)
            train.csv (3820210 lines)
            train.csv.zip (285.9 MB)
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
        working/
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
```

-> data/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv has 97320 rows and 2 columns.
The columns are: id, prediction

-> data/jigsaw-unintended-bias-in-toxicity-classification/test.csv has 205780 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-unintended-bias-in-toxicity-classification/train.csv has 3820209 rows and 45 columns.
The columns are: id, target, comment_text, severe_toxicity, obscene, identity_attack, insult, threat, asian, atheist, bisexual, black, buddhist, christian, female... and 30 more columns

-> data/sample_submission.csv has 97320 rows and 2 columns.
The columns are: id, prediction

-> data/test.csv has 205780 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 3820209 rows and 45 columns.
The columns are: id, target, comment_text, severe_toxicity, obscene, identity_attack, insult, threat, asian, atheist, bisexual, black, buddhist, christian, female... and 30 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.79277

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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train =pd.read_csv('/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test = pd.read_csv('/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
df_test = test['comment_text']
df = train[['target','comment_text']]


## === cell 2
m = df['comment_text']
m.shape


## === cell 3
from imblearn.under_sampling import RandomUnderSampler
X = df['comment_text']
X = X.values.reshape(-1,1)
y = df['target']
y = np.where(y>=0.5,1.,0.)
undersample = RandomUnderSampler(sampling_strategy='majority')
X_sample, y_sample = undersample.fit_resample(X, y)


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1697808178.py in <cell line: 0>()
----> 1 from imblearn.under_sampling import RandomUnderSampler
      2 X = df['comment_text']
      3 X = X.values.reshape(-1,1)
      4 y = df['target']
      5 y = np.where(y>=0.5,1.,0.)

/usr/local/lib/python3.11/dist-packages/imblearn/__init__.py in <module>
     50     # process, as it may not be compiled yet
     51 else:
---> 52     from . import (
     53         combine,
     54         ensemble,

/usr/local/lib/python3.11/dist-packages/imblearn/combine/__init__.py in <module>
      3 """
      4 
----> 5 from ._smote_enn import SMOTEENN
      6 from ._smote_tomek import SMOTETomek
      7 

/usr/local/lib/python3.11/dist-packages/imblearn/combine/_smote_enn.py in <module>
     10 from sklearn.utils import check_X_y
     11 
---> 12 from ..base import BaseSampler
     13 from ..over_sampling import SMOTE
     14 from ..over_sampling.base import BaseOverSampler

/usr/local/lib/python3.11/dist-packages/imblearn/base.py in <module>
     10 from sklearn.base import BaseEstimator, OneToOneFeatureMixin
     11 from sklearn.preprocessing import label_binarize
---> 12 from sklearn.utils._metadata_requests import METHODS
     13 from sklearn.utils.multiclass import check_classification_targets
     14 

ModuleNotFoundError: No module named 'sklearn.utils._metadata_requests'

## === cell 4
unique, counts = np.unique(y_sample, return_counts=True)
unique, counts


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3719114638.py in <cell line: 0>()
----> 1 unique, counts = np.unique(y_sample, return_counts=True)
      2 unique, counts

NameError: name 'y_sample' is not defined

## === cell 5
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_sample.reshape(-1), y_sample, 
                                                    test_size=0.3, random_state=42)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2807539679.py in <cell line: 0>()
      1 from sklearn.model_selection import train_test_split
----> 2 x_train, x_test, y_train, y_test = train_test_split(X_sample.reshape(-1), y_sample, 
      3                                                     test_size=0.3, random_state=42)

NameError: name 'X_sample' is not defined

## === cell 6
x_train.shape


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2758771984.py in <cell line: 0>()
----> 1 x_train.shape

NameError: name 'x_train' is not defined

## === cell 7
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

vec = CountVectorizer()
clf = LogisticRegression()
pipe = make_pipeline(vec, clf)
pipe.fit(x_train, y_train);


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3111572446.py in <cell line: 0>()
      7 clf = LogisticRegression()
      8 pipe = make_pipeline(vec, clf)
----> 9 pipe.fit(x_train, y_train);

NameError: name 'x_train' is not defined

## === cell 8
from sklearn import metrics

def print_report(pipe, x_test, y_test):
    y_pred = pipe.predict(x_test)
    report = metrics.classification_report(y_test, y_pred)
    print(report)
    print("accuracy: {:0.3f}".format(metrics.accuracy_score(y_test, y_pred)))

print_report(pipe, x_test, y_test)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3652637067.py in <cell line: 0>()
      7     print("accuracy: {:0.3f}".format(metrics.accuracy_score(y_test, y_pred)))
      8 
----> 9 print_report(pipe, x_test, y_test)

NameError: name 'x_test' is not defined

## === cell 9
import eli5
eli5.show_weights(clf, vec=vec, top=20)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 10
for _, row in df.sample(5).iterrows():
    print(f"true label: {row['target']}")
    display(eli5.show_prediction(clf, row['comment_text'], vec=vec,))
    print("--"*50)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1451329841.py in <cell line: 0>()
      1 for _, row in df.sample(5).iterrows():
      2     print(f"true label: {row['target']}")
----> 3     display(eli5.show_prediction(clf, row['comment_text'], vec=vec,))
      4     print("--"*50)

/usr/local/lib/python3.11/dist-packages/eli5/ipython.py in show_prediction(estimator, doc, **kwargs)
    305     """
    306     format_kwargs, explain_kwargs = _split_kwargs(kwargs)
--> 307     expl = explain_prediction(estimator, doc, **explain_kwargs)
    308     if expl.image is not None:
    309         # dispatch to image display implementation

/usr/lib/python3.11/functools.py in wrapper(*args, **kw)
    907                             '1 positional argument')
    908 
--> 909         return dispatch(args[0].__class__)(*args, **kw)
    910 
    911     funcname = getattr(func, '__name__', 'singledispatch function')

/usr/local/lib/python3.11/dist-packages/eli5/sklearn/explain_prediction.py in explain_prediction_linear_classifier(clf, doc, vec, top, top_targets, target_names, targets, feature_names, feature_re, feature_filter, vectorized)
    167     is already vectorized.
    168     """
--> 169     vec, feature_names = handle_vec(clf, doc, vec, vectorized, feature_names)
    170     X = get_X(doc, vec=vec, vectorized=vectorized, to_dense=True)
    171 

/usr/local/lib/python3.11/dist-packages/eli5/sklearn/utils.py in handle_vec(clf, doc, vec, vectorized, feature_names, num_features)
    259     feature_names = handle_hashing_vec(
    260         vec, feature_names, coef_scale=None, with_coef_scale=False)
--> 261     feature_names = get_feature_names(
    262         clf, vec, feature_names=feature_names, num_features=num_features)
    263     return vec, feature_names

/usr/local/lib/python3.11/dist-packages/eli5/sklearn/utils.py in get_feature_names(clf, vec, bias_name, feature_names, num_features, estimator_feature_names)
     85         else:
     86             if estimator_feature_names is None:
---> 87                 num_features = num_features or get_num_features(clf)
     88                 return FeatureNames(
     89                     n_features=num_features,

/usr/local/lib/python3.11/dist-packages/eli5/sklearn/utils.py in get_num_features(estimator)
    210         return get_num_features(estimator.estimators_[0])
    211     else:
--> 212         raise ValueError("Can't figure out feature vector size for %s" %
    213                          estimator)
    214 

ValueError: Can't figure out feature vector size for LogisticRegression()

## === cell 11
vec2 = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=.01, max_df=.5)
clf2 = LinearSVC()
pipe_tfidf = make_pipeline(vec2, clf2)
pipe_tfidf.fit(x_train, y_train)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1192594299.py in <cell line: 0>()
      2 clf2 = LinearSVC()
      3 pipe_tfidf = make_pipeline(vec2, clf2)
----> 4 pipe_tfidf.fit(x_train, y_train)

NameError: name 'x_train' is not defined

## === cell 12
print_report(pipe_tfidf, x_test, y_test)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3634445401.py in <cell line: 0>()
----> 1 print_report(pipe_tfidf, x_test, y_test)

NameError: name 'x_test' is not defined

## === cell 13
eli5.show_weights(clf2, vec=vec2, top=20)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2597601980.py in <cell line: 0>()
----> 1 eli5.show_weights(clf2, vec=vec2, top=20)

/usr/local/lib/python3.11/dist-packages/eli5/ipython.py in show_weights(estimator, **kwargs)
    128     """
    129     format_kwargs, explain_kwargs = _split_kwargs(kwargs)
--> 130     expl = explain_weights(estimator, **explain_kwargs)
    131     _set_html_kwargs_defaults(format_kwargs)
    132     html = format_as_html(expl, **format_kwargs)

/usr/lib/python3.11/functools.py in wrapper(*args, **kw)
    907                             '1 positional argument')
    908 
--> 909         return dispatch(args[0].__class__)(*args, **kw)
    910 
    911     funcname = getattr(func, '__name__', 'singledispatch function')

/usr/local/lib/python3.11/dist-packages/eli5/sklearn/explain_weights.py in explain_linear_classifier_weights(clf, vec, top, target_names, targets, feature_names, coef_scale, feature_re, feature_filter)
    216     feature_names, coef_scale = handle_hashing_vec(vec, feature_names,
    217                                                    coef_scale)
--> 218     feature_names, flt_indices = get_feature_names_filtered(
    219         clf, vec,
    220         feature_names=feature_names,

/usr/local/lib/python3.11/dist-packages/eli5/sklearn/utils.py in get_feature_names_filtered(clf, vec, bias_name, feature_names, num_features, feature_filter, feature_re, estimator_feature_names)
    118                                estimator_feature_names=None):
    119     # type: (...) -> Tuple[FeatureNames, List[int]]
--> 120     feature_names = get_feature_names(
    121         clf=clf,
    122         vec=vec,

/usr/local/lib/python3.11/dist-packages/eli5/sklearn/utils.py in get_feature_names(clf, vec, bias_name, feature_names, num_features, estimator_feature_names)
     85         else:
     86             if estimator_feature_names is None:
---> 87                 num_features = num_features or get_num_features(clf)
     88                 return FeatureNames(
     89                     n_features=num_features,

/usr/local/lib/python3.11/dist-packages/eli5/sklearn/utils.py in get_num_features(estimator)
    210         return get_num_features(estimator.estimators_[0])
    211     else:
--> 212         raise ValueError("Can't figure out feature vector size for %s" %
    213                          estimator)
    214 

ValueError: Can't figure out feature vector size for LinearSVC()

## === cell 14
for _, row in df.sample(5).iterrows():
    print(f"true label: {row['target']}")
    display(eli5.show_prediction(clf2, row['comment_text'], vec=vec2,))
    print("--"*50)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3715718323.py in <cell line: 0>()
      1 for _, row in df.sample(5).iterrows():
      2     print(f"true label: {row['target']}")
----> 3     display(eli5.show_prediction(clf2, row['comment_text'], vec=vec2,))
      4     print("--"*50)

/usr/local/lib/python3.11/dist-packages/eli5/ipython.py in show_prediction(estimator, doc, **kwargs)
    305     """
    306     format_kwargs, explain_kwargs = _split_kwargs(kwargs)
--> 307     expl = explain_prediction(estimator, doc, **explain_kwargs)
    308     if expl.image is not None:
    309         # dispatch to image display implementation

/usr/lib/python3.11/functools.py in wrapper(*args, **kw)
    907                             '1 positional argument')
    908 
--> 909         return dispatch(args[0].__class__)(*args, **kw)
    910 
    911     funcname = getattr(func, '__name__', 'singledispatch function')

/usr/local/lib/python3.11/dist-packages/eli5/sklearn/explain_prediction.py in explain_prediction_linear_classifier(clf, doc, vec, top, top_targets, target_names, targets, feature_names, feature_re, feature_filter, vectorized)
    167     is already vectorized.
    168     """
--> 169     vec, feature_names = handle_vec(clf, doc, vec, vectorized, feature_names)
    170     X = get_X(doc, vec=vec, vectorized=vectorized, to_dense=True)
    171 

/usr/local/lib/python3.11/dist-packages/eli5/sklearn/utils.py in handle_vec(clf, doc, vec, vectorized, feature_names, num_features)
    259     feature_names = handle_hashing_vec(
    260         vec, feature_names, coef_scale=None, with_coef_scale=False)
--> 261     feature_names = get_feature_names(
    262         clf, vec, feature_names=feature_names, num_features=num_features)
    263     return vec, feature_names

/usr/local/lib/python3.11/dist-packages/eli5/sklearn/utils.py in get_feature_names(clf, vec, bias_name, feature_names, num_features, estimator_feature_names)
     85         else:
     86             if estimator_feature_names is None:
---> 87                 num_features = num_features or get_num_features(clf)
     88                 return FeatureNames(
     89                     n_features=num_features,

/usr/local/lib/python3.11/dist-packages/eli5/sklearn/utils.py in get_num_features(estimator)
    210         return get_num_features(estimator.estimators_[0])
    211     else:
--> 212         raise ValueError("Can't figure out feature vector size for %s" %
    213                          estimator)
    214 

ValueError: Can't figure out feature vector size for LinearSVC()

## === cell 15
y_pred= pipe.predict(df_test)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/2516971229.py in <cell line: 0>()
----> 1 y_pred= pipe.predict(df_test)

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in predict(self, X, **predict_params)
    478         Xt = X
    479         for _, name, transform in self._iter(with_final=False):
--> 480             Xt = transform.transform(Xt)
    481         return self.steps[-1][1].predict(Xt, **predict_params)
    482 

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in transform(self, raw_documents)
   1428                 "Iterable over raw text documents expected, string object received."
   1429             )
-> 1430         self._check_vocabulary()
   1431 
   1432         # use the same matrix-building strategy as fit_transform

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in _check_vocabulary(self)
    508             self._validate_vocabulary()
    509             if not self.fixed_vocabulary_:
--> 510                 raise NotFittedError("Vocabulary not fitted or provided")
    511 
    512         if len(self.vocabulary_) == 0:

NotFittedError: Vocabulary not fitted or provided

## === cell 16
y_pred = np.where(y_pred>=0.5,1.,0.)
sub_df = pd.read_csv('/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')
sub_df['prediction'] = y_pred
sub_df.to_csv('submission.csv',index = False)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1864292972.py in <cell line: 0>()
----> 1 y_pred = np.where(y_pred>=0.5,1.,0.)
      2 sub_df = pd.read_csv('/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')
      3 sub_df['prediction'] = y_pred
      4 sub_df.to_csv('submission.csv',index = False)

NameError: name 'y_pred' is not defined

## === cell 17
y_pred_2= pipe_tfidf.predict(df_test)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/2220718724.py in <cell line: 0>()
----> 1 y_pred_2= pipe_tfidf.predict(df_test)

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in predict(self, X, **predict_params)
    478         Xt = X
    479         for _, name, transform in self._iter(with_final=False):
--> 480             Xt = transform.transform(Xt)
    481         return self.steps[-1][1].predict(Xt, **predict_params)
    482 

/usr/local/lib/python3.11/dist-packages/sklearn/feature_extraction/text.py in transform(self, raw_documents)
   2153             Tf-idf-weighted document-term matrix.
   2154         """
-> 2155         check_is_fitted(self, msg="The TF-IDF vectorizer is not fitted")
   2156 
   2157         X = super().transform(raw_documents)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: The TF-IDF vectorizer is not fitted

## === cell 18
y_pred_2 = np.where(y_pred_2>=0.5,1.,0.)
sub_df = pd.read_csv('/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')
sub_df['prediction'] = y_pred_2
sub_df.to_csv('submission_2.csv',index = False)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2241376228.py in <cell line: 0>()
----> 1 y_pred_2 = np.where(y_pred_2>=0.5,1.,0.)
      2 sub_df = pd.read_csv('/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')
      3 sub_df['prediction'] = y_pred_2
      4 sub_df.to_csv('submission_2.csv',index = False)

NameError: name 'y_pred_2' is not defined
