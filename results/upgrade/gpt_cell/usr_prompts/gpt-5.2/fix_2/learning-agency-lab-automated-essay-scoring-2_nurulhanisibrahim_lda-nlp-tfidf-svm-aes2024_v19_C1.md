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


## === cell 7
cList = {
  "ain't": "am not","aren't": "are not","can't": "cannot","can't've": "cannot have","'cause": "because",  "could've": "could have","couldn't": "could not","couldn't've": "could not have","didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not","hadn't've": "had not have","hasn't": "has not",
  "haven't": "have not","he'd": "he would","he'd've": "he would have","he'll": "he will","he'll've": "he will have","he's": "he is",
  "how'd": "how did","how'd'y": "how do you","how'll": "how will","how's": "how is","I'd": "I would","I'd've": "I would have","I'll": "I will","I'll've": "I will have","I'm": "I am","I've": "I have",
  "isn't": "is not","it'd": "it had","it'd've": "it would have","it'll": "it will", "it'll've": "it will have","it's": "it is","let's": "let us","ma'am": "madam","mayn't": "may not",
  "might've": "might have","mightn't": "might not","mightn't've": "might not have","must've": "must have","mustn't": "must not","mustn't've": "must not have","needn't": "need not","needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not","oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
  "shan't've": "shall not have","she'd": "she would","she'd've": "she would have","she'll": "she will","she'll've": "she will have","she's": "she is",
  "should've": "should have","shouldn't": "should not","shouldn't've": "should not have","so've": "so have","so's": "so is","that'd": "that would","that'd've": "that would have","that's": "that is","there'd": "there had","there'd've": "there would have","there's": "there is","they'd": "they would","they'd've": "they would have","they'll": "they will","they'll've": "they will have","they're": "they are","they've": "they have","to've": "to have","wasn't": "was not","we'd": "we had",
  "we'd've": "we would have","we'll": "we will","we'll've": "we will have","we're": "we are","we've": "we have",
  "weren't": "were not","what'll": "what will","what'll've": "what will have",
  "what're": "what are","what's": "what is","what've": "what have","when's": "when is","when've": "when have",
  "where'd": "where did","where's": "where is","where've": "where have","who'll": "who will","who'll've": "who will have","who's": "who is","who've": "who have","why's": "why is",
  "why've": "why have","will've": "will have","won't": "will not","won't've": "will not have","would've": "would have","wouldn't": "would not",
  "wouldn't've": "would not have","y'all": "you all","y'alls": "you alls","y'all'd": "you all would",
  "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you had","you'd've": "you would have","you'll": "you you will","you'll've": "you you will have","you're": "you are",  "you've": "you have"
   }


## === cell 8
c_re = re.compile('(%s)' % '|'.join(cList.keys()))


## === cell 9
def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)


## === cell 10
def removeHTML(x):
    html=re.compile(r'<.*?>')
    return html.sub(r'',x)


## === cell 11
def dataPreprocessing(x):
    x = x.apply(lambda s: s.lower()) 
    x = x.apply(removeHTML)
    x = x.apply(lambda s: re.sub("@\w+", '', s))  
    x = x.apply(lambda s: re.sub("'\d+", '', s))
    x = x.apply(lambda s: re.sub("\d+", '',s))
    x = x.apply(lambda s: re.sub("http\w+", '',s))
    x = x.apply(lambda s: re.sub(r"\s+", " ", s))
    x = x.apply(expandContractions)
    x = x.apply(lambda s: re.sub(r"\.+", ".", s))
    x = x.apply(lambda s: re.sub(r"\,+", ",", s))
    x = x.apply(lambda s: re.sub('\n', '',s))
    x = x.apply(lambda s: re.sub('[^\w\s]', '',s))
    x = x.apply(lambda s: s.strip()) 
    return x


## === cell 12
x = dataPreprocessing(train_df1['full_text'])


## === cell 13
x


## === cell 14
test_df1 = pd.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv')


## === cell 15
x0 = dataPreprocessing(test_df1['full_text'])


## === cell 16
y = train_df1.iloc[:, 2:8] 


## === cell 17
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)


## === cell 18
y_train


## === cell 19
X_train


## === cell 20
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


## === cell 21
text_vectorizer = TfidfVectorizer(
    stop_words='english',
    sublinear_tf=False,
    strip_accents='unicode',
    binary=True,
    analyzer='word',
    token_pattern=r'\w{2,}',  
    ngram_range=(1,1),
    norm='l1', 
    use_idf=False, 
    smooth_idf=False,
    max_features=100000,
    min_df=30)


## === cell 22
X_train_features = text_vectorizer.fit_transform(X_train)


## === cell 23
X_train_features


## === cell 24
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC

X_train_features = X_train_features.tocsr()
test_features = text_vectorizer.transform(X_test).tocsr()

clf = make_pipeline(
    MaxAbsScaler(),
    SVC(
        C=1.75,
        kernel="rbf",
        gamma="scale",
        decision_function_shape="ovr",
        random_state=123,
        tol=1e-5,
        shrinking=True,
        verbose=True,
        break_ties=True,
    ),
)
clf.fit(X_train_features, y_train.values.ravel())
y_pred = clf.predict(test_features)


## --- ERROR in cell 24, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3118965246.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     22[0m     ),
[1;32m     23[0m )
[0;32m---> 24[0;31m [0mclf[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX_train_features[0m[0;34m,[0m [0my_train[0m[0;34m.[0m[0mvalues[0m[0;34m.[0m[0mravel[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     25[0m [0my_pred[0m [0;34m=[0m [0mclf[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mtest_features[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py[0m in [0;36mfit[0;34m(self, X, y, **fit_params)[0m
[1;32m    399[0m         """
[1;32m    400[0m         [0mfit_params_steps[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_check_fit_params[0m[0;34m([0m[0;34m**[0m[0mfit_params[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 401[0;31m         [0mXt[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_fit[0m[0;34m([0m[0mX[0m[0;34m,[0m [0my[0m[0;34m,[0m [0;34m**[0m[0mfit_params_steps[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    402[0m         [0;32mwith[0m [0m_print_elapsed_time[0m[0;34m([0m[0;34m"Pipeline"[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m_log_message[0m[0;34m([0m[0mlen[0m[0;34m([0m[0mself[0m[0;34m.[0m[0msteps[0m[0;34m)[0m [0;34m-[0m [0;36m1[0m[0;34m)[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    403[0m             [0;32mif[0m [0mself[0m[0;34m.[0m[0m_final_estimator[0m [0;34m!=[0m [0;34m"passthrough"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py[0m in [0;36m_fit[0;34m(self, X, y, **fit_params_steps)[0m
[1;32m    357[0m                 [0mcloned_transformer[0m [0;34m=[0m [0mclone[0m[0;34m([0m[0mtransformer[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    358[0m             [0;31m# Fit or load from cache the current transformer[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 359[0;31m             X, fitted_transformer = fit_transform_one_cached(
[0m[1;32m    360[0m                 [0mcloned_transformer[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    361[0m                 [0mX[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/joblib/memory.py[0m in [0;36m__call__[0;34m(self, *args, **kwargs)[0m
[1;32m    324[0m [0;34m[0m[0m
[1;32m    325[0m     [0;32mdef[0m [0m__call__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 326[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mfunc[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    327[0m [0;34m[0m[0m
[1;32m    328[0m     [0;32mdef[0m [0mcall_and_shelve[0m[0;34m([0m[0mself[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py[0m in [0;36m_fit_transform_one[0;34m(transformer, X, y, weight, message_clsname, message, **fit_params)[0m
[1;32m    891[0m     [0;32mwith[0m [0m_print_elapsed_time[0m[0;34m([0m[0mmessage_clsname[0m[0;34m,[0m [0mmessage[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    892[0m         [0;32mif[0m [0mhasattr[0m[0;34m([0m[0mtransformer[0m[0;34m,[0m [0;34m"fit_transform"[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 893[0;31m             [0mres[0m [0;34m=[0m [0mtransformer[0m[0;34m.[0m[0mfit_transform[0m[0;34m([0m[0mX[0m[0;34m,[0m [0my[0m[0;34m,[0m [0;34m**[0m[0mfit_params[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    894[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    895[0m             [0mres[0m [0;34m=[0m [0mtransformer[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX[0m[0;34m,[0m [0my[0m[0;34m,[0m [0;34m**[0m[0mfit_params[0m[0;34m)[0m[0;34m.[0m[0mtransform[0m[0;34m([0m[0mX[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py[0m in [0;36mwrapped[0;34m(self, X, *args, **kwargs)[0m
[1;32m    138[0m     [0;34m@[0m[0mwraps[0m[0;34m([0m[0mf[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    139[0m     [0;32mdef[0m [0mwrapped[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mX[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 140[0;31m         [0mdata_to_wrap[0m [0;34m=[0m [0mf[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mX[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    141[0m         [0;32mif[0m [0misinstance[0m[0;34m([0m[0mdata_to_wrap[0m[0;34m,[0m [0mtuple[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    142[0m             [0;31m# only wrap the first output for cross decomposition[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/base.py[0m in [0;36mfit_transform[0;34m(self, X, y, **fit_params)[0m
[1;32m    879[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    880[0m             [0;31m# fit method of arity 2 (supervised transformation)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 881[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX[0m[0;34m,[0m [0my[0m[0;34m,[0m [0;34m**[0m[0mfit_params[0m[0;34m)[0m[0;34m.[0m[0mtransform[0m[0;34m([0m[0mX[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    882[0m [0;34m[0m[0m
[1;32m    883[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py[0m in [0;36mfit[0;34m(self, X, y)[0m
[1;32m   1167[0m         [0;31m# Reset internal state before fitting[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1168[0m         [0mself[0m[0;34m.[0m[0m_reset[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1169[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mpartial_fit[0m[0;34m([0m[0mX[0m[0;34m,[0m [0my[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1170[0m [0;34m[0m[0m
[1;32m   1171[0m     [0;32mdef[0m [0mpartial_fit[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mX[0m[0;34m,[0m [0my[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py[0m in [0;36mpartial_fit[0;34m(self, X, y)[0m
[1;32m   1202[0m [0;34m[0m[0m
[1;32m   1203[0m         [0;32mif[0m [0msparse[0m[0;34m.[0m[0missparse[0m[0;34m([0m[0mX[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1204[0;31m             [0mmins[0m[0;34m,[0m [0mmaxs[0m [0;34m=[0m [0mmin_max_axis[0m[0;34m([0m[0mX[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;36m0[0m[0;34m,[0m [0mignore_nan[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1205[0m             [0mmax_abs[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mmaximum[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mabs[0m[0;34m([0m[0mmins[0m[0;34m)[0m[0;34m,[0m [0mnp[0m[0;34m.[0m[0mabs[0m[0;34m([0m[0mmaxs[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1206[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py[0m in [0;36mmin_max_axis[0;34m(X, axis, ignore_nan)[0m
[1;32m    504[0m     [0;32mif[0m [0misinstance[0m[0;34m([0m[0mX[0m[0;34m,[0m [0;34m([0m[0msp[0m[0;34m.[0m[0mcsr_matrix[0m[0;34m,[0m [0msp[0m[0;34m.[0m[0mcsc_matrix[0m[0;34m)[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    505[0m         [0;32mif[0m [0mignore_nan[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 506[0;31m             [0;32mreturn[0m [0m_sparse_nan_min_max[0m[0;34m([0m[0mX[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    507[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    508[0m             [0;32mreturn[0m [0m_sparse_min_max[0m[0;34m([0m[0mX[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py[0m in [0;36m_sparse_nan_min_max[0;34m(X, axis)[0m
[1;32m    472[0m [0;34m[0m[0m
[1;32m    473[0m [0;32mdef[0m [0m_sparse_nan_min_max[0m[0;34m([0m[0mX[0m[0;34m,[0m [0maxis[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 474[0;31m     [0;32mreturn[0m [0;34m([0m[0m_sparse_min_or_max[0m[0;34m([0m[0mX[0m[0;34m,[0m [0maxis[0m[0;34m,[0m [0mnp[0m[0;34m.[0m[0mfmin[0m[0;34m)[0m[0;34m,[0m [0m_sparse_min_or_max[0m[0;34m([0m[0mX[0m[0;34m,[0m [0maxis[0m[0;34m,[0m [0mnp[0m[0;34m.[0m[0mfmax[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    475[0m [0;34m[0m[0m
[1;32m    476[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py[0m in [0;36m_sparse_min_or_max[0;34m(X, axis, min_or_max)[0m
[1;32m    459[0m         [0maxis[0m [0;34m+=[0m [0;36m2[0m[0;34m[0m[0;34m[0m[0m
[1;32m    460[0m     [0;32mif[0m [0;34m([0m[0maxis[0m [0;34m==[0m [0;36m0[0m[0;34m)[0m [0;32mor[0m [0;34m([0m[0maxis[0m [0;34m==[0m [0;36m1[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 461[0;31m         [0;32mreturn[0m [0m_min_or_max_axis[0m[0;34m([0m[0mX[0m[0;34m,[0m [0maxis[0m[0;34m,[0m [0mmin_or_max[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    462[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    463[0m         [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"invalid axis, use 0 for rows, or 1 for columns"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/sparsefuncs.py[0m in [0;36m_min_or_max_axis[0;34m(X, axis, min_or_max)[0m
[1;32m    442[0m             [0;34m([0m[0mvalue[0m[0;34m,[0m [0;34m([0m[0mmajor_index[0m[0;34m,[0m [0mnp[0m[0;34m.[0m[0mzeros[0m[0;34m([0m[0mlen[0m[0;34m([0m[0mvalue[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mX[0m[0;34m.[0m[0mdtype[0m[0;34m,[0m [0mshape[0m[0;34m=[0m[0;34m([0m[0mM[0m[0;34m,[0m [0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    443[0m         )
[0;32m--> 444[0;31m     [0;32mreturn[0m [0mres[0m[0;34m.[0m[0mA[0m[0;34m.[0m[0mravel[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    445[0m [0;34m[0m[0m
[1;32m    446[0m [0;34m[0m[0m

[0;31mAttributeError[0m: 'coo_matrix' object has no attribute 'A'

## === cell 25
print(confusion_matrix(y_test.values.ravel(), y_pred.ravel()))
