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

3.7

# 3. Installed packages

gensim==4.4.0
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
tf_keras==2.18.0

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

0.5325

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
import re
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss



## === cell 1
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print(train_df.shape, test_df.shape)




## === cell 2
def clean_text(text):
    """lower‑case and remove punctuation/underscores"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"_", "", text)
    return text




## === cell 3
train_df["text"] = (
    train_df["text"].map(lambda x: clean_text(x)).map(lambda x: x.strip().split())
)
train_df.head()



## === cell 4
test_df["text"] = (
    test_df["text"].map(lambda x: clean_text(x)).map(lambda x: x.strip().split())
)
test_df.head()



## === cell 5
data = list(train_df["text"]) + list(test_df["text"])
print(f"Total sentences for embedding: {len(data)}")



## === cell 6
embedding = Word2Vec(
    sentences=data, vector_size=50, window=10, min_count=1, sg=0  # CBOW
)

print(f"Embedding vocab size: {len(embedding.wv)}")



## === cell 7
train_df["author"] = pd.Categorical(train_df["author"])
df_dummies = pd.get_dummies(train_df["author"], prefix="author")
train_df = pd.concat([train_df, df_dummies], axis=1)



## === cell 8
X = train_df["text"].values
Y = train_df[["author_EAP", "author_HPL", "author_MWS"]].values
print(f"X shape: {X.shape}, Y shape: {Y.shape}")



## === cell 9
X_test = test_df["text"].values
print(f"X_test shape: {X_test.shape}")




## === cell 10
def text_to_avg(words):
    """Average Word2Vec vectors for a list of tokens."""
    avg = np.zeros(embedding.vector_size)
    for w in words:
        avg += embedding.wv[w]
    if len(words) > 0:
        avg /= len(words)
    return avg




## === cell 11
X_avg = np.vstack([text_to_avg(txt) for txt in X])
print(f"X_avg shape: {X_avg.shape}")



## === cell 12
X_test_avg = np.vstack([text_to_avg(txt) for txt in X_test])
print(f"X_test_avg shape: {X_test_avg.shape}")



## === cell 13
X_train, X_dev, Y_train, Y_dev = train_test_split(
    X_avg, Y, test_size=0.2, random_state=123, stratify=Y
)
print(f"Train/Dev shapes: {X_train.shape}/{X_dev.shape}")



## === cell 14
clf = LogisticRegression(
    multi_class="multinomial", solver="lbfgs", C=4.0, max_iter=300, n_jobs=-1
)
clf.fit(X_train, Y_train)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2557618075.py in <cell line: 0>()
      3     multi_class="multinomial", solver="lbfgs", C=4.0, max_iter=300, n_jobs=-1
      4 )
----> 5 clf.fit(X_train, Y_train)
      6 

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py in fit(self, X, y, sample_weight)
   1194             _dtype = [np.float64, np.float32]
   1195 
-> 1196         X, y = self._validate_data(
   1197             X,
   1198             y,

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    582                 y = check_array(y, input_name="y", **check_y_params)
    583             else:
--> 584                 X, y = check_X_y(X, y, **check_params)
    585             out = X, y
    586 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1120     )
   1121 
-> 1122     y = _check_y(y, multi_output=multi_output, y_numeric=y_numeric, estimator=estimator)
   1123 
   1124     check_consistent_length(X, y)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _check_y(y, multi_output, y_numeric, estimator)
   1141     else:
   1142         estimator_name = _check_estimator_name(estimator)
-> 1143         y = column_or_1d(y, warn=True)
   1144         _assert_all_finite(y, input_name="y", estimator_name=estimator_name)
   1145         _ensure_no_complex_data(y)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in column_or_1d(y, dtype, warn)
   1200         return _asarray_with_order(xp.reshape(y, -1), order="C", xp=xp)
   1201 
-> 1202     raise ValueError(
   1203         "y should be a 1d array, got an array of shape {} instead.".format(shape)
   1204     )

ValueError: y should be a 1d array, got an array of shape (14096, 3) instead.

## === cell 15
dev_pred = clf.predict_proba(X_dev)
dev_logloss = log_loss(Y_dev, dev_pred)
print(f"Dev LogLoss: {dev_logloss:.5f}")



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_55/1224985754.py in <cell line: 0>()
      1 # Optional: evaluate on dev set (log loss) to see improvement direction
----> 2 dev_pred = clf.predict_proba(X_dev)
      3 dev_logloss = log_loss(Y_dev, dev_pred)
      4 print(f"Dev LogLoss: {dev_logloss:.5f}")
      5 

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py in predict_proba(self, X)
   1360             where classes are ordered as they are in ``self.classes_``.
   1361         """
-> 1362         check_is_fitted(self)
   1363 
   1364         ovr = self.multi_class in ["ovr", "warn"] or (

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This LogisticRegression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

## === cell 16
test_preds = clf.predict_proba(X_test_avg)
print(f"Test predictions shape: {test_preds.shape}")



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_55/2199603352.py in <cell line: 0>()
      1 # Predict probabilities for the test set
----> 2 test_preds = clf.predict_proba(X_test_avg)
      3 print(f"Test predictions shape: {test_preds.shape}")
      4 

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py in predict_proba(self, X)
   1360             where classes are ordered as they are in ``self.classes_``.
   1361         """
-> 1362         check_is_fitted(self)
   1363 
   1364         ovr = self.multi_class in ["ovr", "warn"] or (

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This LogisticRegression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

## === cell 17
result = pd.DataFrame(test_preds, columns=["EAP", "HPL", "MWS"])
result.insert(0, "id", test_df["id"])
result.head()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1149809142.py in <cell line: 0>()
      1 # Build submission dataframe matching required format
----> 2 result = pd.DataFrame(test_preds, columns=["EAP", "HPL", "MWS"])
      3 result.insert(0, "id", test_df["id"])
      4 result.head()
      5 

NameError: name 'test_preds' is not defined

## === cell 18
result.to_csv("submission.csv", index=False, float_format="%.20f")
print("Submission file written to submission.csv")

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1850285080.py in <cell line: 0>()
      1 # Save submission CSV
----> 2 result.to_csv("submission.csv", index=False, float_format="%.20f")
      3 print("Submission file written to submission.csv")

NameError: name 'result' is not defined
