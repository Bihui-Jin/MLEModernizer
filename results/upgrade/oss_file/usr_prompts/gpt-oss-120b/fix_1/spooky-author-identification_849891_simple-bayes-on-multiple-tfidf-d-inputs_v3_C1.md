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
scipy==1.15.3
sklearn-pandas==2.2.0

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

0.3395638810196698

# 6. Current score

0.38005

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

traindata = pd.read_csv("../input/train.csv")
testdata = pd.read_csv("../input/test.csv")

corpus = np.hstack((traindata.loc[:, "text"].values,testdata.loc[:, "text"].values)).astype(str)

## === cell 1
y = traindata.loc[:, "author"].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

## === cell 2
from sklearn.feature_extraction.text import TfidfVectorizer

tv_words = TfidfVectorizer(analyzer = "word", binary = False, ngram_range = (1, 2),
                           stop_words = None, min_df = 2, max_df = 0.8, lowercase = False)
tv_words.fit(corpus)
tfedtext_wordlevel = tv_words.transform(traindata["text"].values)

tfv_charlevel = TfidfVectorizer(analyzer = "char", binary = False, ngram_range = (1, 1),
                                stop_words = None, min_df = 2, max_df = 1.0, lowercase = False)
tfv_charlevel.fit(corpus)
tfedtext_charlevel = tfv_charlevel.transform(traindata["text"].values)

tfv_charlevel_2 = TfidfVectorizer(analyzer = "char", binary = False, ngram_range = (5, 6),
                                  stop_words = None, min_df = 2, max_df = 0.8, lowercase = False)
tfv_charlevel_2.fit(corpus)
tfedtext_charlevel_2 = tfv_charlevel_2.transform(traindata["text"].values)

## === cell 3
import scipy.sparse as sparse #time for hacker stacker
X = sparse.csr_matrix(sparse.hstack((tfedtext_wordlevel, tfedtext_charlevel, tfedtext_charlevel_2)))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB(alpha = 0.12)
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))

from sklearn.metrics import log_loss, accuracy_score
y_probs = mnb.predict_proba(X_test)
print("log loss: ", log_loss(y_test, y_probs))

## === cell 5
from sklearn.model_selection import KFold

def maxindex(anarray):
    return list(map(lambda x: np.where(max(x) == x)[0][0], anarray))

def kfold_predsAndReport(X, y, num_splits, mymodel):
    kf = KFold(n_splits = num_splits, shuffle = True, random_state = 0)
    pred_all = np.zeros([y.shape[0], 3])
    for dev_index, val_index in kf.split(X):
        dev_X, val_X = X[dev_index], X[val_index]
        dev_y, val_y = y[dev_index], y[val_index]
        model = mymodel
        model.fit(dev_X, dev_y)
        pred_val_y = model.predict_proba(val_X)
        pred_all[val_index] = pred_val_y
    y_pred_labels = maxindex(pred_all)
    cm = confusion_matrix(y, y_pred_labels)
    print(cm)
    print(classification_report(y, y_pred_labels))
    print("accuracy:", accuracy_score(y, y_pred_labels))
    print("log loss:", log_loss(y, pred_all))
    return pred_all, model

## === cell 6
y_probs, model = kfold_predsAndReport(X, y, 5, MultinomialNB(alpha = 0.13))

## === cell 7
def trivectorize_words(mytestdata):
    tv_words = TfidfVectorizer(analyzer = "word", binary = False, ngram_range = (1, 2),
                               stop_words = None, min_df = 2, max_df = 0.8, lowercase = False)
    tv_words.fit(corpus)
    tfedtext_wordlevel = tv_words.transform(mytestdata["text"].values)
    
    tfv_charlevel = TfidfVectorizer(analyzer = "char", binary = False, ngram_range = (1, 1),
                                    stop_words = None, min_df = 2, max_df = 1.0, lowercase = False)
    tfv_charlevel.fit(corpus)
    tfedtext_charlevel = tfv_charlevel.transform(mytestdata["text"].values)
    
    tfv_charlevel_2 = TfidfVectorizer(analyzer = "char", binary = False, ngram_range = (5, 6),
                                      stop_words = None, min_df = 2, max_df = 0.8, lowercase = False)
    tfv_charlevel_2.fit(corpus)
    tfedtext_charlevel_2 = tfv_charlevel_2.transform(mytestdata["text"].values)
    
    X = sparse.csr_matrix(sparse.hstack((tfedtext_wordlevel, tfedtext_charlevel, tfedtext_charlevel_2)))
    return X

testsparse = trivectorize_words(testdata)
testsparse

## === cell 8
y_probs, model = kfold_predsAndReport(X, y, 5, MultinomialNB(alpha = 0.13))

## === cell 9
ypreds_test = model.predict_proba(testsparse)

## === cell 10
sub = pd.DataFrame(ypreds_test, columns = list(le.classes_))
sub.insert(0, 'id', testdata.id)
sub.to_csv("tfidf_words_chars_hstacked.csv", index=False)
sub[0:4]

## === cell 12
def stack_charlevelonly(mytestdata):
    
    tfv_charlevel = TfidfVectorizer(analyzer = "char", binary = False, ngram_range = (1, 1),
                                    stop_words = None, min_df = 2, max_df = 1.0, lowercase = False)
    tfv_charlevel.fit(corpus)
    tfedtext_charlevel = tfv_charlevel.transform(mytestdata["text"].values)
    
    tfv_charlevel_2 = TfidfVectorizer(analyzer = "char", binary = False, ngram_range = (5, 6),
                                      stop_words = None, min_df = 2, max_df = 0.8, lowercase = False)
    tfv_charlevel_2.fit(corpus)
    tfedtext_charlevel_2 = tfv_charlevel_2.transform(mytestdata["text"].values)
    
    X = sparse.csr_matrix(sparse.hstack((tfedtext_charlevel, tfedtext_charlevel_2)))
    return X

X_chars = stack_charlevelonly(traindata)
testX_chars = stack_charlevelonly(testdata)

## === cell 13
y_probs_chars, model_chars = kfold_predsAndReport(X_chars, y, 5, MultinomialNB(alpha = 0.04))

## === cell 14
ypreds_test = model_chars.predict_proba(testX_chars)

## === cell 15
sub = pd.DataFrame(ypreds_test, columns = list(le.classes_))
sub.insert(0, 'id', testdata.id)
sub.to_csv("tfidf_chars_unigrams_bigrams.csv", index=False)
sub[0:4]
