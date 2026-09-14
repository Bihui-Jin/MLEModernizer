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

0.74973

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')


## === cell 2
word = []

for text in train['text']:
    word.append( text )

for text in test['text']:
    word.append( text )


## === cell 3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count_vec = CountVectorizer( ngram_range = (1, 3) )
tfid_ = TfidfTransformer( )

print('Extracing Count Information')
count_vec.fit(word)
train_sparse = count_vec.transform( train['text'] )
test_sparse = count_vec.transform( test['text'] )

print('Normalizing Count Information')
tfid_.fit( train_sparse )

train_tfid = tfid_.transform( train_sparse )
test_tfid = tfid_.transform( test_sparse )


## === cell 4
author_dict = { 'EAP' : 0, 'HPL' : 1, 'MWS' : 2 }

author_labels = train['author'].apply(author_dict.get)
train = train.drop('author', axis = 1)
train.drop('id', axis = 1, inplace = True)


## === cell 5
test_preds = {}


## === cell 6
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(loss = 'log', max_iter = 2000, n_jobs = -1)

sgd_clf.fit( train_tfid, author_labels )

test_preds['sgd_clf'] = sgd_clf.predict_proba( test_tfid )


## === cell 7
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB( )

nb.fit( train_tfid, author_labels )

test_preds['nb_clf'] = nb.predict_proba( test_tfid )


## === cell 8
from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression( solver = 'saga', multi_class = 'multinomial', 
                             max_iter = 500, n_jobs = -1)

log_clf.fit( train_tfid, author_labels )

test_preds['log_clf'] = log_clf.predict_proba( test_tfid )


## === cell 9
cols = ['EAP', 'HPL', 'MWS']
sub[cols] = 0.0

n = len( test_preds.keys() )

for key in test_preds.keys():
    sub[cols] += (1.0/n)*( test_preds.get(key) ** -1.0)
    
sub[cols] = ( sub[cols].values ) ** -1.0


## === cell 10
sub.to_csv('sub.csv', index = False)
