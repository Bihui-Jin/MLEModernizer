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

0.75978

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
import matplotlib.pyplot as plt
from spellchecker import SpellChecker


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1234348325.py in <cell line: 0>()
      2 import numpy as np
      3 import matplotlib.pyplot as plt
----> 4 from spellchecker import SpellChecker

ModuleNotFoundError: No module named 'spellchecker'

## === cell 1
train = pd.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv', index_col='essay_id')
test = pd.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv', index_col='essay_id')


## === cell 2
train


## === cell 3
test


## === cell 4
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(train['score'])


## === cell 5
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
train['tokens'] = train['full_text'].apply(word_tokenize)
test['tokens'] = test['full_text'].apply(word_tokenize)


## === cell 6
train


## === cell 7
test


## === cell 8
spell_checker = SpellChecker()
def count_errors(tokens):
    misspelled = spell_checker.unknown(token for token in tokens if token.isalpha())
    return len(misspelled)
train['count_spelling_errors'] = train['tokens'].apply(lambda x: count_errors(x))
test['count_spelling_errors'] = test['tokens'].apply(lambda x: count_errors(x))


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1565461449.py in <cell line: 0>()
----> 1 spell_checker = SpellChecker()
      2 def count_errors(tokens):
      3     misspelled = spell_checker.unknown(token for token in tokens if token.isalpha())
      4     return len(misspelled)
      5 train['count_spelling_errors'] = train['tokens'].apply(lambda x: count_errors(x))

NameError: name 'SpellChecker' is not defined

## === cell 9
train


## === cell 10
test


## === cell 11
class FeatureExtract:
    def __init__(self):
        pass
    
    def word_count(self, text):
        return len(text.split())
    
    def sen_count(self, text):
        return len(text.split('.'))
    
    def ave_word_length(self, text):
        words = text.split()
        total_length = 0
        for word in words:
            total_length += len(word)
        if len(words) == 0:
            return 0
        else:
            return total_length / len(words)
    
    
    def lexical_diversity(self, text):
        words = text.split()
        if len(words) == 0:
            return 0
        return len(set(words)) / len(words)
    
    def extract_features(self, text):
        features = {
            'word_count': self.word_count(text),
            'sen_count': self.sen_count(text),
            'ave_word_length': self.ave_word_length(text),
            'total_stopwords' : self.total_stopwords(text),
            'count_spelling_errors' : self.count_spelling_errors(text),
            'lexical_diversity': self.lexical_diversity(text),
            'sentiment' : self.sentiment(text),
        }
        return features


## === cell 12
def insert_features(df, text):
    extractor = FeatureExtract()
    for feature in ['word_count', 'sen_count', 'ave_word_length', 'lexical_diversity']:
        df[feature] = df[text].apply(lambda x: getattr(extractor, feature)(x))

insert_features(train, 'full_text')
insert_features(test, 'full_text')


## === cell 13
train


## === cell 14
test


## === cell 15
features = ['count_spelling_errors', 'word_count', 'sen_count', 'ave_word_length', 'lexical_diversity']
X = train[features]
X_test = test[features]


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1461794869.py in <cell line: 0>()
      1 features = ['count_spelling_errors', 'word_count', 'sen_count', 'ave_word_length', 'lexical_diversity']
----> 2 X = train[features]
      3 X_test = test[features]

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['count_spelling_errors'] not in index"

## === cell 16
print("Value of X:")
print(X.head())


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3491193523.py in <cell line: 0>()
      1 print("Value of X:")
----> 2 print(X.head())

NameError: name 'X' is not defined

## === cell 17
print("\nValue of X_test:")
print(X_test.head())


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/738077036.py in <cell line: 0>()
      1 print("\nValue of X_test:")
----> 2 print(X_test.head())

NameError: name 'X_test' is not defined

## === cell 18
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2079466733.py in <cell line: 0>()
      1 from sklearn.model_selection import train_test_split
----> 2 X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

NameError: name 'X' is not defined

## === cell 19
print("Value of X_train:")
print(X_train.head())


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/907959466.py in <cell line: 0>()
      1 print("Value of X_train:")
----> 2 print(X_train.head())

NameError: name 'X_train' is not defined

## === cell 20
print("\nValue of X_valid:")
print(X_valid.head())


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1286356872.py in <cell line: 0>()
      1 print("\nValue of X_valid:")
----> 2 print(X_valid.head())

NameError: name 'X_valid' is not defined

## === cell 21
print("\nValue of y_train:")
print(y_train)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4056841827.py in <cell line: 0>()
      1 print("\nValue of y_train:")
----> 2 print(y_train)

NameError: name 'y_train' is not defined

## === cell 22
print("\nValue of y_valid:")
print(y_valid)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2852977407.py in <cell line: 0>()
      1 print("\nValue of y_valid:")
----> 2 print(y_valid)

NameError: name 'y_valid' is not defined

## === cell 23
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()


## === cell 24
model.fit(X_train, y_train)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/60041271.py in <cell line: 0>()
----> 1 model.fit(X_train, y_train)

NameError: name 'X_train' is not defined

## === cell 25
y_pred = model.predict(X_valid)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1605125439.py in <cell line: 0>()
----> 1 y_pred = model.predict(X_valid)

NameError: name 'X_valid' is not defined

## === cell 26
print("Predicted values on the validation set:")
print(y_pred)


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1184319330.py in <cell line: 0>()
      1 print("Predicted values on the validation set:")
----> 2 print(y_pred)

NameError: name 'y_pred' is not defined

## === cell 27
y_pred = le.inverse_transform(y_pred)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1578876944.py in <cell line: 0>()
----> 1 y_pred = le.inverse_transform(y_pred)

NameError: name 'y_pred' is not defined

## === cell 28
from sklearn.metrics import cohen_kappa_score, accuracy_score
kappa_score = cohen_kappa_score(y_valid, y_pred, weights='quadratic')
print("Cohen's Kappa Score:", kappa_score)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1637377133.py in <cell line: 0>()
      1 from sklearn.metrics import cohen_kappa_score, accuracy_score
----> 2 kappa_score = cohen_kappa_score(y_valid, y_pred, weights='quadratic')
      3 print("Cohen's Kappa Score:", kappa_score)

NameError: name 'y_valid' is not defined

## === cell 29
accuracy = accuracy_score(y_valid, y_pred)
print("Accuracy:", accuracy)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3607901725.py in <cell line: 0>()
----> 1 accuracy = accuracy_score(y_valid, y_pred)
      2 print("Accuracy:", accuracy)

NameError: name 'y_valid' is not defined

## === cell 30
plt.figure(figsize=(8, 6))
plt.scatter(y_valid, y_pred, color='blue')
plt.plot([min(y_valid), max(y_valid)], [min(y_valid), max(y_valid)], color='red', linestyle='--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.grid(True)
plt.show()


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/666905662.py in <cell line: 0>()
      1 plt.figure(figsize=(8, 6))
----> 2 plt.scatter(y_valid, y_pred, color='blue')
      3 plt.plot([min(y_valid), max(y_valid)], [min(y_valid), max(y_valid)], color='red', linestyle='--')
      4 plt.xlabel('Actual')
      5 plt.ylabel('Predicted')

NameError: name 'y_valid' is not defined

## === cell 31
plt.figure(figsize=(8, 6))
plt.hist(y_valid, bins=10, alpha=0.5, label='Actual', color='blue')
plt.hist(y_pred, bins=10, alpha=0.5, label='Predicted', color='orange')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title('Distribution of Actual and Predicted Scores')
plt.legend()
plt.grid(True)
plt.show()


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3566329051.py in <cell line: 0>()
      1 plt.figure(figsize=(8, 6))
----> 2 plt.hist(y_valid, bins=10, alpha=0.5, label='Actual', color='blue')
      3 plt.hist(y_pred, bins=10, alpha=0.5, label='Predicted', color='orange')
      4 plt.xlabel('Score')
      5 plt.ylabel('Frequency')

NameError: name 'y_valid' is not defined

## === cell 32
predictions = model.predict(X_test)


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4004083374.py in <cell line: 0>()
----> 1 predictions = model.predict(X_test)

NameError: name 'X_test' is not defined

## === cell 33
print("Predicted values on the test set:")
print(predictions)


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1828079910.py in <cell line: 0>()
      1 print("Predicted values on the test set:")
----> 2 print(predictions)

NameError: name 'predictions' is not defined

## === cell 34
predictions = le.inverse_transform(predictions)


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/330419008.py in <cell line: 0>()
----> 1 predictions = le.inverse_transform(predictions)

NameError: name 'predictions' is not defined

## === cell 35
predictions


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4129765282.py in <cell line: 0>()
----> 1 predictions

NameError: name 'predictions' is not defined

## === cell 36
score_counts = pd.Series(predictions).value_counts()
plt.figure(figsize=(8, 6))
plt.pie(score_counts, labels=score_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Predicted Scores')
plt.axis('equal')
plt.show()


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2026071260.py in <cell line: 0>()
----> 1 score_counts = pd.Series(predictions).value_counts()
      2 plt.figure(figsize=(8, 6))
      3 plt.pie(score_counts, labels=score_counts.index, autopct='%1.1f%%', startangle=140)
      4 plt.title('Distribution of Predicted Scores')
      5 plt.axis('equal')

NameError: name 'predictions' is not defined

## === cell 37
submission = pd.DataFrame({'essay_id': test.index, 'score': predictions})
submission.to_csv('submission.csv', index=False)


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1848590796.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({'essay_id': test.index, 'score': predictions})
      2 submission.to_csv('submission.csv', index=False)

NameError: name 'predictions' is not defined
