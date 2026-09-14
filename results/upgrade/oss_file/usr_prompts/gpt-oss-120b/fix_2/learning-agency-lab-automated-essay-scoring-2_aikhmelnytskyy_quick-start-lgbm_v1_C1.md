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
imbalanced-learn==0.13.0
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
ydata-profiling==4.17.0

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

0.54845

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import nltk
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import re
import random
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
)
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    cohen_kappa_score,
)

nltk.download("wordnet", quiet=True)




## === cell 1
train = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)
print(train.head())




## === cell 2
from ydata_profiling import ProfileReport

profile = ProfileReport(train, title="Pandas Profiling Report")
profile




## === cell 3
def remove_punctuations(data):
    punct_tag = re.compile(r"[^\w\s]")
    return punct_tag.sub(r"", data)


def remove_html(data):
    html_tag = re.compile(r"<.*?>")
    return html_tag.sub(r"", data)


def remove_url(data):
    url_clean = re.compile(r"https://\S+|www\.\S+")
    return url_clean.sub(r"", data)


def remove_emoji(data):
    emoji_clean = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\u2702-\u27B0"
        "\u24C2-\u1F251"
        "]+",
        flags=re.UNICODE,
    )
    data = emoji_clean.sub(r"", data)
    return data.lower()


DESCRIPTION = train["full_text"]
DESCRIPTION = DESCRIPTION.apply(remove_punctuations)
DESCRIPTION = DESCRIPTION.apply(remove_html)
DESCRIPTION = DESCRIPTION.apply(remove_url)
DESCRIPTION = DESCRIPTION.apply(remove_emoji)
train["text"] = DESCRIPTION
train




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_11/978641289.py in <cell line: 0>()
     36 DESCRIPTION = DESCRIPTION.apply(remove_html)
     37 DESCRIPTION = DESCRIPTION.apply(remove_url)
---> 38 DESCRIPTION = DESCRIPTION.apply(remove_emoji)
     39 train["text"] = DESCRIPTION
     40 train

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in apply(self, func, convert_dtype, args, by_row, **kwargs)
   4922             args=args,
   4923             kwargs=kwargs,
-> 4924         ).apply()
   4925 
   4926     def _reindex_indexer(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
   1425 
   1426         # self.func is Callable
-> 1427         return self.apply_standard()
   1428 
   1429     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1505         #  Categorical (GH51645).
   1506         action = "ignore" if isinstance(obj.dtype, CategoricalDtype) else None
-> 1507         mapped = obj._map_values(
   1508             mapper=curried, na_action=action, convert=self.convert_dtype
   1509         )

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _map_values(self, mapper, na_action, convert)
    919             return arr.map(mapper, na_action=na_action)
    920 
--> 921         return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
    922 
    923     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/algorithms.py in map_array(arr, mapper, na_action, convert)
   1741     values = arr.astype(object, copy=False)
   1742     if na_action is None:
-> 1743         return lib.map_infer(values, mapper, convert=convert)
   1744     else:
   1745         return lib.map_infer_mask(

lib.pyx in pandas._libs.lib.map_infer()

/tmp/ipykernel_11/978641289.py in remove_emoji(data)
     16 
     17 def remove_emoji(data):
---> 18     emoji_clean = re.compile(
     19         "["
     20         "\U0001F600-\U0001F64F"

/usr/lib/python3.11/re/__init__.py in compile(pattern, flags)
    225 def compile(pattern, flags=0):
    226     "Compile a regular expression pattern, returning a Pattern object."
--> 227     return _compile(pattern, flags)
    228 
    229 def purge():

/usr/lib/python3.11/re/__init__.py in _compile(pattern, flags)
    292                   "Don't use it.",
    293                   DeprecationWarning)
--> 294     p = _compiler.compile(pattern, flags)
    295     if not (flags & DEBUG):
    296         if len(_cache) >= _MAXCACHE:

/usr/lib/python3.11/re/_compiler.py in compile(p, flags)
    743     if isstring(p):
    744         pattern = p
--> 745         p = _parser.parse(p, flags)
    746     else:
    747         pattern = None

/usr/lib/python3.11/re/_parser.py in parse(str, flags, state)
    987     state.str = str
    988 
--> 989     p = _parse_sub(source, state, flags & SRE_FLAG_VERBOSE, 0)
    990     p.state.flags = fix_flags(str, p.state.flags)
    991 

/usr/lib/python3.11/re/_parser.py in _parse_sub(source, state, verbose, nested)
    462     start = source.tell()
    463     while True:
--> 464         itemsappend(_parse(source, state, verbose, nested + 1,
    465                            not nested and not items))
    466         if not sourcematch("|"):

/usr/lib/python3.11/re/_parser.py in _parse(source, state, verbose, nested, first)
    619                     if hi < lo:
    620                         msg = "bad character range %s-%s" % (this, that)
--> 621                         raise source.error(msg, len(this) + 1 + len(that))
    622                     setappend((RANGE, (lo, hi)))
    623                 else:

error: bad character range Ⓜ-ἥ at position 16

## === cell 4
X = train["text"].astype(str)
y = train["score"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.33, random_state=0
)




## --- ERROR in cell 4, traceback:
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

KeyError: 'text'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1513139518.py in <cell line: 0>()
----> 1 X = train["text"].astype(str)
      2 y = train["score"].astype(int)
      3 
      4 X_train, X_test, y_train, y_test = train_test_split(
      5     X, y, stratify=y, test_size=0.33, random_state=0

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

KeyError: 'text'

## === cell 5
model = Pipeline(
    [
        (
            "Vectorizer",
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                lowercase=True,
                max_features=150000,
            ),
        ),
        ("feature_selection", SelectFromModel(ExtraTreesClassifier(random_state=0))),
        ("clf", LinearSVC(class_weight="balanced")),
    ]
)

model.fit(X_train, y_train)

pred_train = model.predict(X_train)
f1_train = f1_score(y_train, pred_train, average="weighted")
kappa_train = cohen_kappa_score(y_train, pred_train)

print(f"F1 score (train): {f1_train:.4f}")
print(f"Cohen Kappa (train): {kappa_train:.4f}")




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4010788295.py in <cell line: 0>()
     16 )
     17 
---> 18 model.fit(X_train, y_train)
     19 
     20 pred_train = model.predict(X_train)

NameError: name 'X_train' is not defined

## === cell 6
pred_val = model.predict(X_test)

cm = confusion_matrix(y_test, pred_val, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()

f1_val = f1_score(y_test, pred_val, average="weighted")
kappa_val = cohen_kappa_score(y_test, pred_val)

print(f"F1 score (validation): {f1_val:.4f}")
print(f"Cohen Kappa (validation): {kappa_val:.4f}")




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4065393314.py in <cell line: 0>()
      1 # Evaluate on held‑out validation set
----> 2 pred_val = model.predict(X_test)
      3 
      4 cm = confusion_matrix(y_test, pred_val, labels=model.classes_)
      5 disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

NameError: name 'X_test' is not defined

## === cell 7
test = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
DESCRIPTION = test["full_text"]
DESCRIPTION = DESCRIPTION.apply(remove_punctuations)
DESCRIPTION = DESCRIPTION.apply(remove_html)
DESCRIPTION = DESCRIPTION.apply(remove_url)
DESCRIPTION = DESCRIPTION.apply(remove_emoji)
test["text"] = DESCRIPTION
test




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_11/198643292.py in <cell line: 0>()
      6 DESCRIPTION = DESCRIPTION.apply(remove_html)
      7 DESCRIPTION = DESCRIPTION.apply(remove_url)
----> 8 DESCRIPTION = DESCRIPTION.apply(remove_emoji)
      9 test["text"] = DESCRIPTION
     10 test

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in apply(self, func, convert_dtype, args, by_row, **kwargs)
   4922             args=args,
   4923             kwargs=kwargs,
-> 4924         ).apply()
   4925 
   4926     def _reindex_indexer(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
   1425 
   1426         # self.func is Callable
-> 1427         return self.apply_standard()
   1428 
   1429     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1505         #  Categorical (GH51645).
   1506         action = "ignore" if isinstance(obj.dtype, CategoricalDtype) else None
-> 1507         mapped = obj._map_values(
   1508             mapper=curried, na_action=action, convert=self.convert_dtype
   1509         )

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _map_values(self, mapper, na_action, convert)
    919             return arr.map(mapper, na_action=na_action)
    920 
--> 921         return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
    922 
    923     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/algorithms.py in map_array(arr, mapper, na_action, convert)
   1741     values = arr.astype(object, copy=False)
   1742     if na_action is None:
-> 1743         return lib.map_infer(values, mapper, convert=convert)
   1744     else:
   1745         return lib.map_infer_mask(

lib.pyx in pandas._libs.lib.map_infer()

/tmp/ipykernel_11/978641289.py in remove_emoji(data)
     16 
     17 def remove_emoji(data):
---> 18     emoji_clean = re.compile(
     19         "["
     20         "\U0001F600-\U0001F64F"

/usr/lib/python3.11/re/__init__.py in compile(pattern, flags)
    225 def compile(pattern, flags=0):
    226     "Compile a regular expression pattern, returning a Pattern object."
--> 227     return _compile(pattern, flags)
    228 
    229 def purge():

/usr/lib/python3.11/re/__init__.py in _compile(pattern, flags)
    292                   "Don't use it.",
    293                   DeprecationWarning)
--> 294     p = _compiler.compile(pattern, flags)
    295     if not (flags & DEBUG):
    296         if len(_cache) >= _MAXCACHE:

/usr/lib/python3.11/re/_compiler.py in compile(p, flags)
    743     if isstring(p):
    744         pattern = p
--> 745         p = _parser.parse(p, flags)
    746     else:
    747         pattern = None

/usr/lib/python3.11/re/_parser.py in parse(str, flags, state)
    987     state.str = str
    988 
--> 989     p = _parse_sub(source, state, flags & SRE_FLAG_VERBOSE, 0)
    990     p.state.flags = fix_flags(str, p.state.flags)
    991 

/usr/lib/python3.11/re/_parser.py in _parse_sub(source, state, verbose, nested)
    462     start = source.tell()
    463     while True:
--> 464         itemsappend(_parse(source, state, verbose, nested + 1,
    465                            not nested and not items))
    466         if not sourcematch("|"):

/usr/lib/python3.11/re/_parser.py in _parse(source, state, verbose, nested, first)
    619                     if hi < lo:
    620                         msg = "bad character range %s-%s" % (this, that)
--> 621                         raise source.error(msg, len(this) + 1 + len(that))
    622                     setappend((RANGE, (lo, hi)))
    623                 else:

error: bad character range Ⓜ-ἥ at position 16

## === cell 8
test_predictions = model.predict(test["text"])

submission = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)
submission["score"] = test_predictions
submission.to_csv("submission.csv", index=False)

display(submission.head())

## --- ERROR in cell 8, traceback:
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

KeyError: 'text'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/941687597.py in <cell line: 0>()
----> 1 test_predictions = model.predict(test["text"])
      2 
      3 submission = pd.read_csv(
      4     "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
      5 )

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

KeyError: 'text'
