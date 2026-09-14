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
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# 2. Python version

3.10

# 3. Installed packages

catboost==1.2.8
cuml-cu12==25.2.1
geopandas==0.14.4
libcuml-cu12==25.2.1
lightgbm==4.6.0
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
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Target score

0.2114

# 6. Current score

nan

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.23804) has done: 'I slightly weaken the model to bring the Pearson score closer to the target by (1) limiting the TF‑IDF to unigrams only and (2) increasing regularisation in the LogisticRegression (C=0.1). These are minimal, safe tweaks that keep the overall pipeline intact while reducing performance toward the desired range.'
- What this solution (achieved 0.06684) has done: 'I slightly increase regularisation and limit the TF‑IDF vocabulary to reduce model capacity a bit more, which should lower the Pearson score toward the target (from 0.238 → ≈0.21). The changes keep the same pipeline, model type, and output format.'
- What this solution (achieved nan) has done: 'I raise the model’s capacity slightly to improve the Pearson correlation toward the target.  
- In the TF‑IDF step I increase the n‑gram range to include bigrams and double the feature limit, and I also add the `context` column so the model can use that information.  
- I set the logistic regression regularisation parameter `C` to 0.05, which is between the overly‑regularised (C=0.01) and the previously higher‑scoring (C=0.1) settings, aiming for a score nearer the target without over‑fitting.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from cuml.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from sklearn.compose import make_column_transformer




## === cell 1
train = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/train.csv")
test = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/test.csv")
le = LabelEncoder()




## === cell 2
y = train.score
X = train.drop(["id", "context", "score"], axis=1)

y = le.fit_transform(y)
y = y.astype("float32")




## === cell 3
vectorizer = TfidfVectorizer(
    tokenizer=word_tokenize,
    ngram_range=(1, 2),
    max_features=10000,
)
transformer = make_column_transformer(
    (vectorizer, "anchor"),
    (vectorizer, "target"),
    (vectorizer, "context"),
)

X = transformer.fit_transform(X)




## --- ERROR in cell 3, traceback:
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

KeyError: 'context'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/sklearn/utils/__init__.py in _get_column_indices(X, key)
    447             for col in columns:
--> 448                 col_idx = all_columns.get_loc(col)
    449                 if not isinstance(col_idx, numbers.Integral):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:

KeyError: 'context'

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2899690226.py in <cell line: 0>()
     11 )
     12 
---> 13 X = transformer.fit_transform(X)
     14 
     15 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/compose/_column_transformer.py in fit_transform(self, X, y)
    722         self._check_n_features(X, reset=True)
    723         self._validate_transformers()
--> 724         self._validate_column_callables(X)
    725         self._validate_remainder(X)
    726 

/usr/local/lib/python3.11/dist-packages/sklearn/compose/_column_transformer.py in _validate_column_callables(self, X)
    424                 columns = columns(X)
    425             all_columns.append(columns)
--> 426             transformer_to_input_indices[name] = _get_column_indices(X, columns)
    427 
    428         self._columns = all_columns

/usr/local/lib/python3.11/dist-packages/sklearn/utils/__init__.py in _get_column_indices(X, key)
    454 
    455         except KeyError as e:
--> 456             raise ValueError("A given column is not a column of the dataframe") from e
    457 
    458         return column_indices

ValueError: A given column is not a column of the dataframe

## === cell 4
model = LogisticRegression(C=0.05)
model.fit(X, y)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3485789245.py in <cell line: 0>()
      1 # C set between 0.01 and 0.1 to move correlation toward the target
      2 model = LogisticRegression(C=0.05)
----> 3 model.fit(X, y)
      4 
      5 

/usr/local/lib/python3.11/dist-packages/cuml/internals/api_decorators.py in wrapper(*args, **kwargs)
    191 
    192                     if process_return:
--> 193                         ret = func(*args, **kwargs)
    194                     else:
    195                         return func(*args, **kwargs)

/usr/local/lib/python3.11/dist-packages/cuml/internals/api_decorators.py in dispatch(self, *args, **kwargs)
    414         if hasattr(self, "dispatch_func"):
    415             func_name = gpu_func.__name__
--> 416             return self.dispatch_func(func_name, gpu_func, *args, **kwargs)
    417         else:
    418             return gpu_func(self, *args, **kwargs)

/usr/local/lib/python3.11/dist-packages/cuml/internals/api_decorators.py in wrapper(*args, **kwargs)
    193                         ret = func(*args, **kwargs)
    194                     else:
--> 195                         return func(*args, **kwargs)
    196 
    197                 return cm.process_return(ret)

base.pyx in cuml.internals.base.UniversalBase.dispatch_func()

logistic_regression.pyx in cuml.linear_model.logistic_regression.LogisticRegression.fit()

/usr/local/lib/python3.11/dist-packages/cuml/internals/api_decorators.py in wrapper(*args, **kwargs)
    191 
    192                     if process_return:
--> 193                         ret = func(*args, **kwargs)
    194                     else:
    195                         return func(*args, **kwargs)

qn.pyx in cuml.solvers.qn.QN.fit()

/usr/local/lib/python3.11/dist-packages/cuml/internals/input_utils.py in input_to_cuml_array(X, order, deepcopy, check_dtype, convert_to_dtype, check_mem_type, convert_to_mem_type, safe_dtype_conversion, check_cols, check_rows, fail_on_order, force_contiguous)
    410 
    411     """
--> 412     arr = CumlArray.from_input(
    413         X,
    414         order=order,

/usr/local/lib/python3.11/dist-packages/cuml/internals/memory_utils.py in cupy_rmm_wrapper(*args, **kwargs)
     85         if GPU_ENABLED:
     86             with cupy_using_allocator(rmm_cupy_allocator):
---> 87                 return func(*args, **kwargs)
     88         return func(*args, **kwargs)
     89 

/usr/local/lib/python3.11/dist-packages/cuml/internals/array.py in from_input(cls, X, order, deepcopy, check_dtype, convert_to_dtype, check_mem_type, convert_to_mem_type, safe_dtype_conversion, check_cols, check_rows, fail_on_order, force_contiguous)
   1160                     X = cp.asarray(X)
   1161                 if (
-> 1162                     (X < target_dtype_range.min) | (X > target_dtype_range.max)
   1163                 ).any():
   1164                     raise TypeError(

TypeError: '<' not supported between instances of 'str' and 'float'

## === cell 5
test.head()




## === cell 6
t = test[["anchor", "target", "context"]]
t = transformer.transform(t)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3245019317.py in <cell line: 0>()
      1 t = test[["anchor", "target", "context"]]
----> 2 t = transformer.transform(t)
      3 
      4 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/compose/_column_transformer.py in transform(self, X)
    776 
    777         if fit_dataframe_and_transform_dataframe:
--> 778             named_transformers = self.named_transformers_
    779             # check that all names seen in fit are in transform, unless
    780             # they were dropped

/usr/local/lib/python3.11/dist-packages/sklearn/compose/_column_transformer.py in named_transformers_(self)
    459         """
    460         # Use Bunch object to improve autocomplete
--> 461         return Bunch(**{name: trans for name, trans, _ in self.transformers_})
    462 
    463     def _get_feature_name_out_for_transformer(

AttributeError: 'ColumnTransformer' object has no attribute 'transformers_'

## === cell 7
pred = model.predict(t)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/2432253908.py in <cell line: 0>()
----> 1 pred = model.predict(t)
      2 
      3 

/usr/local/lib/python3.11/dist-packages/cuml/internals/api_decorators.py in wrapper(*args, **kwargs)
    191 
    192                     if process_return:
--> 193                         ret = func(*args, **kwargs)
    194                     else:
    195                         return func(*args, **kwargs)

/usr/local/lib/python3.11/dist-packages/cuml/internals/api_decorators.py in dispatch(self, *args, **kwargs)
    414         if hasattr(self, "dispatch_func"):
    415             func_name = gpu_func.__name__
--> 416             return self.dispatch_func(func_name, gpu_func, *args, **kwargs)
    417         else:
    418             return gpu_func(self, *args, **kwargs)

/usr/local/lib/python3.11/dist-packages/cuml/internals/api_decorators.py in wrapper(*args, **kwargs)
    193                         ret = func(*args, **kwargs)
    194                     else:
--> 195                         return func(*args, **kwargs)
    196 
    197                 return cm.process_return(ret)

base.pyx in cuml.internals.base.UniversalBase.dispatch_func()

logistic_regression.pyx in cuml.linear_model.logistic_regression.LogisticRegression.predict()

/usr/local/lib/python3.11/dist-packages/cuml/internals/api_decorators.py in wrapper(*args, **kwargs)
    191 
    192                     if process_return:
--> 193                         ret = func(*args, **kwargs)
    194                     else:
    195                         return func(*args, **kwargs)

qn.pyx in cuml.solvers.qn.QN.predict()

AttributeError: 'NoneType' object has no attribute 'dtype'

## === cell 8
pred = pred.astype("int")
pr = le.inverse_transform(pred)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/477647580.py in <cell line: 0>()
----> 1 pred = pred.astype("int")
      2 pr = le.inverse_transform(pred)
      3 
      4 

NameError: name 'pred' is not defined

## === cell 9
sample = pd.read_csv(
    "../input/us-patent-phrase-to-phrase-matching/sample_submission.csv"
)




## === cell 10
sample.score = pr




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3542124091.py in <cell line: 0>()
----> 1 sample.score = pr
      2 
      3 

NameError: name 'pr' is not defined

## === cell 11
sample.to_csv("submission.csv", index=False)
