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

cudf-polars-cu12==25.6.0
geopandas==0.14.4
joblib==1.5.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
polars==1.25.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
tqdm==4.67.1
xgboost==2.0.3

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

0.7879701644258821

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
import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler, LabelEncoder, MinMaxScaler, MaxAbsScaler
from sklearn.utils import class_weight


from tqdm.auto import tqdm
import datetime

import c24lal_utils as utils

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1284671568.py in <cell line: 0>()
     13 
     14 #import the util script:
---> 15 import c24lal_utils as utils

ModuleNotFoundError: No module named 'c24lal_utils'

## === cell 2
utils.install_libs_1()
print("Installing additional libs is complete.")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2255401621.py in <cell line: 0>()
      1 #install the libs:
----> 2 utils.install_libs_1()
      3 print("Installing additional libs is complete.")

NameError: name 'utils' is not defined

## === cell 3
import textstat
from spellchecker import SpellChecker
import textdescriptives as td 
import asent # sentiment analysis
from spacy.language import Language
import spacy
from spacy.tokens import Doc


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2459204707.py in <cell line: 0>()
      1 #import the installed objects:
----> 2 import textstat
      3 from spellchecker import SpellChecker
      4 import textdescriptives as td
      5 import asent # sentiment analysis

ModuleNotFoundError: No module named 'textstat'

## === cell 4
def spell_check(doc):
    spell = SpellChecker()
    words = []
    for tokens in doc:
        if tokens.pos_ == 'NOUN':
            words.append(doc[tokens.i].text)
        elif tokens.pos_ == "ADJ":
            words.append(doc[tokens.i].text)
        elif tokens.pos_ == "ADV":
            words.append(doc[tokens.i].text)
        elif tokens.pos_ == "DET":
            words.append(doc[tokens.i].text)
        elif tokens.pos_ == "PRON":
            words.append(doc[tokens.i].text)
        elif tokens.pos_ == "CCONJ":
            words.append(doc[tokens.i].text)

    return len(spell.unknown(words))/len(words) 
    



def get_stat(doc):
    test_data = doc.text
    t = {}
    t['F_FRE'] = textstat.flesch_reading_ease(test_data)
    t['F_FKG'] = textstat.flesch_kincaid_grade(test_data)
    t['F_SI'] = textstat.smog_index(test_data)
    t['F_CLI'] = textstat.coleman_liau_index(test_data)
    t['F_ARI'] = textstat.automated_readability_index(test_data)
    t['F_DCRS'] = textstat.dale_chall_readability_score(test_data)
    t['F_DW'] = textstat.difficult_words(test_data)
    t['F_LWF'] = textstat.linsear_write_formula(test_data)
    t['F_GF'] = textstat.gunning_fog(test_data)
    t['F_FH'] = textstat.fernandez_huerta(test_data)
    t['F_SP'] = textstat.szigriszt_pazos(test_data)
    t['F_GP'] = textstat.gutierrez_polini(test_data)
    t['F_C'] =  textstat.crawford(test_data)
    t['F_GI'] = textstat.gulpease_index(test_data)
    t['F_OSM'] = textstat.osman(test_data)

    return t
    
if not Doc.has_extension("get_stat"):
    Doc.set_extension("get_stat", method=get_stat)

if not Doc.has_extension("spell_check"):
    Doc.set_extension("spell_check", method=spell_check)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4030915453.py in <cell line: 0>()
     47 
     48 # Set extension on the Doc with method
---> 49 if not Doc.has_extension("get_stat"):
     50     Doc.set_extension("get_stat", method=get_stat)
     51 

NameError: name 'Doc' is not defined

## === cell 5
import joblib

config = joblib.load("/kaggle/input/nlp-spacy-model/nlp_config.joblib")
lang_cls = spacy.util.get_lang_class(config["nlp"]["lang"])
nlp = lang_cls.from_config(config)
bytes_data = joblib.load("/kaggle/input/nlp-spacy-model/nlp_bytes.joblib")
nlp.from_bytes(bytes_data)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1695720986.py in <cell line: 0>()
      3 import joblib
      4 
----> 5 config = joblib.load("/kaggle/input/nlp-spacy-model/nlp_config.joblib")
      6 lang_cls = spacy.util.get_lang_class(config["nlp"]["lang"])
      7 nlp = lang_cls.from_config(config)

/usr/local/lib/python3.11/dist-packages/joblib/numpy_pickle.py in load(filename, mmap_mode, ensure_native_byte_order)
    733             obj = _unpickle(fobj, ensure_native_byte_order=ensure_native_byte_order)
    734     else:
--> 735         with open(filename, "rb") as f:
    736             with _validate_fileobject_and_memmap(f, filename, mmap_mode) as (
    737                 fobj,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/nlp-spacy-model/nlp_config.joblib'

## === cell 6
print("Generating test set.")
test = utils.generate_data(nlp_model = nlp, filename = "test.csv", cpu = -1)
print("Test set generation completed.")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3665804902.py in <cell line: 0>()
      1 # get test set:
      2 print("Generating test set.")
----> 3 test = utils.generate_data(nlp_model = nlp, filename = "test.csv", cpu = -1)
      4 print("Test set generation completed.")

NameError: name 'utils' is not defined

## === cell 7
train = pl.read_parquet("/kaggle/input/c14-lal-traindataset/train_processed.parquet")
col_selected = train.select(pl.all().exclude(['essay_id', 'full_text', 'score'])).columns
train.sample(1)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1872946569.py in <cell line: 0>()
----> 1 train = pl.read_parquet("/kaggle/input/c14-lal-traindataset/train_processed.parquet")
      2 #select the columns in order
      3 col_selected = train.select(pl.all().exclude(['essay_id', 'full_text', 'score'])).columns
      4 train.sample(1)

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
    112                 old_name, new_name, kwargs, function.__qualname__, version
    113             )
--> 114             return function(*args, **kwargs)
    115 
    116         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
    112                 old_name, new_name, kwargs, function.__qualname__, version
    113             )
--> 114             return function(*args, **kwargs)
    115 
    116         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/io/parquet/functions.py in read_parquet(source, columns, n_rows, row_index_name, row_index_offset, parallel, use_statistics, hive_partitioning, glob, schema, hive_schema, try_parse_hive_dates, rechunk, low_memory, storage_options, credential_provider, retries, use_pyarrow, pyarrow_options, memory_map, include_file_paths, allow_missing_columns)
    250             lf = lf.select(columns)
    251 
--> 252     return lf.collect()
    253 
    254 

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
     86                 kwargs["engine"] = "old-streaming"
     87 
---> 88             return function(*args, **kwargs)
     89 
     90         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/lazyframe/frame.py in collect(self, type_coercion, _type_check, predicate_pushdown, projection_pushdown, simplify_expression, slice_pushdown, comm_subplan_elim, comm_subexpr_elim, cluster_with_columns, collapse_joins, no_optimization, engine, background, _check_order, _eager, **_kwargs)
   2186         # Only for testing purposes
   2187         callback = _kwargs.get("post_opt_callback", callback)
-> 2188         return wrap_df(ldf.collect(engine, callback))
   2189 
   2190     @overload

FileNotFoundError: No such file or directory (os error 2): /kaggle/input/c14-lal-traindataset/train_processed.parquet

This error occurred with the following context stack:
	[1] 'parquet scan'
	[2] 'sink'


## === cell 8
rs = joblib.load("/kaggle/input/c24-lal-m1-xgb/robust_scaler.joblib")
le = joblib.load("/kaggle/input/c24-lal-m1-xgb/label_encoder.joblib")
test_data = test.select(pl.col(col_selected))

test_data = rs.transform(test_data)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/929252420.py in <cell line: 0>()
      1 #obtain transformers and preprocess test :
----> 2 rs = joblib.load("/kaggle/input/c24-lal-m1-xgb/robust_scaler.joblib")
      3 le = joblib.load("/kaggle/input/c24-lal-m1-xgb/label_encoder.joblib")
      4 test_data = test.select(pl.col(col_selected))
      5 

/usr/local/lib/python3.11/dist-packages/joblib/numpy_pickle.py in load(filename, mmap_mode, ensure_native_byte_order)
    733             obj = _unpickle(fobj, ensure_native_byte_order=ensure_native_byte_order)
    734     else:
--> 735         with open(filename, "rb") as f:
    736             with _validate_fileobject_and_memmap(f, filename, mmap_mode) as (
    737                 fobj,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/c24-lal-m1-xgb/robust_scaler.joblib'

## === cell 9
models = []
for i in range(4):
    m = xgb.Booster()
    f_name = f"/kaggle/input/c24-lal-m1-xgb/model_xgb_{i}.json"
    print(f'loading {f_name}')
    m.load_model(f_name)
    models.append(m)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_11/205120018.py in <cell line: 0>()
      5     f_name = f"/kaggle/input/c24-lal-m1-xgb/model_xgb_{i}.json"
      6     print(f'loading {f_name}')
----> 7     m.load_model(f_name)
      8     models.append(m)

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in load_model(self, fname)
   2586             # from URL.
   2587             fname = os.fspath(os.path.expanduser(fname))
-> 2588             _check_call(_LIB.XGBoosterLoadModel(self.handle, c_str(fname)))
   2589         elif isinstance(fname, bytearray):
   2590             buf = fname

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _check_call(ret)
    280     """
    281     if ret != 0:
--> 282         raise XGBoostError(py_str(_LIB.XGBGetLastError()))
    283 
    284 

XGBoostError: [16:47:34] /workspace/src/common/io.cc:147: Opening /kaggle/input/c24-lal-m1-xgb/model_xgb_0.json failed: No such file or directory
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x1ba24e) [0x7fff786fb24e]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x1e66f3) [0x7fff787276f3]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x16e731) [0x7fff786af731]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterLoadModel+0xb9) [0x7fff786af9f9]
  [bt] (4) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (5) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]
  [bt] (6) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7ffff63bc4d8]
  [bt] (7) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0x9c8e) [0x7ffff63bbc8e]
  [bt] (8) /usr/bin/python3(_PyObject_MakeTpCall+0x27c) [0x52f85c]



## === cell 10
print(test.shape)
submission = {'essay_id': test.select('essay_id')}
submission['pre_score'] = np.zeros(shape = (test.shape[0], 4))
for i in range(4): 
    best_iteration = models[i].best_iteration
    print(f"best iteration: {best_iteration}")
    pred = models[i].predict(xgb.DMatrix(test_data), iteration_range = (0, best_iteration + 1))
    submission['pre_score'][:,i] = pred
    
        
        
submission['score'] = np.mean(submission['pre_score'], axis = 1).round().astype(np.uint8)
submission['score'] = le.inverse_transform(submission['score'])
submission.pop('pre_score')

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3345365525.py in <cell line: 0>()
      1 #predict:
----> 2 print(test.shape)
      3 submission = {'essay_id': test.select('essay_id')}
      4 submission['pre_score'] = np.zeros(shape = (test.shape[0], 4))
      5 for i in range(4):

NameError: name 'test' is not defined

## === cell 11
submission
subs = pl.DataFrame(submission)
subs.write_csv("submission.csv")
subs.head()

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/396348094.py in <cell line: 0>()
----> 1 submission
      2 subs = pl.DataFrame(submission)
      3 subs.write_csv("submission.csv")
      4 subs.head()

NameError: name 'submission' is not defined
