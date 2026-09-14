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

0.0108952342277427

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

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3877250672.py in <cell line: 0>()
      1 #install the libs:
----> 2 utils.install_libs_1()

NameError: name 'utils' is not defined

## === cell 3
import textstat
from spellchecker import SpellChecker
import textdescriptives as td 
import asent # sentiment analysis
from spacy.language import Language
import spacy


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/938202697.py in <cell line: 0>()
      1 #import the installed objects:
----> 2 import textstat
      3 from spellchecker import SpellChecker
      4 import textdescriptives as td
      5 import asent # sentiment analysis

ModuleNotFoundError: No module named 'textstat'

## === cell 4
@Language.component("spellcheck")
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

    spell_ratio =len(spell.unknown(words))/len(words) 
    doc._.ratio_new_spelling = spell_ratio

    return doc 

import joblib

config = joblib.load("/kaggle/input/nlp-model-large/nlp_config.joblib")
lang_cls = spacy.util.get_lang_class(config["nlp"]["lang"])
nlp = lang_cls.from_config(config)
bytes_data = joblib.load("/kaggle/input/nlp-model-large/nlp_bytes.joblib")
nlp.from_bytes(bytes_data)


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3893375352.py in <cell line: 0>()
      1 #get the nlp model and generate test set:
----> 2 @Language.component("spellcheck")
      3 def spell_check(doc):
      4     spell = SpellChecker()
      5     words = []

NameError: name 'Language' is not defined

## === cell 5
print("Generating test set.")
test = utils.generate_data(nlp_model = nlp, filename = "test.csv", cpu = -1)
print("Test set generation completed.")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3665804902.py in <cell line: 0>()
      1 # get test set:
      2 print("Generating test set.")
----> 3 test = utils.generate_data(nlp_model = nlp, filename = "test.csv", cpu = -1)
      4 print("Test set generation completed.")

NameError: name 'utils' is not defined

## === cell 6
rs = joblib.load("/kaggle/input/c24-lal-m1-xgb/robust_scaler.joblib")
le = joblib.load("/kaggle/input/c24-lal-m1-xgb/label_encoder.joblib")
test_data = test.select(pl.all().exclude(['essay_id', 'full_text']))
test_data = rs.transform(test_data.to_numpy())


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1427070487.py in <cell line: 0>()
      1 #obtain transformers and preprocess test :
----> 2 rs = joblib.load("/kaggle/input/c24-lal-m1-xgb/robust_scaler.joblib")
      3 le = joblib.load("/kaggle/input/c24-lal-m1-xgb/label_encoder.joblib")
      4 test_data = test.select(pl.all().exclude(['essay_id', 'full_text']))
      5 test_data = rs.transform(test_data.to_numpy())

NameError: name 'joblib' is not defined

## === cell 7
models = []
for i in range(4):
    m = xgb.Booster()
    f_name = f"/kaggle/input/c24-lal-m1-xgb/model_xgb_{i}.json"
    print(f'loading {f_name}')
    m.load_model(f_name)
    models.append(m)

## --- ERROR in cell 7, traceback:
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

XGBoostError: [03:02:13] /workspace/src/common/io.cc:147: Opening /kaggle/input/c24-lal-m1-xgb/model_xgb_0.json failed: No such file or directory
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



## === cell 8
submission = {'essay_id': test.select('essay_id')}
for i in range(4):
    if i == 0:
        best_iteration = models[i].best_iteration
        pred = models[i].predict(xgb.DMatrix(test_data), iteration_range = (0, best_iteration + 1))
        submission['score'] = pred
    else:
        best_iteration = models[i].best_iteration
        pred = models[i].predict(xgb.DMatrix(test_data), iteration_range = (0, best_iteration + 1))
        submission['score'] += pred
        
submission['score'] = (submission['score']/4).round().astype(np.int64)
submission['score'] = le.inverse_transform(submission['score'])

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3872447061.py in <cell line: 0>()
      1 #predict:
----> 2 submission = {'essay_id': test.select('essay_id')}
      3 for i in range(4):
      4     if i == 0:
      5         best_iteration = models[i].best_iteration

NameError: name 'test' is not defined

## === cell 9
subs = pl.DataFrame(submission)
subs.write_csv("submission.csv")
subs.head()

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2454152631.py in <cell line: 0>()
----> 1 subs = pl.DataFrame(submission)
      2 subs.write_csv("submission.csv")
      3 subs.head()

NameError: name 'submission' is not defined
