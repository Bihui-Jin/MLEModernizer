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
Given time series of breaths, predict the airway pressure in the respiratory circuit during the breath, given the time series of control inputs.

The best submissions will take lung attributes compliance and resistance into account.

## Metric
Mean absolute error between the predicted and actual pressures during the inspiratory phase of each breath. The expiratory phase is not scored.

## Submission Format
For each `id` in the test set, you must predict a value for the `pressure` variable. The file should contain a header and have the following format:

```
id,pressure
1,20
2,23
3,24
etc.
```

## Dataset
The ventilator data used in this competition was produced using a modified [open-source ventilator](https://pvp.readthedocs.io/) connected to an [artificial bellows test lung](https://www.ingmarmed.com/product/quicklung/) via a respiratory circuit. The diagram below illustrates the setup, with the two control inputs highlighted in green and the state variable (airway pressure) to predict in blue. The first control input is a continuous variable from 0 to 100 representing the percentage the inspiratory solenoid valve is open to let air into the lung (i.e., 0 is completely closed and no air is let in and 100 is completely open). The second control input is a binary variable representing whether the exploratory valve is open (1) or closed (0) to let air out.

![Ventilator diagram](https://raw.githubusercontent.com/google/deluca-lung/main/assets/2020-10-02%20Ventilator%20diagram.svg)

Each time series represents an approximately 3-second breath. The files are organized such that each row is a time step in a breath and gives the two control signals, the resulting airway pressure, and relevant attributes of the lung, described below.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - globally-unique time step identifier across an entire file
- `breath_id` - globally-unique time step for breaths
- `R` - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change `R` by changing the diameter of the straw, with higher `R` being harder to blow.
- `C` - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change `C` by changing the thickness of the balloon’s latex, with higher `C` having thinner latex and easier to blow.
- `time_step` - the actual time stamp.
- `u_in` - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.
- `u_out` - the control input for the exploratory solenoid valve. Either 0 or 1.
- `pressure` - the airway pressure measured in the respiratory circuit, measured in cmH2O.

# 2. Python version

3.9

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        input/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        working/
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
```

-> data/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/ventilator-pressure-prediction/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> input/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> (stopped after 10 files for performance)

# 5. Target score

8.821502695187858

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
!pip install -Uqq fastbook kaggle waterfallcharts treeinterpreter dtreeviz
import fastbook
fastbook.setup_book()

import seaborn as sns
from fastbook import *
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from fastai.tabular.all import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from dtreeviz.trees import *
from IPython.display import Image, display_svg, SVG, clear_output
from tqdm.auto import tqdm

pd.options.display.max_rows = 20
pd.options.display.max_columns = 10


def predict_batch(self, df):
    dl = self.dls.test_dl(df_test)
    dl.dataset.conts = dl.dataset.conts.astype(np.float32)
    preds, targs = self.get_preds(dl=dl)
    return preds, targs

Learner.predict_batch = predict_batch

## === cell 2
path = Path("../input/ventpressure2")
save_path = Path("/kaggle/working")

df_nn_final = pd.read_csv("../input/ventpressure2/train_preprocessed.csv")
to_drop = ['breathId_uIn_diffmean', 'uIn_diff3', 'breathId_uIn_diffmax']
try: df_nn_final = df_nn_final.drop(to_drop, axis=1)
except Exception: pass

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/777311192.py in <cell line: 0>()
      2 save_path = Path("/kaggle/working")
      3 
----> 4 df_nn_final = pd.read_csv("../input/ventpressure2/train_preprocessed.csv")
      5 to_drop = ['breathId_uIn_diffmean', 'uIn_diff3', 'breathId_uIn_diffmax']
      6 try: df_nn_final = df_nn_final.drop(to_drop, axis=1)

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '../input/ventpressure2/train_preprocessed.csv'

## === cell 3
dep_var = "pressure"
splits = load_pickle("../input/ventpressure1/split.pkl")
cont_nn, cat_nn = cont_cat_split(df_nn_final, max_card=9000, dep_var=dep_var)
cont_nn

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/142655683.py in <cell line: 0>()
      1 dep_var = "pressure"
----> 2 splits = load_pickle("../input/ventpressure1/split.pkl")
      3 cont_nn, cat_nn = cont_cat_split(df_nn_final, max_card=9000, dep_var=dep_var)
      4 cont_nn

/usr/local/lib/python3.11/dist-packages/fastcore/xtras.py in load_pickle(fn)
    263     "Load a pickle file from a file name or opened file"
    264     import pickle
--> 265     with open_file(fn, 'rb') as f: return pickle.load(f)
    266 
    267 # %% ../nbs/03_xtras.ipynb

/usr/local/lib/python3.11/dist-packages/fastcore/xtras.py in open_file(fn, mode, **kwargs)
    251     elif fn.suffix=='.gz' : return gzip.GzipFile(fn, mode, **kwargs)
    252     elif fn.suffix=='.zip': return zipfile.ZipFile(fn, mode, **kwargs)
--> 253     else: return open(fn,mode, **kwargs)
    254 
    255 # %% ../nbs/03_xtras.ipynb

FileNotFoundError: [Errno 2] No such file or directory: '../input/ventpressure1/split.pkl'

## === cell 4
df_nn_final[cat_nn].nunique()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1048349347.py in <cell line: 0>()
----> 1 df_nn_final[cat_nn].nunique()

NameError: name 'df_nn_final' is not defined

## === cell 6
procs_nn = [Categorify, FillMissing, Normalize]
to_nn = TabularPandas(df_nn_final, procs_nn, cat_nn, cont_nn, splits=splits, 
                     y_names=dep_var)
dls = to_nn.dataloaders(1024)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1687340293.py in <cell line: 0>()
      1 procs_nn = [Categorify, FillMissing, Normalize]
----> 2 to_nn = TabularPandas(df_nn_final, procs_nn, cat_nn, cont_nn, splits=splits, 
      3                      y_names=dep_var)
      4 dls = to_nn.dataloaders(1024)

NameError: name 'df_nn_final' is not defined

## === cell 8
y = to_nn.train.y
y.min(), y.max()

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/910844998.py in <cell line: 0>()
----> 1 y = to_nn.train.y
      2 y.min(), y.max()

NameError: name 'to_nn' is not defined

## === cell 10
learn = tabular_learner(dls, y_range=(-2, 65), layers=[400, 300, 200, 100], 
                       n_out=1, loss_func=F.mse_loss, metrics=mae)
learn.lr_find()

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/571344904.py in <cell line: 0>()
----> 1 learn = tabular_learner(dls, y_range=(-2, 65), layers=[400, 300, 200, 100], 
      2                        n_out=1, loss_func=F.mse_loss, metrics=mae)
      3 learn.lr_find()

NameError: name 'dls' is not defined

## === cell 12
learn.fit_one_cycle(7, 1e-2)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2809390495.py in <cell line: 0>()
----> 1 learn.fit_one_cycle(7, 1e-2)

NameError: name 'learn' is not defined

## === cell 13
preds, targs = learn.get_preds()
mean_absolute_error(targs, preds)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1009132729.py in <cell line: 0>()
----> 1 preds, targs = learn.get_preds()
      2 mean_absolute_error(targs, preds)

NameError: name 'learn' is not defined

## === cell 14
learn.save(save_path/"nn")

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3257013725.py in <cell line: 0>()
----> 1 learn.save(save_path/"nn")

NameError: name 'learn' is not defined

## === cell 16
def rf(xs, y, n_estimators=40, max_samples=200_000, 
      max_features=0.5, min_samples_leaf=5, **kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators,
        max_samples=max_samples, max_features=max_features,
        min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)


def m_mae(m, xs, y): return mean_absolute_error(y, m.predict(xs))

## === cell 17
cont, cat = cont_cat_split(df_nn_final, 5, dep_var=dep_var)
procs = [Categorify, FillMissing]
to = TabularPandas(df_nn_final, procs, cat, cont, y_names=dep_var, splits=splits)
xs_final, y = to.train.xs, to.train.y
valid_xs_final, valid_y = to.valid.xs, to.valid.y

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2037976217.py in <cell line: 0>()
----> 1 cont, cat = cont_cat_split(df_nn_final, 5, dep_var=dep_var)
      2 procs = [Categorify, FillMissing]
      3 to = TabularPandas(df_nn_final, procs, cat, cont, y_names=dep_var, splits=splits)
      4 xs_final, y = to.train.xs, to.train.y
      5 valid_xs_final, valid_y = to.valid.xs, to.valid.y

NameError: name 'df_nn_final' is not defined

## === cell 18
try:
    to_drop = ["breathId_uIn_diffmean", "uIn_diff3", "breathId_uIn_diffmax"]
    xs_final = xs_final.drop(to_drop, axis=1)
    valid_xs_final = valid_xs_final.drop(to_drop, axis=1)
    print("Dropped")
except Exception as e: print(f"{type(e)}: {e}")

m = rf(xs_final, y, n_estimators=45)
m_mae(m, valid_xs_final, valid_y)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3170078451.py in <cell line: 0>()
      6 except Exception as e: print(f"{type(e)}: {e}")
      7 
----> 8 m = rf(xs_final, y, n_estimators=45)
      9 m_mae(m, valid_xs_final, valid_y)

NameError: name 'xs_final' is not defined

## === cell 19
rf_preds = m.predict(valid_xs_final)
ens_preds = (to_np(preds.squeeze()) + rf_preds) / 2
mean_absolute_error(valid_y, ens_preds)

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2914613171.py in <cell line: 0>()
----> 1 rf_preds = m.predict(valid_xs_final)
      2 ens_preds = (to_np(preds.squeeze()) + rf_preds) / 2
      3 mean_absolute_error(valid_y, ens_preds)

NameError: name 'm' is not defined

## === cell 21
df_test = pd.read_csv("../input/ventpressure2/test_preprocessed.csv")
df_test.columns

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2178166711.py in <cell line: 0>()
----> 1 df_test = pd.read_csv("../input/ventpressure2/test_preprocessed.csv")
      2 df_test.columns

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '../input/ventpressure2/test_preprocessed.csv'

## === cell 22
df_nn_final.columns

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2423884039.py in <cell line: 0>()
----> 1 df_nn_final.columns

NameError: name 'df_nn_final' is not defined

## === cell 23
to_drop_test = list(set(df_test.columns).difference(set(df_nn_final.columns)))
df_test = df_test.drop(to_drop_test, axis=1)
df_test.columns

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2166583541.py in <cell line: 0>()
----> 1 to_drop_test = list(set(df_test.columns).difference(set(df_nn_final.columns)))
      2 df_test = df_test.drop(to_drop_test, axis=1)
      3 df_test.columns

NameError: name 'df_test' is not defined

## === cell 24
preds, targs = learn.predict_batch(df_test)

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4258856602.py in <cell line: 0>()
----> 1 preds, targs = learn.predict_batch(df_test)

NameError: name 'learn' is not defined

## === cell 25
cont, cat = cont_cat_split(df_test, 5, dep_var=dep_var)
to_test = TabularPandas(df_test, procs, cat, cont)
to_test.xs

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3650602066.py in <cell line: 0>()
----> 1 cont, cat = cont_cat_split(df_test, 5, dep_var=dep_var)
      2 to_test = TabularPandas(df_test, procs, cat, cont)
      3 to_test.xs

NameError: name 'df_test' is not defined

## === cell 26
rf_preds = m.predict(to_test.xs)

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/463554006.py in <cell line: 0>()
----> 1 rf_preds = m.predict(to_test.xs)

NameError: name 'm' is not defined

## === cell 27
rf_preds = m.predict(to_test.xs)
ens_preds = (to_np(preds.squeeze()) + rf_preds) / 2

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/867548348.py in <cell line: 0>()
      1 # del rf_preds, ens_preds
----> 2 rf_preds = m.predict(to_test.xs)
      3 ens_preds = (to_np(preds.squeeze()) + rf_preds) / 2

NameError: name 'm' is not defined

## === cell 28
plt.plot(rf_preds[:80])

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3449647861.py in <cell line: 0>()
----> 1 plt.plot(rf_preds[:80])

NameError: name 'rf_preds' is not defined

## === cell 30
plt.plot(preds.squeeze()[:80])

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2722256351.py in <cell line: 0>()
----> 1 plt.plot(preds.squeeze()[:80])

NameError: name 'preds' is not defined

## === cell 32
ens_preds

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/781058706.py in <cell line: 0>()
----> 1 ens_preds

NameError: name 'ens_preds' is not defined

## === cell 33
submission = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")
submission.pressure = to_np(preds.squeeze())
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2275094450.py in <cell line: 0>()
      1 submission = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")
----> 2 submission.pressure = to_np(preds.squeeze())
      3 submission.to_csv("submission.csv", index=False)

NameError: name 'preds' is not defined
