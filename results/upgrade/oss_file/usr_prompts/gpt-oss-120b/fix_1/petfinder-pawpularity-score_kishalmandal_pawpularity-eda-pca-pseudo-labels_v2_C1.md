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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
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
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

43.04566903651843

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier

## === cell 1
df = pd.read_csv('../input/same-old-creating-folds/train_5folds.csv')

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3744023005.py in <cell line: 0>()
----> 1 df = pd.read_csv('../input/same-old-creating-folds/train_5folds.csv')

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/same-old-creating-folds/train_5folds.csv'

## === cell 2
df.shape

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/857714779.py in <cell line: 0>()
----> 1 df.shape

NameError: name 'df' is not defined

## === cell 3
df.head()

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/964094849.py in <cell line: 0>()
----> 1 df.head()

NameError: name 'df' is not defined

## === cell 4
X = df.drop(['Id', 'Pawpularity', 'kfold'], axis=1)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2299174940.py in <cell line: 0>()
----> 1 X = df.drop(['Id', 'Pawpularity', 'kfold'], axis=1)

NameError: name 'df' is not defined

## === cell 5
y = df['Pawpularity']

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2732942187.py in <cell line: 0>()
----> 1 y = df['Pawpularity']

NameError: name 'df' is not defined

## === cell 6
correlations = X.corr()
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1, cmap='RdPu')
fig.colorbar(cax)
ticks = np.arange(0,12,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(X.columns)
ax.set_yticklabels(X.columns)
plt.show()

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/981332644.py in <cell line: 0>()
----> 1 correlations = X.corr()
      2 fig = plt.figure(figsize=(10,10))
      3 ax = fig.add_subplot(111)
      4 cax = ax.matshow(correlations, vmin=-1, vmax=1, cmap='RdPu')
      5 fig.colorbar(cax)

NameError: name 'X' is not defined

## === cell 7
model = RandomForestRegressor()
model.fit(X, y)
importance1 = model.feature_importances_
d = pd.DataFrame()
d['imp'] = importance1
d['f'] = X.columns

d=d.sort_values('imp', ascending=False)

fig, ax = plt.subplots(figsize=(20,7))

ax.barh(d.f, d.imp, color='grey')
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)
    
ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
ax.invert_yaxis()
plt.show()


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/876177896.py in <cell line: 0>()
      1 model = RandomForestRegressor()
----> 2 model.fit(X, y)
      3 importance1 = model.feature_importances_
      4 d = pd.DataFrame()
      5 d['imp'] = importance1

NameError: name 'X' is not defined

## === cell 8
model = ExtraTreesRegressor()
model.fit(X, y)
importance2 = model.feature_importances_
d = pd.DataFrame()
d['imp'] = importance2
d['f'] = X.columns

d=d.sort_values('imp', ascending=False)

fig, ax = plt.subplots(figsize=(20,7))

ax.barh(d.f, d.imp, color='green')
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)
    
ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
ax.invert_yaxis()
plt.show()


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2187975918.py in <cell line: 0>()
      1 model = ExtraTreesRegressor()
----> 2 model.fit(X, y)
      3 importance2 = model.feature_importances_
      4 d = pd.DataFrame()
      5 d['imp'] = importance2

NameError: name 'X' is not defined

## === cell 9
model = XGBRegressor()
model.fit(X, y)
importance3 = model.feature_importances_
d = pd.DataFrame()
d['imp'] = importance3
d['f'] = X.columns

d=d.sort_values('imp', ascending=False)

fig, ax = plt.subplots(figsize=(20,7))

ax.barh(d.f, d.imp, color='violet')
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)
    
ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
ax.invert_yaxis()
plt.show()


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1174442963.py in <cell line: 0>()
      1 model = XGBRegressor()
----> 2 model.fit(X, y)
      3 importance3 = model.feature_importances_
      4 d = pd.DataFrame()
      5 d['imp'] = importance3

NameError: name 'X' is not defined

## === cell 10
importance = (importance1+importance2+importance3)/3

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/679773219.py in <cell line: 0>()
----> 1 importance = (importance1+importance2+importance3)/3

NameError: name 'importance1' is not defined

## === cell 11

d['imp'] = importance
d['f'] = X.columns

d=d.sort_values('imp', ascending=False)

fig, ax = plt.subplots(figsize=(20,7))

ax.barh(d.f, d.imp, color='crimson')
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)
    
ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
ax.invert_yaxis()
plt.show()


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1145451765.py in <cell line: 0>()
----> 1 d['imp'] = importance
      2 d['f'] = X.columns
      3 
      4 d=d.sort_values('imp', ascending=False)
      5 

NameError: name 'importance' is not defined

## === cell 12
rf = RandomForestRegressor()
et = ExtraTreesRegressor()
xgb = XGBRegressor()

## === cell 13
testdf = pd.read_csv('../input/petfinder-pawpularity-score/test.csv')

## === cell 14
xpred = testdf.drop(['Id'], axis=1).values

## === cell 15
pseudolabel = pd.DataFrame()
y_pred_all =[]
for fold in range(5):
    folddf = pd.DataFrame()
    train = df[df['kfold']!=fold]
    valid = df[df['kfold']==fold]
    
    xtrain = train.drop(['Id', 'Pawpularity', 'kfold'], axis=1).values
    xtest = valid.drop(['Id', 'Pawpularity', 'kfold'], axis=1)
    ytrain = train['Pawpularity'].values
    ytest = valid['Pawpularity'].values
    
    rf = RandomForestRegressor()
    et = ExtraTreesRegressor()
    xgb = XGBRegressor()
    
    rf.fit(xtrain, ytrain)
    et.fit(xtrain, ytrain)
    xgb.fit(xtrain, ytrain)
    
    ypred1 = rf.predict(xpred)
    ypred2 = et.predict(xpred)
    ypred3 = xgb.predict(xpred)
    ypred = (ypred1+ypred2+ypred3)/3
    
    y_pred_all.append(np.vstack(ypred/5))
    
    


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1837873874.py in <cell line: 0>()
      4 for fold in range(5):
      5     folddf = pd.DataFrame()
----> 6     train = df[df['kfold']!=fold]
      7     valid = df[df['kfold']==fold]
      8 

NameError: name 'df' is not defined

## === cell 16
y_pred_all = sum(y_pred_all)
y_pred_all = [y[0]/100 for y in y_pred_all]

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1828042991.py in <cell line: 0>()
      1 y_pred_all = sum(y_pred_all)
----> 2 y_pred_all = [y[0]/100 for y in y_pred_all]

TypeError: 'int' object is not iterable

## === cell 17
from xgboost import XGBClassifier
xgbc = XGBClassifier()

## === cell 18
xgb.fit(xpred, y_pred_all)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/744086382.py in <cell line: 0>()
----> 1 xgb.fit(xpred, y_pred_all)

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in fit(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)
   1053         with config_context(verbosity=self.verbosity):
   1054             evals_result: TrainingCallback.EvalsLog = {}
-> 1055             train_dmatrix, evals = _wrap_evaluation_matrices(
   1056                 missing=self.missing,
   1057                 X=X,

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in _wrap_evaluation_matrices(missing, X, y, group, qid, sample_weight, base_margin, feature_weights, eval_set, sample_weight_eval_set, base_margin_eval_set, eval_group, eval_qid, create_dmatrix, enable_categorical, feature_types)
    519     """Convert array_like evaluation matrices into DMatrix.  Perform validation on the
    520     way."""
--> 521     train_dmatrix = create_dmatrix(
    522         data=X,
    523         label=y,

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in _create_dmatrix(self, ref, **kwargs)
    961             except TypeError:  # `QuantileDMatrix` supports lesser types than DMatrix
    962                 pass
--> 963         return DMatrix(**kwargs, nthread=self.n_jobs)
    964 
    965     def _set_evaluation_result(self, evals_result: TrainingCallback.EvalsLog) -> None:

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in __init__(self, data, label, weight, base_margin, missing, silent, feature_names, feature_types, nthread, group, qid, label_lower_bound, label_upper_bound, feature_weights, enable_categorical, data_split_mode)
    867         self.handle = handle
    868 
--> 869         self.set_info(
    870             label=label,
    871             weight=weight,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in set_info(self, label, weight, base_margin, group, qid, label_lower_bound, label_upper_bound, feature_names, feature_types, feature_weights)
    930 
    931         if label is not None:
--> 932             self.set_label(label)
    933         if weight is not None:
    934             self.set_weight(weight)

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in set_label(self, label)
   1068         from .data import dispatch_meta_backend
   1069 
-> 1070         dispatch_meta_backend(self, label, "label", "float")
   1071 
   1072     def set_weight(self, weight: ArrayLike) -> None:

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in dispatch_meta_backend(matrix, data, name, dtype)
   1255         _meta_from_numpy(array, name, dtype, handle)
   1256         return
-> 1257     raise TypeError("Unsupported type for " + name, str(type(data)))
   1258 
   1259 

TypeError: ('Unsupported type for label', "<class 'int'>")

## === cell 19
y_pred_final = xgb.predict(xpred)

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/3342484305.py in <cell line: 0>()
----> 1 y_pred_final = xgb.predict(xpred)

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in predict(self, X, output_margin, validate_features, base_margin, iteration_range)
   1166             if self._can_use_inplace_predict():
   1167                 try:
-> 1168                     predts = self.get_booster().inplace_predict(
   1169                         data=X,
   1170                         iteration_range=iteration_range,

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in get_booster(self)
    723             from sklearn.exceptions import NotFittedError
    724 
--> 725             raise NotFittedError("need to call fit or load_model beforehand")
    726         return self._Booster
    727 

NotFittedError: need to call fit or load_model beforehand

## === cell 20
testdf['Pawpularity'] = y_pred_final

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3310234916.py in <cell line: 0>()
----> 1 testdf['Pawpularity'] = y_pred_final

NameError: name 'y_pred_final' is not defined

## === cell 21
testdf[['Id', 'Pawpularity']].to_csv('submission.csv', index=False)

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3468153449.py in <cell line: 0>()
----> 1 testdf[['Id', 'Pawpularity']].to_csv('submission.csv', index=False)

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

KeyError: "['Pawpularity'] not in index"
