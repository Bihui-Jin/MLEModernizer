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
Predict whether a lesion is malignant (0 denotes **benign**, and 1 indicates **malignant**).

## Metric
Area under the ROC curve.

## Submission Format
For each `image_name` in the test set, you must predict the probability (`target`) that the sample is **malignant**. The file should contain a header and have the following format:

```
image_name,target
ISIC_0052060,0.7
ISIC_0052349,0.9
ISIC_0058510,0.8
ISIC_0073313,0.5
ISIC_0073502,0.5
etc.
```

## Dataset 
The images are provided in DICOM format.

Images are also provided in JPEG and TFRecord format (in the `jpeg` and `tfrecords` directories, respectively). Images in TFRecord format have been resized to a uniform 1024x1024.

Metadata is also provided outside of the DICOM format, in CSV files. See the `Columns` section for a description.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `image_name` - unique identifier, points to filename of related DICOM image
- `patient_id` - unique patient identifier
- `sex` - the sex of the patient (when unknown, will be blank)
- `age_approx` - approximate patient age at time of imaging
- `anatom_site_general_challenge` - location of imaged site
- `diagnosis` - detailed diagnosis information (train only)
- `benign_malignant` - indicator of malignancy of imaged lesion
- `target` - binarized version of the target variable

# 2. Python version

3.8

# 3. Installed packages

bayesian-optimization==3.1.0
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
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
        input/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
        working/
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
```

-> data/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/siim-isic-melanoma-classification/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> data/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> input/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> (stopped after 10 files for performance)

# 5. Target score

0.9413178099846004

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from sklearn import metrics

from scipy.stats import rankdata
from bayes_opt import BayesianOptimization



train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
test = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
sub = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')


## === cell 4
models = [ "getting-started-with-tfrecords", "melanoma-efficientnetb6-with-attention-mechanism", "triple-stratified-kfold-with-tfrecords"]
models = ["384-E6-with-2018","512-E6","768-E2","512-E5","effb3-fulldata-upsample","effb2-fulldata-upsample"]
models = ["effb0-fulldata-upsample","effb1-fulldata-upsample","effb2-512-fulldata-upsample","effb3-fulldata-upsample","effb4-fulldata-upsample","effb5-fulldata-upsample"]

models = ["effb1-fulldata-upsample",
          "melanoma-efficientnetb6-with-attention-mechanism",
          "effb2-fulldata-upsample",
          "effb4-fulldata-upsample",
          "384-E6-with-2018",
          "effb0-fulldata-upsample"]

models = ["384-E6-with-2018","512-E6","768-E2","512-E5","effb2-fulldata-upsample",
         "effb1-fulldata-upsample"]


for model in models:
    dirname = "/kaggle/input/siim3reorg/" + model
    _oof = pd.read_csv(os.path.join(dirname, "oof.csv"))
    score = metrics.roc_auc_score(_oof['target'], _oof['pred'])
    print(f"{model}: OOF auc:{score:.4}")

    _oof = _oof.rename(columns={"pred":model}).drop(["target"],axis=1)
    if "fold" in _oof.columns:
        _oof = _oof.drop(["fold"],axis=1)

    train = train.merge(_oof, on="image_name")   


    _sub = pd.read_csv(os.path.join(dirname, "submission.csv"))
    _sub.columns = ["image_name",model]    
    test = test.merge(_sub, on="image_name")   


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2810477661.py in <cell line: 0>()
     16 for model in models:
     17     dirname = "/kaggle/input/siim3reorg/" + model
---> 18     _oof = pd.read_csv(os.path.join(dirname, "oof.csv"))
     19     score = metrics.roc_auc_score(_oof['target'], _oof['pred'])
     20     print(f"{model}: OOF auc:{score:.4}")

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/siim3reorg/384-E6-with-2018/oof.csv'

## === cell 5
train.head()

## === cell 8
train["pred_rank"] = 0
train["pred_power"] = 0
train["pred_avg"] = 0

for c in models:
    train["pred_rank"] += train[c].rank() / train[c].rank().max()
    train["pred_power"] += np.power(train[c],2)/np.power(train[c],2).max()
    train["pred_avg"] += train [c]/train [c].max()
    
train["pred_rank"] = train["pred_rank"]/len(models)
train["pred_power"] = train["pred_power"]/len(models)
train["pred_avg"] = train["pred_avg"]/len(models)


score = metrics.roc_auc_score(train['target'], train["pred_avg"])
print(f'OOF avg_auc:{score}')
   
    
score = metrics.roc_auc_score(train['target'], train["pred_rank"])
print(f'OOF rank_auc:{score}')

score = metrics.roc_auc_score(train['target'], train["pred_power"])
print(f'OOF pow_auc:{score}')

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

KeyError: '384-E6-with-2018'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2144316096.py in <cell line: 0>()
      4 
      5 for c in models:
----> 6     train["pred_rank"] += train[c].rank() / train[c].rank().max()
      7     train["pred_power"] += np.power(train[c],2)/np.power(train[c],2).max()
      8     train["pred_avg"] += train [c]/train [c].max()

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

KeyError: '384-E6-with-2018'

## === cell 10
test["target"] = 0.0
for c in models:
    test["target"] += test[c].rank() / test[c].rank().max()
test["target"] = test["target"]/len(models) 
    
sub = test[["image_name","target"]]
sub.to_csv("submission_rank.csv",index=False)
sub.head()

## --- ERROR in cell 10, traceback:
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

KeyError: '384-E6-with-2018'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/447542546.py in <cell line: 0>()
      1 test["target"] = 0.0
      2 for c in models:
----> 3     test["target"] += test[c].rank() / test[c].rank().max()
      4 test["target"] = test["target"]/len(models)
      5 

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

KeyError: '384-E6-with-2018'

## === cell 11
test["target"] = 0.0
for c in models:
    test["target"] += np.power(test[c],2)/np.power(test[c],2).max()
test["target"] = test["target"]/len(models) 
    
sub = test[["image_name","target"]]
sub.to_csv("submission_pow.csv",index=False)
sub.head()

## --- ERROR in cell 11, traceback:
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

KeyError: '384-E6-with-2018'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1959298526.py in <cell line: 0>()
      1 test["target"] = 0.0
      2 for c in models:
----> 3     test["target"] += np.power(test[c],2)/np.power(test[c],2).max()
      4 test["target"] = test["target"]/len(models)
      5 

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

KeyError: '384-E6-with-2018'

## === cell 12
test["target"] = 0.0
for c in models:
    test["target"] += test[c]/test[c].max()
test["target"] = test["target"]/len(models) 
    
sub = test[["image_name","target"]]
sub.to_csv("submission_avg.csv",index=False)
sub.head()

## --- ERROR in cell 12, traceback:
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

KeyError: '384-E6-with-2018'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3420094480.py in <cell line: 0>()
      1 test["target"] = 0.0
      2 for c in models:
----> 3     test["target"] += test[c]/test[c].max()
      4 test["target"] = test["target"]/len(models)
      5 

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

KeyError: '384-E6-with-2018'

## === cell 14
def dim_optimizer (df_oof, features, init_points = 20, n_iter = 30  ):
    pbounds = {'c0': (0.0, 1.0), 'c1': (0.0, 1.0), 'c2': (0.0, 1.0),'c3': (0.0, 1.0),'c4': (0.0, 1.0),'c5': (0.0, 1.0)}
   
    features = features

    def dim_opt (df_oof, c0,c1,c2,c3,c4,c5):

        x = c0*df_oof[  features[0] ] + c1*df_oof[ features[1]] + c2*df_oof[ features[2]] + c3*df_oof[ features[3]] + c4*df_oof[ features[4]] + c5*df_oof[ features[5]]
        return metrics.roc_auc_score(df_oof['target'], x)



    def q (c0, c1,c2,c3,c4,c5):
        return dim_opt  ( df_oof,  c0, c1,c2,c3,c4,c5)

    optimizer = BayesianOptimization(
        f=q,
        pbounds=pbounds,
        random_state=42,
    )


    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter,
    )

    c0 = optimizer.max["params"]["c0"]
    c1 = optimizer.max["params"]["c1"]
    c2= optimizer.max["params"]["c2"]
    c3= optimizer.max["params"]["c3"]
    c4= optimizer.max["params"]["c4"]
    c5= optimizer.max["params"]["c5"]
    
    t = optimizer.max["target"]
    print ( f'bo auc:{t}, c0:{c0}, c1:{c1}, c2:{c2},c3:{c3},c4:{c4},c5:{c5}' )
    
    
    return c0, c1, c2,c3,c4,c5


c0, c1, c2,c3,c4,c5 = dim_optimizer (train, models, init_points = 40, n_iter = 40  )

print (models[0],c0)
print (models[1],c1)
print (models[2],c2)
print (models[3],c3)
print (models[4],c4)
print (models[5],c5)


## --- ERROR in cell 14, traceback:
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

KeyError: '384-E6-with-2018'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2293439790.py in <cell line: 0>()
     40 
     41 
---> 42 c0, c1, c2,c3,c4,c5 = dim_optimizer (train, models, init_points = 40, n_iter = 40  )
     43 
     44 print (models[0],c0)

/tmp/ipykernel_11/2293439790.py in dim_optimizer(df_oof, features, init_points, n_iter)
     21 
     22 
---> 23     optimizer.maximize(
     24         init_points=init_points,
     25         n_iter=n_iter,

/usr/local/lib/python3.11/dist-packages/bayes_opt/bayesian_optimization.py in maximize(self, init_points, n_iter)
    320                 x_probe = self.suggest()
    321                 iteration += 1
--> 322             self.probe(x_probe, lazy=False)
    323 
    324             if self._bounds_transformer and iteration > 0:

/usr/local/lib/python3.11/dist-packages/bayes_opt/bayesian_optimization.py in probe(self, params, lazy)
    237             self._queue.append(params)
    238         else:
--> 239             self._space.probe(params)
    240             self.logger.log_optimization_step(
    241                 self._space.keys, self._space.res()[-1], self._space.params_config, self.max

/usr/local/lib/python3.11/dist-packages/bayes_opt/target_space.py in probe(self, params)
    553             error_msg = "No target function has been provided."
    554             raise ValueError(error_msg)
--> 555         target = self.target_func(**dict_params)
    556 
    557         if self._constraint is None:

/tmp/ipykernel_11/2293439790.py in q(c0, c1, c2, c3, c4, c5)
     12 
     13     def q (c0, c1,c2,c3,c4,c5):
---> 14         return dim_opt  ( df_oof,  c0, c1,c2,c3,c4,c5)
     15 
     16     optimizer = BayesianOptimization(

/tmp/ipykernel_11/2293439790.py in dim_opt(df_oof, c0, c1, c2, c3, c4, c5)
      6     def dim_opt (df_oof, c0,c1,c2,c3,c4,c5):
      7 
----> 8         x = c0*df_oof[  features[0] ] + c1*df_oof[ features[1]] + c2*df_oof[ features[2]] + c3*df_oof[ features[3]] + c4*df_oof[ features[4]] + c5*df_oof[ features[5]]
      9         return metrics.roc_auc_score(df_oof['target'], x)
     10 

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

KeyError: '384-E6-with-2018'

## === cell 15
def bo_pred (df):
    x = c0*df[  models[0] ] + c1*df[ models[1]] + c2*df[ models[2]] + c3*df[ models[3]] + c4*df[ models[4]] + c5*df[ models[5]]
    return x

train["pred"] = bo_pred (train)
score = metrics.roc_auc_score(train['target'], train['pred'])
print(f"auc bo:{score}")



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1050604188.py in <cell line: 0>()
      3     return x
      4 
----> 5 train["pred"] = bo_pred (train)
      6 score = metrics.roc_auc_score(train['target'], train['pred'])
      7 print(f"auc bo:{score}")

/tmp/ipykernel_11/1050604188.py in bo_pred(df)
      1 def bo_pred (df):
----> 2     x = c0*df[  models[0] ] + c1*df[ models[1]] + c2*df[ models[2]] + c3*df[ models[3]] + c4*df[ models[4]] + c5*df[ models[5]]
      3     return x
      4 
      5 train["pred"] = bo_pred (train)

NameError: name 'c0' is not defined

## === cell 17
test["target"] = bo_pred (test)
    
sub = test[["image_name","target"]]
sub.to_csv("submission_bo.csv",index=False)
sub.head()

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2463619523.py in <cell line: 0>()
----> 1 test["target"] = bo_pred (test)
      2 
      3 sub = test[["image_name","target"]]
      4 sub.to_csv("submission_bo.csv",index=False)
      5 sub.head()

/tmp/ipykernel_11/1050604188.py in bo_pred(df)
      1 def bo_pred (df):
----> 2     x = c0*df[  models[0] ] + c1*df[ models[1]] + c2*df[ models[2]] + c3*df[ models[3]] + c4*df[ models[4]] + c5*df[ models[5]]
      3     return x
      4 
      5 train["pred"] = bo_pred (train)

NameError: name 'c0' is not defined
