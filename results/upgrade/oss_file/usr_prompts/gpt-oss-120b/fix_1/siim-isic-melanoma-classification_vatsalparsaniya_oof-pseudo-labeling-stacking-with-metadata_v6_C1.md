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

No external packages required in the script and installed.

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

0.9402423884987422

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 3
import re,os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from colorama import Fore, Back, Style

import lightgbm as lgb # CLF1
from sklearn.linear_model import LogisticRegression # CLF2
from xgboost import XGBRegressor # CLF3
from sklearn.naive_bayes import GaussianNB  # CLF4
from sklearn.ensemble import RandomForestClassifier # CLF5
from sklearn.linear_model import LinearRegression # CLF6
from sklearn.linear_model import Lasso # CLF7
from sklearn.linear_model import ElasticNet # CLF8
from sklearn.neighbors import KNeighborsRegressor # CLF9
from sklearn.tree import DecisionTreeRegressor # CLF10
from sklearn.ensemble import GradientBoostingRegressor # CLF11
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis #CLF12
from mlxtend.classifier import StackingClassifier # SCF

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_auc_score,roc_curve

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning, module='sklearn')
warnings.simplefilter('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

def seed_everything(SEED):
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

## === cell 4
FOLDS = 3
SEED = 123
Setup_Parameters = True
seed_everything(SEED)
file_add_list = [1,2,3,4,5]
pesudo_label = True
test_pipeline = True

## === cell 6
BASE_PATH = '../input/siim-isic-melanoma-classification'
train_metadata = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
test_metadata = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))
sample_submission = pd.read_csv(os.path.join(BASE_PATH, 'sample_submission.csv'))
tfrecord_number_df =  pd.read_csv('../input/stacking-data/Image_Name_TFRecord_number.csv')

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1881390649.py in <cell line: 0>()
      3 test_metadata = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))
      4 sample_submission = pd.read_csv(os.path.join(BASE_PATH, 'sample_submission.csv'))
----> 5 tfrecord_number_df =  pd.read_csv('../input/stacking-data/Image_Name_TFRecord_number.csv')

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/stacking-data/Image_Name_TFRecord_number.csv'

## === cell 8
print('Train data shape : ',train_metadata.shape)
print('Test data shape : ',test_metadata.shape)

## === cell 10
print('Unique values in column with frequency : ')

print('\nsex : ', dict(train_metadata.sex.value_counts()))
print('\nage_approx : ', dict(train_metadata.age_approx.value_counts()))
print('\nanatom_site_general_challenge : ', dict(train_metadata.anatom_site_general_challenge.value_counts()))
print('\ndiagnosis : ', dict(train_metadata.diagnosis.value_counts()))
print('\nbenign_malignant : ', dict(train_metadata.benign_malignant.value_counts()))
print('\ntarget : ', dict(train_metadata.target.value_counts()))

## === cell 12
print('Unique values in column with frequency : ')

print('\nsex : ', dict(test_metadata.sex.value_counts()))
print('\nage_approx : ', dict(test_metadata.age_approx.value_counts()))
print('\nanatom_site_general_challenge : ', dict(test_metadata.anatom_site_general_challenge.value_counts()))

## === cell 14
train = train_metadata.copy()
train['age_approx'] = train['age_approx'].fillna(train.age_approx.mean())
sex_code = pd.get_dummies(train.sex, prefix='sex')
anatom_site_general_challenge_code = pd.get_dummies(train.anatom_site_general_challenge, prefix='anatom_site')
age_aprox_normalized = (train.age_approx-train.age_approx.mean())/train.age_approx.std()
train_coded = pd.concat([train.image_name, sex_code, age_aprox_normalized, anatom_site_general_challenge_code , train.target], axis=1)
print('Shape : ',train_coded.shape)
train_coded.tail()

## === cell 16
def add_OOF_pred(train_coded,num):
    for n in file_add_list:
        df_ = pd.read_csv(f'../input/95-cv-oof-submission/oof_{n}.csv')
        train_coded = pd.merge(train_coded, df_[['image_name','pred']], on="image_name",how='right')
        train_coded.rename({'pred': f'pred_{n}'}, axis=1, inplace=True)
    return train_coded
train_coded = pd.merge(tfrecord_number_df, train_coded, on="image_name",how='left')
train_coded = add_OOF_pred(train_coded,5)
train_coded.to_csv('train_coded.csv',index=False)
train_coded.tail()

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/676292511.py in <cell line: 0>()
      5         train_coded.rename({'pred': f'pred_{n}'}, axis=1, inplace=True)
      6     return train_coded
----> 7 train_coded = pd.merge(tfrecord_number_df, train_coded, on="image_name",how='left')
      8 train_coded = add_OOF_pred(train_coded,5)
      9 train_coded.to_csv('train_coded.csv',index=False)

NameError: name 'tfrecord_number_df' is not defined

## === cell 18
test = test_metadata.copy()
test['age_approx'] = test['age_approx'].fillna(test.age_approx.mean())
sex_code = pd.get_dummies(test.sex, prefix='sex')
anatom_site_general_challenge_code = pd.get_dummies(test.anatom_site_general_challenge, prefix='anatom_site')
age_aprox_normalized = (test.age_approx-test.age_approx.mean())/test.age_approx.std()
test_coded = pd.concat([test.image_name, sex_code, age_aprox_normalized , anatom_site_general_challenge_code], axis=1)
print('Shape : ',test_coded.shape)
test_coded.tail()

## === cell 20
def add_submission_pred(test_coded,num):
    for n in file_add_list:
        df_ = pd.read_csv(f'../input/95-cv-oof-submission/submission_{n}.csv')
        test_coded = pd.merge(test_coded, df_[['image_name','target']], on="image_name",how='right')
        test_coded.rename({'target': f'pred_{n}'}, axis=1, inplace=True)
    return test_coded
test_coded = add_submission_pred(test_coded,5)
test_coded.to_csv('test_coded.csv',index=False)
test_coded.tail()

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3937713603.py in <cell line: 0>()
      5         test_coded.rename({'target': f'pred_{n}'}, axis=1, inplace=True)
      6     return test_coded
----> 7 test_coded = add_submission_pred(test_coded,5)
      8 test_coded.to_csv('test_coded.csv',index=False)
      9 test_coded.tail()

/tmp/ipykernel_11/3937713603.py in add_submission_pred(test_coded, num)
      1 def add_submission_pred(test_coded,num):
      2     for n in file_add_list:
----> 3         df_ = pd.read_csv(f'../input/95-cv-oof-submission/submission_{n}.csv')
      4         test_coded = pd.merge(test_coded, df_[['image_name','target']], on="image_name",how='right')
      5         test_coded.rename({'target': f'pred_{n}'}, axis=1, inplace=True)

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/95-cv-oof-submission/submission_1.csv'

## === cell 21
def crossValidate(CLF,X=train_coded,X_test=test_coded,FOLDS = 5,SEED = 123,show_roc_curve = False,pesudo_label = False):
    print(Fore.YELLOW)
    print('#'*60)
    model_name = type(CLF).__name__
    print('#### ',model_name)
    print('#'*60,Style.RESET_ALL)
    
    CV_Score = []
    Val_preds = []
    Val_imagenames = []
    val_targets = []
    
    CV_Score_pesudo = []
    Val_preds_pesudo = []
    Val_imagenames_pesudo = []
    val_targets_pesudo = []
    
    skf = KFold(n_splits=FOLDS,shuffle=True,random_state=SEED)

    for fold,(idxT,idxV) in enumerate(skf.split(np.arange(15))):

        idxT, idxV = X.tfrecord.isin(idxT), X.tfrecord.isin(idxV)

        X_train_main, y_train_main = X[idxT], X[idxT]
        X_val_main, y_val_main = X[idxV], X[idxV]
        print(Fore.MAGENTA)
        print('#'*60,Style.RESET_ALL)
        print(Fore.BLUE)
        print('FOLD : ',fold)
        print('Train TFrecords : ',X_train_main.tfrecord.unique())
        print('Validation TFrecords : ',X_val_main.tfrecord.unique())
        image_names = list(X_val_main['image_name'])
        
        X_train = X_train_main.drop(['target','tfrecord'],axis=1).iloc[:,1:]
        y_train = y_train_main['target']
        
        X_val = X_val_main.drop(['target','tfrecord'],axis=1).iloc[:,1:]
        y_val = y_val_main['target']
        
        CLF_pesudo = CLF
        CLF.fit(X_train, y_train)
        
        try:
            y_train_pred  = CLF.predict_proba(X_train)[:,1]
        except:
            y_train_pred = CLF.predict(X_train)
            
        print('Train AUC : ', roc_auc_score(y_train,y_train_pred))
        try:
            Val_pred  = CLF.predict_proba(X_val)[:,1]
        except:
            Val_pred = CLF.predict(X_val)
            
        Val_auc = roc_auc_score(y_val,Val_pred)
        print('Val AUC : ', Val_auc)
        
        CV_Score.append(Val_auc)
        Val_preds.append(Val_pred)
        Val_imagenames.append(image_names)
        val_targets.append(list(y_val))
        
        if pesudo_label:
            
            train2_Pesudo = X_train_main.copy()
            train2_Pesudo['target_label'] = y_train_main['target']
            test2_Pesudo = X_val_main.copy()
            test2_Pesudo['target_label'] = Val_pred
            
            test2_Pesudo = test2_Pesudo[ (test2_Pesudo['target_label'] >= 0.99) |  
                                            (test2_Pesudo['target_label'] >= 0.01)]
            test2_Pesudo.loc[ test2_Pesudo['target_label']>=0.5, 'target_label' ] = 1
            test2_Pesudo.loc[ test2_Pesudo['target_label']<0.5, 'target_label' ] = 0
            
            print(Fore.CYAN)
            print('Number of Pesudo Labeled Data added : ',len(test2_Pesudo))
            print('target_label = 1 : ',len(test2_Pesudo[test2_Pesudo['target_label'] == 1]))
            print('target_label = 0 : ',len(test2_Pesudo[test2_Pesudo['target_label'] == 0]))
            
            train_pesudo = pd.concat([train2_Pesudo,test2_Pesudo],axis=0)
            
            X_train_pesudo = train_pesudo.drop(['target_label','target','tfrecord'],axis=1).iloc[:,1:]
            y_train_pesudo = train_pesudo['target_label']
            
            CLF_pesudo.fit(X_train_pesudo, y_train_pesudo)
            
            try:
                y_train_pred_pesudo  = CLF_pesudo.predict_proba(X_train_pesudo)[:,1]
            except:
                y_train_pred_pesudo = CLF_pesudo.predict(X_train_pesudo)

            print('Pesudo Train AUC : ', roc_auc_score(y_train_pesudo,y_train_pred_pesudo))
            try:
                Val_pred_pesudo  = CLF_pesudo.predict_proba(X_val)[:,1]
            except:
                Val_pred_pesudo = CLF_pesudo.predict(X_val)
            
            Val_auc_pesudo = roc_auc_score(y_val,Val_pred_pesudo)
            print('Pesudo Val AUC : ', Val_auc_pesudo)
            print(Style.RESET_ALL)
            
            CV_Score_pesudo.append(Val_auc_pesudo)
            Val_preds_pesudo.append(Val_pred_pesudo)
            Val_imagenames_pesudo.append(image_names)
            val_targets_pesudo.append(list(y_val))
            
    valtargets = np.concatenate(val_targets)
    valpreds = np.concatenate(Val_preds)
    valimagenames = np.concatenate(Val_imagenames)
    
    auc_score = roc_auc_score(valtargets,valpreds)
    
    print(Fore.YELLOW)
    print('#'*60)
    print('\nCV(auc_score) : ',auc_score)
    print(f'Mean CV : {np.mean(CV_Score)} +/- {np.std(CV_Score)}\n')
    
    
    oof = pd.DataFrame()
    oof['image_name'] = valimagenames
    oof['pred'] = valpreds
    oof['target'] = valtargets
    
    
    Test_imagenames = X_test['image_name']
    X_test = X_test.iloc[:,1:]
    try:
        test_pred = CLF.predict_proba(X_test)[:,1]
    except:
        test_pred = CLF.predict(X_test)
        
    submission = pd.DataFrame()
    submission['image_name'] = Test_imagenames
    submission['target'] = test_pred
    
    if pesudo_label:
        valtargets_pesudo = np.concatenate(val_targets_pesudo)
        valpreds_pesudo = np.concatenate(Val_preds_pesudo)
        valimagenames_pesudo = np.concatenate(Val_imagenames_pesudo)
        
        auc_score_pesudo = roc_auc_score(valtargets_pesudo,valpreds_pesudo)
        
        print('Pesudo CV(auc_score) : ',auc_score_pesudo)
        print(f'Pesudo Mean CV : {np.mean(CV_Score_pesudo)} +/- {np.std(CV_Score_pesudo)}\n')
        
        oof_pesudo = pd.DataFrame()
        oof_pesudo['image_name'] = valimagenames_pesudo
        oof_pesudo['pred'] = valpreds_pesudo
        oof_pesudo['target'] = valtargets_pesudo
        
        try:
            test_pred_pesudo = CLF_pesudo.predict_proba(X_test)[:,1]
        except:
            test_pred_pesudo = CLF_pesudo.predict(X_test)

        submission_pesudo = pd.DataFrame()
        submission_pesudo['image_name'] = Test_imagenames
        submission_pesudo['target'] = test_pred_pesudo
        
    
    print('#'*60,Style.RESET_ALL)
    if show_roc_curve:
        fpr, tpr, _ = roc_curve(valtargets,valpreds)
        if pesudo_label:
            fpr_p, tpr_p, _ = roc_curve(valtargets_pesudo,valpreds_pesudo)
            
        plt.figure()
        lw = 2
        if pesudo_label:
            plt.plot(fpr_p, tpr_p, color='red',
                 lw=lw, label=f'Pesudo ROC curve (area = {auc_score_pesudo:0.4f})')
            
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label=f'ROC curve (area = {auc_score:0.4f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve by : {model_name}')
        plt.legend(loc="lower right")
        plt.show()
    
    if pesudo_label:
        return oof , submission, auc_score, model_name, oof_pesudo, submission_pesudo, auc_score_pesudo, str(model_name) + '_Pesudo'
    return oof , submission, auc_score, model_name

## === cell 24
clf1 = lgb.LGBMClassifier(max_depth=5, 
                          metric="auc", 
                          n_estimators=100, 
                          num_leaves=5, 
                          boosting_type="gbdt", 
                          learning_rate=0.1, 
                          feature_fraction=0.05, 
                          colsample_bytree=0.1, 
                          bagging_fraction=0.8, 
                          bagging_freq=2, 
                          reg_lambda=0.2)

## === cell 26
clf2 = LogisticRegression(
            C= 1.0,
            class_weight=None,
            dual= False,
            fit_intercept= True,
            intercept_scaling= 1,
            l1_ratio= None,
            max_iter= 100,
            multi_class= 'auto',
            n_jobs= None,
            penalty= 'l2',
            random_state= None,
            solver= 'lbfgs',
            tol= 0.0001,
            verbose= 0,
            warm_start= False)

## === cell 28
clf3 = XGBRegressor(base_score=0.5, 
                    booster=None, 
                    colsample_bylevel=1,
                    colsample_bynode=1, 
                    colsample_bytree=0.8, 
                    gamma=1, 
                    gpu_id=-1,
                    importance_type='gain', 
                    interaction_constraints=None,
                    learning_rate=0.002, 
                    max_delta_step=0, 
                    max_depth=10,
                    min_child_weight=1, 
                    missing=None, 
                    monotone_constraints=None,
                    n_estimators=700, 
                    n_jobs=-1, 
                    nthread=-1, 
                    num_parallel_tree=1,
                    objective='binary:logistic', 
                    random_state=0,
                    reg_alpha=0, 
                    reg_lambda=1, 
                    scale_pos_weight=1,
                    subsample=0.8,
                    tree_method=None, 
                    validate_parameters=False, 
                    verbosity=None)

## === cell 30
clf4 = GaussianNB(
    priors= None, 
    var_smoothing= 1e-09)

## === cell 32
clf5 = RandomForestClassifier(
        bootstrap= True,
        ccp_alpha= 0.0,
        class_weight= None,
        criterion= 'gini',
        max_depth= 5,
        max_features= 'auto',
        max_leaf_nodes= 30,
        max_samples= None,
        min_impurity_decrease= 0.0,
        min_impurity_split= None,
        min_samples_leaf= 2,
        min_samples_split= 100,
        min_weight_fraction_leaf= 0.0,
        n_estimators= 300,
        n_jobs= None,
        oob_score= False,
        random_state= None,
        verbose= 0,
        warm_start= False)

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1176959523.py in <cell line: 0>()
----> 1 clf5 = RandomForestClassifier(
      2         bootstrap= True,
      3         ccp_alpha= 0.0,
      4         class_weight= None,
      5         criterion= 'gini',

TypeError: RandomForestClassifier.__init__() got an unexpected keyword argument 'min_impurity_split'

## === cell 40
clf9 = KNeighborsRegressor(algorithm= 'auto',
                            leaf_size= 30,
                            metric= 'minkowski',
                            metric_params= None,
                            n_jobs= None,
                            n_neighbors= 10,
                            p= 5,
                            weights= 'uniform')

## === cell 42
clf10 = DecisionTreeRegressor(ccp_alpha= 0.0,
                             criterion= 'mse',
                             max_depth= 5,
                             max_features= 'auto',
                             max_leaf_nodes= 30,
                             min_impurity_decrease= 0.0,
                             min_impurity_split= None,
                             min_samples_leaf= 2,
                             min_samples_split= 100,
                             min_weight_fraction_leaf= 0.0,
                             presort= 'deprecated',
                             random_state= SEED,
                             splitter= 'best')

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4004772298.py in <cell line: 0>()
----> 1 clf10 = DecisionTreeRegressor(ccp_alpha= 0.0,
      2                              criterion= 'mse',
      3                              max_depth= 5,
      4                              max_features= 'auto',
      5                              max_leaf_nodes= 30,

TypeError: DecisionTreeRegressor.__init__() got an unexpected keyword argument 'min_impurity_split'

## === cell 44
clf11 = GradientBoostingRegressor()

## === cell 46
SCF = StackingClassifier(classifiers=[clf1, clf5], 
                         meta_classifier=clf2,
                         use_probas=True,
                         average_probas=True)

## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2027236434.py in <cell line: 0>()
----> 1 SCF = StackingClassifier(classifiers=[clf1, clf5], 
      2                          meta_classifier=clf2,
      3                          use_probas=True,
      4                          average_probas=True)

NameError: name 'clf5' is not defined

## === cell 47
if test_pipeline:
    pipelines = []
    pipelines.append(('LGBMClassifier', Pipeline([('LGBMClassifier',clf1)])))
    pipelines.append(('LogisticRegression', Pipeline([('LogisticRegression',clf2)])))
    pipelines.append(('XGBRegressor', Pipeline([('XGBRegressor',clf3)])))
    pipelines.append(('GaussianNB', Pipeline([('GaussianNB',clf4)])))
    pipelines.append(('RandomForestClassifier', Pipeline([('RandomForestClassifier',clf5)])))
    pipelines.append(('KNeighborsRegressor', Pipeline([('KNeighborsRegressor', clf9)])))
    pipelines.append(('DecisionTreeRegressor', Pipeline([('DecisionTreeRegressor', clf10)])))
    pipelines.append(('GradientBoostingRegressor', Pipeline([('GradientBoostingRegressor', clf11)])))
    pipelines.append(('StackingClassifier', Pipeline([('StackingClassifier', SCF)])))

    X_train = train_coded.drop('target',axis=1).iloc[:,1:]
    y_train = train_coded['target']

    M_name = []
    M_auc_score = []
    for name, model in pipelines:
        cv_results = cross_val_score(model, X_train, y_train, cv=FOLDS, scoring='roc_auc')
        M_auc_score.append(np.mean(cv_results))
        M_name.append(name)
        print("%s: %f +/- %f" % (name, cv_results.mean(), cv_results.std()))

## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3626445933.py in <cell line: 0>()
      5     pipelines.append(('XGBRegressor', Pipeline([('XGBRegressor',clf3)])))
      6     pipelines.append(('GaussianNB', Pipeline([('GaussianNB',clf4)])))
----> 7     pipelines.append(('RandomForestClassifier', Pipeline([('RandomForestClassifier',clf5)])))
      8     # pipelines.append(('LinearRegression', Pipeline([('LinearRegression',clf6)])))
      9     # pipelines.append(('Lasso', Pipeline([('Lasso', clf7)])))

NameError: name 'clf5' is not defined

## === cell 48
if test_pipeline:
    fig, ax = plt.subplots(figsize=(7,7))

    bars = ax.bar(
        x=M_name,
        height=M_auc_score,
        tick_label=M_name
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    bar_color = bars[0].get_facecolor()
    plt.xticks(rotation=90)

    for bar in bars:
        ax.text(
          bar.get_x() + bar.get_width() / 2,
          bar.get_height() + 0.05,
          round(bar.get_height(), 4),
          horizontalalignment='center',
          color=bar_color,
          weight='bold'
        )

    fig.tight_layout()

## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/868544373.py in <cell line: 0>()
      4     # Save the chart so we can loop through the bars below.
      5     bars = ax.bar(
----> 6         x=M_name,
      7         height=M_auc_score,
      8         tick_label=M_name

NameError: name 'M_name' is not defined

## === cell 50
select_classifier = clf2
select_classifier.get_params()

## === cell 51
if Setup_Parameters:

    X_train = train_coded.drop('target',axis=1).iloc[:,1:]
    y_train = train_coded['target']

    params = dict(
        C = [0.001, 0.01, 0.1, 1, 10],
        max_iter = [100,150,50]
    )

    grid = GridSearchCV(estimator=select_classifier,
                        param_grid=params, 
                        cv=FOLDS,
                        scoring='roc_auc',
                        refit='AUC',
                        n_jobs = -1)

    grid.fit(X_train, y_train)

    for r, _ in enumerate(grid.cv_results_['mean_test_score']):
        print("AUC : %0.5f +/- %0.5f %r"
              % (grid.cv_results_['mean_test_score'][r],
                 grid.cv_results_['std_test_score'][r] / 2.0,
                 grid.cv_results_['params'][r]))

    print(f'\nBest parameters: {grid.best_params_}')

    oof, submission, auc_score, model_name = crossValidate(grid,X=train_coded,X_test=test_coded,FOLDS=FOLDS,SEED=SEED,show_roc_curve=True)

## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2414628387.py in <cell line: 0>()
     26     print(f'\nBest parameters: {grid.best_params_}')
     27 
---> 28     oof, submission, auc_score, model_name = crossValidate(grid,X=train_coded,X_test=test_coded,FOLDS=FOLDS,SEED=SEED,show_roc_curve=True)

/tmp/ipykernel_11/1144636025.py in crossValidate(CLF, X, X_test, FOLDS, SEED, show_roc_curve, pesudo_label)
     20     for fold,(idxT,idxV) in enumerate(skf.split(np.arange(15))):
     21 
---> 22         idxT, idxV = X.tfrecord.isin(idxT), X.tfrecord.isin(idxV)
     23 
     24         X_train_main, y_train_main = X[idxT], X[idxT]

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'tfrecord'

## === cell 53
M_name = []
M_auc_score = []
for CLF in [clf1,clf2,clf4,clf5,clf10,clf11,SCF]:
    if pesudo_label:
        oof, submission, auc_score, model_name, \
        pesudo_oof, pesudo_submission, pesudo_auc_score, pesudo_model_name = crossValidate(CLF,X=train_coded,
                                                                                           X_test=test_coded,
                                                                                           FOLDS=FOLDS,SEED=SEED,
                                                                                           show_roc_curve=True,
                                                                                           pesudo_label=True)
    else:
        oof, submission, auc_score, model_name = crossValidate(CLF,X=train_coded,X_test=test_coded,FOLDS=FOLDS,SEED=SEED,show_roc_curve=True,pesudo_label=False)
    
    oof.to_csv(f'oof_{model_name}_{auc_score:0.4f}.csv',index=False)
    submission.to_csv(f'submission_{model_name}_{auc_score:0.4f}.csv',index=False)
    M_name.append(model_name)
    M_auc_score.append(auc_score)
    
    if pesudo_label:
        pesudo_oof.to_csv(f'Pesudo_oof_{model_name}_{pesudo_auc_score:0.4f}.csv',index=False)
        pesudo_submission.to_csv(f'Pesudo_submission_{model_name}_{pesudo_auc_score:0.4f}.csv',index=False)
        M_name.append(pesudo_model_name)
        M_auc_score.append(pesudo_auc_score)

## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2148711841.py in <cell line: 0>()
      1 M_name = []
      2 M_auc_score = []
----> 3 for CLF in [clf1,clf2,clf4,clf5,clf10,clf11,SCF]:
      4     if pesudo_label:
      5         oof, submission, auc_score, model_name, \

NameError: name 'clf5' is not defined

## === cell 55
fig, ax = plt.subplots(figsize=(15,7))

bars = ax.bar(
    x=M_name,
    height=M_auc_score,
    tick_label=M_name
)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)

bar_color = bars[0].get_facecolor()
plt.xticks(rotation=90)

for bar in bars:
    ax.text(
      bar.get_x() + bar.get_width() / 2,
      bar.get_height() + 0.05,
      round(bar.get_height(), 5),
      horizontalalignment='center',
      color=bar_color,
      weight='bold'
    )

fig.tight_layout()

## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/1035584827.py in <cell line: 0>()
     18 ax.xaxis.grid(False)
     19 
---> 20 bar_color = bars[0].get_facecolor()
     21 plt.xticks(rotation=90)
     22 

IndexError: tuple index out of range
