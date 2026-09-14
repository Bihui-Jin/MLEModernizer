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
Predict a patient’s severity of decline in lung function based on a CT scan of their lungs. Lung function is assessed based on output from a spirometer, which measures the forced vital capacity (`FVC`), i.e. the volume of air exhaled.

## Metric
A modified version of the Laplace Log Likelihood. 

For each true FVC measurement, you will predict both an FVC and a confidence measure (standard deviation 𝜎𝜎). The metric is computed as:

$$
\begin{gathered}
\sigma_{\text {clipped }}=\max (\sigma, 70), \\
\Delta=\min \left(\left|F V C_{\text {true }}-F V C_{\text {predicted }}\right|, 1000\right), \\
\text { metric }=-\frac{\sqrt{2} \Delta}{\sigma_{\text {clipped }}}-\ln \left(\sqrt{2} \sigma_{\text {clipped }}\right) .
\end{gathered}
$$

The error is thresholded at 1000 ml to avoid large errors adversely penalizing results, while the confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty in FVC. The final score is calculated by averaging the metric across all test set `Patient_Week`s (three per patient). 

Metric values will be negative and higher is better.

## Submission Format
For each `Patient_Week`, you must predict the `FVC` and a confidence. You are asked to predict every patient's `FVC` measurement for every possible week. Those weeks which are not in the final three visits are ignored in scoring.

The file should contain a header and have the following format:

```
Patient_Week,FVC,Confidence
ID00002637202176704235138_1,2000,100
ID00002637202176704235138_2,2000,100
ID00002637202176704235138_3,2000,100
etc.

```

## Dataset
In the dataset, you are provided with a baseline chest CT scan and associated clinical information for a set of patients. A patient has an image acquired at time `Week = 0` and has numerous follow up visits over the course of approximately 1-2 years, at which time their `FVC` is measured.

- In the training set, you are provided with an anonymized, baseline CT scan and the entire history of FVC measurements.
- In the test set, you are provided with a baseline CT scan and only the initial FVC measurement. **You are asked to predict the final three `FVC` measurements for each patient, as well as a confidence value in your prediction.**

- **train.csv** - the training set, contains full history of clinical information
- **test.csv** - the test set, contains only the baseline measurement
- **train/** - contains the training patients' baseline CT scan in DICOM format
- **test/** - contains the test patients' baseline CT scan in DICOM format
- **sample_submission.csv** - demonstrates the submission format

**train.csv and test.csv**

- `Patient`a unique Id for each patient (also the name of the patient's DICOM folder)
- `Weeks`the relative number of weeks pre/post the baseline CT (may be negative)
- `FVC` - the recorded lung capacity in ml
- `Percent`a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics
- `Age`
- `Sex`
- `SmokingStatus`

**sample submission.csv**

- `Patient_Week` - a unique Id formed by concatenating the `Patient` and `Weeks` columns (i.e. ABC_22 is a prediction for patient ABC at week 22)
- `FVC` - the predicted FVC in ml
- `Confidence` - a confidence value of your prediction (also has units of ml)

# 2. Python version

3.8

# 3. Installed packages

category_encoders==2.7.0
geopandas==0.14.4
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pydicom==3.0.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        input/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        working/
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
```

-> data/osic-pulmonary-fibrosis-progression/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/osic-pulmonary-fibrosis-progression/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/osic-pulmonary-fibrosis-progression/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> (stopped after 10 files for performance)

# 5. Target score

-6.8591

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import category_encoders as ce
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import typing as tp


def seed_everything(seed: int = 42) -> None:
    """Set seeds for reproducibility across random, numpy, and torch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


OUTPUT_DICT = "./"
INPUT_ROOT = os.path.join(os.getcwd(), "data", "osic-pulmonary-fibrosis-progression")
ID = "Patient_Week"
TARGET = "virtual_FVC"
SEED = 42
VIRTUAL_BASE_FVC = 2000
seed_everything(seed=SEED)
N_FOLD = 4



## === cell 1
train_path = os.path.join(INPUT_ROOT, "train.csv")
train = pd.read_csv(train_path)
train[ID] = train["Patient"].astype(str) + "_" + train["Weeks"].astype(str)
print(train.shape)
train.head()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2056646269.py in <cell line: 0>()
      1 train_path = os.path.join(INPUT_ROOT, "train.csv")
----> 2 train = pd.read_csv(train_path)
      3 train[ID] = train["Patient"].astype(str) + "_" + train["Weeks"].astype(str)
      4 print(train.shape)
      5 train.head()

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/data/osic-pulmonary-fibrosis-progression/train.csv'

## === cell 2
output = pd.DataFrame()
gb = train.groupby("Patient")
tk0 = tqdm(gb, total=len(gb))
for _, usr_df in tk0:
    usr_output = pd.DataFrame()
    for week, tmp in usr_df.groupby("Weeks"):
        rename_cols = {
            "Weeks": "base_Week",
            "FVC": "base_FVC",
            "Percent": "base_Percent",
            "Age": "base_Age",
        }
        tmp = tmp.drop(columns="Patient_Week").rename(columns=rename_cols)
        drop_cols = ["Age", "Sex", "SmokingStatus", "Percent"]
        _usr_output = (
            usr_df.drop(columns=drop_cols)
            .rename(columns={"Weeks": "predict_Week"})
            .merge(tmp, on="Patient")
        )
        _usr_output["Week_passed"] = (
            _usr_output["predict_Week"] - _usr_output["base_Week"]
        )
        usr_output = pd.concat([usr_output, _usr_output], ignore_index=True)
    output = pd.concat([output, usr_output], ignore_index=True)

train = output[output["Week_passed"] != 0].reset_index(drop=True)
print(train.shape)
train.head()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/322670910.py in <cell line: 0>()
      1 output = pd.DataFrame()
----> 2 gb = train.groupby("Patient")
      3 tk0 = tqdm(gb, total=len(gb))
      4 for _, usr_df in tk0:
      5     usr_output = pd.DataFrame()

NameError: name 'train' is not defined

## === cell 3
train["virtual_FVC"] = train["FVC"] / train["base_FVC"] * VIRTUAL_BASE_FVC



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1336593402.py in <cell line: 0>()
----> 1 train["virtual_FVC"] = train["FVC"] / train["base_FVC"] * VIRTUAL_BASE_FVC
      2 

NameError: name 'train' is not defined

## === cell 4
test_path = os.path.join(INPUT_ROOT, "test.csv")
submission_path = os.path.join(INPUT_ROOT, "sample_submission.csv")

test_raw = pd.read_csv(test_path)
submission = pd.read_csv(submission_path)

test = test_raw.rename(
    columns={
        "Weeks": "base_Week",
        "FVC": "base_FVC",
        "Percent": "base_Percent",
        "Age": "base_Age",
    }
)

submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["predict_Week"] = (
    submission["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
)

test = submission.drop(columns=["FVC", "Confidence"]).merge(test, on="Patient")
test["Week_passed"] = test["predict_Week"] - test["base_Week"]
print(test.shape)
test.head()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1297338111.py in <cell line: 0>()
      2 submission_path = os.path.join(INPUT_ROOT, "sample_submission.csv")
      3 
----> 4 test_raw = pd.read_csv(test_path)
      5 submission = pd.read_csv(submission_path)
      6 

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/data/osic-pulmonary-fibrosis-progression/test.csv'

## === cell 5
test["FVC"] = np.nan
test["virtual_FVC"] = np.nan



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2573219536.py in <cell line: 0>()
----> 1 test["FVC"] = np.nan
      2 test["virtual_FVC"] = np.nan
      3 

NameError: name 'test' is not defined

## === cell 6
train.head()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2745801949.py in <cell line: 0>()
----> 1 train.head()
      2 

NameError: name 'train' is not defined

## === cell 7
test.head()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3214727096.py in <cell line: 0>()
----> 1 test.head()
      2 

NameError: name 'test' is not defined

## === cell 8
submission = pd.read_csv(submission_path)
print(submission.shape)
submission.head()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/4064857978.py in <cell line: 0>()
----> 1 submission = pd.read_csv(submission_path)
      2 print(submission.shape)
      3 submission.head()
      4 

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/data/osic-pulmonary-fibrosis-progression/sample_submission.csv'

## === cell 9
folds = train[[ID, "Patient", TARGET]].copy()
Fold = GroupKFold(n_splits=N_FOLD)
groups = folds["Patient"].values
folds["fold"] = -1
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[TARGET], groups)):
    folds.loc[val_index, "fold"] = int(n)
folds["fold"] = folds["fold"].astype(int)
folds.head()




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3242670184.py in <cell line: 0>()
----> 1 folds = train[[ID, "Patient", TARGET]].copy()
      2 Fold = GroupKFold(n_splits=N_FOLD)
      3 groups = folds["Patient"].values
      4 folds["fold"] = -1
      5 for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[TARGET], groups)):

NameError: name 'train' is not defined

## === cell 10
class OSICLossForLGBM:
    """
    Custom Loss for LightGBM.

    * Objective: return grad & hess of NLL of gaussian
    * Evaluation: return competition metric
    """

    def __init__(self, epsilon: float = 1e-09) -> None:
        """Initialize."""
        self.name = "osic_loss"
        self.n_class = 2  # FVC & Confidence
        self.epsilon = epsilon

    def __call__(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> float:
        """Calc loss."""
        mu = preds[:, 0]
        sigma = preds[:, 1]
        sigma_t = np.log(1 + np.exp(sigma))
        loss_by_sample = ((labels - mu) / sigma_t) ** 2 / 2 + np.log(
            np.sqrt(2 * np.pi) * sigma_t
        )
        loss = np.average(loss_by_sample, weight)

        return loss

    def _calc_grad_and_hess(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        """Calc Grad and Hess"""
        mu = preds[:, 0]
        sigma = preds[:, 1]

        sigma_t = np.log(1 + np.exp(sigma))
        grad_sigma_t = 1 / (1 + np.exp(-sigma))
        hess_sigma_t = grad_sigma_t * (1 - grad_sigma_t)

        grad = np.zeros_like(preds)
        hess = np.zeros_like(preds)
        grad[:, 0] = -(labels - mu) / sigma_t**2
        hess[:, 0] = 1 / sigma_t**2

        tmp = ((labels - mu) / sigma_t) ** 2
        grad[:, 1] = 1 / sigma_t * (1 - tmp) * grad_sigma_t
        hess[:, 1] = (
            -1 / sigma_t**2 * (1 - 3 * tmp) * grad_sigma_t**2
            + 1 / sigma_t * (1 - tmp) * hess_sigma_t
        )
        if weight is not None:
            grad = grad * weight[:, None]
            hess = hess * weight[:, None]
        return grad, hess

    def return_loss(
        self, preds: np.ndarray, data: lgb.Dataset
    ) -> tp.Tuple[str, float, bool]:
        """Return Loss for lightgbm"""
        labels = data.get_label()
        weight = data.get_weight()
        preds = preds.reshape(self.n_class, len(labels)).T
        loss = self(preds, labels, weight)

        return self.name, loss, False

    def return_grad_and_hess(
        self, preds: np.ndarray, data: lgb.Dataset
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        """Return Grad and Hess for lightgbm"""
        labels = data.get_label()
        weight = data.get_weight()
        preds = preds.reshape(self.n_class, len(labels)).T
        grad, hess = self._calc_grad_and_hess(preds, labels, weight)

        grad = grad.T.reshape(len(labels) * self.n_class)
        hess = hess.T.reshape(len(labels) * self.n_class)

        return grad, hess

    def calc_comp_metric(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        weight: tp.Optional[np.ndarray] = None,
    ) -> float:
        """Calc competition metric."""
        sigma_clip = np.maximum(preds[:, 1], 70)
        Delta = np.minimum(np.abs(preds[:, 0] - labels), 1000)
        loss_by_sample = -np.sqrt(2) * Delta / sigma_clip - np.log(
            np.sqrt(2) * sigma_clip
        )
        loss = np.average(loss_by_sample, weight)

        return loss




## === cell 11
cat_features = ["Sex", "SmokingStatus"]
num_features = [
    c for c in test.columns if (test.dtypes[c] != "object") and (c not in cat_features)
]
features = num_features + cat_features
drop_features = [ID, TARGET, "predict_Week", "base_Week", "FVC", "base_FVC"]
features = [c for c in features if c not in drop_features]

if cat_features:
    ce_oe = ce.OrdinalEncoder(cols=cat_features, handle_unknown="impute")
    ce_oe.fit(train)
    train = ce_oe.transform(train)
    test = ce_oe.transform(test)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1012353471.py in <cell line: 0>()
      1 cat_features = ["Sex", "SmokingStatus"]
      2 num_features = [
----> 3     c for c in test.columns if (test.dtypes[c] != "object") and (c not in cat_features)
      4 ]
      5 features = num_features + cat_features

NameError: name 'test' is not defined

## === cell 12
oof_pred = np.zeros(len(train))
test_pred = np.zeros(len(test))

for fold in range(N_FOLD):
    tr_idx = folds[folds.fold != fold].index
    val_idx = folds[folds.fold == fold].index

    model = lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.1,
        n_estimators=2000,
        max_depth=1,
        reg_lambda=5e-03,
        random_state=SEED,
        n_jobs=5,
    )

    model.fit(
        train.loc[tr_idx, features],
        train.loc[tr_idx, TARGET],
        eval_set=[(train.loc[val_idx, features], train.loc[val_idx, TARGET])],
        eval_metric="l2",
        early_stopping_rounds=50,
        verbose=False,
    )

    oof_pred[val_idx] = model.predict(train.loc[val_idx, features])
    test_pred += model.predict(test[features]) / N_FOLD

oof = np.column_stack([oof_pred, np.full_like(oof_pred, 100.0)])
predictions = np.column_stack([test_pred, np.full_like(test_pred, 100.0)])



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3646137569.py in <cell line: 0>()
----> 1 oof_pred = np.zeros(len(train))
      2 test_pred = np.zeros(len(test))
      3 
      4 for fold in range(N_FOLD):
      5     tr_idx = folds[folds.fold != fold].index

NameError: name 'train' is not defined

## === cell 13
oof = oof / VIRTUAL_BASE_FVC * train["base_FVC"].values[:, None]
predictions = predictions / VIRTUAL_BASE_FVC * test["base_FVC"].values[:, None]



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/902366294.py in <cell line: 0>()
----> 1 oof = oof / VIRTUAL_BASE_FVC * train["base_FVC"].values[:, None]
      2 predictions = predictions / VIRTUAL_BASE_FVC * test["base_FVC"].values[:, None]
      3 

NameError: name 'oof' is not defined

## === cell 14
print("Training metric:", OSICLossForLGBM().calc_comp_metric(oof, train["FVC"].values))



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3229913472.py in <cell line: 0>()
----> 1 print("Training metric:", OSICLossForLGBM().calc_comp_metric(oof, train["FVC"].values))
      2 

NameError: name 'oof' is not defined

## === cell 15
train["FVC_pred"] = oof[:, 0]
train["Confidence"] = oof[:, 1]
test["FVC_pred"] = predictions[:, 0]
test["Confidence"] = predictions[:, 1]



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1873625654.py in <cell line: 0>()
----> 1 train["FVC_pred"] = oof[:, 0]
      2 train["Confidence"] = oof[:, 1]
      3 test["FVC_pred"] = predictions[:, 0]
      4 test["Confidence"] = predictions[:, 1]
      5 

NameError: name 'oof' is not defined

## === cell 16
sub = submission.drop(columns=["FVC", "Confidence"]).merge(
    test[["Patient_Week", "FVC_pred", "Confidence"]], on="Patient_Week"
)
sub.columns = submission.columns
sub.to_csv("submission.csv", index=False)
sub.head()

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3064380265.py in <cell line: 0>()
----> 1 sub = submission.drop(columns=["FVC", "Confidence"]).merge(
      2     test[["Patient_Week", "FVC_pred", "Confidence"]], on="Patient_Week"
      3 )
      4 sub.columns = submission.columns
      5 sub.to_csv("submission.csv", index=False)

NameError: name 'submission' is not defined
