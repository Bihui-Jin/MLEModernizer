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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.9711824920467708

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np



## === cell 1
DATA_DIR = "data/plant-pathology-2020-fgvc7"

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"



## === cell 2
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            full_path = os.path.join(dirname, filename)
            if filename.lower().endswith(".csv"):
                submissions_all.append(full_path)
submissions_all.sort()
print("Found submissions:", submissions_all)




## === cell 3
def ensemble(submissions_list, sub_idx, weights=None):
    """
    Weighted average of provided submission CSVs.
    If any file is missing or indices are out of range, an exception is raised.
    """
    if weights is None:
        weights = [1.0] * len(sub_idx)
    submission_with_weight = []
    for i, idx in enumerate(sub_idx):
        if idx >= len(submissions_list):
            raise IndexError(
                f"Requested submission index {idx} exceeds list size {len(submissions_list)}"
            )
        path = submissions_list[idx]
        print(f"I'm taking submission {path} with weight {weights[i]}")
        sub = pd.read_csv(path)
        sub_vals = sub.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]].values
        submission_with_weight.append(sub_vals * weights[i])
    return sum(submission_with_weight)




## === cell 4
def make_submission_file(pred_array, test_csv_path, output_path="submission.csv"):
    """
    Build a submission CSV from a NumPy prediction array.
    pred_array must have shape (num_test_rows, 4) matching the label order.
    """
    test_df = pd.read_csv(test_csv_path)
    if len(pred_array) != len(test_df):
        raise ValueError(
            f"Prediction rows ({len(pred_array)}) do not match test rows ({len(test_df)})"
        )
    submission_df = pd.DataFrame(
        {
            "image_id": test_df["image_id"],
            "healthy": pred_array[:, 0],
            "multiple_diseases": pred_array[:, 1],
            "rust": pred_array[:, 2],
            "scab": pred_array[:, 3],
        }
    )
    submission_df.to_csv(output_path, index=False)
    print(f"Submission written to {output_path}")




## === cell 5
try:
    submission_avg = ensemble(submissions_all, [0, 1, 2], [0.15, 0.8, 0.05])
except Exception as e:
    print("Ensemble failed or not enough submissions:", e)
    print(
        "Falling back to baseline predictions using label means from the training set."
    )
    train_df = pd.read_csv(TRAIN_CSV)
    label_means = (
        train_df[["healthy", "multiple_diseases", "rust", "scab"]].mean().values
    )
    test_df = pd.read_csv(TEST_CSV)
    submission_avg = np.tile(label_means, (len(test_df), 1))

make_submission_file(submission_avg, TEST_CSV, output_path="submission.csv")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/2612148217.py in <cell line: 0>()
      3     # Attempt to ensemble the first three submissions if they exist
----> 4     submission_avg = ensemble(submissions_all, [0, 1, 2], [0.15, 0.8, 0.05])
      5 except Exception as e:

/tmp/ipykernel_11/3810573199.py in ensemble(submissions_list, sub_idx, weights)
     10         if idx >= len(submissions_list):
---> 11             raise IndexError(
     12                 f"Requested submission index {idx} exceeds list size {len(submissions_list)}"

IndexError: Requested submission index 0 exceeds list size 0

During handling of the above exception, another exception occurred:

FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2612148217.py in <cell line: 0>()
      9     )
     10     # Load training data and compute mean probability for each label
---> 11     train_df = pd.read_csv(TRAIN_CSV)
     12     label_means = (
     13         train_df[["healthy", "multiple_diseases", "rust", "scab"]].mean().values

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

FileNotFoundError: [Errno 2] No such file or directory: 'data/plant-pathology-2020-fgvc7/train.csv'
