# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.11

# 2. Installed packages

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
sklearn-pandas==2.2.0

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (129 lines)
            sample_submission.csv (1857 lines)
            sample_submission.csv.zip (21.4 kB)
            test.zip (26.9 GB)
            train.zip (270.6 GB)
            train_metadata.json (1 lines)
            validation.zip (27.0 GB)
            validation_metadata.json (1 lines)
            google-research-identify-contrails-reduce-global-warming/
                description.md (129 lines)
                sample_submission.csv (1857 lines)
                ... and 6 other files
                google-research-identify-contrails-reduce-global-warming/
                test/
                    1006714073984511039/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 7 other files
                    1011991214639847439/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 7 other files
                    ... and 1855 other folders
                train/
                    1000216489776414077/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 9 other files
                    1000603527582775543/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 9 other files
                    ... and 18672 other folders
                validation/
                    1000834164244036115/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 8 other files
                    1002653297254493116/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 8 other files
                    ... and 1854 other folders
            test/
                1006714073984511039/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 7 other files
                1011991214639847439/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 7 other files
                ... and 1855 other folders
            train/
                1000216489776414077/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 9 other files
                1000603527582775543/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 9 other files
                ... and 18672 other folders
            validation/
                1000834164244036115/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 8 other files
                1002653297254493116/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 8 other files
                ... and 1854 other folders
        input/
            description.md (129 lines)
            sample_submission.csv (1857 lines)
            sample_submission.csv.zip (21.4 kB)
            test.zip (26.9 GB)
            train.zip (270.6 GB)
            train_metadata.json (1 lines)
            validation.zip (27.0 GB)
            validation_metadata.json (1 lines)
            google-research-identify-contrails-reduce-global-warming/
                description.md (129 lines)
                sample_submission.csv (1857 lines)
                ... and 6 other files
                google-research-identify-contrails-reduce-global-warming/
                test/
                    1006714073984511039/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 7 other files
                    1011991214639847439/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 7 other files
                    ... and 1855 other folders
                train/
                    1000216489776414077/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 9 other files
                    1000603527582775543/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 9 other files
                    ... and 18672 other folders
                validation/
                    1000834164244036115/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 8 other files
                    1002653297254493116/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 8 other files
                    ... and 1854 other folders
            test/
                1006714073984511039/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 7 other files
                1011991214639847439/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 7 other files
                ... and 1855 other folders
            train/
                1000216489776414077/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 9 other files
                1000603527582775543/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 9 other files
                ... and 18672 other folders
            validation/
                1000834164244036115/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 8 other files
                1002653297254493116/
                    band_08.npy (2.1 MB)
                    band_09.npy (2.1 MB)
                    ... and 8 other files
                ... and 1854 other folders
        working/
            google-research-identify-contrails-reduce-global-warming/
                description.md (129 lines)
                sample_submission.csv (1857 lines)
                ... and 6 other files
                google-research-identify-contrails-reduce-global-warming/
                test/
                    1006714073984511039/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 7 other files
                    1011991214639847439/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 7 other files
                    ... and 1855 other folders
                train/
                    1000216489776414077/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 9 other files
                    1000603527582775543/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 9 other files
                    ... and 18672 other folders
                validation/
                    1000834164244036115/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 8 other files
                    1002653297254493116/
                        band_08.npy (2.1 MB)
                        band_09.npy (2.1 MB)
                        ... and 8 other files
                    ... and 1854 other folders
```

-> data/google-research-identify-contrails-reduce-global-warming/sample_submission.csv has 1856 rows and 4 columns.
The columns are: record_id, encoded_pixels, height, width

-> data/google-research-identify-contrails-reduce-global-warming/train_metadata.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "record_id": {
        "type": "string"
      },
      "projection_wkt": {
        "type": "string"
      },
      "row_min": {
        "type": "number"
      },
      "row_size": {
        "type": "number"
      },
      "col_min": {
        "type": "number"
      },
      "col_size": {
        "type": "number"
      },
      "timestamp": {
        "type": "number"
      }
    },
    "required": [
      "col_min",
      "col_size",
      "projection_wkt",
      "record_id",
      "row_min",
      "row_size",
      "timestamp"
    ]
  }
}

-> data/google-research-identify-contrails-reduce-global-warming/validation_metadata.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "record_id": {
        "type": "string"
      },
      "projection_wkt": {
        "type": "string"
      },
      "row_min": {
        "type": "number"
      },
      "row_size": {
        "type": "number"
      },
      "col_min": {
        "type": "number"
      },
      "col_size": {
        "type": "number"
      },
      "timestamp": {
        "type": "number"
      }
    },
    "required": [
      "col_min",
      "col_size",
      "projection_wkt",
      "record_id",
      "row_min",
      "row_size",
      "timestamp"
    ]
  }
}

-> data/sample_submission.csv has 1856 rows and 4 columns.
The columns are: record_id, encoded_pixels, height, width

-> data/train_metadata.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "record_id": {
        "type": "string"
      },
      "projection_wkt": {
        "type": "string"
      },
      "row_min": {
        "type": "number"
      },
      "row_size": {
        "type": "number"
      },
      "col_min": {
        "type": "number"
      },
      "col_size": {
        "type": "number"
      },
      "timestamp": {
        "type": "number"
      }
    },
    "required": [
      "col_min",
      "col_size",
      "projection_wkt",
      "record_id",
      "row_min",
      "row_size",
      "timestamp"
    ]
  }
}

-> data/validation_metadata.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "record_id": {
        "type": "string"
      },
      "projection_wkt": {
        "type": "string"
      },
      "row_min": {
        "type": "number"
      },
      "row_size": {
        "type": "number"
      },
      "col_min": {
        "type": "number"
      },
      "col_size": {
        "type": "number"
      },
      "timestamp": {
        "type": "number"
      }
    },
    "required": [
      "col_min",
      "col_size",
      "projection_wkt",
      "record_id",
      "row_min",
      "row_size",
      "timestamp"
    ]
  }
}

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path
data_path = Path('/kaggle/input/google-climate-and-energy-contrails-detection')


## === cell 1
def rle_encode(x, fg_val=1):
    """
        Returns run length as list
    Args:
        x:  numpy array of shape (height, width), 1 - mask, 0 - background
        fg_val
    Returns:
    """
    dots = np.where(
        x.T.flatten() == fg_val)[0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def rle_decode(mask_rle, shape=(256, 256)):
    '''
    mask_rle: run-length as string formatted (start length)
    shape: (height, width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')  # Needed to align to RLE direction

def list_to_string(x):
    return str(x).replace("[", "").replace("]", "").replace(",", "")


## === cell 2
test_recs = os.listdir(data_path / 'test')
print(test_recs)


## --- ERROR in cell 2, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3784617818.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mtest_recs[0m [0;34m=[0m [0mos[0m[0;34m.[0m[0mlistdir[0m[0;34m([0m[0mdata_path[0m [0;34m/[0m [0;34m'test'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mprint[0m[0;34m([0m[0mtest_recs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '/kaggle/input/google-climate-and-energy-contrails-detection/test'

## === cell 3
n = 1000
band_08 = np.load(data_path / 'test' / test_recs[0] / 'band_08.npy').sum(axis=2)
preds = np.c_[np.unravel_index(np.argpartition(band_08.ravel(),-n)[-n:],band_08.shape)]
mask = np.zeros((256, 266))
mask[preds[:, 0], preds[:, 1]] = 1

plt.imshow(mask, cmap='Greys')
plt.title("Obviously Not Contrails", fontsize='16')
plt.show()
