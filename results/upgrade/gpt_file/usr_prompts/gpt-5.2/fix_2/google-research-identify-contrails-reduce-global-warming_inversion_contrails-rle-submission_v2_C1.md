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
Identify contrails in satellite imagery.

## Metric
Global Dice coefficient. The Dice coefficient formula is given by:

$$
\frac{2 \cdot |X \cap Y|}{|X| + |Y|}
$$

where X is the entire set of predicted contrail pixels for all observations in the test data and Y is the ground truth set of all contrail pixels in the test data.

## Submission Format
Use run-length encoding. For example, '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the mask should be binary, meaning the masks for all objects in an image are joined into a single large mask. A value of 0 should indicate pixels that are not masked, and a value of 1 will indicate pixels that are masked.

Use a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

Empty predictions must be marked with '-' in the submission file.

The file should contain a header and have the following format:

```
record_id,encoded_pixels  
1000834164244036115,1 1 5 1  
1002653297254493116,-  
etc.
```

## Dataset 
Some key labeling guidance:
- Contrails must contain at least 10 pixels
- At some time in their life, Contrails must be at least 3x longer than they are wide
- Contrails must either appear suddenly or enter from the sides of the image
- Contrails should be visible in at least two image

A sequence of images at 10-minute intervals are provided. Each example (`record_id`) contains exactly one labeled frame.

- **train/** - the training set; each folder represents a `record_id` and contains the following data:
    - **band_{08-16}.npy**: array with size of `H x W x T`, where `T = n_times_before + n_times_after + 1`, representing the number of images in the sequence. There are `n_times_before` and `n_times_after` images before and after the labeled frame respectively. In our dataset all examples have `n_times_before=4` and `n_times_after=3`. Each band represents an infrared channel at different wavelengths and is converted to brightness temperatures based on the calibration parameters. The number in the filename corresponds to the GOES-16 ABI band number. Details of the ABI bands can be found [here](https://www.goes-r.gov/mission/ABI-bands-quick-info.html).
    - **human_individual_masks.npy**: array with size of `H x W x 1 x R`. Each example is labeled by `R` individual human labelers. `R` is not the same for all samples. The labeled masks have value either 0 or 1 and correspond to the `(n_times_before+1)`-th image in `band_{08-16}.npy`. They are available only in the training set.
    - **human_pixel_masks.npy**: array with size of `H x W x 1` containing the binary ground truth. A pixel is regarded as contrail pixel in evaluation if it is labeled as contrail by more than half of the labelers.
- **validation/** - the same as the training set, without the individual label annotations; it is permitted to use this as training data if desired
- **test/** - the test set; your objective is to identify contrails found in these records.
- **{train|validation}_metadata.json** - metadata information for each record; contains the timestamps and the projection parameters to reproduce the satellite images.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

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
sklearn-pandas==2.2.0

# 4. Data file paths

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

# 5. Target score

0.00041

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path

data_path = Path(
    "/kaggle/input/google-research-identify-contrails-reduce-global-warming"
)

if (data_path / "google-research-identify-contrails-reduce-global-warming").exists():
    data_path = data_path / "google-research-identify-contrails-reduce-global-warming"

print("Using data_path:", data_path)
print("Exists:", data_path.exists())




## === cell 1
def rle_encode(x, fg_val=1):
    """
    Returns run length as list
    Args:
        x: numpy array of shape (height, width), 1 - mask, 0 - background
        fg_val: foreground value
    Returns:
        list of ints [start_1, len_1, start_2, len_2, ...]
    """
    dots = np.where(x.T.flatten() == fg_val)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def rle_decode(mask_rle, shape=(256, 256)):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def list_to_string(x):
    if x is None or len(x) == 0:
        return "-"
    return str(x).replace("[", "").replace("]", "").replace(",", "")




## === cell 2
test_dir = data_path / "test"
if not test_dir.exists():
    raise FileNotFoundError(
        f"Expected test directory at {test_dir}, but it does not exist."
    )

test_recs = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
print("n_test_records:", len(test_recs))
print("first 5:", test_recs[:5])



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3691285512.py in <cell line: 0>()
      2 test_dir = data_path / "test"
      3 if not test_dir.exists():
----> 4     raise FileNotFoundError(
      5         f"Expected test directory at {test_dir}, but it does not exist."
      6     )

FileNotFoundError: Expected test directory at /kaggle/input/google-research-identify-contrails-reduce-global-warming/google-research-identify-contrails-reduce-global-warming/test, but it does not exist.

## === cell 3
n = 1000

sample_band = np.load(test_dir / test_recs[0] / "band_08.npy")  # H x W x T
H, W, T = sample_band.shape
print("Sample band_08 shape:", sample_band.shape)

band_08 = sample_band.sum(axis=2)
preds = np.c_[
    np.unravel_index(np.argpartition(band_08.ravel(), -n)[-n:], band_08.shape)
]
mask = np.zeros((H, W), dtype=np.uint8)
mask[preds[:, 0], preds[:, 1]] = 1

plt.figure(figsize=(4, 4))
plt.imshow(mask, cmap="Greys")
plt.title("Baseline mask (top-N pixels)", fontsize=12)
plt.axis("off")
plt.show()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3296220629.py in <cell line: 0>()
      4 # Fix: the mask must match the real image shape (the original code used 256x266).
      5 # Confirm shape from one file.
----> 6 sample_band = np.load(test_dir / test_recs[0] / "band_08.npy")  # H x W x T
      7 H, W, T = sample_band.shape
      8 print("Sample band_08 shape:", sample_band.shape)

NameError: name 'test_recs' is not defined

## === cell 4
enc = list_to_string(rle_encode(mask))
print("Encoded example (first 80 chars):", enc[:80])



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3087095910.py in <cell line: 0>()
      1 # Quick sanity check of encoding output
----> 2 enc = list_to_string(rle_encode(mask))
      3 print("Encoded example (first 80 chars):", enc[:80])
      4 

NameError: name 'mask' is not defined

## === cell 5
sub_path = data_path / "sample_submission.csv"
if not sub_path.exists():
    raise FileNotFoundError(
        f"Expected sample submission at {sub_path}, but it does not exist."
    )

submission = pd.read_csv(sub_path)
if "record_id" not in submission.columns or "encoded_pixels" not in submission.columns:
    raise ValueError(
        f"Unexpected sample submission columns: {submission.columns.tolist()}"
    )

submission["record_id"] = submission["record_id"].astype(str)
submission = submission.set_index("record_id", drop=False)

print("Sample submission shape:", submission.shape)
print("Columns:", submission.columns.tolist())
submission.head()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1540399018.py in <cell line: 0>()
      4 sub_path = data_path / "sample_submission.csv"
      5 if not sub_path.exists():
----> 6     raise FileNotFoundError(
      7         f"Expected sample submission at {sub_path}, but it does not exist."
      8     )

FileNotFoundError: Expected sample submission at /kaggle/input/google-research-identify-contrails-reduce-global-warming/google-research-identify-contrails-reduce-global-warming/sample_submission.csv, but it does not exist.

## === cell 6
for rec in test_recs:
    band_08 = np.load(test_dir / rec / "band_08.npy").sum(axis=2)
    preds = np.c_[
        np.unravel_index(np.argpartition(band_08.ravel(), -n)[-n:], band_08.shape)
    ]
    mask = np.zeros((band_08.shape[0], band_08.shape[1]), dtype=np.uint8)
    mask[preds[:, 0], preds[:, 1]] = 1
    submission.loc[rec, "encoded_pixels"] = list_to_string(rle_encode(mask))

submission[["record_id", "encoded_pixels"]].head()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/257176568.py in <cell line: 0>()
      1 # Generate predictions for each test record and fill encoded_pixels.
      2 # Keep the core logic the same: use band_08 sum over time and pick top-n pixels.
----> 3 for rec in test_recs:
      4     band_08 = np.load(test_dir / rec / "band_08.npy").sum(axis=2)
      5     preds = np.c_[

NameError: name 'test_recs' is not defined

## === cell 7
out = submission[["record_id", "encoded_pixels"]].copy()
out.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", out.shape)
print(out.head())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/68830292.py in <cell line: 0>()
      1 # Write a valid submission: exactly required columns, correct filename suffix.
----> 2 out = submission[["record_id", "encoded_pixels"]].copy()
      3 out.to_csv("submission.csv", index=False)
      4 
      5 print("Wrote submission.csv with shape:", out.shape)

NameError: name 'submission' is not defined
