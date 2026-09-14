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
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tqdm==4.67.1

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

0.00344

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, math, pathlib
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm



## === cell 1
DATA_DIR = "/kaggle/input/google-research-identify-contrails-reduce-global-warming"
N_TIMES_BEFORE = 4  # index of the labeled frame within the time series



## === cell 2
train_ids = os.listdir(os.path.join(DATA_DIR, "train"))
valid_ids = os.listdir(os.path.join(DATA_DIR, "validation"))
test_ids = os.listdir(os.path.join(DATA_DIR, "test"))
print(f"train: {len(train_ids)}, valid: {len(valid_ids)}, test: {len(test_ids)}")



## === cell 3
_T11_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)


def normalize_range(data, bounds):
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def get_ash_colors(sample_id, split_dir):
    """
    Build a 3‑channel ash‑color image from bands 11, 14, 15.
    Returns an array of shape (H, W, 3, T) where T is the time dimension.
    """
    band15 = np.load(f"{DATA_DIR}/{split_dir}/{sample_id}/band_15.npy")
    band14 = np.load(f"{DATA_DIR}/{split_dir}/{sample_id}/band_14.npy")
    band11 = np.load(f"{DATA_DIR}/{split_dir}/{sample_id}/band_11.npy")

    r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
    g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(band14, _T11_BOUNDS)
    ash_colors = np.clip(np.stack([r, g, b], axis=2), 0, 1)
    return ash_colors  # shape (H, W, 3, T)




## === cell 4
def predict_mask(sample_id):
    """
    Produce a binary contrail mask for the labeled time step
    (index N_TIMES_BEFORE) by thresholding the red channel of the
    ash‑color image.
    """
    ash = get_ash_colors(sample_id, "test")
    frame = ash[..., N_TIMES_BEFORE]  # (H, W, 3)
    red_channel = frame[..., 0]  # use the red channel
    mask = (red_channel > 0.5).astype(np.uint8)
    return mask




## === cell 5
def rle_encode(x, fg_val=1):
    """
    Encode a binary mask `x` (2‑D) using run‑length encoding.
    Returns a list of integers.
    """
    dots = np.where(x.T.flatten() == fg_val)[0]  # Fortran order (column‑major)
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def list_to_string(x):
    """
    Convert a list of integers to the space‑separated string required by Kaggle.
    Empty list becomes '-'.
    """
    if x:
        return str(x).replace("[", "").replace("]", "").replace(",", "")
    else:
        return "-"




## === cell 6
print("Generating predictions for test set...")
predictions = []
for test_id in tqdm(test_ids):
    mask = predict_mask(test_id)
    predictions.append(mask)

print(f"Generated {len(predictions)} prediction masks.")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3263536621.py in <cell line: 0>()
      5 predictions = []
      6 for test_id in tqdm(test_ids):
----> 7     mask = predict_mask(test_id)
      8     predictions.append(mask)
      9 

/tmp/ipykernel_55/3855085338.py in predict_mask(sample_id)
      8     ash‑color image.
      9     """
---> 10     ash = get_ash_colors(sample_id, "test")
     11     # Select the frame that corresponds to the ground‑truth label
     12     frame = ash[..., N_TIMES_BEFORE]  # (H, W, 3)

/tmp/ipykernel_55/1028651396.py in get_ash_colors(sample_id, split_dir)
     16     Returns an array of shape (H, W, 3, T) where T is the time dimension.
     17     """
---> 18     band15 = np.load(f"{DATA_DIR}/{split_dir}/{sample_id}/band_15.npy")
     19     band14 = np.load(f"{DATA_DIR}/{split_dir}/{sample_id}/band_14.npy")
     20     band11 = np.load(f"{DATA_DIR}/{split_dir}/{sample_id}/band_11.npy")

/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py in load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size)
    425             own_fid = False
    426         else:
--> 427             fid = stack.enter_context(open(os_fspath(file), "rb"))
    428             own_fid = True
    429 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/google-research-identify-contrails-reduce-global-warming/test/test/band_15.npy'

## === cell 7
sample_submission_path = os.path.join(DATA_DIR, "sample_submission.csv")
submission = pd.read_csv(sample_submission_path, index_col="record_id")[
    0:0
]  # keep column structure

for test_id, mask in zip(test_ids, predictions):
    submission.loc[int(test_id), "encoded_pixels"] = list_to_string(rle_encode(mask))

output_path = os.path.join("/kaggle/working", "submission.csv")
submission.to_csv(output_path)
print(f"Submission written to {output_path}")

## --- ERROR in outputing the csv:
Invalid submission: Expected submission to have 1856 rows, but got 1823 instead!
