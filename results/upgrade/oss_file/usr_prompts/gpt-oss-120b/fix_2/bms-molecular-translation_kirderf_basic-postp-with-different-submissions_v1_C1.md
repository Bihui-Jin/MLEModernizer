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
Provided with images of chemicals, predict the corresponding International Chemical Identifier (InChI) text string of the image.

## Metric
Mean [Levenshtein distance](http://en.wikipedia.org/wiki/Levenshtein_distance) between the InChi strings you submit and the ground truth InChi values.

## Submission Format
For each `image_id` in the test set, you must predict the InChi string of the molecule in the corresponding image. The file should contain a header and have the following format:

```
image_id,InChI
00000d2a601c,InChI=1S/H2O/h1H2
00001f7fc849,InChI=1S/H2O/h1H2
000037687605,InChI=1S/H2O/h1H2
etc.
```

## Dataset
- **train/** - the training images, arranged in a 3-level folder structure by `image_id`
- **test/** - the test images, arranged in the same folder structure as `train/`
- **train_labels.csv** - ground truth InChi labels for the training images
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
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
            description.md (68 lines)
            extra_approved_InChIs.csv (9998712 lines)
            extra_approved_InChIs.csv.zip (428.7 MB)
            sample_submission.csv (484839 lines)
            sample_submission.csv.zip (4.1 MB)
            test.zip (883.2 MB)
            train.zip (3.5 GB)
            train_labels.csv (1939349 lines)
            train_labels.csv.zip (102.5 MB)
            bms-molecular-translation/
                description.md (68 lines)
                extra_approved_InChIs.csv (9998712 lines)
                ... and 7 other files
                bms-molecular-translation/
                test/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
                train/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
            test/
                0/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                1/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                ... and 15 other folders
            train/
                0/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                1/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                ... and 15 other folders
        input/
            description.md (68 lines)
            extra_approved_InChIs.csv (9998712 lines)
            extra_approved_InChIs.csv.zip (428.7 MB)
            sample_submission.csv (484839 lines)
            sample_submission.csv.zip (4.1 MB)
            test.zip (883.2 MB)
            train.zip (3.5 GB)
            train_labels.csv (1939349 lines)
            train_labels.csv.zip (102.5 MB)
            bms-molecular-translation/
                description.md (68 lines)
                extra_approved_InChIs.csv (9998712 lines)
                ... and 7 other files
                bms-molecular-translation/
                test/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
                train/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
            test/
                0/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                1/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                ... and 15 other folders
            train/
                0/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                1/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                ... and 15 other folders
        working/
            bms-molecular-translation/
                description.md (68 lines)
                extra_approved_InChIs.csv (9998712 lines)
                ... and 7 other files
                bms-molecular-translation/
                test/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
                train/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
```

-> data/bms-molecular-translation/extra_approved_InChIs.csv has 9998711 rows and 1 columns.
The columns are: InChI

-> data/bms-molecular-translation/sample_submission.csv has 484838 rows and 2 columns.
The columns are: image_id, InChI

-> data/bms-molecular-translation/train_labels.csv has 1939348 rows and 2 columns.
The columns are: image_id, InChI

-> data/extra_approved_InChIs.csv has 9998711 rows and 1 columns.
The columns are: InChI

-> data/sample_submission.csv has 484838 rows and 2 columns.
The columns are: image_id, InChI

-> data/train_labels.csv has 1939348 rows and 2 columns.
The columns are: image_id, InChI

-> input/bms-molecular-translation/extra_approved_InChIs.csv has 9998711 rows and 1 columns.
The columns are: InChI

-> (stopped after 10 files for performance)

# 5. Target score

4.662682011302231

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
from pathlib import Path
from collections import Counter




## === cell 1
def find_file(fname: str) -> Path:
    possible = [
        Path.cwd() / fname,
        Path.cwd() / "data" / fname,
        Path.cwd() / "input" / fname,
    ]
    for p in possible:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Could not find {fname} in expected locations.")


train_labels_path = find_file("train_labels.csv")
sample_submission_path = find_file("sample_submission.csv")



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3214620910.py in <cell line: 0>()
     14 
     15 
---> 16 train_labels_path = find_file("train_labels.csv")
     17 sample_submission_path = find_file("sample_submission.csv")
     18 

/tmp/ipykernel_11/3214620910.py in find_file(fname)
     11         if p.is_file():
     12             return p
---> 13     raise FileNotFoundError(f"Could not find {fname} in expected locations.")
     14 
     15 

FileNotFoundError: Could not find train_labels.csv in expected locations.

## === cell 2
train_labels = pd.read_csv(train_labels_path)
most_common_inchi = Counter(train_labels["InChI"]).most_common(1)[0][0]



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2234366602.py in <cell line: 0>()
      1 # Load the training labels and determine the most common InChI string.
----> 2 train_labels = pd.read_csv(train_labels_path)
      3 most_common_inchi = Counter(train_labels["InChI"]).most_common(1)[0][0]
      4 

NameError: name 'train_labels_path' is not defined

## === cell 3
submission = pd.read_csv(sample_submission_path)
submission["InChI"] = most_common_inchi



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3823165171.py in <cell line: 0>()
      1 # Load the sample submission (which contains the correct test image_id order)
----> 2 submission = pd.read_csv(sample_submission_path)
      3 # Fill every prediction with the most common InChI from the training set.
      4 submission["InChI"] = most_common_inchi
      5 

NameError: name 'sample_submission_path' is not defined

## === cell 4
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4012133464.py in <cell line: 0>()
      1 # Write the final submission file expected by Kaggle.
----> 2 submission.to_csv("submission.csv", index=False)

NameError: name 'submission' is not defined
