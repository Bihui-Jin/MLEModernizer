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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

fastai==2.8.5
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
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.8922866227853757

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch

from fastai.vision.all import *



## === cell 1
data_dir = Path("/kaggle/input/aptos2019-blindness-detection")
if not data_dir.exists():
    data_dir = Path("../input/aptos2019-blindness-detection")

model_dir = Path("/kaggle/input/my-aptos2019-blindness-detection")
if not model_dir.exists():
    model_dir = Path("../input/my-aptos2019-blindness-detection")

batch_size = 8
im_size = (528, 528)

train = False
ssl = False
c = 1 / 3

model_name = "efficientnet-b6"




## === cell 2
def preprocess_pil(im: PILImage):
    arr = np.array(im)
    if arr.ndim == 3:
        mask = arr.sum(axis=2) > 0
    else:
        mask = arr > 0
    if not mask.any():
        return im
    ys, xs = np.where(mask)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return im.crop((int(x0), int(y0), int(x1) + 1, int(y1) + 1))




## === cell 3
pkl_path = model_dir / f"{model_name}.pkl"
if not pkl_path.exists():
    raise FileNotFoundError(f"Expected exported learner not found: {pkl_path}")

learn = load_learner(pkl_path, cpu=False)
learn.model.eval()

learn.path = Path(".")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/513088719.py in <cell line: 0>()
      4 if not pkl_path.exists():
      5     # Provide a clear error if the expected artifact is missing.
----> 6     raise FileNotFoundError(f"Expected exported learner not found: {pkl_path}")
      7 
      8 learn = load_learner(pkl_path, cpu=False)

FileNotFoundError: Expected exported learner not found: ../input/my-aptos2019-blindness-detection/efficientnet-b6.pkl

## === cell 4
mean = tensor(imagenet_stats[0])
std = tensor(imagenet_stats[1])


def diagnose_id(id_code: str) -> int:
    img_path = data_dir / "test_images" / f"{id_code}.png"
    im = PILImage.create(img_path)
    im = preprocess_pil(im)
    im = im.resize(im_size, resample=Image.BILINEAR)  # deterministic and fast
    x = TensorImage(np.array(im).transpose(2, 0, 1)).float() / 255.0  # CHW, 0..1
    x = (x - mean[:, None, None]) / std[:, None, None]
    x = x[None].to(learn.dls.device)

    with torch.no_grad():
        pred = learn.model(x)
        pred_class = int(torch.argmax(pred, dim=1).item())
    return pred_class


test_df = pd.read_csv(data_dir / "test.csv")
test_df["diagnosis"] = [diagnose_id(x) for x in test_df["id_code"].tolist()]

sub_path = Path("submission.csv")
test_df[["id_code", "diagnosis"]].to_csv(sub_path, index=False, lineterminator="\n")

print(f"Wrote {sub_path} with shape {test_df[['id_code','diagnosis']].shape}")
print(test_df.head())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2103898755.py in <cell line: 0>()
     26 
     27 test_df = pd.read_csv(data_dir / "test.csv")
---> 28 test_df["diagnosis"] = [diagnose_id(x) for x in test_df["id_code"].tolist()]
     29 
     30 # Ensure correct submission format and .csv suffix

/tmp/ipykernel_55/2103898755.py in <listcomp>(.0)
     26 
     27 test_df = pd.read_csv(data_dir / "test.csv")
---> 28 test_df["diagnosis"] = [diagnose_id(x) for x in test_df["id_code"].tolist()]
     29 
     30 # Ensure correct submission format and .csv suffix

/tmp/ipykernel_55/2103898755.py in diagnose_id(id_code)
     17     x = TensorImage(np.array(im).transpose(2, 0, 1)).float() / 255.0  # CHW, 0..1
     18     x = (x - mean[:, None, None]) / std[:, None, None]
---> 19     x = x[None].to(learn.dls.device)
     20 
     21     with torch.no_grad():

NameError: name 'learn' is not defined
