# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.9

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

0.8925204558744121

# 6. Current score

None

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.01119) has done: 'I add an SSL‑context override before importing `pretrainedmodels` so the ImageNet weights can be downloaded without certificate errors. This small change removes the runtime failure while keeping the original model architecture and processing unchanged, allowing the script to run end‑to‑end and produce a valid `submission.csv`.'
- What this solution (achieved -0.02923) has done: 'I replace the handcrafted thresholding with a simple rounding‑and‑clipping step, which aligns the model’s single‑output regression to the required integer classes (0‑4). This tiny change keeps the core model unchanged while producing predictions that better match the evaluation metric, moving the score toward the target.'

# 9. Code solution

## === cell 0
import sys, subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "pretrainedmodels==0.7.4"]
)
print("pretrainedmodels installed")



## === cell 1
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import os
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms

import pretrainedmodels


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.item():.4f}, eps={self.eps})"


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


def get_se_resnet50_gem(pretrain="imagenet"):
    """Create SE‑ResNet‑50 with GeM pooling and a single‑output regression head."""
    if pretrain == "imagenet":
        model = pretrainedmodels.__dict__["se_resnet50"](
            num_classes=1000, pretrained="imagenet"
        )
    else:
        model = pretrainedmodels.__dict__["se_resnet50"](
            num_classes=1000, pretrained=None
        )
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(2048, 1)
    return model


BASE_INPUT = "/kaggle/input/aptos2019-blindness-detection"
TEST_IMAGE_PATH = os.path.join(BASE_INPUT, "test_images")
MODEL_PATH = "/kaggle/input/seresnet384/model_epoch33.pth"  # may not exist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_se_resnet50_gem(pretrain="imagenet")
model.to(device)

if os.path.exists(MODEL_PATH):
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint from {MODEL_PATH}")
else:
    print(
        f"Checkpoint not found at {MODEL_PATH}; using ImageNet‑pretrained weights only."
    )

model.eval()
ImageFile.LOAD_TRUNCATED_IMAGES = True  # safeguard for corrupted images

preprocess = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_images = sorted(glob(os.path.join(TEST_IMAGE_PATH, "*.png")))
predictions = []

with torch.no_grad():
    for i, im_path in enumerate(test_images):
        if i % 50 == 0 or i == len(test_images) - 1:
            print(f"Processing {i+1}/{len(test_images)}")
        try:
            image = Image.open(im_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open {im_path}: {e}")
            continue
        tensor = preprocess(image).to(device)
        output = model(tensor.unsqueeze(0))
        pred_value = output.item()
        img_id = os.path.splitext(os.path.basename(im_path))[0]
        predictions.append((img_id, pred_value))

submission = pd.DataFrame(predictions, columns=["id_code", "diagnosis"])
raw_path = "submission_raw_value.csv"
submission.to_csv(raw_path, index=False)
print(f"Raw predictions saved to {raw_path}")

min_pred = submission["diagnosis"].min()
max_pred = submission["diagnosis"].max()
if max_pred > min_pred:
    calibrated = (submission["diagnosis"] - min_pred) / (max_pred - min_pred) * 4.0
else:
    calibrated = pd.Series(0.0, index=submission.index)

submission["diagnosis"] = calibrated.round().clip(lower=0, upper=4).astype(int)

final_path = "submission.csv"
submission.to_csv(final_path, index=False)
print(f"Final submission saved to {final_path}")

submission.head()
