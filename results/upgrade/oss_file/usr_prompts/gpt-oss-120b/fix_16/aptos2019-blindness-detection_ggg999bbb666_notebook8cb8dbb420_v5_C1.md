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
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
        input/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
            test_images/
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
            train_images/
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> input/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> working/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.8836900126751313

# 6. Current score

-0.07398

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I add the missing pandas import, ensure the submission respects the original test CSV ordering by merging predictions with the test file, and renumber the cells so they start at 1 as required. These fixes remove the NameError, guarantee a correctly‑formatted CSV, and keep the core model logic unchanged.'
- What this solution (achieved 0.0) has done: 'I renumber the cells so they start at 1, keep the original model loading unchanged, and replace the fixed threshold mapping with a simple min‑max linear scaling of the raw model outputs to the 0‑4 class range. This yields varied predictions instead of all zeros, moving the score toward the target while preserving the core logic.'
- What this solution (achieved -0.27246) has done: 'I keep the overall pipeline unchanged but improve the fallback model so it produces diverse predictions instead of a constant zero. By loading a pretrained ResNet‑50 from torchvision and replacing its pooling with GeM and its final layer with a single linear output, the model generate varied raw scores that are then scaled to the 0‑4 range, moving the quadratic weighted kappa toward the target. I also fix the cell numbering and ensure all imports are present.'
- What this solution (achieved -0.07313) has done: 'I keep the model loading and image‑processing unchanged, but replace the naive min‑max scaling with a quantile‑based mapping that respects the class distribution in the training data. By aligning the predicted score quantiles to the observed label frequencies, the predictions better match the target distribution, which should raise the quadratic weighted kappa toward the target while preserving the core pipeline.'
- What this solution (achieved 0.12005) has done: 'I add ImageNet‑style normalization to the image tensors and apply a simple test‑time augmentation (horizontal flip) whose predictions are averaged, which usually improves the raw model scores without altering the core architecture or training logic. These modest changes are expected to raise the quadratic weighted kappa toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.0) has done: 'I keep the existing model and image‑processing pipeline, but add a tiny calibration step: after loading the training labels I run the same model on the training images to obtain raw regression scores, fit a simple linear regression that maps those raw scores to the true diagnosis, and then apply this learned linear mapping to the test raw predictions before rounding to the 0‑4 classes. This inexpensive calibration keeps the core architecture unchanged while creating a stronger monotonic relationship between model outputs and the target, which should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.43805) has done: 'I align the training predictions with their true labels (instead of assuming CSV order) and replace the linear‑regression calibration with a rank‑based mapping that preserves the monotonic relationship between raw model outputs and diagnoses. This calibration is more robust for a model that produces random‑looking scores, and it keeps the original architecture and processing unchanged while still improving the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.05297) has done: 'I keep the model architecture unchanged but replace the rank‑based interpolation calibration with a simple linear‑regression mapping from the raw model outputs to the true diagnoses. This monotonic linear fit usually yields a tighter correlation and thus a higher quadratic weighted kappa while preserving the original pipeline. I also renumber the cells to start at 1 so the script runs end‑to‑end and writes a valid submission.csv.'
- What this solution (achieved 0.33477) has done: 'I replace the linear‑regression calibration with a monotonic rank‑based mapping that aligns the distribution of raw model scores on the training set with the observed class frequencies. By using the training raw predictions to compute quantile thresholds for each diagnosis class and then assigning test predictions according to those thresholds, we preserve the core model logic while creating a more meaningful mapping that should raise the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.21592) has done: 'I replace the rank‑based calibration with a simple linear regression (using NumPy polyfit) that maps the raw model scores to the true diagnoses on the training set. This monotonic mapping usually yields a tighter correlation and therefore a higher quadratic weighted kappa, moving the score closer to the target while leaving the model architecture and inference pipeline unchanged. I also renumber the notebook cells to start at 1 so the script runs end‑to‑end.'
- What this solution (achieved 0.15439) has done: 'I replace the simple linear‑regression calibration with a rank‑based (quantile) mapping that respects the observed class frequencies in the training set, which has been shown to raise the quadratic weighted kappa toward the target. I also renumber the notebook cells so they start at 1, keeping the original model and preprocessing unchanged, and ensure the submission CSV is written correctly.'
- What this solution (achieved 0.00221) has done: 'I keep the same model and image preprocessing but change the raw score generation to a sigmoid‑scaled value (now bounded between 0 and 4) and replace the quantile‑based calibration with a simple linear regression (via numpy polyfit). This small, monotonic mapping usually aligns the predictions better with the true diagnoses, moving the quadratic weighted kappa toward the target while leaving the core architecture untouched.'
- What this solution (achieved 0.01663) has done: 'I add a monotonic isotonic‑regression calibration step (using sklearn IsotonicRegression) which better aligns the raw model scores to the true diagnosis distribution without altering the model itself. This small change keeps the core architecture untouched while providing a more appropriate mapping than the simple linear fit, and therefore should move the quadratic weighted kappa much closer to the target. I also renumber the notebook cells to start at 1 so the script runs end‑to‑end without indexing issues.'
- What this solution (achieved -0.07398) has done: 'I replace the isotonic‑regression calibration with a rank‑based (quantile) mapping that aligns the distribution of raw model scores on the training set to the observed class frequencies – this approach previously raised the score to ~0.44 and should move the current 0.01663 score much closer to the target. The rest of the pipeline, model, and preprocessing stay unchanged, ensuring the core logic is preserved while improving the prediction calibration.'

# 9. Code solution

## === cell 0
import os
import warnings
from glob import glob

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

from sklearn.isotonic import IsotonicRegression


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.item():.4f}, eps={self.eps})"


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


def get_se_resnet50_gem(pretrain: str = None):
    try:
        import pretrainedmodels  # type: ignore

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
    except Exception as e:
        warnings.warn(
            f"pretrainedmodels not available or failed to load ({e}). "
            "Using a torchvision ResNet‑50 fallback model."
        )
        model = models.resnet50(pretrained=True)
        model.avgpool = GeM()
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model




## === cell 1
BASE_DIR = "/kaggle/input/aptos2019-blindness-detection"
TEST_IMAGE_PATH = os.path.join(BASE_DIR, "test_images")
TRAIN_IMAGE_PATH = os.path.join(BASE_DIR, "train_images")
MODEL_PATH = "/kaggle/input/224best/224best.pth"  # may not exist in this kernel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_se_resnet50_gem(pretrain=None)
model.to(device)
model.eval()

if os.path.isfile(MODEL_PATH):
    try:
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state, strict=False)
    except Exception as load_err:
        warnings.warn(f"Failed to load pretrained weights: {load_err}")
else:
    warnings.warn(
        f"Model checkpoint not found at {MODEL_PATH}. Using fallback model with random (but varied) outputs."
    )

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
train_images = sorted(glob(os.path.join(TRAIN_IMAGE_PATH, "*.png")))
if len(train_images) != len(train_df):
    warnings.warn(
        "Number of training images does not match train.csv rows; calibration may be noisy."
    )

train_raw_preds = []
train_ids = []

for im_path in train_images:
    img_id = os.path.splitext(os.path.basename(im_path))[0]
    with Image.open(im_path) as img:
        img = img.convert("RGB")
        img = img.resize((224, 224), resample=Image.BILINEAR)
        tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out1 = model(tensor)
        flipped = torch.flip(tensor, dims=[3])
        out2 = model(flipped)
    pred = torch.sigmoid((out1.squeeze() + out2.squeeze()) / 2.0).cpu().item() * 4.0
    train_raw_preds.append(pred)
    train_ids.append(img_id)

if len(train_raw_preds) > 0:
    train_pred_df = pd.DataFrame({"id_code": train_ids, "raw_pred": train_raw_preds})
    train_merged = train_pred_df.merge(train_df, on="id_code", how="inner")
    if len(train_merged) < len(train_df):
        warnings.warn(
            "Some training ids were not matched; calibration will use the matched subset."
        )

    raw_vals = train_merged["raw_pred"].values.astype(float)
    labels = train_merged["diagnosis"].values.astype(int)

    if raw_vals.size == 0 or np.all(np.isnan(raw_vals)):
        warnings.warn("Invalid raw prediction values; falling back to simple rounding.")

        def calibrate(raw_vals_input):
            return np.round(raw_vals_input).astype(int).clip(0, 4)

    else:
        label_counts = np.bincount(labels, minlength=5).astype(float)
        if label_counts.sum() == 0:
            warnings.warn("No label counts found; using rounding fallback.")

            def calibrate(raw_vals_input):
                return np.round(raw_vals_input).astype(int).clip(0, 4)

        else:
            cum_frac = np.cumsum(label_counts) / label_counts.sum()
            thresholds = np.percentile(raw_vals, cum_frac[:-1] * 100)

            def calibrate(raw_vals_input):
                return np.searchsorted(thresholds, raw_vals_input, side="right").astype(
                    int
                )

else:
    warnings.warn("No training predictions available; using simple rounding fallback.")

    def calibrate(raw_vals_input):
        return np.round(raw_vals_input).astype(int).clip(0, 4)


test_images = sorted(glob(os.path.join(TEST_IMAGE_PATH, "*.png")))
if not test_images:
    raise RuntimeError(f"No test images found in {TEST_IMAGE_PATH}")

test_raw_preds = []
test_ids = []

for idx, im_path in enumerate(test_images):
    if idx % 100 == 0 or idx == len(test_images) - 1:
        print(f"Processing {idx + 1}/{len(test_images)}")
    with Image.open(im_path) as img:
        img = img.convert("RGB")
        img = img.resize((224, 224), resample=Image.BILINEAR)
        tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out1 = model(tensor)
        flipped_tensor = torch.flip(tensor, dims=[3])
        out2 = model(flipped_tensor)

    pred = torch.sigmoid((out1.squeeze() + out2.squeeze()) / 2.0).cpu().item() * 4.0
    test_raw_preds.append(pred)
    test_ids.append(os.path.splitext(os.path.basename(im_path))[0])

pred_df = pd.DataFrame({"id_code": test_ids, "raw_pred": test_raw_preds})
pred_df["diagnosis"] = calibrate(pred_df["raw_pred"].values)

test_df = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))
submission = test_df.merge(pred_df[["id_code", "diagnosis"]], on="id_code", how="left")

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")
