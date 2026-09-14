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
numpy==1.26.4
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

0.9194098121606712

# 6. Current score

0.12083

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the model loading robust and use ImageNet‑pretrained weights (instead of random weights) so predictions are non‑trivial even if the fine‑tuned checkpoints are missing. I also add a safe fallback that fills predictions with the global mean diagnosis from the training set when no model predictions are available. This ensures a valid CSV is always written and moves the score toward the target.'
- What this solution (achieved 0.01091) has done: 'I replace the missing custom model imports with standard torchvision models, keeping the same GeM pooling and single‑output head so predictions are non‑trivial. I also load checkpoints with `strict=False` to avoid shape mismatches, and keep the rest of the pipeline unchanged. This minimal change lets the ensemble produce varied predictions instead of a constant fallback, moving the kappa score toward the target.'
- What this solution (achieved 0.0) has done: 'I adjust the prediction step to map the raw model output into the 0‑4 diagnosis range using a sigmoid scaling, and then simplify the post‑processing by rounding the averaged predictions to the nearest integer (clipped to 0‑4). I also add the standard ImageNet normalization to the Densenet transform so its features are on a comparable scale. These small, targeted changes keep the core architecture unchanged while producing more realistic score values, which should move the quadratic weighted kappa much closer to the target.'
- What this solution (achieved 0.29348) has done: 'We adjust the prediction scaling to use the raw model output (instead of a sigmoid) before multiplying by 4, allowing a wider spread of values and thus a higher quadratic weighted kappa. The change is limited to the `make_predictions` function and does not alter model architecture or training logic.'
- What this solution (achieved 0.17897) has done: 'I adjust the ensemble post‑processing to spread the averaged predictions across the full 0‑4 range using a min‑max scaling step. This small change keeps the model architecture and training untouched while giving the predictions more variance, which should move the quadratic weighted kappa closer to the target score.'
- What this solution (achieved -0.23764) has done: 'I keep the model loading and prediction pipeline unchanged but improve the ensemble step: use weighted averaging of model outputs (favoring the stronger Densenet), apply a modest gamma‑stretch after the min‑max scaling to increase prediction variance, and fix the typo in the SE‑ResNet‑50 loading call. These small adjustments are expected to raise the quadratic weighted kappa toward the target without altering the core architecture.'
- What this solution (achieved -0.53707) has done: 'I add a global training‑set standard‑deviation variable and, after the ensemble scaling, calibrate the pooled predictions to match the training label distribution (mean ± std). This simple linear re‑scaling keeps the original model architecture and ensemble unchanged while giving the predictions a more realistic spread, which should raise the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.00706) has done: 'I add missing imports, define the global mean diagnosis from the training labels, and fix the prediction scaling (use raw linear output × 4 instead of a sigmoid) to give predictions more variance. These minimal fixes resolve the NameErrors, ensure a valid CSV is written, and should move the quadratic weighted kappa closer to the target score.'
- What this solution (achieved 0.12083) has done: 'I add a min‑max scaling step to the ensemble predictions (cell 6) so the averaged outputs are stretched to the full 0‑4 range before rounding. This simple post‑processing change preserves the model architecture and training logic, yet gives the predictions much more variance, which should raise the quadratic weighted kappa toward the target score. No other parts of the pipeline are altered.'

# 9. Code solution

## === cell 0
import os
import glob
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.item():.4f}, eps={self.eps})"


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


def get_se_resnet50_gem(pretrain="imagenet"):
    """Use torchvision ResNet50 as a drop‑in for the missing SE‑ResNet50."""
    if pretrain == "imagenet":
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        base = models.resnet50(weights=None)
    base.avgpool = GeM()
    base.fc = nn.Linear(base.fc.in_features, 1)
    return base


def get_densenet121_gem(pretrain="imagenet"):
    """Use torchvision DenseNet121 as a drop‑in for the missing custom model."""
    if pretrain == "imagenet":
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    else:
        base = models.densenet121(weights=None)
    base.features.avgpool = GeM()
    base.classifier = nn.Linear(base.classifier.in_features, 1)
    return base


TRAIN_CSV_PATHS = [
    "/kaggle/input/aptos2019-blindness-detection/train.csv",
    "/kaggle/input/train.csv",
]
train_csv_path = next((p for p in TRAIN_CSV_PATHS if os.path.exists(p)), None)
if train_csv_path is None:
    raise FileNotFoundError("train.csv not found in expected locations.")
train_df = pd.read_csv(train_csv_path)
GLOBAL_MEAN_DIAGNOSIS = train_df["diagnosis"].mean()




## === cell 1
TEST_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/test_images"
device = torch.device("cpu")
test_images = glob.glob(os.path.join(TEST_IMAGE_PATH, "*.png"))




## === cell 2
def make_predictions(
    model, test_images, transforms, size=256, device=torch.device("cpu")
):
    """
    Produce predictions scaled to the 0‑4 diagnosis range.
    The raw linear output is multiplied by 4 to obtain a value in [0,4].
    """
    predictions = []
    model.eval()
    with torch.no_grad():
        for im_path in test_images:
            image = Image.open(im_path).convert("RGB")
            image = image.resize((size, size), resample=Image.BILINEAR)
            image = transforms(image).to(device)
            output = model(image.unsqueeze(0))
            output_flip = model(torch.flip(image.unsqueeze(0), dims=(3,)))
            raw = (output.item() + output_flip.item()) / 2.0
            final_prediction = raw * 4.0  # direct scaling
            predictions.append(
                (os.path.splitext(os.path.basename(im_path))[0], final_prediction)
            )
    return predictions




## === cell 3
MODEL_PATH_DENSE = "/kaggle/input/densenet121/model_densenet121_bs64_30.pth"
try:
    model = get_densenet121_gem(pretrain="imagenet")
    model.to(device)
    if os.path.exists(MODEL_PATH_DENSE):
        state = torch.load(MODEL_PATH_DENSE, map_location="cpu")
        try:
            model.load_state_dict(state, strict=False)
        except Exception as e:
            print(
                f"Warning: checkpoint load issue ({e}); proceeding with pretrained weights."
            )
    else:
        print("Warning: Densenet checkpoint not found; using ImageNet weights.")
    norm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    predictions_densenet = make_predictions(
        model, test_images, norm, size=224, device=device
    )
except Exception as e:
    print(
        f"Warning: Densenet model could not be loaded ({e}); using empty predictions."
    )
    predictions_densenet = []




## === cell 4
MODEL_PATH_SERES = "/kaggle/input/seresnet50pretrain/fine_tune_256_model30.pth"
try:
    model = get_se_resnet50_gem(pretrain="imagenet")
    model.to(device)
    if os.path.exists(MODEL_PATH_SERES):
        state = torch.load(MODEL_PATH_SERES, map_location="cpu")
        try:
            model.load_state_dict(state, strict=False)
        except Exception as e:
            print(
                f"Warning: checkpoint load issue ({e}); proceeding with pretrained weights."
            )
    else:
        print("Warning: SE‑ResNet50 checkpoint not found; using ImageNet weights.")
    norm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    predictions_seresnet = make_predictions(
        model, test_images, norm, size=256, device=device
    )
except Exception as e:
    print(
        f"Warning: SE‑ResNet50 model could not be loaded ({e}); using empty predictions."
    )
    predictions_seresnet = []




## === cell 5
MODEL_PATH_SERES_512 = "/kaggle/input/seresnet50-512/model30_512.pth"
try:
    model = get_se_resnet50_gem(pretrain="imagenet")
    model.to(device)
    if os.path.exists(MODEL_PATH_SERES_512):
        state = torch.load(MODEL_PATH_SERES_512, map_location="cpu")
        try:
            model.load_state_dict(state, strict=False)
        except Exception as e:
            print(
                f"Warning: checkpoint load issue ({e}); proceeding with pretrained weights."
            )
    else:
        print("Warning: SE‑ResNet50‑512 checkpoint not found; using ImageNet weights.")
    norm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    predictions_seresnet_512 = make_predictions(
        model, test_images, norm, size=512, device=device
    )
except Exception as e:
    print(
        f"Warning: SE‑ResNet50‑512 model could not be loaded ({e}); using empty predictions."
    )
    predictions_seresnet_512 = []




## === cell 6
available = [
    p
    for p in [predictions_densenet, predictions_seresnet, predictions_seresnet_512]
    if p
]
if not available:
    ids = [os.path.splitext(os.path.basename(p))[0] for p in test_images]
    avg_vals = np.full(len(ids), GLOBAL_MEAN_DIAGNOSIS, dtype=float)
    final_predictions = list(zip(ids, avg_vals))
else:
    base_weights = np.array([0.5, 0.3, 0.2])
    weights = base_weights[: len(available)]
    weights = weights / weights.sum()  # normalize
    ids = [pid for pid, _ in available[0]]
    stacked_vals = np.vstack([[val for _, val in preds] for preds in available])
    avg_vals = np.average(stacked_vals, axis=0, weights=weights)

    min_val = avg_vals.min()
    max_val = avg_vals.max()
    if max_val > min_val:
        avg_vals = (avg_vals - min_val) / (max_val - min_val) * 4.0
    else:
        avg_vals = np.full_like(avg_vals, GLOBAL_MEAN_DIAGNOSIS)

    avg_vals = np.clip(avg_vals, 0.0, 4.0)
    final_predictions = list(zip(ids, avg_vals))




## === cell 7
submission = pd.DataFrame(final_predictions, columns=["id_code", "diagnosis"])
submission["diagnosis"] = np.clip(np.rint(submission["diagnosis"]), 0, 4).astype(int)

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
