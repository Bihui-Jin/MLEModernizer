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

0.9049475539208136

# 6. Current score

-0.04406

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The fix updates the image resize dimension to 224 pixels so the DenseNet feature map size matches the hard‑coded 7×7 average‑pooling in the model, eliminating the tensor‑shape mismatch. This allows predictions to be generated, after which the submission dataframe is created and saved as a proper submission.csv file.'
- What this solution (achieved 0.0) has done: 'I add a lightweight fallback that uses the overall training label distribution when the model’s predictions are essentially uninformative (near‑constant). After generating predictions, if their standard deviation is very low we replace them with the rounded mean diagnosis from the training set. I also simplify the post‑processing by rounding predictions to the nearest integer and clipping to the valid range 0‑4, which aligns better with the quadratic weighted kappa metric while keeping the original model‑based pipeline untouched. This small change should raise the score from 0 towards the target without altering the core architecture or training logic.'
- What this solution (achieved 0.0) has done: 'I loosen the “uninformative‑prediction” check so that constant‑fallback replacement only occurs when the model truly outputs an identical value for every image (std = 0). This preserves the original predictions when they have any variation, giving the model a chance to achieve a higher quadratic weighted kappa instead of being forced to the mean class.'
- What this solution (achieved -0.01713) has done: 'I broaden the fallback that replaces model outputs with the overall training‑set mean when the predictions have little variation (standard deviation < 0.1). This keeps the core model unchanged but ensures a sensible, non‑constant prediction that yields a positive quadratic weighted kappa instead of the current zero score. The rest of the pipeline remains the same, and the script still writes a proper `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I adjust the prediction step to bound the model output with a sigmoid and scale it to the 0‑4 diagnosis range, and I stop the “low‑variance fallback” from overwriting predictions unless the output is truly constant. These small changes keep the core model untouched while giving predictions that better match the label scale, which should raise the quadratic weighted kappa toward the target.'
- What this solution (achieved -0.37841) has done: 'I add a lightweight calibration step that rescales the raw model outputs to match the training label distribution (mean ± std). This keeps the core architecture unchanged, ensures predictions vary instead of being near‑constant, and should raise the quadratic weighted kappa toward the target while still writing a proper `submission.csv`.'
- What this solution (achieved -0.02482) has done: 'I remove the extra calibration step that was distorting the model’s raw outputs and instead keep the predictions directly after sigmoid scaling, only clipping them to the valid 0‑4 range. This small change preserves the core architecture while avoiding the negative impact of the previous mean‑std rescaling, helping the quadratic weighted kappa move upward toward the target. The rest of the pipeline remains unchanged.'
- What this solution (achieved -0.17364) has done: 'I keep the model architecture unchanged and only adjust the post‑processing of the raw predictions. After obtaining the sigmoid‑scaled values (0‑4) I rescale them to match the training label distribution (using the training mean and std) and then clip to the valid range. This simple linear calibration should correct systematic bias and improve the quadratic weighted kappa without altering any core modeling code.'
- What this solution (achieved 0.0) has done: 'I simplify the post‑processing step: instead of re‑scaling predictions to match the training mean and standard deviation (which can distort the outputs and produce a negative kappa), I keep the raw sigmoid‑scaled values (scaled to 0‑4) and only apply a small‑variance fallback when the model predicts a constant. This minimal change preserves the core model and inference logic while making the predictions more aligned with the evaluation metric, moving the score toward the target.'
- What this solution (achieved -0.04406) has done: 'I added all necessary imports (torch, torchvision, PIL, numpy, pandas, os, glob, etc.) and fixed the model‑loading function to use the current torchvision API. The GeM pooling class now imports the required torch modules. With these corrections the script runs end‑to‑end, creates predictions (or falls back to the mean class when images are missing), rounds/clips them to the valid 0‑4 range, and writes a proper **submission.csv** file.'

# 9. Code solution

## === cell 0
import os
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.models import densenet121

__all__ = [
    "alexnet",
    "densenet121",
    "densenet169",
    "densenet201",
    "densenet161",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "inceptionv3",
    "squeezenet1_0",
    "squeezenet1_1",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
]

model_urls = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
    "densenet121": "http://data.lip6.fr/cadene/pretrainedmodels/densenet121-fbdb23505.pth",
    "densenet169": "http://data.lip6.fr/cadene/pretrainedmodels/densenet169-f470b90a4.pth",
    "densenet201": "http://data.lip6.fr/cadene/pretrainedmodels/densenet201-5750cbb1e.pth",
    "densenet161": "http://data.lip6.fr/cadene/pretrainedmodels/densenet161-347e6b360.pth",
    "inceptionv3": "https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth",
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "squeezenet1_0": "https://download.pytorch.org/models/squeezenet1_0-a815701f.pth",
    "squeezenet1_1": "https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth",
    "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}




## === cell 1
class GeM(nn.Module):
    """Generalized Mean (GeM) pooling layer."""

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.item():.4f}, eps={self.eps})"


def gem(x, p=3, eps=1e-6):
    """Helper for GeM pooling."""
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


def get_densenet121_gem(pretrain="imagenet"):
    """
    Returns a DenseNet‑121 model with GeM pooling and a single linear output.
    If `pretrain` is "imagenet" we load ImageNet weights; otherwise the model
    is left uninitialised.
    """
    if pretrain == "imagenet":
        model = densenet121(pretrained=True)
    else:
        model = densenet121(pretrained=False)

    model.features.avg_pool = GeM()
    model.classifier = nn.Linear(1024, 1)
    return model




## === cell 2
device = torch.device("cpu")

possible_paths = [
    "/kaggle/input/aptos2019-blindness-detection/test_images",
    "/kaggle/input/test_images",
    "./data/aptos2019-blindness-detection/test_images",
    "./input/aptos2019-blindness-detection/test_images",
    "./data/test_images",
    "./input/test_images",
]
TEST_IMAGE_PATH = None
for p in possible_paths:
    if os.path.isdir(p):
        TEST_IMAGE_PATH = p
        break
if TEST_IMAGE_PATH is None:
    print("Warning: Test image directory not found – will fall back to CSV IDs.")
    test_images = []
else:
    test_images = glob(os.path.join(TEST_IMAGE_PATH, "*.png"))


def _load_test_ids():
    """Read the test CSV and return a list of image IDs."""
    possible_test_paths = [
        "./test.csv",
        "./data/test.csv",
        "./input/test.csv",
        "/kaggle/input/aptos2019-blindness-detection/test.csv",
        "/kaggle/input/test.csv",
    ]
    for pt in possible_test_paths:
        if os.path.isfile(pt):
            df = pd.read_csv(pt)
            return df["id_code"].tolist()
    raise FileNotFoundError("Test CSV not found.")




## === cell 3
def make_predictions(
    model, test_images, transform, size=256, device=torch.device("cpu")
):
    """
    Run inference on each image, apply horizontal flip augmentation,
    average the sigmoid outputs and map them to the 0‑4 diagnosis range.
    """
    model.eval()
    predictions = []
    for im_path in test_images:
        image = Image.open(im_path).convert("RGB")
        image = image.resize((size, size), resample=Image.BILINEAR)

        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            out1 = model(tensor)
            out2 = model(torch.flip(tensor, dims=(3,)))  # horizontal flip
            pred1 = torch.sigmoid(out1).item()
            pred2 = torch.sigmoid(out2).item()
            pred = (pred1 + pred2) / 2.0 * 4.0  # average and map to 0‑4

        img_id = os.path.splitext(os.path.basename(im_path))[0]
        predictions.append((img_id, pred))
    return predictions


def _load_train_stats():
    """Return (mean, std) of the training diagnosis column."""
    possible_train_paths = [
        "./train.csv",
        "./data/train.csv",
        "./input/train.csv",
        "/kaggle/input/aptos2019-blindness-detection/train.csv",
        "/kaggle/input/train.csv",
    ]
    for pt in possible_train_paths:
        if os.path.isfile(pt):
            df = pd.read_csv(pt)
            mean = df["diagnosis"].mean()
            std = df["diagnosis"].std(ddof=0)
            return mean, max(std, 1e-3)  # avoid division by zero
    return 2.0, 1.0


def _load_train_mean():
    """Legacy helper – returns rounded mean (used for constant fallback)."""
    mean, _ = _load_train_stats()
    return int(round(mean))


try:
    model = get_densenet121_gem(pretrain="imagenet")
    print("Model loaded with ImageNet weights.")
except Exception as e:
    print(f"ImageNet weight loading failed ({e}), using untrained model.")
    model = get_densenet121_gem(pretrain=None)

model.to(device)

custom_weight_path = "../input/densenet121/model_densenet121_bs64_30.pth"
if os.path.exists(custom_weight_path):
    try:
        state = torch.load(custom_weight_path, map_location=device)
        model.load_state_dict(state)
        print("Custom weights loaded.")
    except Exception as e:
        print(
            f"Warning: could not load custom weights ({e}); using current model weights."
        )
else:
    print("Custom weight file not found; using current model weights.")

norm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

predictions = make_predictions(model, test_images, norm, size=224, device=device)

if len(predictions) == 0:
    test_ids = _load_test_ids()
    mean_diag = _load_train_mean()
    predictions = [(tid, float(mean_diag)) for tid in test_ids]
    print(
        f"No image files found – generated baseline predictions (class {mean_diag}) for {len(test_ids)} test IDs."
    )
else:
    pred_vals = np.array([p[1] for p in predictions])
    if pred_vals.std() < 1e-3:
        mean_diag = _load_train_mean()
        predictions = [(img_id, float(mean_diag)) for img_id, _ in predictions]
        print(
            f"Predictions were near‑constant; replaced with baseline class {mean_diag}."
        )
    else:
        predictions = [(pid, float(np.clip(val, 0.0, 4.0))) for pid, val in predictions]




## === cell 4
pred_df = pd.DataFrame(predictions, columns=["id_code", "diagnosis"])
pred_df["diagnosis"] = pred_df["diagnosis"].round().astype(int)
pred_df["diagnosis"] = pred_df["diagnosis"].clip(0, 4)

submission_path = "submission.csv"
pred_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
print(pred_df.head())
