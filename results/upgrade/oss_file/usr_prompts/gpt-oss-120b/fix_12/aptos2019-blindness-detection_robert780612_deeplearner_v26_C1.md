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

0.8627159209634188

# 6. Current score

0.29851

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I adjust the prediction function so that it converts the model’s 1000‑dimensional output into a single scalar (by averaging) before averaging with the flipped image prediction. This removes the runtime error and ensures the `predictions` list is created, allowing the subsequent cell to build and save a valid `submission.csv` file.'
- What this solution (achieved 0.14325) has done: 'I keep the overall model and prediction pipeline unchanged, but replace the hard‑coded threshold bins with a simple linear scaling of the raw regression output to the 0‑4 range followed by rounding. This spreads the predictions across all classes, avoiding the all‑zero result that gave a score of 0 and moving the validation metric toward the target. The modification is limited to the post‑processing cell, preserving the core logic while improving the expected Quadratic Weighted Kappa.'
- What this solution (achieved 0.0) has done: 'I replace the aggressive min‑max scaling of the raw model outputs with a simple bias‑adjustment followed by clipping and rounding. This keeps the core model unchanged while providing a more realistic mapping of the continuous predictions to the 0‑4 class range, which should improve the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.0) has done: 'The post‑processing was only bias‑shifting the raw regression output, which can keep the predictions poorly calibrated for the 0‑4 severity scale.  
Replacing that with a sigmoid → scale → round mapping aligns the model’s continuous output to the required class range more naturally, while keeping the rest of the pipeline unchanged. This modest calibration should move the quadratic weighted kappa closer to the target score.'
- What this solution (achieved 0.03145) has done: 'I add a lightweight calibration step that fits a simple linear mapping from the model’s raw output to the true diagnosis values using the training set. This mapping is then applied to the test predictions before rounding and clipping, which spreads the predictions across all classes and should move the quadratic weighted kappa toward the target score while keeping the core model unchanged.'
- What this solution (achieved -0.02148) has done: 'I replace the simple linear‐fit calibration with a min‑max scaling that stretches the model’s raw outputs to the full 0‑4 diagnosis range before rounding. This keeps the model unchanged but spreads predictions across all classes, which should raise the quadratic weighted kappa toward the target while staying within the original pipeline.'
- What this solution (achieved -0.01448) has done: 'I replace the simplistic min‑max calibration with a proper linear regression (least‑squares) fit between the model’s raw predictions on the training set and the true diagnosis labels. This keeps the overall pipeline unchanged while providing a more accurate mapping, which should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved -0.04429) has done: 'I replace the linear‑calibration step in the post‑processing cell with a sigmoid‑based mapping that squashes the raw model output into the 0‑4 range before rounding. This keeps the core model and prediction pipeline unchanged while providing a more stable, bounded conversion that should spread predictions across classes and raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.29851) has done: 'I adjust the post‑processing step so that the linear calibration parameters `a` and `b` (computed on the training set) are actually applied to the raw model predictions before clipping and rounding. The original code ignored this calibration and used a sigmoid transformation, which drives the score far below the target. By mapping the raw outputs with `a*pred + b` and then converting them to the 0‑4 class range, we keep the core model unchanged while improving the alignment with the evaluation metric.'

# 9. Code solution

## === cell 0
import os, sys, math, types
from glob import glob
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



## === cell 1
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
    "vgg19",
    "vgg19_bn",
]




## === cell 2
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


def se_resnet50(num_classes=1000, pretrained=None):
    model = models.resnet50(pretrained=(pretrained == "imagenet"))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_se_resnet50_gem(pretrain):
    if pretrain == "imagenet":
        model = se_resnet50(num_classes=1000, pretrained="imagenet")
    else:
        model = se_resnet50(num_classes=1000, pretrained=None)
    model.avgpool = GeM()
    model.last_linear = nn.Linear(2048, 1)
    return model




## === cell 3
TEST_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/test_images"
test_images = sorted(glob(os.path.join(TEST_IMAGE_PATH, "*.png")))

norm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)




## === cell 4
def _tensor_to_scalar(tensor):
    """
    Convert any tensor output to a single scalar.
    If the tensor has more than one element, use the mean value.
    """
    if tensor.numel() == 1:
        return tensor.item()
    else:
        return tensor.mean().item()


def make_predictions(model, image_paths, transform, size=256, device=device):
    model.eval()
    predictions = []
    for im_path in image_paths:
        img = Image.open(im_path).convert("RGB")
        img = img.resize((size, size), resample=Image.BILINEAR)
        tensor = transform(img).to(device)

        with torch.no_grad():
            out = model(tensor.unsqueeze(0))
            out_flip = model(torch.flip(tensor.unsqueeze(0), dims=(3,)))
        out_scalar = _tensor_to_scalar(out.squeeze())
        out_flip_scalar = _tensor_to_scalar(out_flip.squeeze())
        pred = (out_scalar + out_flip_scalar) / 2.0
        img_id = os.path.splitext(os.path.basename(im_path))[0]
        predictions.append((img_id, pred))
    return predictions




## === cell 5
MODEL_PATH = "/kaggle/input/seresnet50pretrain/fine_tune_256_model30.pth"
model = get_se_resnet50_gem(pretrain=False)
model.to(device)

try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("Checkpoint loaded successfully.")
except Exception as e:
    print(f"Warning: could not load checkpoint ({e}); using random weights.")

TRAIN_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/train.csv"
TRAIN_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/train_images"

if os.path.exists(TRAIN_CSV_PATH) and os.path.isdir(TRAIN_IMAGE_PATH):
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    train_image_paths = [
        os.path.join(TRAIN_IMAGE_PATH, f"{img_id}.png")
        for img_id in train_df["id_code"]
    ]
    print(f"Running calibration on {len(train_image_paths)} training images...")
    train_preds = make_predictions(
        model, train_image_paths, norm, size=256, device=device
    )
    train_pred_vals = np.array([p[1] for p in train_preds])
    train_labels = train_df["diagnosis"].values.astype(float)

    if train_pred_vals.size > 0 and np.isfinite(train_pred_vals).all():
        X = np.column_stack((train_pred_vals, np.ones_like(train_pred_vals)))
        coeffs, residuals, rank, s = np.linalg.lstsq(X, train_labels, rcond=None)
        a, b = coeffs[0], coeffs[1]
        if np.isfinite(a) and np.isfinite(b):
            print(f"Linear calibration parameters: a={a:.5f}, b={b:.5f}")
        else:
            a, b = 1.0, 0.0
            print("Calibration produced non‑finite parameters; fallback to identity.")
    else:
        a, b = 1.0, 0.0
        print("Calibration fallback to identity.")
else:
    a, b = 1.0, 0.0
    print("Training data not found; using identity calibration.")



## === cell 6
predictions = make_predictions(model, test_images, norm, size=256, device=device)



## === cell 7
submission = pd.DataFrame(predictions, columns=["id_code", "diagnosis"])

raw_vals = submission["diagnosis"].values.astype(float)
calibrated = a * raw_vals + b

final_vals = np.clip(np.round(calibrated), 0, 4).astype(int)

submission["diagnosis"] = final_vals

submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
