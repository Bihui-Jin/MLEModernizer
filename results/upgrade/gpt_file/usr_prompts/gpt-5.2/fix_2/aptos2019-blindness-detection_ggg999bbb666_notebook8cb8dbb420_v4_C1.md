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

0.8857740987904832

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
from glob import glob

import torch
import pandas as pd
from PIL import Image
from torchvision import transforms, models

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

try:
    os.makedirs("/kaggle/working", exist_ok=True)
    os.chdir("/kaggle/working")
except Exception:
    pass




## === cell 1
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        p = self.p.data.tolist()[0]
        return f"{self.__class__.__name__}(p={p:.4f}, eps={self.eps})"


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class SEResNet50Like(nn.Module):
    """
    Torchvision ResNet50 backbone with GeM pooling and a single output head.
    This is a compatibility replacement for se_resnet50 used in the original code,
    preserving the overall inference semantics (CNN -> pooling -> linear -> scalar).
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        self.avg_pool = GeM()
        self.last_linear = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.avg_pool(x)  # (B, 2048, 1, 1)
        x = torch.flatten(x, 1)  # (B, 2048)
        x = self.last_linear(x)  # (B, 1)
        return x


def get_se_resnet50_gem(pretrain):
    if pretrain == "imagenet":
        return SEResNet50Like(pretrained=True)
    return SEResNet50Like(pretrained=False)




## === cell 2
CANDIDATE_MODEL_PATHS = [
    "/kaggle/input/128best/128best.pth",
    "../input/128best/128best.pth",
    "/kaggle/input/aptos2019-blindness-detection/128best/128best.pth",
]
MODEL_PATH = next((p for p in CANDIDATE_MODEL_PATHS if os.path.exists(p)), None)

CANDIDATE_TEST_IMG_DIRS = [
    "/kaggle/input/aptos2019-blindness-detection/test_images",
    "/kaggle/data/aptos2019-blindness-detection/test_images",
    "../input/aptos2019-blindness-detection/test_images",
    "/kaggle/input/test_images",
]
TEST_IMAGE_PATH = next((p for p in CANDIDATE_TEST_IMG_DIRS if os.path.isdir(p)), None)

CANDIDATE_TEST_CSVS = [
    "/kaggle/input/aptos2019-blindness-detection/test.csv",
    "/kaggle/data/aptos2019-blindness-detection/test.csv",
    "../input/aptos2019-blindness-detection/test.csv",
    "/kaggle/input/test.csv",
]
TEST_CSV_PATH = next((p for p in CANDIDATE_TEST_CSVS if os.path.exists(p)), None)

if TEST_IMAGE_PATH is None:
    raise FileNotFoundError(
        "Could not find test_images directory in expected Kaggle input paths."
    )
if TEST_CSV_PATH is None:
    raise FileNotFoundError("Could not find test.csv in expected Kaggle input paths.")
if MODEL_PATH is None:
    raise FileNotFoundError(
        "Could not find the model checkpoint 128best.pth in expected Kaggle input paths."
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_56/465167295.py in <cell line: 0>()
     31 if MODEL_PATH is None:
     32     # The provided script expects a checkpoint; if missing, fail loudly rather than produce junk.
---> 33     raise FileNotFoundError(
     34         "Could not find the model checkpoint 128best.pth in expected Kaggle input paths."
     35     )

FileNotFoundError: Could not find the model checkpoint 128best.pth in expected Kaggle input paths.

## === cell 3
model = get_se_resnet50_gem(pretrain=None).to(device)

ckpt = torch.load(MODEL_PATH, map_location="cpu")

if (
    isinstance(ckpt, dict)
    and "state_dict" in ckpt
    and isinstance(ckpt["state_dict"], dict)
):
    state_dict = ckpt["state_dict"]
elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
    state_dict = ckpt
else:
    state_dict = ckpt

new_state = {}
for k, v in state_dict.items():
    nk = k
    if nk.startswith("module."):
        nk = nk[len("module.") :]
    if nk.startswith("model."):
        nk = nk[len("model.") :]
    if nk.startswith("backbone."):
        pass
    new_state[nk] = v

missing, unexpected = model.load_state_dict(new_state, strict=False)
model.eval()

len(missing), len(unexpected)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2473008784.py in <cell line: 0>()
      1 # Model build + checkpoint load with common-issues handling (map_location, key prefixes)
----> 2 model = get_se_resnet50_gem(pretrain=None).to(device)
      3 
      4 ckpt = torch.load(MODEL_PATH, map_location="cpu")
      5 

NameError: name 'device' is not defined

## === cell 4
infer_tfms = transforms.Compose(
    [
        transforms.Resize(
            (128, 128), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

test_df = pd.read_csv(TEST_CSV_PATH)
test_ids = test_df["id_code"].astype(str).tolist()

pred_map = {}
with torch.no_grad():
    for i, id_code in enumerate(test_ids):
        if i % 50 == 0:
            print(i, "/", len(test_ids))
        im_path = os.path.join(TEST_IMAGE_PATH, f"{id_code}.png")
        if not os.path.exists(im_path):
            candidates = glob(os.path.join(TEST_IMAGE_PATH, f"{id_code}.*"))
            if len(candidates) == 0:
                raise FileNotFoundError(f"Missing image for id_code={id_code}")
            im_path = candidates[0]

        image = Image.open(im_path).convert("RGB")
        x = infer_tfms(image).unsqueeze(0).to(device)
        output = model(x)
        pred_map[id_code] = float(output.item())

raw_pred = pd.DataFrame(
    {"id_code": test_ids, "diagnosis": [pred_map[i] for i in test_ids]}
)
raw_pred.to_csv("submission_raw_value.csv", index=False)
raw_pred.head()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2802187298.py in <cell line: 0>()
     29 
     30         image = Image.open(im_path).convert("RGB")
---> 31         x = infer_tfms(image).unsqueeze(0).to(device)
     32         output = model(x)
     33         pred_map[id_code] = float(output.item())

NameError: name 'device' is not defined

## === cell 5
submission = raw_pred.copy()

submission.loc[submission.diagnosis < 0.7, "diagnosis"] = 0
submission.loc[
    (0.7 <= submission.diagnosis) & (submission.diagnosis < 1.5), "diagnosis"
] = 1
submission.loc[
    (1.5 <= submission.diagnosis) & (submission.diagnosis < 2.5), "diagnosis"
] = 2
submission.loc[
    (2.5 <= submission.diagnosis) & (submission.diagnosis < 3.5), "diagnosis"
] = 3
submission.loc[3.5 <= submission.diagnosis, "diagnosis"] = 4

submission["diagnosis"] = submission["diagnosis"].astype(int)

submission = submission[["id_code", "diagnosis"]]
submission.to_csv("submission.csv", index=False)

submission.head()

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2782601405.py in <cell line: 0>()
      1 # Post-processing: keep the original thresholding logic unchanged.
----> 2 submission = raw_pred.copy()
      3 
      4 submission.loc[submission.diagnosis < 0.7, "diagnosis"] = 0
      5 submission.loc[

NameError: name 'raw_pred' is not defined
