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

0.8536147975702988

# 6. Current score

-0.06045

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00254) has done: 'I remove the failing dependency on a missing `pretrainedmodels` package and instead use the equivalent `torchvision` ResNet50 backbone, keeping the same GeM pooling and single-output regression head. I also fix file/path issues by using the provided competition directory under `/kaggle/input/aptos2019-blindness-detection/` and ensure predictions are generated in the exact `test.csv` order (not filesystem glob order), which prevents submission misalignment. Since the provided external weights file (`../input/mymodels/model15.pth`) is not present in your environment, I make the checkpoint loading optional (safe fallback) so the notebook always produces a valid `submission.csv`. Finally, I keep your original thresholding logic (score-neutral vs your intent) and ensure the output CSV has the required columns and `.csv` suffix.'
- What this solution (achieved 0.0) has done: 'Your current 0.00254 score is consistent with running a randomly initialized ResNet50 because the intended checkpoint isn’t available; the smallest legitimate step toward the 0.8536 target is to use ImageNet pretrained weights (same architecture/head/pooling) so the model produces non-random features. I keep your GeM + single-output regression head and the exact same thresholding logic, only switching `pretrain=None` to `pretrain="imagenet"` and making the test-time preprocessing match standard ResNet inference a bit better by using a center crop after resize (still deterministic, no TTA). This should move the score substantially upward without changing the core approach or training loop (there is none). The script still write `submission.csv` in the required format and order.'
- What this solution (achieved -0.06045) has done: 'Your current 0.0 score is consistent with a submission-format or row-alignment problem rather than model quality, so the smallest change toward the target is to guarantee the `submission.csv` exactly matches `sample_submission.csv` row order and schema. I also make inference deterministic and robust by forcing CPU/GPU eval settings and ensuring we never reorder or drop any `id_code`. Finally, I clip/round predictions defensively to valid class integers 0–4 after your existing thresholding, without changing your model or preprocessing, so Kaggle always receives valid labels.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
from glob import glob

import torch
import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision import models

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_DIR = "/kaggle/input/aptos2019-blindness-detection"
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_IMAGE_PATH = os.path.join(DATA_DIR, "test_images")

MODEL_PATH = "/kaggle/input/mymodels/model15.pth"  # may not exist in this environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("device:", device)
print("DATA_DIR exists:", os.path.exists(DATA_DIR))
print("TEST_IMAGE_PATH exists:", os.path.exists(TEST_IMAGE_PATH))
print("MODEL_PATH exists:", os.path.exists(MODEL_PATH))
print("TEST_CSV exists:", os.path.exists(TEST_CSV))
print("SAMPLE_SUB_CSV exists:", os.path.exists(SAMPLE_SUB_CSV))




## === cell 1
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


def get_resnet50_gem(pretrain=None):
    if pretrain == "imagenet":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
    else:
        weights = None

    model = models.resnet50(weights=weights)
    model.avgpool = GeM()
    model.fc = nn.Linear(2048, 1)
    return model


model = get_resnet50_gem(pretrain="imagenet")
model.to(device)

if os.path.exists(MODEL_PATH):
    try:
        state = torch.load(MODEL_PATH, map_location=device)
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        if isinstance(state, dict):
            new_state = {}
            for k, v in state.items():
                nk = k[7:] if k.startswith("module.") else k
                new_state[nk] = v
            state = new_state
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Loaded MODEL_PATH with strict=False")
        print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
    except Exception as e:
        print("Warning: failed to load checkpoint:", repr(e))
        print("Proceeding with ImageNet-pretrained model.")
else:
    print("Checkpoint not found; using ImageNet-pretrained model.")

model.eval()



## === cell 2
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
assert list(sample_sub.columns) == ["id_code", "diagnosis"]
test_df = pd.read_csv(TEST_CSV)
assert "id_code" in test_df.columns

id_list = sample_sub["id_code"].tolist()

test_ids = set(test_df["id_code"].tolist())
missing_in_test = [x for x in id_list if x not in test_ids]
if len(missing_in_test) > 0:
    raise ValueError(
        f"Some sample_submission ids are missing in test.csv, example: {missing_in_test[:5]}"
    )

tfm = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

predictions = []
with torch.no_grad():
    for i, id_code in enumerate(id_list):
        if i % 50 == 0:
            print(i, "/", len(id_list))
        im_path = os.path.join(TEST_IMAGE_PATH, f"{id_code}.png")
        if not os.path.exists(im_path):
            raise FileNotFoundError(f"Missing test image: {im_path}")
        image = Image.open(im_path).convert("RGB")
        image = tfm(image).to(device)
        output = model(image.unsqueeze(0))
        predictions.append(float(output.item()))

raw_df = pd.DataFrame({"id_code": id_list, "diagnosis": predictions})
raw_df.to_csv("submission_raw_value.csv", index=False)
raw_df.head()



## === cell 3
submission = pd.DataFrame({"id_code": id_list, "diagnosis": predictions})

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

submission["diagnosis"] = submission["diagnosis"].astype(int).clip(0, 4)

submission = submission[["id_code", "diagnosis"]]
submission = sample_sub[["id_code"]].merge(submission, on="id_code", how="left")
if submission["diagnosis"].isna().any():
    raise ValueError(
        "NaNs found in diagnosis after merge; would create invalid submission."
    )
submission["diagnosis"] = submission["diagnosis"].astype(int)

submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("Unique labels:", sorted(submission["diagnosis"].unique().tolist()))
