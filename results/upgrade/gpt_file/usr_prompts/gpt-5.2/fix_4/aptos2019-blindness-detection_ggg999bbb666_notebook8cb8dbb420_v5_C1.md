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

0.8836900126751313

# 6. Current score

-0.01193

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.021) has done: 'I remove the failing dependency-install cell that points to a non-existent `../input/pretrainmodels` directory and instead use a locally available model backbone from `torchvision` so the notebook runs in the Kaggle environment without extra packages. To preserve the core inference logic (single-image loop, 224 resize, regression head producing a scalar), I keep the same GeM pooling + linear regression head structure and load weights if possible; if the provided checkpoint is missing or incompatible, the code fail loudly with a clear message rather than silently producing nonsense. I also fix path handling to use the provided `/kaggle/input/aptos2019-blindness-detection/...` structure, ensure deterministic ordering by predicting in `test.csv` order (not glob order), and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'I fix the runtime failure by making the checkpoint optional: if `/kaggle/input/224best/224best.pth` isn’t available, the code still run end-to-end and create a valid `submission.csv` instead of crashing. To avoid the extremely poor score caused by using an untrained random model, I add a safe fallback that outputs the dataset’s majority class (0) when no checkpoint is found; this should move the score upward toward a reasonable baseline without changing the intended model/inference logic when the checkpoint is present. I also add ImageNet normalization (a standard requirement for ResNet) to improve predictions when a real checkpoint or ImageNet weights are used, while keeping the architecture and loop intact. Finally, I keep prediction order aligned to `test.csv` and always write both raw and final submission CSVs.'
- What this solution (achieved -0.01193) has done: 'Your current 0.0 score is coming from the “no checkpoint → predict constant class” fallback, which typically yields a near-zero quadratic weighted kappa. To move toward the 0.8837 target with minimal change and without altering the model/training setup, I replace the constant-class fallback with a deterministic, label-free heuristic that uses each test image’s mean brightness to assign one of the 5 classes (still producing a valid submission when no checkpoint exists). When a checkpoint is present, the model inference path and the existing regression→threshold mapping remain unchanged. This should materially increase score from 0.0 while keeping changes small and runtime within limits.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision import transforms, models

ImageFile.LOAD_TRUNCATED_IMAGES = True

INPUT_DIR = Path("/kaggle/input/aptos2019-blindness-detection")
TEST_CSV = INPUT_DIR / "test.csv"
TRAIN_CSV = INPUT_DIR / "train.csv"
TEST_IMAGE_DIR = INPUT_DIR / "test_images"
MODEL_PATH = Path(
    "/kaggle/input/224best/224best.pth"
)  # keep user's intended model path

WORKING_DIR = Path("/kaggle/working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("Device:", device)
print("TEST_IMAGE_DIR exists:", TEST_IMAGE_DIR.exists())
print("MODEL_PATH exists:", MODEL_PATH.exists())
print("TEST_CSV exists:", TEST_CSV.exists())
print("TRAIN_CSV exists:", TRAIN_CSV.exists())




## === cell 1
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    @staticmethod
    def gem(x, p=3.0, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        p = self.p.data.tolist()[0]
        return f"{self.__class__.__name__}(p={p:.4f}, eps={self.eps})"


class ResNet50_GeM_Regressor(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        self.backbone.avgpool = GeM()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x):
        return self.backbone(x)


model = ResNet50_GeM_Regressor(pretrained=False).to(device)
model.eval()

has_checkpoint = MODEL_PATH.exists()
if has_checkpoint:
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = ckpt
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state_dict = ckpt[k]
                break

    if isinstance(state_dict, dict) and any(
        key.startswith("module.") for key in state_dict.keys()
    ):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Loaded checkpoint from:", MODEL_PATH)
    print("Missing keys (show up to 10):", missing[:10])
    print("Unexpected keys (show up to 10):", unexpected[:10])
else:
    print(
        f"WARNING: Checkpoint not found at {MODEL_PATH}. "
        "Will generate a valid submission using a deterministic image-statistics fallback."
    )




## === cell 2
test_df = pd.read_csv(TEST_CSV)
assert "id_code" in test_df.columns
print("Test rows:", len(test_df))

tfm = transforms.Compose(
    [
        transforms.Resize(
            (224, 224), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def brightness_based_class(img_rgb: Image.Image) -> int:
    g = img_rgb.convert("L")
    g = g.resize((64, 64), resample=Image.Resampling.BILINEAR)
    arr = np.asarray(g, dtype=np.float32)
    m = float(arr.mean())

    if m >= 140.0:
        return 0
    elif m >= 120.0:
        return 1
    elif m >= 105.0:
        return 2
    elif m >= 90.0:
        return 3
    else:
        return 4


pred_rows = []
model.eval()

with torch.no_grad():
    for i, id_code in enumerate(test_df["id_code"].tolist()):
        if i % 50 == 0:
            print(f"{i}/{len(test_df)}")
        img_path = TEST_IMAGE_DIR / f"{id_code}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing test image: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if has_checkpoint:
            x = tfm(image).unsqueeze(0).to(device)
            y = model(x)
            pred_rows.append((id_code, float(y.item())))
        else:
            pred_rows.append((id_code, float(brightness_based_class(image))))

raw_sub = pd.DataFrame(pred_rows, columns=["id_code", "diagnosis"])
raw_path = WORKING_DIR / "submission_raw_value.csv"
raw_sub.to_csv(raw_path, index=False)
print("Wrote:", raw_path)
print(raw_sub.head())




## === cell 3
submission = raw_sub.copy()

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

out_path = WORKING_DIR / "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())
print("Class counts:\n", submission["diagnosis"].value_counts().sort_index())
