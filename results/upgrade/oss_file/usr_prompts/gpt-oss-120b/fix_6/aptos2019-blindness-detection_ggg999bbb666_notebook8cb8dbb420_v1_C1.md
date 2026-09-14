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

None

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I add a quick calibration step that learns a linear mapping from the image‑wise mean pixel intensity (the same feature you already use) to the true diagnosis on the training set. This small regression replaces the fixed “*mean × 4*” scaling, keeps the overall pipeline unchanged, and should move the quadratic weighted‑kappa score closer to the target. I also renumber the cells to start at 1 and keep the final CSV generation unchanged.'
- What this solution (achieved 0.14498) has done: 'I replace the simple linear regression with a GradientBoostingRegressor, which can capture the non‑linear relationship between mean pixel intensity and the DR severity while keeping the same single‑feature pipeline. The rest of the code (image processing, thresholding and CSV writing) stays unchanged, so the script still runs end‑to‑end and produces a valid `submission.csv`. This modest model upgrade is expected to raise the quadratic weighted‑kappa score from 0 toward the target 0.85.'

# 9. Code solution

## === cell 0
import os
from glob import glob

import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision import transforms, models

from sklearn.ensemble import GradientBoostingRegressor

ImageFile.LOAD_TRUNCATED_IMAGES = True

TEST_CSV_PATH = "../input/aptos2019-blindness-detection/test.csv"
TEST_IMAGE_PATH = "../input/aptos2019-blindness-detection/test_images"
TRAIN_CSV_PATH = "../input/aptos2019-blindness-detection/train.csv"
TRAIN_IMAGE_PATH = "../input/aptos2019-blindness-detection/train_images"

test_df = pd.read_csv(TEST_CSV_PATH)
id_codes = test_df["id_code"].tolist()




## === cell 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
except Exception:
    backbone = None

if backbone is not None:
    backbone = torch.nn.Sequential(*list(backbone.children())[:-1])  # remove final FC
    backbone.eval()
    backbone.to(device)
    torch.set_grad_enabled(False)

resnet_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

simple_transform = transforms.ToTensor()  # returns C×H×W in [0,1]


def extract_features(image_path: str) -> np.ndarray:
    """Return a 512‑dim feature vector (ResNet) or a single mean intensity as fallback."""
    if not os.path.isfile(image_path):
        return (
            np.zeros(512, dtype=np.float32)
            if backbone is not None
            else np.array([0.0], dtype=np.float32)
        )

    img = Image.open(image_path).convert("RGB")
    if backbone is not None:
        inp = resnet_transform(img).unsqueeze(0).to(device)  # 1×3×224×224
        with torch.no_grad():
            feats = backbone(inp)  # 1×512×1×1
        return feats.squeeze().cpu().numpy()
    else:
        inp = simple_transform(img)
        return np.array([float(inp.mean().item())], dtype=np.float32)




## === cell 2
train_df = pd.read_csv(TRAIN_CSV_PATH)

train_features = []
train_labels = []

print("Extracting features for training images...")
for idx, row in train_df.iterrows():
    img_path = os.path.join(TRAIN_IMAGE_PATH, f"{row['id_code']}.png")
    feats = extract_features(img_path)
    train_features.append(feats)
    train_labels.append(row["diagnosis"])

train_features_np = np.stack(train_features)  # shape (N, D)
train_labels_np = np.array(train_labels)

regressor = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)
regressor.fit(train_features_np, train_labels_np)
print("Regressor trained on extracted features.")




## === cell 3
test_features = []

print("Extracting features for test images...")
for idx, id_code in enumerate(id_codes):
    img_path = os.path.join(TEST_IMAGE_PATH, f"{id_code}.png")
    feats = extract_features(img_path)
    test_features.append(feats)
    if (idx + 1) % 50 == 0 or (idx + 1) == len(id_codes):
        print(f"Processed {idx + 1}/{len(id_codes)}: {id_code}")

test_features_np = np.stack(test_features)

raw_preds = regressor.predict(test_features_np)
raw_preds = np.clip(raw_preds, 0.0, 4.0)

submission = pd.DataFrame({"id_code": id_codes, "diagnosis": raw_preds})
submission.to_csv("submission_raw_value.csv", index=False)

submission["diagnosis"] = np.rint(submission["diagnosis"]).astype(int)
submission.to_csv("submission.csv", index=False)

print("Submission files saved: submission_raw_value.csv, submission.csv")
