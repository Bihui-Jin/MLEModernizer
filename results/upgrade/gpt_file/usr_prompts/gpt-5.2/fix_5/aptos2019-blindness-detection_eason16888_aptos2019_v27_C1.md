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

3.10

# 3. Installed packages

No external packages required in the script and installed.

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

0.489960185339643

# 6. Current score

0.14153

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.09891) has done: 'I remove the failing `pip install tensorflow-addons`/`tensorflow_addons` dependency that’s crashing the notebook, and consolidate imports so `tf`, `pd`, etc. are always defined before later cells run. I also fix the image preprocessing at inference time to match the existing `preprocessing()` logic (including proper normalization), which is a minimal, metric-aligned correction and should improve the kappa score versus feeding unnormalized pixels. Finally, I make the weights loading robust: if the external weights file isn’t available in this environment, the script fall back to building the model and still produce a valid `submission.csv` (but with worse score), rather than erroring out.'
- What this solution (achieved 0.0) has done: 'The immediate blocker is the TensorFlow import crash (`MessageFactory.GetPrototype`), which is a known protobuf/TensorFlow incompatibility in some Kaggle images; we fix it by forcing the pure-Python protobuf implementation before importing TensorFlow. Next, your current weights path points to a dataset that doesn’t exist here, so you are effectively predicting with random weights (explaining the negative kappa); we fix this by using EfficientNetB0 ImageNet weights (same architecture) as a stable, score-improving default while still loading external weights if present. Finally, we keep your preprocessing and prediction loop logic intact, but make I/O paths robust to both `/kaggle/input/...` and `../input/...` layouts and ensure we always write a valid `submission.csv` with the correct columns.'
- What this solution (achieved 0.14153) has done: 'I fix the TensorFlow import crash by avoiding TensorFlow entirely (it’s not usable in this environment due to the protobuf `MessageFactory.GetPrototype` error) while keeping the same core “EfficientNetB0 + softmax(5)” inference intent by switching to a built-in, Kaggle-safe PyTorch EfficientNet-B0 ImageNet backbone. I also fix the submission merge bug that creates `diagnosis_x/diagnosis_y` columns (causing the `KeyError: 'diagnosis'`) by directly constructing the submission from `test.csv` (or by overwriting the column). Finally, I keep your preprocessing semantics (Ben Graham-style crop/blur/unsharp + resize + normalization) and ensure we always write a valid `submission.csv` with `id_code,diagnosis`.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torchvision

np.random.seed(42)
torch.manual_seed(42)



## === cell 1
"""
    Config
"""
IMG_SIZE = 224
BATCH_SIZE = 16


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def load_ben_color(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


"""
    Preprocessing for ImageDataGenerator since ImageDataGenerator reads images in rgb mode, while opencv in bgr
"""


def preprocessing(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0




## === cell 2
class EffNetB0_5(nn.Module):
    def __init__(self):
        super().__init__()
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = torchvision.models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 5)

    def forward(self, x):
        return self.backbone(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EffNetB0_5().to(device)
model.eval()



## === cell 3
WEIGHTS_CANDIDATES = [
    "../input/none-eff-b0-model/none_eff_b0_model.pth",
    "/kaggle/input/none-eff-b0-model/none_eff_b0_model.pth",
]

loaded = False
for wp in WEIGHTS_CANDIDATES:
    if os.path.exists(wp):
        state = torch.load(wp, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded weights from: {wp}")
        loaded = True
        break

if not loaded:
    print(
        "WARNING: External PyTorch weights file not found. Proceeding with ImageNet EfficientNetB0 backbone and randomly initialized 5-class head."
    )



## === cell 4
with torch.no_grad():
    _ = model(
        torch.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float32, device=device)
    )
print("Model ready for inference.")



## === cell 5
INPUT_ROOTS = [
    "../input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/input",  # fallback if files are directly under /kaggle/input
    "/kaggle/data",  # fallback if files are directly under /kaggle/data
]
DATA_ROOT = None
for r in INPUT_ROOTS:
    cand = r
    if os.path.isdir(os.path.join(r, "aptos2019-blindness-detection")):
        cand = os.path.join(r, "aptos2019-blindness-detection")
    if os.path.exists(os.path.join(cand, "test.csv")) and (
        os.path.isdir(os.path.join(cand, "test_images"))
        or os.path.exists(os.path.join(cand, "test_images.zip"))
    ):
        DATA_ROOT = cand
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find aptos2019-blindness-detection dataset folder/files in expected locations."
    )

test_csv_path = os.path.join(DATA_ROOT, "test.csv")
test_img_dir = os.path.join(DATA_ROOT, "test_images")

test_df = pd.read_csv(test_csv_path)
id_code = test_df["id_code"].values

test_prediction = np.empty(len(id_code), dtype=np.int64)

imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

batch_tensors = []
batch_indices = []

for i in range(len(id_code)):
    img_path = os.path.join(test_img_dir, f"{id_code[i]}.png")
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessing(img)  # float32 in [0,1], shape HWC
    img = (img - imagenet_mean) / imagenet_std
    img = np.transpose(img, (2, 0, 1))  # CHW
    batch_tensors.append(torch.from_numpy(img).float())
    batch_indices.append(i)

    if len(batch_tensors) == BATCH_SIZE or (i == len(id_code) - 1):
        X = torch.stack(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            logits = model(X)
            cls = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
        for bi, ci in zip(batch_indices, cls):
            test_prediction[bi] = int(ci)

        batch_tensors.clear()
        batch_indices.clear()
        del X, logits
        gc.collect()

sub_df = pd.DataFrame(
    {"id_code": id_code, "diagnosis": test_prediction.astype("int64")}
)

out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)

unique, counts = np.unique(test_prediction, return_counts=True)
tmp = dict(zip(unique.tolist(), counts.tolist()))
print(tmp)
print(
    f"Done! Wrote {out_path} with columns: {sub_df.columns.tolist()} and shape {sub_df.shape}"
)
