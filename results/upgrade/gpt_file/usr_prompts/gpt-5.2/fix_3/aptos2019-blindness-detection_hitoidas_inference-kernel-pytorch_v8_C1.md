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

3.7

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.8674232603201573

# 6. Current score

0.02059

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.02059) has done: 'You’re failing early because the script expects a pretrained `model_aptos.pth` that isn’t present in the dataset, so `torch.load` raises `FileNotFoundError`. Even if it existed, the loaded object can be a full fastai `Learner` (or something not a `nn.Module`), which caused the `'function' object has no attribute 'to'` error when you pass it into a new `Learner`. I fix this by (1) robustly locating the dataset under `/kaggle/input/...`, (2) loading a model only if the file exists and extracting `.model` if a `Learner` was saved, otherwise falling back to a standard torchvision classifier so inference can run end-to-end, and (3) bypassing fastai’s training callbacks by doing plain PyTorch inference and then writing a valid `submission.csv`. These changes are minimal and aimed at producing a valid submission reliably (score is unknown currently because no submission was generated).'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

CAND_INPUT_ROOTS = ["../input", "/kaggle/input", "/kaggle/data/input"]
INPUT_ROOT = next((p for p in CAND_INPUT_ROOTS if os.path.isdir(p)), "../input")

COMP_SUBDIR = "aptos2019-blindness-detection"
CAND_COMP_DIRS = [
    os.path.join(INPUT_ROOT, COMP_SUBDIR),
    os.path.join(INPUT_ROOT, "kaggle", "data", COMP_SUBDIR),
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/data/input/aptos2019-blindness-detection",
]
COMP_DIR = next(
    (p for p in CAND_COMP_DIRS if os.path.isdir(p)),
    os.path.join(INPUT_ROOT, COMP_SUBDIR),
)

print("Using INPUT_ROOT:", INPUT_ROOT)
print("Using COMP_DIR:", COMP_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

MODEL_PATH_CANDS = [
    os.path.join(COMP_DIR, "model_aptos.pth"),
    os.path.join(INPUT_ROOT, "model_aptos.pth"),
    os.path.join(COMP_DIR, "model.pth"),
    os.path.join(INPUT_ROOT, "model.pth"),
]
MODEL_PATH = next((p for p in MODEL_PATH_CANDS if os.path.exists(p)), None)
print(
    (
        "MODEL_PATH found:"
        if MODEL_PATH
        else "MODEL_PATH not found; will use torchvision fallback."
    ),
    MODEL_PATH,
)


def _build_fallback_model(num_classes=5):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


mod = None
if MODEL_PATH is not None:
    obj = torch.load(MODEL_PATH, map_location="cpu")
    if hasattr(obj, "model") and isinstance(getattr(obj, "model"), nn.Module):
        mod = obj.model
        print("Loaded a fastai Learner; using its .model for inference.")
    elif isinstance(obj, nn.Module):
        mod = obj
        print("Loaded a torch nn.Module for inference.")
    elif isinstance(obj, dict) and any(
        k.endswith("state_dict") or k == "state_dict" for k in obj.keys()
    ):
        mod = _build_fallback_model(num_classes=5)
        sd = obj.get("state_dict", None)
        if sd is None:
            for k in obj.keys():
                if k.endswith("state_dict"):
                    sd = obj[k]
                    break
        sd2 = (
            {k.replace("module.", ""): v for k, v in sd.items()}
            if sd is not None
            else {}
        )
        missing, unexpected = mod.load_state_dict(sd2, strict=False)
        print(
            "Loaded state_dict checkpoint. Missing keys:",
            len(missing),
            "Unexpected keys:",
            len(unexpected),
        )
    else:
        print("Unrecognized model object type:", type(obj), "-> using fallback model.")
        mod = _build_fallback_model(num_classes=5)
else:
    mod = _build_fallback_model(num_classes=5)

mod = mod.to(device)
mod.eval()

if torch.cuda.is_available():
    torch.cuda.empty_cache()



## === cell 1
test_csv = os.path.join(COMP_DIR, "test.csv")
test_img_dir = os.path.join(COMP_DIR, "test_images")

CAND_TEST_DIRS = [
    test_img_dir,
    os.path.join(COMP_DIR, "test_images", "test_images"),
    os.path.join(INPUT_ROOT, "test_images"),
    os.path.join(INPUT_ROOT, "test_images", "test_images"),
]
test_img_dir = next((p for p in CAND_TEST_DIRS if os.path.isdir(p)), test_img_dir)

dff = pd.read_csv(test_csv)
assert "id_code" in dff.columns, "test.csv must contain id_code"
dff["file"] = dff["id_code"].astype(str) + ".png"

print("Test rows:", len(dff))
print("Using test image dir:", test_img_dir)

tfm = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@torch.no_grad()
def predict_batch(file_list):
    imgs = []
    for fn in file_list:
        fp = os.path.join(test_img_dir, fn)
        try:
            im = Image.open(fp).convert("RGB")
        except Exception:
            im = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        imgs.append(tfm(im))
    xb = torch.stack(imgs, dim=0).to(device)
    out = mod(xb)
    if out.ndim != 2 or out.shape[1] != 5:
        raise RuntimeError(
            f"Model output has unexpected shape {tuple(out.shape)}; expected (bs,5)."
        )
    return out.argmax(dim=1).detach().cpu().numpy().astype(int)


bs = 32
labels = []
files = dff["file"].tolist()
for i in range(0, len(files), bs):
    labels.extend(predict_batch(files[i : i + bs]).tolist())

print("Generated predictions:", len(labels), "for test rows:", len(dff))
assert len(labels) == len(dff)



## === cell 2
submit = pd.DataFrame({"id_code": dff["id_code"].tolist(), "diagnosis": labels})
submit = submit[["id_code", "diagnosis"]]
submit["diagnosis"] = submit["diagnosis"].astype(int)

out_path = "./submission.csv"
submit.to_csv(out_path, index=False)

print("Wrote submission to:", out_path)
print(submit.head())
print("Submission shape:", submit.shape)
print(
    "Diagnosis value counts:\n",
    submit["diagnosis"].value_counts(dropna=False).sort_index(),
)
