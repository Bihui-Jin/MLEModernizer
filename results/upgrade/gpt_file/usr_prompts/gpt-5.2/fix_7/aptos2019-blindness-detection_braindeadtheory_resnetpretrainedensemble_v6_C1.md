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

3.8

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tqdm==4.67.1

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

0.872404072348969

# 6. Current score

0.00088

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.01027) has done: 'The immediate runtime failure is caused by trying to load a non-existent checkpoint from `../input/resnet/FinalResnet01.pt`. I fix this by making checkpoint loading robust: search for the file under `/kaggle/input` and, if it truly doesn’t exist, fall back to using an ImageNet-pretrained ResNet101 (same architecture) so you can still generate a valid submission and substantially improve score from random. I also fix the progress-bar import (`tqdm._tqdm_notebook` is unreliable in Kaggle script mode) and add `model.eval()`/`torch.no_grad()` handling plus safe RGB conversion to avoid occasional PIL mode issues. The output always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'I fix the torchvision `resnet101(weights=..., num_classes=5)` misuse by loading ImageNet weights and then replacing the final `fc` layer to output 5 classes, which keeps the intended fallback behavior but avoids the ValueError. I also fix the CUDA/CPU mismatch by ensuring the model is moved to the same device after *all* weight-loading/model-rebuilding branches. Finally, I make inference robust (non-blocking where safe, correct dtype) and guarantee `submission.csv` is written in the required format and row order.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with using an ImageNet-pretrained ResNet101 with a randomly initialized 5-class head (because the checkpoint isn’t found), which yields essentially random predictions. To move the score upward toward the 0.872 target while preserving your exact inference pipeline, I make checkpoint discovery robust by also searching common Kaggle dataset folders (including `../input/**`) and accepting more filename variants, so the intended trained weights are actually loaded when present. I also add a safe “strip common prefixes” load to handle `module.` / `model.` key mismatches without changing model logic. If no checkpoint is found, the fallback behavior remains the same, but when the checkpoint exists this should materially increase the score from 0.0.'
- What this solution (achieved 0.00088) has done: 'Your 0.0 score is most consistent with the fallback branch being used (checkpoint not found), which leaves the 5-class head randomly initialized and produces near-random predictions. I make checkpoint loading more robust by (1) searching additional common filename variants and (2) allowing non-strict loading while still verifying that key tensors (especially `fc.weight`/`fc.bias`) actually load with matching shapes; this keeps your ResNet101 core logic intact but prevents silent “random head” behavior. If no compatible checkpoint is found, I keep your current ImageNet fallback, but I also default predictions to the argmax class (instead of softmax expected-value rounding) because expected-value rounding tends to collapse toward middle classes and can further hurt kappa under a random/untrained head. These are minimal inference-time changes intended to move the score up toward the target without changing the architecture or introducing new training.'
- What this solution (achieved 0.00088) has done: 'Your score (0.00088) strongly suggests you’re still running the fallback ImageNet ResNet101 with a randomly initialized 5-class head, i.e., the intended trained checkpoint is not being loaded. I make the checkpoint search/load more robust without changing the model architecture or inference semantics: (1) search for any `.pt/.pth` files containing `resnet` under `/kaggle/input` and `../input`, (2) support common checkpoint layouts like `{'model': ...}` and `{'model_state_dict': ...}`, and (3) if `fc.*` is missing/mismatched, attempt a safe “head-only remap” from common names like `classifier.*`/`head.*` to `fc.*` so the trained 5-class head actually loads when present. If no compatible checkpoint exists, behavior remains the same as your current fallback (so it won’t break), but when a proper checkpoint is available this should move the score substantially upward toward your target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import glob
import time
import copy
import gc

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
transform = transforms.Compose(
    [
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.460, 0.247, 0.080], [0.249, 0.138, 0.081]),
    ]
)


class APTOSDataset(Dataset):
    """Eye images dataset."""

    def __init__(self, csv_file, filetype, transform=None):
        self.eye_frame = pd.read_csv(csv_file)
        self.filetype = filetype
        self.transform = transform

    def __len__(self):
        return len(self.eye_frame)

    def __getitem__(self, idx):
        if self.filetype == "train":
            img_name = os.path.join(
                "../input/aptos2019-blindness-detection/train_images",
                self.eye_frame.loc[idx, "id_code"] + ".png",
            )
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, int(self.eye_frame.loc[idx, "diagnosis"])
        else:
            img_name = os.path.join(
                "../input/aptos2019-blindness-detection/test_images",
                self.eye_frame.loc[idx, "id_code"] + ".png",
            )
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, self.eye_frame.loc[idx, "id_code"]




## === cell 2
test_dataset = APTOSDataset(
    csv_file="../input/aptos2019-blindness-detection/test.csv",
    filetype="test",
    transform=transform,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=24, shuffle=False, num_workers=4, pin_memory=True
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torchvision.models.resnet101(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)


def _find_checkpoint():
    patterns = [
        "/kaggle/input/**/FinalResnet01.pt",
        "/kaggle/input/**/finalresnet01.pt",
        "/kaggle/input/**/FinalResNet01.pt",
        "/kaggle/input/**/finalResnet01.pt",
        "/kaggle/input/**/FinalResnet*.pt",
        "/kaggle/input/**/finalresnet*.pt",
        "/kaggle/input/**/resnet*.pt",
        "../input/**/FinalResnet01.pt",
        "../input/**/finalresnet01.pt",
        "../input/**/FinalResNet01.pt",
        "../input/**/finalResnet01.pt",
        "../input/**/FinalResnet*.pt",
        "../input/**/finalresnet*.pt",
        "../input/**/resnet*.pt",
        "/kaggle/input/**/*resnet*.pth",
        "/kaggle/input/**/*resnet*.pt",
        "../input/**/*resnet*.pth",
        "../input/**/*resnet*.pt",
    ]
    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(p, recursive=True))

    def rank(path: str) -> int:
        base = os.path.basename(path).lower()
        if base == "finalresnet01.pt":
            return 0
        if base == "finalresnet01.pth":
            return 0
        if "finalresnet" in base:
            return 1
        if "resnet" in base:
            return 2
        return 3

    candidates = sorted(
        list(dict.fromkeys(candidates)), key=lambda x: (rank(x), len(x))
    )
    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            return c
    return None


def _clean_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if len(keys) == 0:
        return state_dict

    def strip_prefix(sd, prefix):
        return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}

    for prefix in ("module.", "model.", "net."):
        if sum(k.startswith(prefix) for k in keys) >= max(1, int(0.8 * len(keys))):
            stripped = strip_prefix(state_dict, prefix)
            if len(stripped) > 0:
                return stripped
    return state_dict


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _remap_common_head_to_fc(state, model_state):
    if not isinstance(state, dict):
        return state
    if "fc.weight" in state and "fc.bias" in state:
        return state

    fcw_shape = model_state["fc.weight"].shape
    fcb_shape = model_state["fc.bias"].shape

    candidates_w = [
        "classifier.weight",
        "head.weight",
        "last_linear.weight",
        "fc1.weight",
    ]
    candidates_b = [
        "classifier.bias",
        "head.bias",
        "last_linear.bias",
        "fc1.bias",
    ]

    found_w, found_b = None, None
    for k in candidates_w:
        if (
            k in state
            and hasattr(state[k], "shape")
            and tuple(state[k].shape) == tuple(fcw_shape)
        ):
            found_w = k
            break
    for k in candidates_b:
        if (
            k in state
            and hasattr(state[k], "shape")
            and tuple(state[k].shape) == tuple(fcb_shape)
        ):
            found_b = k
            break

    if found_w is not None and found_b is not None:
        new_state = dict(state)
        new_state["fc.weight"] = new_state[found_w]
        new_state["fc.bias"] = new_state[found_b]
        return new_state
    return state


def _try_load_checkpoint_into_model(model, ckpt_path: str) -> bool:
    state = torch.load(ckpt_path, map_location="cpu")
    state = _extract_state_dict(state)
    state = _clean_state_dict_keys(state)
    if not isinstance(state, dict) or len(state) == 0:
        return False

    model_state = model.state_dict()

    state = _remap_common_head_to_fc(state, model_state)

    filtered = {
        k: v
        for k, v in state.items()
        if (
            k in model_state and hasattr(v, "shape") and v.shape == model_state[k].shape
        )
    }
    if len(filtered) == 0:
        return False

    model.load_state_dict(filtered, strict=False)

    fcw_ok = ("fc.weight" in filtered) and (
        filtered["fc.weight"].shape == model_state["fc.weight"].shape
    )
    fcb_ok = ("fc.bias" in filtered) and (
        filtered["fc.bias"].shape == model_state["fc.bias"].shape
    )
    return bool(fcw_ok and fcb_ok)


ckpt_path = _find_checkpoint()
loaded_ckpt = False

if ckpt_path is not None and os.path.exists(ckpt_path):
    loaded_ckpt = _try_load_checkpoint_into_model(model, ckpt_path)

if not loaded_ckpt:
    weights = torchvision.models.ResNet101_Weights.DEFAULT
    model = torchvision.models.resnet101(weights=weights)  # 1000-class head
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)  # replace head for 5 classes

model = model.to(device)



## === cell 3
from tqdm.auto import tqdm


def compute_predictions(model, model_type, data_loader, device):
    model.eval()
    if model_type == "train":
        predictions = []
        correct_pred, num_examples = 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, total=len(data_loader)):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.append(preds.detach().cpu())
                num_examples += labels.size(0)
                correct_pred += (preds == labels).sum()
        return predictions, correct_pred.item() / num_examples * 100
    else:
        predictions = []
        img_ids = []

        with torch.no_grad():
            for inputs, img_id in tqdm(data_loader, total=len(data_loader)):
                inputs = inputs.to(device, non_blocking=True)
                logits = model(inputs)

                preds = torch.argmax(logits, dim=1).to(torch.int64)

                predictions.extend(preds.detach().cpu().tolist())
                img_ids.extend(list(img_id))

        final_predictions = pd.DataFrame({"id_code": img_ids, "diagnosis": predictions})

        test_ids = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")[
            "id_code"
        ].tolist()
        final_predictions = (
            final_predictions.set_index("id_code").loc[test_ids].reset_index()
        )

        final_predictions.to_csv("submission.csv", index=False)
        return final_predictions




## === cell 4
print("Computing Test Predictions")
test_predictions_df = compute_predictions(model, "test", test_loader, device)

assert isinstance(test_predictions_df, pd.DataFrame)
assert list(test_predictions_df.columns) == ["id_code", "diagnosis"]
assert test_predictions_df.shape[0] == len(
    pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
)
print(test_predictions_df.head())
print("Wrote submission.csv")
