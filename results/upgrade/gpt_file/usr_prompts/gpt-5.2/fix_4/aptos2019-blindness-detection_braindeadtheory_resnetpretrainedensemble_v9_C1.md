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

0.8791714649727451

# 6. Current score

0.01829

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.08307) has done: 'I fix the crash by removing the hardcoded dependency on a missing `../input/resnet152/FinalResnet152_0.pt` file and instead load torchvision’s built-in ResNet-152 ImageNet weights (available in the Kaggle environment). I also make the tqdm import compatible with scripts (the notebook-specific tqdm module can break) and ensure images are always converted to RGB to avoid occasional channel-related errors. Finally, I keep your inference logic intact (argmax over 5 logits) and guarantee a valid `submission.csv` with the required `id_code,diagnosis` columns is written.'
- What this solution (achieved 0.01829) has done: 'Your score is strongly negative because the model is doing inference with a randomly initialized final classification layer (and no fine-tuning), so predictions are essentially random. To move the score up toward the 0.879 target while keeping your core inference logic (argmax over 5 logits) unchanged, the minimal legitimate fix is to load a trained checkpoint for the modified 5-class head (and full model) if it exists in the dataset working directory. I add a small, robust checkpoint search/load step that looks for common `.pt/.pth` files under `../input/aptos2019-blindness-detection/` (and falls back to current behavior if none are found), plus deterministic settings to avoid run-to-run drift. The output submission format/path remains identical (`submission.csv` with `id_code,diagnosis`).'
- What this solution (achieved 0.01829) has done: 'Your current score is low because the checkpoint search is accidentally picking up large non-model `.pth/.pt` files (e.g., the 7.7GB `train_images.zip` placeholder or other irrelevant binaries) and then “loading” them with `strict=False`, which leaves the 5-class head effectively random. I make the checkpoint finder strict to only consider plausible PyTorch state_dict files by (1) excluding known non-model paths/files and huge archive-like sizes, and (2) verifying the loaded object looks like a model/state_dict with at least the `fc.weight` tensor of shape `[5, in_features]` before applying it. If no valid checkpoint exists, the code fall back exactly as before, still producing `submission.csv`. These are minimal changes focused only on making sure a real trained checkpoint (if present) is loaded correctly, which should move the kappa score upward toward your target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

from tqdm import tqdm



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
    test_dataset,
    batch_size=24,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

weights = torchvision.models.ResNet152_Weights.DEFAULT
model = torchvision.models.resnet152(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)
model = model.to(device)


def _is_plausible_model_file(path, sz):
    fn = os.path.basename(path).lower()
    bad_name_tokens = [
        "train_images",
        "test_images",
        "sample_submission",
        "train.csv",
        "test.csv",
        ".zip",
        ".csv",
        ".md",
        ".txt",
        ".json",
    ]
    if any(tok in fn for tok in bad_name_tokens):
        return False
    if sz >= 2_000_000_000:  # 2GB
        return False
    if sz < 500_000:  # 0.5MB
        return False
    return True


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    return None


def _normalize_state_dict_keys(state):
    if state is None:
        return None
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _state_dict_matches_5class_resnet_head(state, in_features):
    w = state.get("fc.weight", None)
    b = state.get("fc.bias", None)
    if not isinstance(w, torch.Tensor):
        return False
    if list(w.shape) != [5, in_features]:
        return False
    if isinstance(b, torch.Tensor) and list(b.shape) != [5]:
        return False
    return True


def _find_and_load_best_checkpoint(model, search_roots, device):
    exts = (".pt", ".pth", ".bin")
    candidates = []
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                lfn = fn.lower()
                if not lfn.endswith(exts):
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    sz = os.path.getsize(full)
                except OSError:
                    continue
                if not _is_plausible_model_file(full, sz):
                    continue
                candidates.append((sz, full))

    candidates.sort(reverse=True)

    for _, path in candidates:
        try:
            ckpt = torch.load(path, map_location=device)
        except Exception:
            continue
        state = _extract_state_dict(ckpt)
        state = _normalize_state_dict_keys(state)
        if state is None:
            continue
        if not _state_dict_matches_5class_resnet_head(state, model.fc.in_features):
            continue

        missing, unexpected = model.load_state_dict(state, strict=False)
        return path, missing, unexpected

    return None, None, None


search_roots = [
    "../input/aptos2019-blindness-detection",
    "../input",
]
ckpt_path, missing, unexpected = _find_and_load_best_checkpoint(
    model, search_roots, device
)
if ckpt_path is not None:
    print(f"Loaded checkpoint (validated): {ckpt_path}")
    print(
        f"Missing keys (non-fatal): {len(missing)}; Unexpected keys (non-fatal): {len(unexpected)}"
    )
else:
    print(
        "No valid trained checkpoint found in ../input; using ImageNet backbone + randomly initialized 5-class head (likely low score)."
    )




## === cell 3
def compute_predictions(model, model_type, data_loader, device):
    if model_type == "train":
        predictions = []
        correct_pred, num_examples = 0, 0
        for inputs, labels in tqdm(data_loader, desc="Predict (train)"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.append(preds)
            num_examples += labels.size(0)
            correct_pred += (preds == labels).sum()
        return predictions, correct_pred.item() / num_examples * 100

    else:
        predictions = []
        img_ids = []
        for inputs, img_id in tqdm(data_loader, desc="Predict (test)"):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.detach().cpu().tolist())
            img_ids.extend(list(img_id))

        final_predictions = pd.DataFrame({"id_code": img_ids, "diagnosis": predictions})
        final_predictions.to_csv("submission.csv", index=False)
        return final_predictions




## === cell 4
with torch.set_grad_enabled(False):
    model.eval()
    print("Computing Test Predictions")
    test_predictions = compute_predictions(model, "test", test_loader, device)
    print(test_predictions.head())
    print("Wrote submission.csv")
