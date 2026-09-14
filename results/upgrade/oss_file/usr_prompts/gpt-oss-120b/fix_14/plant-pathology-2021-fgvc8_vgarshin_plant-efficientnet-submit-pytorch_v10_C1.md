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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.8273499538319488

# 6. Current score

0.26014

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'Implemented fixes:
- Resolved name clash between torchvision `models` and the model list by aliasing the import and renaming the list to `model_list`.
- Adjusted EfficientNet loading to use the aliased torchvision models.
- Filtered only image files when building the submission template to avoid directory entries.
- Updated downstream code to reference the new `model_list`.
- Minor clean‑ups for robustness while preserving original logic.'
- What this solution (achieved 0.26014) has done: 'I fix the NameError caused by deleting an undefined variable, lower the prediction threshold to make the model output more disease labels, and adjust the label‑decoding logic so that “healthy” is only used when no disease is predicted. These changes keep the original model architecture and training approach intact while correcting bugs and nudging the F1‑score toward the target.'
- What this solution (achieved 0.19163) has done: 'I raise the prediction threshold to a more selective value and average the predictions from both loaded folds instead of using only the first model. This should reduce false‑positive disease tags and improve the mean F1 score, moving it closer to the target while preserving the original architecture and training logic.'
- What this solution (achieved 0.26014) has done: 'The refactor reduces duplicated image loading and I/O by using a single DataLoader for all test‑time augmentations, applying flips directly on the in‑memory tensors, and increasing the data‑loading workers. The inference loop now averages predictions across folds and flips without rebuilding loaders, preserving the exact ensemble logic and output while cutting runtime dramatically.'
- What this solution (achieved 0.28049) has done: 'I lower the prediction threshold a bit and replace the simple “any‑above‑threshold” decoding with a small top‑k selection strategy: we keep the strongest non‑healthy predictions (up to two) that exceed the new lower threshold, falling back to “healthy” only when none qualify. This modest change preserves the whole model‑inference pipeline while making the label post‑processing a bit more discriminative, which should move the F1 score closer to the target.'
- What this solution (achieved 0.16361) has done: 'Implemented three focused adjustments to boost the F1‑score while keeping the original model pipeline intact:  
1. **Raised the prediction threshold** from 0.10 to 0.20 to reduce false‑positive disease tags.  
2. **Added ImageNet‑style normalization** inside the dataset loader so test images are processed with the same mean/std used during training.  
3. **Extended the top‑k label selection** to up to three disease predictions, allowing more accurate multi‑disease labeling.  

These minimal changes align inference with training preprocessing and calibration, moving the score toward the target.'
- What this solution (achieved 0.26014) has done: 'I lower the prediction threshold to 0.10 to increase recall, expand the top‑k selection to 5 labels, and ensure the EfficientNet backbone is loaded with ImageNet pretrained weights (using `from_pretrained` when available). These minimal adjustments keep the original pipeline intact while providing a stronger baseline that should move the F1 score closer to the target.'
- What this solution (achieved 0.22698) has done: 'I raise the prediction threshold and limit the number of predicted classes to make the model’s output more precise, and I disable test‑time augmentation (use only the original image). These small adjustments keep the core architecture and inference logic unchanged while moving the F1 score closer to the target.'
- What this solution (achieved 0.26014) has done: 'The changes lower the decision threshold, enable all four test‑time augmentations, and allow up to five predicted disease labels per image, which should increase recall and overall F1‑score, moving the metric closer to the target while keeping the original pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import gc
import json
import time
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.sampler import SequentialSampler
from torchvision import transforms, models as tv_models  # alias to avoid name clash
from IPython.display import display  # for Jupyter preview calls

KAGGLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from efficientnet_pytorch import model as enet
except Exception:
    enet = None  # will use torchvision later.




## === cell 1
TEST = True
VER = "v4"
if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"../input/plant-models-{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"
TH = 0.15  # lowered threshold to improve recall
TTAS = [0, 1, 2, 3]  # enable all test‑time augmentations
FOLDS = [3, 4]
IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"

start_time = time.time()




## === cell 2
image_files = [
    f
    for f in sorted(os.listdir(IMGS_PATH))
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
df_sub = pd.DataFrame(image_files, columns=["image"])
df_sub["labels"] = "healthy"  # placeholder; will be overwritten after inference.
print("Submission template preview:")
display(df_sub.head())




## === cell 3
def flip(img, axis=0):
    if axis == 1:
        return img[::-1, :, :]
    elif axis == 2:
        return img[:, ::-1, :]
    elif axis == 3:
        return img[::-1, ::-1, :]
    else:
        return img


class PlantDataset(data.Dataset):
    def __init__(self, df, size, labels, transform=None):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.labels = labels
        self.transform = transform
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.image
        img_path = f"{IMGS_PATH}/{img_name}"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        img = img.transpose(2, 0, 1)  # C, H, W
        return torch.tensor(img.copy())


class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        backbone_name = params["backbone"]
        if enet is not None:
            try:
                self.enet = enet.EfficientNet.from_pretrained(backbone_name)
            except Exception:
                self.enet = enet.EfficientNet.from_name(backbone_name)
        else:
            try:
                ctor_name = f"efficientnet_{backbone_name.split('-')[-1]}"
                self.enet = getattr(tv_models, ctor_name)(pretrained=True)
            except Exception as e:
                raise ValueError(f"Backbone {backbone_name} not supported.") from e

        nc = (
            self.enet.classifier[1].in_features
            if hasattr(self.enet, "classifier")
            else self.enet._fc.in_features
        )
        if hasattr(self.enet, "classifier"):
            self.enet.classifier = nn.Identity()
        else:
            self.enet._fc = nn.Identity()
        self.myfc = nn.Sequential(
            nn.Dropout(params["dropout"]),
            nn.Linear(nc, nc // 4),
            nn.Dropout(params["dropout"]),
            nn.Linear(nc // 4, out_dim),
        )

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x




## === cell 4
params_path = f"{MDLS_PATH}/params.json"
if os.path.isfile(params_path):
    with open(params_path) as file:
        params = json.load(file)
else:
    print("params.json not found – using default parameters.")
    params = {
        "img_size": 224,
        "batch_size": 16,
        "workers": 0,
        "dropout": 0.2,
        "backbone": "efficientnet-b0",
        "labels_": ["healthy", "multiple_diseases", "rust", "scab", "complex"],
        "labels": {
            "0": "healthy",
            "1": "multiple_diseases",
            "2": "rust",
            "3": "scab",
            "4": "complex",
        },
    }
LABELS_ = params["labels_"]
LABELS = params["labels"]
WORKERS = params.get("workers", 0)
if WORKERS == 0:
    WORKERS = min(4, os.cpu_count() or 1)
print("Loaded params:", params)




## === cell 5
model_list = []
for n_fold in FOLDS:
    model = EffNet(params, out_dim=len(LABELS_))
    ckpt_path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
    if os.path.isfile(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"Checkpoint not found for fold {n_fold}, using pretrained weights.")
    model = model.to(DEVICE)
    model.eval()
    model_list.append(model)
try:
    del model
except NameError:
    pass
try:
    del state_dict
except NameError:
    pass




## === cell 6
dataset = PlantDataset(df=df_sub, size=params["img_size"], labels=None, transform=None)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=params["batch_size"],
    sampler=SequentialSampler(dataset),
    num_workers=WORKERS,
    pin_memory=True,
)




## === cell 7
def get_labels(row, labels_dict, th, top_k=3):
    """
    Decode a prediction row into space‑delimited labels.
    Keeps up to `top_k` strongest non‑healthy classes whose probability exceeds `th`.
    Falls back to “healthy” when none meet the criteria.
    """
    probs = np.array(row)
    sorted_idx = probs.argsort()[::-1]
    selected = []
    for idx in sorted_idx:
        label = labels_dict.get(str(idx), "healthy")
        if label == "healthy":
            continue
        if probs[idx] >= th:
            selected.append(label)
        if len(selected) >= top_k:
            break
    if not selected:
        return "healthy"
    return " ".join(selected)


def apply_tta(batch_tensor, tta_code):
    if tta_code == 0:
        return batch_tensor
    if tta_code == 1:
        return torch.flip(batch_tensor, dims=[2])  # vertical flip (height)
    if tta_code == 2:
        return torch.flip(batch_tensor, dims=[3])  # horizontal flip (width)
    if tta_code == 3:
        return torch.flip(batch_tensor, dims=[2, 3])  # both axes
    return batch_tensor


fold_preds = []
for model in model_list:
    model.eval()
    preds_per_batch = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            tta_preds = []
            for tta in TTAS:
                batch_tta = apply_tta(batch, tta)
                preds = torch.sigmoid(model(batch_tta))
                tta_preds.append(preds)
            avg_preds = torch.mean(torch.stack(tta_preds), dim=0)  # (B, C)
            preds_per_batch.append(avg_preds.cpu().numpy())
    fold_preds.append(np.vstack(preds_per_batch))  # (N, C) for this fold

logits = np.mean(np.stack(fold_preds), axis=0)  # (num_images, num_classes)

df_sub["labels"] = [get_labels(x, LABELS, TH, top_k=5) for x in logits]

elapsed = time.time() - start_time
print(f"Time elapsed: {int(elapsed // 60)} min {int(elapsed % 60)} sec")
print("Sample predictions:")
display(df_sub.head())




## === cell 8
output_path = "submission.csv"
df_sub.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
