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
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.10

# 3. Installed packages

albumentations==2.0.8
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
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Target score

0.8373268929321263

# 6. Current score

0.59388

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.56719) has done: 'I remove the incompatible `albumentations==0.5.2` install (it breaks with the provided SciPy/Numpy stack) and rely on the preinstalled albumentations 2.x API, fixing the dataset `__getitem__` to return the `'image'` key consistently. I also remove the dependency on `efficientnet_pytorch` (to avoid extra installs) and use the equivalent pretrained EfficientNet-B0 from `torchvision`, keeping the same overall classifier head (4-way softmax) and inference logic. I fix the incorrect pretrained weight path (it points to a non-existent dataset) by making it optional: load it only if the file exists, otherwise run with torchvision pretrained weights so a valid submission is always produced. Finally, I ensure the submission file is written as `submission.csv` with the exact `Id,Label` columns aligned to the test images.'
- What this solution (achieved 0.59388) has done: 'The timeout is dominated by huge, unnecessary data loading/augmentation for training/validation (225k+ images resized to 512) when a checkpoint is missing; that path cannot finish in 600s. To guarantee runtime while preserving the same inference/core model logic, the script (a) strictly require the provided checkpoint and skip training entirely (training was already conditional and not needed for submission), and (b) speed up test-time augmentation by avoiding a 3× batch forward with `torch.cat` and instead doing three forwards on the same batch (same math, much lower peak memory and better GPU scheduling stability). Additionally, dataloading is optimized safely via a single shared decode function and using `torchvision.transforms.functional.normalize` to avoid Albumentations overhead in the hot path while keeping identical resize/normalize semantics. All randomness settings and paths are preserved.'
- What this solution (achieved 0.59388) has done: 'The run currently fails because it hard-requires a checkpoint path that does not exist in your provided `/kaggle/input/alaska2-image-steganalysis` dataset, so I change checkpoint discovery to search common locations (including the working directory) and gracefully proceed with ImageNet-pretrained weights if none is found (so a valid `submission.csv` is always produced). This is a correctness/runtime fix only and keeps the same model architecture and inference semantics; it just avoids crashing before submission. To nudge score upward toward the target without changing the core approach, I also fix the weighted-AUC implementation to match the official “partial AUC over TPR bands” definition (the current implementation pads incorrectly and can mis-estimate). Finally, I keep the rest of the pipeline intact and ensure the submission has exactly `Id,Label` with 5000 rows.'
- What this solution (achieved 0.59388) has done: 'Your current score is far below target mainly because inference is often running with only ImageNet-pretrained weights when the competition checkpoint is not found, which severely hurts steganalysis performance. I make a minimal, runtime-safe change to reliably discover and load the checkpoint by searching recursively under the provided dataset directories (without changing the model or inference math). I also ensure state_dict loading is robust to common checkpoint wrappers (e.g., `{'state_dict': ...}` or `module.` prefixes) so you actually use the intended weights when present. These changes should increase the score toward the target while preserving your exact architecture, loss, and prediction post-processing.'
- What this solution (achieved 0.59388) has done: 'Your score is far below the target, so we should safely increase performance without changing the model or training/inference approach. The most likely reason you’re stuck around ~0.59 is that the intended competition checkpoint still isn’t being loaded due to key-prefix mismatches (e.g., `model.` from Lightning) and/or choosing an unrelated `.pth` during recursive search. I make a minimal, targeted fix: (1) improve checkpoint selection to prefer the exact filename (and otherwise prefer “epoch/auc” style ALASKA2 checkpoints over arbitrary `.pth` files), and (2) extend state_dict key-stripping to handle common prefixes (`model.`, `net.`, `module.`) so the weights actually land in the EfficientNet module. This preserves your architecture, preprocessing, TTA math, and output semantics, but should move the score upward toward the target by ensuring the correct weights are used.'

# 9. Code solution

## === cell 0
import os
import random
import gc
from glob import glob
import time

import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset

from tqdm.auto import tqdm
from sklearn import metrics

import albumentations as A
from albumentations.pytorch import ToTensorV2


seed = 42
print(f"setting everything to seed {seed}")
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = (seed + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed)

try:
    cv2.setNumThreads(0)
except Exception:
    pass

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass



## === cell 1
data_dir = "../input/alaska2-image-steganalysis"

sample_size = 75000
val_size = int(sample_size * 0.25)

train_fn, val_fn = [], []
train_labels, val_labels = [], []

folder_names = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]  # labels 0..3

for label, folder in enumerate(folder_names):
    fns = sorted(glob(f"{data_dir}/{folder}/*.jpg"))[:sample_size]
    fns = np.array(fns)
    rng = np.random.RandomState(seed + label)
    rng.shuffle(fns)
    fns = fns.tolist()

    val_part = fns[:val_size]
    train_part = fns[val_size:]

    train_fn.extend(train_part)
    train_labels.extend([label] * len(train_part))
    val_fn.extend(val_part)
    val_labels.extend([label] * len(val_part))

assert len(train_labels) == len(train_fn), "wrong labels"
assert len(val_labels) == len(val_fn), "wrong labels"

train_df = pd.DataFrame({"ImageFileName": train_fn, "Label": train_labels})[
    ["ImageFileName", "Label"]
]
train_df["Label"] = train_df["Label"].astype(int)

val_df = pd.DataFrame({"ImageFileName": val_fn, "Label": val_labels})[
    ["ImageFileName", "Label"]
]
val_df["Label"] = val_df["Label"].astype(int)

print(train_df.head())
print(val_df.head())




## === cell 2
def _read_rgb_reduced(fn: str) -> np.ndarray:
    im = cv2.imread(fn, cv2.IMREAD_COLOR | cv2.IMREAD_REDUCED_COLOR_2)
    if im is None:
        raise FileNotFoundError(f"Could not read image: {fn}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


class Alaska2Dataset(Dataset):
    def __init__(self, df, augmentations=None):
        self.fns = df["ImageFileName"].to_numpy()
        self.labels = df["Label"].to_numpy(dtype=np.int64)
        self.augment = augmentations

    def __len__(self):
        return self.fns.shape[0]

    def __getitem__(self, idx):
        fn = self.fns[idx]
        label = int(self.labels[idx])

        im = _read_rgb_reduced(fn)
        if self.augment is not None:
            im = self.augment(image=im)["image"]
        return im, label


img_size = 512

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

AUGMENTATIONS_TRAIN = A.Compose(
    [
        A.Resize(img_size, img_size),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_range=(75, 100), p=0.5),
        A.ToFloat(max_value=255.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=1.0),
        ToTensorV2(),
    ]
)

AUGMENTATIONS_TEST = A.Compose(
    [
        A.Resize(img_size, img_size),
        A.ToFloat(max_value=255.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=1.0),
        ToTensorV2(),
    ]
)



## === cell 3
if False:
    temp_df = train_df.sample(16, random_state=seed).reset_index(drop=True)
    temp_dataset = Alaska2Dataset(temp_df, augmentations=AUGMENTATIONS_TEST)
    temp_loader = torch.utils.data.DataLoader(
        temp_dataset, batch_size=16, num_workers=0, shuffle=False
    )

    images, labels = next(iter(temp_loader))
    images_np = images.permute(0, 2, 3, 1).cpu().numpy()

    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    axs = axs.ravel()
    for i in range(16):
        axs[i].imshow(
            np.clip(
                (images_np[i] * np.array(IMAGENET_STD)) + np.array(IMAGENET_MEAN), 0, 1
            )
        )
        axs[i].set_title(str(int(labels[i])))
        axs[i].axis("off")
    plt.suptitle("0: COVER, 1: JMiPOD, 2: JUNIWARD, 3: UERD")
    plt.tight_layout()
    plt.show()

    del images, labels, images_np, temp_df, temp_dataset, temp_loader
    gc.collect()




## === cell 4
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = torchvision.models.efficientnet_b0(weights=weights)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 4)

    def forward(self, x):
        return self.model(x)




## === cell 5
def alaska_weighted_auc(y_true, y_score):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)

    if fpr[0] != 0.0 or tpr[0] != 0.0:
        fpr = np.concatenate([[0.0], fpr])
        tpr = np.concatenate([[0.0], tpr])
    if fpr[-1] != 1.0 or tpr[-1] != 1.0:
        fpr = np.concatenate([fpr, [1.0]])
        tpr = np.concatenate([tpr, [1.0]])

    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])
    normalization = float(np.dot(areas, weights))

    total = 0.0
    for i, w in enumerate(weights):
        y_min = tpr_thresholds[i]
        y_max = tpr_thresholds[i + 1]

        tpr_clipped = np.clip(tpr, y_min, y_max)
        y = tpr_clipped - y_min
        band_area = metrics.auc(fpr, y)
        total += float(w) * float(band_area)

    return total / normalization




## === cell 6
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Net().to(device)


def _find_ckpt():
    exact_name = "epoch_13_val_loss_6.67_auc_0.819.pth"
    preferred = [
        "../input/alaska/epoch_13_val_loss_6.67_auc_0.819.pth",
        "../input/alaska2-image-steganalysis/epoch_13_val_loss_6.67_auc_0.819.pth",
        "../input/alaska2-image-steganalysis/alaska2-image-steganalysis/epoch_13_val_loss_6.67_auc_0.819.pth",
        "../working/epoch_13_val_loss_6.67_auc_0.819.pth",
        "./epoch_13_val_loss_6.67_auc_0.819.pth",
    ]
    for p in preferred:
        if os.path.exists(p):
            return p

    roots = ["../input/alaska2-image-steganalysis", "../input", "../working", "."]
    patterns = [
        f"**/{exact_name}",
        "**/*val_loss*auc*.pth",
        "**/*auc*.pth",
        "**/*.pth",
    ]

    candidates = []
    for root in roots:
        if not os.path.exists(root):
            continue
        for pat in patterns:
            hits = glob(os.path.join(root, pat), recursive=True)
            if hits:
                candidates.extend(hits)

    if not candidates:
        return None

    candidates = sorted(set(candidates))
    exact_hits = [c for c in candidates if os.path.basename(c) == exact_name]
    if exact_hits:
        return sorted(exact_hits)[0]

    def _rank(p):
        base = os.path.basename(p).lower()
        path = p.lower()
        score = 0
        score += 10 if "alaska" in path else 0
        score += 6 if "epoch" in base else 0
        score += 4 if "val_loss" in base else 0
        score += 4 if "auc" in base else 0
        score -= 2 if "best" in base else 0  # neutral/weak; keep minimal
        return -score, p  # smaller tuple is better

    candidates_sorted = sorted(candidates, key=_rank)
    return candidates_sorted[0]


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            sd = ckpt_obj["state_dict"]
        elif "model_state_dict" in ckpt_obj and isinstance(
            ckpt_obj["model_state_dict"], dict
        ):
            sd = ckpt_obj["model_state_dict"]
        else:
            sd = ckpt_obj
    else:
        sd = ckpt_obj

    if not isinstance(sd, dict):
        return sd

    prefixes = ("module.", "model.", "net.")
    changed = True
    while changed:
        changed = False
        keys = list(sd.keys())
        for pref in prefixes:
            if keys and all(k.startswith(pref) for k in keys):
                sd = {k[len(pref) :]: v for k, v in sd.items()}
                changed = True
                break

    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    return sd


ckpt_path = _find_ckpt()

if ckpt_path is not None:
    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    state = _extract_state_dict(ckpt_obj)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    if missing or unexpected:
        print(
            f"NOTE: load_state_dict strict=False; missing={len(missing)}, unexpected={len(unexpected)}"
        )
else:
    print(
        "WARNING: checkpoint not found; running with torchvision ImageNet pretrained weights only. "
        "Submission will be produced, but score may be lower than a competition-trained checkpoint."
    )

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

batch_size = 8
_cpu = os.cpu_count() or 4
num_workers = min(8, max(2, _cpu // 2))

train_dataset = valid_dataset = None
train_loader = valid_loader = None



## === cell 7
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 0  # checkpoint inference-only for timeout compliance

train_loss, val_loss = [], []

for epoch in range(num_epochs):
    print(f"Epoch {epoch}/{num_epochs - 1}")
    print("-" * 10)

    model.train()
    running_loss = 0.0

    tk0 = tqdm(train_loader, total=len(train_loader), leave=False, mininterval=1.0)
    for images, labels in tk0:
        inputs = images.to(device, dtype=torch.float, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())

    epoch_loss = running_loss / max(1, len(train_loader))
    train_loss.append(epoch_loss)
    print(f"Training Loss: {epoch_loss:.8f}")

    model.eval()
    running_loss = 0.0

    n_valid = len(valid_dataset)
    preds = np.empty((n_valid, 4), dtype=np.float32)
    y = np.empty((n_valid,), dtype=np.int64)

    offset = 0
    tk1 = tqdm(valid_loader, total=len(valid_loader), leave=False, mininterval=1.0)
    with torch.inference_mode():
        for images, labels in tk1:
            bs = images.shape[0]
            inputs = images.to(device, dtype=torch.float, non_blocking=True)
            labels_t = labels.to(device, dtype=torch.long, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels_t)

            probs = (
                F.softmax(outputs, 1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            preds[offset : offset + bs] = probs
            y[offset : offset + bs] = labels.numpy().astype(np.int64, copy=False)

            offset += bs
            running_loss += float(loss.item())

    epoch_loss = running_loss / max(1, len(valid_loader))
    val_loss.append(epoch_loss)

    y_bin = (y != 0).astype(int)
    stego_prob = 1.0 - preds[:, 0]
    auc_score = alaska_weighted_auc(y_bin, stego_prob)

    hard = preds.argmax(1)
    acc = (hard == y).mean() * 100.0
    print(f"Val Loss: {epoch_loss:.3f}, Weighted AUC:{auc_score:.3f}, Acc: {acc:.3f}")

    torch.save(
        model.state_dict(),
        f"epoch_{epoch}_val_loss_{epoch_loss:.3f}_auc_{auc_score:.3f}.pth",
    )



## === cell 8
if len(train_loss) > 0:
    plt.figure(figsize=(15, 7))
    plt.plot(train_loss, c="r")
    plt.plot(val_loss, c="b")
    plt.legend(["train_loss", "val_loss"])
    plt.title("Loss Plot")
    plt.show()



## === cell 9
from torchvision.transforms import functional as TF

_MEAN_T = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
_STD_T = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)


class Alaska2TestDataset(Dataset):
    def __init__(self, df, img_size=512):
        self.fns = df["ImageFileName"].to_numpy()
        self.img_size = int(img_size)

    def __len__(self):
        return self.fns.shape[0]

    def __getitem__(self, idx):
        fn = self.fns[idx]
        im = _read_rgb_reduced(fn)
        im = cv2.resize(
            im, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
        )
        t = (
            torch.from_numpy(im)
            .permute(2, 0, 1)
            .contiguous()
            .to(dtype=torch.float32)
            .mul_(1.0 / 255.0)
        )
        t = (t - _MEAN_T) / _STD_T
        return t


test_filenames = sorted(glob(f"{data_dir}/Test/*.jpg"))
test_df = pd.DataFrame({"ImageFileName": test_filenames})[["ImageFileName"]]

batch_size = 32
_cpu = os.cpu_count() or 4
num_workers = min(8, max(2, _cpu // 2))
test_dataset = Alaska2TestDataset(test_df, img_size=img_size)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)

print("Test images:", len(test_df))



## === cell 10
model.eval()

_prev_bench = torch.backends.cudnn.benchmark
torch.backends.cudnn.benchmark = True

n_test = len(test_dataset)
preds = np.empty((n_test, 4), dtype=np.float32)

offset = 0
tk0 = tqdm(test_loader, total=len(test_loader), leave=False, mininterval=1.0)
with torch.inference_mode():
    for images in tk0:
        bs = images.shape[0]
        inputs = images.to(device, dtype=torch.float, non_blocking=True)

        out1 = model(inputs)
        out2 = model(inputs.flip(2))
        out3 = model(inputs.flip(3))
        outputs = 0.5 * out1 + 0.25 * out2 + 0.25 * out3

        probs = F.softmax(outputs, 1).cpu().numpy().astype(np.float32, copy=False)
        preds[offset : offset + bs] = probs
        offset += bs

torch.backends.cudnn.benchmark = _prev_bench

new_preds = 1.0 - preds[:, 0].astype(np.float32)

test_df["Id"] = pd.Series(test_df["ImageFileName"].to_numpy()).map(os.path.basename)
test_df["Label"] = new_preds
sub = test_df[["Id", "Label"]].copy()

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with", len(sub), "rows")
assert sub.shape[0] == 5000 and list(sub.columns) == ["Id", "Label"]
assert sub["Id"].isna().sum() == 0 and sub["Label"].isna().sum() == 0
