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

0.8139304115471919

# 6. Current score

0.5866

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.57635) has done: 'Your code didn’t yield a score mainly because it’s set up to train EfficientNet-B0 from scratch-ish (only ImageNet pretrained backbone) on a very large dataset for 2 epochs at 512px, which is unlikely to finish within the 600s budget and thus never reaches the submission-writing cell. I keep the same model/loss/training loop, but make the run reliably complete by (1) reducing image size to 256 and (2) using a much smaller but still class-balanced sample per folder so training finishes quickly and the notebook always writes `submission.csv`. I also make DataLoader seeding deterministic per worker (no semantic change) and avoid the `pip install efficientnet_pytorch` path by using `torchvision`’s EfficientNet-B0 if the external package isn’t present (same architecture family, but this is only to ensure the environment doesn’t stall). These minimal changes prioritize getting a valid submission first; once you can measure a current score, we can tune toward the target more precisely.'
- What this solution (achieved 0.57242) has done: 'Your current score (0.57635) is far below the target (0.81393), so we should improve generalization while keeping your core model/training loop intact. The biggest low-risk issue is that you currently take the first `sample_size` files per folder (sorted), which can create an unrepresentative subset; we instead randomly sample from the full folder deterministically, keeping the same sample_size/val split and the same 4-class training objective. We also fix a small metric bug: the weighted-AUC integration should include boundary points (`<=`/`>=`) and should explicitly anchor each TPR segment; this won’t change training, but it makes validation feedback accurate. Finally, we apply the exact same “stego_score = sum of non-cover probabilities” but use logits-based computation (`logsumexp`) for slightly more stable ranking (same semantics: monotonic in stego probability), which typically nudges AUC upward without changing architecture or training.'
- What this solution (achieved 0.58281) has done: 'We’re far below the target (0.57242 vs 0.81393), so we should improve generalization without changing your core model/loss/loop. The smallest high-impact change is to increase the effective training signal while staying within the runtime budget by (1) modestly increasing the per-class sample size and (2) increasing `num_epochs` from 2 to 3, leaving everything else intact. We also make a tiny but meaningful train-time fix by matching the same simple TTA used at test time (original + flips) during validation scoring only (not changing training), which typically improves score stability and correlates better with leaderboard behavior. All changes preserve the same architecture, cross-entropy objective, and prediction semantics, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.5866) has done: 'Your current score (0.58281) is far below the target (0.81393), so we should increase performance with the smallest changes that keep your same EfficientNet-B0 + cross-entropy training loop and the same “stego score = non-cover vs cover from logits” submission semantics. The biggest low-risk gain here is fixing an evaluation/train mismatch: you train on highly augmented JPEG-compressed images but validate on clean images, which can mislead model selection and hurt generalization; we make validation use the same resize+compression (but no flips) to better match the JPEG variability described in the dataset. We also align the torchvision EfficientNet-B0 preprocessing to its ImageNet weights (mean/std normalization) without changing architecture or loss, which usually provides a meaningful boost for pretrained backbones. Finally, we slightly raise the learning rate (same optimizer) to better adapt within only 3 epochs on a sampled dataset, which is a minimal knob that tends to improve undertraining without changing the core approach.'

# 9. Code solution

## === cell 0
import os
import random
import gc
import time
from glob import glob

import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from tqdm.auto import tqdm
from sklearn import metrics

import albumentations as A
from albumentations.pytorch import ToTensorV2

_EFFICIENTNET_BACKEND = None
try:
    from efficientnet_pytorch import EfficientNet  # original dependency

    _EFFICIENTNET_BACKEND = "efficientnet_pytorch"
except Exception:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    _EFFICIENTNET_BACKEND = "torchvision"

print("Imports OK; EfficientNet backend:", _EFFICIENTNET_BACKEND)



## === cell 1
seed = 42
print(f"setting everything to seed {seed}")
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 2
data_dir = "../input/alaska2-image-steganalysis"

sample_size = 12000  # per folder (was 6000)
val_size = int(sample_size * 0.25)

train_fn, val_fn = [], []
train_labels, val_labels = [], []

folder_names = ["Cover/", "JMiPOD/", "JUNIWARD/", "UERD/"]  # labels: 0,1,2,3

rng = np.random.default_rng(seed)

for label, folder in enumerate(folder_names):
    all_fns = sorted(glob(f"{data_dir}/{folder}*.jpg"))
    if len(all_fns) < sample_size:
        raise ValueError(
            f"Not enough files in {folder}: {len(all_fns)} < {sample_size}"
        )

    idx = rng.choice(len(all_fns), size=sample_size, replace=False)
    fns = np.array(all_fns, dtype=object)[idx]
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

train_df = pd.DataFrame({"ImageFileName": train_fn, "Label": train_labels})
train_df["Label"] = train_df["Label"].astype(int)

val_df = pd.DataFrame({"ImageFileName": val_fn, "Label": val_labels})
val_df["Label"] = val_df["Label"].astype(int)

print(train_df.head())
_ = train_df.Label.hist()
plt.title("Train label distribution")
plt.show()




## === cell 3
class Alaska2Dataset(Dataset):
    def __init__(self, df, augmentations=None):
        self.data = df.reset_index(drop=True)
        self.augment = augmentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fn = self.data.loc[idx, "ImageFileName"]
        label = int(self.data.loc[idx, "Label"])

        im = cv2.imread(fn)
        if im is None:
            raise FileNotFoundError(f"Could not read image: {fn}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        if self.augment is not None:
            im = self.augment(image=im)["image"]

        return im, label


img_size = 256

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

AUGMENTATIONS_VALID = A.Compose(
    [
        A.Resize(img_size, img_size),
        A.ImageCompression(quality_range=(75, 100), p=1.0),
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



## === cell 4
temp_df = train_df.sample(16, random_state=seed).reset_index(drop=True)
temp_dataset = Alaska2Dataset(temp_df, augmentations=AUGMENTATIONS_TEST)

temp_loader = torch.utils.data.DataLoader(
    temp_dataset, batch_size=16, num_workers=0, shuffle=False
)

images, labels = next(iter(temp_loader))
images = images.permute(0, 2, 3, 1).cpu().numpy()

vis = (
    images * np.array(IMAGENET_STD)[None, None, None, :]
    + np.array(IMAGENET_MEAN)[None, None, None, :]
)
vis = np.clip(vis, 0, 1)

fig, axs = plt.subplots(4, 4, figsize=(8, 8))
axs = axs.reshape(-1)
for i in range(16):
    axs[i].imshow(vis[i])
    axs[i].set_title(str(int(labels[i])))
    axs[i].axis("off")
plt.suptitle("0: COVER, 1: JMiPOD, 2: JUNIWARD, 3: UERD")
plt.show()

del images, labels, temp_loader, temp_dataset, vis
gc.collect()




## === cell 5
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        if _EFFICIENTNET_BACKEND == "efficientnet_pytorch":
            self.model = EfficientNet.from_pretrained("efficientnet-b0")
            self._mode = "efficientnet_pytorch"
            self.dense_output = nn.Linear(1280, 4)
        else:
            self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            self._mode = "torchvision"
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, 4)

    def forward(self, x):
        if self._mode == "efficientnet_pytorch":
            feat = self.model.extract_features(x)
            feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, 1280)
            return self.dense_output(feat)
        else:
            return self.model(x)




## === cell 6
batch_size = 8
num_workers = 4


def seed_worker(worker_id):
    worker_seed = (seed + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed)

train_dataset = Alaska2Dataset(train_df, augmentations=AUGMENTATIONS_TRAIN)
valid_dataset = Alaska2Dataset(val_df, augmentations=AUGMENTATIONS_VALID)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g,
    persistent_workers=(num_workers > 0),
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=batch_size * 2,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g,
    persistent_workers=(num_workers > 0),
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Net().to(device)

ckpt_path_candidates = [
    "../input/alaska/epoch_9_val_loss_6.67_auc_0.798.pth",  # original (likely missing)
    "../input/alaska2-image-steganalysis/epoch_9_val_loss_6.67_auc_0.798.pth",
]
loaded = False
for p in ckpt_path_candidates:
    if os.path.exists(p):
        state = torch.load(p, map_location="cpu")
        model.load_state_dict(state)
        loaded = True
        print(f"Loaded checkpoint: {p}")
        break
if not loaded:
    print(
        "No checkpoint found; will train from ImageNet pretrained EfficientNet-B0 + random head."
    )

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)




## === cell 7
def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, _ = metrics.roc_curve(y_true, y_valid, pos_label=1)

    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])
    normalization = np.dot(areas, weights)

    competition_metric = 0.0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]

        mask = (tpr >= y_min) & (tpr <= y_max)
        if mask.sum() < 2:
            continue

        x_seg = fpr[mask]
        y_seg = tpr[mask]

        x = np.concatenate([[x_seg[0]], x_seg, [x_seg[-1]]])
        y = np.concatenate([[y_min], y_seg, [y_max]])

        y = (y - y_min) / (y_max - y_min + 1e-12)
        score = metrics.auc(x, y) * (y_max - y_min)
        competition_metric += score * weight

    return competition_metric / normalization




## === cell 8
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 3  # was 2

train_loss, val_loss = [], []


def stego_score_from_logits(logits_tensor: torch.Tensor) -> torch.Tensor:
    log_p_stego_unnorm = torch.logsumexp(logits_tensor[:, 1:], dim=1)
    log_p_cover_unnorm = logits_tensor[:, 0]
    return log_p_stego_unnorm - torch.logaddexp(log_p_stego_unnorm, log_p_cover_unnorm)


for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch, num_epochs - 1))
    print("-" * 10)

    model.train()
    running_loss = 0.0
    tk0 = tqdm(train_loader, total=len(train_loader))
    for inputs, labels in tk0:
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        tk0.set_postfix(loss=float(loss.item()))

    epoch_loss = running_loss / max(1, len(train_loader))
    train_loss.append(epoch_loss)
    print("Training Loss: {:.8f}".format(epoch_loss))

    model.eval()
    running_loss = 0.0
    y, stego_scores = [], []
    tk1 = tqdm(valid_loader, total=len(valid_loader))
    with torch.no_grad():
        for inputs, labels in tk1:
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            out_o = model(inputs)
            loss = criterion(out_o, labels)

            out_h = model(inputs.flip(2))
            out_w = model(inputs.flip(3))
            outputs = 0.25 * out_h + 0.25 * out_w + 0.5 * out_o

            y.extend(labels.cpu().numpy().astype(int).tolist())
            stego_scores.extend(
                stego_score_from_logits(outputs).detach().cpu().numpy().tolist()
            )

            running_loss += float(loss.item())
            tk1.set_postfix(loss=float(loss.item()))

    epoch_loss = running_loss / max(1, len(valid_loader))
    val_loss.append(epoch_loss)

    y_np = np.asarray(y)
    y_bin = (y_np != 0).astype(int)
    stego_scores = np.asarray(stego_scores, dtype=np.float64)

    auc_score = alaska_weighted_auc(y_bin, stego_scores)

    with torch.no_grad():
        preds_for_acc = []
        for inputs, _labels in valid_loader:
            inputs = inputs.to(device, dtype=torch.float)
            preds_for_acc.append(F.softmax(model(inputs), 1).cpu())
    preds_for_acc = torch.cat(preds_for_acc, dim=0).numpy()
    pred_class = preds_for_acc.argmax(1)
    acc = (pred_class == y_np).mean() * 100.0

    print(f"Val Loss: {epoch_loss:.3}, Weighted AUC:{auc_score:.3}, Acc: {acc:.3}")

    torch.save(
        model.state_dict(),
        f"epoch_{epoch+1}_val_loss_{epoch_loss:.3}_auc_{auc_score:.3}.pth",
    )



## === cell 9
if len(train_loss) > 0:
    plt.figure(figsize=(15, 7))
    plt.plot(train_loss, c="r")
    plt.plot(val_loss, c="b")
    plt.legend(["train_loss", "val_loss"])
    plt.title("Loss Plot")
    plt.show()
else:
    print("Skipped loss plot because num_epochs=0 (as configured).")




## === cell 10
class Alaska2TestDataset(Dataset):
    def __init__(self, df, augmentations=None):
        self.data = df.reset_index(drop=True)
        self.augment = augmentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fn = self.data.loc[idx, "ImageFileName"]
        im = cv2.imread(fn)
        if im is None:
            raise FileNotFoundError(f"Could not read image: {fn}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        if self.augment is not None:
            im = self.augment(image=im)["image"]

        return im


test_filenames = sorted(glob(f"{data_dir}/Test/*.jpg"))
test_df = pd.DataFrame({"ImageFileName": list(test_filenames)})

batch_size = 16
num_workers = 4
test_dataset = Alaska2TestDataset(test_df, augmentations=AUGMENTATIONS_TEST)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g,
    persistent_workers=(num_workers > 0),
)

print("Test images:", len(test_df))



## === cell 11
model.eval()

preds_logits = []
tk0 = tqdm(test_loader, total=len(test_loader))
with torch.no_grad():
    for inputs in tk0:
        inputs = inputs.to(device, dtype=torch.float)

        im_h = inputs.flip(2)
        out_h = model(im_h)

        im_w = inputs.flip(3)
        out_w = model(im_w)

        out_o = model(inputs)

        outputs = 0.25 * out_h + 0.25 * out_w + 0.5 * out_o
        preds_logits.append(outputs.detach().cpu())

preds_logits = torch.cat(preds_logits, dim=0).numpy()

logits = torch.from_numpy(preds_logits)
log_p_stego_unnorm = torch.logsumexp(logits[:, 1:], dim=1)
log_p_cover_unnorm = logits[:, 0]
stego_score = (
    (log_p_stego_unnorm - torch.logaddexp(log_p_stego_unnorm, log_p_cover_unnorm))
    .numpy()
    .astype(np.float32)
)

test_df["Id"] = test_df["ImageFileName"].apply(lambda x: os.path.basename(x))
test_df["Label"] = stego_score
sub = test_df[["Id", "Label"]].copy()

sub = sub.sort_values("Id").reset_index(drop=True)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
assert sub.shape[0] == 5000, f"Expected 5000 test rows, got {sub.shape[0]}"
assert list(sub.columns) == ["Id", "Label"]
