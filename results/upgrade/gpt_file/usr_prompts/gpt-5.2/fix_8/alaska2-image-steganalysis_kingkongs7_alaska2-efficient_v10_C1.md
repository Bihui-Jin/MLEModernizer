# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import gc
import time
import random
from glob import glob

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from tqdm import tqdm
from sklearn import metrics

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchvision

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)



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

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def seed_worker(worker_id):
    worker_seed = (seed + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g_torch = torch.Generator()
g_torch.manual_seed(seed)



## === cell 2
data_dir = "../input/alaska2-image-steganalysis"
sample_size = 75000
val_size = int(sample_size * 0.25)

ckpt_path = "../input/alaska/epoch_14_val_loss_6.63_auc_0.824.pth"
fallback_ckpt_dir = os.path.dirname(data_dir)
fallback_ckpt_glob = glob(
    os.path.join(fallback_ckpt_dir, "**", "*.pth"), recursive=True
)
if (not os.path.exists(ckpt_path)) and len(fallback_ckpt_glob) > 0:
    fallback_ckpt_glob = sorted(fallback_ckpt_glob)
    ckpt_path = fallback_ckpt_glob[0]
    print(f"Using available checkpoint instead: {ckpt_path}")

will_train = not os.path.exists(ckpt_path)

if will_train:
    train_fn, val_fn = [], []
    train_labels, val_labels = [], []

    folder_names = ["Cover/", "JMiPOD/", "JUNIWARD/", "UERD/"]  # label 0,1,2,3

    for label, folder in enumerate(folder_names):
        filenames = sorted(glob(f"{data_dir}/{folder}/*.jpg"))[:sample_size]
        filenames = np.array(filenames)
        np.random.shuffle(filenames)

        val_part = filenames[:val_size].tolist()
        train_part = filenames[val_size:].tolist()

        train_fn.extend(train_part)
        train_labels.extend([label] * len(train_part))

        val_fn.extend(val_part)
        val_labels.extend([label] * len(val_part))

    assert len(train_labels) == len(train_fn), "wrong labels"
    assert len(val_labels) == len(val_fn), "wrong labels"

    train_df = pd.DataFrame({"ImageFileName": train_fn, "Label": train_labels})[
        ["ImageFileName", "Label"]
    ]
    val_df = pd.DataFrame({"ImageFileName": val_fn, "Label": val_labels})[
        ["ImageFileName", "Label"]
    ]

    train_df["Label"] = train_df["Label"].astype(int)
    val_df["Label"] = val_df["Label"].astype(int)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(train_df.head())
    _ = train_df.Label.hist()
else:
    train_df = None
    val_df = None
    print(
        f"Checkpoint found at {ckpt_path}; skipping train/val file enumeration to save time."
    )




## === cell 3
class Alaska2Dataset(Dataset):
    def __init__(self, df, augmentations=None, fast_resize=False, img_size=512):
        self.data = df.reset_index(drop=True)
        self.augment = augmentations
        self.fast_resize = fast_resize
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fn = row["ImageFileName"]
        label = int(row["Label"])

        im = cv2.imread(fn)
        if im is None:
            raise FileNotFoundError(f"Failed to read image: {fn}")
        im = im[:, :, ::-1]  # BGR->RGB

        if self.fast_resize:
            im = cv2.resize(
                im, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA
            )
            im = torch.from_numpy(im).permute(2, 0, 1).contiguous().float().div_(255.0)
        else:
            if self.augment is not None:
                im = self.augment(image=im)["image"]

        return im, label


img_size = 512

AUGMENTATIONS_TRAIN = A.Compose(
    [
        A.Resize(img_size, img_size),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_range=(75, 100), p=0.5),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ],
    p=1.0,
)

AUGMENTATIONS_TEST = A.Compose(
    [
        A.Resize(img_size, img_size),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ],
    p=1.0,
)



## === cell 4
if will_train:
    temp_df = train_df.sample(16, random_state=seed).reset_index(drop=True)
    temp_dataset = Alaska2Dataset(
        temp_df, augmentations=AUGMENTATIONS_TEST, fast_resize=False, img_size=img_size
    )

    temp_loader = torch.utils.data.DataLoader(
        temp_dataset, batch_size=16, num_workers=0, shuffle=False
    )
    images, labels = next(iter(temp_loader))
    images = images.permute(0, 2, 3, 1).cpu().numpy()

    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    axs = axs.ravel()
    for i in range(16):
        axs[i].imshow(np.clip(images[i], 0, 1))
        axs[i].set_title(str(int(labels[i])))
        axs[i].axis("off")
    plt.suptitle("0: COVER, 1: JMiPOD, 2: JUNIWARD, 3: UERD")
    plt.tight_layout()
    plt.show()
    del images, labels, temp_loader, temp_dataset, temp_df
    gc.collect()




## === cell 5
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.efficientnet_b0(
            weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.dense_output = nn.Linear(1280, 4)

    def forward(self, x):
        feat = self.model.features(x)
        feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, 1280)
        return self.dense_output(feat)




## === cell 6
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Net().to(device)

if device == "cuda":
    model = model.to(memory_format=torch.channels_last)

if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            new_state[nk] = v
        state = new_state
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    if missing:
        print(f"Missing keys (ignored): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(
            f"Unexpected keys (ignored): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}"
        )
else:
    print(
        f"Checkpoint not found at {ckpt_path}. Will train from scratch to produce a valid submission."
    )

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

if hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        print("torch.compile enabled (reduce-overhead).")
    except Exception as e:
        print(f"torch.compile not enabled (fallback to eager): {type(e).__name__}: {e}")

if will_train:
    batch_size = 8
    cpu_cnt = os.cpu_count() or 4
    num_workers = min(8, max(2, cpu_cnt // 2))

    train_dataset = Alaska2Dataset(
        train_df,
        augmentations=AUGMENTATIONS_TRAIN,
        fast_resize=False,
        img_size=img_size,
    )
    valid_dataset = Alaska2Dataset(
        val_df, augmentations=None, fast_resize=True, img_size=img_size
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4,
        worker_init_fn=seed_worker,
        generator=g_torch,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size * 2,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4,
        worker_init_fn=seed_worker,
        generator=g_torch,
    )




## === cell 7
def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)

    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])
    normalization = np.dot(areas, weights)

    competition_metric = 0.0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]

        mask = (tpr >= y_min) & (tpr <= y_max)
        if not np.any(mask):
            continue

        fpr_sub = fpr[mask]
        tpr_sub = tpr[mask]

        if fpr_sub[-1] < 1.0:
            x_padding = np.linspace(fpr_sub[-1], 1.0, 100)
            x = np.concatenate([fpr_sub, x_padding])
            y = np.concatenate([tpr_sub, np.full_like(x_padding, y_max)])
        else:
            x, y = fpr_sub, tpr_sub

        y = y - y_min
        score = metrics.auc(x, y)
        competition_metric += score * weight

    return competition_metric / normalization




## === cell 8
criterion = torch.nn.CrossEntropyLoss()

train_loss, val_loss = [], []

num_epochs = 1 if not os.path.exists(ckpt_path) else 0

for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch, num_epochs - 1))
    print("-" * 10)
    model.train()
    running_loss = 0.0

    tk0 = tqdm(train_loader, total=len(train_loader))
    for im, labels in tk0:
        inputs = im.to(device, dtype=torch.float, non_blocking=True)
        if device == "cuda":
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        tk0.set_postfix(loss=float(loss.item()))

    epoch_loss = running_loss / max(1, len(train_loader))
    train_loss.append(epoch_loss)
    print("Training Loss: {:.8f}".format(epoch_loss))

    tk1 = tqdm(valid_loader, total=len(valid_loader))
    model.eval()
    running_vloss = 0.0
    y, preds = [], []

    with torch.no_grad():
        for im, labels in tk1:
            inputs = im.to(device, dtype=torch.float, non_blocking=True)
            if device == "cuda":
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            y.extend(labels.cpu().numpy().astype(int))
            preds.extend(F.softmax(outputs, 1).cpu().numpy())

            running_vloss += loss.item()
            tk1.set_postfix(loss=float(loss.item()))

    epoch_vloss = running_vloss / max(1, len(valid_loader))
    val_loss.append(epoch_vloss)

    preds = np.array(preds)
    pred_labels = preds.argmax(1)
    acc = (pred_labels == np.array(y)).mean() * 100.0

    new_preds = np.zeros((len(preds),), dtype=np.float32)
    temp = preds[pred_labels != 0, 1:]
    new_preds[pred_labels != 0] = temp.sum(1)
    new_preds[pred_labels == 0] = preds[pred_labels == 0, 0]

    y_bin = np.array(y)
    y_bin[y_bin != 0] = 1
    auc_score = alaska_weighted_auc(y_bin, new_preds)
    print(f"Val Loss: {epoch_vloss:.3}, Weighted AUC:{auc_score:.3}, Acc: {acc:.3}")

    torch.save(
        model.state_dict(),
        f"epoch_{epoch+1}_val_loss_{epoch_vloss:.3}_auc_{auc_score:.3}.pth",
    )



## === cell 9
if len(train_loss) > 0:
    plt.figure(figsize=(12, 5))
    plt.plot(train_loss, c="r")
    plt.plot(val_loss, c="b")
    plt.legend(["train_loss", "val_loss"])
    plt.title("Loss Plot")
    plt.show()




## === cell 10
class Alaska2TestDataset(Dataset):
    def __init__(self, df, fast_resize=True, img_size=512):
        self.data = df.reset_index(drop=True)
        self.fast_resize = fast_resize
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]["ImageFileName"]


def _test_collate_filenames_to_tensor(batch_filenames, img_size=512):
    ims = []
    for fn in batch_filenames:
        im = cv2.imread(fn)
        if im is None:
            raise FileNotFoundError(f"Failed to read image: {fn}")
        im = im[:, :, ::-1]  # BGR->RGB
        im = cv2.resize(im, (img_size, img_size), interpolation=cv2.INTER_AREA)
        ims.append(im)
    arr = np.stack(ims, axis=0)  # NHWC uint8
    ten = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    return ten


test_dir = f"{data_dir}/Test"
test_filenames = sorted(
    os.path.join(test_dir, fn) for fn in os.listdir(test_dir) if fn.endswith(".jpg")
)

test_df = pd.DataFrame({"ImageFileName": test_filenames})[
    ["ImageFileName"]
].reset_index(drop=True)

batch_size = 32 if device == "cuda" else 16
cpu_cnt = os.cpu_count() or 4
num_workers = min(8, max(2, cpu_cnt // 2))

test_dataset = Alaska2TestDataset(test_df, fast_resize=True, img_size=img_size)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=8,
    worker_init_fn=seed_worker,
    generator=g_torch,
    collate_fn=lambda fns: _test_collate_filenames_to_tensor(fns, img_size=img_size),
)

print("Test images:", len(test_df))

model.eval()

if device == "cuda":
    torch.backends.cudnn.benchmark = True

n_test = len(test_dataset)
preds = np.empty((n_test, 4), dtype=np.float32)

tk0 = tqdm(test_loader, total=len(test_loader))
offset = 0

with torch.inference_mode():
    for im in tk0:
        inputs = im.to(device, dtype=torch.float, non_blocking=True)
        if device == "cuda":
            inputs = inputs.contiguous(memory_format=torch.channels_last)

        x0 = inputs
        x2 = inputs.flip(2)
        x3 = inputs.flip(3)
        x_cat = torch.cat([x0, x2, x3], dim=0)
        logits_cat = model(x_cat)
        b = inputs.shape[0]
        logits1 = logits_cat[:b]
        logits2 = logits_cat[b : 2 * b]
        logits3 = logits_cat[2 * b : 3 * b]
        outputs = 0.5 * logits1 + 0.25 * logits2 + 0.25 * logits3

        prob = (
            F.softmax(outputs, 1).detach().cpu().numpy().astype(np.float32, copy=False)
        )
        preds[offset : offset + b] = prob
        offset += b

labels = preds.argmax(1)

new_preds = np.zeros((len(preds),), dtype=np.float32)
mask = labels != 0
new_preds[mask] = preds[mask, 1:].sum(1)
new_preds[~mask] = preds[~mask, 0]

test_ids = [os.path.basename(p) for p in test_filenames]
sub_df = pd.DataFrame({"Id": test_ids, "Label": new_preds})

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print(sub_df.head())
print(
    f"Wrote submission to: {sub_path}  rows={len(sub_df)}  cols={list(sub_df.columns)}"
)
