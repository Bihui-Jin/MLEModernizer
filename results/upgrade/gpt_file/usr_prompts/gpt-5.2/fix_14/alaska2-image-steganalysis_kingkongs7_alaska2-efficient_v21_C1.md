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

0.59266

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.57635) has done: 'Your code didn’t yield a score mainly because it’s set up to train EfficientNet-B0 from scratch-ish (only ImageNet pretrained backbone) on a very large dataset for 2 epochs at 512px, which is unlikely to finish within the 600s budget and thus never reaches the submission-writing cell. I keep the same model/loss/training loop, but make the run reliably complete by (1) reducing image size to 256 and (2) using a much smaller but still class-balanced sample per folder so training finishes quickly and the notebook always writes `submission.csv`. I also make DataLoader seeding deterministic per worker (no semantic change) and avoid the `pip install efficientnet_pytorch` path by using `torchvision`’s EfficientNet-B0 if the external package isn’t present (same architecture family, but this is only to ensure the environment doesn’t stall). These minimal changes prioritize getting a valid submission first; once you can measure a current score, we can tune toward the target more precisely.'
- What this solution (achieved 0.57242) has done: 'Your current score (0.57635) is far below the target (0.81393), so we should improve generalization while keeping your core model/training loop intact. The biggest low-risk issue is that you currently take the first `sample_size` files per folder (sorted), which can create an unrepresentative subset; we instead randomly sample from the full folder deterministically, keeping the same sample_size/val split and the same 4-class training objective. We also fix a small metric bug: the weighted-AUC integration should include boundary points (`<=`/`>=`) and should explicitly anchor each TPR segment; this won’t change training, but it makes validation feedback accurate. Finally, we apply the exact same “stego_score = sum of non-cover probabilities” but use logits-based computation (`logsumexp`) for slightly more stable ranking (same semantics: monotonic in stego probability), which typically nudges AUC upward without changing architecture or training.'
- What this solution (achieved 0.58281) has done: 'We’re far below the target (0.57242 vs 0.81393), so we should improve generalization without changing your core model/loss/loop. The smallest high-impact change is to increase the effective training signal while staying within the runtime budget by (1) modestly increasing the per-class sample size and (2) increasing `num_epochs` from 2 to 3, leaving everything else intact. We also make a tiny but meaningful train-time fix by matching the same simple TTA used at test time (original + flips) during validation scoring only (not changing training), which typically improves score stability and correlates better with leaderboard behavior. All changes preserve the same architecture, cross-entropy objective, and prediction semantics, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.5866) has done: 'Your current score (0.58281) is far below the target (0.81393), so we should increase performance with the smallest changes that keep your same EfficientNet-B0 + cross-entropy training loop and the same “stego score = non-cover vs cover from logits” submission semantics. The biggest low-risk gain here is fixing an evaluation/train mismatch: you train on highly augmented JPEG-compressed images but validate on clean images, which can mislead model selection and hurt generalization; we make validation use the same resize+compression (but no flips) to better match the JPEG variability described in the dataset. We also align the torchvision EfficientNet-B0 preprocessing to its ImageNet weights (mean/std normalization) without changing architecture or loss, which usually provides a meaningful boost for pretrained backbones. Finally, we slightly raise the learning rate (same optimizer) to better adapt within only 3 epochs on a sampled dataset, which is a minimal knob that tends to improve undertraining without changing the core approach.'
- What this solution (achieved 0.56779) has done: 'Main bottlenecks are (1) training from scratch when the checkpoint is missing, (2) slow JPEG decode + Albumentations per-sample CPU work, and (3) extra forward passes from TTA during validation/test. To finish under 600s without changing the model, loss, or training/eval semantics, the script below (a) makes execution “inference-only by default” when no checkpoint exists (so it won’t spend minutes training in a submission run), (b) enables safe CUDA/cuDNN autotuning and TF32 (negligible FP diffs) to accelerate EfficientNet, and (c) speeds up dataloading with larger worker pool, prefetching, pinned memory, and OpenCV thread tuning while keeping deterministic seeds. It also removes a small amount of avoidable overhead (vectorize basename, avoid storing full logits list on CPU) while keeping identical predictions.'
- What this solution (achieved 0.58167) has done: 'Your current gap to target is large (0.56779 vs 0.81393), and the biggest score-killer in the latest script is that it often runs inference-only from a random head when no checkpoint is found. To move toward the target while keeping the same model/loss/loop, I re-enable a short, guaranteed training run when no checkpoint exists by shrinking the training subset and epochs so it reliably fits the 600s budget. I also remove test-time `ImageCompression(p=1.0)` because it injects *random* JPEG artifacts into every test image and makes predictions noisy (worse AUC); instead, we keep deterministic resize+normalize at test while leaving the core scoring logic unchanged. All changes preserve the same EfficientNet-B0 classifier, cross-entropy training, and “stego score from logits” submission semantics, and still write a valid `submission.csv`.'
- What this solution (achieved 0.58892) has done: 'Your current score is far below the target (0.58167 vs 0.81393), so we should improve generalization while keeping your same EfficientNet-B0 + cross-entropy training loop and the same “stego score from logits” submission semantics. The smallest high-impact change is to make training see *all three stego algorithms as the same positive class* by oversampling the single Cover class (binary-balanced training) while still keeping the model output as 4 logits and the exact same loss/training loop (just remapping labels per batch). This aligns the supervised signal with the competition metric (cover vs stego), usually giving a sizable AUC lift without changing architecture or loss. We also make train augmentations match the stated JPEG QFs more tightly (75/90/95) to reduce augmentation-domain mismatch while keeping the same augmentation type. Submission writing and format remain unchanged.'
- What this solution (achieved 0.58765) has done: 'You’re far below the target (0.58892 vs 0.81393), so we should push score upward with the smallest safe changes that don’t alter your core model/loss/loop. The biggest low-risk issue is an augmentation-domain mismatch: you train with JPEG artifacts only 50% of the time, but your validation always applies compression; we make training compression deterministic (p=1.0) and restricted to the known QFs (75/90/95) so the model consistently learns the right artifacts. Second, your current binary training only updates logits 0/1, leaving logits 2/3 essentially random; we keep the exact same training objective but compute the stego score from only logits [0,1] to remove noise from unused heads. These two changes are minimal, keep the same architecture and training approach, and should move AUC upward while still finishing and writing a valid `submission.csv`.'
- What this solution (achieved 0.59266) has done: 'Your current score (0.58765) is far below the target (0.81393), so we should improve generalization with the smallest changes that keep your EfficientNet-B0 + CE loop and “stego score from logits” submission semantics intact. The main issue is a train/test domain mismatch: you train with always-on JPEG compression but test with none, so I make test-time preprocessing also apply deterministic JPEG recompression using the known QFs (75/90/95), matching your validation/train regime. Second, your validation loss is currently computed on 4-class labels while you actually train a binary head (logits 0/1), which can mislead checkpoint selection; I compute validation loss on the same binary head/labels (no change to training). These changes are minimal, keep architecture/loss/training approach the same, and should move the score upward while still producing a valid `submission.csv`.'

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

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

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
torch.backends.cudnn.benchmark = (
    True  # faster convolution algorithm selection per input shape
)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



## === cell 2
data_dir = "../input/alaska2-image-steganalysis"

sample_size = 6000  # per folder
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
        self.fns = self.data["ImageFileName"].to_numpy(dtype=object)
        self.labels = self.data["Label"].to_numpy(dtype=np.int64)
        self.augment = augmentations

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        fn = self.fns[idx]
        label = int(self.labels[idx])

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
        A.ImageCompression(quality_range=(75, 95), quality_value=[75, 90, 95], p=1.0),
        A.ToFloat(max_value=255.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=1.0),
        ToTensorV2(),
    ]
)

AUGMENTATIONS_VALID = A.Compose(
    [
        A.Resize(img_size, img_size),
        A.ImageCompression(quality_range=(75, 95), quality_value=[75, 90, 95], p=1.0),
        A.ToFloat(max_value=255.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=1.0),
        ToTensorV2(),
    ]
)

AUGMENTATIONS_TEST = A.Compose(
    [
        A.Resize(img_size, img_size),
        A.ImageCompression(quality_range=(75, 95), quality_value=[75, 90, 95], p=1.0),
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
batch_size = 16
num_workers = min(8, (os.cpu_count() or 4))
prefetch_factor = 4 if num_workers > 0 else None


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
    prefetch_factor=prefetch_factor,
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
    prefetch_factor=prefetch_factor,
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

num_epochs = 2 if not loaded else 4

train_loss, val_loss = [], []


def stego_score_from_logits(logits_tensor: torch.Tensor) -> torch.Tensor:
    logits2 = logits_tensor[:, :2]
    log_p_stego_unnorm = logits2[:, 1]
    log_p_cover_unnorm = logits2[:, 0]
    return log_p_stego_unnorm - torch.logaddexp(log_p_stego_unnorm, log_p_cover_unnorm)


def tta_forward_logits(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    x_o = inputs
    x_h = inputs.flip(2)
    x_w = inputs.flip(3)
    x = torch.cat([x_o, x_h, x_w], dim=0)
    out = model(x)
    b = inputs.shape[0]
    out_o = out[:b]
    out_h = out[b : 2 * b]
    out_w = out[2 * b :]
    return 0.5 * out_o + 0.25 * out_h + 0.25 * out_w


cover_idx = np.where(train_df["Label"].to_numpy() == 0)[0]
stego_idx = np.where(train_df["Label"].to_numpy() != 0)[0]
rng_local = np.random.default_rng(seed)
if len(cover_idx) > 0 and len(stego_idx) > 0:
    cover_oversampled_idx = rng_local.choice(
        cover_idx, size=len(stego_idx), replace=True
    )
    train_df_bal = (
        pd.concat(
            [train_df.iloc[stego_idx], train_df.iloc[cover_oversampled_idx]],
            axis=0,
            ignore_index=True,
        )
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    print(
        "Balanced train_df size:",
        len(train_df_bal),
        " (stego:",
        len(stego_idx),
        "cover_oversampled:",
        len(cover_oversampled_idx),
        ")",
    )
else:
    train_df_bal = train_df.copy()
    print(
        "Warning: could not balance (missing cover or stego); using original train_df."
    )

train_dataset_bal = Alaska2Dataset(train_df_bal, augmentations=AUGMENTATIONS_TRAIN)
train_loader_bal = torch.utils.data.DataLoader(
    train_dataset_bal,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g,
    persistent_workers=(num_workers > 0),
    prefetch_factor=prefetch_factor,
)

for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch, num_epochs - 1))
    print("-" * 10)

    model.train()
    running_loss = 0.0
    tk0 = tqdm(train_loader_bal, total=len(train_loader_bal), mininterval=0.5)
    for inputs, labels in tk0:
        inputs = inputs.to(device, dtype=torch.float, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        labels_train = labels.clone()
        labels_train[labels_train != 0] = 1

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)

        outputs_bin = outputs[:, :2]
        loss = criterion(outputs_bin, labels_train)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        tk0.set_postfix(loss=float(loss.item()))

    epoch_loss = running_loss / max(1, len(train_loader_bal))
    train_loss.append(epoch_loss)
    print("Training Loss: {:.8f}".format(epoch_loss))

    model.eval()
    running_loss = 0.0
    y, stego_scores = [], []
    correct = 0
    total = 0

    tk1 = tqdm(valid_loader, total=len(valid_loader), mininterval=0.5)
    with torch.inference_mode():
        for inputs, labels in tk1:
            inputs = inputs.to(device, dtype=torch.float, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)

            labels_val_bin = labels.clone()
            labels_val_bin[labels_val_bin != 0] = 1
            out_o = model(inputs)
            loss = criterion(out_o[:, :2], labels_val_bin)

            outputs = tta_forward_logits(model, inputs)

            pred_class = outputs.argmax(1)
            correct += int((pred_class == labels).sum().item())
            total += int(labels.numel())

            y.extend(labels.cpu().numpy().astype(int).tolist())
            stego_scores.extend(stego_score_from_logits(outputs).cpu().numpy().tolist())

            running_loss += float(loss.item())
            tk1.set_postfix(loss=float(loss.item()))

    epoch_loss = running_loss / max(1, len(valid_loader))
    val_loss.append(epoch_loss)

    y_np = np.asarray(y)
    y_bin = (y_np != 0).astype(int)
    stego_scores = np.asarray(stego_scores, dtype=np.float64)

    auc_score = alaska_weighted_auc(y_bin, stego_scores)
    acc = (correct / max(1, total)) * 100.0

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
        self.fns = self.data["ImageFileName"].to_numpy(dtype=object)
        self.augment = augmentations

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        fn = self.fns[idx]
        im = cv2.imread(fn)
        if im is None:
            raise FileNotFoundError(f"Could not read image: {fn}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        if self.augment is not None:
            im = self.augment(image=im)["image"]

        return im


test_filenames = sorted(glob(f"{data_dir}/Test/*.jpg"))
test_df = pd.DataFrame({"ImageFileName": list(test_filenames)})

batch_size = 32
num_workers = min(8, (os.cpu_count() or 4))
prefetch_factor = 4 if num_workers > 0 else None

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
    prefetch_factor=prefetch_factor,
)

print("Test images:", len(test_df))



## === cell 11
model.eval()

all_scores = np.empty((len(test_dataset),), dtype=np.float32)

offset = 0
tk0 = tqdm(test_loader, total=len(test_loader), mininterval=0.5)
with torch.inference_mode():
    for inputs in tk0:
        b = inputs.shape[0]
        inputs = inputs.to(device, dtype=torch.float, non_blocking=True)
        outputs = tta_forward_logits(model, inputs)

        scores = (
            stego_score_from_logits(outputs).detach().cpu().numpy().astype(np.float32)
        )
        all_scores[offset : offset + b] = scores
        offset += b

assert offset == len(test_dataset)

test_df["Id"] = test_df["ImageFileName"].map(os.path.basename)
test_df["Label"] = all_scores

sub = test_df[["Id", "Label"]].copy()
sub = sub.sort_values("Id").reset_index(drop=True)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
assert sub.shape[0] == 5000, f"Expected 5000 test rows, got {sub.shape[0]}"
assert list(sub.columns) == ["Id", "Label"]
