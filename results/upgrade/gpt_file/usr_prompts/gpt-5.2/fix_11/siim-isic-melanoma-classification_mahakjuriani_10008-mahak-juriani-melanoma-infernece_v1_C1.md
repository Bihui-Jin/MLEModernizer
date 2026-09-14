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
Predict whether a lesion is malignant (0 denotes **benign**, and 1 indicates **malignant**).

## Metric
Area under the ROC curve.

## Submission Format
For each `image_name` in the test set, you must predict the probability (`target`) that the sample is **malignant**. The file should contain a header and have the following format:

```
image_name,target
ISIC_0052060,0.7
ISIC_0052349,0.9
ISIC_0058510,0.8
ISIC_0073313,0.5
ISIC_0073502,0.5
etc.
```

## Dataset 
The images are provided in DICOM format.

Images are also provided in JPEG and TFRecord format (in the `jpeg` and `tfrecords` directories, respectively). Images in TFRecord format have been resized to a uniform 1024x1024.

Metadata is also provided outside of the DICOM format, in CSV files. See the `Columns` section for a description.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `image_name` - unique identifier, points to filename of related DICOM image
- `patient_id` - unique patient identifier
- `sex` - the sex of the patient (when unknown, will be blank)
- `age_approx` - approximate patient age at time of imaging
- `anatom_site_general_challenge` - location of imaged site
- `diagnosis` - detailed diagnosis information (train only)
- `benign_malignant` - indicator of malignancy of imaged lesion
- `target` - binarized version of the target variable

# 2. Python version

3.14

# 3. Installed packages

albumentations==2.0.8
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
timm==1.0.19
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
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
        input/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
        working/
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
```

-> data/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/siim-isic-melanoma-classification/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> data/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> input/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> (stopped after 10 files for performance)

# 5. Target score

0.8224236665538093

# 6. Current score

0.58396

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.55872) has done: 'Your timeout is dominated by (1) decoding/resizing/normalizing ~4k test JPEGs at 384px in Python/OpenCV and (2) `torch.compile(mode="max-autotune")`, which can spend minutes compiling for a small 1‑epoch run. I keep the exact same model, loss, training loop, and data selection, but remove compile and reduce per-sample CPU overhead by caching deterministic resized+normalized images in memory (safe because each file is read once and augmentations are only applied on train). I also switch the training DataLoader to `shuffle=False` because the subset sampling is already randomized deterministically; this preserves the same data points and objective while avoiding extra sampler overhead. Finally, I set DataLoader `prefetch_factor` a bit higher and enable `inference_mode()` for prediction to reduce framework overhead while preserving outputs.'
- What this solution (achieved 0.53554) has done: 'To move your AUC up toward the 0.822 target with minimal disruption, I keep the exact same model, loss, and 1‑epoch training loop, but fix two high-impact issues that depress performance: (1) training currently starts from pretrained weights but never uses the test-time augmentations pipeline you defined, and (2) the head is trained with MSE on logits, so the learned scale can be poor—so I only adjust **prediction post-processing** by applying a temperature scaling (no training change) to better calibrate probabilities for ROC-AUC ranking. I also change the train subset sampling to be **stratified by target** (same subset size, still deterministic) so the one-epoch “warm” sees both classes more reliably, which typically improves ranking without changing the core approach. These are small, safe changes that should increase your score from ~0.56 toward the target band while keeping runtime under control and still producing a valid `submission.csv`.'
- What this solution (achieved 0.52161) has done: 'Your current gap to the target AUC is large (0.53554 → 0.82242), so we need a small but meaningful boost without changing the model/loss/training loop structure. The biggest low-risk gain is to fix a train/test mismatch: you already use Albumentations Normalize in both, but in the dataset you *skip* that normalization when `augmentations` is provided, feeding unnormalized float images into ResNet; we apply the same mean/std normalization in both paths so the pretrained backbone sees the expected distribution. I also remove the deterministic per-index NumPy RNG augmentation (which currently makes augmentations constant across epochs and duplicates Albumentations flips/rotations), relying on your existing `train_aug` for training-time randomness; this preserves the same augmentation intent while improving stochasticity and avoiding double-augmenting. Finally, I keep your temperature scaling but set it to 1.0 by default (calibration is unlikely to help ROC-AUC here and can hurt ranking), which should nudge AUC upward toward the target.'
- What this solution (achieved 0.58396) has done: 'Your current AUC (0.52161) is far below the target (0.82242), so we should make small, legitimate changes that improve ranking without changing the model, loss, or training loop structure. The biggest issue is that ResNet50 is being used with its default 1000-class classifier and then you add another linear layer on top; switching to a proper feature backbone (global pooled features) keeps the same overall approach (ResNet50 + dropout + linear) but removes an unnecessary bottleneck that heavily hurts AUC. I also keep the exact same augmentations but ensure the augmented path always returns normalized CHW float32 (Albumentations already does this) and keep submission alignment checks so the CSV is always valid. These changes are minimal, runtime-safe, and should move AUC upward toward the target band.'

# 9. Code solution

## === cell 0
import os
import random
import math
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations
import timm
from tqdm import tqdm




## === cell 1
class args:
    batch_size = 64
    image_size = 384

    epochs = 1
    num_workers = 2
    seed = 42

    train_subset_size = 4096  # deterministic fixed subset

    inference_temperature = 1.0




## === cell 2
def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))




## === cell 3
class MelanomaDataset(Dataset):
    def __init__(
        self, image_paths, dense_features, targets, augmentations, is_train: bool
    ):
        self.image_paths = image_paths
        self.dense_features = dense_features
        self.targets = targets
        self.augmentations = augmentations
        self.is_train = is_train

        self._fallback = np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)

        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self._inv_255 = np.float32(1.0 / 255.0)

        self._base_cache = {}

    def __len__(self):
        return len(self.image_paths)

    def _read_rgb_resized(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
        if img is None:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            img = self._fallback
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.shape[0] != args.image_size or img.shape[1] != args.image_size:
            img = cv2.resize(
                img, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA
            )
        return img

    def _normalize_chw(self, image_hwc_uint8: np.ndarray) -> np.ndarray:
        image = image_hwc_uint8.astype(np.float32, copy=False)
        image *= self._inv_255
        image = (image - self._mean) / self._std
        image = np.transpose(image, (2, 0, 1))
        return np.ascontiguousarray(image)

    def __getitem__(self, item):
        path = self.image_paths[item]

        base = self._base_cache.get(path)
        if base is None:
            base = self._read_rgb_resized(path)
            self._base_cache[path] = base

        image = base

        if self.augmentations is not None:
            out = self.augmentations(image=image)
            image = out["image"]
            if image.dtype != np.float32:
                image = image.astype(np.float32, copy=False)
            image = np.transpose(image, (2, 0, 1))
            image = np.ascontiguousarray(image)
        else:
            image = self._normalize_chw(image)

        features = self.dense_features[item, :]
        targets = self.targets[item]

        return {
            "image": torch.from_numpy(image),
            "features": torch.from_numpy(features),  # already float32
            "targets": torch.tensor(targets, dtype=torch.float32),
        }




## === cell 4
class MelanomaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "resnet50", pretrained=True, in_chans=3, num_classes=0, global_pool="avg"
        )
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(self.model.num_features, 1)

        self.step_scheduler_after = "epoch"

    def monitor_metrics(self, outputs, targets, loss):
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        accuracy = (preds == targets.view(-1, 1)).float().mean()
        metrics = {"accuracy": accuracy, "loss": loss}
        return metrics

    def optimizer_scheduler(self):
        opt = torch.optim.AdamW(self.parameters(), lr=2.5e-05, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return opt, sch

    def forward(self, image, features, targets=None):
        x = self.model(image)
        x = self.dropout(x)
        x = self.out(x)

        if targets is not None:
            loss = nn.MSELoss()(x, targets.view(-1, 1))
            metrics = self.monitor_metrics(x, targets, loss)
            return x, loss, metrics
        return x, 0, {}




## === cell 5
test_aug = albumentations.Compose(
    [
        albumentations.Resize(args.image_size, args.image_size, p=1),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

train_aug = albumentations.Compose(
    [
        albumentations.Resize(args.image_size, args.image_size, p=1),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.RandomRotate90(p=0.5),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)




## === cell 6
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(args.seed)

try:
    cv2.setNumThreads(0)
except Exception:
    pass

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

cpu = os.cpu_count() or 4
if args.num_workers <= 2:
    args.num_workers = min(8, max(2, cpu // 2))

DATA_ROOT = "/kaggle/input/siim-isic-melanoma-classification"
TRAIN_CSV = f"{DATA_ROOT}/train.csv"
TEST_CSV = f"{DATA_ROOT}/test.csv"
TRAIN_IMG_DIR = f"{DATA_ROOT}/jpeg/train"
TEST_IMG_DIR = f"{DATA_ROOT}/jpeg/test"

df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)

train_img_paths = [f"{TRAIN_IMG_DIR}/{x}.jpg" for x in df_train["image_name"].values]
test_img_paths = [f"{TEST_IMG_DIR}/{x}.jpg" for x in df_test["image_name"].values]

dense_features = []

X_train_dense = (
    df_train[dense_features].to_numpy(dtype=np.float32, copy=False)
    if len(dense_features) > 0
    else np.zeros((len(df_train), 0), dtype=np.float32)
)
X_test_dense = (
    df_test[dense_features].to_numpy(dtype=np.float32, copy=False)
    if len(dense_features) > 0
    else np.zeros((len(df_test), 0), dtype=np.float32)
)

y_train = df_train["target"].to_numpy(dtype=np.float32, copy=False)

n_train = len(train_img_paths)
subset_n = min(args.train_subset_size, n_train)
rng = np.random.RandomState(args.seed)

pos_idx = np.where(y_train > 0.5)[0]
neg_idx = np.where(y_train <= 0.5)[0]
rng.shuffle(pos_idx)
rng.shuffle(neg_idx)

prev = float(y_train.mean()) if n_train > 0 else 0.0
n_pos = int(round(subset_n * prev))
n_pos = max(32, n_pos) if len(pos_idx) > 0 else 0
n_pos = min(n_pos, len(pos_idx))
n_neg = subset_n - n_pos
n_neg = min(n_neg, len(neg_idx))
if n_pos + n_neg < subset_n:
    need = subset_n - (n_pos + n_neg)
    if len(neg_idx) - n_neg > 0:
        add = min(need, len(neg_idx) - n_neg)
        n_neg += add
        need -= add
    if need > 0 and len(pos_idx) - n_pos > 0:
        add = min(need, len(pos_idx) - n_pos)
        n_pos += add
        need -= add

subset_idx = np.concatenate([pos_idx[:n_pos], neg_idx[:n_neg]])
rng.shuffle(subset_idx)

train_img_paths_sub = [train_img_paths[i] for i in subset_idx]
X_train_dense_sub = X_train_dense[subset_idx]
y_train_sub = y_train[subset_idx]

train_dataset = MelanomaDataset(
    image_paths=train_img_paths_sub,
    dense_features=X_train_dense_sub,
    targets=y_train_sub,
    augmentations=train_aug,
    is_train=True,
)

test_dataset = MelanomaDataset(
    image_paths=test_img_paths,
    dense_features=X_test_dense,
    targets=np.ones(len(test_img_paths), dtype=np.float32),
    augmentations=test_aug,
    is_train=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MelanomaModel().to(device)
optimizer, scheduler = model.optimizer_scheduler()

if torch.cuda.is_available():
    model = model.to(memory_format=torch.channels_last)

_loader_kwargs = dict(
    num_workers=args.num_workers,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
    persistent_workers=(args.num_workers > 0),
    prefetch_factor=4 if args.num_workers > 0 else None,
)

if _loader_kwargs["prefetch_factor"] is None:
    _loader_kwargs.pop("prefetch_factor")

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    **_loader_kwargs,
)

model.train()
for epoch in range(args.epochs):
    pbar = tqdm(train_loader, total=len(train_loader), mininterval=0.5)
    for i, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        if torch.cuda.is_available():
            images = images.contiguous(memory_format=torch.channels_last)

        features = batch["features"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs, loss, metrics = model(images, features, targets)
        loss.backward()
        optimizer.step()

        if (i & 31) == 0:
            pbar.set_description(
                f"epoch {epoch+1}/{args.epochs} loss={loss.item():.4f} acc={metrics['accuracy'].item():.4f}"
            )

    scheduler.step(epoch + 1)

test_loader = DataLoader(
    test_dataset,
    batch_size=2 * args.batch_size,
    shuffle=False,
    **_loader_kwargs,
)

model.eval()
final_test_predictions = []

with torch.inference_mode():
    for batch in tqdm(test_loader, total=len(test_loader), mininterval=0.5):
        images = batch["image"].to(device, non_blocking=True)
        if torch.cuda.is_available():
            images = images.contiguous(memory_format=torch.channels_last)

        features = batch["features"].to(device, non_blocking=True)
        logits, _, _ = model(images, features, targets=None)

        logits = logits / float(args.inference_temperature)

        probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        final_test_predictions.extend(probs.tolist())

assert len(final_test_predictions) == len(df_test), (
    len(final_test_predictions),
    len(df_test),
)

df_sub = df_test[["image_name"]].copy()
df_sub["target"] = np.asarray(final_test_predictions, dtype=np.float32)
df_sub["target"] = df_sub["target"].clip(0.0, 1.0)
df_sub.to_csv("submission.csv", index=False)

df_sub.head()
