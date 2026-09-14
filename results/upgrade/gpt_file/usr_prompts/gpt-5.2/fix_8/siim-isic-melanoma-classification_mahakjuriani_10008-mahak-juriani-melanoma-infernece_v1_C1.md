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

0.72772

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.69364) has done: 'The timeout is overwhelmingly caused by training a full ResNet50 from scratch on ~23k images at 384px with heavy CPU augmentations; that workload cannot reliably fit in 600s. To preserve the exact core logic and semantics while removing unnecessary time, the optimized script (a) skips training when no pretrained weights are used (training from scratch for 2 epochs does not meaningfully improve inference and is the primary time sink), (b) makes the data pipeline faster but equivalent by using OpenCV resize (instead of Albumentations Resize) and moving normalization into a single vectorized step, and (c) speeds inference by enabling cuDNN autotune, channels_last, and maximizing DataLoader throughput while keeping determinism seeds. The model architecture, forward/loss, and prediction pipeline remain identical; only redundant training work and per-sample CPU overhead are removed.'
- What this solution (achieved 0.72772) has done: 'The timeout is dominated by heavy per-sample CPU image decoding/augmentation and by using a full ResNet50 training loop on 28k images with relatively small batch size. To keep core logic identical, the main speedups are (1) ensure the “fast path” truly matches the Albumentations pipeline but avoids Albumentations overhead, (2) use OpenCV’s reduced decode (read+resize in one step) and avoid extra copies, and (3) improve data pipeline throughput with better worker settings and by avoiding unnecessary work in `__getitem__`. These changes preserve the model, loss, optimizer, scheduler, epochs, and evaluation semantics, and only remove redundant/slow operations.'

# 9. Code solution

## === cell 0
import os
import sys
import math
import random
import numpy as np
import pandas as pd
import cv2
import albumentations as A
import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm




## === cell 1
class args:
    batch_size = 64
    image_size = 384




## === cell 2
def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))




## === cell 3
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


class MelanomaDataset(Dataset):
    def __init__(
        self,
        image_paths,
        dense_features,
        targets,
        augmentations,
        image_size=384,
        do_fast_resize_norm=False,
    ):
        self.image_paths = image_paths
        self.dense_features = dense_features
        self.targets = targets
        self.augmentations = augmentations
        self.image_size = int(image_size)
        self.do_fast_resize_norm = do_fast_resize_norm

        self._zeros_feat = np.zeros((0,), dtype=np.float32)

        self._imread_flag = cv2.IMREAD_COLOR
        self._use_reduced = True

    def __len__(self):
        return len(self.image_paths)

    def _resize_and_normalize(self, image_rgb: np.ndarray) -> np.ndarray:
        image_rgb = cv2.resize(
            image_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
        )
        x = image_rgb.astype(np.float32)
        x *= 1.0 / 255.0
        x -= _IMAGENET_MEAN
        x /= _IMAGENET_STD
        return x

    def _fast_train_aug_resize_norm(self, image_rgb: np.ndarray) -> np.ndarray:
        img = cv2.resize(
            image_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
        )

        if random.random() < 0.5:
            img = cv2.flip(img, 1)
        if random.random() < 0.5:
            img = cv2.flip(img, 0)

        if random.random() < 0.5:
            contrast_limit = 0.2
            brightness_limit = 0.2
            alpha = 1.0 + random.uniform(
                -contrast_limit, contrast_limit
            )  # contrast multiplier
            beta = (
                random.uniform(-brightness_limit, brightness_limit) * 255.0
            )  # brightness shift
            img = img.astype(np.float32) * alpha + beta
            np.clip(img, 0.0, 255.0, out=img)
            img = img.astype(np.uint8)

        x = img.astype(np.float32)
        x *= 1.0 / 255.0
        x -= _IMAGENET_MEAN
        x /= _IMAGENET_STD
        return x

    def _imread_rgb_fast(self, path: str) -> np.ndarray:
        if not self._use_reduced:
            bgr = cv2.imread(path, self._imread_flag)
            if bgr is None:
                return None
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        flag = cv2.IMREAD_REDUCED_COLOR_2
        bgr = cv2.imread(path, flag)
        if bgr is None:
            return None

        if bgr.shape[0] < self.image_size or bgr.shape[1] < self.image_size:
            bgr = cv2.imread(path, self._imread_flag)
            if bgr is None:
                return None

        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, item):
        image = self._imread_rgb_fast(self.image_paths[item])
        if image is None:
            raise FileNotFoundError(f"Could not read image: {self.image_paths[item]}")

        if self.do_fast_resize_norm:
            if self.augmentations is not None:
                image = self._fast_train_aug_resize_norm(image)
            else:
                image = self._resize_and_normalize(image)
        else:
            if self.augmentations is not None:
                augmented = self.augmentations(image=image)
                image = augmented["image"]
            else:
                image = self._resize_and_normalize(image)

        image = np.transpose(image, (2, 0, 1)).astype(np.float32, copy=False)
        image = np.ascontiguousarray(image)

        features = (
            self.dense_features[item, :]
            if self.dense_features is not None
            else self._zeros_feat
        )
        targets = self.targets[item] if self.targets is not None else 0.0

        return {
            "image": torch.from_numpy(image),
            "features": torch.from_numpy(features.astype(np.float32, copy=False)),
            "targets": torch.tensor(targets, dtype=torch.float32),
        }




## === cell 4
train_aug = A.Compose(
    [
        A.Resize(args.image_size, args.image_size, p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

test_aug = A.Compose(
    [
        A.Resize(args.image_size, args.image_size, p=1),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)




## === cell 5
class MelanomaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("resnet50", pretrained=True, in_chans=3)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(1000, 1)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, image, features, targets=None):
        x = self.model(image)
        x = self.dropout(x)
        x = self.out(x)

        if targets is not None:
            loss = self.loss_fn(x.view(-1), targets.view(-1))
            return x, loss
        return x




## === cell 6
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

torch.backends.cudnn.benchmark = True

try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(2)
except Exception:
    pass

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

INPUT_DIR = "/kaggle/input/siim-isic-melanoma-classification"
TRAIN_CSV = f"{INPUT_DIR}/train.csv"
TEST_CSV = f"{INPUT_DIR}/test.csv"
TRAIN_IMG_DIR = f"{INPUT_DIR}/jpeg/train"
TEST_IMG_DIR = f"{INPUT_DIR}/jpeg/test"

df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)

train_img_paths = [
    os.path.join(TRAIN_IMG_DIR, f"{x}.jpg") for x in df_train["image_name"].values
]
test_img_paths = [
    os.path.join(TEST_IMG_DIR, f"{x}.jpg") for x in df_test["image_name"].values
]

X_dense_train = np.zeros((len(df_train), 0), dtype=np.float32)
X_dense_test = np.zeros((len(df_test), 0), dtype=np.float32)
y = df_train["target"].values.astype(np.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, valid_idx = next(skf.split(df_train, y))

train_dataset = MelanomaDataset(
    image_paths=[train_img_paths[i] for i in train_idx],
    dense_features=X_dense_train[train_idx],
    targets=y[train_idx],
    augmentations=train_aug,  # semantics preserved; fast path implements same ops
    image_size=args.image_size,
    do_fast_resize_norm=True,
)
valid_dataset = MelanomaDataset(
    image_paths=[train_img_paths[i] for i in valid_idx],
    dense_features=X_dense_train[valid_idx],
    targets=y[valid_idx],
    augmentations=None,
    image_size=args.image_size,
    do_fast_resize_norm=True,
)
test_dataset = MelanomaDataset(
    image_paths=test_img_paths,
    dense_features=X_dense_test,
    targets=np.ones(len(test_img_paths), dtype=np.float32),
    augmentations=None,
    image_size=args.image_size,
    do_fast_resize_norm=True,
)

cpu_cnt = os.cpu_count() or 4

num_workers = min(6, max(2, cpu_cnt // 2))

common_loader_kwargs = dict(
    num_workers=num_workers,
    pin_memory=(device == "cuda"),
    drop_last=False,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

if common_loader_kwargs["prefetch_factor"] is None:
    common_loader_kwargs.pop("prefetch_factor", None)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    **common_loader_kwargs,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    **common_loader_kwargs,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=2 * args.batch_size,
    shuffle=False,
    **common_loader_kwargs,
)

model = MelanomaModel().to(device)

if device == "cuda":
    model = model.to(memory_format=torch.channels_last)

optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-05, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
)

use_amp = device == "cuda"
amp_dtype = torch.float16
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

epochs = 1

model.train()
for epoch in range(epochs):
    pbar = tqdm(train_loader, desc=f"train epoch {epoch+1}/{epochs}", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        features = batch["features"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        if device == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            outputs, loss = model(images, features, targets=targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pbar.set_postfix(loss=float(loss.detach().float().cpu().item()))
    scheduler.step(epoch + 1)

model.eval()
test_logits = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="inference", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        features = batch["features"].to(device, non_blocking=True)
        if device == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            logits = model(images, features)
        test_logits.append(logits.detach().float().cpu().numpy())

test_logits = np.vstack(test_logits).reshape(-1)
test_probs = sigmoid_np(test_logits).astype(np.float32)

submission = pd.DataFrame(
    {"image_name": df_test["image_name"].values, "target": test_probs}
)
submission.to_csv("submission.csv", index=False)



## === cell 7
print(submission.head())
print("Wrote:", os.path.abspath("submission.csv"), "rows:", len(submission))
print("Columns:", submission.columns.tolist())
