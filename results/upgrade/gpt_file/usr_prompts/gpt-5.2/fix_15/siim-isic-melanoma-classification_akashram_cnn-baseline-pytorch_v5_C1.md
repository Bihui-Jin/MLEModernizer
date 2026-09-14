# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.8

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
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

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

try:
    import cv2

    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    from skimage.io import imread
    from skimage.transform import resize

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

train = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
test = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")

test_path = "../input/siim-isic-melanoma-classification/jpeg/test/"
train_path = "../input/siim-isic-melanoma-classification/jpeg/train/"

sample_submission = pd.read_csv(
    "../input/siim-isic-melanoma-classification/sample_submission.csv"
)

print(train.shape, test.shape, sample_submission.shape)
print(sample_submission.head())



## === cell 1
IMG_SIZE = 32

from concurrent.futures import ThreadPoolExecutor


def load_images_from_df(df, img_dir, img_size=32, verbose=True, cache_prefix=None):
    cache_fp = None
    if cache_prefix is not None:
        cache_fp = os.path.join("/kaggle/working", f"{cache_prefix}_{img_size}.mmap")
        meta_fp = cache_fp + ".meta.npy"
        if os.path.exists(cache_fp) and os.path.exists(meta_fp):
            meta = np.load(meta_fp, allow_pickle=True).item()
            if (
                meta.get("n") == len(df)
                and tuple(meta.get("shape", ())) == (len(df), img_size, img_size, 3)
                and meta.get("dtype") == "float32"
            ):
                x = np.memmap(
                    cache_fp,
                    mode="r",
                    dtype=np.float32,
                    shape=(len(df), img_size, img_size, 3),
                )
                if verbose:
                    print(f"Loaded memmap cache: {cache_fp} -> {x.shape} {x.dtype}")
                return x

    names = df["image_name"].to_numpy()
    fps = [os.path.join(img_dir, f"{name}.jpg") for name in names]
    n = len(fps)

    if _HAS_CV2:
        try:
            cv2.setUseOptimized(True)
        except Exception:
            pass
        try:
            cv2.setNumThreads(
                0
            )  # let OpenCV manage threads; we parallelize at file level
        except Exception:
            pass

        def _read_one(fp):
            try:
                img = cv2.imread(fp, cv2.IMREAD_COLOR)
                if img is None:
                    return None
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(
                    img, (img_size, img_size), interpolation=cv2.INTER_AREA
                )
                return img.astype(np.float32) * (1.0 / 255.0)
            except Exception:
                return None

        x = np.empty((n, img_size, img_size, 3), dtype=np.float32)
        missing = 0

        max_workers = min(32, (os.cpu_count() or 1))
        it = range(n)
        if verbose:
            it = tqdm(it, total=n, miniters=1024)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, img in zip(it, ex.map(_read_one, fps, chunksize=256)):
                if img is None:
                    x[i] = 0.0
                    missing += 1
                else:
                    x[i] = img
    else:
        x = np.empty((n, img_size, img_size, 3), dtype=np.float32)
        missing = 0
        it = range(n)
        if verbose:
            it = tqdm(it, total=n, miniters=256)
        for i in it:
            fp = fps[i]
            if not os.path.exists(fp):
                x[i] = 0.0
                missing += 1
                continue
            img = imread(fp)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[-1] == 4:
                img = img[..., :3]
            img = resize(
                img, (img_size, img_size, 3), anti_aliasing=True, preserve_range=True
            ).astype(np.float32)
            x[i] = img * (1.0 / 255.0)

    if verbose and missing > 0:
        print(f"Warning: {missing} images missing in {img_dir}")

    if cache_fp is not None:
        mm = np.memmap(cache_fp, mode="w+", dtype=np.float32, shape=x.shape)
        mm[:] = x
        mm.flush()
        np.save(
            meta_fp, {"n": n, "shape": x.shape, "dtype": "float32"}, allow_pickle=True
        )
        if verbose:
            print(f"Saved memmap cache: {cache_fp} (+ meta)")

    return x


train_y = train["target"].values
train_x = load_images_from_df(
    train, train_path, img_size=IMG_SIZE, verbose=True, cache_prefix="train_x_cache"
)
print("train_x:", train_x.shape, "train_y:", train_y.shape)



## === cell 2
i = 0
plt.figure(figsize=(8, 8))
plt.subplot(221), plt.imshow(train_x[i])
plt.subplot(222), plt.imshow(train_x[i + 25])
plt.subplot(223), plt.imshow(train_x[i + 50])
plt.subplot(224), plt.imshow(train_x[i + 75])
plt.tight_layout()



## === cell 3
train_x, val_x, train_y, val_y = train_test_split(
    train_x, train_y, test_size=0.2, random_state=42, stratify=train_y
)
print((train_x.shape, train_y.shape), (val_x.shape, val_y.shape))



## === cell 4
train_x = np.ascontiguousarray(np.transpose(train_x, (0, 3, 1, 2)), dtype=np.float32)
val_x = np.ascontiguousarray(np.transpose(val_x, (0, 3, 1, 2)), dtype=np.float32)

train_x = torch.from_numpy(train_x).pin_memory()
val_x = torch.from_numpy(val_x).pin_memory()

train_y = torch.from_numpy(train_y.astype(np.int64, copy=False)).pin_memory()
val_y = torch.from_numpy(val_y.astype(np.int64, copy=False)).pin_memory()

print(train_x.shape, train_y.shape, train_x.dtype, train_y.dtype)
print(val_x.shape, val_y.shape, val_x.dtype, val_y.dtype)



## === cell 5
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.use_deterministic_algorithms(False)




## === cell 6
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 4)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.adapt = nn.AdaptiveMaxPool2d((5, 7))
        self.fc1 = nn.Linear(16 * 5 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Sequential(nn.Linear(84, 2))

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = self.adapt(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




## === cell 7
model = Net()
optimizer = Adam(model.parameters(), lr=0.07)
criterion = CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = criterion.to(device)

print(device)
print(model)



## === cell 8
_train_x_dev = train_x.to(device, non_blocking=True)
_train_y_dev = train_y.to(device, non_blocking=True)
_val_x_dev = val_x.to(device, non_blocking=True)
_val_y_dev = val_y.to(device, non_blocking=True)

_last_train_logits = None
_last_val_logits = None


def train(epoch):
    global _last_train_logits, _last_val_logits

    model.train()
    optimizer.zero_grad(set_to_none=True)

    output_train = model(_train_x_dev)
    loss_train = criterion(output_train, _train_y_dev)

    model.eval()
    with torch.inference_mode():
        output_val = model(_val_x_dev)
        loss_val = criterion(output_val, _val_y_dev)

    model.train()

    train_losses.append(loss_train.item())
    val_losses.append(loss_val.item())

    loss_train.backward()
    optimizer.step()

    _last_train_logits = output_train.detach()
    _last_val_logits = output_val.detach()

    if epoch % 2 == 0:
        print(
            "Epoch :", epoch + 1, "\t", "loss :", float(loss_val.detach().cpu().item())
        )




## === cell 9
n_epochs = 11
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    train(epoch)

print("Final val loss:", val_losses[-1])



## === cell 10
model.eval()
with torch.inference_mode():
    if _last_train_logits is None:
        output = model(_train_x_dev).detach()
    else:
        output = _last_train_logits
    probs = F.softmax(output, dim=1).cpu().numpy()
    predictions = np.argmax(probs, axis=1)

print("Train accuracy:", accuracy_score(train_y.numpy(), predictions))



## === cell 11
model.eval()
with torch.inference_mode():
    if _last_val_logits is None:
        output = model(_val_x_dev).detach()
    else:
        output = _last_val_logits
    probs = F.softmax(output, dim=1).cpu().numpy()
    predictions = np.argmax(probs, axis=1)

print("Val accuracy:", accuracy_score(val_y.numpy(), predictions))



## === cell 12
test_x = load_images_from_df(
    test, test_path, img_size=IMG_SIZE, verbose=True, cache_prefix="test_x_cache"
)

test_x = np.ascontiguousarray(np.transpose(test_x, (0, 3, 1, 2)), dtype=np.float32)
test_x = torch.from_numpy(test_x).pin_memory()
print("test_x:", test_x.shape, test_x.dtype)

test_x = test_x.to(device, non_blocking=True)



## === cell 13
model.eval()
with torch.inference_mode():
    output = model(test_x)
    preds = F.softmax(output, dim=1)[:, 1].detach().cpu().numpy()

sub = sample_submission.copy()
sub["target"] = preds[: len(sub)]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
