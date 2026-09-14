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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

try:
    import torchvision

    torchvision.set_image_backend("accimage")
except Exception:
    pass

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)

try:
    torch.backends.image.set_image_backend("torchvision")
except Exception:
    pass




## === cell 1
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 2
BASE = "/kaggle/input/siim-isic-melanoma-classification"
TRAIN_CSV = os.path.join(BASE, "train.csv")
TEST_CSV = os.path.join(BASE, "test.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "jpeg", "train")
TEST_IMG_DIR = os.path.join(BASE, "jpeg", "test")

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
train.shape, test.shape




## === cell 3
def append_ext(fn):
    return fn + ".jpg"


train["image_name"] = train["image_name"].apply(append_ext)
test["image_name"] = test["image_name"].apply(append_ext)

train[["image_name", "target"]].head()




## === cell 4
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
for tr_idx, va_idx in sss.split(train, train["target"]):
    train_df = train.iloc[tr_idx].reset_index(drop=True)
    valid_df = train.iloc[va_idx].reset_index(drop=True)

train_df.shape, valid_df.shape




## === cell 5
img_size = 256

train_tfms = transforms.Compose(
    [
        transforms.ConvertImageDtype(torch.float32),  # scales to [0,1]
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=40),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.2, 0.2),
            shear=20,
            scale=(0.8, 1.2),
        ),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

valid_tfms = transforms.Compose(
    [
        transforms.ConvertImageDtype(torch.float32),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_tfms = valid_tfms


class MelanomaDataset(Dataset):
    def __init__(
        self, df, img_dir, transforms_=None, has_target=True, cache_transformed=False
    ):
        df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms_ = transforms_
        self.has_target = has_target
        self.cache_transformed = cache_transformed
        self._cache = {} if cache_transformed else None

        self.image_names = df["image_name"].to_numpy()
        base = img_dir if img_dir.endswith(os.sep) else (img_dir + os.sep)
        self.paths = (base + self.image_names).astype(object)

        if has_target:
            self.targets = df["target"].to_numpy(dtype=np.int64)
        else:
            self.targets = None

    def __len__(self):
        return self.image_names.shape[0]

    def _load_image(self, path):
        return read_image(path, mode=ImageReadMode.RGB)  # uint8, [C,H,W]

    def __getitem__(self, idx):
        if self._cache is not None:
            cached = self._cache.get(idx, None)
            if cached is not None:
                return cached

        img = self._load_image(self.paths[idx])

        if self.transforms_ is not None:
            img = self.transforms_(img)

        if self.has_target:
            y = int(self.targets[idx])
            out = (img, y)
        else:
            out = (img, self.image_names[idx])

        if self._cache is not None:
            self._cache[idx] = out
        return out




## === cell 6
batch_size = 32


def seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

if torch.cuda.is_available():
    num_workers = min(8, (os.cpu_count() or 8))
    pin_memory = True
    persistent_workers = True
    prefetch_factor = 4
else:
    num_workers = 0
    pin_memory = False
    persistent_workers = False
    prefetch_factor = None

train_ds = MelanomaDataset(
    train_df,
    TRAIN_IMG_DIR,
    transforms_=train_tfms,
    has_target=True,
    cache_transformed=False,
)
valid_ds = MelanomaDataset(
    valid_df,
    TRAIN_IMG_DIR,
    transforms_=valid_tfms,
    has_target=True,
    cache_transformed=True,
)
test_ds = MelanomaDataset(
    test, TEST_IMG_DIR, transforms_=test_tfms, has_target=False, cache_transformed=True
)

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
    worker_init_fn=seed_worker if num_workers > 0 else None,
    generator=g,
)
valid_loader = DataLoader(
    valid_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
    worker_init_fn=seed_worker if num_workers > 0 else None,
    generator=g,
)
test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
    worker_init_fn=seed_worker if num_workers > 0 else None,
    generator=g,
)

len(train_loader), len(valid_loader), len(test_loader)




## === cell 7
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
backbone = efficientnet_b0(weights=weights)

in_features = backbone.classifier[1].in_features
backbone.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(in_features, 2))
model2 = backbone.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model2.parameters(), lr=1e-4)




## === cell 8
def run_epoch(model, loader, train_mode=True):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    context = torch.enable_grad() if train_mode else torch.inference_mode()
    with context:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)
            if train_mode:
                loss.backward()
                optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total += bs
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()

    return total_loss / total, correct / total


history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
epochs = 3

for ep in range(1, epochs + 1):
    tr_loss, tr_acc = run_epoch(model2, train_loader, train_mode=True)
    va_loss, va_acc = run_epoch(model2, valid_loader, train_mode=False)
    history["loss"].append(tr_loss)
    history["acc"].append(tr_acc)
    history["val_loss"].append(va_loss)
    history["val_acc"].append(va_acc)
    print(
        f"Epoch {ep}/{epochs} - loss: {tr_loss:.4f} acc: {tr_acc:.4f} - val_loss: {va_loss:.4f} val_acc: {va_acc:.4f}"
    )




## === cell 9
model2.eval()
all_image_names = []
all_probs_1 = []

with torch.inference_mode():
    for x, names in test_loader:
        x = x.to(device, non_blocking=True)
        logits = model2(x)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_probs_1.append(probs)
        all_image_names.extend(names)

all_probs_1 = np.concatenate(all_probs_1, axis=0)
len(all_image_names), all_probs_1.shape




## === cell 10
image_name_no_ext = (
    pd.Series(all_image_names).str.replace(".jpg", "", regex=False).tolist()
)

pred_df = pd.DataFrame(
    {"image_name": image_name_no_ext, "target": all_probs_1.astype(float)}
)

pred_df = (
    pred_df.set_index("image_name")
    .loc[test["image_name"].str.replace(".jpg", "", regex=False)]
    .reset_index()
)

pred_df.head(), pred_df.shape




## === cell 11
sub_path = "efficient_net.csv"
pred_df.to_csv(sub_path, index=False)

print("Wrote:", sub_path)
print(pred_df.columns.tolist())
print(pred_df.isna().sum().to_dict())
print(pred_df["target"].min(), pred_df["target"].max())
