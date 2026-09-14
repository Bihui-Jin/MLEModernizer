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
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
import concurrent.futures

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True  # enable cuDNN autotuning for faster GPU ops
np.random.seed(42)
torch.manual_seed(42)

torch.set_num_threads(min(os.cpu_count() or 1, 8))




## === cell 1
base_path = "../input/siim-isic-melanoma-classification"
train_csv_path = os.path.join(base_path, "train.csv")
test_csv_path = os.path.join(base_path, "test.csv")
sample_sub_path = os.path.join(base_path, "sample_submission.csv")
train_img_dir = os.path.join(base_path, "jpeg/train")
test_img_dir = os.path.join(base_path, "jpeg/test")
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
sample_submission = pd.read_csv(sample_sub_path)




## === cell 2
def _load_and_resize(name_img_dir):
    """Load a single JPEG, convert to RGB, resize to 32×32 and return a float32 array."""
    name, img_dir = name_img_dir
    img_path = os.path.join(img_dir, f"{name}.jpg")
    img = Image.open(img_path).convert("RGB")
    img = img.resize((32, 32), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def load_and_preprocess(image_names, img_dir, cache_path):
    """
    Load and resize images in parallel using a ProcessPoolExecutor.
    Results are cached to ``cache_path`` so subsequent runs read the pre‑processed
    NumPy array directly, eliminating the expensive I/O and resize step.
    The preprocessing logic is identical to the original implementation.
    """
    if os.path.exists(cache_path):
        return np.load(cache_path)

    n = len(image_names)
    imgs = np.empty((n, 32, 32, 3), dtype=np.float32)

    max_workers = min(8, os.cpu_count() or 1)  # limit to avoid oversubscription
    chunksize = 256
    args = [(name, img_dir) for name in image_names]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, arr in enumerate(
            executor.map(_load_and_resize, args, chunksize=chunksize)
        ):
            imgs[i] = arr

    np.save(cache_path, imgs)
    return imgs




## === cell 3
train_cache = os.path.join(train_img_dir, "train_images_cache.npy")
train_images = load_and_preprocess(
    train_df["image_name"].values, train_img_dir, train_cache
)
train_targets = train_df["target"].values.astype(np.int64)
X_train, X_val, y_train, y_val = train_test_split(
    train_images, train_targets, test_size=0.2, random_state=42, stratify=train_targets
)




## === cell 4
def to_tensor(x):
    x = np.transpose(x, (0, 3, 1, 2))  # N C H W
    return torch.from_numpy(x)


train_X = to_tensor(X_train)
val_X = to_tensor(X_val)
train_y = torch.from_numpy(y_train)
val_y = torch.from_numpy(y_val)




## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 4)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.adapt = nn.AdaptiveMaxPool2d((5, 7))
        self.fc1 = nn.Linear(16 * 5 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = self.adapt(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.07)




## === cell 6
batch_size = 256
train_loader = DataLoader(
    TensorDataset(train_X, train_y),
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    prefetch_factor=2,
    persistent_workers=True,
)
val_loader = DataLoader(
    TensorDataset(val_X, val_y),
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
    prefetch_factor=2,
    persistent_workers=True,
)
n_epochs = 11
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(train_loader.dataset)

    model.eval()
    val_probs = []
    val_labels = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            val_probs.extend(probs)
            val_labels.extend(yb.numpy())
    val_auc = roc_auc_score(val_labels, val_probs)
    if (epoch + 1) % 2 == 0:
        print(
            f"Epoch {epoch+1}/{n_epochs} - loss: {epoch_loss:.4f} - val AUC: {val_auc:.4f}"
        )




## === cell 7
test_cache = os.path.join(test_img_dir, "test_images_cache.npy")
test_images = load_and_preprocess(
    test_df["image_name"].values, test_img_dir, test_cache
)
test_tensor = to_tensor(test_images)

test_loader = DataLoader(
    TensorDataset(test_tensor),
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=2,
    prefetch_factor=2,
)

model.eval()
test_probs = []
with torch.no_grad():
    for (xb,) in test_loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        test_probs.extend(probs)

sample_submission["target"] = np.array(test_probs)
sample_submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
