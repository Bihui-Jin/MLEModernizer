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
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
from joblib import Parallel, delayed  # parallel image loading

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score




## === cell 1
train = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
test = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")
train_path = "../input/siim-isic-melanoma-classification/jpeg/train/"
test_path = "../input/siim-isic-melanoma-classification/jpeg/test/"
sample_submission = pd.read_csv(
    "../input/siim-isic-melanoma-classification/sample_submission.csv"
)




## === cell 2
IMG_SIZE = 32


def _load_and_resize(img_name, base_path):
    """Fast JPEG loading with Pillow and simple bilinear resize."""
    img_path = f"{base_path}{img_name}.jpg"
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img_array = np.array(img, dtype=np.float32) / 255.0  # scale to [0,1]
    return img_array


print("Loading training images …")
n_jobs = min(os.cpu_count() or 1, 8)  # cap to avoid oversubscription
train_imgs = Parallel(n_jobs=n_jobs, backend="loky")(
    delayed(_load_and_resize)(img_name, train_path)
    for img_name in tqdm(train["image_name"].values)
)
train_x = np.stack(train_imgs)  # (N, H, W, C)
train_y = train["target"].values.astype(np.int64)
print("train_x shape:", train_x.shape)




## === cell 3
i = 0
plt.figure(figsize=(8, 8))
plt.subplot(221)
plt.imshow(train_x[i])
plt.title("0")
plt.subplot(222)
plt.imshow(train_x[i + 25])
plt.title("25")
plt.subplot(223)
plt.imshow(train_x[i + 50])
plt.title("50")
plt.subplot(224)
plt.imshow(train_x[i + 75])
plt.title("75")
plt.show()




## === cell 4
train_x, val_x, train_y, val_y = train_test_split(
    train_x, train_y, test_size=0.2, random_state=42, stratify=train_y
)
print("After split:", train_x.shape, val_x.shape)




## === cell 5
train_x = np.transpose(train_x, (0, 3, 1, 2))
val_x = np.transpose(val_x, (0, 3, 1, 2))
train_x = torch.from_numpy(train_x).float()
val_x = torch.from_numpy(val_x).float()
train_y = torch.from_numpy(train_y).long()
val_y = torch.from_numpy(val_y).long()

if torch.cuda.is_available():
    train_x = train_x.cuda()
    val_x = val_x.cuda()
    train_y = train_y.cuda()
    val_y = val_y.cuda()

print("Tensor shapes:", train_x.shape, train_y.shape, val_x.shape, val_y.shape)




## === cell 6
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True  # speed‑up on GPU if available
torch.set_num_threads(4)  # limit intra‑op parallelism to avoid oversubscription




## === cell 7
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 4)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.adapt = nn.AdaptiveMaxPool2d((5, 7))
        self.fc1 = nn.Linear(16 * 5 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # 2 classes

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = self.adapt(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




## === cell 8
model = Net()
optimizer = Adam(model.parameters(), lr=0.07)
criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

print(model)




## === cell 9
def train_one_epoch(epoch):
    model.train()
    optimizer.zero_grad()
    x_tr, y_tr = train_x, train_y
    out = model(x_tr)
    loss = criterion(out, y_tr)
    loss.backward()
    optimizer.step()
    if epoch % 2 == 0:
        print(f"Epoch {epoch+1}/{n_epochs} - loss: {loss.item():.4f}")
    return loss.item()




## === cell 10
n_epochs = 8
train_losses = []
for epoch in range(n_epochs):
    loss = train_one_epoch(epoch)
    train_losses.append(loss)




## === cell 11
model.eval()
with torch.no_grad():
    x_val, y_val = val_x, val_y
    val_out = model(x_val)
    val_probs = F.softmax(val_out, dim=1)[:, 1].cpu().numpy()
    val_pred = (val_probs > 0.5).astype(int)
    val_acc = accuracy_score(val_y.cpu().numpy(), val_pred)
    val_auc = roc_auc_score(val_y.cpu().numpy(), val_probs)
print(f"Validation Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}")




## === cell 12
model.eval()
with torch.no_grad():
    x_tr, y_tr = train_x, train_y
    tr_out = model(x_tr)
    tr_probs = F.softmax(tr_out, dim=1)[:, 1].cpu().numpy()
    tr_pred = (tr_probs > 0.5).astype(int)
    tr_acc = accuracy_score(train_y.cpu().numpy(), tr_pred)
    tr_auc = roc_auc_score(train_y.cpu().numpy(), tr_probs)
print(f"Train Accuracy: {tr_acc:.4f}, AUC: {tr_auc:.4f}")




## === cell 13
print("Loading test images …")
n_jobs = min(os.cpu_count() or 1, 8)  # same logic as for training
test_imgs = Parallel(n_jobs=n_jobs, backend="loky")(
    delayed(_load_and_resize)(img_name, test_path)
    for img_name in tqdm(test["image_name"].values)
)
test_x = np.stack(test_imgs)  # (N, H, W, C)
test_x = np.transpose(test_x, (0, 3, 1, 2))  # NCHW
test_x = torch.from_numpy(test_x).float()
if torch.cuda.is_available():
    test_x = test_x.cuda()
print("test_x shape:", test_x.shape)




## === cell 14
model.eval()
with torch.no_grad():
    tx = test_x
    out = model(tx)
    probs = (
        F.softmax(out, dim=1)[:, 1].cpu().numpy()
    )  # probability of malignant (class 1)

sample_submission["target"] = probs
submission_path = "submission.csv"
sample_submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
