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
import sys, subprocess, pkgutil


def _pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


if not pkgutil.find_loader("efficientnet_pytorch"):
    _pip_install("efficientnet_pytorch")



## === cell 1
import os
import time
import random
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from efficientnet_pytorch import EfficientNet
import torchvision.transforms as T


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 2
BASE_INPUT = "/kaggle/input/siim-isic-melanoma-classification"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/data/siim-isic-melanoma-classification"

TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
TEST_CSV = os.path.join(BASE_INPUT, "test.csv")
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "jpeg", "train")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "jpeg", "test")

assert os.path.exists(TRAIN_CSV), f"Missing train.csv at {TRAIN_CSV}"
assert os.path.exists(TEST_CSV), f"Missing test.csv at {TEST_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing train jpeg dir at {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing test jpeg dir at {TEST_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV).reset_index(drop=True)
test_df = pd.read_csv(TEST_CSV).reset_index(drop=True)

train_df.shape, test_df.shape, train_df.columns.tolist()




## === cell 3
class MelanomaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, imfolder: str, transforms_object=None):
        self.data_frame = df.reset_index(drop=True)
        self.path_to_folder = imfolder
        self.transforms_object = transforms_object

    def __getitem__(self, index):
        image_name_at_index = self.data_frame.loc[index, "image_name"]
        load_path = os.path.join(self.path_to_folder, image_name_at_index + ".jpg")

        image_bgr = cv2.imread(load_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {load_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        image_pil = T.functional.to_pil_image(image_rgb)

        if self.transforms_object is not None:
            image_tensor = self.transforms_object(image_pil)
        else:
            image_tensor = T.ToTensor()(image_pil)

        if "target" in self.data_frame.columns.values:
            y = self.data_frame.loc[index, "target"]
        else:
            y = 1
        return image_tensor, y

    def __len__(self):
        return self.data_frame.shape[0]




## === cell 4
train_transform = T.Compose(
    [
        T.RandomResizedCrop(size=256, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

test_transform = T.Compose(
    [
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)




## === cell 5
class deeper_network(nn.Module):
    def __init__(self, arch):
        super(deeper_network, self).__init__()
        self.arch = arch
        self.arch._fc = nn.Linear(in_features=1280, out_features=512, bias=True)
        self.fin_net = nn.Sequential(
            self.arch,
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, inputs):
        output = self.fin_net(inputs)
        return output




## === cell 6
train_dataset = MelanomaDataset(
    train_df, TRAIN_IMG_DIR, transforms_object=train_transform
)
train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=64,
    num_workers=2,
    pin_memory=(device.type == "cuda"),
)

test_dataset = MelanomaDataset(test_df, TEST_IMG_DIR, transforms_object=test_transform)
test_loader = DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=10,
    num_workers=2,
    pin_memory=(device.type == "cuda"),
)

len(train_loader), len(test_loader)



## === cell 7
arch = EfficientNet.from_pretrained("efficientnet-b1")
deep_net = deeper_network(arch).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(deep_net.parameters())

deep_net.train()



## === cell 8
model = deep_net

for e in range(2, 4):
    running_loss = 0.0
    total_predictions = 0.0
    correct_predictions = 0.0
    start_time = time.time()

    model.train()
    for batch_idx, (image_data_array, target) in enumerate(train_loader):
        optimizer.zero_grad()

        image_data_array = image_data_array.float().to(device, non_blocking=True)
        target_t = torch.as_tensor(target, dtype=torch.float32, device=device).view(
            -1, 1
        )

        outputs = model(image_data_array)
        loss = criterion(outputs, target_t)

        predictions = torch.round(torch.sigmoid(outputs))
        total_predictions += target_t.size(0)
        correct_predictions += (
            (target_t.detach().cpu().view(-1) == predictions.detach().cpu().view(-1))
            .sum()
            .item()
        )

        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    acc = (correct_predictions / total_predictions) * 100.0
    end_time = time.time()
    running_loss /= max(1, len(train_loader))

    print("Epoch:", e)
    print(
        "Training Loss:",
        round(running_loss, 3),
        "Time:",
        round(end_time - start_time, 3),
        "s",
    )
    print("Training Accuracy:", round(acc, 3), "%")

torch.save(model.state_dict(), f"/kaggle/working/effnet_v{e}.pth")



## === cell 9
model.eval()
fin_temp = np.empty((0,), dtype=np.float32)

with torch.no_grad():
    for batch_idx, (image_data_array, _) in enumerate(test_loader):
        image_data_array = image_data_array.float().to(device, non_blocking=True)
        outputs = model(image_data_array)
        temp = torch.sigmoid(outputs).detach().cpu().numpy().squeeze()
        temp = np.atleast_1d(temp).astype(np.float32)
        fin_temp = np.concatenate([fin_temp, temp])

assert fin_temp.shape[0] == test_df.shape[0], (fin_temp.shape, test_df.shape)



## === cell 10
submission = test_df[["image_name"]].copy()
submission["target"] = fin_temp

sub_path = "/kaggle/working/submission.csv"
submission.to_csv(sub_path, index=False)

print("Wrote:", sub_path)
print(submission.head())
print("File size (bytes):", os.path.getsize(sub_path))
