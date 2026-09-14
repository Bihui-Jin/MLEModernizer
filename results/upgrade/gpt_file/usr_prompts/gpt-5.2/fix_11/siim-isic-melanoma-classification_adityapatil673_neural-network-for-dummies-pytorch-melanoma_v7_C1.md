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
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import time
import random
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 1
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



## === cell 2
try:
    from torchvision.transforms import v2 as TV2
except Exception:
    TV2 = None

_OUT_SIZE = 256
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

if TV2 is not None:
    gpu_train_tf = TV2.Compose(
        [
            TV2.RandomResizedCrop(
                size=(_OUT_SIZE, _OUT_SIZE),
                scale=(0.7, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
                antialias=True,
            ),
            TV2.RandomHorizontalFlip(p=0.5),
            TV2.RandomVerticalFlip(p=0.5),
            TV2.ToDtype(torch.float32, scale=True),  # uint8 -> float32 in [0,1]
            TV2.Normalize(mean=_MEAN, std=_STD),
        ]
    )
    gpu_test_tf = TV2.Compose(
        [
            TV2.ToDtype(torch.float32, scale=True),
            TV2.Normalize(mean=_MEAN, std=_STD),
        ]
    )
else:
    gpu_train_tf = None
    gpu_test_tf = None




## === cell 3
class MelanomaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, imfolder: str, transforms_object=None):
        self.data_frame = df.reset_index(drop=True)
        self.path_to_folder = imfolder
        self.transforms_object = transforms_object  # kept for compatibility

        self._image_names = self.data_frame["image_name"].values
        self._paths = [
            os.path.join(self.path_to_folder, n + ".jpg") for n in self._image_names
        ]

        self._has_target = "target" in self.data_frame.columns.values
        if self._has_target:
            self._targets = (
                self.data_frame["target"].to_numpy(copy=True).astype(np.float32)
            )

        self._is_train = self._has_target
        self._out_size = 256

    def __getitem__(self, index):
        load_path = self._paths[index]

        img = cv2.imread(load_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            raise FileNotFoundError(load_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(
            img, (self._out_size, self._out_size), interpolation=cv2.INTER_AREA
        )

        img = np.transpose(img, (2, 0, 1))  # CHW
        image_tensor = torch.from_numpy(img)  # uint8

        if self._has_target:
            y = self._targets[index]
        else:
            y = 1.0
        return image_tensor, y

    def __len__(self):
        return self.data_frame.shape[0]




## === cell 4
train_transform = T.Compose(
    [
        T.RandomResizedCrop(size=256, scale=(0.7, 1.0), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

test_transform = T.Compose(
    [
        T.Resize((256, 256), antialias=True),
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
_num_workers = min(8, max(2, (os.cpu_count() or 2) - 2))
_pin = device.type == "cuda"

try:
    cv2.setNumThreads(0)  # avoid OpenCV oversubscribing worker processes
except Exception:
    pass

try:
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
except Exception:
    pass


def _seed_worker(worker_id: int):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(42)

train_dataset = MelanomaDataset(
    train_df, TRAIN_IMG_DIR, transforms_object=train_transform
)
train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=64,
    num_workers=_num_workers,
    pin_memory=_pin,
    pin_memory_device="cuda" if _pin else "",
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,  # SPEED: keep workers fed
    worker_init_fn=_seed_worker if _num_workers > 0 else None,
    generator=g,
)

test_dataset = MelanomaDataset(test_df, TEST_IMG_DIR, transforms_object=test_transform)
test_loader = DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=64,
    num_workers=_num_workers,
    pin_memory=_pin,
    pin_memory_device="cuda" if _pin else "",
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
    worker_init_fn=_seed_worker if _num_workers > 0 else None,
    generator=g,
)

len(train_loader), len(test_loader)



## === cell 7
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
arch = torchvision.models.efficientnet_b1(weights=weights)

arch._fc = arch.classifier[1]

arch.classifier[1] = nn.Linear(in_features=1280, out_features=512, bias=True)
arch._fc = arch.classifier[1]

deep_net = deeper_network(arch).to(device)

if device.type == "cuda":
    deep_net = deep_net.to(memory_format=torch.channels_last)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(deep_net.parameters())

deep_net.train()



## === cell 8
model = deep_net
_USE_COMPILE = os.environ.get("USE_TORCH_COMPILE", "0") == "1"
if _USE_COMPILE and hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="max-autotune")
    except Exception:
        model = deep_net

_fallback_norm = T.Normalize(list(_MEAN), list(_STD))

for e in range(2, 4):
    running_loss = 0.0
    start_time = time.time()

    model.train()
    for batch_idx, (image_data_array, target) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)

        image_data_array = image_data_array.to(device, non_blocking=True)
        if device.type == "cuda":
            image_data_array = image_data_array.contiguous(
                memory_format=torch.channels_last
            )

        if gpu_train_tf is not None:
            image_data_array = gpu_train_tf(image_data_array)
        else:
            image_data_array = image_data_array.float().div_(255.0)
            image_data_array = _fallback_norm(image_data_array)

        if isinstance(target, torch.Tensor):
            target_t = target.to(
                device=device, dtype=torch.float32, non_blocking=True
            ).view(-1, 1)
        else:
            target_t = torch.as_tensor(target, device=device, dtype=torch.float32).view(
                -1, 1
            )

        outputs = model(image_data_array)
        loss = criterion(outputs, target_t)

        running_loss += float(loss.detach())
        loss.backward()
        optimizer.step()

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

torch.save(model.state_dict(), f"/kaggle/working/effnet_v{e}.pth")



## === cell 9
model.eval()
fin_temp = np.empty((test_df.shape[0],), dtype=np.float32)
offset = 0

_fallback_norm_test = _fallback_norm

with torch.inference_mode():
    for batch_idx, (image_data_array, _) in enumerate(test_loader):
        image_data_array = image_data_array.to(device, non_blocking=True)
        if device.type == "cuda":
            image_data_array = image_data_array.contiguous(
                memory_format=torch.channels_last
            )

        if gpu_test_tf is not None:
            image_data_array = gpu_test_tf(image_data_array)
        else:
            image_data_array = image_data_array.float().div_(255.0)
            image_data_array = _fallback_norm_test(image_data_array)

        outputs = model(image_data_array)
        temp = (
            torch.sigmoid(outputs).detach().cpu().numpy().reshape(-1).astype(np.float32)
        )

        n = temp.shape[0]
        fin_temp[offset : offset + n] = temp
        offset += n

assert offset == test_df.shape[0], (offset, test_df.shape[0])
assert fin_temp.shape[0] == test_df.shape[0], (fin_temp.shape, test_df.shape)

fin_temp = np.clip(fin_temp, 0.0, 1.0)



## === cell 10
submission = test_df[["image_name"]].copy()
submission["target"] = fin_temp

sub_path = "/kaggle/working/submission.csv"
submission.to_csv(sub_path, index=False)

print("Wrote:", sub_path)
print(submission.head())
print(
    "target range:",
    float(submission["target"].min()),
    float(submission["target"].max()),
)
print("File size (bytes):", os.path.getsize(sub_path))
