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
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

_max_threads = min(4, os.cpu_count() or 1)
torch.set_num_threads(_max_threads)
os.environ.setdefault("OMP_NUM_THREADS", str(_max_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(_max_threads))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_max_threads))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_max_threads))

os.environ.setdefault("OPENCV_FOR_THREADS_NUM", "1")
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass




## === cell 1
import pandas as pd
import cv2

try:
    cv2.setNumThreads(1)
except Exception:
    pass
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F




## === cell 2
BASE_CANDIDATES = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/input",
    "/kaggle/data",
]
BASE = None
for b in BASE_CANDIDATES:
    if os.path.exists(b):
        if b.endswith("siim-isic-melanoma-classification"):
            BASE = b
            break
        if os.path.exists(os.path.join(b, "train.csv")) and os.path.exists(
            os.path.join(b, "test.csv")
        ):
            BASE = b
            break

if BASE is None:
    raise FileNotFoundError(
        "Could not find dataset base path under /kaggle/input or /kaggle/data"
    )

TRAIN_CSV = os.path.join(BASE, "train.csv")
TEST_CSV = os.path.join(BASE, "test.csv")
JPEG_TRAIN_DIR = os.path.join(BASE, "jpeg", "train")
JPEG_TEST_DIR = os.path.join(BASE, "jpeg", "test")

print("BASE:", BASE)
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("TEST_CSV exists:", os.path.exists(TEST_CSV))
print("JPEG_TRAIN_DIR exists:", os.path.exists(JPEG_TRAIN_DIR))
print("JPEG_TEST_DIR exists:", os.path.exists(JPEG_TEST_DIR))




## === cell 3
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print(train_df.shape, test_df.shape)
print(train_df.columns)




## === cell 4
class _FastTrainTransform:
    def __init__(self, img_size: int):
        self.img_size = int(img_size)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self.scale = (0.7, 1.0)
        self.ratio = (3.0 / 4.0, 4.0 / 3.0)
        self._log_ratio = (float(np.log(self.ratio[0])), float(np.log(self.ratio[1])))

    def __call__(self, img_rgb_uint8: np.ndarray) -> torch.Tensor:
        h, w = img_rgb_uint8.shape[:2]

        area = float(h * w)

        i = j = 0
        th, tw = h, w
        for _ in range(10):
            target_area = area * float(np.random.uniform(self.scale[0], self.scale[1]))
            aspect = float(
                np.exp(np.random.uniform(self._log_ratio[0], self._log_ratio[1]))
            )

            tw_ = int(round(np.sqrt(target_area * aspect)))
            th_ = int(round(np.sqrt(target_area / aspect)))

            if 0 < tw_ <= w and 0 < th_ <= h:
                i = int(np.random.randint(0, h - th_ + 1))
                j = int(np.random.randint(0, w - tw_ + 1))
                th, tw = th_, tw_
                break

        crop = img_rgb_uint8[i : i + th, j : j + tw]

        resized = cv2.resize(
            crop, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
        )

        if float(np.random.rand()) < 0.5:
            resized = resized[:, ::-1, :]
        if float(np.random.rand()) < 0.5:
            resized = resized[::-1, :, :]

        x = resized.astype(np.float32) * (1.0 / 255.0)
        x = (x - self.mean) / self.std

        return torch.from_numpy(x).permute(2, 0, 1).contiguous()


class _FastTestTransform:
    def __init__(self, img_size: int):
        self.img_size = int(img_size)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, img_rgb_uint8: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(
            img_rgb_uint8,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_LINEAR,
        )
        x = resized.astype(np.float32) * (1.0 / 255.0)
        x = (x - self.mean) / self.std
        return torch.from_numpy(x).permute(2, 0, 1).contiguous()




## === cell 5
def _imread_rgb_uint8(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


class CachedResizedImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        imfolder: str,
        transforms_object=None,
        cache_size: int = 256,
    ):
        self.data_frame = df.reset_index(drop=True)
        self.path_to_folder = imfolder
        self.transforms_object = transforms_object
        self.cache_size = int(cache_size)

        names = self.data_frame["image_name"].astype(str).tolist()
        self._paths = [os.path.join(imfolder, f"{n}.jpg") for n in names]

        if "target" in self.data_frame.columns:
            self._targets = torch.from_numpy(
                self.data_frame["target"].astype(np.float32).to_numpy()
            )
        else:
            self._targets = None

        self._cache = [None] * len(self._paths)

    def build_cache(self):
        for idx, p in enumerate(self._paths):
            img = _imread_rgb_uint8(p)
            if img is None:
                raise FileNotFoundError(f"Could not read image at: {p}")
            img = cv2.resize(
                img, (self.cache_size, self.cache_size), interpolation=cv2.INTER_AREA
            )
            self._cache[idx] = img

    def __getitem__(self, index):
        img_rgb = self._cache[index]
        if img_rgb is None:
            load_path = self._paths[index]
            img_rgb = _imread_rgb_uint8(load_path)
            if img_rgb is None:
                raise FileNotFoundError(f"Could not read image at: {load_path}")
            img_rgb = cv2.resize(
                img_rgb,
                (self.cache_size, self.cache_size),
                interpolation=cv2.INTER_AREA,
            )
            self._cache[index] = img_rgb

        x = (
            self.transforms_object(img_rgb)
            if self.transforms_object is not None
            else img_rgb
        )

        if self._targets is not None:
            y = self._targets[index]
        else:
            y = torch.tensor(1.0, dtype=torch.float32)
        return x, y

    def __len__(self):
        return len(self._paths)




## === cell 6
IMG_SIZE = 128

train_transform = _FastTrainTransform(IMG_SIZE)
test_transform = _FastTestTransform(IMG_SIZE)




## === cell 7
path_for_jpeg_train = JPEG_TRAIN_DIR
path_for_jpeg_test = JPEG_TEST_DIR

train_dataset = CachedResizedImageDataset(
    train_df, path_for_jpeg_train, transforms_object=train_transform, cache_size=256
)
test_dataset = CachedResizedImageDataset(
    test_df, path_for_jpeg_test, transforms_object=test_transform, cache_size=256
)

import time

_t0 = time.time()
train_dataset.build_cache()
_t1 = time.time()
test_dataset.build_cache()
_t2 = time.time()
print(f"Built train cache: {len(train_dataset)} images in {(_t1-_t0):.1f}s")
print(f"Built test  cache: {len(test_dataset)} images in {(_t2-_t1):.1f}s")


def _seed_worker(worker_id):
    seed = SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _fast_collate(batch):
    imgs, ys = zip(*batch)
    x = torch.stack(imgs, dim=0)
    y = torch.stack(ys).to(dtype=torch.float32)
    return x, y


_num_workers = 0

g = torch.Generator()
g.manual_seed(SEED)

_persistent = False
_pin_device = "cuda" if torch.cuda.is_available() else ""

train_loader_args = dict(
    shuffle=True,
    batch_size=64,
    num_workers=_num_workers,
    pin_memory=True,
    persistent_workers=_persistent,
    generator=g,
    collate_fn=_fast_collate,
    drop_last=True,
)
if _pin_device:
    train_loader_args["pin_memory_device"] = _pin_device
train_loader = data.DataLoader(train_dataset, **train_loader_args)

test_loader_args = dict(
    shuffle=False,
    batch_size=64,
    num_workers=_num_workers,
    pin_memory=True,
    persistent_workers=_persistent,
    generator=g,
    collate_fn=_fast_collate,
)
if _pin_device:
    test_loader_args["pin_memory_device"] = _pin_device
test_loader = data.DataLoader(test_dataset, **test_loader_args)

print("num_workers:", _num_workers, "persistent_workers:", _persistent)
print("train/test batches:", len(train_loader), len(test_loader))




## === cell 8
class deeper_network(nn.Module):
    def __init__(self, arch):
        super(deeper_network, self).__init__()
        self.arch = arch

        in_features = None
        if hasattr(self.arch, "classifier") and isinstance(
            self.arch.classifier, nn.Sequential
        ):
            for m in reversed(self.arch.classifier):
                if isinstance(m, nn.Linear):
                    in_features = m.in_features
                    break
        if in_features is None:
            in_features = 1280

        self.arch.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=in_features, out_features=512, bias=True),
        )

        self.fin_net = nn.Sequential(
            self.arch,
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, inputs):
        return self.fin_net(inputs)




## === cell 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V2
arch = torchvision.models.efficientnet_b1(weights=weights)

deep_net = deeper_network(arch).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(deep_net.parameters())

if device.type == "cuda":
    deep_net = deep_net.to(memory_format=torch.channels_last)

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

if hasattr(torch, "compile"):
    try:
        deep_net = torch.compile(deep_net, mode="reduce-overhead")
        print("torch.compile enabled")
    except Exception as _e:
        print("torch.compile not available, continuing without it:", repr(_e))

print("Device:", device)




## === cell 10
import time

model = deep_net
model.train()

EPOCHS = 2
last_epoch = -1

for e in range(EPOCHS):
    last_epoch = e
    start_time = time.time()

    loss_sum = torch.zeros((), device=device)
    correct_sum = torch.zeros((), device=device)
    total = 0

    for batch_idx, (image_data_array, target) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)

        image_data_array = image_data_array.to(device, non_blocking=True)
        if device.type == "cuda":
            image_data_array = image_data_array.to(memory_format=torch.channels_last)
        target = target.to(device, non_blocking=True).view(-1, 1)

        outputs = model(image_data_array)
        loss = criterion(outputs, target)

        predictions = (outputs >= 0).to(dtype=target.dtype)
        correct_sum += (predictions.view(-1) == target.view(-1)).sum()
        total += target.size(0)

        loss_sum += loss.detach()

        loss.backward()
        optimizer.step()

    running_loss = (loss_sum / len(train_loader)).item()
    acc = (correct_sum / float(total) * 100.0).item()
    end_time = time.time()

    print("Epoch:", e)
    print(
        "Training Loss:",
        round(running_loss, 3),
        "Time:",
        round(end_time - start_time, 3),
        "s",
    )
    print("Training Accuracy:", round(acc, 3), "%")




## === cell 11
save_name = f"effnet_v{last_epoch}.pth"
torch.save(model.state_dict(), save_name)
print("Saved:", save_name)
print("Working dir files:", os.listdir("/kaggle/working")[:50])




## === cell 12
model.eval()

n_test = len(test_dataset)
fin_temp = np.empty((n_test,), dtype=np.float32)
write_pos = 0

with torch.inference_mode():
    for batch_idx, (image_data_array, _) in enumerate(test_loader):
        image_data_array = image_data_array.to(device, non_blocking=True)
        if device.type == "cuda":
            image_data_array = image_data_array.to(memory_format=torch.channels_last)
        outputs = model(image_data_array)
        probs = (
            torch.sigmoid(outputs).detach().cpu().numpy().reshape(-1).astype(np.float32)
        )
        bs = probs.shape[0]
        fin_temp[write_pos : write_pos + bs] = probs
        write_pos += bs

if write_pos != n_test:
    raise RuntimeError(f"Prediction write mismatch: wrote {write_pos} of {n_test}")

print(
    "Preds:", fin_temp.shape, "min/max:", float(fin_temp.min()), float(fin_temp.max())
)

Y_submission = test_df[["image_name"]].copy()
if len(Y_submission) != len(fin_temp):
    raise RuntimeError(
        f"Prediction length mismatch: {len(fin_temp)} preds vs {len(Y_submission)} rows"
    )

Y_submission["target"] = fin_temp

sub_path = "/kaggle/working/submission.csv"
Y_submission.to_csv(sub_path, index=False)
print("Wrote submission to:", sub_path)
print(Y_submission.head())
