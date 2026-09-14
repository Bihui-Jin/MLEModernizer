# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.8

# 3. Installed packages

No external packages required in the script and installed.

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

0.8687963943891686

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.57536) has done: 'I fix the immediate runtime failure caused by an environment-level protobuf/TensorFlow import issue by pinning TensorFlow’s protobuf implementation to the pure-Python backend before importing TensorFlow. Then I remove the hard dependency on an external weights file that is not present in your provided `/kaggle/input/...` paths by switching to EfficientNetB0’s built-in ImageNet weights (same architecture) so inference can run end-to-end. Finally, I keep your TTA/inference pipeline intact and ensure the submission is aligned to `sample_submission.csv` ordering so `submission.csv` is always valid and correctly formatted.'
- What this solution (achieved 0.56393) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf backend *and* disabling the C++ one explicitly, and I also make the TensorFlow import more robust by setting these env vars before any TF-related import happens. Next, I load the EfficientNetB0 ImageNet weights as before but also add the missing compilation/weight loading logic check so we never run with random weights unintentionally (which is the main reason the AUC is far below target). Finally, I keep your TFRecord + TTA pipeline intact while ensuring deterministic ID/pred alignment by deriving `test_ids` from the same finite, non-repeated dataset slice used for predictions, so the submission rows correctly match the averaged predictions.'
- What this solution (achieved 0.50358) has done: 'You’re crashing before any training/inference because the Kaggle TensorFlow build is hitting a protobuf incompatibility (`MessageFactory.GetPrototype`). I fix this by forcing the pure-Python protobuf implementation *and* disabling the C++ backend before TensorFlow is imported, which avoids that specific attribute error in this environment. Then, to move the score toward your target (your current AUC strongly suggests random/untrained head weights), I minimally load a standard pretrained EfficientNetB0 classification checkpoint (same architecture family) and adapt it to your 1-output sigmoid head by copying the backbone weights and leaving only the final Dense randomly initialized. Finally, I keep your TFRecord + TTA pipeline intact but make ID/pred alignment deterministic by generating `test_ids` from the exact same non-repeated base dataset used for counting and prediction ordering.'
- What this solution (achieved 0.52677) has done: 'We fix the immediate TensorFlow/protobuf import crash by setting the protobuf environment variables *before* Python imports the `google.protobuf` module, and by forcing TensorFlow to use the Python protobuf implementation in a way that actually takes effect in Kaggle. Then we keep your TFRecord + TTA inference pipeline intact but make sure the model is properly initialized for inference by compiling it (score-neutral) and ensuring pretrained backbone weights are successfully loaded (otherwise predictions can collapse toward random). Finally, we keep the submission alignment/merge with `sample_submission.csv` and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.50643) has done: 'I fix the TensorFlow/protobuf import crash by ensuring the pure-Python protobuf backend is selected *before* any protobuf/TensorFlow import and by removing the early `google.protobuf` import that can lock in the wrong implementation. I also add a safe fallback path to use the duplicate dataset location under `/kaggle/data/...` if `/kaggle/input/...` isn’t the one actually mounted in your runtime, so TFRecords and sample submission are always found. Finally, I keep your model/TTA pipeline intact but make the prediction/ID ordering consistent by deriving `test_ids` from the exact same repeated/augmented stream used for predictions (rather than a potentially differently-ordered base dataset), which should legitimately improve AUC toward the target without changing the core approach.'
- What this solution (achieved 0.3584) has done: 'I fix the immediate TensorFlow/protobuf crash by forcing the pure-Python protobuf backend before any TensorFlow import, and by ensuring no early `google.protobuf` import can lock in the incompatible C++ implementation. Then I make the TFRecord ID/prediction alignment deterministic and correct by removing the non-deterministic dataset option and by deriving `test_ids` from the same base (non-augmented, non-repeated) dataset order as the TFRecords themselves. Finally, I keep your EfficientNetB0 + TTA inference core logic intact (same model, same transforms, same TTA count), while ensuring a valid `submission.csv` is always written in the exact `sample_submission.csv` row order.'
- What this solution (achieved 0.36908) has done: 'The crash happens before any of your code runs because the Kaggle environment’s TensorFlow/protobuf combination is incompatible with this import path, so the fix is to avoid importing TensorFlow entirely. To keep your core approach (EfficientNetB0 + deterministic ordering + submission alignment) while making it run end-to-end, I switch inference to PyTorch/TorchVision EfficientNet-B0 pretrained weights (available in the standard Kaggle image) and perform the same single-model probability prediction over the JPEG test images. This also moves the score up substantially versus essentially-random outputs (your current 0.3584), without changing the task semantics (still outputs malignant probability per image). The script still writes `submission.csv` with the exact required columns and in `sample_submission.csv` order.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
CANDIDATE_ROOTS = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/input",
    "/kaggle/data",
]

LOCAL_DS_PATH = None
for root in CANDIDATE_ROOTS:
    if (
        os.path.exists(os.path.join(root, "train.csv"))
        and os.path.exists(os.path.join(root, "test.csv"))
        and (
            os.path.exists(os.path.join(root, "jpeg"))
            or os.path.exists(os.path.join(root, "jpeg", "test"))
        )
    ):
        LOCAL_DS_PATH = root
        break

if LOCAL_DS_PATH is None:
    raise FileNotFoundError(f"Could not locate dataset root. Tried: {CANDIDATE_ROOTS}")

print("Using dataset root:", LOCAL_DS_PATH)

train_csv_path = os.path.join(LOCAL_DS_PATH, "train.csv")
test_csv_path = os.path.join(LOCAL_DS_PATH, "test.csv")
sample_path = os.path.join(LOCAL_DS_PATH, "sample_submission.csv")

if not os.path.exists(sample_path):
    alt = "/kaggle/input/sample_submission.csv"
    if os.path.exists(alt):
        sample_path = alt

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
sample = pd.read_csv(sample_path)

jpeg_test_dir_candidates = [
    os.path.join(LOCAL_DS_PATH, "jpeg", "test"),
    os.path.join(LOCAL_DS_PATH, "siim-isic-melanoma-classification", "jpeg", "test"),
]
JPEG_TEST_DIR = None
for d in jpeg_test_dir_candidates:
    if os.path.exists(d):
        JPEG_TEST_DIR = d
        break
if JPEG_TEST_DIR is None:
    raise FileNotFoundError(
        f"Could not find jpeg test directory. Tried: {jpeg_test_dir_candidates}"
    )

jpeg_train_dir_candidates = [
    os.path.join(LOCAL_DS_PATH, "jpeg", "train"),
    os.path.join(LOCAL_DS_PATH, "siim-isic-melanoma-classification", "jpeg", "train"),
]
JPEG_TRAIN_DIR = None
for d in jpeg_train_dir_candidates:
    if os.path.exists(d):
        JPEG_TRAIN_DIR = d
        break
if JPEG_TRAIN_DIR is None:
    raise FileNotFoundError(
        f"Could not find jpeg train directory. Tried: {jpeg_train_dir_candidates}"
    )

print("Using JPEG train dir:", JPEG_TRAIN_DIR)
print("Using JPEG test dir:", JPEG_TEST_DIR)
print(
    "Train rows:",
    len(train_df),
    "Test rows:",
    len(test_df),
    "Sample rows:",
    len(sample),
)



## === cell 2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from PIL import Image

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

BATCH_SIZE = 64

weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
model = models.efficientnet_b0(weights=weights)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 1)
model = model.to(device)

for p in model.parameters():
    p.requires_grad = False
for p in model.classifier.parameters():
    p.requires_grad = True

_wt = weights.transforms()
_MEAN = _wt.mean
_STD = _wt.std

train_names = train_df["image_name"].to_numpy()
train_targets = train_df["target"].to_numpy(dtype=np.float32)
test_names = test_df["image_name"].to_numpy()



## === cell 3
import multiprocessing as mp

IMG_SIZE = 128
CHW = (3, IMG_SIZE, IMG_SIZE)


def _load_and_preprocess_jpeg_fast(path: str):
    img = Image.open(path).convert("RGB")
    img.load()
    img = TF.resize(
        img, [IMG_SIZE, IMG_SIZE], interpolation=transforms.InterpolationMode.BILINEAR
    )
    x = TF.to_tensor(img)
    x = TF.normalize(x, mean=_MEAN, std=_STD)
    return x


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


CACHE_DIR = "/kaggle/working/isic_cache"
_ensure_dir(CACHE_DIR)

TRAIN_MEMMAP = os.path.join(CACHE_DIR, f"train_{IMG_SIZE}x{IMG_SIZE}_float32_chw.dat")
TEST_MEMMAP = os.path.join(CACHE_DIR, f"test_{IMG_SIZE}x{IMG_SIZE}_float32_chw.dat")


def _build_memmap(memmap_path: str, names: np.ndarray, img_dir: str):
    n = int(names.shape[0])
    mm = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=(n, *CHW))

    cpu_cnt = os.cpu_count() or 2
    workers = min(8, max(2, cpu_cnt - 1))

    def _worker(i_name):
        i, name = i_name
        path = os.path.join(img_dir, f"{name}.jpg")
        try:
            x = _load_and_preprocess_jpeg_fast(path)  # torch tensor float32 CHW
            arr = x.numpy()  # shares data copy to numpy float32
        except Exception:
            arr = np.zeros(CHW, dtype=np.float32)
        return i, arr

    chunksize = 64

    with mp.get_context("fork").Pool(processes=workers) as pool:
        for i, arr in pool.imap_unordered(
            _worker, enumerate(names.tolist()), chunksize=chunksize
        ):
            mm[i, :, :, :] = arr

    mm.flush()
    return memmap_path


if not os.path.exists(TRAIN_MEMMAP):
    print("Building train cache (one-time)...")
    _build_memmap(TRAIN_MEMMAP, train_names, JPEG_TRAIN_DIR)
else:
    print("Train cache exists:", TRAIN_MEMMAP)

if not os.path.exists(TEST_MEMMAP):
    print("Building test cache (one-time)...")
    _build_memmap(TEST_MEMMAP, test_names, JPEG_TEST_DIR)
else:
    print("Test cache exists:", TEST_MEMMAP)

train_mm = np.memmap(
    TRAIN_MEMMAP, dtype=np.float32, mode="r", shape=(len(train_names), *CHW)
)
test_mm = np.memmap(
    TEST_MEMMAP, dtype=np.float32, mode="r", shape=(len(test_names), *CHW)
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3324157113.py in <cell line: 0>()
     64 if not os.path.exists(TRAIN_MEMMAP):
     65     print("Building train cache (one-time)...")
---> 66     _build_memmap(TRAIN_MEMMAP, train_names, JPEG_TRAIN_DIR)
     67 else:
     68     print("Train cache exists:", TRAIN_MEMMAP)

/tmp/ipykernel_11/3324157113.py in _build_memmap(memmap_path, names, img_dir)
     53 
     54     with mp.get_context("fork").Pool(processes=workers) as pool:
---> 55         for i, arr in pool.imap_unordered(
     56             _worker, enumerate(names.tolist()), chunksize=chunksize
     57         ):

/usr/lib/python3.11/multiprocessing/pool.py in <genexpr>(.0)
    449                     result._set_length
    450                 ))
--> 451             return (item for chunk in result for item in chunk)
    452 
    453     def apply_async(self, func, args=(), kwds={}, callback=None,

/usr/lib/python3.11/multiprocessing/pool.py in next(self, timeout)
    871         if success:
    872             return value
--> 873         raise value
    874 
    875     __next__ = next                    # XXX

/usr/lib/python3.11/multiprocessing/pool.py in _handle_tasks(taskqueue, put, outqueue, pool, cache)
    538                         break
    539                     try:
--> 540                         put(task)
    541                     except Exception as e:
    542                         job, idx = task[:2]

/usr/lib/python3.11/multiprocessing/connection.py in send(self, obj)
    204         self._check_closed()
    205         self._check_writable()
--> 206         self._send_bytes(_ForkingPickler.dumps(obj))
    207 
    208     def recv_bytes(self, maxlength=None):

/usr/lib/python3.11/multiprocessing/reduction.py in dumps(cls, obj, protocol)
     49     def dumps(cls, obj, protocol=None):
     50         buf = io.BytesIO()
---> 51         cls(buf, protocol).dump(obj)
     52         return buf.getbuffer()
     53 

AttributeError: Can't pickle local object '_build_memmap.<locals>._worker'

## === cell 4
class MelanomaTrainDataset(Dataset):
    def __init__(self, memmap_arr, targets):
        self.mm = memmap_arr
        self.targets = targets

    def __len__(self):
        return int(self.targets.shape[0])

    def __getitem__(self, idx):
        x = torch.from_numpy(self.mm[idx])  # float32 CHW
        y = float(self.targets[idx])
        return x, torch.tensor([y], dtype=torch.float32)


class MelanomaTestDataset(Dataset):
    def __init__(self, memmap_arr, image_names):
        self.mm = memmap_arr
        self.image_names = image_names

    def __len__(self):
        return int(self.image_names.shape[0])

    def __getitem__(self, idx):
        x = torch.from_numpy(self.mm[idx])  # float32 CHW
        return x, self.image_names[idx]


train_dataset = MelanomaTrainDataset(train_mm, train_targets)
test_dataset = MelanomaTestDataset(test_mm, test_names)

cpu_cnt = os.cpu_count() or 2
num_workers = min(4, max(2, cpu_cnt - 1))
g = torch.Generator()
g.manual_seed(SEED)

common_loader_kwargs = dict(
    batch_size=BATCH_SIZE,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=(4 if num_workers > 0 else None),
)

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    generator=g,
    **{k: v for k, v in common_loader_kwargs.items() if v is not None},
)

test_loader = DataLoader(
    test_dataset,
    shuffle=False,
    **{k: v for k, v in common_loader_kwargs.items() if v is not None},
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/773401927.py in <cell line: 0>()
     28 
     29 
---> 30 train_dataset = MelanomaTrainDataset(train_mm, train_targets)
     31 test_dataset = MelanomaTestDataset(test_mm, test_names)
     32 

NameError: name 'train_mm' is not defined

## === cell 5
pos = float(train_df["target"].sum())
neg = float(len(train_df) - pos)
pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device, dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=3e-3, weight_decay=1e-4
)

model.train()
EPOCHS = 2  # unchanged core logic

for epoch in range(EPOCHS):
    running = 0.0
    n = 0
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        running += float(loss.detach().cpu()) * bs
        n += bs

    print(f"epoch {epoch+1}/{EPOCHS} - train_loss: {running/max(n,1):.5f}")

model.eval()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2161419599.py in <cell line: 0>()
     15     running = 0.0
     16     n = 0
---> 17     for xb, yb in train_loader:
     18         xb = xb.to(device, non_blocking=True)
     19         yb = yb.to(device, non_blocking=True)

NameError: name 'train_loader' is not defined

## === cell 6
n_test = len(test_dataset)
all_ids = np.empty((n_test,), dtype=object)
all_preds = np.empty((n_test,), dtype=np.float32)

offset = 0
with torch.no_grad():
    for xb, ids in test_loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb).squeeze(1)
        probs = (
            torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32, copy=False)
        )

        bs = probs.shape[0]
        all_ids[offset : offset + bs] = np.asarray(ids, dtype=object)
        all_preds[offset : offset + bs] = probs
        offset += bs

sub_raw = pd.DataFrame({"image_name": all_ids, "target": all_preds})

sub = sample[["image_name"]].merge(sub_raw, on="image_name", how="left")
sub["target"] = (
    sub["target"]
    .fillna(sub["target"].mean() if sub["target"].notna().any() else 0.5)
    .clip(0.0, 1.0)
)

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print(
    "Target stats:",
    float(sub["target"].min()),
    float(sub["target"].mean()),
    float(sub["target"].max()),
)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/828120756.py in <cell line: 0>()
      1 # Speed fix: keep efficient preallocation and avoid extra list(ids) conversions when possible.
----> 2 n_test = len(test_dataset)
      3 all_ids = np.empty((n_test,), dtype=object)
      4 all_preds = np.empty((n_test,), dtype=np.float32)
      5 

NameError: name 'test_dataset' is not defined
