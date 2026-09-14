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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.7357710064635279

# 6. Current score

0.26731

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.26731) has done: 'The timeout is dominated by per-image CPU preprocessing: `skimage.transform.resize` is slow, `RandomCrop` introduces extra work, and logits are accumulated with repeated `np.vstack` (quadratic growth). I replace resizing/cropping with equivalent OpenCV operations (same geometry semantics), make the crop deterministic (center crop) to avoid random overhead while keeping the same transform structure, and preallocate the logits array to avoid repeated concatenations. I also enable faster DataLoader settings (more workers, persistent workers, prefetching) and reduce per-batch CPU/GPU sync by writing into the preallocated array directly—all of which preserve the model, checkpoint loading, inference loop semantics, and output formatting.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

import cv2
from skimage import transform

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchvision.models as models
from torchvision.transforms import transforms

from tqdm import tqdm

DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

test_fram = pd.read_csv(SAMPLE_SUB_CSV)
print(test_fram.head())
print("Num test rows:", len(test_fram))
print("Example image id:", test_fram.values[1][0])




## === cell 1
class LeafDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform):
        df = pd.read_csv(csv_file)
        self.image_names = df.iloc[:, 0].to_numpy()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.image_names[idx])

        image = cv2.imread(img_name, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        labels = [1]
        sample = {"image": image, "labels": labels}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    def __call__(self, sample):
        image = sample["image"]
        labels = sample["labels"]
        image = np.ascontiguousarray(image.transpose((2, 0, 1)), dtype=np.float32)
        return {"image": torch.from_numpy(image), "labels": labels}


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample["image"], sample["labels"]
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return {"image": img, "labels": labels}


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = (
            (output_size, output_size) if isinstance(output_size, int) else output_size
        )
        assert len(self.output_size) == 2

    def __call__(self, sample):
        image, labels = sample["image"], sample["labels"]
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        if h < new_h or w < new_w:
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            return {"image": image, "labels": labels}

        top = (h - new_h) // 2
        left = (w - new_w) // 2
        image = image[top : top + new_h, left : left + new_w]
        return {"image": image, "labels": labels}


class Normalize(object):
    """
    Fix: ResNet expects ImageNet normalization; without it predictions are badly calibrated.
    This is score-improving but does not change model architecture/training logic.
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)

    def __call__(self, sample):
        x = sample["image"]
        x = x.astype(np.float32) * (1.0 / 255.0)
        x = (x - self.mean) / self.std
        sample["image"] = x
        return sample


leafDatasets = LeafDataset(
    SAMPLE_SUB_CSV,
    TEST_IMG_DIR,
    transform=transforms.Compose(
        [
            Rescale(256),
            RandomCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
)

print("Dataset item keys:", leafDatasets[0].keys())
print("Image tensor shape:", leafDatasets[0]["image"].shape)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_56/935143056.py in <cell line: 0>()
    123 )
    124 
--> 125 print("Dataset item keys:", leafDatasets[0].keys())
    126 print("Image tensor shape:", leafDatasets[0]["image"].shape)
    127 

/tmp/ipykernel_56/935143056.py in __getitem__(self, idx)
     26 
     27         if self.transform:
---> 28             sample = self.transform(sample)
     29 
     30         return sample

/usr/local/lib/python3.11/dist-packages/torchvision/transforms/transforms.py in __call__(self, img)
     93     def __call__(self, img):
     94         for t in self.transforms:
---> 95             img = t(img)
     96         return img
     97 

/tmp/ipykernel_56/935143056.py in __call__(self, sample)
    104         x = sample["image"]
    105         # Speed: fuse operations; also ensure float32 once here.
--> 106         x = x.astype(np.float32) * (1.0 / 255.0)
    107         x = (x - self.mean) / self.std
    108         sample["image"] = x

AttributeError: 'Tensor' object has no attribute 'astype'

## === cell 2
batch_size = 64  # Speed: larger batches reduce Python/DataLoader overhead; inference semantics unchanged.

num_workers = min(8, os.cpu_count() or 2)
pin = torch.cuda.is_available()
test_loader = DataLoader(
    leafDatasets,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)




## === cell 3
try:
    resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
except Exception:
    resnet = models.resnet152(weights=None)

num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 6)

print(resnet.fc)




## === cell 4
load_path = "../input/modelres/resnet.pkl"
if os.path.exists(load_path):
    state = torch.load(load_path, map_location="cpu")
    resnet.load_state_dict(state)
    print("Loaded checkpoint:", load_path)
else:
    print("Checkpoint not found; using torchvision initialization:", load_path)




## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)
resnet.eval()

n_test = len(leafDatasets)
y_pred_logits = np.empty((n_test, 6), dtype=np.float32)

with torch.no_grad():
    stream = tqdm(
        test_loader,
        desc=f"Infer ({device})",
        total=(n_test + batch_size - 1) // batch_size,
    )
    offset = 0
    for sample in stream:
        X = sample["image"].to(device, non_blocking=True)
        X = X.float()
        pred = resnet(X)  # logits
        bs = pred.shape[0]
        y_pred_logits[offset : offset + bs] = pred.detach().cpu().numpy()
        offset += bs

print("Pred logits shape:", y_pred_logits.shape)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_56/1724295643.py in <cell line: 0>()
     14     )
     15     offset = 0
---> 16     for sample in stream:
     17         X = sample["image"].to(device, non_blocking=True)
     18         # Keep float() to preserve original semantics exactly.

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

AttributeError: Caught AttributeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_56/935143056.py", line 28, in __getitem__
    sample = self.transform(sample)
             ^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/transforms.py", line 95, in __call__
    img = t(img)
          ^^^^^^
  File "/tmp/ipykernel_56/935143056.py", line 106, in __call__
    x = x.astype(np.float32) * (1.0 / 255.0)
        ^^^^^^^^
AttributeError: 'Tensor' object has no attribute 'astype'


## === cell 6
y_pred_logits[:2]




## === cell 7
probs = 1.0 / (1.0 + np.exp(-y_pred_logits))

thr = 0.50

mask = probs >= thr
indices = []
argmax_idx = np.argmax(probs, axis=1)
for i in range(probs.shape[0]):
    idxs = np.flatnonzero(mask[i]).tolist()
    if len(idxs) == 0:
        idxs = [int(argmax_idx[i])]
    indices.append(idxs)

print("Example predicted indices:", indices[0])




## === cell 8
labels = ["complex", "frog_eye_leaf_spot", "healthy", "powdery_mildew", "rust", "scab"]

testlabels = [" ".join(labels[i] for i in idxs) for idxs in indices]

print(testlabels[:5], " ... total:", len(testlabels))




## === cell 9
sub = pd.read_csv(SAMPLE_SUB_CSV)
assert len(sub) == len(
    testlabels
), f"Predictions ({len(testlabels)}) != submission rows ({len(sub)})"
sub["labels"] = testlabels
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
