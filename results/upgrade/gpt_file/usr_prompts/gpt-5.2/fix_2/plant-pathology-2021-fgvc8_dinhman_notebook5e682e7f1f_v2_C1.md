# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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

test_fram = pd.read_csv(SAMPLE_SUB_CSV)
print(test_fram.head())
print("Num test rows:", len(test_fram))
print("Example image id:", test_fram.values[1][0])




## === cell 1
class LeafDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform):
        self.train_fram = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.train_fram)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.train_fram.values[idx][0])

        image = cv2.imread(img_name)
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
        image = image.transpose((2, 0, 1)).astype(np.float32)
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
        img = transform.resize(
            image, (new_h, new_w), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
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

        if h <= new_h or w <= new_w:
            top = max((h - new_h) // 2, 0)
            left = max((w - new_w) // 2, 0)
            image = image[top : top + new_h, left : left + new_w]
            if image.shape[0] != new_h or image.shape[1] != new_w:
                image = transform.resize(
                    image, (new_h, new_w), preserve_range=True, anti_aliasing=True
                ).astype(np.float32)
            return {"image": image, "labels": labels}

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
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
        x = x / 255.0
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



## === cell 2
batch_size = 32
num_workers = 2  # safer in Kaggle notebook/containers
test_loader = DataLoader(
    leafDatasets,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
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


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


y_pred_logits = np.empty(shape=(0, 6), dtype=np.float32)

with torch.no_grad():
    stream = tqdm(test_loader, desc=f"Infer ({device})")
    for _, sample in enumerate(stream):
        X = sample["image"].to(device, non_blocking=True)
        X = X.float()
        pred = resnet(X)  # logits
        y_pred_logits = np.vstack((y_pred_logits, to_numpy(pred)))

print("Pred logits shape:", y_pred_logits.shape)



## === cell 6
y_pred_logits[:2]



## === cell 7
probs = 1.0 / (1.0 + np.exp(-y_pred_logits))

thr = 0.50

indices = []
for p in probs:
    idxs = np.where(p >= thr)[0].tolist()
    if len(idxs) == 0:
        idxs = [int(np.argmax(p))]
    indices.append(idxs)

print("Example predicted indices:", indices[0])



## === cell 8
labels = ["complex", "frog_eye_leaf_spot", "healthy", "powdery_mildew", "rust", "scab"]

testlabels = []
for idxs in indices:
    temp = [labels[i] for i in idxs]
    testlabels.append(" ".join(temp))

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
