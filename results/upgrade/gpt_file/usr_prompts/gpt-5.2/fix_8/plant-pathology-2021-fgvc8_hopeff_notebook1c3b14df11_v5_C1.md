# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
pillow==11.3.0
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

0.71898430286242

# 6. Current score

0.80738

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.23671) has done: 'I fix the runtime failure by removing the missing external checkpoint dependency and instead load the standard ImageNet pretrained weights for the same ResNeXt50 backbone (this keeps the architecture and inference semantics intact while allowing the notebook to run end-to-end). I also make the dataset image loading robust (force RGB) and ensure deterministic ordering of test files and no shuffling so the submission is stable and aligned. Finally, I add a safe CPU fallback if CUDA isn’t available and keep the submission format exactly as required (`image,labels`) with a `.csv` suffix.'
- What this solution (achieved 0.90653) has done: 'The timeout is most likely dominated by slow per-sample CPU preprocessing (cv2 → RGB → PIL conversion → torchvision Resize/ToTensor) repeated 2× over ~15k training images plus test-time preprocessing. I keep the same model, epochs, loss, thresholding, and training loop, but replace the transform pipeline with an equivalent OpenCV+NumPy implementation (resize + normalize) that avoids PIL and reduces overhead. I also pre-encode the multi-label targets once in the dataset init (same semantics) to remove per-__getitem__ string parsing, and enable cuDNN benchmarking safely (determinism is already enforced) plus minor DataLoader tuning to reduce input stalls. These changes are provably equivalent in output (same resize size, same normalization, same label mapping) and primarily cut constant-factor CPU time.'
- What this solution (achieved 0.87679) has done: 'Your current score (0.90653) is much higher than the target (0.71898), so the goal is to *decrease* performance slightly toward the target while keeping the core model/training/inference logic intact. The smallest safe lever here (without changing architecture, loss, training loop, or feature extraction) is to adjust only the inference decision threshold used to convert sigmoid probabilities into labels, because this directly trades precision/recall and move mean F1 predictably. I keep everything else identical and just raise the threshold from 0.5 to a slightly more conservative value (0.80) to reduce predicted positives and lower F1 toward your target band. The submission format, ordering, and file writing remain unchanged.'
- What this solution (achieved 0.83969) has done: 'Your current score (0.87679) is above the target (0.71898), so we should *reduce* performance toward the target band with the smallest safe change. The most controlled lever that preserves the model/training/inference core logic is the sigmoid-to-label decision threshold, because it directly changes precision/recall and therefore mean F1 without altering architecture, loss, or training. I increase `PRED_THRESHOLD` from 0.80 to 0.92 to make predictions more conservative and typically lower mean F1 closer to the target. Everything else (data reading, preprocessing, model, optimizer, epochs, submission writing) remains identical to avoid unintended shifts.'
- What this solution (achieved 0.80738) has done: 'Your current score (0.83969) is still above the target (0.71898), so we should deliberately reduce performance toward the target band using the smallest, safest lever that doesn’t alter training or the model: the inference probability threshold. I only increase `PRED_THRESHOLD` (more conservative predictions → typically lower mean F1) and keep everything else identical to avoid unintended shifts. This should move the score downward toward ~0.72 while preserving submission format and deterministic alignment. The notebook still run end-to-end and write `submission.csv` with the required `image,labels` columns.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from PIL import Image  # kept for compatibility, though we avoid PIL in the hot path


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 1
PRED_THRESHOLD = 0.96


class Model(nn.Module):
    def __init__(self, num_classes, pred_threshold: float = 0.5):
        super().__init__()

        backbone = models.resnext50_32x4d(
            weights=models.ResNeXt50_32X4D_Weights.DEFAULT
        )
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.backbone = backbone

        self.loss = nn.BCEWithLogitsLoss()
        self.pred_threshold = float(pred_threshold)

    def forward(self, images, labels=None):
        if self.training:
            y = self.backbone(images)
            loss = self.loss(y, labels)
            return loss
        else:
            pred = self.backbone(images)
            logits = pred.sigmoid()
            mask = logits > self.pred_threshold
            return [
                torch.nonzero(mask[i], as_tuple=False).squeeze(1).tolist()
                for i in range(mask.size(0))
            ]




## === cell 2
_w = models.ResNeXt50_32X4D_Weights.DEFAULT
_MEAN = np.array(_w.transforms().mean, dtype=np.float32)
_STD = np.array(_w.transforms().std, dtype=np.float32)


def _read_rgb_cv2(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR uint8
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _cv2_preprocess_300(img_rgb_uint8: np.ndarray) -> torch.Tensor:
    img = cv2.resize(img_rgb_uint8, (300, 300), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0  # HWC, [0,1]
    img = (img - _MEAN) / _STD  # broadcast over channels
    img = np.transpose(img, (2, 0, 1))  # CHW
    return torch.from_numpy(img)  # float32 tensor on CPU


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_root):
        self.df = pd.read_csv(csv_path)
        self.root = image_root

        self.label_map = {
            "complex": ["疑难复杂", 0],
            "rust": ["锈菌，生锈", 1],
            "scab": ["疮痂病，斑点病", 2],
            "frog_eye_leaf_spot": ["青蛙眼叶斑", 3],
            "healthy": ["健康的", 4],
            "powdery_mildew": ["白粉病", 5],
        }
        self.index_to_name = {self.label_map[key][1]: key for key in self.label_map}
        self.name_to_index = {key: self.label_map[key][1] for key in self.label_map}
        self.num_classes = len(self.label_map)

        labels_series = self.df["labels"].astype(str).values
        y = np.zeros((len(self.df), self.num_classes), dtype=np.float32)
        for i, s in enumerate(labels_series):
            for lab in s.split():
                j = self.name_to_index.get(lab, None)
                if j is not None:
                    y[i, j] = 1.0
        self._targets = torch.from_numpy(y)  # CPU tensor, float32

    def __getitem__(self, index):
        row = self.df.iloc[index]
        name = row["image"]
        path = os.path.join(self.root, name)

        image = _read_rgb_cv2(path)
        image = _cv2_preprocess_300(image)
        return image, self._targets[index]

    def __len__(self):
        return len(self.df)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.root = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"
        self.label_map = {
            "complex": ["疑难复杂", 0],
            "rust": ["锈菌，生锈", 1],
            "scab": ["疮痂病，斑点病", 2],
            "frog_eye_leaf_spot": ["青蛙眼叶斑", 3],
            "healthy": ["健康的", 4],
            "powdery_mildew": ["白粉病", 5],
        }
        self.index_to_name = {self.label_map[key][1]: key for key in self.label_map}
        self.num_classes = len(self.label_map)

        files = []
        with os.scandir(self.root) as it:
            for e in it:
                if not e.is_file():
                    continue
                fn = e.name.lower()
                if fn.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                    files.append((e.path, e.name))
        files.sort(key=lambda x: x[1])
        self.files = files

    def __getitem__(self, index):
        file, name = self.files[index]
        image = _read_rgb_cv2(file)
        image = _cv2_preprocess_300(image)
        return image, name

    def __len__(self):
        return len(self.files)




## === cell 3
batch_size = 32
device = "cuda:0" if torch.cuda.is_available() else "cpu"

train_csv = "/kaggle/input/plant-pathology-2021-fgvc8/train.csv"
train_root = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"

train_dataset = TrainDataset(train_csv, train_root)

_num_workers = min(8, (os.cpu_count() or 2))
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=torch.cuda.is_available(),
    num_workers=_num_workers,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)

test_dataset = TestDataset()
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,  # keep deterministic file order for correct alignment
    pin_memory=torch.cuda.is_available(),
    num_workers=_num_workers,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)

model = Model(train_dataset.num_classes, pred_threshold=PRED_THRESHOLD).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

model.train()
epochs = 2  # unchanged
for epoch in range(epochs):
    running = 0.0
    n = 0
    for images, targets in train_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = model(images, targets)
        loss.backward()
        optimizer.step()

        running += float(loss.item()) * images.size(0)
        n += images.size(0)

    print(f"epoch {epoch+1}/{epochs} - train_loss: {running/max(n,1):.5f}")



## === cell 4
model.eval()
all_predict = []

with torch.no_grad():
    for images, names in test_loader:
        images = images.to(device, non_blocking=True)
        batched_labels = model(images)
        idx2name = test_loader.dataset.index_to_name
        for name, labels in zip(names, batched_labels):
            all_predict.append([name, " ".join([idx2name[index] for index in labels])])

data = pd.DataFrame(all_predict, columns=("image", "labels"))
data.to_csv("submission.csv", index=False)
data
