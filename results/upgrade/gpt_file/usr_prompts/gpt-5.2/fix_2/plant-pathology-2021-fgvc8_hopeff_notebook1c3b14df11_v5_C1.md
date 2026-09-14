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

0.23671

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.23671) has done: 'I fix the runtime failure by removing the missing external checkpoint dependency and instead load the standard ImageNet pretrained weights for the same ResNeXt50 backbone (this keeps the architecture and inference semantics intact while allowing the notebook to run end-to-end). I also make the dataset image loading robust (force RGB) and ensure deterministic ordering of test files and no shuffling so the submission is stable and aligned. Finally, I add a safe CPU fallback if CUDA isn’t available and keep the submission format exactly as required (`image,labels`) with a `.csv` suffix.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)




## === cell 1
class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        backbone = models.resnext50_32x4d(
            weights=models.ResNeXt50_32X4D_Weights.DEFAULT
        )
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.backbone = backbone

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, images, labels=None):
        if self.training:
            y = self.backbone(images)
            loss = self.loss(y, labels)
            return loss
        else:
            pred = self.backbone(images)
            logits = pred.sigmoid()
            batched_labels = []
            for item in logits:
                batched_labels.append(torch.where(item > 0.5)[0].tolist())
            return batched_labels




## === cell 2
class Dataset:
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

        self.files = []
        for dirname, _, filenames in os.walk(self.root):
            for filename in filenames:
                if filename.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
                ):
                    self.files.append([os.path.join(dirname, filename), filename])

        self.files.sort(key=lambda x: x[1])

        self.trans = T.Compose(
            [T.Resize((300, 300)), T.ToTensor(), T.Normalize(mean=0.5, std=1.0)]
        )

    def __getitem__(self, index):
        file, name = self.files[index]
        image = Image.open(file).convert("RGB")
        image = self.trans(image)
        return image, name

    def __len__(self):
        return len(self.files)




## === cell 3
batch_size = 32
device = "cuda:0" if torch.cuda.is_available() else "cpu"

dataset = Dataset()
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,  # bugfix: don't shuffle test; keeps deterministic file order
    pin_memory=torch.cuda.is_available(),
    num_workers=2,
)

model = Model(dataloader.dataset.num_classes)
model.to(device)
_ = model.eval()

all_predict = []
with torch.no_grad():
    for images, names in dataloader:
        images = images.to(device, non_blocking=True)
        batched_labels = model(images)
        for name, labels in zip(names, batched_labels):
            all_predict.append(
                [
                    name,
                    " ".join(
                        [dataloader.dataset.index_to_name[index] for index in labels]
                    ),
                ]
            )

data = pd.DataFrame(all_predict, columns=("image", "labels"))
data.to_csv("submission.csv", index=False)
data
