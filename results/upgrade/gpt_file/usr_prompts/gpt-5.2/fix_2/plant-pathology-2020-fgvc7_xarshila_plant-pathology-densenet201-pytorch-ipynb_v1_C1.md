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
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

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
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.9429711711064372

# 6. Current score

0.54161

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.54161) has done: 'I fix the immediate runtime error by removing the missing external weight file dependency and instead using a valid torchvision pretrained DenseNet201 backbone available in the Kaggle environment. I keep your overall approach (DenseNet backbone + linear head + log-softmax + exp at inference) but correct the output dimension from 6 to 4 to match the competition’s four labels, which should substantially improve ROC AUC versus the current misaligned outputs. I also fix the test dataset length bug (`__len__` currently breaks when `preload=False`) and ensure deterministic, correct image loading (RGB) and correct submission column order. The script run end-to-end and write `submission.csv`.'

# 9. Code solution

## === cell 0
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch import nn
import random
import gc
import os



## === cell 1
DATA_DIR = Path("../input/plant-pathology-2020-fgvc7")
CLASS_NAMES = np.array(["healthy", "multiple_diseases", "rust", "scab"])
BATCH_SIZE = 8
IMAGE_SIZE = (512, 512)
TEST_SPLIT = 0.2



## === cell 2
device = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(42)




## === cell 3
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        weights = torchvision.models.DenseNet201_Weights.DEFAULT
        self.backbone = torchvision.models.densenet201(weights=weights)
        self.fc = nn.Linear(1000, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)




## === cell 4
model = MyModel().to(device)

fallback_paths = [
    Path("../input/plant-pathology-models/acc_96_size_512.pth"),
    Path("/kaggle/input/plant-pathology-models/acc_96_size_512.pth"),
]
loaded = False
for p in fallback_paths:
    if p.exists():
        state = torch.load(p, map_location="cpu")
        model.load_state_dict(state, strict=False)
        loaded = True
        break

model.eval()




## === cell 5
class PlantPathologyTestDataset(Dataset):
    def __init__(self, root, data_df, transform=None, preload=False):
        self.root = Path(root)
        self.data_df = data_df.reset_index(drop=True)
        self.images = None
        self.transform = transform

        if preload:
            self.images = []
            for idx in range(len(self.data_df)):
                image_path = (
                    self.root / "images" / (self.data_df.loc[idx, "image_id"] + ".jpg")
                )
                image = Image.open(str(image_path)).convert("RGB")
                image = image.resize(IMAGE_SIZE)
                self.images.append(image.copy())

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if self.images is None:
            image_path = (
                self.root / "images" / (self.data_df.loc[idx, "image_id"] + ".jpg")
            )
            image = Image.open(str(image_path)).convert("RGB")
            image = image.resize(IMAGE_SIZE)
        else:
            image = self.images[idx]

        if self.transform:
            image = self.transform(image)
        return self.data_df.loc[idx, "image_id"], image




## === cell 6
weights = torchvision.models.DenseNet201_Weights.DEFAULT
preprocess = weights.transforms()
test_transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=preprocess.mean, std=preprocess.std),
    ]
)

data_df = pd.read_csv(DATA_DIR / "test.csv")
submission_data = PlantPathologyTestDataset(
    DATA_DIR, data_df, transform=test_transform, preload=True
)
submission_loader = DataLoader(
    submission_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=1,
    pin_memory=torch.cuda.is_available(),
)



## === cell 7
image_ids = []
h, m, r, s = [], [], [], []

model.eval()
with torch.no_grad():
    for batch_idx, (image_id_batch, data) in enumerate(submission_loader):
        data = data.to(device, non_blocking=True)
        pred_batch = model(data)  # log-probs
        prob_batch = torch.exp(pred_batch).detach().cpu().numpy()

        for image_id, res in zip(image_id_batch, prob_batch):
            h.append(float(res[0]))
            m.append(float(res[1]))
            r.append(float(res[2]))
            s.append(float(res[3]))
            image_ids.append(image_id)

        del data, pred_batch
        gc.collect()



## === cell 8
sub = pd.DataFrame(
    {"image_id": image_ids, "healthy": h, "multiple_diseases": m, "rust": r, "scab": s}
)

sub = data_df[["image_id"]].merge(sub, on="image_id", how="left")



## === cell 9
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Saved submission.csv with shape:", sub.shape)
