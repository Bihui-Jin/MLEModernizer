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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

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
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8375642187972197

# 6. Current score

0.53587

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the dataset/IO bugs that prevent the notebook from running: the test image folder contains a nested `test_images/` directory that must be filtered out, and `MyTestDataset.__len__` incorrectly references `self.labels`. I also fix the missing model weights path by using torchvision’s built-in pretrained SqueezeNet weights (keeping the same model architecture and inference logic) so the code can run end-to-end in this environment. Finally, I ensure inference runs in `eval()` mode with `torch.no_grad()` and writes a valid `submission.csv` with the required `image_id,label` columns.'
- What this solution (achieved 0.53587) has done: 'Your 0.0 score is because the current code uses an ImageNet-pretrained SqueezeNet (1000 classes) to predict cassava labels (0–4), so predictions are essentially random and won’t approach the target accuracy. To move the score toward 0.8376 with minimal changes while preserving the core approach (single pretrained CNN, no training loop), I keep SqueezeNet inference but add a lightweight “nearest-prototype” classifier: compute class prototypes from the provided `train_images/train.csv` using the same pretrained model, then classify each test image by nearest prototype in feature space. This keeps the model architecture and loss/training logic unchanged (still no training), but makes predictions aligned to the competition’s 5 classes. I also switch inference to use a DataLoader for correctness/performance and keep the submission format identical.'

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision

from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

torch.manual_seed(0)
np.random.seed(0)



## === cell 1
TEST_PATH = "../input/cassava-leaf-disease-classification/test_images/"

files = sorted(
    [f for f in os.listdir(TEST_PATH) if os.path.isfile(os.path.join(TEST_PATH, f))]
)

assert (
    len(files) > 0
), f"No test image files found in {TEST_PATH}. Found entries: {os.listdir(TEST_PATH)[:10]}"



## === cell 2
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ]
)


def default_loader(path):
    img_pil = Image.open(path).convert("RGB")
    img_tensor = preprocess(img_pil)
    return img_tensor




## === cell 3
class MyTestDataset(Dataset):
    def __init__(self, files, loader=default_loader):
        self.files = files
        self.loader = loader

    def __getitem__(self, index):
        img_path = os.path.join(TEST_PATH, self.files[index])
        img = self.loader(img_path)
        return self.files[index], img

    def __len__(self):
        return len(self.files)




## === cell 4
testset = MyTestDataset(files)



## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = torchvision.models.SqueezeNet1_0_Weights.DEFAULT
model = torchvision.models.squeezenet1_0(weights=weights)
model = model.to(device)
model.eval()

feature_extractor = model.features
feature_extractor.eval()



## === cell 6
TRAIN_CSV = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMG_DIR = "../input/cassava-leaf-disease-classification/train_images/"

train_df = pd.read_csv(TRAIN_CSV)
assert {"image_id", "label"}.issubset(train_df.columns)


class MyTrainDataset(Dataset):
    def __init__(self, df, img_dir, loader=default_loader):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.loader = loader

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        x = self.loader(img_path)
        y = int(row["label"])
        return x, y


CAP = min(6000, len(train_df))
train_df_cap = train_df.iloc[:CAP].copy()

trainset = MyTrainDataset(train_df_cap, TRAIN_IMG_DIR)
trainloader = DataLoader(
    trainset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

num_classes = 5
feat_dim = 512  # SqueezeNet1.0 final conv channels
sum_feats = torch.zeros(num_classes, feat_dim, device=device)
count = torch.zeros(num_classes, device=device)

with torch.no_grad():
    for xb, yb in trainloader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        feats = feature_extractor(xb)  # [B, 512, H, W]
        feats = feats.mean(dim=(2, 3))  # global average pool -> [B, 512]
        feats = torch.nn.functional.normalize(feats, dim=1)

        for c in range(num_classes):
            mask = yb == c
            if mask.any():
                sum_feats[c] += feats[mask].sum(dim=0)
                count[c] += mask.sum()

prototypes = sum_feats / count.clamp_min(1.0).unsqueeze(1)
prototypes = torch.nn.functional.normalize(prototypes, dim=1)

print(
    "Built prototypes from",
    int(count.sum().item()),
    "train images; per-class counts:",
    count.detach().cpu().numpy().astype(int).tolist(),
)



## === cell 7
testloader = DataLoader(
    testset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

pred_labels = []
pred_files = []

with torch.no_grad():
    for file_batch, xb in testloader:
        xb = xb.to(device, non_blocking=True)
        feats = feature_extractor(xb).mean(dim=(2, 3))
        feats = torch.nn.functional.normalize(feats, dim=1)

        sims = feats @ prototypes.T  # [B, 5]
        yhat = torch.argmax(sims, dim=1).detach().cpu().numpy().astype(int).tolist()

        pred_labels.extend(yhat)
        pred_files.extend(list(file_batch))

submission = pd.DataFrame({"image_id": pred_files, "label": pred_labels})



## === cell 8
submission



## === cell 9
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
