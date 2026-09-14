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

0.7630470914127425

# 6. Current score

0.20825

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.20825) has done: 'I remove the unused “labeled_train_df” load (which can cause a file‑not‑found error) and simplify the label handling.  
The prediction step is changed to use a soft‑max followed by the top 3 class IDs; their label strings are joined with spaces so the submission matches the multi‑label F1 format. This small adjustment should raise the F1 score toward the target while keeping the original model architecture and training logic intact, and it guarantees that a valid `submission.csv` file is written.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing



## === cell 1
BATCH = 6
EPOCHS = 10
WEIGHT_DECAY = 0.0
LR = 1e-6
IM_SIZE = 728
TOP_K = 3  # number of classes to output per image

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"



## === cell 2
train_df = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
print(f"Train records: {len(train_df)}")



## === cell 3
le = preprocessing.LabelEncoder()
le.fit(train_df["labels"])
train_df["label_id"] = le.transform(train_df["labels"])
NUM_CL = len(le.classes_)
print(f"Number of classes: {NUM_CL}")



## === cell 4
class_map = dict(sorted(train_df[["label_id", "labels"]].values.tolist()))



## === cell 5
trainnum = 14800
valnum = 3700
tr_df = train_df[:trainnum]
X_Train, Y_Train = tr_df["image"].values, tr_df["label_id"].values

val_df = train_df[-valnum:]
X_val, Y_val = val_df["image"].values, val_df["label_id"].values



## === cell 6
Transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

Transformval = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.CenterCrop(int(IM_SIZE * 0.8)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)




## === cell 7
class GetData(Dataset):
    def __init__(self, dir_path, fnames, labels, transform):
        self.dir = dir_path
        self.fnames = fnames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir, self.fnames[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        if self.labels is None:
            return img, self.fnames[idx]  # test mode
        else:
            return img, self.labels[idx]  # train/val mode




## === cell 8
trainset = GetData(TRAIN_DIR, X_Train, Y_Train, Transform)
trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True)

valset = GetData(TRAIN_DIR, X_val, Y_val, Transformval)
valloader = DataLoader(valset, batch_size=BATCH, shuffle=False)



## === cell 9
model = torchvision.models.resnext101_32x8d()
model.fc = nn.Linear(2048, NUM_CL, bias=True)

checkpoint_path = os.path.join("../input/resnet-model/ResNext16.pth")
if os.path.exists(checkpoint_path):
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    print("Loaded custom ResNext checkpoint.")
else:
    print("Custom checkpoint not found – using ImageNet‑pretrained weights.")

model = model.to(DEVICE)
model.eval()



## === cell 10
test_fnames = [
    f for f in os.listdir(TEST_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))
]
print(f"Number of test images: {len(test_fnames)}")



## === cell 11
testset = GetData(TEST_DIR, test_fnames, None, Transformval)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

pred_records = []

with torch.no_grad():
    for imgs, fnames in testloader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        topk = torch.topk(probs, k=TOP_K, dim=1).indices.squeeze(0).cpu().numpy()
        pred_labels = " ".join([class_map[int(cls_id)] for cls_id in topk])
        pred_records.append([fnames[0], pred_labels])



## === cell 12
submission = pd.DataFrame(pred_records, columns=["image", "labels"])
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
