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
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.10

# 3. Installed packages

albumentations==2.0.8
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
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Target score

0.8373268929321263

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, random, gc
import numpy as np, pandas as pd
import cv2, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models
from tqdm.notebook import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
from glob import glob

seed = 42
print(f"setting everything to seed {seed}")
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



## === cell 1
data_dir = "../input/alaska2-image-steganalysis"
sample_size = 75000
val_size = int(sample_size * 0.25)

folder_names = ["Cover/", "JMiPOD/", "JUNIWARD/", "UERD/"]  # labels 0‑3
train_fn, val_fn = [], []
train_labels, val_labels = [], []

for label, folder in enumerate(folder_names):
    all_files = sorted(glob(f"{data_dir}/{folder}/*.jpg"))[:sample_size]
    np.random.shuffle(all_files)
    train_fn.extend(all_files[val_size:])
    train_labels.extend([label] * len(all_files[val_size:]))
    val_fn.extend(all_files[:val_size])
    val_labels.extend([label] * len(all_files[:val_size]))

train_df = pd.DataFrame({"ImageFileName": train_fn, "Label": train_labels})
val_df = pd.DataFrame({"ImageFileName": val_fn, "Label": val_labels})

print("Train samples:", len(train_df), "Val samples:", len(val_df))




## === cell 2
class Alaska2Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fn = self.df.loc[idx, "ImageFileName"]
        img = cv2.imread(fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = int(self.df.loc[idx, "Label"])
        if self.transform:
            img = self.transform(img)
        return {"image": img}, label


class Alaska2TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fn = self.df.loc[idx, "ImageFileName"]
        img = cv2.imread(fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return {"image": img}




## === cell 3
img_size = 512
train_transform = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
    ]
)
test_transform = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ]
)




## === cell 4
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, 4)

    def forward(self, x):
        return self.backbone(x)


model = Net().to(device)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/4119970509.py in <cell line: 0>()
     10 
     11 
---> 12 model = Net().to(device)
     13 # No checkpoint loading – we use the pretrained weights as‑is
     14 

/tmp/ipykernel_55/4119970509.py in __init__(self)
      4         super().__init__()
      5         self.backbone = models.efficientnet_b0(pretrained=True)
----> 6         self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, 4)
      7 
      8     def forward(self, x):

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in __getattr__(self, name)
   1926             if name in modules:
   1927                 return modules[name]
-> 1928         raise AttributeError(
   1929             f"'{type(self).__name__}' object has no attribute '{name}'"
   1930         )

AttributeError: 'Sequential' object has no attribute 'in_features'

## === cell 5
batch_size = 8
num_workers = 4

train_dataset = Alaska2Dataset(train_df, transform=train_transform)
val_dataset = Alaska2Dataset(val_df, transform=test_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)




## === cell 6
def alaska_weighted_auc(y_true, y_score):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
    areas = np.diff(tpr_thresholds)
    normalization = np.dot(areas, weights)

    competition_metric = 0.0
    for i, w in enumerate(weights):
        y_min, y_max = tpr_thresholds[i], tpr_thresholds[i + 1]
        mask = (tpr > y_min) & (tpr < y_max)
        if not np.any(mask):
            continue
        x_pad = np.linspace(fpr[mask][-1], 1, 100)
        x = np.concatenate([fpr[mask], x_pad])
        y = np.concatenate([tpr[mask], np.full_like(x_pad, y_max)]) - y_min
        score = metrics.auc(x, y)
        competition_metric += score * w
    return competition_metric / normalization




## === cell 7
test_filenames = sorted(glob(f"{data_dir}/Test/*.jpg"))
test_df = pd.DataFrame({"ImageFileName": test_filenames})

test_dataset = Alaska2TestDataset(test_df, transform=test_transform)
test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
)

model.eval()
all_preds = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting test"):
        inputs = batch["image"].to(device, dtype=torch.float)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        all_preds.append(probs)

all_preds = np.vstack(all_preds)  # shape (N, 4)
pred_labels = all_preds.argmax(axis=1)

new_scores = np.zeros(len(all_preds))
mask_nonzero = pred_labels != 0
new_scores[mask_nonzero] = all_preds[mask_nonzero, 1:].sum(axis=1)
new_scores[~mask_nonzero] = all_preds[~mask_nonzero, 0]

test_df["Id"] = test_df["ImageFileName"].apply(lambda x: os.path.basename(x))
test_df["Label"] = new_scores
submission_path = "submission.csv"
test_df[["Id", "Label"]].to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(test_df.head())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2213582444.py in <cell line: 0>()
      8 )
      9 
---> 10 model.eval()
     11 all_preds = []
     12 with torch.no_grad():

NameError: name 'model' is not defined
