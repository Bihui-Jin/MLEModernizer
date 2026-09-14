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

# 8. Previous improvement plans

- What this solution (achieved 0.6813) has done: 'Implemented fixes:
- Replaced missing `efficientnet_pytorch` with torchvision’s EfficientNet implementation.
- Added necessary imports (`torchvision.models`, `PIL.Image`).
- Updated the `Net` class to use `models.efficientnet_b0` and correctly extract features.
- Adjusted dataset classes to convert OpenCV images to PIL before applying transforms, ensuring compatibility with torchvision augmentations.
- Kept the rest of the pipeline unchanged, guaranteeing a valid CSV submission.'
- What this solution (achieved 0.72032) has done: 'I add ImageNet‑style normalization to the training and test augmentations, raise the training epochs to three, and simplify the post‑processing so the weighted AUC uses the summed probability of the three stego classes directly (instead of relying on the predicted label). These minimal tweaks keep the EfficientNet‑B0 backbone unchanged while giving the model better calibrated inputs and more learning time, which should raise the validation Weighted AUC toward the target score.'
- What this solution (achieved 0.74895) has done: 'Optimized data loading by avoiding unnecessary PIL conversions for validation and test, increased DataLoader workers with persistent workers, and added mixed‑precision (AMP) training/inference to speed up GPU computation while keeping the model architecture and training logic unchanged. These changes reduce I/O and compute overhead, ensuring the script completes within the 600‑second limit without affecting result accuracy.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.models as models
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
default_data_dir = os.path.join(os.getcwd(), "input", "alaska2-image-steganalysis")
data_dir = os.getenv("DATA_DIR", default_data_dir)

train_transform = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class Alaska2Dataset(Dataset):
    def __init__(self, df, augmentations=None):
        self.df = df.reset_index(drop=True)
        self.augment = augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "ImageFileName"]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if self.augment:
            img = self.augment(img)
        label = self.df.loc[idx, "Label"]
        return img, label


model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)



## === cell 2
sample_size_per_class = 5000  # smaller subset for quicker I/O
val_ratio = 0.25
folder_names = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]  # labels 0‑3

train_fns, train_labels = [], []
val_fns, val_labels = [], []

for label, folder in enumerate(folder_names):
    all_files = sorted(glob.glob(os.path.join(data_dir, folder, "*.jpg")))
    if len(all_files) > sample_size_per_class:
        all_files = all_files[:sample_size_per_class]
    random.shuffle(all_files)
    split = int(len(all_files) * val_ratio)
    val_fns.extend(all_files[:split])
    val_labels.extend([label] * split)
    train_fns.extend(all_files[split:])
    train_labels.extend([label] * (len(all_files) - split))

train_df = pd.DataFrame({"ImageFileName": train_fns, "Label": train_labels})
val_df = pd.DataFrame({"ImageFileName": val_fns, "Label": val_labels})

print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")



## === cell 3
batch_size = 32
num_workers = 8

train_loader = DataLoader(
    Alaska2Dataset(train_df, augmentations=train_transform),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    Alaska2Dataset(val_df, augmentations=test_transform),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
num_epochs = 6  # a modest increase to improve performance
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/551697747.py in <cell line: 0>()
      2 num_workers = 8
      3 
----> 4 train_loader = DataLoader(
      5     Alaska2Dataset(train_df, augmentations=train_transform),
      6     batch_size=batch_size,

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __init__(self, dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context, generator, prefetch_factor, persistent_workers, pin_memory_device, in_order)
    381             else:  # map-style
    382                 if shuffle:
--> 383                     sampler = RandomSampler(dataset, generator=generator)  # type: ignore[arg-type]
    384                 else:
    385                     sampler = SequentialSampler(dataset)  # type: ignore[arg-type]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/sampler.py in __init__(self, data_source, replacement, num_samples, generator)
    163 
    164         if not isinstance(self.num_samples, int) or self.num_samples <= 0:
--> 165             raise ValueError(
    166                 f"num_samples should be a positive integer value, but got num_samples={self.num_samples}"
    167             )

ValueError: num_samples should be a positive integer value, but got num_samples=0

## === cell 4
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for images, targets in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
    ):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item() * images.size(0)

    scheduler.step()
    avg_loss = epoch_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1} - Training loss: {avg_loss:.4f}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * images.size(0)
    val_avg = val_loss / len(val_loader.dataset)
    print(f"Epoch {epoch+1} - Validation loss: {val_avg:.4f}")
    model.train()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/325697399.py in <cell line: 0>()
      1 model.train()
----> 2 for epoch in range(num_epochs):
      3     epoch_loss = 0.0
      4     for images, targets in tqdm(
      5         train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False

NameError: name 'num_epochs' is not defined

## === cell 5
test_files = sorted(glob.glob(os.path.join(data_dir, "Test", "*.jpg")))
test_df = pd.DataFrame({"ImageFileName": test_files})
test_dataset = Alaska2Dataset(test_df, augmentations=test_transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

model.eval()
ids = []
preds = []
softmax = nn.Softmax(dim=1)

with torch.no_grad():
    for images, _ in tqdm(test_loader, desc="Inference"):
        images = images.to(device)
        logits = model(images)
        probs = softmax(logits)
        stego_score = probs[:, 1:].sum(dim=1).cpu().numpy()
        preds.extend(stego_score.tolist())

ids = [os.path.basename(p) for p in test_files]

submission = pd.DataFrame({"Id": ids, "Label": preds})
output_path = os.path.join(os.getcwd(), "submission.csv")
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers must have the same length
