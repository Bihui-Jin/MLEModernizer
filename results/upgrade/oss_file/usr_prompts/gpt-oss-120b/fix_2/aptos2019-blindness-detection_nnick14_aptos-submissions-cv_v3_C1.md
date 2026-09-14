# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.8

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
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score



## === cell 1
train_transform = transforms.Compose(
    [
        transforms.Resize((320, 320)),
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=(-360, 360), shear=(-45, 45))], p=0.5
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.460, 0.247, 0.080], [0.249, 0.138, 0.081]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.460, 0.247, 0.080], [0.249, 0.138, 0.081]),
    ]
)




## === cell 2
class APTOSDataset(Dataset):
    """Eye images dataset."""

    def __init__(self, csv_file, filetype, transform=None):
        self.eye_frame = pd.read_csv(csv_file)
        self.filetype = filetype
        self.transform = transform

    def __len__(self):
        return len(self.eye_frame)

    def __getitem__(self, idx):
        if self.filetype == "train":
            img_name = os.path.join(
                "../input/aptos2019-blindness-detection/train_images",
                self.eye_frame.loc[idx, "id_code"] + ".png",
            )
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            label = int(self.eye_frame.loc[idx, "diagnosis"])
            return image, label
        else:
            img_name = os.path.join(
                "../input/aptos2019-blindness-detection/test_images",
                self.eye_frame.loc[idx, "id_code"] + ".png",
            )
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            img_id = self.eye_frame.loc[idx, "id_code"]
            return image, img_id




## === cell 3
full_train_dataset = APTOSDataset(
    csv_file="../input/aptos2019-blindness-detection/train.csv",
    filetype="train",
    transform=train_transform,
)
test_dataset = APTOSDataset(
    csv_file="../input/aptos2019-blindness-detection/test.csv",
    filetype="test",
    transform=test_transform,
)

train_len = int(0.9 * len(full_train_dataset))
val_len = len(full_train_dataset) - train_len
train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_len, val_len],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



## === cell 4
model = torchvision.models.resnet101(pretrained=True)

for ct, child in enumerate(model.children(), start=1):
    if ct < 5:
        for param in child.parameters():
            param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)
model = model.to(device)

weights_path = "/kaggle/input/trainingweights/train_valid_best.pth"
if os.path.exists(weights_path):
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        pretrained_loaded = True
    except Exception:
        pretrained_loaded = False
else:
    pretrained_loaded = False



## === cell 5
if not pretrained_loaded:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    num_epochs = 3  # keep short to respect runtime limits

    print("Starting fine‑tuning...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
        ):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
        print(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Val QWK={val_kappa:.4f}")

    torch.save(model.state_dict(), "fine_tuned_resnet101.pth")
else:
    print("Pre‑trained weights loaded; skipping training.")




## === cell 6
def compute_predictions(model, mode, data_loader, device):
    model.eval()
    if mode == "train":
        predictions = []
        correct_pred, num_examples = 0, 0
        for inputs, labels in tqdm(data_loader, desc="Train Eval", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            num_examples += labels.size(0)
            correct_pred += (preds == labels).sum().item()
        accuracy = correct_pred / num_examples * 100
        return predictions, accuracy
    else:
        predictions = []
        img_ids = []
        for inputs, img_id in tqdm(data_loader, desc="Test Inference", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            img_ids.extend(img_id)
        final_predictions = pd.DataFrame({"id_code": img_ids, "diagnosis": predictions})
        final_predictions.to_csv("submission.csv", index=False)
        return final_predictions




## === cell 7
val_preds, val_acc = compute_predictions(model, "train", val_loader, device)
print(f"Validation accuracy: {val_acc:.2f}%")

print("Computing Test Predictions...")
test_predictions_df = compute_predictions(model, "test", test_loader, device)
print("Submission file saved as submission.csv")
