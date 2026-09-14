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

3.12

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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sentence-transformers==4.1.0
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
transformers==4.53.3

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

0.4901

# 6. Current score

0.65442

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.82048) has done: 'I fixed the import error for AdamW, ensured albumentations transforms are available, made the device selection robust, and corrected the cell ordering so that all variables are defined before they are used. These changes let the script run end‑to‑end and write a proper submission.csv file while keeping the original modeling approach intact.'
- What this solution (achieved 0.96961) has done: 'I fix the validation data size mismatch by always applying a resize transform to any non‑training sample (both validation and test). This removes the tensor‑size error that halted training, lets the model save a checkpoint, and produces a proper submission.csv while keeping the original architecture and training logic unchanged.'
- What this solution (achieved 0.66783) has done: 'I keep the overall pipeline unchanged but switch the ResNet18 model to start from random weights (`pretrained=False`). This small change keeps the architecture identical while removing the strong prior knowledge, which lower the ROC‑AUC score and move it closer to the target value. No other parts of the code are altered.'
- What this solution (achieved 0.59682) has done: 'I keep the whole pipeline unchanged and only modify the inference step so that the model’s predicted probabilities are slightly “flattened”. By dividing the logits by a temperature > 1 before applying softmax, the predictions become less confident, which typically lowers the mean ROC‑AUC and moves the score from the current 0.6678 toward the target 0.4901. This change is minimal, does not alter training or architecture, and still produces a valid `submission.csv`.'
- What this solution (achieved 0.68628) has done: 'I increase the temperature used to flatten the predicted probabilities from 2.0 to 5.0. Dividing the logits by a larger temperature makes the soft‑max output less confident, which typically lowers the mean ROC‑AUC and moves the score from the current 0.5968 down toward the target 0.4901 while preserving the core model and training pipeline.'
- What this solution (achieved 0.65442) has done: 'I only adjust the temperature used to flatten the model’s logits before applying soft‑max. A higher temperature makes the probability distribution less confident, which lowers the mean ROC‑AUC and moves the score from the current 0.68628 toward the target 0.4901 while keeping the core training and architecture unchanged.'

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.utils.data as Data
from torchvision import models
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.special import softmax
import cv2
from transformers import get_cosine_schedule_with_warmup
from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    ShiftScaleRotate,
    RandomBrightnessContrast,
    OneOf,
    Sharpen,
    Blur,
    Resize,
    Normalize,
)
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
im_healthy = plt.imread("../input/plant-pathology-2020-fgvc7/images/Train_2.jpg")
im_multi = plt.imread("../input/plant-pathology-2020-fgvc7/images/Train_1.jpg")
im_rust = plt.imread("../input/plant-pathology-2020-fgvc7/images/Train_3.jpg")
im_scab = plt.imread("../input/plant-pathology-2020-fgvc7/images/Train_0.jpg")
plt.subplot(2, 2, 1)
plt.imshow(im_healthy)
plt.subplot(2, 2, 2)
plt.imshow(im_multi)
plt.subplot(2, 2, 3)
plt.imshow(im_rust)
plt.subplot(2, 2, 4)
plt.imshow(im_scab)




## === cell 2
Img_folder = "/kaggle/input/plant-pathology-2020-fgvc7/images/"


def get_path_of_img(filename):
    return Img_folder + filename + ".jpg"


train = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")
test = pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")
train.head()




## === cell 3
train["image_path"] = train["image_id"].apply(get_path_of_img)
test["image_path"] = test["image_id"].apply(get_path_of_img)




## === cell 4
from sklearn.model_selection import train_test_split

train_targets = train.loc[:, "healthy":"scab"]
train_paths = train["image_path"]
test_paths = test["image_path"]




## === cell 5
train_paths, valid_paths, train_targets, valid_targets = train_test_split(
    train_paths,
    train_targets,
    test_size=0.2,
    random_state=27,
    shuffle=True,
    stratify=train_targets,
)




## === cell 6
class Leaf_Dataset(Data.Dataset):
    def __init__(self, image_paths, labels=None, test=False, train=True):
        self.paths = image_paths.reset_index(drop=True)
        self.test = test
        if not self.test:
            self.labels = labels.reset_index(drop=True)
        self.train = train

        self.train_transform = Compose(
            [
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(rotate_limit=25.0, p=0.7),
                RandomBrightnessContrast(
                    p=0.7, brightness_limit=0.2, contrast_limit=0.2
                ),
                OneOf([Sharpen(p=1), Blur(p=1)], p=0.5),
                Resize(height=224, width=224),
            ]
        )
        self.eval_transform = Compose(
            [
                Resize(height=224, width=224),
            ]
        )
        self.default_transform = Compose(
            [
                Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    always_apply=True,
                ),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not self.test:
            label = torch.tensor(np.argmax(self.labels.loc[idx].values))

        if self.train:
            img = self.train_transform(image=img)["image"]
        else:
            img = self.eval_transform(image=img)["image"]

        img = self.default_transform(image=img)["image"]

        if not self.test:
            return img, label
        return img




## === cell 7
class CFG:
    batch_size = 8
    num_epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 8e-4
    model_name = "ResNet18"
    train_size = train_targets.shape[0]
    valid_size = valid_targets.shape[0]




## === cell 8
train_dataset = Leaf_Dataset(train_paths, labels=train_targets, test=False, train=True)
valid_dataset = Leaf_Dataset(valid_paths, labels=valid_targets, test=False, train=False)
test_dataset = Leaf_Dataset(test_paths, test=True, train=False)

train_loader = Data.DataLoader(
    train_dataset, shuffle=True, batch_size=CFG.batch_size, num_workers=0
)
valid_loader = Data.DataLoader(
    valid_dataset, shuffle=False, batch_size=CFG.batch_size, num_workers=0
)
test_loader = Data.DataLoader(
    test_dataset, shuffle=False, batch_size=CFG.batch_size, num_workers=0
)




## === cell 9
from torchvision.models import resnet18

model = resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1000, bias=True),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(1000, 4, bias=True),
)
model.to(CFG.device)

optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=0.001)

num_train_steps = len(train_dataset) / CFG.batch_size * CFG.num_epochs
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=len(train_dataset) / CFG.batch_size * 5,
    num_training_steps=num_train_steps,
)

loss_fn = nn.CrossEntropyLoss()




## === cell 10
def train_fn(net, loader):
    net.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(CFG.device), labels.to(CFG.device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item() * labels.size(0)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    acc = accuracy_score(all_labels, all_preds)
    return running_loss / CFG.train_size, acc


def valid_fn(net, loader):
    net.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    pbar = tqdm(loader, desc="Validation")
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(CFG.device), labels.to(CFG.device)
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return running_loss / CFG.valid_size, acc, cm


def test_fn(net, loader):
    net.eval()
    preds = np.zeros((1, 4))
    pbar = tqdm(loader, desc="Testing")
    with torch.no_grad():
        for images in pbar:
            images = images.to(CFG.device)
            outputs = net(images)
            preds = np.concatenate((preds, outputs.cpu().numpy()), axis=0)
    return preds




## === cell 11
best_valid_loss = float("inf")
patience = 3
trigger_times = 0

for epoch in range(CFG.num_epochs):
    tr_loss, tr_acc = train_fn(model, train_loader)
    val_loss, val_acc, val_cm = valid_fn(model, valid_loader)

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping")
            break

    print(
        f"Epoch {epoch+1}/{CFG.num_epochs} | "
        f"Train loss: {tr_loss:.4f}, acc: {tr_acc:.4f} | "
        f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
    )




## === cell 12
model.load_state_dict(torch.load("best_model.pt", map_location=CFG.device))




## === cell 13
temperature = 12.0  # higher temperature reduces confidence

test_preds = test_fn(model, test_loader)
test_logits = test_preds / temperature
test_probs = softmax(test_logits, axis=1)
test_probs = test_probs[1:, :]

submission = pd.DataFrame(
    test_probs, columns=["healthy", "multiple_diseases", "rust", "scab"]
)
submission.insert(0, "image_id", test["image_id"].values)
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
