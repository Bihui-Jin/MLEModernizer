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
from PIL import Image
from tqdm import tqdm
import copy
import pandas as pd
from torchvision import transforms
import torchvision
from torch import optim
import torch
import os
import random
from glob import glob



## === cell 1
transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomAffine(
            degrees=10, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
transform_valid = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)



## === cell 2
with open("../input/plant-pathology-2021-fgvc8/train.csv", "r") as f:
    csv_lines = f.readlines()[1:]
    for _ in range(5):
        random.shuffle(csv_lines)
    cnt = int(len(csv_lines) * 0.9)
    train_csv = csv_lines[:cnt]
    valid_csv = csv_lines[cnt:]



## === cell 3
all_tokens = set()
for line in train_csv:
    parts = line.strip().split(",", 1)
    if len(parts) != 2:
        continue
    labels_str = parts[1].strip()
    for tok in labels_str.split():
        all_tokens.add(tok)

label = sorted(list(all_tokens))
label2idx = {lab: idx for idx, lab in enumerate(label)}
label2idx




## === cell 4
class torchvision_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_root, csv, label2idx, transforms=None):
        self.data = csv
        self.image_path = data_root
        self.label2idx = label2idx
        self.transform = transforms
        self.num_classes = len(label2idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, label_str = self.data[idx].strip().split(",", 1)
        img = Image.open(os.path.join(self.image_path, image_name)).convert("RGB")
        x = self.transform(img) if self.transform else transforms.ToTensor()(img)

        y = torch.zeros(self.num_classes, dtype=torch.float32)
        for tok in label_str.strip().split():
            if tok in self.label2idx:
                y[self.label2idx[tok]] = 1.0
        return x, y




## === cell 5
TRAIN_IMG_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_IMG_DIR = "../input/plant-pathology-2021-fgvc8/test_images"

train_dataset = torchvision_Dataset(
    TRAIN_IMG_DIR, train_csv, label2idx, transform_train
)
valid_dataset = torchvision_Dataset(
    TRAIN_IMG_DIR, valid_csv, label2idx, transform_valid
)



## === cell 6
train_dataloaders = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True
)
valid_dataloaders = torch.utils.data.DataLoader(
    valid_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)



## === cell 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



## === cell 8
num_classes = len(label2idx)
model_ft = torchvision.models.efficientnet_b4(weights=None)
in_features = model_ft.classifier[1].in_features
model_ft.classifier[1] = torch.nn.Linear(in_features, num_classes)
model_ft.to(device)




## === cell 9
class FocalLoss(torch.nn.Module):
    """
    The focal loss for fighting against class-imbalance
    Expects:
      logits: [batch, num_classes]
      target: [batch, num_classes] (multi-hot float)
    """

    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1e-12  # prevent training from Nan-loss error

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        one_subtract_probs = 1.0 - probs
        probs_new = probs + self.epsilon
        one_subtract_probs_new = one_subtract_probs + self.epsilon
        log_pt = target * torch.log(probs_new) + (1.0 - target) * torch.log(
            one_subtract_probs_new
        )
        pt = torch.exp(log_pt)
        focal_loss = -1.0 * (self.alpha * (1 - pt) ** self.gamma) * log_pt
        return torch.mean(focal_loss)




## === cell 10
criterion = FocalLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-3)
exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, 40, eta_min=1e-6)




## === cell 11
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    os.makedirs("outputs", exist_ok=True)

    for epoch in range(num_epochs):
        running_loss = 0.0
        train_corrects = 0
        train_data_cnt = 0

        train_progress_bar = tqdm(train_dataloaders)
        for inputs, labels in train_progress_bar:
            model.train()

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)  # [B, C]
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            hard_labels = torch.argmax(labels, dim=1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == hard_labels).item()
            train_data_cnt += inputs.size(0)

            train_progress_bar.set_description(
                f" Epoch[{epoch+1}/{num_epochs}] train : runing_Loss {running_loss / max(1, train_data_cnt):.5f}, train_acc {train_corrects / max(1, train_data_cnt):.5f}"
            )

        scheduler.step()

        valid_corrects = 0
        valid_data_cnt = 0
        valid_progress_bar = tqdm(valid_dataloaders)
        for inputs, labels in valid_progress_bar:
            model.eval()

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                hard_labels = torch.argmax(labels, dim=1)

            valid_corrects += torch.sum(preds == hard_labels).item()
            valid_data_cnt += inputs.size(0)
            valid_progress_bar.set_description(
                f" Epoch[{epoch+1}/{num_epochs}] valid : valid_acc {valid_corrects / max(1, valid_data_cnt):.5f}"
            )

        epoch_acc = valid_corrects / len(valid_dataset)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f"outputs/{best_epoch}.pth")
            print(f"best epoch : {best_epoch}")

    model.load_state_dict(best_model_wts)
    return best_model_wts




## === cell 12
_ = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=1)



## === cell 13
ckpt_path = "../input/efficient-adam-focalloss/39.pth"
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    model_ft.load_state_dict(state, strict=False)
model_ft.to(device)



## === cell 14
idx2label = {idx: lab for lab, idx in label2idx.items()}



## === cell 15
sample_sub_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)

model_ft.eval()
submit = []

threshold = 0.5

for img_name in tqdm(sample_sub["image"].tolist()):
    img_path = os.path.join(TEST_IMG_DIR, img_name)
    img = Image.open(img_path).convert("RGB")
    img_t = transform_valid(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model_ft(img_t)
        probs = torch.sigmoid(logits).squeeze(0).detach().cpu()

    pred_idxs = (probs >= threshold).nonzero(as_tuple=False).view(-1).tolist()
    if len(pred_idxs) == 0:
        pred_idxs = [int(torch.argmax(probs).item())]

    pred_labels = " ".join([idx2label[i] for i in pred_idxs])
    submit.append([img_name, pred_labels])

submission = pd.DataFrame(submit, columns=["image", "labels"])
submission.to_csv("/kaggle/working/submission.csv", index=False)
print(submission.head())
print("Wrote /kaggle/working/submission.csv")
