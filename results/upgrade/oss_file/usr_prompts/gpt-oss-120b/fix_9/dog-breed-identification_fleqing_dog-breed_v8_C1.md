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
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

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
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Target score

1.04161

# 6. Current score

1.35993

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.35993) has done: 'I add the missing imports, fix undefined names, and ensure the script writes a proper CSV submission. No core modeling logic is changed; the fixes simply make the notebook runnable and produce a valid `dog_breed.csv` file.'

# 9. Code solution

## === cell 0
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn
import torch.utils.data
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import torchvision.models as models

from sklearn.model_selection import train_test_split, KFold

torch.backends.cudnn.benchmark = True



## === cell 1
train_data = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
labels = sorted(list(set(train_data["breed"])))
labels_num = [labels.index(b) for b in train_data["breed"]]
train_data["number"] = labels_num
train_data.shape



## === cell 2
file_names = sorted(
    [
        name
        for name in os.listdir("/kaggle/input/dog-breed-identification/test")
        if name.lower().endswith(".jpg")
    ]
)
file_names = [name[:-4] for name in file_names]  # strip .jpg
test_data = pd.DataFrame({"id": file_names})
test_data.head()



## === cell 3
transforms_train = A.Compose(
    [
        A.Resize(height=256, width=256),
        A.RandomCrop(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

transforms_test = A.Compose(
    [
        A.Resize(height=256, width=256),
        A.CenterCrop(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 4
class Dog_Breed(Dataset):
    def __init__(self, csv_data, transform=None, test=False):
        super().__init__()
        self.csv_data = csv_data
        self.image_ids = list(self.csv_data["id"])
        self.test = test
        if not self.test:
            self.label_nums = list(self.csv_data["number"])
        self.transform = transform

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(
            (
                "/kaggle/input/dog-breed-identification/test"
                if self.test
                else "/kaggle/input/dog-breed-identification/train"
            ),
            img_id + ".jpg",
        )
        img = cv2.imread(img_path)  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        if self.test:
            return img
        else:
            label = self.label_nums[idx]
            return img, label

    def __len__(self):
        return len(self.image_ids)




## === cell 5
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


device = get_device()
device



## === cell 6
seed = 42
train_split, valid_split = train_test_split(
    train_data, test_size=0.2, random_state=seed
)



## === cell 7
batch_size = 128
num_workers = min(8, os.cpu_count() or 1)

trainset_opt = Dog_Breed(train_split, transform=transforms_train)
validset_opt = Dog_Breed(valid_split, transform=transforms_test)

train_loader_opt = torch.utils.data.DataLoader(
    trainset_opt,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)
valid_loader_opt = torch.utils.data.DataLoader(
    validset_opt,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)



## === cell 8
dataiter = iter(train_loader_opt)
images, labels_batch = next(dataiter)
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 4))

for i, ax in enumerate(axes):
    img_np = images[i].cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    ax.imshow(np.clip(img_np, 0, 1))
    ax.set_title(f"Label: {labels[labels_batch[i].item()]}")
    ax.axis("off")

plt.tight_layout()
plt.show()




## === cell 9
class MyResNet50(nn.Module):
    def __init__(self, num_classes=len(labels)):
        super(MyResNet50, self).__init__()
        self.net = models.resnet50(pretrained=True)
        for param in self.net.parameters():
            param.requires_grad = False
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.net(x)




## === cell 10
class EfficientNetCustom(nn.Module):
    def __init__(self, num_classes=len(labels)):
        super(EfficientNetCustom, self).__init__()
        raise NotImplementedError("EfficientNet not available in this environment.")

    def forward(self, x):
        raise NotImplementedError




## === cell 11
def train_model(
    model,
    train_loader,
    valid_loader,
    loss_fn,
    optimizer,
    epochs,
    device,
    test_loader,
):
    net = model.to(device)
    best_epoch = 0
    best_score = 0.0
    best_state = None
    early_stopping_round = 3
    scaler = amp.GradScaler()
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-4)

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.0
        correct = 0
        for x, y in tqdm(train_loader, leave=False):
            optimizer.zero_grad()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with amp.autocast():
                logits = net(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            correct += (logits.argmax(dim=1) == y).sum().item()
        scheduler.step()
        train_acc = correct / (len(train_loader.dataset))
        print(
            f"epoch {epoch} train_loss={epoch_loss/len(train_loader):.4f} train_acc={train_acc:.4f}",
            end="",
        )

        net.eval()
        val_correct = 0
        with torch.no_grad():
            for x, y in tqdm(valid_loader, leave=False):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with amp.autocast():
                    logits = net(x)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
        val_acc = val_correct / (len(valid_loader.dataset))
        print(f" valid_acc={val_acc:.4f}")

        if val_correct > best_score:
            best_state = net.state_dict()
            best_score = val_correct
            best_epoch = epoch
            print("  ** best model saved **")
        if epoch - best_epoch >= early_stopping_round:
            print("Early stopping")
            break

    net.load_state_dict(best_state)

    net.eval()
    all_preds = []
    with torch.no_grad():
        for x in tqdm(test_loader, leave=False):
            x = x.to(device, non_blocking=True)
            with amp.autocast():
                logits = net(x)
            all_preds.append(logits.cpu())
    return torch.cat(all_preds, dim=0)




## === cell 12
learn_rate = 0.001
epochs = 15



## === cell 13
kfold = KFold(n_splits=5, shuffle=True, random_state=2021)

testset = Dog_Breed(test_data, transform=transforms_test, test=True)
test_loader = torch.utils.data.DataLoader(
    testset,
    batch_size=128,
    shuffle=False,
    drop_last=False,
    num_workers=min(8, os.cpu_count() or 1),
    pin_memory=True,
    persistent_workers=True,
)

all_predictions_sum = torch.zeros((len(test_data), len(labels)), dtype=torch.float32)

criterion = nn.CrossEntropyLoss()

for train_idx, val_idx in kfold.split(train_data):
    train_fold = train_data.iloc[train_idx].reset_index(drop=True)
    valid_fold = train_data.iloc[val_idx].reset_index(drop=True)

    trainset = Dog_Breed(train_fold, transform=transforms_train)
    validset = Dog_Breed(valid_fold, transform=transforms_test)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        drop_last=False,
        num_workers=min(8, os.cpu_count() or 1),
        pin_memory=True,
        persistent_workers=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        validset,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=min(8, os.cpu_count() or 1),
        pin_memory=True,
        persistent_workers=True,
    )

    model = MyResNet50()
    optimizer = optim.Adam(model.net.fc.parameters(), lr=learn_rate, weight_decay=1e-5)

    fold_preds = train_model(
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        epochs,
        device,
        test_loader,
    )
    all_predictions_sum += fold_preds

probabilities = torch.softmax(all_predictions_sum, dim=1).cpu().numpy()

submission = pd.DataFrame(probabilities, columns=labels)
submission = pd.concat([test_data.reset_index(drop=True), submission], axis=1)

submission_path = "dog_breed.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
