# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
from torchvision.transforms import ToPILImage
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR
from sklearn.model_selection import train_test_split, KFold
from PIL import Image
import cv2
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
import torchvision.utils as vutil
from mpl_toolkits.axes_grid1 import ImageGrid




## === cell 1
train_data = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
labels = sorted(train_data["breed"].unique())
label_to_idx = {label: idx for idx, label in enumerate(labels)}
train_data["number"] = train_data["breed"].map(label_to_idx)
train_data.shape




## === cell 2
test_dir = "/kaggle/input/dog-breed-identification/test"
file_names = [f for f in sorted(os.listdir(test_dir)) if f.lower().endswith(".jpg")]
file_ids = [os.path.splitext(f)[0] for f in file_names]
test_data = pd.DataFrame({"id": file_ids})
test_data.head()




## === cell 3
transforms_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
transforms_test = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)




## === cell 4
class Dog_Breed(Dataset):
    def __init__(self, df, transform=None, test=False, preloaded_images=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.image_ids = list(self.df["id"])
        self.test = test
        self.transform = transform

        base_path = (
            "/kaggle/input/dog-breed-identification/test"
            if self.test
            else "/kaggle/input/dog-breed-identification/train"
        )
        self.base_path = base_path

        if preloaded_images is not None:
            self.cached_images = preloaded_images
        else:
            self.cached_images = [
                Image.open(os.path.join(base_path, img_id + ".jpg")).convert("RGB")
                for img_id in self.image_ids
            ]

        if not self.test:
            self.label_nums = list(self.df["number"])

    def __getitem__(self, idx):
        image = self.cached_images[idx]
        if self.transform is not None:
            image = self.transform(image)
        if self.test:
            return image
        else:
            return image, self.label_nums[idx]

    def __len__(self):
        return len(self.image_ids)




## === cell 5
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


device = get_device()
torch.backends.cudnn.benchmark = True
device




## === cell 6
train = None
valid = None




## === cell 7
train_loader = None
valid_loader = None




## === cell 8
pass




## === cell 9
class MyResNet50(nn.Module):
    def __init__(self, num_classes=120):
        super(MyResNet50, self).__init__()
        self.net = models.resnet50(pretrained=True)
        for param in self.net.parameters():
            param.requires_grad = False
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.net(x)
        return x




## === cell 10
class EfficientNetCustom(nn.Module):
    def __init__(self, num_classes=120):
        super(EfficientNetCustom, self).__init__()
        self.net = EfficientNet.from_pretrained("efficientnet-b0")
        for param in self.net.parameters():
            param.requires_grad = False
        in_features = self.net._fc.in_features
        self.net._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.net(x)




## === cell 11
def train_model(
    model,
    train_loader,
    valid_loader,
    loss_fn,
    optimizer,
    epochs,
    device=torch.device("cuda:0"),
    test_loader=None,  # optional cached test loader
):
    net = model.to(device)
    best_epoch = 0
    best_score = 0.0
    best_model_state = None
    early_stopping_round = 3
    losses = []

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-4)
    for epoch_idx in range(epochs):
        acc = 0
        loss_sum = 0.0
        net.train()
        for x, y in tqdm(train_loader):
            optimizer.zero_grad()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_hat = net(x)
            loss_val = loss_fn(y_hat, y)
            loss_sum += loss_val.item()
            loss_val.backward()
            optimizer.step()
            acc += torch.sum(y_hat.argmax(dim=1) == y).item()
        scheduler.step()
        losses.append(loss_sum / len(train_loader))
        print(
            f"epoch: {epoch_idx} loss={loss_sum/(len(train_loader)*train_loader.batch_size):.4f} "
            f"train_acc={(acc/(len(train_loader)*train_loader.batch_size)):.4f}",
            end="",
        )

        test_acc = 0
        net.eval()
        with torch.no_grad():
            for x, y in tqdm(valid_loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                y_hat = net(x)
                test_acc += torch.sum(y_hat.argmax(dim=1) == y).item()
        print(f"  val_acc={(test_acc/(len(valid_loader)*valid_loader.batch_size)):.4f}")

        if test_acc > best_score:
            best_model_state = copy.deepcopy(net.state_dict())
            best_score = test_acc
            best_epoch = epoch_idx
            print("best epoch saved!")

        if epoch_idx - best_epoch >= early_stopping_round:
            break

    net.load_state_dict(best_model_state)

    if test_loader is None:
        testset = Dog_Breed(test_data, transform=transforms_test, test=True)
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=NUM_WORKERS_GLOBAL,
            pin_memory=True if device == "cuda" else False,
            persistent_workers=True,
        )

    predictions = []
    net.eval()
    with torch.no_grad():
        for x in tqdm(test_loader):
            x = x.to(device, non_blocking=True)
            y_hat = net(x)
            predictions.append(y_hat.cpu())
    predictions = torch.cat(predictions, dim=0)  # shape (N, 120)
    return predictions




## === cell 12
learn_rate = 0.001
momentum = 0.9
epoch = 15




## === cell 13
def preload_images(base_path, ids):
    """Load all images once and return a list of PIL images."""
    return [
        Image.open(os.path.join(base_path, img_id + ".jpg")).convert("RGB")
        for img_id in ids
    ]


NUM_WORKERS_GLOBAL = min(8, os.cpu_count() or 1)  # for test loader (parallel)
NUM_WORKERS_TRAIN = 0  # no extra workers when images are preloaded
BATCH_SIZE = 128

train_image_ids = train_data["id"].tolist()
train_images_full = preload_images(
    "/kaggle/input/dog-breed-identification/train", train_image_ids
)

kfold = KFold(n_splits=5, shuffle=True, random_state=2021)

testset_global = Dog_Breed(test_data, transform=transforms_test, test=True)
test_loader_global = torch.utils.data.DataLoader(
    testset_global,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=NUM_WORKERS_GLOBAL,
    pin_memory=True if device == "cuda" else False,
    persistent_workers=True,
)

all_predictions_sum = torch.zeros((len(test_data), 120), dtype=torch.float32)

for train_idx, val_idx in kfold.split(train_data):
    train_images = [train_images_full[i] for i in train_idx]
    val_images = [train_images_full[i] for i in val_idx]

    train_fold = train_data.iloc[train_idx].reset_index(drop=True)
    valid_fold = train_data.iloc[val_idx].reset_index(drop=True)

    trainset = Dog_Breed(
        train_fold, transform=transforms_train, preloaded_images=train_images
    )
    validset = Dog_Breed(
        valid_fold, transform=transforms_test, preloaded_images=val_images
    )

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS_TRAIN,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        validset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS_TRAIN,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=False,
    )

    model = MyResNet50()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.net.fc.parameters(), lr=learn_rate, weight_decay=1e-5)

    fold_pred = train_model(
        model,
        train_loader,
        valid_loader,
        loss_fn,
        optimizer,
        epoch,
        device,
        test_loader=test_loader_global,  # reuse cached loader
    )

    all_predictions_sum += fold_pred * 0.2  # each fold contributes 1/5

result = pd.DataFrame(all_predictions_sum.numpy(), columns=labels)
result = pd.concat([test_data, result], axis=1)
result.to_csv("dog_breed.csv", index=False)




## === cell 14
df = pd.read_csv("/kaggle/working/dog_breed.csv")
logits = df.iloc[:, 1:].values
prob_tensor = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=1)
df.iloc[:, 1:] = prob_tensor.numpy()
df.to_csv("dog_breed.csv", index=False)
