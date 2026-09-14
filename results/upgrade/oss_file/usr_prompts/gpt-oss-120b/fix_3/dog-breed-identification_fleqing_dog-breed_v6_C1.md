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
import torchvision.utils as vutils
from mpl_toolkits.axes_grid1 import ImageGrid

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")




## === cell 1
train_data = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
labels = sorted(list(set(train_data["breed"])))
labels_num = [labels.index(b) for b in train_data["breed"]]
train_data = train_data.assign(number=labels_num)
train_data.shape




## === cell 2
test_dir = "/kaggle/input/dog-breed-identification/test"
test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
file_names = [
    os.path.splitext(f)[0] for f in sorted(test_files) if os.path.splitext(f)[0]
]
test_data = pd.DataFrame({"id": file_names})
test_data.head()




## === cell 3
transforms_train = transforms.Compose(
    [
        transforms.Resize(256),
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
    def __init__(self, csv_df, transform=None, test=False):
        super().__init__()
        self.csv_df = csv_df
        self.image_ids = list(csv_df["id"])
        self.test = test
        if not self.test:
            self.labels = list(csv_df["number"])
        self.transform = transform

    def __getitem__(self, idx):
        if self.test:
            img_path = os.path.join(
                "/kaggle/input/dog-breed-identification/test",
                self.image_ids[idx] + ".jpg",
            )
        else:
            img_path = os.path.join(
                "/kaggle/input/dog-breed-identification/train",
                self.image_ids[idx] + ".jpg",
            )
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.test:
            return image
        else:
            return image, self.labels[idx]

    def __len__(self):
        return len(self.image_ids)




## === cell 5
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


device = get_device()
device




## === cell 6
train_df, valid_df = train_test_split(train_data, test_size=0.2, random_state=42)




## === cell 7
num_workers = min(4, os.cpu_count() or 1)
trainset_simple = Dog_Breed(train_df, transform=transforms_train)
validset_simple = Dog_Breed(valid_df, transform=transforms_test)
train_loader_simple = DataLoader(
    trainset_simple,
    batch_size=32,
    shuffle=True,
    drop_last=False,
    num_workers=num_workers,
    pin_memory=True,
)
valid_loader_simple = DataLoader(
    validset_simple,
    batch_size=32,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
    pin_memory=True,
)




## === cell 8
dataiter = iter(train_loader_simple)
images, label = next(dataiter)
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 4))
for i, ax in enumerate(axes):
    img = images[i].cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    ax.imshow(np.clip(img, 0, 1))
    ax.set_title(f"Label: {labels[label[i].item()]}")
    ax.axis("off")
plt.tight_layout()
plt.show()




## === cell 9
class MyResNet50(nn.Module):
    def __init__(self, num_classes=120):
        super(MyResNet50, self).__init__()
        self.net = models.resnet50(pretrained=True)
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.net(x)




## === cell 10
class EfficientNetCustom(nn.Module):
    def __init__(self, num_classes=120):
        super(EfficientNetCustom, self).__init__()
        from efficientnet_pytorch import EfficientNet

        self.net = EfficientNet.from_pretrained("efficientnet-b0")
        for p in self.net.parameters():
            p.requires_grad = False
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
    device,
    predict_test: bool = True,
):
    """
    Core training loop unchanged. Added `predict_test` flag to optionally skip
    the expensive test‑set inference (used for intermediate CV folds).
    """
    net = model.to(device)
    best_epoch = 0
    best_score = 0.0
    best_state = None
    early_stop = 3
    scheduler = ExponentialLR(optimizer, gamma=0.9, verbose=False)

    for epoch in range(epochs):
        net.train()
        acc = 0
        loss_sum = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            logits = net(x)
            loss = loss_fn(logits, y)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
            acc += (logits.argmax(dim=1) == y).sum().item()
        scheduler.step()
        train_acc = acc / len(train_loader.dataset)
        print(
            f"epoch {epoch} loss={loss_sum/len(train_loader):.4f} train_acc={train_acc:.4f}",
            end=" ",
        )

        net.eval()
        val_acc = 0
        with torch.no_grad():
            for x, y in tqdm(valid_loader, desc=f"Epoch {epoch} [val]"):
                x, y = x.to(device), y.to(device)
                logits = net(x)
                val_acc += (logits.argmax(dim=1) == y).sum().item()
        val_acc = val_acc / len(valid_loader.dataset)
        print(f"val_acc={val_acc:.4f}")

        if val_acc > best_score:
            best_score = val_acc
            best_state = copy.deepcopy(net.state_dict())
            best_epoch = epoch
            print("  --> new best model saved")
        if epoch - best_epoch >= early_stop:
            print("Early stopping")
            break

    net.load_state_dict(best_state)

    if not predict_test:
        return None

    test_dataset = Dog_Breed(test_data, transform=transforms_test, test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    preds = []
    net.eval()
    with torch.no_grad():
        for x in tqdm(test_loader, desc="Predict test"):
            x = x.to(device)
            logits = net(x)
            preds.append(logits.cpu())
    preds = torch.cat(preds, dim=0)  # shape (N_test, 120)
    return preds




## === cell 12
learn_rate = 0.001
epoch_num = 10




## === cell 13
kfold = KFold(n_splits=5, shuffle=True, random_state=2021)
preds = None  # will hold predictions from the final fold
for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
    print(f"--- Fold {fold+1} ---")
    train_fold = train_data.iloc[train_idx]
    valid_fold = train_data.iloc[val_idx]

    train_set = Dog_Breed(train_fold, transform=transforms_train)
    valid_set = Dog_Breed(valid_fold, transform=transforms_test)

    train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=32,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = MyResNet50()  # re‑initialize for each fold
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.net.fc.parameters(), lr=learn_rate, weight_decay=1e-5)

    predict_test = fold == kfold.get_n_splits() - 1
    fold_preds = train_model(
        model,
        train_loader,
        valid_loader,
        loss_fn,
        optimizer,
        epoch_num,
        device,
        predict_test=predict_test,
    )
    if predict_test:
        preds = fold_preds

result_df = pd.DataFrame(preds.numpy(), columns=labels)
result_df = pd.concat([test_data.reset_index(drop=True), result_df], axis=1)
result_df.to_csv("dog_breed.csv", index=False)




## === cell 14
df = pd.read_csv("dog_breed.csv")
logits = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32)
probs = torch.softmax(logits, dim=1).numpy()
df.iloc[:, 1:] = probs
df.to_csv("dog_breed.csv", index=False)
