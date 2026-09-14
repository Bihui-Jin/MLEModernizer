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

0.43748

# 6. Current score

11.47699

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 11.47699) has done: 'I fix the undefined `i` used for `random_state` and correct how test IDs are collected (the current code is accidentally listing a directory entry, producing an empty `id` and the path `.../test/.jpg`). I also ensure predictions are converted to proper probabilities with `softmax` before writing the submission, so the CSV matches the competition’s log-loss requirement without needing a second failing post-process cell. Finally, I write the submission using the exact sample submission column order to guarantee format correctness and avoid column mismatches.'

# 9. Code solution

## === cell 0
import os
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
from PIL import Image


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(42)



## === cell 1
train_data = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
labels = sorted(list(set(list(train_data["breed"]))))
labels_num = []
for i in range(len(train_data)):
    labels_num.append(labels.index(train_data["breed"][i]))
train_data["number"] = labels_num
train_data.shape



## === cell 2
test_dir = "/kaggle/input/dog-breed-identification/test"
file_names = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
file_names = [name[:-4] for name in file_names]
test_data = pd.DataFrame({"id": file_names})
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
    def __init__(self, train_csv, transform=None, test=False):
        super().__init__()
        self.train_csv = train_csv.reset_index(drop=True)
        self.image_path = list(self.train_csv["id"])
        self.test = test
        if not self.test:
            self.label_nums = list(self.train_csv["number"])
        self.transform = transform

    def __getitem__(self, idx):
        if self.test:
            img_path = os.path.join(
                "/kaggle/input/dog-breed-identification/test",
                self.image_path[idx] + ".jpg",
            )
        else:
            img_path = os.path.join(
                "/kaggle/input/dog-breed-identification/train",
                self.image_path[idx] + ".jpg",
            )

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if not self.test:
            label = self.label_nums[idx]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.image_path)




## === cell 5
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


device = get_device()
device



## === cell 6
train, valid = train_test_split(
    train_data, test_size=0.2, random_state=42, stratify=train_data["breed"]
)
trainset = Dog_Breed(train, transform=transforms_train)
validset = Dog_Breed(valid, transform=transforms_test)



## === cell 7
train_loader = DataLoader(
    trainset,
    batch_size=32,
    shuffle=True,
    drop_last=False,
    num_workers=2,
    pin_memory=True,
)
valid_loader = DataLoader(
    validset,
    batch_size=32,
    shuffle=False,
    drop_last=False,
    num_workers=2,
    pin_memory=True,
)




## === cell 8
class MyResNet50(nn.Module):
    def __init__(self, num_classes=120):
        super(MyResNet50, self).__init__()
        self.net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in self.net.parameters():
            param.requires_grad = False
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.net(x)




## === cell 9
def train_model(model, train_loader, valid_loader, loss, optimizer, epoch, device):
    net = model.to(device)
    best_epoch = 0
    best_score = -1.0
    best_model_state = None
    early_stopping_round = 10

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-4)

    for ep in range(epoch):
        acc = 0
        loss_sum = 0.0
        net.train()

        for x, y in tqdm(train_loader, desc=f"train ep {ep}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_hat = net(x)
            loss_temp = loss(y_hat, y)
            loss_sum += float(loss_temp.detach().cpu())
            loss_temp.backward()
            optimizer.step()

            acc += int(torch.sum(y_hat.argmax(dim=1) == y).detach().cpu())

        scheduler.step()

        train_acc = acc / len(trainset)
        train_loss = loss_sum / len(train_loader)
        print("epoch:", ep, "loss=", train_loss, "训练集准确度=", train_acc, end="")

        test_acc = 0
        net.eval()
        with torch.no_grad():
            for x, y in tqdm(valid_loader, desc=f"valid ep {ep}", leave=False):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                y_hat = net(x)
                test_acc += int(torch.sum(y_hat.argmax(dim=1) == y).detach().cpu())

        valid_acc = test_acc / len(validset)
        print(" 验证集准确度", valid_acc)

        if test_acc > best_score:
            best_model_state = copy.deepcopy(net.state_dict())
            best_score = test_acc
            best_epoch = ep
            print("best epoch save!")
        if ep - best_epoch >= early_stopping_round:
            break

    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    testset = Dog_Breed(test_data, transform=transforms_test, test=True)
    test_loader = DataLoader(
        testset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
    )

    predictions = []
    net.eval()
    with torch.no_grad():
        for x in tqdm(test_loader, desc="infer", leave=False):
            x = x.to(device, non_blocking=True)
            logits = net(x)
            probs = F.softmax(
                logits, dim=1
            )  # Metric-aligned: submission must be probabilities.
            predictions.append(probs.detach().cpu())

    prediction = torch.cat(predictions, dim=0)  # (N,120) float tensor on CPU
    return prediction




## === cell 10
learn_rate = 0.001
momentum = 0.9
epoch = 30



## === cell 11
model = MyResNet50()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.net.fc.parameters(), lr=learn_rate, weight_decay=1e-5)

prediction = train_model(
    model, train_loader, valid_loader, loss_fn, optimizer, epoch, device
)

sample_sub = pd.read_csv("/kaggle/input/dog-breed-identification/sample_submission.csv")
sub = sample_sub.copy()
sub.iloc[:, 1:] = prediction.numpy()

vals = sub.iloc[:, 1:].to_numpy(dtype=np.float64)
vals = np.clip(vals, 1e-15, 1.0)
vals = vals / vals.sum(axis=1, keepdims=True)
sub.iloc[:, 1:] = vals

sub_path = "/kaggle/working/dog_breed.csv"
sub.to_csv(sub_path, index=False)
print("Wrote submission to:", sub_path)



## === cell 12
assert os.path.exists("/kaggle/working/dog_breed.csv")
df_check = pd.read_csv("/kaggle/working/dog_breed.csv")
print(df_check.shape)
print(df_check.head(2))
