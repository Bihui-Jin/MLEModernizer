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
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import train_test_split, KFold
from PIL import Image
import cv2

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.manual_seed(2021)
np.random.seed(2021)
random.seed(2021)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

DATA_DIR = "/kaggle/input/dog-breed-identification"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")



## === cell 1
train_data = pd.read_csv(LABELS_CSV)

labels = sorted(train_data["breed"].unique().tolist())
label_to_idx = {b: i for i, b in enumerate(labels)}
train_data["number"] = train_data["breed"].map(label_to_idx).astype(np.int64)

train_data.shape



## === cell 2
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
test_data = sample_sub[["id"]].copy()
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
    def __init__(self, train_csv, transform=None, test=False):
        super().__init__()
        self.train_csv = train_csv.reset_index(drop=True)

        if "id" in self.train_csv.columns:
            self.image_path = list(self.train_csv["id"].astype(str).values)
        else:
            self.image_path = []

        self.image_path = [
            p for p in self.image_path if isinstance(p, str) and len(p) > 0
        ]

        self.test = test
        if not self.test:
            self.label_nums = list(self.train_csv["number"].values)
        self.transform = transform

        if self.test and len(self.image_path) == 0:
            fns = sorted(
                [fn for fn in os.listdir(TEST_DIR) if fn.lower().endswith(".jpg")]
            )
            self.image_path = [fn[:-4] for fn in fns]

    def __getitem__(self, idx):
        if self.test:
            img_path = os.path.join(TEST_DIR, self.image_path[idx] + ".jpg")
        else:
            img_path = os.path.join(TRAIN_DIR, self.image_path[idx] + ".jpg")

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            image = Image.open(img_path).convert("RGB")
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img)

        if self.transform is not None:
            image = self.transform(image)

        if not self.test:
            label = int(self.label_nums[idx])
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
    train_data, test_size=0.2, random_state=2021, stratify=train_data["breed"]
)
trainset = Dog_Breed(train, transform=transforms_train)
validset = Dog_Breed(valid, transform=transforms_test)




## === cell 7
def seed_worker(worker_id):
    worker_seed = 2021 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(2021)

NUM_WORKERS = min(8, os.cpu_count() or 2)
PIN_MEMORY = torch.cuda.is_available()

train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=32,
    shuffle=True,
    drop_last=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)
valid_loader = torch.utils.data.DataLoader(
    validset,
    batch_size=32,
    shuffle=False,
    drop_last=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)



## === cell 8
DO_VIS = False
if DO_VIS:
    import matplotlib.pyplot as plt

    try:
        dataiter = iter(train_loader)
        images, label = next(dataiter)
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 4))

        for j, ax in enumerate(axes):
            image = images[j].numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            ax.imshow(image)
            ax.set_title(f"Label: {labels[int(label[j].item())]}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Visualization skipped:", repr(e))




## === cell 9
class MyResNet50(nn.Module):
    def __init__(self, num_classes=120):
        super(MyResNet50, self).__init__()
        try:
            self.net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except Exception:
            self.net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.net(x)
        return x




## === cell 10
class EfficientNetCustom(nn.Module):
    def __init__(self, num_classes=120):
        super(EfficientNetCustom, self).__init__()
        raise ImportError(
            "EfficientNetCustom requires the 'efficientnet_pytorch' package, which is not available here."
        )

    def forward(self, x):
        raise RuntimeError("EfficientNetCustom is unavailable in this environment.")




## === cell 11
testset_global = Dog_Breed(test_data, transform=transforms_test, test=True)
test_loader_global = torch.utils.data.DataLoader(
    testset_global,
    batch_size=64,
    shuffle=False,
    drop_last=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)




## === cell 12
def train_model(
    model,
    train_loader,
    valid_loader,
    loss,
    optimizer,
    epoch,
    device=torch.device("cuda:0"),
    test_loader=None,
):
    device = torch.device(device) if not isinstance(device, torch.device) else device

    net = model.to(device)

    if device.type != "cpu":
        net = net.to(memory_format=torch.channels_last)


    best_epoch = 0
    best_score = -1.0
    best_model_state = None
    early_stopping_round = 3
    losses = []

    scheduler = ExponentialLR(optimizer, gamma=0.9, verbose=True)
    for i in range(epoch):
        acc = 0
        loss_sum = 0.0
        net.train()
        for x, y in tqdm(train_loader, leave=False):
            optimizer.zero_grad(set_to_none=True)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if device.type != "cpu":
                x = x.contiguous(memory_format=torch.channels_last)
            y_hat = net(x)
            loss_temp = loss(y_hat, y)
            loss_sum += float(loss_temp.item())
            loss_temp.backward()
            optimizer.step()
            acc += torch.sum(y_hat.argmax(dim=1) == y).item()
        scheduler.step()
        losses.append(loss_sum / max(1, len(train_loader)))
        train_acc = acc / max(1, len(train_loader.dataset))
        print(
            "epoch:",
            i,
            "loss=",
            loss_sum / max(1, len(train_loader.dataset)),
            "训练集准确度=",
            train_acc,
            end="",
        )

        test_acc = 0
        net.eval()
        with torch.no_grad():
            for x, y in tqdm(valid_loader, leave=False):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if device.type != "cpu":
                    x = x.contiguous(memory_format=torch.channels_last)
                y_hat = net(x)
                test_acc += torch.sum(y_hat.argmax(dim=1) == y).item()
        val_acc = test_acc / max(1, len(valid_loader.dataset))
        print(" 验证集准确度", val_acc)
        if test_acc > best_score:
            best_model_state = copy.deepcopy(net.state_dict())
            best_score = test_acc
            best_epoch = i
            print("best epoch save!")
        if i - best_epoch >= early_stopping_round:
            break

    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    if test_loader is None:
        raise ValueError("test_loader must be provided (reused global loader).")

    predictions = []
    net.eval()
    with torch.no_grad():
        for x in tqdm(test_loader, leave=False):
            x = x.to(device, non_blocking=True)
            if device.type != "cpu":
                x = x.contiguous(memory_format=torch.channels_last)
            y_hat = net(x)
            predictions.append(y_hat.detach().cpu())
    predictions = torch.cat(predictions, dim=0)  # logits [N, 120]
    return predictions




## === cell 13
learn_rate = 0.001
momentum = 0.9
epoch = 10



## === cell 14
base_model = MyResNet50()
base_state = copy.deepcopy(base_model.state_dict())

kfold = KFold(n_splits=5, shuffle=True, random_state=2021)

all_fold_probs = []
for fold, (train_index, val_index) in enumerate(kfold.split(train_data), start=1):
    print(f"\n==== Fold {fold}/5 ====")
    train, valid = train_data.iloc[train_index], train_data.iloc[val_index]
    trainset = Dog_Breed(train, transform=transforms_train)
    validset = Dog_Breed(valid, transform=transforms_test)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=32,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )
    valid_loader = torch.utils.data.DataLoader(
        validset,
        batch_size=32,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )

    model = MyResNet50()
    model.load_state_dict(base_state)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.net.fc.parameters(), lr=learn_rate, weight_decay=1e-5)

    fold_logits = train_model(
        model,
        train_loader,
        valid_loader,
        loss,
        optimizer,
        epoch,
        device,
        test_loader=test_loader_global,
    )
    fold_probs = F.softmax(fold_logits, dim=1).cpu().numpy()
    all_fold_probs.append(fold_probs)

mean_probs = np.mean(np.stack(all_fold_probs, axis=0), axis=0)  # [Ntest, 120]

result = pd.DataFrame(mean_probs, columns=labels)
result = pd.concat([test_data.reset_index(drop=True), result], axis=1)

result = result.reindex(columns=sample_sub.columns, fill_value=0.0)

probs = result.iloc[:, 1:].to_numpy(dtype=np.float64)
probs[np.isnan(probs)] = 0.0
row_sums = probs.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
probs = probs / row_sums
result.iloc[:, 1:] = probs

out_path = "/kaggle/working/dog_breed.csv"
result.to_csv(out_path, index=False)
print("Saved submission to", out_path, "with shape:", result.shape)



## === cell 15
df = pd.read_csv("/kaggle/working/dog_breed.csv")
probs = df.iloc[:, 1:].to_numpy(dtype=np.float64)
probs[np.isnan(probs)] = 0.0
row_sums = probs.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
probs = probs / row_sums
df.iloc[:, 1:] = probs
df = df.reindex(columns=sample_sub.columns, fill_value=0.0)
df.to_csv("/kaggle/working/dog_breed.csv", index=False)
print("Submission verified and rewritten to /kaggle/working/dog_breed.csv")
