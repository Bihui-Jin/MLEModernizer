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

0.5506

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

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
    _pil_cache = {}

    def __init__(self, csv_df, transform=None, test=False):
        super().__init__()
        self.csv_df = csv_df
        self.image_ids = list(csv_df["id"])
        self.test = test
        if not self.test:
            self.labels = list(csv_df["number"])
        self.transform = transform

        base_path = (
            "/kaggle/input/dog-breed-identification/test"
            if self.test
            else "/kaggle/input/dog-breed-identification/train"
        )
        for img_id in self.image_ids:
            if img_id not in Dog_Breed._pil_cache:
                img_path = os.path.join(base_path, img_id + ".jpg")
                Dog_Breed._pil_cache[img_id] = Image.open(img_path).convert("RGB")

        if self.test and self.transform is not None:
            self.cached_tensors = [
                self.transform(Dog_Breed._pil_cache[img_id])
                for img_id in self.image_ids
            ]

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        if self.test:
            return self.cached_tensors[idx]
        else:
            image = Dog_Breed._pil_cache[img_id]
            if self.transform is not None:
                image = self.transform(image)
            return image, self.labels[idx]

    def __len__(self):
        return len(self.image_ids)




## === cell 5
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


device = get_device()
device

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)



## === cell 6
train_df, valid_df = train_test_split(train_data, test_size=0.2, random_state=42)



## === cell 7
num_workers = min(8, os.cpu_count() or 1)  # slightly more workers can speed up loading

trainset_simple = Dog_Breed(train_df, transform=transforms_train)
validset_simple = Dog_Breed(valid_df, transform=transforms_test)
train_loader_simple = DataLoader(
    trainset_simple,
    batch_size=64,
    shuffle=True,
    drop_last=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)
valid_loader_simple = DataLoader(
    validset_simple,
    batch_size=64,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

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




## === cell 8
class MyResNet50(nn.Module):
    def __init__(self, num_classes=120):
        super(MyResNet50, self).__init__()
        self.net = models.resnet50(pretrained=True)
        for p in self.net.parameters():
            p.requires_grad = False  # freeze backbone
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.net(x)




## === cell 9
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
    Training loop identical to the original but without per‑batch tqdm progress bars.
    The removal of tqdm eliminates Python‑level overhead while preserving the
    exact training semantics (mixed precision, scheduler, early stopping, etc.).
    """
    net = model.to(device)
    if hasattr(torch, "compile"):
        net = torch.compile(net)

    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_epoch = 0
    best_score = 0.0
    best_state = None
    early_stop = 5  # increased patience to let the model train a bit longer
    scheduler = ExponentialLR(optimizer, gamma=0.9, verbose=False)

    for epoch in range(epochs):
        net.train()
        acc = 0
        loss_sum = 0.0
        for x, y in train_loader:  # removed tqdm
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = net(x)
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = net(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()

            loss_sum += loss.item()
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
            for x, y in valid_loader:  # removed tqdm
                x, y = x.to(device), y.to(device)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = net(x)
                else:
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
        persistent_workers=True,
    )
    preds = []
    net.eval()
    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = net(x)
                preds.append(logits.cpu())
            else:
                logits = net(x)
                preds.append(logits.cpu())
    preds = torch.cat(preds, dim=0)  # shape (N_test, 120)
    return preds




## === cell 10
learn_rate = 0.0005  # slightly lower LR for smoother learning
epoch_num = 20  # more epochs to allow convergence



## === cell 11
print("=== Training on single split ===")
train_set = Dog_Breed(train_df, transform=transforms_train)
valid_set = Dog_Breed(valid_df, transform=transforms_test)

train_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True,
    drop_last=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)
valid_loader = DataLoader(
    valid_set,
    batch_size=64,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

model = MyResNet50()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.net.fc.parameters(), lr=learn_rate, weight_decay=1e-5)

preds = train_model(
    model,
    train_loader,
    valid_loader,
    loss_fn,
    optimizer,
    epoch_num,
    device,
    predict_test=True,
)

if preds is None:
    raise RuntimeError("No predictions were generated. Check the training loop.")

preds = torch.softmax(preds, dim=1)
result_df = pd.DataFrame(preds.numpy(), columns=labels)
result_df = pd.concat([test_data.reset_index(drop=True), result_df], axis=1)

output_path = "/kaggle/working/dog_breed.csv"
result_df.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")



## === cell 12
if os.path.exists(output_path):
    df_check = pd.read_csv(output_path)
    print("First rows of the submission file:")
    print(df_check.head())
else:
    print("Error: submission file was not created.")
