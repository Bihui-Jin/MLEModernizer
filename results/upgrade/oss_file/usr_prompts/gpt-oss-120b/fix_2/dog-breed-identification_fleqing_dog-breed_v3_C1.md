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

Not yielded

# 7. Whether higher score is better

Lower is better

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
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms import ToPILImage
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
from PIL import Image



## === cell 1
train_data = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
labels = sorted(train_data["breed"].unique())
label_to_idx = {label: idx for idx, label in enumerate(labels)}
train_data["number"] = train_data["breed"].map(label_to_idx)
train_data.shape



## === cell 2
test_folder = "/kaggle/input/dog-breed-identification/dog-breed-identification/test"
test_ids = sorted(
    [fname[:-4] for fname in os.listdir(test_folder) if fname.endswith(".jpg")]
)
test_data = pd.DataFrame({"id": test_ids})
test_data.head()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2275941193.py in <cell line: 0>()
      2 test_folder = "/kaggle/input/dog-breed-identification/dog-breed-identification/test"
      3 test_ids = sorted(
----> 4     [fname[:-4] for fname in os.listdir(test_folder) if fname.endswith(".jpg")]
      5 )
      6 test_data = pd.DataFrame({"id": test_ids})

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/dog-breed-identification/dog-breed-identification/test'

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
    """
    PyTorch Dataset for dog breed images.
    Handles both training/validation and test mode.
    """

    def __init__(self, df, transform=None, test=False):
        self.df = df
        self.ids = list(df["id"])
        self.test = test
        self.transform = transform
        self.train_dir = (
            "/kaggle/input/dog-breed-identification/dog-breed-identification/train"
        )
        self.test_dir = (
            "/kaggle/input/dog-breed-identification/dog-breed-identification/test"
        )
        if not self.test:
            self.labels = list(df["number"])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        if self.test:
            img_path = os.path.join(self.test_dir, img_id + ".jpg")
        else:
            img_path = os.path.join(self.train_dir, img_id + ".jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.test:
            return image
        else:
            label = self.labels[idx]
            return image, label




## === cell 5
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


device = get_device()
print(f"Using device: {device}")



## === cell 6
train_df, valid_df = train_test_split(
    train_data, test_size=0.2, random_state=42, stratify=train_data["number"]
)
trainset = Dog_Breed(train_df, transform=transforms_train, test=False)
validset = Dog_Breed(valid_df, transform=transforms_test, test=False)



## === cell 7
train_loader = DataLoader(trainset, batch_size=32, shuffle=True, drop_last=False)
valid_loader = DataLoader(validset, batch_size=32, shuffle=False, drop_last=False)




## === cell 8
class MyResNet50(nn.Module):
    def __init__(self, num_classes=120):
        super(MyResNet50, self).__init__()
        self.net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for param in self.net.parameters():
            param.requires_grad = False
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.net(x)




## === cell 9
def train_model(model, train_loader, valid_loader, loss_fn, optimizer, epochs, device):
    net = model.to(device)
    best_epoch = 0
    best_score = -1.0  # higher validation accuracy is better
    best_state = None
    early_stop_rounds = 10
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-4)

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            optimizer.zero_grad()
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        scheduler.step()
        train_acc = correct / total
        print(
            f"epoch {epoch}: loss={epoch_loss/len(train_loader):.4f}, train_acc={train_acc:.4f}",
            end=" ",
        )

        net.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, targets in tqdm(valid_loader, desc=f"Epoch {epoch} [val]"):
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = net(imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_acc = val_correct / val_total
        print(f"val_acc={val_acc:.4f}")

        if val_acc > best_score:
            best_score = val_acc
            best_state = copy.deepcopy(net.state_dict())
            best_epoch = epoch
            print(">>> New best model saved")

        if epoch - best_epoch >= early_stop_rounds:
            print("Early stopping triggered")
            break

    net.load_state_dict(best_state)

    testset = Dog_Breed(test_data, transform=transforms_test, test=True)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False, drop_last=False)

    all_logits = []
    net.eval()
    with torch.no_grad():
        for imgs in tqdm(test_loader, desc="Test inference"):
            imgs = imgs.to(device)
            logits = net(imgs)
            all_logits.append(logits.cpu())

    logits_tensor = torch.cat(all_logits, dim=0)  # shape (num_test, num_classes)
    probs = F.softmax(logits_tensor, dim=1)  # convert to probabilities
    return probs




## === cell 10
learn_rate = 0.001
epochs = 30



## === cell 11
model = MyResNet50()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.net.fc.parameters(), lr=learn_rate, weight_decay=1e-5)

test_probabilities = train_model(
    model, train_loader, valid_loader, criterion, optimizer, epochs, device
)

submission = pd.DataFrame(test_probabilities.numpy(), columns=labels)
submission.insert(0, "id", test_data["id"])
submission_path = "/kaggle/working/dog_breed.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3644021508.py in <cell line: 0>()
      4 
      5 # Train and get probability predictions for the test set
----> 6 test_probabilities = train_model(
      7     model, train_loader, valid_loader, criterion, optimizer, epochs, device
      8 )

/tmp/ipykernel_55/1604340249.py in train_model(model, train_loader, valid_loader, loss_fn, optimizer, epochs, device)
     12         correct = 0
     13         total = 0
---> 14         for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
     15             optimizer.zero_grad()
     16             imgs = imgs.to(device)

/usr/local/lib/python3.11/dist-packages/tqdm/notebook.py in __iter__(self)
    248         try:
    249             it = super().__iter__()
--> 250             for obj in it:
    251                 # return super(tqdm...) will not catch exception
    252                 yield obj

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_55/695234035.py in __getitem__(self, idx)
     29         else:
     30             img_path = os.path.join(self.train_dir, img_id + ".jpg")
---> 31         image = Image.open(img_path).convert("RGB")
     32         if self.transform:
     33             image = self.transform(image)

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/dog-breed-identification/dog-breed-identification/train/401756f7b5bd79cce6e7c0c9f973d196.jpg'
