# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
timm==1.0.19
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
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os, random, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import timm
from matplotlib import pyplot as plt

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)



## === cell 1
path = "../input/cassava-leaf-disease-classification"



## === cell 2
df = pd.read_csv(os.path.join(path, "train.csv"))



## === cell 3
df["path"] = df["image_id"].map(lambda x: os.path.join(path, "train_images", x))
df = df.drop(columns=["image_id"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # deterministic shuffle



## === cell 4
from sklearn import model_selection

train_df, valid_df = model_selection.train_test_split(
    df, test_size=0.2, random_state=42, stratify=df.label.values
)



## === cell 5
image_size = 512
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class make_mask_image:
    def __init__(self, p, mask_size=50):
        self.p = p
        self.mask_size = mask_size

    def __call__(self, image):
        if random.random() < self.p:
            draw = ImageDraw.Draw(image)
            w, h = image.size
            for _ in range(10):
                x = random.randrange(0, w - self.mask_size)
                y = random.randrange(0, h - self.mask_size)
                draw.rectangle(
                    (x, y, x + self.mask_size, y + self.mask_size),
                    fill=(0, 0, 0),
                    outline=(0, 0, 0),
                )
        return image


train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(image_size),
        make_mask_image(p=0.5, mask_size=50),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

valid_transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)




## === cell 6
class CassavaDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with open(row["path"], "rb") as f:
            img = Image.open(f).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row["label"])




## === cell 7
train_dataset = CassavaDataset(train_df, train_transform)



## === cell 8
with open(os.path.join(path, "label_num_to_disease_map.json"), "r") as f:
    label_to_name = json.load(f)


class Unnormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


unnorm = Unnormalize(mean, std)



## === cell 9
num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 16
epochs = 3



## === cell 10
resNet = timm.create_model("resnet50", pretrained=True)
resNet.fc = nn.Linear(resNet.fc.in_features, num_classes)
resNet = resNet.to(device)

ef_model = timm.create_model("tf_efficientnet_b2_ns", pretrained=True)
ef_model.classifier = nn.Linear(ef_model.classifier.in_features, num_classes)
ef_model = ef_model.to(device)



## === cell 11
ef_optimizer = torch.optim.AdamW(ef_model.parameters(), lr=1e-4, weight_decay=1e-4)
ef_scheduler = torch.optim.lr_scheduler.StepLR(ef_optimizer, step_size=2, gamma=0.1)

resNet_optimizer = torch.optim.AdamW(resNet.parameters(), lr=1e-4, weight_decay=1e-4)
resNet_scheduler = torch.optim.lr_scheduler.StepLR(
    resNet_optimizer, step_size=2, gamma=0.1
)

criterion = nn.CrossEntropyLoss()




## === cell 12
def calc_correction(model, df):
    model.eval()
    loader = DataLoader(
        CassavaDataset(df, valid_transform),
        batch_size=32,
        shuffle=False,
        num_workers=min(8, os.cpu_count()),
        pin_memory=True,
        persistent_workers=True,
    )
    correct = 0
    total = 0
    pred_counts = torch.zeros(num_classes, dtype=torch.long, device="cpu")
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
            pred_counts += torch.bincount(preds.cpu(), minlength=num_classes)
    return correct / total, pred_counts.tolist()




## === cell 13
def plot_losses(title, train_losses, valid_losses):
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, label="train loss")
    plt.plot(epochs_range, valid_losses, label="valid loss")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()




## === cell 14
def train_model(
    model, dataset, batch_size, optimizer, criterion, scheduler, epochs, save_path
):
    best_state = None
    best_loss = float("inf")
    train_losses, valid_losses = [], []

    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_set,
        batch_size,
        shuffle=True,
        num_workers=min(8, os.cpu_count()),
        pin_memory=True,
        persistent_workers=True,
    )
    valid_loader = DataLoader(
        val_set,
        batch_size,
        shuffle=False,
        num_workers=min(8, os.cpu_count()),
        pin_memory=True,
        persistent_workers=True,
    )

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        tr_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * imgs.size(0)
        tr_loss /= len(train_loader.dataset)
        train_losses.append(tr_loss)

        model.eval()
        val_loss = 0.0
        accs = []
        with torch.no_grad():
            for imgs, lbls in valid_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                loss = criterion(out, lbls)
                val_loss += loss.item() * imgs.size(0)
                accs.append((out.argmax(1) == lbls).float().mean().item())
        val_loss /= len(valid_loader.dataset)
        valid_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()

        scheduler.step()
        epoch_time = time.time() - start
        print(
            f"Epoch {epoch} | Time {epoch_time:.2f}s | TrainLoss {tr_loss:.4f} | "
            f"ValidLoss {val_loss:.4f} | Acc {np.mean(accs):.4f}"
        )

    torch.save(best_state, save_path)
    model.load_state_dict(best_state)
    return model, train_losses, valid_losses




## === cell 15
def train_models(resNet, ef_model):
    resNet, tr_l, val_l = train_model(
        resNet,
        train_dataset,
        batch_size,
        resNet_optimizer,
        criterion,
        resNet_scheduler,
        epochs,
        "./res_model.pth",
    )
    print("ResNet validation accuracy:", calc_correction(resNet, valid_df)[0])
    plot_losses("ResNet losses", tr_l, val_l)

    ef_model, tr_l, val_l = train_model(
        ef_model,
        train_dataset,
        batch_size,
        ef_optimizer,
        criterion,
        ef_scheduler,
        epochs,
        "./ef_model.pth",
    )
    print("EffNet validation accuracy:", calc_correction(ef_model, valid_df)[0])
    plot_losses("EffNet losses", tr_l, val_l)




## === cell 16
train_models(resNet, ef_model)



## === cell 17
resNet.load_state_dict(torch.load("./res_model.pth", map_location=device))
ef_model.load_state_dict(torch.load("./ef_model.pth", map_location=device))




## === cell 18
class CassavaClassifier(nn.Module):
    def __init__(self, model_a, model_b):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b

    def forward(self, x):
        return 0.5 * self.model_a(x) + 0.5 * self.model_b(x)

    def test(self, x, rate):
        return rate * self.model_a(x) + (1 - rate) * self.model_b(x)




## === cell 19
classifier = CassavaClassifier(resNet, ef_model).to(device)




## === cell 20
def test_rate():
    for rate in range(1, 10):
        loader = DataLoader(
            CassavaDataset(valid_df, valid_transform),
            batch_size=32,
            shuffle=False,
            num_workers=min(8, os.cpu_count()),
            pin_memory=True,
            persistent_workers=True,
        )
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                preds = classifier.test(imgs, rate / 10.0).argmax(1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)
        print(f"rate: {rate/10:.1f} accuracy: {correct/total:.4f}")




## === cell 21
test_path = os.path.join(path, "test_images")
image_ids = sorted(
    [f for f in os.listdir(test_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
)
image_paths = [os.path.join(test_path, f) for f in image_ids]




## === cell 22
class TestDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


test_loader = DataLoader(
    TestDataset(image_paths, valid_transform),
    batch_size=32,
    shuffle=False,
    num_workers=min(8, os.cpu_count()),
    pin_memory=True,
    persistent_workers=True,
)



## === cell 23
preds = []
classifier.eval()
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = classifier(batch)
        preds.extend(out.argmax(1).cpu().tolist())



## === cell 24
submission = pd.DataFrame({"image_id": image_ids, "label": preds})
submission.to_csv("submission.csv", index=False)
print("Submission written, shape:", submission.shape)
