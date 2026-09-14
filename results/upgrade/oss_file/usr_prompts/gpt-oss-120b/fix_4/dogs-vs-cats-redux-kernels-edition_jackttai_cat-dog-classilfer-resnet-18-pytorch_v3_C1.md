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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
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

# 4. Data file paths

```
/
    kaggle/
        data/
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
        input/
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
            test/
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
            train/
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
        working/
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

# 5. Target score

5.73755

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import zipfile, os, shutil


def unzip_to_dir(zip_path, extract_to="."):
    """
    Extract zip_path only if it exists.
    This prevents errors when the dataset has already been extracted
    (as is the case in the provided environment).
    """
    if os.path.isfile(zip_path):
        print(f"Unzipping {zip_path} → {extract_to}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
    else:
        print(
            f"Zip file {zip_path} not found – skipping extraction (folder likely already exists)."
        )


unzip_to_dir("../input/dogs-vs-cats-redux-kernels-edition/train.zip")
unzip_to_dir("../input/dogs-vs-cats-redux-kernels-edition/test.zip")




## === cell 1
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

plt.style.use("ggplot")
import pandas as pd
import random, time, os, numpy as np
from PIL import Image



## === cell 2
train_dir = "./dogs-vs-cats-redux-kernels-edition/train"
test_dir = "./dogs-vs-cats-redux-kernels-edition/test"


def build_train_df(root):
    rows = []
    for label_name, label_val in [("cat", 0), ("dog", 1)]:
        class_dir = os.path.join(root, label_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            rows.append(
                {"filename": os.path.join(class_dir, fname), "label": label_val}
            )
    return pd.DataFrame(rows)


train_df_full = build_train_df(train_dir)

if train_df_full.empty or "label" not in train_df_full.columns:
    raise RuntimeError(
        "Training data not found or incorrectly loaded. Check train_dir path."
    )

train_df, val_df = train_test_split(
    train_df_full, test_size=0.04, random_state=42, stratify=train_df_full["label"]
)

test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
test_df = pd.DataFrame({"filename": [os.path.join(test_dir, f) for f in test_files]})

print(
    f"Training images: {len(train_df)}, Validation images: {len(val_df)}, Test images: {len(test_df)}"
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/931015139.py in <cell line: 0>()
     19 
     20 if train_df_full.empty or "label" not in train_df_full.columns:
---> 21     raise RuntimeError(
     22         "Training data not found or incorrectly loaded. Check train_dir path."
     23     )

RuntimeError: Training data not found or incorrectly loaded. Check train_dir path.

## === cell 3
def show_6_photos(df):
    sample_df = df.sample(6)
    for path in sample_df["filename"]:
        img = plt.imread(path)
        plt.figure(figsize=(3, 3))
        plt.imshow(img)
        plt.axis("off")
        plt.show()




## === cell 4
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}




## === cell 5
class image_set(Dataset):
    def __init__(self, dataframe, transform=None, test=False):
        self.df = dataframe
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.test:
            return image
        else:
            label = torch.tensor(self.df.iloc[idx, 1], dtype=torch.float32).unsqueeze(0)
            return image, label




## === cell 6
batch_size = 32
num_workers = 0  # safe for most Kaggle environments

train_set = image_set(train_df, transform=data_transforms["train"])
val_set = image_set(val_df, transform=data_transforms["val"])
test_set = image_set(test_df, transform=data_transforms["val"], test=True)

train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
val_loader = DataLoader(
    val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
)
test_loader = DataLoader(
    test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1589364244.py in <cell line: 0>()
      2 num_workers = 0  # safe for most Kaggle environments
      3 
----> 4 train_set = image_set(train_df, transform=data_transforms["train"])
      5 val_set = image_set(val_df, transform=data_transforms["val"])
      6 test_set = image_set(test_df, transform=data_transforms["val"], test=True)

NameError: name 'train_df' is not defined

## === cell 7
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## === cell 8
def train_model(model, cost_function, optimizer, num_epochs=5):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    train_acc_metric = Accuracy(task="binary").to(device)
    val_acc_metric = Accuracy(task="binary").to(device)

    for epoch in range(num_epochs):
        print("-" * 20)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        model.train()
        epoch_train_losses = []
        train_acc_metric.reset()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)  # y shape (B,1)
            optimizer.zero_grad()
            outputs = model(x)  # (B,1) after sigmoid
            loss = cost_function(outputs, y)
            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())
            train_acc_metric.update(outputs.detach(), y.int())

        model.eval()
        epoch_val_losses = []
        val_acc_metric.reset()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = cost_function(outputs, y)
                epoch_val_losses.append(loss.item())
                val_acc_metric.update(outputs, y.int())

        train_losses.append(np.mean(epoch_train_losses))
        val_losses.append(np.mean(epoch_val_losses))

        train_acc = train_acc_metric.compute().item()
        val_acc = val_acc_metric.compute().item()
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"train loss: {train_losses[-1]:.4f}, acc: {train_acc:.4f} | "
            f"val loss: {val_losses[-1]:.4f}, acc: {val_acc:.4f}"
        )

    print("Training finished.")
    return train_losses, val_losses, train_accs, val_accs




## === cell 9
class net(nn.Module):
    def __init__(self, resnet):
        super(net, self).__init__()
        self.resnet = resnet
        self.linear1 = nn.Linear(1000, 512)
        self.linear2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.resnet(x))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x




## === cell 10
base_resnet = models.resnet18(pretrained=True)
for p in base_resnet.parameters():
    p.requires_grad = False

model_final = net(base_resnet).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model_final.parameters()), lr=0.009
)

EPOCHS = 5  # reduced for quicker runs; can be increased later




## === cell 11
train_losses, val_losses, train_acc, val_acc = train_model(
    model=model_final, cost_function=criterion, optimizer=optimizer, num_epochs=EPOCHS
)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1875817111.py in <cell line: 0>()
----> 1 train_losses, val_losses, train_acc, val_acc = train_model(
      2     model=model_final, cost_function=criterion, optimizer=optimizer, num_epochs=EPOCHS
      3 )
      4 
      5 

/tmp/ipykernel_11/4280523159.py in train_model(model, cost_function, optimizer, num_epochs)
     15         train_acc_metric.reset()
     16 
---> 17         for x, y in train_loader:
     18             x, y = x.to(device), y.to(device)  # y shape (B,1)
     19             optimizer.zero_grad()

NameError: name 'train_loader' is not defined

## === cell 12
def plot_result(train_losses, val_losses, train_acc, val_acc):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))
    ax1.plot(train_losses, label="train loss")
    ax1.plot(val_losses, label="val loss")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax2.plot(train_acc, label="train acc", color="brown")
    ax2.plot(val_acc, label="val acc", color="pink")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    plt.show()


plot_result(train_losses, val_losses, train_acc, val_acc)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2816249075.py in <cell line: 0>()
     12 
     13 
---> 14 plot_result(train_losses, val_losses, train_acc, val_acc)
     15 
     16 

NameError: name 'train_losses' is not defined

## === cell 13
def predict_on_loader(loader, model):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            preds = model(x)  # (B,1)
            all_preds.append(preds.cpu())
    return torch.cat(all_preds).squeeze().numpy()  # shape (N,)




## === cell 14
predictions = predict_on_loader(test_loader, model_final)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2302048121.py in <cell line: 0>()
----> 1 predictions = predict_on_loader(test_loader, model_final)
      2 
      3 

NameError: name 'test_loader' is not defined

## === cell 15
import pathlib

possible_paths = [
    pathlib.Path("../input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv"),
    pathlib.Path("input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv"),
    pathlib.Path("../working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv"),
]

submission_path = next((p for p in possible_paths if p.is_file()), None)
if submission_path is None:
    raise FileNotFoundError("sample_submission.csv not found in any expected location.")

submission = pd.read_csv(submission_path)
assert len(predictions) == len(submission), "Prediction size mismatch"
submission["label"] = predictions
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv with shape:", submission.shape)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4192856418.py in <cell line: 0>()
     13 
     14 submission = pd.read_csv(submission_path)
---> 15 assert len(predictions) == len(submission), "Prediction size mismatch"
     16 submission["label"] = predictions
     17 submission.to_csv("submission.csv", index=False)

NameError: name 'predictions' is not defined

## === cell 16
torch.save(model_final.state_dict(), "model_state_dict.pt")
print("Model checkpoint saved.")
