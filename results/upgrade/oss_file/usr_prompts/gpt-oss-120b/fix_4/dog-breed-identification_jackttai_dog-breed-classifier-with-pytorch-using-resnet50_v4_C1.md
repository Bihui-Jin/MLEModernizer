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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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

0.50039

# 6. Current score

0.95902

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63153) has done: 'I fixed the stratified split error by using the full training set and ensuring the validation split contains at least as many samples as there are classes (120). This resolves the `ValueError` and defines `train_loader`, `val_loader`, and `batch_size` so the rest of the pipeline can run and produce a proper `submission.csv`.'
- What this solution (achieved 0.95902) has done: 'I unfreeze the last block of the pretrained ResNet‑50 so the model can fine‑tune useful features, and I train a few more epochs (8 instead of 5). These minimal adjustments should improve validation loss and move the log‑loss closer to the target without altering the core architecture or training logic.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader




## === cell 1
comp_df = pd.read_csv("../input/dog-breed-identification/labels.csv")
test_df = pd.read_csv("../input/dog-breed-identification/sample_submission.csv")
print("Training set rows:", comp_df.shape[0], "Test set rows:", test_df.shape[0])




## === cell 2
le = LabelEncoder()
comp_df["label"] = le.fit_transform(comp_df["breed"])

dict_df = comp_df[["label", "breed"]].drop_duplicates().set_index("label")
index_to_breed = dict_df["breed"].to_dict()




## === cell 3
train_dir = "../input/dog-breed-identification/train"
comp_df["id"] = comp_df["id"].apply(lambda x: os.path.join(train_dir, f"{x}.jpg"))
comp_df = comp_df.drop(columns=["breed"])




## === cell 4
def show_images(df, img_num):
    sample = df.sample(img_num)
    paths = sample["id"].tolist()
    for path in paths:
        plt.figure(figsize=(3, 3))
        img = plt.imread(path)
        plt.imshow(img)
        plt.axis("off")
        plt.show()




## === cell 5
class img_dataset(Dataset):
    def __init__(self, dataframe, transform=None, test=False):
        self.df = dataframe
        self.transform = transform
        self.test = test

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.test:
            return img
        else:
            label = self.df.iloc[idx, 1]
            return img, label

    def __len__(self):
        return len(self.df)




## === cell 6
train_transformer = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_transformer = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)




## === cell 7
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## === cell 8
training_samples = comp_df.shape[0]  # use all rows
sample_df = comp_df.sample(training_samples, random_state=42)

val_frac = max(0.1, 120 / training_samples)  # at least 10% or enough rows
x_train, x_val = train_test_split(
    sample_df, test_size=val_frac, random_state=42, stratify=sample_df["label"]
)

train_set = img_dataset(x_train, transform=train_transformer)
val_set = img_dataset(x_val, transform=val_transformer)

batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

print("Training subset:", x_train.shape[0], "Validation subset:", x_val.shape[0])




## === cell 9
class Net(torch.nn.Module):
    def __init__(self, base_model, num_classes):
        super(Net, self).__init__()
        self.base = torch.nn.Sequential(
            *list(base_model.children())[:-1]
        )  # remove original FC
        self.fc1 = torch.nn.Linear(base_model.fc.in_features, 512)
        self.out = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.out(x)
        return x


resnet = torchvision.models.resnet50(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False
for param in resnet.layer4.parameters():
    param.requires_grad = True

model_final = Net(resnet, num_classes=len(le.classes_)).to(device)




## === cell 10
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model_final.parameters()), lr=3e-4
)




## === cell 11
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def train_model(model, epochs):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, device)

        train_losses.append(tr_loss)
        train_accuracies.append(tr_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch}/{epochs} - "
            f"train loss: {tr_loss:.4f}, acc: {tr_acc:.4f} | "
            f"val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )
    return train_losses, train_accuracies, val_losses, val_accuracies




## === cell 12
EPOCHS = 8  # increased modestly to allow better fine‑tuning
train_losses, train_acc, val_losses, val_acc = train_model(model_final, EPOCHS)




## === cell 13
plt.style.use("ggplot")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(range(1, EPOCHS + 1), train_losses, label="train loss")
ax1.plot(range(1, EPOCHS + 1), val_losses, label="val loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(range(1, EPOCHS + 1), train_acc, label="train acc")
ax2.plot(range(1, EPOCHS + 1), val_acc, label="val acc")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()
plt.show()




## === cell 14
test_dir = "../input/dog-breed-identification/test"
test_df_paths = test_df[["id"]].copy()
test_df_paths["id"] = test_df_paths["id"].apply(
    lambda x: os.path.join(test_dir, f"{x}.jpg")
)
test_set = img_dataset(test_df_paths, transform=val_transformer, test=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)




## === cell 15
model_final.eval()
all_preds = []
with torch.no_grad():
    for imgs in test_loader:
        imgs = imgs.to(device)
        outputs = model_final(imgs)
        all_preds.append(outputs.cpu())
pred_tensor = torch.cat(all_preds, dim=0)




## === cell 16
probabilities = F.softmax(pred_tensor, dim=1).numpy()

answer_ids = pd.read_csv("../input/dog-breed-identification/sample_submission.csv")[
    "id"
].tolist()
submission_df = pd.DataFrame(probabilities, index=answer_ids)
submission_df.columns = [index_to_breed[i] for i in range(len(index_to_breed))]
submission_df.index.name = "id"
submission_path = "submission.csv"
submission_df.to_csv(submission_path)
print(f"Submission file written to {submission_path}")
