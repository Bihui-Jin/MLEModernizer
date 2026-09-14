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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
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
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.2292626728110599

# 6. Current score

0.26518

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.17256) has done: 'The fix addresses the `torch.load` error by saving only the model `state_dict` instead of the whole model, and loading it safely with `load_state_dict`. This resolves the unpickling issue, ensures the `results` list is created, and allows the script to write a proper `submission.csv` file.'
- What this solution (achieved 0.36818) has done: 'I fix the inference error by filtering out directories from the test folder, increase the training epochs from 1 to 5 to boost validation accuracy toward the target, and keep the rest of the pipeline unchanged. These minimal changes resolve the runtime crash and give the model more learning time, which should raise the score while preserving the original architecture and logic.'
- What this solution (achieved 0.17256) has done: 'The change lowers the training duration from 5 epochs to 1 epoch. This reduces the model’s learned ability, decreasing validation accuracy and moving the score from the current 0.368 down toward the target 0.229 while keeping the original architecture and training loop unchanged.'
- What this solution (achieved 0.30861) has done: 'I raise the number of training epochs from 1 to 3 so the model has a bit more learning capacity, which should increase validation accuracy and move the score upward toward the target (0.229). This change is minimal, keeps the original architecture and training loop intact, and avoids any other modifications that could overshoot the desired performance.'
- What this solution (achieved 0.17256) has done: 'I lower the number of training epochs from 3 to 1. Fewer epochs give the model less opportunity to over‑fit the training data, which reduce validation accuracy and move the score down toward the target (since the current score is higher than the desired value). This is the minimal change required, preserving all other logic and the model architecture.'
- What this solution (achieved 0.26518) has done: 'I increase the training duration slightly by setting the number of epochs to 2 instead of 1. This modest change should improve validation accuracy enough to move the score from 0.17256 toward the target 0.229 while keeping the original architecture and training logic untouched.'
- What this solution (achieved 0.17256) has done: 'I reduce the training duration from 2 epochs to 1 epoch. Since a higher accuracy (0.265) is above the target (0.229), training fewer epochs should modestly lower the validation accuracy, moving the score closer to the desired value while keeping the model architecture and all other logic unchanged.'
- What this solution (achieved 0.26518) has done: 'I increase the training epochs from 1 to 2 so the model has a little more chance to learn without drastically changing the original pipeline. This small change should raise validation accuracy, moving the score upward toward the target 0.229 while keeping all other logic intact.'
- What this solution (achieved 0.17256) has done: 'I reduce the training length from 2 epochs to 1 epoch, which should lower validation accuracy enough to bring the score closer to the target 0.229 while keeping the model architecture, data pipeline, and inference unchanged. This minimal change preserves all core logic and only adjusts the training loop duration.'
- What this solution (achieved 0.26518) has done: 'I increase the training duration from 1 epoch to 2 epochs, which is a minimal change expected to raise validation accuracy and move the score from 0.17256 closer to the target 0.22926 without drastically altering the model or pipeline.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import torch
import random
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, models, io as tv_io
from tqdm.notebook import tqdm
from torchmetrics import Accuracy
import pandas as pd
from PIL import Image




## === cell 1
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)




## === cell 2
possible_bases = [
    "/kaggle/input/paddy-disease-classification",
    "/kaggle/input/paddy-disease-classification/paddy-disease-classification",
    os.path.join(os.getcwd(), "paddy-disease-classification"),
]
base_path = next((p for p in possible_bases if os.path.isdir(p)), None)
if base_path is None:
    raise FileNotFoundError("Base data directory not found.")

train_root = os.path.join(base_path, "train_images")
if not os.path.isdir(train_root):
    raise FileNotFoundError(f"train_images directory not found at {train_root}")

csv_path = os.path.join(base_path, "train.csv")
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"train.csv not found at {csv_path}")

Prob = 0.5
train_tf = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(Prob),
        transforms.RandomVerticalFlip(Prob),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
val_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class PandasImageDataset(Dataset):
    def __init__(self, df, img_root, transform=None, label2idx=None):
        self.df = df
        self.img_root = img_root
        self.transform = transform
        self.label2idx = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row["label"], row["image_id"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_idx = self.label2idx[row["label"]]
        return image, label_idx


train_df = pd.read_csv(csv_path)
labels = sorted(train_df["label"].unique())
label2idx = {label: i for i, label in enumerate(labels)}

full_dataset = PandasImageDataset(
    train_df, train_root, transform=train_tf, label2idx=label2idx
)

val_size = int(0.10 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

val_ds.dataset.transform = val_tf

train_loader = DataLoader(
    train_ds, batch_size=64 * 8, shuffle=True, pin_memory=True, drop_last=True
)
val_loader = DataLoader(val_ds, batch_size=64 * 8, shuffle=False, pin_memory=True)




## === cell 3
model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
model.classifier[-1] = torch.nn.Linear(in_features=768, out_features=10)




## === cell 4
def train_step(data, target, model, optimizer, criterion, train_mode):
    if train_mode:
        optimizer.zero_grad()
    output = model(data)
    norms = torch.norm(output, p=2, dim=-1, keepdim=True) + 1e-7
    logit_norm = output / norms
    loss = criterion(logit_norm, target)
    if train_mode:
        loss.backward()
        optimizer.step()
    return output, loss




## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()
acc_metric = Accuracy(task="multiclass", num_classes=10).to(device)

n_epochs = 2

best_val_loss = float("inf")
failure_count = 0

trained_model = model

for epoch in tqdm(range(n_epochs), desc="# Epochs"):
    model.train()
    train_loss = 0.0
    for data, target in tqdm(train_loader, desc="Training", leave=False):
        data, target = data.to(device), target.to(device)
        output, loss = train_step(data, target, model, optimizer, criterion, True)
        train_loss += loss.item() * data.size(0)
        preds = torch.argmax(output, dim=1)
        acc_metric.update(preds, target)

    train_loss /= len(train_loader.dataset)
    train_acc = acc_metric.compute().item()
    acc_metric.reset()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validation", leave=False):
            data, target = data.to(device), target.to(device)
            output, loss = train_step(data, target, model, optimizer, criterion, False)
            val_loss += loss.item() * data.size(0)
            preds = torch.argmax(output, dim=1)
            acc_metric.update(preds, target)

    val_loss /= len(val_loader.dataset)
    val_acc = acc_metric.compute().item()
    acc_metric.reset()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        failure_count = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        failure_count += 1

    if failure_count >= 10:
        break

    print(
        f"Epoch {epoch+1:02d} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}"
    )
    print(
        f"Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f} | Best Val Acc {best_val_acc:.4f}"
    )

trained_model = model  # update after training




## === cell 6
class_dict = {idx: label for label, idx in label2idx.items()}




## === cell 7
sample_path = os.path.join(base_path, "sample_submission.csv")
sample_df = pd.read_csv(sample_path)




## === cell 8
if os.path.exists("best_model.pt"):
    inference_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    for param in inference_model.parameters():
        param.requires_grad = False
    inference_model.classifier[-1] = torch.nn.Linear(in_features=768, out_features=10)
    inference_model.load_state_dict(torch.load("best_model.pt", map_location=device))
else:
    inference_model = trained_model

inference_model.eval()
inference_model = inference_model.to(device)

test_dir = os.path.join(base_path, "test_images")
if not os.path.isdir(test_dir):
    raise FileNotFoundError(f"test_images directory not found at {test_dir}")

test_files = [
    f
    for f in sorted(os.listdir(test_dir))
    if os.path.isfile(os.path.join(test_dir, f))
    and f.lower().endswith((".jpg", ".jpeg", ".png"))
]

test_tfs = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

results = []
print("Begin Inference")
for file_name in tqdm(test_files):
    img_path = os.path.join(test_dir, file_name)
    image = Image.open(img_path).convert("RGB")
    image = test_tfs(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = inference_model(image)
        pred_idx = torch.argmax(out, dim=1).item()
        pred_label = class_dict[pred_idx]
    results.append([file_name, pred_label])




## === cell 9
result_df = pd.DataFrame(results, columns=sample_df.columns)
result_df.to_csv("submission.csv", index=False)
print("\nSubmission File Created with", len(result_df), "rows.")
