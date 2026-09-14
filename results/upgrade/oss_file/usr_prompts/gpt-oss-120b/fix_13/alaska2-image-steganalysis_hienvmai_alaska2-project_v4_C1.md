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
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.13

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Target score

0.6894357242271227

# 6. Current score

0.59164

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.59581) has done: 'I cut the training data size (CFG.subset_size) to a much smaller number and increase batch sizes and data‑loader workers so each epoch runs far quicker while keeping the same model, loss, training loop, and evaluation logic. These hyper‑parameter changes reduce the amount of I/O and GPU work but do not alter the core algorithm, so the results remain comparable and the script finishes well within the 600‑second limit.'
- What this solution (achieved 0.60031) has done: 'I modestly increase the training data size and the number of epochs (still using the same model, loss, and training loop) so the model can learn a bit more without altering the core logic. These small adjustments are expected to raise the weighted AUC toward the target while keeping the script fast enough to finish within the time limit.'
- What this solution (achieved 0.61038) has done: 'The changes keep the same model, loss, and training loops but add mixed‑precision (AMP) for faster GPU computation, enable persistent workers and prefetching in the DataLoaders, and turn off the deterministic flag that slows cuDNN. The tqdm progress bars are run in disabled mode to avoid extra overhead while still preserving the loop logic.'
- What this solution (achieved 0.5945) has done: 'The changes reduce expensive repeated disk I/O by pre‑loading all images into memory once, shrink the training subset to keep total work within the 600 s limit, and keep the same model, loss, and training loops so the algorithmic behavior is unchanged. Pre‑loading is safe because the subset fits in memory, and the smaller but still representative subset preserves the training logic while cutting runtime.'
- What this solution (achieved 0.5949) has done: 'I increase the training subset slightly and run a few more epochs to give the model more data and learning time, and I add a simple test‑time augmentation (horizontal flip) whose predictions are averaged with the original ones. These minimal changes keep the core architecture, loss and training loop unchanged while providing a modest boost in weighted AUC, moving the score toward the target.'
- What this solution (achieved 0.59164) has done: 'The main slowdown comes from loading and caching ≈ 10 k full‑size images in memory, which explodes RAM usage and initialization time. By shrinking the training subset (still a representative sample) and loading images lazily inside `__getitem__` instead of pre‑caching, we keep the same model, transforms, and training loop while dramatically reducing I/O and memory overhead. Increasing the batch size and a few worker adjustments further cut the total runtime without altering any core algorithmic logic.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from tqdm.notebook import tqdm
import timm
import warnings

warnings.filterwarnings("ignore")




## === cell 1
class CFG:
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = "/kaggle/input/alaska2-image-steganalysis/"
    subset_size = 800  # was 2500
    image_size = 512

    model_name = "tf_efficientnet_b2_ns"
    num_classes = 1

    n_folds = 5
    fold_to_train = 0
    epochs = 12
    train_batch_size = 128
    valid_batch_size = 128
    weight_decay = 1e-6

    lr = 1e-4
    T_0 = 5
    eta_min = 1e-6

    checkpoint_save_path = "/kaggle/working/latest_checkpoint.pth"
    best_model_save_path = f"/kaggle/working/best_model_fold_{fold_to_train}.pth"

    checkpoint_load_path = None

    use_amp = True  # mixed‑precision retained


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(CFG.seed)




## === cell 2
def alaska_weighted_auc(y_true, y_pred):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    if len(fpr) < 2:
        return 0.5
    areas = np.array([0.0] * len(weights))
    for i, lower in enumerate(tpr_thresholds[:-1]):
        upper = tpr_thresholds[i + 1]
        mask = (tpr >= lower) & (tpr < upper)
        if np.any(mask):
            idx = np.where(mask)[0]
            start_idx, end_idx = idx[0], idx[-1]
            tpr_slice = np.concatenate([[lower], tpr[start_idx : end_idx + 1], [upper]])
            fpr_slice = np.concatenate(
                [
                    [np.interp(lower, tpr, fpr)],
                    fpr[start_idx : end_idx + 1],
                    [np.interp(upper, tpr, fpr)],
                ]
            )
            tpr_slice, uniq = np.unique(tpr_slice, return_index=True)
            fpr_slice = fpr_slice[uniq]
            areas[i] = np.trapz(fpr_slice, tpr_slice)
    return np.sum(areas * weights) / np.sum(weights)




## === cell 3
image_folders = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]
subset_files = []

print(f"Creating a subset of {CFG.subset_size} images from each folder...")
for folder in image_folders:
    folder_path = os.path.join(CFG.data_path, folder)
    all_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".jpg")
    ]
    random.shuffle(all_files)
    subset_files.extend(all_files[: CFG.subset_size])
    print(f"  - Took {len(all_files[:CFG.subset_size])} images from {folder}")

subset_files = list(subset_files)  # ensure plain Python list
df = pd.DataFrame({"image_path": subset_files})
df["label"] = df["image_path"].apply(lambda x: 0 if "Cover" in x else 1)
df["image_id"] = df["image_path"].apply(os.path.basename)

skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
df["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
    df.loc[val_idx, "fold"] = fold

print("\nSubset dataset distribution:")
print(df["label"].value_counts())
print("\nFold distribution:")
print(df.groupby("fold")["label"].value_counts())




## === cell 4
def _basic_transform(image, img_size):
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image)


def get_transforms(mode="train"):
    def transform_fn(image):
        if mode == "train" and random.random() < 0.5:
            image = cv2.flip(image, 1)
        return _basic_transform(image, CFG.image_size)

    return transform_fn


class AlaskaDataset(Dataset):
    """
    Lazily loads images on each __getitem__ call.
    This removes the huge memory overhead of pre‑caching all images,
    while keeping the same transformation pipeline.
    """

    def __init__(self, df, transforms=None):
        self.df = df.reset_index(drop=True)
        self.labels = df["label"].values.astype(np.float32)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, "image_path"]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(img)
        else:
            img = _basic_transform(img, CFG.image_size)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return img, label




## === cell 5
def train_fn(loader, model, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    scaler = torch.cuda.amp.GradScaler() if CFG.use_amp else None
    pbar = tqdm(loader, desc="Training", disable=True)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=CFG.use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler:
            scheduler.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def eval_fn(loader, model, criterion, device):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []
    scaler = torch.cuda.amp.GradScaler() if CFG.use_amp else None
    pbar = tqdm(loader, desc="Evaluating", disable=True)
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            with torch.cuda.amp.autocast(enabled=CFG.use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item()
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    val_loss = running_loss / len(loader)
    score = alaska_weighted_auc(all_labels, all_preds)
    return val_loss, score




## === cell 6
import torch

torch.cuda.empty_cache()
import gc

gc.collect()




## === cell 7
def run_training(fold):
    print(f"========== Starting Training for Fold {fold} ==========")

    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)

    train_dataset = AlaskaDataset(train_df, transforms=get_transforms("train"))
    valid_dataset = AlaskaDataset(valid_df, transforms=get_transforms("valid"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = timm.create_model(
        CFG.model_name, pretrained=True, num_classes=CFG.num_classes
    ).to(CFG.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=CFG.T_0 * len(train_loader), eta_min=CFG.eta_min
    )
    criterion = nn.BCEWithLogitsLoss()

    best_score = 0.0
    start_epoch = 0

    if CFG.checkpoint_load_path and os.path.exists(CFG.checkpoint_load_path):
        print(f"Resuming from {CFG.checkpoint_load_path}")
        checkpoint = torch.load(
            CFG.checkpoint_load_path, map_location=CFG.device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint.get("best_score", 0.0)
        print(f"Resumed at epoch {start_epoch}, best score {best_score:.4f}")
    else:
        print("No checkpoint found, training from scratch.")

    for epoch in range(start_epoch, CFG.epochs):
        print(f"\n--- Epoch {epoch+1}/{CFG.epochs} ---")
        train_loss = train_fn(
            train_loader, model, criterion, optimizer, scheduler, CFG.device
        )
        val_loss, val_score = eval_fn(valid_loader, model, criterion, CFG.device)
        print(
            f"Epoch {epoch+1} -> Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, Valid Weighted AUC: {val_score:.4f}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), CFG.best_model_save_path)
            print(f"New best model saved with score {best_score:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_score": best_score,
            },
            CFG.checkpoint_save_path,
        )

    print(f"\n========== Finished Fold {fold}. Best Score: {best_score:.4f} ==========")


run_training(CFG.fold_to_train)




## === cell 8
def create_submission():
    print("\nStarting inference on the test set...")
    test_folder = os.path.join(CFG.data_path, "Test")
    test_image_ids = [f for f in os.listdir(test_folder) if f.lower().endswith(".jpg")]

    test_df = pd.DataFrame({"image_id": test_image_ids})
    test_df["image_path"] = test_df["image_id"].apply(
        lambda x: os.path.join(test_folder, x)
    )
    test_df["label"] = 0  # dummy

    test_dataset = AlaskaDataset(test_df, transforms=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = timm.create_model(
        CFG.model_name, pretrained=False, num_classes=CFG.num_classes
    )
    model.load_state_dict(torch.load(CFG.best_model_save_path, map_location=CFG.device))
    model.to(CFG.device)
    model.eval()

    predictions = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Predicting", disable=True)
        for images, _ in pbar:
            images = images.to(CFG.device)
            with torch.cuda.amp.autocast(enabled=CFG.use_amp):
                out_orig = model(images)
                out_hflip = model(torch.flip(images, dims=[3]))
                out_vflip = model(torch.flip(images, dims=[2]))
            probs_orig = torch.sigmoid(out_orig).cpu().numpy()
            probs_hflip = torch.sigmoid(out_hflip).cpu().numpy()
            probs_vflip = torch.sigmoid(out_vflip).cpu().numpy()
            avg_probs = (probs_orig + probs_hflip + probs_vflip) / 3.0
            predictions.extend(avg_probs.flatten())

    submission_df = pd.DataFrame({"Id": test_image_ids, "Label": predictions})
    submission_df = submission_df.sort_values("Id")
    submission_df.to_csv("submission.csv", index=False)
    print("\nSubmission file 'submission.csv' created successfully!")
    print(submission_df.head())


create_submission()
