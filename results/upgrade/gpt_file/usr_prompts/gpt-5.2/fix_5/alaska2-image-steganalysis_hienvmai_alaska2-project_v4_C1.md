# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
from tqdm.auto import tqdm
import timm
import warnings
import gc

warnings.filterwarnings("ignore")

try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)
except Exception:
    pass


class CFG:
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = "/kaggle/input/alaska2-image-steganalysis/"
    subset_size = 75000
    image_size = 512

    model_name = "tf_efficientnet_b2_ns"
    num_classes = 1

    n_folds = 5
    fold_to_train = 0
    epochs = 1
    train_batch_size = 16
    valid_batch_size = 32
    weight_decay = 1e-6

    lr = 1e-4
    T_0 = 5
    eta_min = 1e-6

    checkpoint_save_path = "/kaggle/working/latest_checkpoint.pth"
    best_model_save_path = f"/kaggle/working/best_model_fold_{fold_to_train}.pth"

    checkpoint_load_path = None


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


set_seed(CFG.seed)




## === cell 1
def alaska_weighted_auc(y_true, y_pred):
    """
    Calculates the weighted AUC score for the ALASKA2 competition.
    """
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    if len(fpr) < 2:
        return 0.5

    areas = np.array([0.0] * len(weights))
    for i, lower in enumerate(tpr_thresholds[:-1]):
        upper = tpr_thresholds[i + 1]
        mask = (tpr >= lower) & (tpr < upper)
        if np.any(mask):
            mask_indices = np.where(mask)[0]
            start_idx, end_idx = mask_indices[0], mask_indices[-1]
            tpr_slice = np.concatenate([[lower], tpr[start_idx : end_idx + 1], [upper]])
            fpr_slice = np.concatenate(
                [
                    [np.interp(lower, tpr, fpr)],
                    fpr[start_idx : end_idx + 1],
                    [np.interp(upper, tpr, fpr)],
                ]
            )
            tpr_slice, unique_indices = np.unique(tpr_slice, return_index=True)
            fpr_slice = fpr_slice[unique_indices]
            areas[i] = np.trapz(fpr_slice, tpr_slice)

    return np.sum(areas * weights) / np.sum(weights)




## === cell 2
image_folders = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]

per_folder = CFG.subset_size // len(image_folders)
remainder = CFG.subset_size - per_folder * len(image_folders)

subset_files = []
print(
    f"Creating a subset of {CFG.subset_size} total images ({per_folder} per folder + remainder {remainder})..."
)

rng = np.random.RandomState(CFG.seed)

for i, folder in enumerate(image_folders):
    folder_path = os.path.join(CFG.data_path, folder)
    files = [
        e.name
        for e in os.scandir(folder_path)
        if e.is_file() and e.name.endswith(".jpg")
    ]

    n_take = per_folder + (1 if i < remainder else 0)
    if n_take > len(files):
        n_take = len(files)

    idx = rng.choice(len(files), size=n_take, replace=False)
    taken = [os.path.join(folder_path, files[j]) for j in idx]
    subset_files.extend(taken)
    print(f"  - Took {len(taken)} images from {folder}")

df = pd.DataFrame({"image_path": np.array(subset_files, dtype=object)})

df["label"] = (~df["image_path"].astype(str).str.contains(r"/Cover/")).astype(np.int64)
df["image_id"] = df["image_path"].map(os.path.basename)

skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
df["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(df, df["label"])):
    df.loc[val_idx, "fold"] = fold

print("\nSubset dataset distribution:")
print(df["label"].value_counts())
print("\nFold distribution:")
print(df.groupby("fold")["label"].value_counts())



## === cell 3
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_transforms(data_type="train"):
    def _transform(image):
        if data_type == "train":
            if random.random() < 0.5:
                image = np.ascontiguousarray(np.flip(image, axis=1))  # horizontal
            if random.random() < 0.5:
                image = np.ascontiguousarray(np.flip(image, axis=0))  # vertical

        image = cv2.resize(
            image, (CFG.image_size, CFG.image_size), interpolation=cv2.INTER_AREA
        )
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(image).float()

    return _transform


class AlaskaDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df.reset_index(drop=True)
        self.image_paths = self.df["image_path"].astype(str).tolist()
        self.labels = self.df["label"].astype(np.float32).tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label




## === cell 4
_TQDM_MININTERVAL = 0.5


def train_fn(loader, model, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Training", leave=False, mininterval=_TQDM_MININTERVAL)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        running_loss += loss.item()
        pbar.set_postfix(
            loss=float(loss.item()), lr=float(optimizer.param_groups[0]["lr"])
        )

    return running_loss / max(1, len(loader))


def eval_fn(loader, model, criterion, device):
    model.eval()
    running_loss = 0.0

    n = len(loader.dataset)
    all_preds = np.empty(n, dtype=np.float32)
    all_labels = np.empty(n, dtype=np.float32)
    offset = 0

    with torch.no_grad():
        pbar = tqdm(
            loader, desc="Evaluating", leave=False, mininterval=_TQDM_MININTERVAL
        )
        for images, labels in pbar:
            bsz = images.size(0)
            images = images.to(device, non_blocking=True)
            labels_t = labels.to(device, non_blocking=True).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels_t)
            running_loss += loss.item()

            probs = torch.sigmoid(outputs).detach().cpu().numpy().ravel()
            labs = labels_t.detach().cpu().numpy().ravel()
            all_preds[offset : offset + bsz] = probs
            all_labels[offset : offset + bsz] = labs
            offset += bsz

    val_loss = running_loss / max(1, len(loader))
    score = alaska_weighted_auc(all_labels, all_preds)
    return val_loss, score




## === cell 5
torch.cuda.empty_cache()
gc.collect()




## === cell 6
def run_training(fold):
    print(f"========== Starting Training for Fold {fold} ==========")

    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)

    train_dataset = AlaskaDataset(train_df, transforms=get_transforms("train"))
    valid_dataset = AlaskaDataset(valid_df, transforms=get_transforms("valid"))

    ncpu = os.cpu_count() or 4
    num_workers = min(4, max(2, ncpu // 4))

    g = torch.Generator()
    g.manual_seed(CFG.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        generator=g,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )

    model = timm.create_model(
        CFG.model_name, pretrained=True, num_classes=CFG.num_classes
    ).to(CFG.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=CFG.T_0 * max(1, len(train_loader)), eta_min=CFG.eta_min
    )
    criterion = nn.BCEWithLogitsLoss()

    start_epoch = 0
    best_score = 0.0
    if CFG.checkpoint_load_path and os.path.exists(CFG.checkpoint_load_path):
        print(f"Resuming training from checkpoint: {CFG.checkpoint_load_path}")
        checkpoint = torch.load(
            CFG.checkpoint_load_path, map_location=CFG.device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_score = float(checkpoint.get("best_score", 0.0))
        print(
            f"Loaded model from epoch {start_epoch-1} with best score: {best_score:.4f}"
        )
    else:
        print("No checkpoint found, starting training from scratch.")

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
            print(
                f"Validation score improved! ({best_score:.4f} -> {val_score:.4f}). Saving best model..."
            )
            best_score = val_score
            torch.save(model.state_dict(), CFG.best_model_save_path)

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }
        torch.save(checkpoint, CFG.checkpoint_save_path)
        print(f"Epoch {epoch+1} state saved to checkpoint: {CFG.checkpoint_save_path}")

    print(
        f"\n========== Finished Training for Fold {fold}. Best Score: {best_score:.4f} =========="
    )


run_training(CFG.fold_to_train)




## === cell 7
def create_submission():
    print("\nStarting inference on the test set...")

    test_folder = os.path.join(CFG.data_path, "Test")

    test_image_ids = sorted(
        [
            e.name
            for e in os.scandir(test_folder)
            if e.is_file() and e.name.endswith(".jpg")
        ]
    )

    test_paths = [os.path.join(test_folder, x) for x in test_image_ids]
    test_df = pd.DataFrame({"image_id": np.array(test_image_ids, dtype=object)})
    test_df["image_path"] = np.array(test_paths, dtype=object)
    test_df["label"] = 0  # dummy

    test_dataset = AlaskaDataset(test_df, transforms=get_transforms("test"))

    ncpu = os.cpu_count() or 4
    num_workers = min(4, max(2, ncpu // 4))

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )

    model = timm.create_model(
        CFG.model_name, pretrained=False, num_classes=CFG.num_classes
    )
    model.load_state_dict(
        torch.load(CFG.best_model_save_path, map_location="cpu", weights_only=False)
    )
    model.to(CFG.device)
    model.eval()

    predictions = np.empty(len(test_dataset), dtype=np.float32)
    offset = 0

    with torch.no_grad():
        pbar = tqdm(
            test_loader, desc="Predicting", leave=False, mininterval=_TQDM_MININTERVAL
        )
        for images, _ in pbar:
            bsz = images.size(0)
            images = images.to(CFG.device, non_blocking=True)
            outputs = model(images)
            probs = torch.sigmoid(outputs).detach().cpu().numpy().ravel()
            predictions[offset : offset + bsz] = probs
            offset += bsz

    submission_df = (
        pd.DataFrame({"Id": test_image_ids, "Label": predictions.tolist()})
        .sort_values("Id")
        .reset_index(drop=True)
    )
    submission_path = "/kaggle/working/submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"\nSubmission file created successfully at: {submission_path}")
    print(submission_df.head())
    return submission_df


_ = create_submission()
