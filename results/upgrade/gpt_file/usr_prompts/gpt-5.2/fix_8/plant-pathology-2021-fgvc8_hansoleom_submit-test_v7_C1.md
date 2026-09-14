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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.7137396121883658

# 6. Current score

0.44788

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.45226) has done: 'I fix the crash by ensuring the test image list contains only actual image files (the provided directory includes a nested `test_images/` folder that was being globbed and passed to PIL). I also make the train/valid label-index mapping consistent by building it once from the full `train.csv` (instead of separately per split), which avoids label mismatch and improves correctness without changing the model or training loop. Finally, I generate predictions in the exact order of `sample_submission.csv` to guarantee alignment and produce a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.47107) has done: 'Your current pipeline is training a single-label classifier (CrossEntropy + one class per image), but this competition’s metric/format is multi-label (space-delimited set of diseases). Without changing your model/training core, the smallest legitimate move toward the target is to (1) map each unique space-delimited label *set* to a class (what you already effectively do), then (2) at inference time output the *top-K* most likely label-sets instead of only top-1, joining them with spaces to match the submission semantics and improve mean F1. I’m also fixing a subtle but important label parsing issue: `labels` strings can contain spaces, so splitting CSV lines by comma is unsafe; we parse train/valid via pandas and keep the rest identical. Finally, I keep ordering aligned to `sample_submission.csv` exactly as you already do to avoid accidental score loss from misalignment.'
- What this solution (achieved 0.47585) has done: 'Your score gap to the target is large (0.47107 vs 0.71374), so we need a legitimate improvement while keeping your core single-label EfficientNet+CE training unchanged. The biggest low-risk gain here is to start from pretrained EfficientNet-B4 ImageNet weights (same architecture/training loop, just better initialization), which typically boosts performance substantially for leaf-disease images. I also fix your seeding setup to be truly deterministic (you currently set `deterministic=True` but also `benchmark=True`, which conflicts) to avoid score regressions, and I add a tiny, metric-aligned inference tweak: use sigmoid over logits with an absolute probability threshold per label-set plus a safe fallback to top-1 (still predicting label-sets, but avoids forcing 3 label-sets when uncertain). The submission ordering/format remain aligned to `sample_submission.csv` exactly, and it still write `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.62982) has done: 'Your current inference is using `sigmoid` + independent thresholding, but the model was trained with `CrossEntropyLoss` on mutually exclusive “label-set classes”, so probabilities should come from `softmax`; this mismatch can significantly depress F1. I switch inference to `softmax` and keep your existing top‑K/threshold idea, interpreting `PROB_THRESH` as a softmax confidence threshold, with the same safe fallback to top‑1. I also make the train/valid split stratified by the label-set class to reduce validation noise and improve the learned decision boundaries without changing the model, loss, or training loop. Everything else (paths, architecture, training procedure, and submission ordering/format) stays the same and still writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.44788) has done: 'To move your 0.62982 score closer to the 0.71374 target with minimal risk, I keep the same EfficientNet-B4 + CrossEntropy training and only adjust inference to better match the metric’s “set of labels” semantics. Specifically, I (1) stop concatenating multiple *label-sets* (which creates invalid composite strings and hurts F1) and instead predict exactly one label-set per image, and (2) apply a small, validation-calibrated confidence fallback: if the top-1 softmax is below a threshold, use a safer default (the most frequent class in training) rather than forcing a wrong rare label-set. This keeps evaluation semantics intact (still choosing among the same mutually-exclusive label-set classes) while improving macro F1 stability. The submission is still aligned to `sample_submission.csv` order and written to `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
from PIL import Image
from tqdm import tqdm
import copy
import pandas as pd
from torchvision import transforms, models
from torch import optim
import torch
import os
import random
from glob import glob

DATA_ROOT = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 1
transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(640),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

transform_valid = transforms.Compose(
    [
        transforms.Resize(640),
        transforms.CenterCrop(640),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)



## === cell 2
train_df_full = pd.read_csv(TRAIN_CSV_PATH)

all_labels = sorted(train_df_full["labels"].astype(str).unique().tolist())
label2idx = {lab: i for i, lab in enumerate(all_labels)}
idx2label = {i: lab for lab, i in label2idx.items()}
NUM_CLASSES = len(label2idx)
print("NUM_CLASSES:", NUM_CLASSES)

df_tmp = train_df_full.copy()
df_tmp["y"] = df_tmp["labels"].astype(str).map(label2idx)

most_freq_lab = df_tmp["labels"].astype(str).value_counts().idxmax()
MOST_FREQ_IDX = label2idx[most_freq_lab]
print("Most frequent label-set:", most_freq_lab, "-> idx", MOST_FREQ_IDX)

parts = []
for y, g in df_tmp.groupby("y", sort=False):
    parts.append(g.sample(frac=1.0, random_state=42))
df_shuf = (
    pd.concat(parts, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
)

valid_frac = 0.1
train_idx = []
valid_idx = []
for y, g in df_shuf.groupby("y", sort=False):
    n = len(g)
    n_valid = max(1, int(round(n * valid_frac))) if n > 1 else 0
    valid_idx.extend(g.index[:n_valid].tolist())
    train_idx.extend(g.index[n_valid:].tolist())

train_df = df_shuf.loc[train_idx, ["image", "labels"]].reset_index(drop=True)
valid_df = df_shuf.loc[valid_idx, ["image", "labels"]].reset_index(drop=True)

print("train/valid sizes:", len(train_df), len(valid_df))




## === cell 3
class torchvision_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_root, df, label2idx, transforms=None):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.label2idx = label2idx
        self.transform = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df.loc[idx, "image"]
        label_name = str(self.df.loc[idx, "labels"]).strip()

        img = Image.open(os.path.join(self.image_root, image_name)).convert("RGB")
        x = self.transform(img) if self.transform else img
        return x, self.label2idx[label_name]




## === cell 4
train_dataset = torchvision_Dataset(TRAIN_IMG_DIR, train_df, label2idx, transform_train)
valid_dataset = torchvision_Dataset(TRAIN_IMG_DIR, valid_df, label2idx, transform_valid)




## === cell 5
def _default_num_workers():
    return min(4, os.cpu_count() or 2)


NUM_WORKERS = _default_num_workers()
PIN_MEMORY = torch.cuda.is_available()

train_dataloaders = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
)
valid_dataloaders = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
)



## === cell 6
pass



## === cell 7
model_ft = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
in_features = model_ft.classifier[1].in_features
model_ft.classifier[1] = torch.nn.Linear(in_features, NUM_CLASSES)
model_ft.to(device)



## === cell 8
from torch.optim.lr_scheduler import _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [
            base_lr
            * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)




## === cell 9
criterion = torch.nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_ft, 30, eta_min=0, last_epoch=-1
)
exp_lr_scheduler = GradualWarmupScheduler(
    optimizer_ft, multiplier=100, total_epoch=3, after_scheduler=cosine_scheduler
)




## === cell 10
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    os.makedirs("outputs", exist_ok=True)

    for epoch in range(num_epochs):
        running_loss = 0.0
        train_corrects = 0
        train_data_cnt = 0

        model.train()
        train_progress_bar = tqdm(train_dataloaders)
        for inputs, labels in train_progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data).item()
            train_data_cnt += inputs.size(0)
            train_progress_bar.set_description(
                f" Epoch[{epoch+1}/{num_epochs}] train : runing_Loss {running_loss / train_data_cnt:.5f}, "
                f"train_acc {train_corrects / train_data_cnt:.5f}"
            )

        scheduler.step()

        valid_corrects = 0
        valid_data_cnt = 0

        model.eval()
        valid_progress_bar = tqdm(valid_dataloaders)
        for inputs, labels in valid_progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

            valid_corrects += torch.sum(preds == labels.data).item()
            valid_data_cnt += inputs.size(0)
            valid_progress_bar.set_description(
                f" Epoch[{epoch+1}/{num_epochs}] valid : valid_acc {valid_corrects / valid_data_cnt:.5f}"
            )

        epoch_acc = valid_corrects / len(valid_dataset)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f"outputs/{best_epoch}.pth")
            print(f"best epoch : {best_epoch} (acc={best_acc:.5f})")

    return best_model_wts




## === cell 11
pass



## === cell 12
ckpt_path = "/kaggle/input/best-model/28.pth"
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    model_ft.load_state_dict(state)
    model_ft.to(device)
else:
    best_wts = train_model(
        model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=1
    )
    model_ft.load_state_dict(best_wts)
    model_ft.to(device)




## === cell 13
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.transform(img) if self.transform else img
        return x, os.path.basename(p)


valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
all_paths = sorted(glob(os.path.join(TEST_IMG_DIR, "*")))
img_paths = [
    p
    for p in all_paths
    if os.path.isfile(p) and os.path.splitext(p.lower())[1] in valid_ext
]

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
needed = sample_sub["image"].tolist()
path_map = {os.path.basename(p): p for p in img_paths}
missing = [n for n in needed if n not in path_map]
if len(missing) > 0:
    print(
        f"Warning: {len(missing)} images from sample_submission not found in TEST_IMG_DIR. Example:",
        missing[:3],
    )

img_paths_ordered = [path_map[n] for n in needed if n in path_map]

test_dataset = TestDataset(img_paths_ordered, transform_valid)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
)

CONF_THRESH = 0.55

submit = []
model_ft.eval()

with torch.no_grad():
    for inputs, names in tqdm(test_loader):
        inputs = inputs.to(device, non_blocking=True)
        logits = model_ft(inputs)
        probs = torch.softmax(logits, dim=1)  # (B, C)

        top1_probs, top1_idx = torch.max(probs, dim=1)  # (B,), (B,)

        top1_probs = top1_probs.detach().cpu().tolist()
        top1_idx = top1_idx.detach().cpu().tolist()

        for n, p1, i1 in zip(names, top1_probs, top1_idx):
            if float(p1) < CONF_THRESH:
                chosen = idx2label[int(MOST_FREQ_IDX)]
            else:
                chosen = idx2label[int(i1)]
            submit.append([n, chosen])

submission = pd.DataFrame(submit, columns=["image", "labels"])
submission = sample_sub[["image"]].merge(submission, on="image", how="left")
submission["labels"] = submission["labels"].fillna(idx2label[int(MOST_FREQ_IDX)])

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv with shape:", submission.shape)
print(submission.head())
