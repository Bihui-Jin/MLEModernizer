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

0.7650600184672209

# 6. Current score

0.68136

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.32676) has done: 'The timeout is driven mainly by slow input pipelines and per-image test inference (opening images one-by-one in Python) rather than the single training epoch itself. I keep the exact same model, transforms, loss, optimizer, and training loop semantics, but speed up data loading by enabling persistent workers, prefetching, and increasing worker count, plus using faster PIL decoding settings. For test-time, I preserve identical prediction semantics (same transform and argmax) while batching inference with a DataLoader instead of looping per file, which removes thousands of Python/PIL overhead calls and improves GPU utilization. I also add determinism/seed controls and safe cuDNN settings to keep results stable while improving throughput.'
- What this solution (achieved 0.53605) has done: 'Your score is far below the target, and the biggest issue is that you’re training a single-label classifier on a multi-label problem (the CSV `labels` are space-delimited multi-labels), then submitting exactly one label per image. To move the score toward the target with minimal core-logic disruption, I keep EfficientNet-B4, PyTorch training loop style, and standard transforms, but switch the head to multi-label with `BCEWithLogitsLoss` and multi-hot targets (this aligns training/inference with mean F1). I also add a tiny, safe validation-time threshold search (no early stopping, no extra epochs) to choose a global probability threshold that improves F1 on the held-out split, then apply it to test predictions and output space-delimited labels. These changes directly target the metric mismatch and should significantly lift your F1 toward the target while keeping the overall approach intact and still producing a valid `submission.csv`.'
- What this solution (achieved 0.68136) has done: 'I fix the immediate runtime error caused by the newer torchvision weights API by extracting the normalization stats safely (without assuming `weights.transforms()` has a `.transforms` attribute). This restore `transform_train/transform_valid`, which in turn unblocks dataset/dataloader creation and eliminates the downstream `NameError`s. I also make CSV parsing robust against potential commas/quotes by using `pandas.read_csv` (same semantics: 90/10 split after shuffling) so the multi-label targets are built reliably. These changes are correctness/stability focused and keep your multi-label EfficientNet+BCEWithLogitsLoss approach intact while ensuring a valid `/kaggle/working/submission.csv` is produced.'

# 9. Code solution

## === cell 0
from PIL import Image
from tqdm import tqdm
import copy
import pandas as pd
from torchvision import transforms, models
import torch
from torch import optim
import os
import random
from glob import glob

DATA_ROOT = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

assert os.path.exists(TRAIN_CSV_PATH), f"Missing train.csv at {TRAIN_CSV_PATH}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing train_images dir at {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing test_images dir at {TEST_IMG_DIR}"

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

Image.MAX_IMAGE_PIXELS = None
try:
    Image.warnings.simplefilter("ignore", Image.DecompressionBombWarning)
except Exception:
    pass



## === cell 1
weights = models.EfficientNet_B4_Weights.DEFAULT

mean = None
std = None
try:
    mean = getattr(weights, "meta", {}).get("mean", None)
    std = getattr(weights, "meta", {}).get("std", None)
except Exception:
    mean, std = None, None

if mean is None or std is None:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

norm = transforms.Normalize(mean, std)

transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        norm,
    ]
)
transform_valid = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        norm,
    ]
)



## === cell 2
train_df = pd.read_csv(TRAIN_CSV_PATH)
train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

cnt = int(len(train_df) * 0.9)
train_df_split = train_df.iloc[:cnt].copy()
valid_df_split = train_df.iloc[cnt:].copy()




## === cell 3
def parse_labels(label_str: str):
    label_str = str(label_str).strip()
    if not label_str:
        return []
    return label_str.split()


all_labels = set()
for lab_str in train_df_split["labels"].tolist():
    for lab in parse_labels(lab_str):
        all_labels.add(lab)

label = sorted(all_labels)
label2idx = {lab: idx for idx, lab in enumerate(label)}
label2idx




## === cell 4
class torchvision_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_root, df, label2idx, transforms=None):
        self.image_path = data_root
        self.label2idx = label2idx
        self.transform = transforms
        self.data = list(zip(df["image"].tolist(), df["labels"].tolist()))
        self.num_classes = len(self.label2idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, label_name = self.data[idx]
        img = Image.open(os.path.join(self.image_path, image_name)).convert("RGB")
        x = self.transform(img) if self.transform else transforms.ToTensor()(img)

        y = torch.zeros(self.num_classes, dtype=torch.float32)
        for lab in parse_labels(label_name):
            if lab in self.label2idx:
                y[self.label2idx[lab]] = 1.0
        return x, y




## === cell 5
train_dataset = torchvision_Dataset(
    TRAIN_IMG_DIR, train_df_split, label2idx, transform_train
)
valid_dataset = torchvision_Dataset(
    TRAIN_IMG_DIR, valid_df_split, label2idx, transform_valid
)




## === cell 6
def _num_workers():
    try:
        cpu = os.cpu_count() or 2
    except Exception:
        cpu = 2
    return max(2, min(8, cpu))


NW = _num_workers()

train_dataloaders = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=NW,
    pin_memory=True,
    persistent_workers=(NW > 0),
    prefetch_factor=4 if NW > 0 else None,
)
valid_dataloaders = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=NW,
    pin_memory=True,
    persistent_workers=(NW > 0),
    prefetch_factor=4 if NW > 0 else None,
)



## === cell 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



## === cell 8
num_classes = len(label2idx)

model_ft = models.efficientnet_b4(weights=weights)
model_ft.classifier[1] = torch.nn.Linear(
    model_ft.classifier[1].in_features, num_classes
)
model_ft.to(device)



## === cell 9
criterion = torch.nn.BCEWithLogitsLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_ft, 100, eta_min=0, last_epoch=-1
)




## === cell 10
def macro_f1_from_probs(probs: torch.Tensor, tgts: torch.Tensor, thr: float):
    preds = (probs >= thr).to(torch.int32)
    tgts_i = tgts.to(torch.int32)

    tp = (preds & tgts_i).sum(dim=0).to(torch.float32)
    fp = (preds & (1 - tgts_i)).sum(dim=0).to(torch.float32)
    fn = ((1 - preds) & tgts_i).sum(dim=0).to(torch.float32)
    f1_c = (2 * tp) / (2 * tp + fp + fn + 1e-12)
    return f1_c.mean().item()


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, default_thr=0.5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_score = -1.0

    os.makedirs("/kaggle/working/outputs", exist_ok=True)

    for epoch in range(num_epochs):
        running_loss = 0.0
        train_data_cnt = 0

        model.train()
        train_progress_bar = tqdm(train_dataloaders, leave=False)
        for inputs, targets in train_progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)  # logits
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            bs = inputs.size(0)
            running_loss += loss.item() * bs
            train_data_cnt += bs

            if train_data_cnt % (16 * 20) == 0:
                train_progress_bar.set_description(
                    f" Epoch[{epoch+1}/{num_epochs}] train : runing_Loss {running_loss / train_data_cnt:.5f}"
                )

        scheduler.step()

        model.eval()
        all_probs = []
        all_tgts = []
        with torch.no_grad():
            valid_progress_bar = tqdm(valid_dataloaders, leave=False)
            for inputs, targets in valid_progress_bar:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(inputs)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.detach().cpu())
                all_tgts.append(targets.detach().cpu())

        probs = torch.cat(all_probs, dim=0)
        tgts = torch.cat(all_tgts, dim=0)

        epoch_score = macro_f1_from_probs(probs, tgts, default_thr)

        if epoch_score > best_score:
            best_score = epoch_score
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, f"/kaggle/working/outputs/{best_epoch}.pth")
            print(f"best epoch : {best_epoch} (macro_f1~={best_score:.5f})")

    model.load_state_dict(best_model_wts)
    return best_model_wts




## === cell 11
_ = train_model(
    model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=1, default_thr=0.5
)



## === cell 12
out_ckpts = sorted(glob("/kaggle/working/outputs/*.pth"))
if len(out_ckpts) > 0:
    state = torch.load(out_ckpts[-1], map_location="cpu")
    model_ft.load_state_dict(state)
model_ft.to(device)



## === cell 13
idx2label = {idx: lab for lab, idx in label2idx.items()}




## === cell 14
def find_best_threshold(model, loader, device, thresholds=None):
    if thresholds is None:
        thresholds = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]

    model.eval()
    all_probs = []
    all_tgts = []
    with torch.no_grad():
        for xb, yb in tqdm(loader, leave=False):
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            probs = torch.sigmoid(logits).detach().cpu()
            all_probs.append(probs)
            all_tgts.append(yb.detach().cpu())

    probs = torch.cat(all_probs, dim=0)
    tgts = torch.cat(all_tgts, dim=0)

    best_thr = thresholds[0]
    best_score = -1.0
    for thr in thresholds:
        score = macro_f1_from_probs(probs, tgts, thr)
        if score > best_score:
            best_score = score
            best_thr = thr

    print(f"Chosen threshold={best_thr} with val macro_f1~={best_score:.5f}")
    return best_thr


best_thr = find_best_threshold(model_ft, valid_dataloaders, device)




## === cell 15
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.transform(img)
        return x, os.path.basename(p)


img_paths = sorted(glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
test_dataset = TestDataset(img_paths, transform_valid)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=NW,
    pin_memory=True,
    persistent_workers=(NW > 0),
    prefetch_factor=4 if NW > 0 else None,
)

submit = []
model_ft.eval()

with torch.no_grad():
    for xb, names in tqdm(test_loader):
        xb = xb.to(device, non_blocking=True)
        logits = model_ft(xb)
        probs = torch.sigmoid(logits).detach().cpu()
        pred_mask = probs >= best_thr

        for i, n in enumerate(names):
            labs = [idx2label[j] for j in torch.where(pred_mask[i])[0].tolist()]
            if len(labs) == 0:
                labs = ["healthy"]
            submit.append([n, " ".join(labs)])

submission = pd.DataFrame(submit, columns=["image", "labels"])

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
submission = sample_sub[["image"]].merge(submission, on="image", how="left")
submission["labels"] = submission["labels"].fillna("healthy")

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv with shape:", submission.shape)
print(submission.head())
