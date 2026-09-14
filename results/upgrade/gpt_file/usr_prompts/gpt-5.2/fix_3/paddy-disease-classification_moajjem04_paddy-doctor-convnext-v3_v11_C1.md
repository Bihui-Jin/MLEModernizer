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

0.36856

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing input paths by building the train/val datasets directly from `train.csv` and `train_images/` (instead of a non-existent pre-split folder), and I replace the missing external `.pt` weight load with a torchvision ConvNeXt-Tiny initialized from built-in ImageNet weights. I keep your same transforms, training loop structure, and “normalize logits then CrossEntropyLoss” semantics, but make it run reliably on CPU/GPU and with correct torchmetrics usage for multiclass. I also fix inference bugs (device handling, undefined `i` vs `ii`, and model eval/no_grad) and ensure the submission has exactly the sample rows and correct column names/order. This should run end-to-end in the Kaggle environment and produce a valid `submission.csv`.'
- What this solution (achieved 0.36856) has done: 'I fix the DataLoader crash by ensuring the training/validation transforms receive a PIL image (your current pipeline feeds NumPy arrays into transforms that expect PIL). I also fix the inference crash by filtering `test_images/` to include only actual `.jpg` files (the directory contains a nested folder name that `os.listdir` returns, which `imread` can’t read). Finally, I keep your model/training logic intact but unfreeze the classifier parameters so the model can actually learn (previously everything was frozen, so accuracy would stay near chance), which should raise the score toward your low target while still being a minimal change.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision
from tqdm.notebook import tqdm
from torchmetrics import Accuracy
import pandas as pd
from skimage import io
from PIL import Image




## === cell 1
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(42)



## === cell 2
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



## === cell 3
DATA_DIR = "../input/paddy-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_ROOT = os.path.join(DATA_DIR, "train_images")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_DIR = os.path.join(DATA_DIR, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

labels = sorted(train_df["label"].unique().tolist())
label2idx = {lab: i for i, lab in enumerate(labels)}
idx2label = {i: lab for lab, i in label2idx.items()}

from sklearn.model_selection import train_test_split

tr_df, va_df = train_test_split(
    train_df, test_size=0.15, random_state=42, stratify=train_df["label"]
)
tr_df = tr_df.reset_index(drop=True)
va_df = va_df.reset_index(drop=True)


class PaddyCSVDataset(Dataset):
    def __init__(self, df, img_root, transform=None, label2idx=None):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.transform = transform
        self.label2idx = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        label = row["label"]

        img_path = os.path.join(self.img_root, label, image_id)
        img = io.imread(img_path)

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]

        img = Image.fromarray(img.astype(np.uint8), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        target = self.label2idx[label] if self.label2idx is not None else -1
        return img, torch.tensor(target, dtype=torch.long)


train_ds = PaddyCSVDataset(
    tr_df, TRAIN_IMG_ROOT, transform=train_tf, label2idx=label2idx
)
val_ds = PaddyCSVDataset(va_df, TRAIN_IMG_ROOT, transform=val_tf, label2idx=label2idx)

print("Num classes:", len(labels))
print("Train size:", len(train_ds), "Val size:", len(val_ds))



## === cell 4
train_ds.class_to_idx = label2idx
train_ds.class_to_idx



## === cell 5
val_ds.class_to_idx = label2idx
val_ds.class_to_idx



## === cell 6
BATCH_SIZE = 64
NUM_WORKERS = 2

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    num_workers=NUM_WORKERS,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
    num_workers=NUM_WORKERS,
)



## === cell 7
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
model = convnext_tiny(weights=weights)



## === cell 8
for param in model.parameters():
    param.requires_grad = False



## === cell 9
model.classifier



## === cell 10
model.classifier[-1] = torch.nn.Linear(in_features=768, out_features=10)  # tiny



## === cell 11
for param in model.classifier.parameters():
    param.requires_grad = True

for param in model.parameters():
    if param.requires_grad:
        print(param.shape)




## === cell 12
def train(data, target, model, optimizer, criterion, TRAIN):

    if TRAIN:
        optimizer.zero_grad()

    output = model(data)

    norms = torch.norm(output, p=2, dim=-1, keepdim=True) + 1e-7
    logit_norm = torch.div(output, norms)
    loss = criterion(logit_norm, target)

    if TRAIN:
        loss.backward()
        optimizer.step()

    return output, loss




## === cell 13
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

n_epochs = 1

acc = Accuracy(task="multiclass", num_classes=10).to(device)
print(f"\nTraining Model on device={device}")
best_val_ = 1000
failure_count = 0

for epoch in tqdm(range(n_epochs), desc="# Epochs", position=0):
    train_loss = 0
    val_loss = 0

    train_acc = 0
    val_acc = 0

    model.train()

    for i, (data, target) in enumerate(
        tqdm(train_loader, desc="Training", leave=True, position=1)
    ):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output, loss = train(data, target, model, optimizer, criterion, TRAIN=True)

        train_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, dim=1)
        acc(pred, target)

    train_loss = train_loss / len(train_loader.dataset)
    train_acc = acc.compute().item()
    acc.reset()

    with torch.no_grad():
        model.eval()

        for i, (data, target) in enumerate(
            tqdm(val_loader, desc="Validation", leave=True, position=2)
        ):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output, loss = train(data, target, model, optimizer, criterion, TRAIN=False)
            val_loss += loss.item() * data.size(0)
            _, pred = torch.max(output, dim=1)
            acc(pred, target)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = acc.compute().item()
        acc.reset()

    if val_loss < best_val_:
        best_val_ = val_loss
        best_acc = val_acc
        failure_count = 0
        torch.save(model, "best_model.pt")
    else:
        failure_count += 1

    if failure_count >= 10:
        break

    print(f"Epoch # {epoch+1:04d}")
    print(f"Train Loss: {train_loss: .4f},\t Val Loss: {val_loss: .4f}")
    print(f"Train Acc : {train_acc: .4f},\t Val Acc : {val_acc: .4f}")
    print(f"Best Val Loss : {best_val_: .4f},\t Val Acc : {best_acc: .4f}")
    print(f"Failure Count = {failure_count}")



## === cell 14
pass



## === cell 15
train_ds.class_to_idx



## === cell 16
class_dict = {v: k for k, v in train_ds.class_to_idx.items()}
class_dict



## === cell 17
sample_df = pd.read_csv("../input/paddy-disease-classification/sample_submission.csv")
sample_df.head()



## === cell 18
if os.path.exists("best_model.pt"):
    model = torch.load("best_model.pt", map_location=device)
model.to(device)
model.eval()



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
UnpicklingError                           Traceback (most recent call last)
/tmp/ipykernel_55/931523836.py in <cell line: 0>()
      1 if os.path.exists("best_model.pt"):
----> 2     model = torch.load("best_model.pt", map_location=device)
      3 model.to(device)
      4 model.eval()
      5 

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1468                         )
   1469                     except pickle.UnpicklingError as e:
-> 1470                         raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
   1471                 return _load(
   1472                     opened_zipfile,

UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint. 
	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
	WeightsUnpickler error: Unsupported global: GLOBAL torchvision.models.convnext.ConvNeXt was not an allowed global by default. Please use `torch.serialization.add_safe_globals([ConvNeXt])` or the `torch.serialization.safe_globals([ConvNeXt])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.

## === cell 19
test_dir = "../input/paddy-disease-classification/test_images"
file_list = sorted(
    [
        f
        for f in os.listdir(test_dir)
        if os.path.isfile(os.path.join(test_dir, f)) and f.lower().endswith(".jpg")
    ]
)
file_list[:5], len(file_list)



## === cell 20
test_tfs = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 21
result_df = sample_df.copy()

pred_map = {}

print("Begin Inference\n")
with torch.no_grad():
    for ii, file_name in enumerate(tqdm(file_list)):
        image = io.imread(os.path.join(test_dir, file_name))

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]

        image = test_tfs(image).unsqueeze(0).to(device, non_blocking=True)
        out = model(image)
        out = torch.argmax(out, dim=1).item()
        label = class_dict[out]
        pred_map[file_name] = label

        if (ii + 1) % 200 == 0:
            print(f"{ii+1:04d}/{len(file_list)}")
    else:
        print("All Test Images are processed")

result_df["label"] = result_df["image_id"].map(pred_map)

if result_df["label"].isna().any():
    fallback = train_df["label"].mode().iloc[0]
    result_df["label"] = result_df["label"].fillna(fallback)



## === cell 22
result_df = result_df[["image_id", "label"]]
assert len(result_df) == len(
    sample_df
), f"Expected {len(sample_df)} rows, got {len(result_df)}"
result_df.to_csv("submission.csv", index=False)
print("\nSubmission File Created! -> submission.csv")
print(result_df.head())
