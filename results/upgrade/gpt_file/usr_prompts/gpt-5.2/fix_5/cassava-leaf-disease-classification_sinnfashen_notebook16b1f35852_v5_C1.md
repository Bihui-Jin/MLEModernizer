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

# 5. Target score

0.8472348141432456

# 6. Current score

0.69806

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.27354) has done: 'I fix the dataset/test file discovery so it only iterates real image files (the current code hits a nested `test_images/` directory and crashes). I also fix the `MyTestDataset.__len__` bug (it references a non-existent `self.labels`) so the dataset is usable end-to-end. Since your referenced pretrained model file path doesn’t exist in this environment, I keep the same VGG16 core model but replace the missing checkpoint load with a safe fallback (ImageNet pretrained weights) so inference can run and a valid `submission.csv` is always produced. Finally, I switch inference to `model.eval()` + `torch.no_grad()` and ensure the VGG16 classifier outputs 5 classes (required by this competition) before predicting labels.'
- What this solution (achieved 0.25187) has done: 'Your current score is low because the model is effectively an ImageNet VGG16 with a randomly initialized 5-class head, and inference is also using a non-standard resize (400×300) without the ImageNet resize/crop that the pretrained backbone expects. To move accuracy toward your target with minimal semantic change, I keep the same VGG16 core and single-pass inference, but switch preprocessing to the official VGG16 `weights.transforms()` (correct resize/crop/normalization) and run inference via a `DataLoader` for consistent batching (no algorithmic change, just more stable and less error-prone). I also add a simple test-time augmentation that preserves core logic (average of original + horizontal flip logits) to improve predictions without changing training (there is none). These changes should substantially increase accuracy from the random-head behavior while still producing the same required `submission.csv`.'
- What this solution (achieved 0.74253) has done: 'Your score is far below the target because you’re doing zero training: you’re using an ImageNet-pretrained VGG16 backbone with a randomly initialized 5-class classifier head, so predictions are essentially random with a small bias. To move accuracy toward the target while preserving the same core model (VGG16) and the same overall pipeline, I add a minimal train/validation fine-tuning step on `train.csv` using the provided `train_images/` folder, then run inference on test and write `submission.csv` as before. I keep preprocessing aligned to VGG16 ImageNet weights (same transforms you already use) and keep your test-time horizontal-flip averaging. I also keep runtime bounded by training only the classifier head for a few epochs with a small validation split.'
- What this solution (achieved 0.69806) has done: 'Your current score (0.74253) is below the target (0.84723), so we should make a small, legitimate improvement without changing the core VGG16 pipeline. The biggest low-risk gain while preserving your architecture/training loop is to train the classifier head with class-balanced loss (the cassava labels are imbalanced), which typically improves accuracy noticeably with minimal semantic change. I compute per-class weights from `train.csv` and pass them into `CrossEntropyLoss`, keeping the same optimizer, epochs, split, and TTA. I also keep everything deterministic and ensure the submission formatting stays identical.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

BASE1 = "/kaggle/input/cassava-leaf-disease-classification"
BASE2 = "../input/cassava-leaf-disease-classification"
BASE = BASE1 if os.path.exists(BASE1) else BASE2

TRAIN_CSV = os.path.join(BASE, "train.csv")
TRAIN_PATH = os.path.join(BASE, "train_images")
TEST_PATH = os.path.join(BASE, "test_images")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.isdir(TRAIN_PATH), f"Missing: {TRAIN_PATH}"
assert os.path.isdir(TEST_PATH), f"Missing: {TEST_PATH}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"



## === cell 1
valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
files = sorted(
    [
        f
        for f in os.listdir(TEST_PATH)
        if os.path.isfile(os.path.join(TEST_PATH, f)) and f.lower().endswith(valid_ext)
    ]
)
assert (
    len(files) > 0
), f"No image files found in {TEST_PATH}. Found: {os.listdir(TEST_PATH)[:10]}"

train_df = pd.read_csv(TRAIN_CSV)
assert set(train_df.columns) >= {"image_id", "label"}
train_df["image_id"] = train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(int)

train_df = train_df[
    train_df["image_id"].apply(lambda x: os.path.isfile(os.path.join(TRAIN_PATH, x)))
]
train_df = train_df.reset_index(drop=True)
assert len(train_df) > 0



## === cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "../input/cassavaleaffsolmodels/VGG16_1.mdl"

try:
    weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
except Exception:
    weights = None

if weights is not None:
    model = torchvision.models.vgg16(weights=weights)
    weights_preprocess = weights.transforms()
else:
    model = torchvision.models.vgg16()
    weights_preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

in_features = model.classifier[-1].in_features
model.classifier[-1] = torch.nn.Linear(in_features, 5)

loaded_ckpt = False
if os.path.exists(MODEL_PATH):
    state = torch.load(MODEL_PATH, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    loaded_ckpt = True

model = model.to(device)




## === cell 3
def default_loader(path):
    img_pil = Image.open(path).convert("RGB")
    img_tensor = weights_preprocess(img_pil)
    return img_tensor


class MyTrainDataset(Dataset):
    def __init__(self, df, root, loader=default_loader):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.loader = loader

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = os.path.join(self.root, row["image_id"])
        img = self.loader(img_path)
        y = int(row["label"])
        return img, y

    def __len__(self):
        return len(self.df)


class MyTestDataset(Dataset):
    def __init__(self, files, loader=default_loader):
        self.files = files
        self.loader = loader

    def __getitem__(self, index):
        img_path = os.path.join(TEST_PATH, self.files[index])
        img = self.loader(img_path)
        return self.files[index], img

    def __len__(self):
        return len(self.files)




## === cell 4
if not loaded_ckpt:
    for p in model.features.parameters():
        p.requires_grad = False

    n = len(train_df)
    idx = np.arange(n)
    rng = np.random.RandomState(0)
    rng.shuffle(idx)
    split = int(n * 0.9)
    tr_idx, va_idx = idx[:split], idx[split:]

    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)

    trainset = MyTrainDataset(tr_df, TRAIN_PATH)
    valset = MyTrainDataset(va_df, TRAIN_PATH)

    train_loader = DataLoader(
        trainset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        valset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    counts = (
        tr_df["label"]
        .value_counts()
        .reindex(range(5), fill_value=0)
        .sort_index()
        .values
    )
    counts = np.maximum(counts, 1)  # safety against zero-count (shouldn't happen)
    class_weights = (counts.sum() / counts).astype(np.float32)
    class_weights = (
        class_weights / class_weights.mean()
    )  # keep average weight ~1 (stable LR behavior)
    class_weights_t = torch.tensor(class_weights, device=device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=2e-4, weight_decay=1e-4
    )

    model.train()
    for epoch in range(3):  # keep epochs unchanged for minimal change
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = torch.argmax(model(xb), dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        _val_acc = correct / max(1, total)
        model.train()

model.eval()



## === cell 5
testset = MyTestDataset(files)

loader = DataLoader(
    testset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

labels = []
image_ids = []

with torch.no_grad():
    for batch_ids, batch_imgs in loader:
        batch_imgs = batch_imgs.to(device, non_blocking=True)

        logits = model(batch_imgs)

        flipped_imgs = torch.flip(batch_imgs, dims=[3])
        logits_flip = model(flipped_imgs)
        logits_avg = (logits + logits_flip) / 2.0

        preds = (
            torch.argmax(logits_avg, dim=1).detach().cpu().numpy().astype(int).tolist()
        )
        labels.extend(preds)
        image_ids.extend(list(batch_ids))

submission = pd.DataFrame({"image_id": image_ids, "label": labels})



## === cell 6
submission["label"] = submission["label"].astype(int)
submission = submission.sort_values("image_id").reset_index(drop=True)

sample = pd.read_csv(SAMPLE_SUB)
sample["image_id"] = sample["image_id"].astype(str)
submission = sample[["image_id"]].merge(submission, on="image_id", how="left")
assert submission["label"].notna().all(), "Some test image_ids are missing predictions."

submission["label"] = submission["label"].astype(int)
submission.to_csv("submission.csv", index=False)

assert submission.shape[0] == sample.shape[0]
assert list(submission.columns) == ["image_id", "label"]
submission.head()
