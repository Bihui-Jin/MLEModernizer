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

No external packages required in the script and installed.

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

0.899365367180417

# 6. Current score

0.77317

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.17302) has done: 'I remove the failing external pip installs and missing third‑party imports (VisionTransformer/albumentations/skimage/cv2/efficientnet), and replace them with lightweight, built-in equivalents that keep the same overall inference flow (multi-model ensemble → argmax label → submission.csv). I fix the TensorFlow/protobuf crash by avoiding TensorFlow entirely (it isn’t needed for producing a valid submission here) and keep everything in PyTorch+timm, which is available in Kaggle. I also fix multiple NameErrors caused by out-of-order cells by consolidating required imports/definitions and ensuring every referenced symbol exists before use. Finally, I ensure the submission has exactly the required columns (`image_id,label`), correct row count (matching `sample_submission.csv`), and is written with a `.csv` suffix.'
- What this solution (achieved 0.77317) has done: 'Your current score is low because the code is doing pure ImageNet-pretrained inference with random classification heads (timm replaces the final layer when `num_classes=5`, so outputs are essentially untrained), so predictions are near-random. To move accuracy toward the target with minimal change in “core logic” (still timm models + ensemble + argmax + submission), I (1) load the provided Cassava train labels, (2) train only the classification head for a single short epoch on resized images, and (3) reuse the same inference/ensemble pipeline afterward. This keeps the architecture and inference flow intact while making the predictions dataset-relevant instead of random. The rest of the submission formatting and paths remain unchanged.'

# 9. Code solution

## === cell 0
import os, sys, glob, random, warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import timm



## === cell 1
IMAGE_SIZE = 512
CLASSES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(42)



## === cell 2
DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = f"{DATA_ROOT}/test_images"
TRAIN_IMG_DIR = f"{DATA_ROOT}/train_images"
SAMPLE_SUB_PATH = f"{DATA_ROOT}/sample_submission.csv"
TRAIN_CSV_PATH = f"{DATA_ROOT}/train.csv"

assert os.path.exists(TEST_IMG_DIR), f"Missing test_images at {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing train_images at {TRAIN_IMG_DIR}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv at {SAMPLE_SUB_PATH}"
assert os.path.exists(TRAIN_CSV_PATH), f"Missing train.csv at {TRAIN_CSV_PATH}"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_sub = sample_sub[["image_id", "label"]].copy()

train_df = pd.read_csv(TRAIN_CSV_PATH)
train_df = train_df[["image_id", "label"]].copy()
train_df["label"] = train_df["label"].astype(int)



## === cell 3
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def pil_to_tensor_normalized(img: Image.Image, size: int = IMAGE_SIZE) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0,1]
    t = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
    t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
    return t


class PytorchCassavaDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str,
        transforms=None,
        with_label: bool = False,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.data_root = data_root
        self.transforms = transforms
        self.with_label = with_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        image_id = self.df.iloc[idx]["image_id"]
        path = os.path.join(self.data_root, image_id)
        img = Image.open(path)
        if self.transforms is None:
            x = pil_to_tensor_normalized(img, IMAGE_SIZE)
        else:
            x = self.transforms(img)

        if self.with_label:
            y = int(self.df.iloc[idx]["label"])
            return x, y
        return x




## === cell 4
class TimmModel(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=CLASSES
        )

    def forward(self, x):
        return self.model(x)


def inference_simple(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device, non_blocking=True).float()
            logits = model(images)
            preds.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_head_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    lr: float = 3e-3,
):
    model.train()

    for p in model.parameters():
        p.requires_grad = False

    head_params = []
    if hasattr(model.model, "classifier") and isinstance(
        model.model.classifier, nn.Module
    ):
        for p in model.model.classifier.parameters():
            p.requires_grad = True
        head_params = list(model.model.classifier.parameters())
    elif hasattr(model.model, "fc") and isinstance(model.model.fc, nn.Module):
        for p in model.model.fc.parameters():
            p.requires_grad = True
        head_params = list(model.model.fc.parameters())
    elif hasattr(model.model, "head") and isinstance(model.model.head, nn.Module):
        for p in model.model.head.parameters():
            p.requires_grad = True
        head_params = list(model.model.head.parameters())
    else:
        for p in model.parameters():
            p.requires_grad = True
        head_params = [p for p in model.parameters() if p.requires_grad]

    opt = torch.optim.AdamW(head_params, lr=lr)
    crit = nn.CrossEntropyLoss()

    for images, targets in train_loader:
        images = images.to(device, non_blocking=True).float()
        targets = torch.as_tensor(targets, device=device, dtype=torch.long)

        opt.zero_grad(set_to_none=True)
        logits = model(images)
        loss = crit(logits, targets)
        loss.backward()
        opt.step()




## === cell 5
test_df = sample_sub[["image_id"]].copy()
test_dataset = PytorchCassavaDataset(
    df=test_df, data_root=TEST_IMG_DIR, transforms=None, with_label=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=min(4, os.cpu_count() or 1),
    pin_memory=torch.cuda.is_available(),
)

train_dataset = PytorchCassavaDataset(
    df=train_df, data_root=TRAIN_IMG_DIR, transforms=None, with_label=True
)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=min(4, os.cpu_count() or 1),
    pin_memory=torch.cuda.is_available(),
)



## === cell 6
models_to_ensemble = [
    ("resnet50", 0.5),
    ("tf_efficientnet_b0_ns", 0.5),
]

all_preds = []
for mname, w in models_to_ensemble:
    seed_everything(42)

    model = TimmModel(mname, pretrained=True).to(device)

    train_head_one_epoch(model, train_loader, device=device, lr=3e-3)

    preds = inference_simple(model, test_loader, device)
    all_preds.append(w * preds)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

ensemble_preds = np.sum(all_preds, axis=0)  # (N, CLASSES)



## === cell 7
pred_labels = np.argmax(ensemble_preds, axis=1).astype(int)

submission_df = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": pred_labels}
)

submission_df = submission_df.merge(
    sample_sub[["image_id"]], on="image_id", how="right"
)
submission_df["label"] = submission_df["label"].fillna(0).astype(int)
submission_df = submission_df[["image_id", "label"]]

assert submission_df.shape[0] == sample_sub.shape[0], "Submission row count mismatch"
assert list(submission_df.columns) == [
    "image_id",
    "label",
], "Submission columns mismatch"



## === cell 8
submission_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_df.shape)
print(submission_df.head())
