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

albumentations==2.0.8
geopandas==0.14.4
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

0.8750377757630704

# 6. Current score

0.41779

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the albumentations import/transform API break (v2 moved modules) by importing `Resize` and `Normalize` from the top-level package, which unblocks the whole pipeline. I also make the model loading robust in this environment by (a) ensuring `torch` is imported before loading, (b) using `map_location` and a small loader that can handle either a full pickled model or a state_dict, and (c) gracefully falling back to a valid baseline submission if the referenced model files are not available. Finally, I ensure the submission has exactly the same `image_id` order/length as `sample_submission.csv` to avoid “invalid length” errors and always write `submission.csv`.'
- What this solution (achieved 0.55568) has done: 'Your current low score is consistent with the ensemble being effectively empty (missing model files), which makes the code submit constant class-0 predictions. To move the score toward the target with minimal core-logic changes, I keep the same inference pipeline but add a lightweight, on-the-fly training fallback that trains the existing ResNet-18 (already in your code) on `train_images` when pretrained model pickles aren’t available. This preserves your architecture (ResNet-18) and evaluation semantics (argmax of logits), and only changes behavior in the “no models loaded” case to produce meaningful predictions. I also fix the test normalization to use ImageNet mean/std (a small but relevant inference correctness tweak). The script still write a valid `submission.csv` in the same format and order.'
- What this solution (achieved 0.05531) has done: 'The timeout is almost certainly coming from the fallback training path: it loads and augments ~18.7k images at 512×512 for 6 epochs, which is far too slow in 10 minutes. To make runtime deterministic and fast, the refactor avoids ever entering fallback training by using only the provided pretrained ensemble checkpoints; if none are available, it fails fast with a clear error instead of training. For inference, the changes reduce Python overhead and DataLoader stalls by enabling persistent workers, prefetching, and moving `no_grad()` outside the inner ensemble loop (same math), and by using faster PIL decoding (`pillow-simd`-style settings if available) while keeping identical transforms and logits-summing semantics.'
- What this solution (achieved 0.41779) has done: 'The crash happens because none of the referenced pretrained model files exist in this Kaggle environment, and the notebook currently hard-errors before it can write a submission. To both fix the runtime and move the score up toward the target, I re-enable the existing fallback training path but make it fast enough for the 600s limit by training the same ResNet-18 architecture on a small, stratified subset of the training data for a couple of epochs (same loss/optimizer semantics). I also switch the dataloader to use `persistent_workers`/`prefetch_factor` to reduce input stalls, and ensure predictions are produced for every test image in the exact `sample_submission.csv` order. The ensemble path is preserved exactly when model files are present; the changes only affect the “no models loaded” case.'

# 9. Code solution

## === cell 0
import io
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

INPUT_DIR = Path("/kaggle/input/cassava-leaf-disease-classification")
WORKING_DIR = Path("/kaggle/working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

Image.MAX_IMAGE_PIXELS = None
try:
    Image.LOAD_TRUNCATED_IMAGES = True
except Exception:
    pass

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



## === cell 1
batch_size = 32
valid_input_size = 600  # kept for compatibility with original notebook variable
test_img_path = "../input/cassava-leaf-disease-classification/test_images"

if not Path(test_img_path).exists():
    test_img_path = str(INPUT_DIR / "test_images")

sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
if not Path(sample_sub_path).exists():
    sample_sub_path = str(INPUT_DIR / "sample_submission.csv")

sample_sub = pd.read_csv(sample_sub_path)
assert {"image_id", "label"}.issubset(sample_sub.columns)
test_image_ids = sample_sub["image_id"].tolist()

train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
if not Path(train_csv_path).exists():
    train_csv_path = str(INPUT_DIR / "train.csv")

train_img_path = "../input/cassava-leaf-disease-classification/train_images"
if not Path(train_img_path).exists():
    train_img_path = str(INPUT_DIR / "train_images")

train_df = pd.read_csv(train_csv_path)
assert {"image_id", "label"}.issubset(train_df.columns)



## === cell 2
model_fnames = [
    "../input/cassava-notebook-15-resnext-models/resnext101wsl_epoch_6.pickle",
    "../input/cassava-notebook-16-models/resnext101wsl_epoch_8.pickle",
    "../input/cassava-notebook-19-models/resnext101wsl_epoch_8.pickle",
    "../input/cassava-notebook-20-models/resnext101wsl_epoch_6.pickle",
]


def _build_default_model(pretrained: bool = False):
    if pretrained:
        try:
            m = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            )
        except Exception:
            m = torchvision.models.resnet18(weights=None)
    else:
        m = torchvision.models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, 5)
    return m


def _load_model_any(path: str):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, torch.nn.Module):
        return obj
    state = None
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state = obj["state_dict"]
        elif "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            state = obj["model_state_dict"]
        elif all(isinstance(k, str) for k in obj.keys()):
            state = obj
    if state is not None:
        m = _build_default_model(pretrained=False)
        m.load_state_dict(state, strict=False)
        return m
    raise ValueError(f"Unrecognized model format in: {path}")


models = []
missing = []
for p in model_fnames:
    if Path(p).exists():
        try:
            m = _load_model_any(p).to(device).eval()
            models.append(m)
        except Exception as e:
            missing.append((p, f"load_failed: {repr(e)}"))
    else:
        missing.append((p, "missing"))

print(f"Loaded models: {len(models)}; missing/failed: {len(missing)}")
if missing:
    print("Model load issues (first 4):", missing[:4])



## === cell 3
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_SIZE = 512

test_tfms = A.Compose(
    [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
)

train_tfms = A.Compose(
    [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.04, scale_limit=0.10, rotate_limit=12, p=0.5, border_mode=0
        ),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
)


class ImageClassificationDataset(Dataset):
    def __init__(self, img_dir, df, tfms):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.df = df.reset_index(drop=True)
        self.tfms = tfms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        label = int(self.df.loc[idx, "label"])
        img_path = self.img_dir / image_id
        img = np.array(Image.open(img_path).convert("RGB"))
        img = self.tfms(image=img)["image"]
        return img, label


class TestDataset(Dataset):
    def __init__(self, path, tfms, image_ids=None):
        super().__init__()
        self.path = Path(path)
        self.tfms = tfms
        if image_ids is None:
            self.image_ids = sorted(
                [
                    p.name
                    for p in self.path.iterdir()
                    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
                ]
            )
        else:
            self.image_ids = list(image_ids)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        filename = self.image_ids[idx]
        img_path = self.path / filename
        img = np.array(Image.open(img_path).convert("RGB"))
        img = self.tfms(image=img)["image"]
        return filename, img


test_ds = TestDataset(test_img_path, test_tfms, image_ids=test_image_ids)

_num_workers = min(8, os.cpu_count() or 1)
test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    num_workers=_num_workers,
    drop_last=False,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)




## === cell 4
def _stratified_subset(df: pd.DataFrame, n_total: int, seed: int = 42) -> pd.DataFrame:
    if n_total >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    rng = np.random.RandomState(seed)
    counts = df["label"].value_counts().to_dict()
    labels = sorted(counts.keys())
    alloc = {lab: max(1, int(round(n_total * counts[lab] / len(df)))) for lab in labels}
    cur = sum(alloc.values())
    if cur > n_total:
        for lab in sorted(labels, key=lambda x: alloc[x], reverse=True):
            if cur <= n_total:
                break
            if alloc[lab] > 1:
                alloc[lab] -= 1
                cur -= 1
    elif cur < n_total:
        for lab in sorted(labels, key=lambda x: counts[x], reverse=True):
            if cur >= n_total:
                break
            alloc[lab] += 1
            cur += 1

    parts = []
    for lab in labels:
        part = df[df["label"] == lab].sample(
            n=alloc[lab], random_state=int(rng.randint(0, 10**9))
        )
        parts.append(part)
    out = (
        pd.concat(parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return out


def train_fallback_model(
    train_df: pd.DataFrame, img_dir: str, epochs: int = 2, subset_n: int = 6000
):
    model = _build_default_model(pretrained=True).to(device)
    model.train()

    use_df = _stratified_subset(train_df, n_total=subset_n, seed=42)

    ds = ImageClassificationDataset(img_dir, use_df, train_tfms)
    nw = min(4, os.cpu_count() or 1)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    for ep in range(epochs):
        running_loss = 0.0
        n = 0
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu()) * imgs.size(0)
            n += imgs.size(0)

        scheduler.step()
        print(
            f"Fallback training epoch {ep+1}/{epochs} - loss: {running_loss/max(n,1):.4f} - lr: {optimizer.param_groups[0]['lr']:.2e}"
        )

    model.eval()
    return model


if len(models) == 0:
    print(
        "No pretrained ensemble models found at the specified paths; "
        "training a fallback ResNet-18 on a stratified subset to produce meaningful predictions."
    )
    models = [train_fallback_model(train_df, train_img_path, epochs=2, subset_n=6000)]




## === cell 5
class EnsemblePredictor:
    def __init__(self, models):
        super().__init__()
        self.models = list(models)

    def predict_on_loader(self, loader):
        if len(self.models) == 0:
            raise RuntimeError("No models available for prediction.")

        predictions = []
        filenames = []

        with torch.no_grad():
            for file, img in loader:
                img = img.to(device, non_blocking=True)

                pred_sum = None
                for model in self.models:
                    out = model(img)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    out = out.detach().float()
                    pred_sum = out if pred_sum is None else (pred_sum + out)

                if pred_sum is None:
                    raise RuntimeError(
                        "Prediction failed: pred_sum is None (no model outputs)."
                    )

                batch_preds = torch.argmax(pred_sum, dim=1).cpu().tolist()
                predictions.extend(batch_preds)
                filenames.extend(list(file))

        return predictions, filenames


predictor = EnsemblePredictor(models)

try:
    predictions, filenames = predictor.predict_on_loader(test_loader)
except Exception as e:
    print("Prediction failed with exception:", repr(e))
    print(
        "Falling back to constant class-0 predictions to produce a valid submission.csv."
    )
    filenames = list(test_image_ids)
    predictions = [0] * len(filenames)

assert len(predictions) == len(test_image_ids), (len(predictions), len(test_image_ids))
assert (
    filenames == test_image_ids
), "Prediction order must match sample_submission image_id order."



## === cell 6
sub = pd.DataFrame({"image_id": filenames, "label": predictions})

assert sub.shape[0] == sample_sub.shape[0]
assert sub["image_id"].tolist() == sample_sub["image_id"].tolist()

out_path = WORKING_DIR / "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())
