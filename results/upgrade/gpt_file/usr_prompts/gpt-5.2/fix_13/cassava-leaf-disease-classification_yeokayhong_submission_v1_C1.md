# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.13

# 3. Installed packages

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
tqdm==4.67.1

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

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image

test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_data_directory = "/kaggle/input/cassava-leaf-disease-classification/train_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
model_path = "/kaggle/input/vit_l_cassava/pytorch/default/1/model_weights_3.pth"

image_size = 518
num_classes = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_grad_enabled(False)
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

val_transforms = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)




## === cell 1
def _find_checkpoint_fallback(preferred_path: str) -> str | None:
    """
    Return None if no checkpoint is available under /kaggle/input.
    """
    if os.path.isfile(preferred_path):
        return preferred_path

    base_inputs = [
        "/kaggle/input/vit_l_cassava/pytorch/default/1",
        "/kaggle/input/vit_l_cassava",
        "/kaggle/input",
    ]
    candidate_files = [
        "model_weights_3.pth",
        "model_weights.pth",
        "model.pth",
        "weights.pth",
        "best.pth",
        "checkpoint.pth",
    ]

    candidates = []
    for b in base_inputs:
        for fn in candidate_files:
            fp = os.path.join(b, fn)
            if os.path.isfile(fp):
                candidates.append(fp)

    if candidates:
        chosen = candidates[0]
        print(
            f"[WARN] Checkpoint not found at '{preferred_path}'. Using fallback checkpoint: {chosen}"
        )
        return chosen

    print(
        f"[WARN] Checkpoint not found at '{preferred_path}', and no fallback .pth files were found in expected locations. "
        "Falling back to torchvision pretrained ViT weights."
    )
    return None


def _load_state_dict_robust(
    model: torch.nn.Module, ckpt_path: str, device: torch.device
) -> None:
    """
    Handle checkpoints saved as raw state_dict or wrapped dict (e.g., {'state_dict': ...}),
    stripping common prefixes like 'module.' and 'model.'.
    """
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        else:
            sd = obj
    else:
        raise RuntimeError(
            f"Unsupported checkpoint object type: {type(obj)} at {ckpt_path}"
        )

    cleaned = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(
            f"[WARN] Missing keys when loading checkpoint (showing up to 20): {missing[:20]}"
        )
    if unexpected:
        print(
            f"[WARN] Unexpected keys when loading checkpoint (showing up to 20): {unexpected[:20]}"
        )


def _freeze_backbone_train_head_only(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.heads.head.parameters():
        p.requires_grad = True


def _train_head_one_epoch(
    model: torch.nn.Module,
    train_df: pd.DataFrame,
    train_dir: str,
    tfm,
    device: torch.device,
    batch_size: int,
) -> None:
    """
    Minimal, legitimate score-improvement step when no cassava checkpoint exists:
    fine-tune only the 5-class head for 1 epoch (backbone frozen), keeping
    model family + argmax inference semantics unchanged.
    """
    from torch.utils.data import Dataset, DataLoader

    class TrainDS(Dataset):
        def __init__(self, df, root_dir, tfm):
            self.df = df.reset_index(drop=True)
            self.root_dir = root_dir
            self.tfm = tfm

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            image_id = str(self.df.loc[idx, "image_id"])
            y = int(self.df.loc[idx, "label"])
            image_path = os.path.join(self.root_dir, image_id)
            img = Image.open(image_path).convert("RGB")
            x = self.tfm(img)
            return x, y

    def _seed_worker(worker_id: int):
        worker_seed = seed + 2000 + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    ds = TrainDS(train_df, train_dir, tfm)
    num_workers = min(4, os.cpu_count() or 0)  # keep within 600s budget
    pin_memory = device.type == "cuda"

    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
    )

    _freeze_backbone_train_head_only(model)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.heads.head.parameters(), lr=3e-3, weight_decay=0.0
    )

    prev_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    try:
        for batch_x, batch_y in tqdm(
            loader, desc="Train head (1 epoch)", total=len(loader)
        ):
            batch_x = batch_x.to(device, non_blocking=(device.type == "cuda"))
            batch_y = batch_y.to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
    finally:
        torch.set_grad_enabled(prev_grad)

    model.eval()
    torch.set_grad_enabled(False)




## === cell 2
resolved_model_path = _find_checkpoint_fallback(model_path)

using_torchvision_pretrained = resolved_model_path is None
vit_weights = None

if using_torchvision_pretrained:
    try:
        vit_weights = models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
    except Exception:
        vit_weights = models.ViT_H_14_Weights.DEFAULT

    model = models.vit_h_14(weights=vit_weights, image_size=image_size)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    print("[INFO] Using torchvision pretrained backbone (no cassava checkpoint found).")

    val_transforms = vit_weights.transforms()
    print("[INFO] Using ViT_H_14_Weights official preprocessing transforms.")
else:
    model = models.vit_h_14(weights=None, image_size=image_size)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    _load_state_dict_robust(model, resolved_model_path, device)
    print(f"[INFO] Loaded cassava checkpoint from: {resolved_model_path}")

model.to(device)
model.eval()

if using_torchvision_pretrained:
    train_df = pd.read_csv(train_csv_path)
    train_bs = 32 if device.type == "cuda" else 4
    _train_head_one_epoch(
        model=model,
        train_df=train_df,
        train_dir=train_data_directory,
        tfm=val_transforms,
        device=device,
        batch_size=train_bs,
    )



## === cell 3
from torch.utils.data import Dataset, DataLoader

sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].astype(str).tolist()

try:
    from torchvision.io import read_file, decode_jpeg

    _TV_IO_OK = True
except Exception:
    _TV_IO_OK = False


_HAS_TENSOR_TRANSFORMS = True
try:
    from torchvision.transforms import functional as F
    from torchvision.transforms.functional import InterpolationMode
except Exception:
    _HAS_TENSOR_TRANSFORMS = False


def _apply_val_transforms_fast(img_chw_uint8: torch.Tensor) -> torch.Tensor:
    x = img_chw_uint8
    x = F.resize(
        x,
        (image_size, image_size),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    x = x.to(dtype=torch.float32).div_(255.0)
    x = F.normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return x


class CassavaTestDataset(Dataset):
    def __init__(self, image_ids, root_dir, tfm):
        self.image_ids = image_ids
        self.root_dir = root_dir
        self.tfm = tfm

        self._use_fast_tensor_path = (
            _TV_IO_OK and _HAS_TENSOR_TRANSFORMS and self.tfm is val_transforms
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.root_dir, image_id)

        if _TV_IO_OK:
            data = read_file(image_path)
            img = decode_jpeg(data, mode="RGB")  # uint8, [C,H,W]
            if self._use_fast_tensor_path:
                x = _apply_val_transforms_fast(img)
                return image_id, x
            img = transforms.functional.to_pil_image(img)
        else:
            img = Image.open(image_path).convert("RGB")

        x = self.tfm(img)
        return image_id, x


def _seed_worker(worker_id: int):
    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


test_ds = CassavaTestDataset(test_image_ids, test_data_directory, val_transforms)

cpu_cnt = os.cpu_count() or 0
num_workers = min(4, cpu_cnt)
pin_memory = device.type == "cuda"

batch_size = 16 if device.type == "cuda" else 2

g = torch.Generator()
g.manual_seed(seed)

test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
    generator=g,
)

test_predictions = []
test_image_ids_out = []

model.eval()
with torch.inference_mode():
    for batch_ids, batch_x in tqdm(test_loader, desc="Test", total=len(test_loader)):
        batch_x = batch_x.to(device, non_blocking=(device.type == "cuda"))

        output = model(batch_x)  # [B, C]
        pred = output.argmax(dim=1).to("cpu").numpy().astype(int)

        test_predictions.extend(pred.tolist())
        test_image_ids_out.extend(list(batch_ids))

submission_df = pd.DataFrame(
    {"image_id": test_image_ids_out, "label": test_predictions}
)
submission_df["image_id"] = submission_df["image_id"].astype(str)
submission_df["label"] = submission_df["label"].astype(int)

submission_df = sample_sub[["image_id"]].merge(submission_df, on="image_id", how="left")
submission_df["label"] = submission_df["label"].fillna(0).astype(int)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file created: {submission_path}")
print(submission_df.head())
print("Submission shape:", submission_df.shape)
