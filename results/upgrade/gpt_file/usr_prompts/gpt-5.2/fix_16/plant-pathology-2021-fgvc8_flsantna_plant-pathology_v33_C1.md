# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

No external packages required in the script and installed.

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

# 5. Code solution

## === cell 0
import os
import sys
import glob
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from PIL import Image

torch.manual_seed(42)
np.random.seed(42)

output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/conve01/eff7-e3/epoch-3"

image_dims = (300, 300, 3)

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

if len(dataset_labels) != 6:
    print(
        "WARNING: Expected 6 classes but found",
        len(dataset_labels),
        "classes:",
        dataset_labels,
    )

print("Classes:", dataset_labels)
print(
    "Test dir exists:",
    os.path.isdir(test_dir),
    "num test images:",
    len(os.listdir(test_dir)) if os.path.isdir(test_dir) else -1,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

torch.set_num_threads(min(4, os.cpu_count() or 1))




## === cell 1
import timm


def _find_any_torch_weight_candidate(path: str):
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        return None

    exts = (".pth", ".pt", ".bin")

    try:
        files = os.listdir(path)
        candidates = [os.path.join(path, f) for f in files if f.lower().endswith(exts)]
    except Exception:
        candidates = []

    if not candidates:
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith(exts):
                    candidates.append(os.path.join(root, f))

    if not candidates:
        return None

    def score(p):
        base = os.path.basename(p).lower()
        pref = 0
        if "best" in base:
            pref -= 5
        if "epoch" in base:
            pref -= 2
        if "model" in base:
            pref -= 1
        return (pref, len(p), p)

    candidates.sort(key=score)
    return candidates[0]


def _search_kaggle_input_for_checkpoint():
    roots = ["../input", "/kaggle/input"]
    exts = (".pth", ".pt", ".bin")

    candidates = []
    for r in roots:
        if not os.path.isdir(r):
            continue

        try:
            top = os.listdir(r)
        except Exception:
            continue

        dirs_to_check = [r]
        for d in top:
            dp = os.path.join(r, d)
            if os.path.isdir(dp):
                dirs_to_check.append(dp)

        for root in dirs_to_check:
            try:
                for f in os.listdir(root):
                    fp = os.path.join(root, f)
                    if os.path.isfile(fp) and f.lower().endswith(exts):
                        candidates.append(fp)
            except Exception:
                continue

    if not candidates:
        return None

    def score(p):
        base = os.path.basename(p).lower()
        pref = 0
        if "eff" in base or "efficientnet" in base or "b7" in base or "eff7" in base:
            pref -= 5
        if "best" in base:
            pref -= 3
        if "epoch" in base:
            pref -= 2
        if "fold" in base:
            pref -= 1
        return (pref, len(p), p)

    candidates.sort(key=score)
    return candidates[0]


class MultiLabelTorch(nn.Module):
    """
    Minimal PyTorch equivalent for inference:
    EfficientNet-B7 backbone (global pooled) -> 6 independent logits -> sigmoid.
    """

    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b7", pretrained=False, num_classes=0, global_pool="avg"
        )
        in_features = self.backbone.num_features
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        probs = torch.sigmoid(logits)
        return probs


def load_weights_robust_torch(model: nn.Module, model_path: str):
    cand = _find_any_torch_weight_candidate(model_path)

    if cand is None:
        alt = _search_kaggle_input_for_checkpoint()
        if alt is not None:
            print(
                f"WARNING: No checkpoint found under '{model_path}'. "
                f"Falling back to checkpoint found at: {alt}"
            )
            cand = alt

    if cand is None:
        print(
            f"WARNING: Could not find any .pth/.pt/.bin checkpoint under '{model_path}' "
            f"or within Kaggle input roots. Proceeding with randomly initialized weights."
        )
        return False

    ckpt = torch.load(cand, map_location="cpu")

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        raise OSError(f"Unsupported checkpoint type at {cand}: {type(ckpt)}")

    cleaned = {}
    for k, v in state.items():
        nk = k
        for prefix in ("module.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print("Loaded weights from:", cand)
    print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
    if len(missing) > 0:
        print("Sample missing:", missing[:10])
    if len(unexpected) > 0:
        print("Sample unexpected:", unexpected[:10])
    return True


_MEAN_CPU = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_STD_CPU = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

try:
    import torchvision
    from torchvision.io import decode_image, read_file
    from torchvision.transforms.functional import resize as tv_resize
    from torchvision.transforms.functional import InterpolationMode

    _HAS_TORCHVISION = True
    try:
        torchvision.set_image_backend("accimage")
    except Exception:
        pass
except Exception:
    _HAS_TORCHVISION = False


def preprocess_pil(img: Image.Image, size=(300, 300)):
    img = img.convert("RGB").resize(size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)  # [0,255], float32
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    x = torch.from_numpy(arr)  # CPU float32
    x = x / 255.0
    x = (x - _MEAN_CPU) / _STD_CPU
    return x


def preprocess_path_fast(fpath: str, size=(300, 300)):
    if not _HAS_TORCHVISION:
        with Image.open(fpath) as img:
            return preprocess_pil(img, size=size)

    data = read_file(fpath)
    x = decode_image(data, mode=torchvision.io.ImageReadMode.RGB)  # uint8, CHW
    x = tv_resize(
        x, size=list(size), interpolation=InterpolationMode.BILINEAR, antialias=False
    )
    x = x.to(dtype=torch.float32).div_(255.0)
    x = x.sub_(_MEAN_CPU).div_(_STD_CPU)
    return x.contiguous()




## === cell 2
from torch.utils.data import Dataset, DataLoader


class TestImageDataset(Dataset):
    def __init__(self, root_dir, filenames, size_hw):
        self.root_dir = root_dir
        self.filenames = filenames
        self.size_hw = size_hw

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        fpath = os.path.join(self.root_dir, fname)
        x = preprocess_path_fast(fpath, size=self.size_hw)
        return fname, x


def _collate_fn(batch):
    fnames, xs = zip(*batch)
    return list(fnames), torch.stack(xs, dim=0)


def _dl_worker_init_fn(worker_id: int):
    try:
        torch.set_num_threads(1)
    except Exception:
        pass




## === cell 3
if __name__ == "__main__":
    model = MultiLabelTorch(num_classes=len(dataset_labels)).to(device)
    model.eval()

    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    _loaded = load_weights_robust_torch(model, model_dir)

    images_path_list = sorted(
        f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")
    )

    classes = dataset_labels
    threshold = 0.7  # preserve original decision threshold

    cpu_cnt = os.cpu_count() or 1

    if device.type == "cuda":
        num_workers = min(4, cpu_cnt)  # was 8; often faster end-to-end for JPEG decode
        batch_size = 128
        pin_memory = True
        prefetch_factor = 2
    else:
        num_workers = min(2, cpu_cnt) if cpu_cnt > 1 else 0
        batch_size = 32
        pin_memory = False
        prefetch_factor = 2 if num_workers > 0 else None

    ds = TestImageDataset(
        root_dir=test_dir,
        filenames=images_path_list,
        size_hw=(image_dims[0], image_dims[1]),
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=_collate_fn,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
        worker_init_fn=_dl_worker_init_fn if num_workers > 0 else None,
    )

    values = []
    cls_arr = np.asarray(classes, dtype=object)

    with torch.inference_mode():
        for fnames, xb in dl:
            if device.type == "cuda":
                xb = xb.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last
                )
            else:
                xb = xb.to(device)

            probs = model(xb)  # (B, C)
            mask = (probs > threshold).to("cpu").numpy()  # bool (B, C)

            for fname, row in zip(fnames, mask):
                if not row.any():
                    values.append([fname, "healthy"])
                else:
                    values.append([fname, " ".join(cls_arr[row].tolist())])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)
    print("Wrote submission.csv with shape:", csv_pd.shape, "to", out_path)
    print("Checkpoint loaded:", _loaded)
    print(csv_pd.head())
