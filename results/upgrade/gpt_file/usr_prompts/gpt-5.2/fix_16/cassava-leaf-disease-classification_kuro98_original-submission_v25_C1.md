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

3.12

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

0.8955877908733756

# 6. Current score

0.71413

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the pipeline so it always produces a valid `submission.csv` with the correct number of rows and aligned `image_id`s by reading `sample_submission.csv` as the source of truth and iterating test images in that exact order (no shuffling). I also fix the missing external model file by replacing the `torch.load(...)` with a locally constructible ViT from `torchvision` (same overall approach: pretrained image classifier + softmax/argmax inference), and add safe image loading (`RGB`) to avoid shape/type issues. Finally, I correct the TTA batching logic so averaging is done per-image across augmentations, and ensure deterministic settings remain compatible with DataLoader workers.'
- What this solution (achieved 0.05531) has done: 'I fix the runtime error by making the ViT’s expected input resolution match the pipeline’s `img_size` (384), since the model currently asserts `224` while your transforms produce `384`. This is a minimal, score-positive correction because it restores correct forward passes without changing the overall approach (pretrained ViT + linear head + softmax/argmax + optional TTA). I keep the rest of the data loading, TTA averaging, ordering via `sample_submission.csv`, and submission writing unchanged. I also add a small safety assertion to ensure the submission row count matches the sample submission.'
- What this solution (achieved 0.23692) has done: 'I fix the root runtime issue by constructing the ViT with the default pretrained image_size=224 (torchvision enforces this for pretrained weights) and align the transforms/crop to produce 224×224 inputs, which also resolves the downstream `model` undefined error caused by the earlier exception. I keep the same core approach (pretrained ViT backbone + replaced linear head + softmax/argmax + TTA averaging) and only adjust the image sizing to match the pretrained model contract. I also make TTA deterministic-safe by using stateless flips/rotations (instead of randomness) to avoid unstable/inadvertent behavior, while keeping the same TTA averaging semantics. The script then run end-to-end and write a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.7799) has done: 'Your score is far below the target, so the smallest safe way to move toward it is to fix the main semantic issue: you’re using an ImageNet-pretrained ViT but predicting 5 cassava classes with a randomly initialized head (so accuracy is near chance). I keep the same core approach (pretrained ViT + linear head + softmax/argmax + optional TTA) and add a minimal training step on `train.csv`/`train_images` to actually learn the 5-way head. To stay minimal and stable, I train only the classification head (freeze backbone), use the same normalization/transforms, and then run the existing inference/submission pipeline unchanged. This should substantially increase accuracy and move the score toward the target band while keeping runtime within limits.'
- What this solution (achieved 0.05531) has done: 'The timeout is dominated by repeated CPU image decoding/CenterCrop and the expensive default ViT preprocessing applied inside workers, plus extra overhead from PIL’s lazy file handling and DataLoader settings. I keep the exact same model, training loop, loss, and TTA semantics, but make data loading markedly faster by (1) forcing eager image decode/close, (2) caching the fixed CenterCrop result per file within each worker process, and (3) tuning DataLoader to reduce worker restart overhead and improve host→GPU throughput (persistent workers + prefetch + pinned memory). I also enable PyTorch’s CUDA matmul/conv TF32 fast path (not AMP) for a large speedup on modern GPUs with only negligible floating-point differences, without changing the algorithm. No epochs, batch sizes, architecture, transforms, or TTA views are reduced/changed.'
- What this solution (achieved 0.79484) has done: 'I fix the `DataLoader` construction bug that passes `batch_size` twice (once explicitly and once via `loader_kwargs`), which currently prevents `test_loader` from being created and causes the downstream `NameError`. I do this by excluding `batch_size` from the kwargs dict when building the test loader (keeping ordering and all performance-related settings the same). This change is score-neutral; it just makes the pipeline run end-to-end and reliably write a valid `submission.csv` aligned to `sample_submission.csv`. No model/training/inference semantics are changed.'
- What this solution (achieved 0.05531) has done: 'The timeout is dominated by preprocessing and cache warming: you currently iterate through *every* train/val sample up front and, worse, `weights.transforms()` performs random augmentation so caching those results is both incorrect and extremely expensive. I remove the full-dataset warm-cache pass, and make caching provably correct by caching only deterministic preprocessing (CenterCrop+resize/normalize) while keeping stochastic augmentation applied on-the-fly during training exactly as before. I also speed up the TTA collate to avoid Python nested loops, and tune DataLoader settings (workers/prefetch/persistent) to reduce overhead without changing training loops, model, loss, or evaluation. All paths remain unchanged and determinism/seeds are preserved.'
- What this solution (achieved 0.71413) has done: 'I fix the `KeyError: 'mean'` by not relying on `weights.meta["mean"/"std"]` (which isn’t present in this torchvision version) and instead using the standard ImageNet normalization values consistent with ViT-B/16 pretrained weights. This let `base_cache_transform` be created, which unblocks dataset/loader creation and removes the cascading `NameError`s in later cells. I keep the existing model, training loop, caching, and TTA logic unchanged to preserve semantics, only making the minimal normalization fix needed to run end-to-end and improve accuracy vs. the broken run. The script then train and write a valid `submission.csv` aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import hashlib
import pathlib
import random

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

torch.set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

torch.manual_seed(3407)
random.seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = True
cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"

train_csv_path = f"{DATA_DIR}/train.csv"
train_dir = f"{DATA_DIR}/train_images/"

test_dir = f"{DATA_DIR}/test_images/"
sample_sub_path = f"{DATA_DIR}/sample_submission.csv"

img_size = 224

batch_size = 16
num_workers = min(12, (os.cpu_count() or 4))
num_classes = 5
tta = True

train_epochs = 6
train_lr = 3e-4
finetune_epochs = 2
finetune_lr = 1e-5

from torchvision.models import vit_b_16, ViT_B_16_Weights

weights = ViT_B_16_Weights.IMAGENET1K_V1
model = vit_b_16(weights=weights)  # image_size fixed by weights (=224)
in_features = model.heads.head.in_features
model.heads.head = torch.nn.Linear(in_features, num_classes)

model.to(device)



## === cell 1
vit_preprocess = weights.transforms()

train_transforms = vit_preprocess
test_transforms = vit_preprocess

if tta:
    ttas = [
        v2.Identity(),  # include the original view
        v2.RandomHorizontalFlip(p=1.0),  # deterministic flip
        v2.RandomVerticalFlip(p=1.0),  # deterministic flip
        v2.RandomRotation(degrees=(90, 90), interpolation=InterpolationMode.BILINEAR),
    ]
else:
    ttas = None



## === cell 2
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

CACHE_DIR = "/kaggle/working/cassava_cc_cache_png"
pathlib.Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

from collections import OrderedDict

_TENSOR_CACHE = OrderedDict()
_TENSOR_CACHE_MAX = 4096  # bounded in-RAM cache per worker


def _cache_key(img_path: str, key_suffix: str) -> str:
    st = os.stat(img_path)
    sig = f"{img_path}|{st.st_size}|{int(st.st_mtime)}|{key_suffix}"
    h = hashlib.sha1(sig.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.pt")


def _load_rgb_pil(img_path: str) -> Image.Image:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        try:
            im.draft("RGB", (512, 512))
        except Exception:
            pass
        im.load()
    return im


def _atomic_save_tensor(t: torch.Tensor, cache_path: str) -> None:
    tmp_path = cache_path + f".tmp_{os.getpid()}"
    try:
        torch.save(t, tmp_path)
        try:
            os.replace(tmp_path, cache_path)
        except FileExistsError:
            pass
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _cached_tensor(
    img_path: str,
    preprocess_transform,
    key_suffix: str,
) -> torch.Tensor:
    """Returns a cached CPU tensor produced by preprocess_transform(PIL_RGB)."""
    k = (img_path, key_suffix)
    im_t = _TENSOR_CACHE.get(k)
    if im_t is not None:
        _TENSOR_CACHE.move_to_end(k)
        return im_t

    cache_path = _cache_key(img_path, key_suffix)
    if os.path.exists(cache_path):
        t = torch.load(cache_path, map_location="cpu")
    else:
        im = _load_rgb_pil(img_path)
        t = preprocess_transform(im)
        _atomic_save_tensor(t, cache_path)

    _TENSOR_CACHE[k] = t
    if len(_TENSOR_CACHE) > _TENSOR_CACHE_MAX:
        _TENSOR_CACHE.popitem(last=False)
    return t


class CassavaTrainDataset(VisionDataset):
    """Minimal train dataset: (image, label) from train.csv"""

    def __init__(self, data_dir, df, transform=None, base_cache_transform=None):
        super().__init__(root=data_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform

        self.cc = v2.CenterCrop((400, 400))

        self.base_cache_transform = base_cache_transform
        self._key_suffix = f"cc400|base_cache_v1|{type(base_cache_transform).__name__}"

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["image_id"]
        label = int(row["label"])
        img_path = os.path.join(self.root, filename)

        if self.transform is not None and self.base_cache_transform is not None:
            base = _cached_tensor(img_path, self.base_cache_transform, self._key_suffix)
            img = self.transform(base)
        elif self.transform is not None:
            img = self.transform(self.cc(_load_rgb_pil(img_path)))
        else:
            img = self.cc(_load_rgb_pil(img_path))
        return img, label

    def __len__(self):
        return len(self.df)


class CassavaTestDataset(VisionDataset):
    """Custom dataset for the Cassava test data (ordered by sample submission)."""

    def __init__(
        self, data_dir, image_ids, transform=None, ttas=None, base_cache_transform=None
    ):
        super().__init__(root=data_dir)
        self.transform = transform
        self.image_ids = list(image_ids)
        self.ttas = ttas

        self.cc = v2.CenterCrop((400, 400))

        self.base_cache_transform = base_cache_transform
        self._key_suffix_base = (
            f"cc400|base_cache_v1|{type(base_cache_transform).__name__}"
        )

    def __getitem__(self, idx):
        filename = self.image_ids[idx]
        img_path = os.path.join(self.root, filename)

        if self.base_cache_transform is not None:
            base = _cached_tensor(
                img_path, self.base_cache_transform, self._key_suffix_base
            )
        else:
            base = self.cc(_load_rgb_pil(img_path))

        if self.ttas is not None and self.transform is not None:
            imgs = [self.transform(t(base)) for t in self.ttas]
            return imgs, filename
        elif self.transform is not None:
            return self.transform(base), filename
        else:
            return base, filename

    def __len__(self):
        return len(self.image_ids)


def _warm_cache_for_dataset(ds: VisionDataset, max_items: int | None = None) -> None:
    return




## === cell 3
train_df = pd.read_csv(train_csv_path)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

base_cache_transform = v2.Compose(
    [
        v2.CenterCrop((400, 400)),
        v2.Resize(
            (img_size, img_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

g = torch.Generator().manual_seed(3407)
labels = torch.from_numpy(train_df["label"].to_numpy())
idx_all = torch.arange(len(train_df))

val_mask = torch.zeros(len(train_df), dtype=torch.bool)
for lbl in torch.unique(labels).tolist():
    idxs = idx_all[labels == lbl]
    perm = idxs[torch.randperm(len(idxs), generator=g)]
    n_val = max(1, int(round(0.1 * len(idxs))))
    val_mask[perm[:n_val]] = True

val_size = int(0.1 * len(train_df))
val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze(1)[:val_size].tolist()

train_df_val = train_df.iloc[val_idx].reset_index(drop=True)
train_df_tr = train_df.drop(index=val_idx).reset_index(drop=True)

train_dataset = CassavaTrainDataset(
    train_dir,
    df=train_df_tr,
    transform=train_transforms,
    base_cache_transform=base_cache_transform,
)
val_dataset = CassavaTrainDataset(
    train_dir,
    df=train_df_val,
    transform=test_transforms,
    base_cache_transform=base_cache_transform,
)

_warm_cache_for_dataset(train_dataset)
_warm_cache_for_dataset(val_dataset)

loader_kwargs = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    generator=g,  # keep shuffle deterministic across runs
    **{k: v for k, v in loader_kwargs.items() if v is not None},
)

val_loader = DataLoader(
    val_dataset,
    shuffle=False,
    **{k: v for k, v in loader_kwargs.items() if v is not None},
)

for p in model.parameters():
    p.requires_grad = False
for p in model.heads.head.parameters():
    p.requires_grad = True

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.heads.head.parameters(), lr=train_lr)


def _eval_acc(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))
    return correct / max(total, 1)


model.train()
for epoch in range(train_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))

    train_acc = correct / max(total, 1)
    val_acc = _eval_acc(val_loader)
    print(
        f"stage1 epoch {epoch+1}/{train_epochs} - loss: {running_loss/max(total,1):.4f} - train_acc: {train_acc:.4f} - val_acc: {val_acc:.4f}"
    )
    model.train()

for p in model.parameters():
    p.requires_grad = False
for p in model.heads.head.parameters():
    p.requires_grad = True

for p in model.encoder.layers[-1].parameters():
    p.requires_grad = True
for p in model.encoder.ln.parameters():
    p.requires_grad = True

finetune_params = [p for p in model.parameters() if p.requires_grad]
optimizer_ft = torch.optim.AdamW(finetune_params, lr=finetune_lr)

model.train()
for epoch in range(finetune_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer_ft.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer_ft.step()

        running_loss += float(loss.item()) * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))

    train_acc = correct / max(total, 1)
    val_acc = _eval_acc(val_loader)
    print(
        f"stage2 epoch {epoch+1}/{finetune_epochs} - loss: {running_loss/max(total,1):.4f} - train_acc: {train_acc:.4f} - val_acc: {val_acc:.4f}"
    )
    model.train()

model.eval()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
UnpicklingError                           Traceback (most recent call last)
/tmp/ipykernel_55/2622935985.py in <cell line: 0>()
    103     correct = 0
    104     total = 0
--> 105     for images, labels in train_loader:
    106         images = images.to(device, non_blocking=True)
    107         labels = labels.to(device, non_blocking=True)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

UnpicklingError: Caught UnpicklingError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/32940318.py", line 95, in __getitem__
    base = _cached_tensor(img_path, self.base_cache_transform, self._key_suffix)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/32940318.py", line 63, in _cached_tensor
    t = torch.load(cache_path, map_location="cpu")
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/serialization.py", line 1470, in load
    raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint. 
	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
	WeightsUnpickler error: Unsupported global: GLOBAL torchvision.tv_tensors._image.Image was not an allowed global by default. Please use `torch.serialization.add_safe_globals([Image])` or the `torch.serialization.safe_globals([Image])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.


## === cell 4
sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].tolist()

test_dataset = CassavaTestDataset(
    test_dir,
    image_ids=test_image_ids,
    transform=test_transforms,
    ttas=ttas,
    base_cache_transform=base_cache_transform,
)

test_loader_kwargs = {
    k: v for k, v in loader_kwargs.items() if v is not None and k != "batch_size"
}


def _tta_collate(batch):
    imgs_list, names = zip(
        *batch
    )  # tuple length B; each element list length T of [C,H,W]
    per_t = [torch.stack(frames, dim=0) for frames in zip(*imgs_list)]
    return per_t, list(names)


test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=_tta_collate if tta else None,
    **test_loader_kwargs,
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 5
all_names = []
all_preds = []

model.eval()
with torch.inference_mode():
    for batch_idx, (inputs, filenames) in enumerate(test_loader):
        filenames = list(filenames)

        if tta:
            tta_tensor = torch.stack(inputs, dim=0)  # [T,B,C,H,W]
            T, B = tta_tensor.shape[0], tta_tensor.shape[1]
            flat = tta_tensor.reshape(T * B, *tta_tensor.shape[2:]).to(
                device, non_blocking=True
            )

            preds = normalizer(model(flat))  # [T*B, num_classes]
            preds = preds.reshape(T, B, -1).mean(dim=0)  # [B, num_classes]
            pred_labels = torch.argmax(preds, dim=1).tolist()
        else:
            inputs = inputs.to(device, non_blocking=True)
            preds = normalizer(model(inputs))
            pred_labels = torch.argmax(preds, dim=1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)



## === cell 6
pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})

pred_df = pred_df.drop_duplicates(subset=["image_id"], keep="first")

submission = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

submission["label"] = submission["label"].fillna(0).astype(int)

assert submission.shape[0] == sample_sub.shape[0], (submission.shape, sample_sub.shape)
assert submission["label"].between(0, num_classes - 1).all()

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)



## === cell 7
submission.head()
