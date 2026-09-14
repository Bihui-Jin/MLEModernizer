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

0.894983378664249

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the immediate runtime blocker by removing the dependency on a missing external model file and instead instantiate a torchvision ViT with the correct output dimension so inference can run end-to-end. I also fix submission validity issues by making the test DataLoader deterministic (no shuffling) and aligning the final predictions to `sample_submission.csv` order to guarantee matching length and IDs. Finally, I correct the TTA pipeline so it batches augmented views properly and ensures filenames are repeated/aggregated consistently, producing exactly one label per test image. These changes are minimal and focused on unblocking execution and producing a valid `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'You’re hitting a TTA collation shape issue: the default DataLoader collate turns your per-sample list-of-TTA-tensors into a per-TTA tensor batch, so iterating as if `inputs` were a list per sample causes the `torch.stack()` TypeError. I fix cell 4 to handle the actual batched structure (a list of length T, each a tensor of shape B×C×H×W), flatten it to (B*T) for the model, then reshape back to average over T. To move accuracy toward the target, I also load the pretrained ViT weights but correctly replace the classification head and load a finetuned checkpoint if present (otherwise keep pretrained features), and ensure deterministic TTA by using functional transforms with fixed seeds per image. The rest (paths, submission alignment, no shuffling) stays the same.'
- What this solution (achieved 0.13191) has done: 'The runtime error comes from feeding 384×384 tensors into a ViT-B/16 model that was instantiated with ImageNet weights expecting 224×224 inputs; I fix this by using the corresponding 384px pretrained weights so the model’s internal `image_size` matches your `img_size` and transforms. I also switch to using the weights’ built-in preprocess transform to ensure normalization/resizing are exactly aligned with the pretrained checkpoint (score-positive but still same core approach). Finally, I keep your existing TTA batching logic but make it robust to different collate structures and ensure a valid `submission.csv` is always written with the correct row count/order.'
- What this solution (achieved 0.1293) has done: 'The crash is because your `test_images/` directory contains a nested `test_images/` folder, and the dataset currently tries to open that directory as if it were an image file. I minimally fix the dataset to only include real image files (e.g., `.jpg/.png`) and ignore subdirectories, which unblocks the DataLoader and lets inference finish. To move accuracy upward toward the target (without changing your core approach), I also make TTA deterministic by seeding torchvision v2 random transforms per-image so you get stable, reproducible augmented views. Everything else (ViT-B/16 384px weights, head replacement, TTA averaging, submission alignment to `sample_submission.csv`) remains the same.'
- What this solution (achieved 0.05531) has done: 'The timeout is dominated by (1) fine-tuning the head over the full 18.7k training images and (2) expensive per-batch TTA that runs three stochastic geometric transforms on GPU/CPU and then forwards 4× the images through ViT. To keep the same core model/loops/semantics, the main speedups are: use the provided TFRecords for training/test to avoid slow JPEG decode and Python-level file iteration, fuse/collate TTA views into a single batched forward (still identical averaging logic), and reduce dataloader overhead with better worker/prefetch/persistent settings and avoiding repeated per-item work. These changes are equivalent (same pixels/labels, same transforms and averaging, same loss/training steps) but remove major I/O and Python overhead that causes the 10-minute timeout. Determinism is preserved by keeping the same seeds and explicitly seeding TFRecord shuffling and TTA generators.'

# 9. Code solution

## === cell 0
import os

import pandas as pd
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4))))

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = True
cudnn.benchmark = True  # faster deterministic kernel selection

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    torch.multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

train_tfrec_dir = "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords/"
test_tfrec_dir = "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/"

img_size = 384
batch_size = 16
num_workers = min(8, (os.cpu_count() or 4))
num_classes = 5
tta = True

finetune_head = True
finetune_epochs = 3
finetune_lr = 3e-4

from torchvision.models import vit_b_16, ViT_B_16_Weights

weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1  # image_size=384
model = vit_b_16(weights=weights)
model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)

ckpt_candidates = [
    "./model.pth",
    "./best.pth",
    "./checkpoint.pth",
    "/kaggle/working/model.pth",
    "/kaggle/working/best.pth",
    "/kaggle/working/checkpoint.pth",
]
loaded_any_ckpt = False
for p in ckpt_candidates:
    if os.path.isfile(p):
        state = torch.load(p, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict):
            new_state = {}
            for k, v in state.items():
                nk = k.replace("module.", "")
                new_state[nk] = v
            missing, unexpected = model.load_state_dict(new_state, strict=False)
            print(f"Loaded checkpoint: {p}")
            print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            loaded_any_ckpt = True
        break

model.to(device)

if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

try:
    if hasattr(torch, "compile"):
        model = torch.compile(
            model, mode="reduce-overhead", fullgraph=False, dynamic=False
        )
        print("torch.compile enabled")
except Exception as e:
    print("torch.compile not available, continuing eager. Reason:", repr(e))



## === cell 1
import glob
import json
from typing import List, Optional, Tuple

from torch.utils.data import IterableDataset
from torchvision.io import decode_jpeg, ImageReadMode

weights_tfms = weights.transforms()

from torchdata.datapipes.iter import FileLister, FileOpener
from torchdata.datapipes.iter import TFRecordLoader  # type: ignore


@torch.jit.script
def _center_crop_600_tensor(img: torch.Tensor) -> torch.Tensor:
    h = img.shape[1]
    w = img.shape[2]
    th = 600
    tw = 600
    if h == th and w == tw:
        return img
    top = (h - th) // 2
    left = (w - tw) // 2
    if top < 0:
        top = 0
    if left < 0:
        left = 0
    return img[:, top : top + th, left : left + tw]


def _seed_worker(worker_id: int):
    base_seed = 3407
    seed = (base_seed + worker_id) % (2**32 - 1)
    torch.manual_seed(seed)


def _list_tfrec_files(tfrec_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(tfrec_dir, "*.tfrec")))
    if len(files) == 0:
        raise RuntimeError(f"No .tfrec files found in: {tfrec_dir}")
    return files


def _build_tfrec_datapipe(tfrec_files: List[str]):
    dp = FileLister(tfrec_files)
    dp = FileOpener(dp, mode="rb")
    dp = TFRecordLoader(dp)
    return dp


class CassavaTFRecordDataset(IterableDataset):
    """
    Iterable TFRecord dataset for Cassava.
    If labeled=True yields (img_tensor, label_int), else yields (img_tensor, image_id_str).
    """

    def __init__(
        self,
        tfrec_dir: str,
        transform=None,
        labeled: bool = True,
        shuffle_files: bool = False,
        seed: int = 3407,
    ):
        super().__init__()
        self.tfrec_files = _list_tfrec_files(tfrec_dir)
        self.transform = transform
        self.labeled = labeled
        self.shuffle_files = shuffle_files
        self.seed = seed

    def __iter__(self):
        files = self.tfrec_files
        if self.shuffle_files:
            g = torch.Generator()
            g.manual_seed(self.seed)
            idx = torch.randperm(len(files), generator=g).tolist()
            files = [files[i] for i in idx]

        dp = _build_tfrec_datapipe(files)
        for _, ex in dp:
            img_bytes = ex["image"][0]
            name_bytes = ex["image_name"][0]
            img = decode_jpeg(
                torch.frombuffer(img_bytes, dtype=torch.uint8), mode=ImageReadMode.RGB
            )
            img = _center_crop_600_tensor(img)

            if self.transform:
                img = self.transform(img)

            if self.labeled:
                y = (
                    int(ex["target"][0].decode("utf-8"))
                    if isinstance(ex["target"][0], (bytes, bytearray))
                    else int(ex["target"][0])
                )
                yield img, y
            else:
                image_id = (
                    name_bytes.decode("utf-8")
                    if isinstance(name_bytes, (bytes, bytearray))
                    else str(name_bytes)
                )
                yield img, image_id




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_55/1999388244.py in <cell line: 0>()
     11 
     12 # TorchVision TFRecord support is in torchdata (installed). Use it to stream records efficiently.
---> 13 from torchdata.datapipes.iter import FileLister, FileOpener
     14 from torchdata.datapipes.iter import TFRecordLoader  # type: ignore
     15 

ModuleNotFoundError: No module named 'torchdata.datapipes'

## === cell 2
if finetune_head and (not loaded_any_ckpt):
    train_dataset = CassavaTFRecordDataset(
        train_tfrec_dir,
        transform=weights_tfms,
        labeled=True,
        shuffle_files=True,
        seed=3407,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # IterableDataset can't use DataLoader shuffle; dataset handles shuffle_files.
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(
            4 if num_workers > 0 else None
        ),  # lower to reduce RAM pressure/overhead
        drop_last=False,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )

    for p in model.parameters():
        p.requires_grad = False
    for p in model.heads.head.parameters():
        p.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.heads.head.parameters(), lr=finetune_lr)

    model.train()
    for epoch in range(finetune_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            if device.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            running_loss += float(loss.item()) * bs
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == yb).sum().item())
            total += int(bs)

        print(
            f"[finetune head] epoch {epoch+1}/{finetune_epochs} "
            f"loss={running_loss/max(total,1):.4f} acc={correct/max(total,1):.4f}"
        )

    torch.save(model.state_dict(), "/kaggle/working/model.pth")
    model.eval()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1721583608.py in <cell line: 0>()
      3 if finetune_head and (not loaded_any_ckpt):
      4     # Create TFRecord iterable dataset (shuffling at file level keeps training stochasticity).
----> 5     train_dataset = CassavaTFRecordDataset(
      6         train_tfrec_dir,
      7         transform=weights_tfms,

NameError: name 'CassavaTFRecordDataset' is not defined

## === cell 3
if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

test_dataset = CassavaTFRecordDataset(
    test_tfrec_dir,
    transform=weights_tfms,
    labeled=False,
    shuffle_files=False,
    seed=3407,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)

_softmax_dim1 = lambda x: torch.softmax(x, dim=1)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3931369709.py in <cell line: 0>()
     10     ttas = None
     11 
---> 12 test_dataset = CassavaTFRecordDataset(
     13     test_tfrec_dir,
     14     transform=weights_tfms,

NameError: name 'CassavaTFRecordDataset' is not defined

## === cell 4
all_names = []
all_preds = []

model.eval()


def _seed_for_batch_view(batch_idx: int, view_j: int) -> int:
    return int((3407 + batch_idx * 1000003 + view_j * 9176) % (2**31 - 1))


_cpu_g = torch.Generator(device="cpu")
_cuda_g = torch.Generator(device="cuda") if device.type == "cuda" else None

with torch.inference_mode():
    for batch_idx, (inputs, filenames) in enumerate(test_loader):
        xb = inputs.to(device, non_blocking=True)  # (B,C,H,W)
        if device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)

        if tta:
            bsz, c, h, w = xb.shape
            n_tta = len(ttas)
            x_all = xb.new_empty((bsz * (n_tta + 1), c, h, w))
            x_all[0:bsz].copy_(xb)

            for j, t in enumerate(ttas):
                seed = _seed_for_batch_view(batch_idx, j)
                _cpu_g.manual_seed(seed)
                if _cuda_g is not None:
                    _cuda_g.manual_seed(seed)

                try:
                    x_aug = t(
                        xb, generator=_cuda_g if device.type == "cuda" else _cpu_g
                    )
                except TypeError:
                    torch.manual_seed(seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed(seed)
                    x_aug = t(xb)

                if device.type == "cuda":
                    x_aug = x_aug.contiguous(memory_format=torch.channels_last)
                start = (j + 1) * bsz
                x_all[start : start + bsz].copy_(x_aug)

            preds_flat = _softmax_dim1(model(x_all))  # (B*(T+1), num_classes)
            preds_bt = preds_flat.view(n_tta + 1, bsz, -1).transpose(0, 1)
            mean_preds = preds_bt.mean(dim=1)  # (B, num_classes)
            pred_labels = torch.argmax(mean_preds, dim=1).tolist()
        else:
            preds = _softmax_dim1(model(xb))
            pred_labels = torch.argmax(preds, dim=1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

print("Preds generated:", len(all_preds), "Unique files:", len(set(all_names)))



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1741080060.py in <cell line: 0>()
     15 
     16 with torch.inference_mode():
---> 17     for batch_idx, (inputs, filenames) in enumerate(test_loader):
     18         xb = inputs.to(device, non_blocking=True)  # (B,C,H,W)
     19         if device.type == "cuda":

NameError: name 'test_loader' is not defined

## === cell 5
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample = pd.read_csv(sample_path)

pred_ser = pd.Series(all_preds, index=pd.Index(all_names, name="image_id"))
pred_ser = pred_ser[~pred_ser.index.duplicated(keep="first")]

mapped = sample["image_id"].map(pred_ser)
sample["label"] = mapped.fillna(0).astype(int)

my_submission = sample[["image_id", "label"]]
my_submission.to_csv("submission.csv", index=False)

print("Saved submission.csv with shape:", my_submission.shape)
print("Any missing predictions filled with 0:", int(mapped.isna().sum()))



## === cell 6
my_submission.head()
