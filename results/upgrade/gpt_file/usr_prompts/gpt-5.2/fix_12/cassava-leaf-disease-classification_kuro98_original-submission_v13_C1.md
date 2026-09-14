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

0.8948322756119673

# 6. Current score

0.72496

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the missing model file error by adding a robust fallback that loads a built-in torchvision ViT when the external checkpoint path doesn’t exist, so the notebook runs end-to-end and produces predictions. I also fix the invalid submission length issue by ensuring the DataLoader does not shuffle and by aligning/sorting predictions to exactly match `sample_submission.csv`’s `image_id` ordering (this also prevents duplicates/missing IDs). Finally, I make test-time augmentation deterministic (so it’s reproducible under the fixed seed) and remove a large debug print that can slow/timeout inference, without changing the core inference semantics (still TTA with mean probabilities and argmax). The result always write a valid `submission.csv` with the required columns and correct row count.'
- What this solution (achieved 0.05531) has done: 'I fix the runtime error by ensuring the fallback torchvision ViT model is created with `image_size=384` to match your 384×384 preprocessing (the checkpointed model likely expected 384, while the default torchvision ViT expects 224). I also make the test image listing deterministic (sorted) so filenames/predictions are stable and aligned, without changing inference semantics. Finally, I keep your submission alignment via `sample_submission.csv` unchanged so it always outputs a valid `submission.csv` with the correct rows/columns; these changes should also materially improve accuracy versus the broken 224/384 mismatch.'
- What this solution (achieved 0.13117) has done: 'I fix the immediate runtime error caused by trying to override `image_size` while also loading pretrained `ViT_B_16_Weights` (torchvision enforces 224 for that weights enum). To preserve your core ViT approach and 384×384 preprocessing, I switch the fallback to `weights=None` when `image_size=384`, which avoids the exception and ensures `vit_model` is always created (preventing the downstream `NoneType` error). I also make the TTA transforms deterministic per-sample by seeding inside `__getitem__`, keeping the same TTA/mean-prob/argmax semantics but stabilizing predictions and typically improving accuracy vs fully random TTA. The rest of the pipeline (dataset, TTA averaging, and submission alignment to `sample_submission.csv`) remains unchanged.'
- What this solution (achieved 0.13117) has done: 'I fix the crash by filtering the test directory listing so the dataset only includes actual image files (the provided directory contains a nested `test_images/` folder that was being treated like an image). I also make the checkpoint loading more robust (load either a full model object or a state_dict into the same ViT architecture) so the intended weights are used when available, which should move accuracy up toward the target without changing the core ViT+TTA inference semantics. Finally, I keep your submission alignment to `sample_submission.csv` unchanged and ensure the script always writes a valid `submission.csv`.'
- What this solution (achieved 0.17339) has done: 'Your score is far below the target, so the smallest reliable way to move accuracy upward is to make sure inference uses trained weights rather than an untrained ViT fallback. I (1) make checkpoint loading robust to common key mismatches (especially `heads.head.*` vs `head.*` vs `classifier.*`) and enforce `strict=True` when possible so we don’t silently run with random weights, and (2) if the provided checkpoint can’t be used, switch the fallback to a pretrained ViT at its native 224px and adjust the preprocessing accordingly (same ViT core logic, just avoiding the current “random weights at 384px” failure mode). I also remove the submission-time `groupby` (it can hide bugs) and instead assert one prediction per sample and align exactly to `sample_submission.csv`. These changes keep your architecture/inference approach (ViT + optional TTA + mean-prob + argmax) intact while addressing the main reason for the low accuracy.'
- What this solution (achieved 0.22833) has done: 'Your score is far below the target, so we should move accuracy up by ensuring the model actually uses strong, correctly-normalized pretrained weights during inference. The smallest reliable fix is to always build the ViT using the official torchvision weights and their matching preprocessing (mean/std + resize/crop), while keeping your same ViT-B/16 core model and the same “TTA → mean probabilities → argmax” inference semantics. I also make the TTA transforms operate on tensors (after base preprocessing) so they don’t fight the weights’ expected preprocessing pipeline, and I keep your submission alignment to `sample_submission.csv` intact. These changes should substantially increase accuracy versus the current frequent “random-head / mismatched preprocessing” behavior, moving you toward the target.'
- What this solution (achieved 0.61136) has done: 'Your current score is far below the target, so we should push accuracy upward with the smallest safe change: use the pretrained ViT weights exactly as intended at inference time. The main issue is that you replace the pretrained classifier head with a random 5-class head and then often fail to load your finetuned checkpoint, which makes predictions nearly random; we instead keep the pretrained head unless the checkpoint loads successfully. If the checkpoint loads (strictly), we keep your 5-class head and use it; if it doesn’t, we fall back to the pretrained 1000-class head and map it to 5 classes via a tiny calibration layer fit on the provided `train.csv` (no new model/architecture change; just a linear mapping trained quickly). This preserves your core ViT + TTA + mean-prob + argmax inference semantics, stays deterministic, and should move the score much closer to the target.'
- What this solution (achieved 0.05531) has done: 'Your score gap to the target is large and you’re currently in the “no checkpoint loaded” path, so the biggest accuracy limiter is the weak pretrained→5 mapper fitting (it trains on only the first 4096 rows and uses the wrong features: softmaxed probabilities). I keep the exact same ViT+TTA+mean-prob+argmax core inference logic, but improve the mapper fitting by (1) fitting on a larger, class-balanced subset of train.csv and (2) training the linear mapper on raw logits (better linear separability than probabilities) while keeping the same linear head structure. I also fix a subtle TTA batching issue by providing a `collate_fn` that stacks the per-sample TTA tensors into the expected `[t*b, C, H, W]` format deterministically, avoiding shape surprises across PyTorch versions. These minimal changes should move accuracy upward toward the target without changing your model architecture or inference semantics.'
- What this solution (achieved 0.05531) has done: 'I fix the mapper-fitting path so it works with your existing `collate_fn`: in the fit loader (where `ttas=None`), the batch is already a tensor, but your code currently breaks because it sometimes receives a list and then calls `.to()` on it. I make the fit loop robust by handling both tensor and list-of-TTA inputs (without changing the ViT/TTA/mean-prob/argmax core logic). This ensure `mapper` is always trained and non-None when `needs_mapper=True`, which also resolves the downstream `NoneType` error in inference. The rest of the pipeline (model construction/loading, test inference, and submission alignment to `sample_submission.csv`) remains unchanged.'
- What this solution (achieved 0.7201) has done: 'The crash in mapper fitting happens because the fit DataLoader uses the same collate as TTA mode, so it returns a list even when `ttas=None`, and then the code mistakenly indexes into the batch and ends up feeding a 3D tensor (C,H,W) to ViT. I fix this by making `cassava_collate` decide based on whether the batch items are actually lists (i.e., TTA present) rather than the global `tta` flag, keeping your exact ViT+TTA inference semantics unchanged. I also add a hard guard that if `needs_mapper=True` then `mapper` must be trained, so we never silently proceed to inference with `mapper=None`. These changes are score-improving because they enable the intended pretrained->5 mapper path (instead of crashing / falling back to fillna(0) behavior), while preserving your modeling approach.'
- What this solution (achieved 0.72496) has done: 'Your current score (0.7201) is below the target (0.8948), so we should make a small change that reliably increases accuracy without changing the model/training core. The biggest issue is that when `needs_mapper=True`, you train the mapper on raw logits from the 1000-way pretrained head, but at inference you feed it the *mean of per-TTA logits*; this is fine, yet you still compute `mean_probs` from softmax and ignore it, and the mapper is trained without any feature normalization, which can make it poorly conditioned. I keep the exact same ViT + (optional) TTA + mean-aggregation + argmax semantics, but (1) train the mapper on the same aggregation behavior used at inference (use a small fixed-TTA=1 path for train fitting to match distribution) and (2) add a lightweight logits standardization (per-feature mean/std computed on the fit set) applied consistently in both mapper training and inference; this is a minimal calibration step that typically moves accuracy upward toward your target while preserving the core approach. I also add a strict assertion that `t == len(inputs_list)` in TTA to avoid silent shape mismatches.'

# 9. Code solution

## === cell 0
import os

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

batch_size = 16
num_workers = 4
num_classes = 5
tta = True

ckpt_path = "/kaggle/input/vit-v1-update/vit_v1_1.pt"

from torchvision.models import vit_b_16, ViT_B_16_Weights


def _build_vit(image_size: int, weights, replace_head: bool):
    m = vit_b_16(weights=weights, image_size=image_size)
    if replace_head:
        in_features = m.heads.head.in_features
        m.heads.head = torch.nn.Linear(in_features, num_classes)
    return m


weights = ViT_B_16_Weights.DEFAULT
img_size = 224

vit_model = _build_vit(image_size=img_size, weights=weights, replace_head=False)

ckpt_loaded = False
if os.path.exists(ckpt_path):
    obj = torch.load(ckpt_path, map_location="cpu")
    try:
        if isinstance(obj, torch.nn.Module):
            vit_model = obj
            ckpt_loaded = True
        elif isinstance(obj, dict):
            state_dict = obj.get("state_dict", obj)

            cleaned = {}
            for k, v in state_dict.items():
                nk = k
                if nk.startswith("module."):
                    nk = nk[len("module.") :]
                if nk.startswith("model."):
                    nk = nk[len("model.") :]

                if nk.startswith("head."):
                    nk = "heads.head." + nk[len("head.") :]
                if nk.startswith("classifier."):
                    nk = "heads.head." + nk[len("classifier.") :]
                if nk.startswith("fc."):
                    nk = "heads.head." + nk[len("fc.") :]

                cleaned[nk] = v

            if any(k.startswith("heads.head.") for k in cleaned.keys()):
                vit_model = _build_vit(
                    image_size=img_size, weights=weights, replace_head=True
                )

            try:
                vit_model.load_state_dict(cleaned, strict=True)
                ckpt_loaded = True
                print("Loaded checkpoint with strict=True")
            except Exception as e_strict:
                print(
                    "Warning: checkpoint incompatible with strict=True; will use pretrained backbone+head. Error:",
                    repr(e_strict),
                )
                ckpt_loaded = False
    except Exception as e:
        print(
            "Warning: failed to load checkpoint object; will use pretrained backbone+head. Error:",
            repr(e),
        )
        ckpt_loaded = False

vit_model.to(device)




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava images.
    Returns (image or list-of-tta-images, filename[, label]).
    """

    def __init__(self, data_dir, transform=None, ttas=None, df=None, labeled=False):
        super().__init__(root=data_dir)
        self.transform = transform
        self.ttas = ttas
        self.base_seed = 3407
        self.labeled = labeled

        if df is not None:
            self.images = df["image_id"].tolist()
            if labeled:
                self.labels = df["label"].astype(int).tolist()
            else:
                self.labels = None
        else:
            exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            files = []
            for name in os.listdir(data_dir):
                full = os.path.join(data_dir, name)
                if os.path.isfile(full) and name.lower().endswith(exts):
                    files.append(name)
            self.images = sorted(files)
            self.labels = None

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        img = self.transform(img) if self.transform else img

        if self.ttas is not None:
            out = []
            for j, t in enumerate(self.ttas):
                torch.manual_seed(self.base_seed + idx * 1000 + j)
                torch.cuda.manual_seed(self.base_seed + idx * 1000 + j)
                out.append(t(img))
            img = out

        if self.labeled:
            return img, filename, int(self.labels[idx])
        return img, filename

    def __len__(self):
        return len(self.images)




## === cell 2
base_transforms = weights.transforms(antialias=True)

if tta:
    ttas = [
        v2.RandomHorizontalFlip(p=1.0),
        v2.RandomRotation(15, interpolation=InterpolationMode.BILINEAR),
        v2.RandomAffine(
            degrees=0,
            translate=(0.06, 0.06),
            scale=(0.95, 1.05),
            shear=None,
            interpolation=InterpolationMode.BILINEAR,
        ),
    ]
else:
    ttas = None


def cassava_collate(batch):
    labeled = len(batch[0]) == 3
    first_img = batch[0][0]
    has_tta = isinstance(first_img, (list, tuple))

    if not has_tta:
        if labeled:
            imgs, fns, labels = zip(*batch)
            return (
                torch.stack(list(imgs), dim=0),
                list(fns),
                torch.tensor(labels, dtype=torch.long),
            )
        imgs, fns = zip(*batch)
        return torch.stack(list(imgs), dim=0), list(fns)

    if labeled:
        imgs_list, fns, labels = zip(*batch)
    else:
        imgs_list, fns = zip(*batch)

    t = len(imgs_list[0])
    per_t = []
    for j in range(t):
        per_t.append(
            torch.stack([imgs_list[i][j] for i in range(len(imgs_list))], dim=0)
        )
    if labeled:
        return per_t, list(fns), torch.tensor(labels, dtype=torch.long)
    return per_t, list(fns)


test_dataset = CassavaDataset(test_dir, transform=base_transforms, ttas=ttas)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=cassava_collate,
)

needs_mapper = not ckpt_loaded
mapper = None  # torch.nn.Linear(1000 -> 5) set in next cell if needed.

mapper_mu = None
mapper_sigma = None

normalizer = torch.nn.Softmax(dim=1)



## === cell 3
if needs_mapper:
    train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
    train_df = pd.read_csv(train_csv_path)

    max_fit = 12000  # keep runtime reasonable; much stronger than 4096.
    per_class = max_fit // num_classes

    fit_parts = []
    for c in range(num_classes):
        part = train_df[train_df["label"] == c].head(per_class)
        fit_parts.append(part)
    fit_df = pd.concat(fit_parts, axis=0, ignore_index=True)

    if len(fit_df) < max_fit:
        remaining = max_fit - len(fit_df)
        used = set(fit_df["image_id"].tolist())
        topup = train_df[~train_df["image_id"].isin(used)].head(remaining)
        fit_df = pd.concat([fit_df, topup], axis=0, ignore_index=True)

    fit_dataset = CassavaDataset(
        "/kaggle/input/cassava-leaf-disease-classification/train_images/",
        transform=base_transforms,
        ttas=None,
        df=fit_df,
        labeled=True,
    )
    fit_loader = DataLoader(
        fit_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=cassava_collate,
    )

    vit_model.eval()
    X_list = []
    y_list = []
    with torch.no_grad():
        for imgs, _, labels in fit_loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = vit_model(imgs)  # [B, 1000] for pretrained head
            feats = logits.float().cpu()
            X_list.append(feats)
            y_list.append(labels.cpu())

    X = torch.cat(X_list, dim=0)  # [N, 1000]
    y = torch.cat(y_list, dim=0).long()  # [N]

    mapper_mu = X.mean(dim=0, keepdim=True)
    mapper_sigma = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xn = (X - mapper_mu) / mapper_sigma

    mapper = torch.nn.Linear(Xn.shape[1], num_classes, bias=True)
    mapper.to(device)

    opt = torch.optim.AdamW(mapper.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    Xd = Xn.to(device)
    yd = y.to(device)

    mapper.train()
    for _ in range(25):
        opt.zero_grad(set_to_none=True)
        out = mapper(Xd)
        loss = loss_fn(out, yd)
        loss.backward()
        opt.step()

    mapper.eval()
    print(
        f"Fitted pretrained->5 mapper on {len(fit_dataset)} samples (balanced subset) with logits standardization"
    )
else:
    print("Checkpoint loaded; no mapper needed")

if needs_mapper and mapper is None:
    raise RuntimeError(
        "needs_mapper=True but mapper was not fit/created; cannot proceed."
    )



## === cell 4
all_names = []
all_preds = []

vit_model.eval()

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if tta:
            inputs_list, filenames = batch
            filenames = list(filenames)

            assert isinstance(inputs_list, (list, tuple)), type(inputs_list)
            assert len(inputs_list) == len(ttas), (len(inputs_list), len(ttas))

            inputs = torch.cat(list(inputs_list), dim=0).to(device, non_blocking=True)

            logits = vit_model(inputs)
            probs = normalizer(logits)

            b = len(filenames)
            t = len(ttas)

            if needs_mapper:
                mean_logits = logits.view(t, b, -1).mean(dim=0)
                mu = mapper_mu.to(device)
                sigma = mapper_sigma.to(device)
                mean_logits_n = (mean_logits.float() - mu) / sigma
                mapped_logits = mapper(mean_logits_n)
                pred_labels = torch.argmax(mapped_logits, dim=1).tolist()
            else:
                batch_probs = probs.view(t, b, -1)
                mean_probs = batch_probs.mean(dim=0)
                pred_labels = torch.argmax(mean_probs, dim=1).tolist()
        else:
            inputs, filenames = batch
            filenames = list(filenames)

            inputs = inputs.to(device, non_blocking=True)
            logits = vit_model(inputs)

            if needs_mapper:
                mu = mapper_mu.to(device)
                sigma = mapper_sigma.to(device)
                logits_n = (logits.float() - mu) / sigma
                mapped_logits = mapper(logits_n)
                pred_labels = torch.argmax(mapped_logits, dim=1).tolist()
            else:
                probs = normalizer(logits)
                pred_labels = torch.argmax(probs, dim=1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

assert len(all_names) == len(test_dataset), (len(all_names), len(test_dataset))
assert len(all_preds) == len(test_dataset), (len(all_preds), len(test_dataset))



## === cell 5
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample = pd.read_csv(sample_path)

pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})

if pred_df["image_id"].duplicated().any():
    pred_df = pred_df.drop_duplicates("image_id", keep="first")

submission = sample[["image_id"]].merge(pred_df, on="image_id", how="left")
submission["label"] = submission["label"].fillna(0).astype(int)

submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## === cell 6
submission
