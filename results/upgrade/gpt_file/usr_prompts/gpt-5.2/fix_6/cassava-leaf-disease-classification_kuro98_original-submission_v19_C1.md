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

0.8845572680568148

# 6. Current score

0.2216

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.2216) has done: 'I fix the ViT input-size assertion by making the inference resize match the model’s expected `image_size` (224 for `vit_b_16`), which prevents the forward pass crash. Then I fix the submission NaN casting bug by filling missing mapped labels *before* converting to int, so a valid `submission.csv` is always written. I also make weight loading robust in case the saved checkpoint is a `state_dict` rather than a full model object (common Kaggle issue), without changing the model architecture or inference logic. These changes are execution- and submission-correctness focused and should improve score versus a broken pipeline by actually generating predictions.'
- What this solution (achieved 0.2216) has done: 'Your current score (0.2216) is far below the target (0.8846), so we should improve meaningfully while preserving the same ViT inference core. The biggest likely issue is that the checkpoint is not being loaded correctly: using `strict=False` can silently leave the randomly-initialized head (and possibly more) in place, which would produce near-random predictions and a very low accuracy. I keep the same `vit_b_16` model and inference flow, but I (1) robustly extract the correct state_dict (including common Lightning/EMA nesting), (2) verify how many keys are missing/unexpected, and (3) enforce that the classification head weights are actually loaded (otherwise fail fast instead of submitting junk). These changes are directly aimed at moving accuracy up toward the target without changing architecture/training.'
- What this solution (achieved 0.2216) has done: 'Your current score (0.2216) is far below the target (0.8846), and the most likely cause (given your inference-only notebook) is that you are not actually using the trained Cassava weights: your code points to `/kaggle/input/efficient-net/vit_cont_3.pt`, which almost certainly does not exist in this dataset, so you fall back to ImageNet ViT + random head (near-random accuracy). I make the smallest change that meaningfully moves accuracy upward: automatically locate a compatible `.pt/.pth` checkpoint inside `/kaggle/input/**` (including common Kaggle dataset mounts) and load it, while keeping the same `vit_b_16` architecture and inference loop. I also keep the “fail fast if head not loaded” behavior when a checkpoint is found, but avoid failing when none exists by clearly warning and still producing a valid submission. Finally, I set `img_size` to the model’s expected image size to avoid silent mismatch and ensure consistency with the checkpoint training resolution.'
- What this solution (achieved 0.2216) has done: 'Your score is far below the target, so the most likely issue is still that you’re effectively submitting near-random predictions (either no finetuned checkpoint is being used, or the loaded checkpoint doesn’t actually match the model). I keep the same ViT-B/16 inference pipeline, but make checkpoint selection/loading more deterministic and compatible with common training code: (1) handle `Lightning`-style `state_dict` keys (e.g., `model.*`) and heads saved as `head.*` (not only `heads.head.*`), (2) prefer checkpoints whose tensors clearly indicate a 5-class classifier head, and (3) print a clear one-line summary of what was loaded so you can confirm you’re not silently running ImageNet+random head. These are minimal changes focused on actually using the intended finetuned weights, which should move accuracy sharply upward toward your target without changing architecture or inference semantics.'

# 9. Code solution

## === cell 0
import os
import glob

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
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

img_size = 224

batch_size = 16
num_workers = 4
num_classes = 5
tta = False

weight_path = "/kaggle/input/efficient-net/vit_cont_3.pt"
model = None


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
            "ema",
            "ema_state_dict",
        ):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return ckpt_obj
    return None


def _strip_prefixes(state):
    prefixes = ("module.", "model.", "net.", "backbone.", "encoder.")
    out = {}
    for k, v in state.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p) :]
                    changed = True
        out[nk] = v
    return out


def _state_dict_has_5class_head(state):
    cand_keys = [
        "heads.head.weight",
        "heads.head.bias",
        "head.weight",
        "head.bias",
        "classifier.weight",
        "classifier.bias",
        "fc.weight",
        "fc.bias",
    ]
    for wk in cand_keys:
        if wk in state and isinstance(state[wk], torch.Tensor):
            w = state[wk]
            if w.ndim == 2 and w.shape[0] == 5:
                return True
    for bk in cand_keys:
        if bk in state and isinstance(state[bk], torch.Tensor):
            b = state[bk]
            if b.ndim == 1 and b.shape[0] == 5:
                return True
    return False


def _find_candidate_checkpoints(search_root="/kaggle/input"):
    patterns = [
        os.path.join(search_root, "**", "*.pt"),
        os.path.join(search_root, "**", "*.pth"),
        os.path.join(search_root, "**", "*.bin"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))

    files = [f for f in files if os.path.isfile(f) and os.path.getsize(f) > 1_000_000]

    def score_path(p):
        name = os.path.basename(p).lower()
        s = 0
        for token, w in [
            ("cassava", 50),
            ("leaf", 20),
            ("disease", 20),
            ("vit", 30),
            ("transformer", 10),
            ("b16", 10),
            ("finetune", 15),
            ("fine", 5),
            ("best", 10),
            ("final", 5),
            ("fold", 3),
        ]:
            if token in name:
                s += w
        depth = p.count(os.sep)
        s -= depth * 0.2
        s += min(os.path.getsize(p) / 1e8, 10)
        return s

    files = sorted(files, key=score_path, reverse=True)
    return files


def _load_checkpoint_into_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, torch.nn.Module):
        print(f"Loaded full model object from checkpoint: {ckpt_path}")
        return ckpt

    state = _extract_state_dict(ckpt)
    if state is None:
        raise TypeError(
            f"Could not extract a state_dict from checkpoint at {ckpt_path}. Type={type(ckpt)}"
        )

    state = _strip_prefixes(state)

    incompatible = model.load_state_dict(state, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)

    head_weight_keys = (
        "heads.head.weight",
        "head.weight",
        "classifier.weight",
        "fc.weight",
    )
    head_bias_keys = (
        "heads.head.bias",
        "head.bias",
        "classifier.bias",
        "fc.bias",
    )
    head_loaded = any(k in state for k in head_weight_keys) or any(
        k in state for k in head_bias_keys
    )

    if _state_dict_has_5class_head(state) and not head_loaded:
        raise RuntimeError(
            "Checkpoint appears to contain a 5-class classifier, but head keys were not recognized/loaded. "
            f"Refusing checkpoint: {ckpt_path}"
        )

    print(
        f"Checkpoint load summary for {os.path.basename(ckpt_path)}: "
        f"missing_keys={len(missing)} unexpected_keys={len(unexpected)} head_keys_present={head_loaded}"
    )
    return model


import torchvision

model = torchvision.models.vit_b_16(
    weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
)
in_features = model.heads.head.in_features
model.heads.head = torch.nn.Linear(in_features, num_classes)

loaded = False

candidate_paths = []
if os.path.exists(weight_path):
    candidate_paths = [weight_path]
else:
    candidate_paths = _find_candidate_checkpoints("/kaggle/input")
    print(
        f"Did not find weight_path={weight_path}. Auto-discovered {len(candidate_paths)} candidate checkpoints."
    )

scored = []
for p in candidate_paths[:200]:
    try:
        ckpt = torch.load(p, map_location="cpu")
        state = _extract_state_dict(ckpt)
        if state is None:
            continue
        state = _strip_prefixes(state)
        scored.append((1 if _state_dict_has_5class_head(state) else 0, p))
    except Exception:
        continue
scored.sort(key=lambda x: x[0], reverse=True)
candidate_paths = [p for _, p in scored] + [
    p for p in candidate_paths if p not in set([pp for _, pp in scored])
]

for p in candidate_paths[:25]:  # cap attempts for time safety
    try:
        model = _load_checkpoint_into_model(model, p)
        loaded = True
        print(f"Using checkpoint: {p}")
        break
    except Exception:
        continue

if not loaded:
    print(
        "WARNING: No compatible finetuned checkpoint could be loaded. "
        "Proceeding with ImageNet-initialized ViT + random 5-class head (will likely score poorly)."
    )

model.to(device)




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for Cassava test images, driven by sample_submission image_id order."""

    def __init__(self, data_dir, image_ids, transform=None, ttas=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.images = list(image_ids)
        self.ttas = ttas

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        if self.ttas is not None and self.transform is not None:
            img = [self.transform(t(img)) for t in self.ttas]
        elif self.transform:
            img = self.transform(img)

        return img, filename

    def __len__(self):
        return len(self.images)




## === cell 2
model_img_size = int(getattr(model, "image_size", 224))

test_transforms = v2.Compose(
    [
        v2.Resize(
            (model_img_size, model_img_size), interpolation=InterpolationMode.BICUBIC
        ),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].tolist()

test_dataset = CassavaDataset(
    test_dir, image_ids=test_image_ids, transform=test_transforms, ttas=ttas
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,  # important for deterministic filename/pred alignment
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 3
all_names = []
all_preds = []

model.eval()

with torch.no_grad():
    for batch_idx, (inputs, filenames) in enumerate(test_loader):
        if tta:
            inputs = torch.cat(inputs, dim=0).to(device)
            filenames = list(filenames)

            preds = normalizer(model(inputs))

            batch_preds = torch.stack(torch.split(preds, len(filenames)), dim=0)
            mean_preds = torch.mean(batch_preds, dim=0)
            pred_labels = torch.argmax(mean_preds, 1).tolist()
        else:
            inputs = inputs.to(device)
            filenames = list(filenames)

            preds = normalizer(model(inputs))
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)



## === cell 4
pred_map = dict(zip(all_names, all_preds))
my_submission = sample_sub.copy()

mapped = my_submission["image_id"].map(pred_map)
my_submission["label"] = mapped.fillna(0).astype("int64")

assert len(my_submission) == len(
    sample_sub
), "Submission length mismatch vs sample_submission."
assert list(my_submission.columns) == [
    "image_id",
    "label",
], "Submission columns must be image_id,label."

my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)



## === cell 5
my_submission.head()
