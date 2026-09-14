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

0.8715624055605923

# 6. Current score

0.11024

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11211) has done: 'I fix the missing `efficientnet_pytorch` import by removing the unavailable wheel install and instead using `torchvision`’s built-in EfficientNet-B4 with a 5-class classifier head, which preserves the same core model family and allows the provided `.pth` weights to load (with key/shape handling). I update the Albumentations `RandomResizedCrop` call to the new v2 API (expects `size=(h,w)`) and replace deprecated transforms (`Cutout`) with `CoarseDropout` only so augmentation runs. I also fix the test-time augmentation loop bug (it loops 5 times but divides by 10) and ensure tensors are correctly normalized and moved to the right device. Finally, I guarantee a valid `submission.csv` with the exact required columns is written.'
- What this solution (achieved 0.11024) has done: 'I fix the path/IO bug that prevents the model weights from being found by searching for the `.pth` file inside the provided `/kaggle/input` directory structure and falling back safely if it’s not present. Then I correct the preprocessing bug: you currently `A.Normalize()` in Albumentations and then apply `transforms.ToTensor()` which re-scales values again, badly breaking inference and causing the very low score; I replace this with a single consistent “to tensor + normalize” step that matches the EfficientNet expectations. I also switch TTA to deterministic test-time transforms (flips/transpose) only—keeping the same TTA averaging approach but removing training-only random crops/rotations that can harm accuracy at inference. Finally, I ensure `submission.csv` is written with the exact required columns and row order from `sample_submission.csv`.'
- What this solution (achieved 0.11024) has done: 'Your current score is far below the target, so the smallest likely win is to make inference preprocessing match what the EfficientNet-B4 checkpoint expects. I (1) add the missing EfficientNet resize/crop step (B4 typically expects a centered 380×380 crop) before normalization, (2) ensure the transpose-based TTA variants produce a valid square input by resizing after the geometric transform, and (3) speed up/clean inference slightly (batch the TTA stack per image) without changing the model or the TTA averaging logic. These changes keep the same architecture, checkpoint loading, loss/eval semantics, and produce the same submission format, but should materially improve accuracy if the checkpoint was trained with standard EfficientNet sizing.'
- What this solution (achieved 0.11024) has done: 'Your score indicates the model is effectively guessing, which is most consistent with loading the wrong checkpoint (or none) even though inference preprocessing is now sane. I make the checkpoint selection stricter by preferring files that (a) contain “b4/efficientnet/cassava”, (b) are reasonably sized (to avoid tiny optimizer-only stubs), and (c) actually match the model’s tensor shapes, by trying the top candidates and selecting the first that loads cleanly. This keeps the same EfficientNet-B4 architecture and inference/TTA logic, but increases the chance you’re actually using the intended trained weights, which should move accuracy toward your target. Everything still runs end-to-end and writes a valid `submission.csv` in the required format.'

# 9. Code solution

## === cell 0
import os
import warnings
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models

warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



## === cell 1
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

assert os.path.exists(
    sample_sub_path
), f"Missing sample submission at: {sample_sub_path}"
assert os.path.isdir(
    test_images_path
), f"Missing test images dir at: {test_images_path}"

preferred_model_path = "../input/en-b4-tta-calr-15-v2/model(13).pth"


def find_checkpoint_candidates():
    roots = ["../input", "/kaggle/input"]
    candidates = []

    if os.path.exists(preferred_model_path):
        candidates.append(preferred_model_path)

    for r in roots:
        rp = Path(r)
        if rp.exists():
            candidates.extend([str(p) for p in rp.rglob("*.pth")])
            candidates.extend([str(p) for p in rp.rglob("*.pt")])

    seen = set()
    uniq = []
    for p in candidates:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def checkpoint_rank_score(p: str) -> float:
    name = Path(p).name.lower()
    parent = str(Path(p).parent).lower()
    s = 0.0

    for key, val in [
        ("efficientnet", 8),
        ("tf_efficientnet", 8),
        ("b4", 7),
        ("enb4", 7),
        ("cassava", 6),
        ("leaf", 2),
        ("disease", 2),
        ("tta", 2),
        ("fold", 1),
        ("best", 1),
        ("model", 1),
    ]:
        if key in name or key in parent:
            s += val

    try:
        sz = os.path.getsize(p)
        if sz < 1_000_000:
            s -= 50
        elif sz < 10_000_000:
            s -= 10
        elif sz > 40_000_000:
            s += 5
    except OSError:
        s -= 5

    if name.endswith(".pth"):
        s += 1

    return s


candidates = find_checkpoint_candidates()
candidates = sorted(candidates, key=checkpoint_rank_score, reverse=True)
print(f"Found {len(candidates)} checkpoint candidates (top 10 shown):")
for p in candidates[:10]:
    try:
        print(
            "  ",
            p,
            "| size(MB)=",
            round(os.path.getsize(p) / 1024 / 1024, 2),
            "| score=",
            checkpoint_rank_score(p),
        )
    except OSError:
        print("  ", p, "| size(MB)=?", "| score=", checkpoint_rank_score(p))



## === cell 2
model = models.efficientnet_b4(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 5)
model = model.to(device)


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        sd = ckpt_obj.get(
            "state_dict",
            ckpt_obj.get("model_state_dict", ckpt_obj.get("model", ckpt_obj)),
        )
    else:
        sd = ckpt_obj
    if not isinstance(sd, dict):
        return None
    cleaned = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    return cleaned


loaded = False
selected_model_path = None
last_err = None

for p in candidates[:30]:  # limit to keep runtime bounded
    try:
        if not os.path.exists(p):
            continue
        ckpt = torch.load(p, map_location="cpu")
        state_dict = _extract_state_dict(ckpt)
        if state_dict is None:
            continue

        try:
            model.load_state_dict(state_dict, strict=True)
            loaded = True
            selected_model_path = p
            print("Loaded checkpoint with strict=True:", p)
            break
        except RuntimeError as e_strict:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            cls_w_key = "classifier.1.weight"
            cls_b_key = "classifier.1.bias"
            ok_cls = (
                cls_w_key in state_dict
                and cls_b_key in state_dict
                and tuple(state_dict[cls_w_key].shape)
                == tuple(model.state_dict()[cls_w_key].shape)
                and tuple(state_dict[cls_b_key].shape)
                == tuple(model.state_dict()[cls_b_key].shape)
            )

            total_params = len(model.state_dict())
            miss_ratio = len(missing) / max(1, total_params)

            if ok_cls and miss_ratio <= 0.05:
                loaded = True
                selected_model_path = p
                print("Loaded checkpoint with strict=False (accepted):", p)
                print(
                    "  Missing keys:",
                    len(missing),
                    "Unexpected keys:",
                    len(unexpected),
                    "Missing ratio:",
                    round(miss_ratio, 4),
                )
                break
            else:
                last_err = e_strict
                continue

    except Exception as e:
        last_err = e
        continue

if not loaded:
    print(
        "WARNING: No compatible checkpoint found; running with randomly initialized model (submission will be valid but low score)."
    )
    if last_err is not None:
        print("Last checkpoint load error:", repr(last_err))
else:
    print("Checkpoint selected:", selected_model_path)

model.eval()



## === cell 3
B4_SIZE = 380
tta_aug = A.Compose(
    [
        A.ToFloat(max_value=255.0),
        A.Resize(B4_SIZE, B4_SIZE, interpolation=1),
        A.CenterCrop(B4_SIZE, B4_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.transforms.ToTensorV2(),
    ]
)


def apply_tta_variant(img_np, variant_id: int):
    if variant_id == 0:
        x = img_np
    elif variant_id == 1:
        x = np.ascontiguousarray(np.flip(img_np, axis=1))  # H-flip
    elif variant_id == 2:
        x = np.ascontiguousarray(np.flip(img_np, axis=0))  # V-flip
    elif variant_id == 3:
        x = np.ascontiguousarray(np.transpose(img_np, (1, 0, 2)))  # transpose
    elif variant_id == 4:
        x = np.ascontiguousarray(
            np.flip(np.transpose(img_np, (1, 0, 2)), axis=1)
        )  # transpose + H-flip
    else:
        x = img_np
    return x




## === cell 4
sample_sub = pd.read_csv(sample_sub_path)

predictions = []
tta_n = 5  # keep consistent with variants and averaging

with torch.no_grad():
    for _, sample_row in sample_sub.iterrows():
        img_path = os.path.join(test_images_path, sample_row.image_id)

        image_pil = Image.open(img_path).convert("RGB")
        image_np = np.array(image_pil)

        xs = []
        for v in range(tta_n):
            img_v = apply_tta_variant(image_np, v)
            xs.append(tta_aug(image=img_v)["image"])
        x = torch.stack(xs, dim=0).to(device)  # tta_n x C x H x W

        out = model(x)  # tta_n x 5 logits
        logits_avg = out.mean(dim=0, keepdim=True)  # 1 x 5
        pred_label = int(torch.argmax(logits_avg, dim=1).item())
        predictions.append([sample_row.image_id, pred_label])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
sub_df.to_csv("submission.csv", index=False)

print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape, "| weights_loaded:", loaded)
assert sub_df.shape[0] == sample_sub.shape[0]
assert list(sub_df.columns) == ["image_id", "label"]
assert os.path.exists("submission.csv")
