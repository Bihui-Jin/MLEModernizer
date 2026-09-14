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

0.8203384708371109

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the two blockers preventing an end-to-end run: (1) make the ViT weights loading robust to missing external weight files by falling back to timm’s built-in pretrained weights (same architecture, same inference semantics), and (2) remove unconditional `.cuda()` calls by selecting CPU vs GPU at runtime so it works in Kaggle notebook environments without a GPU driver. I also fix the test dataloader ordering/alignment by using `sample_submission.csv` image_id order and disabling shuffle, ensuring the submission exactly matches the required format and ordering. Finally, I replace the deprecated/slow `DataFrame.append` in the inference loop with list accumulation to avoid runtime/performance issues while keeping predictions identical.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) is far below the target (0.82034), which strongly suggests the finetuned checkpoint is not actually being loaded (or is mismatched), so you’re effectively submitting near-random predictions from a generic ImageNet-pretrained ViT head. I make the checkpoint loading robust to common Kaggle checkpoint formats (e.g., `state_dict`, `model`, `module.` prefixes) while keeping the exact same model architecture and inference logic. I also add a quick sanity print of missing/unexpected keys to confirm whether weights were applied, and keep the submission ordering exactly aligned to `sample_submission.csv`. These minimal fixes should move accuracy sharply upward toward the target without changing the model or evaluation semantics.'
- What this solution (achieved 0.11584) has done: 'Your current score is so far below the target that the most likely cause is the finetuned checkpoint isn’t actually being applied to the model (so predictions are essentially random). I keep your model and inference logic intact, but make checkpoint loading more robust specifically for common “timm saved model” cases (e.g., keys prefixed with `model.model.` and classifier saved as `fc.*` instead of `head.*`), and I verify the loaded head weight shapes to ensure the 5-class classifier is restored. I also keep the submission ordering driven by `sample_submission.csv` exactly as you already do. These are minimal changes aimed directly at moving accuracy sharply upward toward the target without altering architecture or evaluation semantics.'
- What this solution (achieved 0.11584) has done: 'Your current score is far below the target, so the smallest likely fix is ensuring the finetuned checkpoint truly matches your ViT head and is actually being loaded (not silently skipped by `strict=False`). I keep your model and inference identical, but make the checkpoint loader “strict when possible”: it (1) adapt common key patterns, (2) explicitly validate that `head.weight/head.bias` exist and match shape `(5, in_features)`, and (3) fail over to a second mapping attempt for other frequent wrappers. I also switch the test DataLoader to a larger batch size (no semantic change) to keep runtime comfortably under limits while producing the same predictions. The submission order/format remains driven by `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import time
import random
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt



## === cell 1
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



## === cell 2
BASE1 = "../input/cassava-leaf-disease-classification"
BASE2 = "/kaggle/input/cassava-leaf-disease-classification"

data_path = BASE1 if os.path.exists(BASE1) else BASE2
train_path = os.path.join(data_path, "train_images") + os.sep
test_path = os.path.join(data_path, "test_images") + os.sep

model_path = "../input/vitbase16224/jx_vit_base_p16_224-80ecf9dd.pth"
Cassava_model = (
    "../input/cassavaaugmtp98epochs4lr175/CassavaViT_Augm_TP98_Epochs4_LR1-75e05.pt"
)

print("data_path:", data_path)
print("test_path exists:", os.path.exists(test_path))
print("Cassava finetuned checkpoint exists:", os.path.exists(Cassava_model))
print("Base ViT weights path exists:", os.path.exists(model_path))



## === cell 3
import timm

print("timm version:", getattr(timm, "__version__", "unknown"))



## === cell 4
print("Available ViT Models (subset):")
print([m for m in timm.list_models("vit*")][:10], "...")




## === cell 5
class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        super(ViTBase16, self).__init__()

        self.model = timm.create_model("vit_base_patch16_224", pretrained=False)

        if pretrained:
            if os.path.exists(model_path):
                state = torch.load(model_path, map_location="cpu")
                self.model.load_state_dict(state, strict=True)
                print(f"Loaded base ViT weights from: {model_path}")
            else:
                self.model = timm.create_model("vit_base_patch16_224", pretrained=True)
                print(
                    "External base ViT weights not found; using timm pretrained=True weights instead."
                )

        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        return self.model(x)




## === cell 6
def _extract_state_dict(ckpt_obj):
    """
    Minimal, robust state_dict extractor to ensure the finetuned checkpoint is actually loaded.
    This directly targets the low-score issue (random-ish predictions) without changing model logic.
    """
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        if any(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    return ckpt_obj  # may already be an OrderedDict


def _strip_prefix_from_state_dict(state_dict, prefixes):
    if not isinstance(state_dict, dict):
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        new_sd[nk] = v
    return new_sd


def _remap_classifier_keys_for_vit(sd):
    """
    Change is directly aimed at improving score by correctly mapping common saved classifier names
    to timm ViT's 'head.*' so the 5-class finetuned head is actually loaded.
    (Does not change architecture; only corrects checkpoint compatibility.)
    """
    if not isinstance(sd, dict):
        return sd
    out = dict(sd)

    if "fc.weight" in out and "head.weight" not in out:
        out["head.weight"] = out.pop("fc.weight")
    if "fc.bias" in out and "head.bias" not in out:
        out["head.bias"] = out.pop("fc.bias")

    if "classifier.weight" in out and "head.weight" not in out:
        out["head.weight"] = out.pop("classifier.weight")
    if "classifier.bias" in out and "head.bias" not in out:
        out["head.bias"] = out.pop("classifier.bias")

    return out


def _try_load_with_prefixes_and_remaps(model, sd, prefixes):
    sd2 = _strip_prefix_from_state_dict(sd, prefixes=prefixes)
    sd2 = _remap_classifier_keys_for_vit(sd2)
    missing, unexpected = model.load_state_dict(sd2, strict=False)
    return sd2, missing, unexpected


def load_finetuned_weights_strictish(model, ckpt_path):
    """
    Loads finetuned weights with minimal assumptions; prints diagnostics.
    To directly move score toward target, we validate the classifier head was restored:
      - 'head.weight'/'head.bias' must be present and shape-compatible with model.model.head
    If head keys are absent/mismatched, we retry alternate common key-prefix patterns.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _extract_state_dict(ckpt)

    attempts = [
        (
            "attempt#1",
            (
                "module.",
                "model.",
                "net.",
                "model.model.",
                "module.model.",
                "module.model.model.",
            ),
        ),
        (
            "attempt#2",
            (
                "module.",
                "model.",
                "net.",
                "backbone.",
                "encoder.",
                "student.",
                "teacher.",
            ),
        ),
    ]

    loaded_sd = None
    loaded_info = None

    for tag, prefixes in attempts:
        sd_try, missing, unexpected = _try_load_with_prefixes_and_remaps(
            model, sd, prefixes
        )

        ok = True
        if not (
            isinstance(sd_try, dict)
            and ("head.weight" in sd_try)
            and ("head.bias" in sd_try)
        ):
            ok = False
        else:
            try:
                hw = model.model.head.weight
                hb = model.model.head.bias
                if tuple(sd_try["head.weight"].shape) != tuple(hw.shape):
                    ok = False
                if tuple(sd_try["head.bias"].shape) != tuple(hb.shape):
                    ok = False
            except Exception:
                ok = False

        print(
            f"[{tag}] missing keys: {len(missing)} | unexpected keys: {len(unexpected)} | head_ok: {ok}"
        )

        if ok:
            loaded_sd = sd_try
            loaded_info = (tag, missing, unexpected)
            break

    print(f"Loaded finetuned checkpoint from: {ckpt_path}")

    if loaded_info is None:
        print(
            "WARNING: Could not confidently validate finetuned head was loaded. Predictions may be near-random."
        )
    else:
        tag, missing, unexpected = loaded_info
        print(f"Using {tag} as final mapping.")
        if len(missing) > 0:
            print("First 10 missing keys:", missing[:10])
        if len(unexpected) > 0:
            print("First 10 unexpected keys:", unexpected[:10])

    if hasattr(model, "model") and hasattr(model.model, "head"):
        hw = model.model.head.weight
        hb = model.model.head.bias
        print("Model head weight shape:", tuple(hw.shape))
        print("Model head bias shape:", tuple(hb.shape))

    if isinstance(loaded_sd, dict):
        head_keys = [k for k in loaded_sd.keys() if k.startswith("head.")]
        print("Loaded checkpoint head-related keys:", head_keys)




## === cell 7
cassava_model = ViTBase16(n_classes=5, pretrained=True)

if os.path.exists(Cassava_model):
    load_finetuned_weights_strictish(cassava_model, Cassava_model)
else:
    print(
        "Finetuned Cassava checkpoint not found; running with (base) pretrained initialization."
    )

cassava_model = cassava_model.to(device)
cassava_model.eval()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cassava_model.parameters(), lr=1.5e-05)

cassava_model



## === cell 8
sample_sub_path = os.path.join(data_path, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].tolist()
print("Num test ids from sample_submission:", len(test_image_ids))
print("First 3 ids:", test_image_ids[:3])




## === cell 9
class TestSet2(Dataset):
    """Cassava Test Dataset driven by image_id list (ensures stable ordering)."""

    def __init__(self, test_dir, image_ids, transform=None):
        super().__init__()
        self.test_dir = test_dir
        self.image_ids = list(image_ids)
        self.transform = transform
        print("Cassava Disease Test Dataset Length = ", len(self.image_ids))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_name = self.image_ids[idx]
        img_path = os.path.join(self.test_dir, image_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(img)
        else:
            image = transforms.ToTensor()(img)
        return image, image_name




## === cell 10
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

testset = TestSet2(
    test_dir=test_path, image_ids=test_image_ids, transform=test_transform
)
print(testset)



## === cell 11
test_batch_size = 32
test_loader = DataLoader(
    dataset=testset,
    batch_size=test_batch_size,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=2,
)
print(test_loader)



## === cell 12
images, names = next(iter(test_loader))
im = make_grid(images[: min(len(images), 16)], nrow=4)
print("Batch names:", list(names[:4]))

plt.figure(figsize=(12, 4))
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
plt.axis("off")
plt.show()



## === cell 13
pred_rows = []
tic = time.time()

with torch.no_grad():
    for X_test, name in test_loader:
        X_test = X_test.to(device, non_blocking=True)
        logits = cassava_model(X_test)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist()
        for n, p in zip(list(name), preds):
            pred_rows.append((n, int(p)))

toc = time.time() - tic
print("Time for test inference is", toc, "seconds")

submission_df = pd.DataFrame(pred_rows, columns=["image_id", "label"])

submission_df = submission_df.set_index("image_id").loc[test_image_ids].reset_index()

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission_df.head())



## === cell 14
df_sub_test = pd.read_csv("submission.csv")
print("submission shape:", df_sub_test.shape)
print("submission columns:", df_sub_test.columns.tolist())
print(df_sub_test.head())

print("Files in cwd:", os.listdir("."))
