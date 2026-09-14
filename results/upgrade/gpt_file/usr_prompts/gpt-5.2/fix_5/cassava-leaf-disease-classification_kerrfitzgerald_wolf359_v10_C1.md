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

0.852825627077667

# 6. Current score

0.537

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24477) has done: 'I fix the two issues that prevent an end-to-end run and submission creation: (1) the fine-tuned checkpoint path is missing, so I make the loader robust by falling back to a standard pretrained ViT (timm weights) when the Kaggle input checkpoint isn’t available, and (2) the model weights are currently on CPU while inputs are on GPU, so I ensure the model is moved to the same device after all weight-loading. These changes preserve the core ViT architecture and inference flow, but guarantee the notebook produces a valid `submission.csv`. This should also yield a reasonable score (likely below your target without the provided fine-tuned weights, but it run and submit correctly).'
- What this solution (achieved 0.50747) has done: 'Your score is far below the target, and the main cause is that your notebook is likely not loading the intended fine-tuned checkpoint (so it falls back to a generic ImageNet ViT head for 5 classes, which performs near-random on cassava). I keep your ViT architecture and inference flow identical, but make the checkpoint loading robust to common training wrappers (e.g., `module.`, `model.`, `net.`, `state_dict` nesting) so the fine-tuned weights actually land in the model. I also ensure the fallback still uses your `ViTBase16` class (same head wiring) and prefer `pretrained=True` there, so even without the finetuned file you don’t end up with an untrained head by accident. These are minimal, execution-safe changes aimed specifically at moving accuracy up toward your target without changing the modeling approach.'
- What this solution (achieved 0.537) has done: 'Your current score (0.50747) is far below the target (0.85283), so we should safely increase performance without changing the model architecture or training approach. The biggest likely remaining issue is that the fine-tuned checkpoint may still not be loading correctly due to key mismatches (e.g., `head.*` vs `model.head.*`, or nested prefixes), so I make the checkpoint loader map keys onto your `ViTBase16` wrapper deterministically and verify that head weights load. I also ensure test-time preprocessing matches typical ViT fine-tuning (center-crop style) while keeping image size and normalization the same, as a minimal transform tweak that can improve accuracy without altering semantics. Finally, I add lightweight sanity prints (counts of loaded/missing keys and head weight stats) to confirm the intended weights are actually used before producing `submission.csv`.'

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
from torchvision import transforms
from torchvision.utils import make_grid

from PIL import Image
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
data_path = "../input/cassava-leaf-disease-classification/"
train_path = "../input/cassava-leaf-disease-classification/train_images/"
test_path = "../input/cassava-leaf-disease-classification/test_images/"

model_path = "../input/vitbase16224/jx_vit_base_p16_224-80ecf9dd.pth"
Cassava_model = (
    "../input/cassavanewaugtp95epochs3/CassavaViT_newaug_TP95_Epochs3_LR1-75e05.pt"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



## === cell 2
wheel_path = "../input/timm034/timm-0.3.4-py3-none-any.whl"
if not os.path.exists(wheel_path):
    print(
        "timm wheel not found at:",
        wheel_path,
        " -> assuming timm is already available.",
    )
else:
    import sys, subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_path, "-q"])



## === cell 3
import timm



## === cell 4
print("Available ViT Models (first 20):")
print(timm.list_models("vit*")[:20])




## === cell 5
class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        super(ViTBase16, self).__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False)

        if pretrained:
            if os.path.exists(model_path):
                state = torch.load(model_path, map_location="cpu")
                self.model.load_state_dict(state)
                print("Loaded backbone weights from:", model_path)
            else:
                print(
                    "Backbone weights not found at:",
                    model_path,
                    " -> using timm pretrained=True weights for backbone.",
                )
                self.model = timm.create_model("vit_base_patch16_224", pretrained=True)

        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        return self.model(x)




## === cell 6
def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        if len(ckpt_obj) and all(isinstance(kk, str) for kk in ckpt_obj.keys()):
            return ckpt_obj
    return ckpt_obj


def _strip_known_prefixes(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    prefixes = ("module.", "model.", "net.")
    out = {}
    for k, v in state_dict.items():
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


def _remap_to_wrapper_keys(state_dict):
    """
    Minimal, score-relevant fix: many fine-tuned checkpoints are saved from a bare timm model
    (keys like 'patch_embed.*', 'head.weight'), while our wrapper expects 'model.patch_embed.*',
    'model.head.*'. This remap helps the fine-tuned weights actually load, improving accuracy.
    """
    if not isinstance(state_dict, dict):
        return state_dict

    remapped = {}
    for k, v in state_dict.items():
        nk = k

        if not nk.startswith("model."):
            nk = "model." + nk

        remapped[nk] = v
    return remapped


cassava_model = ViTBase16(n_classes=5, pretrained=True)

loaded_finetuned = False
if os.path.exists(Cassava_model):
    ckpt = torch.load(Cassava_model, map_location="cpu")
    raw_sd = _extract_state_dict(ckpt)
    raw_sd = _strip_known_prefixes(raw_sd)

    state_dict = _remap_to_wrapper_keys(raw_sd)

    try:
        cassava_model.load_state_dict(state_dict, strict=True)
        print("Loaded fine-tuned checkpoint (strict) from:", Cassava_model)
        loaded_finetuned = True
    except Exception as e:
        print("Strict load failed (will retry strict=False). Reason:", repr(e))
        incompatible = cassava_model.load_state_dict(state_dict, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        print("Loaded fine-tuned checkpoint (strict=False) from:", Cassava_model)
        print(
            "Missing keys count:",
            len(missing),
            "| Unexpected keys count:",
            len(unexpected),
        )
        head_missing = [k for k in missing if "head" in k]
        if head_missing:
            print(
                "WARNING: head-related missing keys (may hurt score):",
                head_missing[:20],
            )
        loaded_finetuned = True
else:
    print(
        "Fine-tuned model checkpoint not found at:",
        Cassava_model,
        " -> using timm pretrained backbone weights instead.",
    )
    cassava_model = ViTBase16(n_classes=5, pretrained=True)

cassava_model = cassava_model.to(device)
cassava_model.eval()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cassava_model.parameters(), lr=1.5e-05)

with torch.no_grad():
    hw = cassava_model.model.head.weight.detach().float().cpu()
    hb = cassava_model.model.head.bias.detach().float().cpu()
    print("Head weight mean/std:", float(hw.mean()), float(hw.std()))
    print("Head bias mean/std:", float(hb.mean()), float(hb.std()))

cassava_model



## === cell 7
test_img_names = [f for f in os.listdir(test_path) if f.lower().endswith(".jpg")]
print("Testing Images:", len(test_img_names))
print("First 5:", test_img_names[:5])




## === cell 8
class TestSet2(Dataset):
    """Cassava Disease Test Dataset"""

    def __init__(self, root_dir, test_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.test_dir = test_dir
        self.transform = transform

        self.files = sorted(
            [f for f in os.listdir(self.test_dir) if f.lower().endswith(".jpg")]
        )
        print(root_dir)
        print(test_dir)
        print("Cassava Disease Test Dataset Length = ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_name = self.files[idx]
        img_path = os.path.join(self.test_dir, image_name)
        img = Image.open(img_path).convert("RGB")

        image = self.transform(img) if self.transform else img
        return (image, image_name)




## === cell 9
test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 10
testset = TestSet2(root_dir="", test_dir=test_path, transform=test_transform)
print(testset)



## === cell 11
test_batch_size = 32
test_loader = DataLoader(
    dataset=testset,
    batch_size=test_batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
print(test_loader)



## === cell 12
images, names = next(iter(test_loader))
im = make_grid(images[:8], nrow=4)
print("Example names:", list(names[:8]))
plt.figure(figsize=(12, 4))
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
plt.axis("off")
plt.show()



## === cell 13
rows = []
tic = time.time()

with torch.no_grad():
    for X_test, name in test_loader:
        X_test = X_test.to(device, non_blocking=True)
        logits = cassava_model(X_test)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int)

        for img_name, label in zip(name, preds):
            rows.append((img_name, int(label)))

toc = time.time() - tic
print("Time for test inference is", toc, "seconds")

pred_df = pd.DataFrame(rows, columns=["image_id", "label"])

sample_sub_path = os.path.join(data_path, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)
sub = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if sub["label"].isna().any():
    fill_label = int(pred_df["label"].mode().iloc[0]) if len(pred_df) else 0
    sub["label"] = sub["label"].fillna(fill_label).astype(int)
else:
    sub["label"] = sub["label"].astype(int)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Saved submission.csv with shape:", sub.shape)

print("Files in current directory:", os.listdir("."))



## === cell 14
df_sub_test = pd.read_csv("submission.csv")
print(df_sub_test.shape)
print(df_sub_test.columns.tolist())
print(df_sub_test.head())
assert df_sub_test.columns.tolist() == ["image_id", "label"]
assert df_sub_test["image_id"].nunique() == len(df_sub_test)
assert df_sub_test["label"].between(0, 4).all()
print("Submission format looks valid.")
