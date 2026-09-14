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

geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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

0.8286

# 6. Current score

0.05568

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I (1) fix the missing-weight crash by loading weights only if they exist and otherwise running a small, deterministic training fallback using the same SqueezeNet-based core model so a valid submission is always produced. I (2) fix test image enumeration to avoid the nested `test_images/test_images` directory and ensure predictions are aligned exactly to `sample_submission.csv` to prevent length/order submission errors. I (3) correct the class count to 5 (cassava has 5 labels) while preserving the same “binary + minority” two-model logic by mapping the 4-way head to classes {0,1,2,4} and the binary head to {not-3 vs 3}. These changes are required for correctness/end-to-end execution and should yield a reasonable accuracy toward the target rather than failing to produce a valid submission.'
- What this solution (achieved 0.46375) has done: 'I fix the root-cause FileNotFoundError by making image directory resolution robust against the duplicated `train_images/train_images` and `test_images/test_images` nesting present in this dataset dump, and ensure both the dataset loader and inference use the same resolved directory. I also make the training dataloader safer by using `num_workers=0` when running the fallback training (so file-not-found/debugging issues don’t get hidden inside worker processes), without changing the training core logic. Finally, I keep the existing two-model prediction logic intact and ensure the submission is aligned exactly to `sample_submission.csv` and written to `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.46375) has done: 'Your current score is far below the target, so we should make small, low-risk changes that improve accuracy without altering the two-model SqueezeNet core setup. The biggest correctness/performance bug here is that `prediction_logic` returns the string `"0_1_2_4"` for the non-3 binary class, which makes those predictions invalid (and effectively destroys accuracy); fixing that to return the minority model’s mapped class when binary predicts “not 3” should immediately move the score up. I also fix the binary gating to use the actual probability of class “3” consistently (and flatten softmax outputs safely), and ensure we always feed logits in the right shape for both training and inference (matching what you already do in training). These are minimal semantic fixes that keep architecture/training intact and should move the score significantly toward the target band.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.46375) is far below the target (0.8286), so we should make a small, low-risk accuracy improvement without changing the two-model SqueezeNet setup. The biggest likely issue is that both models are initialized with `pretrained=False`, so unless external weights exist your predictions are effectively near-random; switching to ImageNet-pretrained backbones keeps the architecture identical but makes both the loaded-weights path and the fallback-training path much stronger. I also fix the SqueezeNet classifier Conv2d padding (should be 0 for a 1×1 conv) to match standard SqueezeNet behavior and avoid unnecessary shape artifacts. Finally, I ensure inference uses `torch.softmax` directly (same semantics as your `softmax` but avoids numpy edge-cases), keeping the gating logic unchanged.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.11584) is far below the target (0.8286), so we should fix a high-impact correctness bug with minimal surface area: your image pipeline swaps RGB↔BGR, which strongly degrades pretrained ImageNet normalization and makes predictions near-random. I remove the channel swap in both training and inference (keeping the same transforms, model architecture, and training loop), so the pretrained backbone sees the expected RGB distribution. I also make the gate threshold slightly less strict (0.65 → 0.55) to reduce missed class-3 detections, which typically improves accuracy when the binary head is undertrained/weak. Everything else (two-model logic, SqueezeNet heads, fallback training behavior, submission alignment/path) stays the same and the script still writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.05531) is far below the target (0.8286), so we need a minimal but high-impact correctness fix rather than tuning. The biggest issue is that the 4-way “minority” head is trained with a `{0,1,2,4} -> {0,1,2,3}` mapping, but inference currently maps its argmax through an incorrect `minority_idx` dict (it even includes label 3, which should never be produced by that head). I fix this by replacing the inference mapping with the exact inverse of the training mapping `{0:0, 1:1, 2:2, 3:4}` so non-3 predictions become valid again, while keeping the two-model gating logic and SqueezeNet architecture unchanged. Everything else (transforms, training loop, submission alignment/path) stays the same so it still runs end-to-end and writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.05531) is far below the target (0.8286), so we should make a small correctness fix that most directly impacts accuracy without changing the two-model SqueezeNet setup. The main issue is the binary gate: with `p_3 = preds_binary[0,1]` you’re implicitly assuming index 1 always corresponds to class “3”, but your binary labels are encoded as `(label==3)` meaning class “3” is index 1 only if training learned that mapping consistently; to make this robust, we compute `p_3` from the logits by explicitly taking the probability of the positive class using a stable softmax and keep the same threshold semantics. Additionally, we remove the unused numpy `softmax()` function to avoid accidental future misuse and ensure inference always uses torch softmax on-device. These are minimal changes that preserve architecture/training while fixing a high-impact gating correctness issue, which should move accuracy substantially upward toward the target band.'
- What this solution (achieved 0.05568) has done: 'Your current score is far below the target, so we should make a minimal, high-impact accuracy fix without changing the two-model SqueezeNet setup. The biggest likely remaining issue is that inference uses `CenterCrop(400)` which can crop away the diseased leaf region (many Cassava images are not tightly centered), hurting both heads; switching to `Resize(256) + CenterCrop(224)` keeps the same “simple deterministic transform” core logic but is much more standard for ImageNet-pretrained backbones. I also apply the exact same transform consistently in both training and inference (already mostly true) and keep the gating, mappings, and training loop untouched. This should improve accuracy materially and move the score toward the target band while staying within the constraints.'
- What this solution (achieved 0.05568) has done: 'Your current score (0.05568) is far below the target (0.8286), so we should apply a minimal but high-impact correctness fix rather than tuning. The biggest remaining likely issue is a train/inference mismatch: during training you feed tensors shaped (B,3,224,224), but during inference `img_transform()` returns (1,3,224,224) already and you additionally `unsqueeze(0)` inside transforms, which is correct; however the model outputs are being flattened with `.view()` and the SqueezeNet classifier includes `AdaptiveAvgPool2d`, so the outputs are already (B,C,1,1) and flattening is fine. The more impactful fix is to ensure the minority head is actually used for all “not-3” cases by making the class-3 gating more robust: instead of a fixed threshold (0.55) that can misroute many samples when the binary head is weak, we route to class 3 only when it is the *argmax* of the binary head (same semantics: binary decides 3 vs not-3, but removes fragile calibration dependence). This keeps the same two-model logic, architecture, and loss, but should materially increase accuracy toward the target.'
- What this solution (achieved 0.05568) has done: 'Your current score is far below the target, so the smallest change that plausibly moves accuracy upward is to fix the gating logic so it uses the binary head as actually trained: class “3” vs “not 3”. Right now gating by argmax over a 2-way head that has never seen “class 0 vs class 3” semantics can misroute many samples; switching back to a stable probability-of-3 threshold (using softmax on the positive class) better matches the training labels and typically improves routing. I keep the same two-model SqueezeNet setup, transforms, training fallback, and class mappings; only the gating decision is adjusted and made numerically robust. The script still runs end-to-end and writes `/kaggle/working/submission.csv` aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os, sys, time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sz = 224
num_classes = 5

proj_dir = "/kaggle/input/cassava-leaf-disease-classification/"
train_dir = os.path.join(proj_dir, "train_images")
test_dir = os.path.join(proj_dir, "test_images")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1
pass




## === cell 2
def light_model(num_classes_out: int):
    weights = torchvision.models.SqueezeNet1_0_Weights.IMAGENET1K_V1
    squeezenet_custom = torchvision.models.squeezenet1_0(weights=weights)

    classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Conv2d(
            in_channels=512,
            out_channels=num_classes_out,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        ),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
    )
    squeezenet_custom.classifier = classifier
    return squeezenet_custom


squeezenet_custom_4 = light_model(4).to(DEVICE)
squeezenet_custom_2 = light_model(2).to(DEVICE)



## === cell 3
leaf_transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 4
minority_inv_map = {0: 0, 1: 1, 2: 2, 3: 4}

binary_idx = {0: 0, 1: 3}



## === cell 5
weight_dir = "../input/cassava-models"
model_1_path = os.path.join(weight_dir, "minority_weights.pth")
model_2_path = os.path.join(weight_dir, "binary_weights.pth")


def _try_load_weights(model: nn.Module, path: str) -> bool:
    """
    Bugfix: original code hard-crashed when weight files are missing.
    Load either full model, state_dict, or checkpoint dict where possible.
    """
    if not os.path.exists(path):
        return False
    obj = torch.load(path, map_location="cpu")
    state_dict = None
    if isinstance(obj, nn.Module):
        state_dict = obj.state_dict()
    elif isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state_dict = obj["state_dict"]
        else:
            state_dict = obj
    if state_dict is None:
        return False
    new_sd = {}
    for k, v in state_dict.items():
        nk = k.replace("module.", "")
        new_sd[nk] = v
    model.load_state_dict(new_sd, strict=False)
    return True


loaded_1 = _try_load_weights(squeezenet_custom_4, model_1_path)
loaded_2 = _try_load_weights(squeezenet_custom_2, model_2_path)

squeezenet_custom_4.eval()
squeezenet_custom_2.eval()

print(f"Loaded minority (4-way) weights: {loaded_1} from {model_1_path}")
print(f"Loaded binary (2-way) weights: {loaded_2} from {model_2_path}")




## === cell 6
def prediction_logic(img, squeezenet_custom_4, squeezenet_custom_2, thresh_3=0.55):
    """
    Score-improving fix (minimal): make gating match how the binary head was trained.
    The binary dataset encodes targets as (label == 3) => 1 else 0, so the correct
    gating signal is P(class==1) from the 2-way softmax, not argmax calibration quirks.
    """
    with torch.no_grad():
        out_min = squeezenet_custom_4(img).detach().float()
        out_bin = squeezenet_custom_2(img).detach().float()

        out_min = out_min.view(out_min.size(0), -1)  # (1,4)
        out_bin = out_bin.view(out_bin.size(0), -1)  # (1,2)

        probs_min = torch.softmax(out_min, dim=1)  # (1,4)
        probs_bin = torch.softmax(out_bin, dim=1)  # (1,2)

        minority_cls_encoded = int(torch.argmax(probs_min, dim=1).item())  # 0..3
        p_3 = float(probs_bin[:, 1].item())  # explicitly "positive" class (label==3)

    if p_3 >= thresh_3:
        return 3
    else:
        return int(minority_inv_map[minority_cls_encoded])




## === cell 7
def img_transform(img_path):
    """
    Keep RGB (no channel swap): pretrained ImageNet normalization expects RGB.
    """
    img = Image.open(img_path).convert("RGB")
    img = leaf_transform(img).float().unsqueeze(0)
    return img




## === cell 8
df = pd.read_csv(os.path.join(proj_dir, "train.csv"))
sample_df = pd.read_csv(os.path.join(proj_dir, "sample_submission.csv"))


def _resolve_image_dir(base_dir: str, df_ids=None) -> str:
    """
    Bugfix: some dataset exports contain nested dirs like train_images/train_images/.
    Prefer the directory that actually contains the image_ids from the CSV.
    """
    candidates = [base_dir, os.path.join(base_dir, os.path.basename(base_dir))]
    candidates = [c for c in candidates if os.path.isdir(c)]
    if not candidates:
        return base_dir

    if df_ids is None or len(df_ids) == 0:
        return candidates[-1]

    probe = list(df_ids[: min(50, len(df_ids))])
    best = candidates[0]
    best_hits = -1
    for c in candidates:
        hits = sum(os.path.exists(os.path.join(c, pid)) for pid in probe)
        if hits > best_hits:
            best_hits = hits
            best = c
    return best


train_img_dir = _resolve_image_dir(train_dir, df["image_id"].values)
test_img_dir = _resolve_image_dir(test_dir, sample_df["image_id"].values)

print("Resolved train image dir:", train_img_dir)
print("Resolved test image dir:", test_img_dir)
print("Train rows:", len(df), "Sample submission rows:", len(sample_df))




## === cell 9
class CassavaDataset(Dataset):
    def __init__(self, df_, img_dir, transform, binary=False, minority=False):
        self.df = df_.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.binary = binary
        self.minority = minority

        if self.binary:
            self.targets = (self.df["label"].values == 3).astype(np.int64)
        elif self.minority:
            keep = self.df["label"].values != 3
            self.df = self.df.loc[keep].reset_index(drop=True)
            lab = self.df["label"].values.astype(np.int64)
            map_to = {0: 0, 1: 1, 2: 2, 4: 3}
            self.targets = np.array([map_to[int(x)] for x in lab], dtype=np.int64)
        else:
            self.targets = self.df["label"].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        img_path = os.path.join(self.img_dir, image_id)
        x = Image.open(img_path).convert("RGB")
        x = self.transform(x).float()
        y = int(self.targets[idx])
        return x, y


def _train_one_model(model, train_loader, epochs=1, lr=1e-3):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            out = out.view(out.size(0), -1)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
    model.eval()


if (not loaded_1) or (not loaded_2):
    df_shuf = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df_sub = df_shuf.iloc[: min(len(df_shuf), 6000)].copy()

    bs = 32 if DEVICE.type == "cuda" else 16
    nw = 0

    if not loaded_2:
        ds_bin = CassavaDataset(df_sub, train_img_dir, leaf_transform, binary=True)
        dl_bin = DataLoader(
            ds_bin,
            batch_size=bs,
            shuffle=True,
            num_workers=nw,
            pin_memory=(DEVICE.type == "cuda"),
        )
        _train_one_model(squeezenet_custom_2, dl_bin, epochs=1, lr=1e-3)
        print("Trained fallback binary model for 1 epoch on subset.")

    if not loaded_1:
        ds_min = CassavaDataset(df_sub, train_img_dir, leaf_transform, minority=True)
        dl_min = DataLoader(
            ds_min,
            batch_size=bs,
            shuffle=True,
            num_workers=nw,
            pin_memory=(DEVICE.type == "cuda"),
        )
        _train_one_model(squeezenet_custom_4, dl_min, epochs=1, lr=1e-3)
        print("Trained fallback minority model for 1 epoch on subset.")



## === cell 10
missing = []
test_preds = []

for i, image_id in enumerate(sample_df["image_id"].tolist()):
    img_path = os.path.join(test_img_dir, image_id)
    if not os.path.exists(img_path):
        missing.append(image_id)
        pred = 3
    else:
        img = img_transform(img_path).to(DEVICE)
        pred = prediction_logic(img, squeezenet_custom_4, squeezenet_custom_2)
    test_preds.append(pred)

print("Missing test images:", len(missing))
if len(missing) > 0:
    print("First missing examples:", missing[:5])

sub = pd.DataFrame(
    {
        "image_id": sample_df["image_id"].values,
        "label": np.array(test_preds, dtype=np.int64),
    }
)
sub.head()



## === cell 11
out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote submission:", out_path, "rows:", len(sub), "cols:", list(sub.columns))
print(sub["label"].value_counts().sort_index())
