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

0.8943789664551224

# 6. Current score

0.17414

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.12332) has done: 'I fix the Vision Transformer input-size mismatch by resizing images to the size expected by the pretrained ViT weights (224), which removes the runtime assertion and allows inference to run. I also load the provided trained weights for the ViT, EfficientNet, and linear head if they exist in common Kaggle working/input locations; otherwise the code still run but be untrained (and score poorly). Finally, I make the dataset read only `.jpg/.png` files to avoid accidental non-image entries and ensure the submission merges correctly and is fully populated. These are minimal execution/stability fixes that preserve your model and inference logic.'
- What this solution (achieved 0.1775) has done: 'Your low score is consistent with using randomly initialized fusion/heads (and possibly not finding your trained checkpoints), so the smallest high-impact fix is to make checkpoint loading robust to common key-prefix patterns (`model.`, `module.`, `vit_model.`, `eff_model.`, `linear_head.`) and to search a few additional likely locations under `/kaggle/input` and `/kaggle/working`. I also align preprocessing with the official torchvision pretrained weights by using `weights.transforms()` for ViT/EfficientNet (same resizing is kept), which typically improves accuracy without changing the model or inference logic. Finally, I keep submission alignment safeguards but ensure deterministic, correct loading and inference behavior.'
- What this solution (achieved 0.1775) has done: 'I fix the preprocessing crash by using the official torchvision `weights.transforms()` pipelines (which removes the missing `meta["mean"/"std"]` KeyError in this torchvision version) while keeping your resize-to-224/528 logic intact. I also ensure `test_loader` is always created by making cell 3 run successfully, which resolves the downstream `NameError`. Finally, I make the test image directory resolution robust to the duplicated nested `test_images/test_images` folder so predictions cover all `sample_submission.csv` rows and the merge produces zero missing labels, yielding a valid `submission.csv`.'
- What this solution (achieved 0.1775) has done: 'Your current score (0.1775) is far below the target (0.8944), which strongly suggests the fusion head (and possibly parts of the backbones) are not being loaded from the intended trained checkpoint(s). I make the checkpoint loading robust to “combined” checkpoints (single file containing vit/eff/head weights) by detecting common key patterns and routing sub-state-dicts to the right module, while keeping your exact model definitions and inference logic unchanged. I also switch the models to `eval()` before loading and force `strict=False` loading diagnostics so we don’t silently skip usable weights. These are minimal, execution-safe changes aimed specifically at moving accuracy upward toward the target by ensuring the intended trained weights actually get used.'
- What this solution (achieved 0.17414) has done: 'Your current score (0.1775) is far below the target (0.8944), so the most likely issue is still that your intended trained fusion weights are not being loaded, leaving the ViT/EfficientNet heads and/or linear_head effectively random for this task. I keep your exact model definitions and inference flow, but make checkpoint loading more robust to common “single-model” checkpoints (where keys don’t include vit/eff/head prefixes) by trying direct loads into each module and also by handling `DataParallel`-style `module.` prefixes cleanly. I also make the image preprocessing strictly match the pretrained weights’ expected input by removing the fixed `CenterCrop(600,600)` (which can damage leaf composition) and relying on the official `weights.transforms()` resize/crop pipeline; this is a minimal, metric-aligned change that typically improves accuracy without changing the architecture. Finally, I keep submission alignment checks and ensure the script always writes a valid `submission.csv`.'

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
from torchvision.transforms import v2
from torchvision import models



## === cell 1
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
test_dir = f"{DATA_DIR}/test_images/"
sample_sub_path = f"{DATA_DIR}/sample_submission.csv"

eff_img_size = 528
vit_img_size = 224

batch_size = 16
num_workers = 4
num_classes = 5
tta = False

weights_vit = models.ViT_B_16_Weights.IMAGENET1K_V1
vit_model = models.vit_b_16(weights=weights_vit)
vit_model.heads.head = torch.nn.Linear(vit_model.heads.head.in_features, num_classes)

weights_eff = models.EfficientNet_B0_Weights.IMAGENET1K_V1
eff_model = models.efficientnet_b0(weights=weights_eff)
eff_model.classifier[1] = torch.nn.Linear(
    eff_model.classifier[1].in_features, num_classes
)

linear_head = torch.nn.Linear(num_classes * 2, num_classes)

vit_model = vit_model.to(device)
eff_model = eff_model.to(device)
linear_head = linear_head.to(device)


def _load_any_checkpoint(
    vit_model: torch.nn.Module,
    eff_model: torch.nn.Module,
    linear_head: torch.nn.Module,
    ckpt_path: str,
) -> bool:
    if not ckpt_path or not os.path.exists(ckpt_path):
        return False

    sd = torch.load(ckpt_path, map_location="cpu")
    if (
        isinstance(sd, dict)
        and "state_dict" in sd
        and isinstance(sd["state_dict"], dict)
    ):
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        return False

    def _strip_prefixes(k: str) -> str:
        nk = k
        for p in ("model.", "module.", "net.", "backbone."):
            if nk.startswith(p):
                nk = nk[len(p) :]
        return nk

    cleaned = {_strip_prefixes(k): v for k, v in sd.items()}

    vit_sd, eff_sd, head_sd, other_sd = {}, {}, {}, {}
    for k, v in cleaned.items():
        if k.startswith(("vit_model.", "vit.")):
            vit_sd[k.split(".", 1)[1]] = v
        elif k.startswith(("eff_model.", "eff.", "efficientnet.", "enet.")):
            eff_sd[k.split(".", 1)[1]] = v
        elif k.startswith(
            ("linear_head.", "head.", "fusion_head.", "classifier_head.")
        ):
            head_sd[k.split(".", 1)[1]] = v
        else:
            other_sd[k] = v

    loaded = False

    if vit_sd or eff_sd or head_sd:
        if vit_sd:
            m, u = vit_model.load_state_dict(vit_sd, strict=False)
            print(
                f"[ckpt:{os.path.basename(ckpt_path)}] vit loaded. missing={len(m)} unexpected={len(u)}"
            )
            loaded = True
        if eff_sd:
            m, u = eff_model.load_state_dict(eff_sd, strict=False)
            print(
                f"[ckpt:{os.path.basename(ckpt_path)}] eff loaded. missing={len(m)} unexpected={len(u)}"
            )
            loaded = True
        if head_sd:
            m, u = linear_head.load_state_dict(head_sd, strict=False)
            print(
                f"[ckpt:{os.path.basename(ckpt_path)}] head loaded. missing={len(m)} unexpected={len(u)}"
            )
            loaded = True
        return loaded

    tried_any = False
    direct_loaded = False

    for name, module in (("vit", vit_model), ("eff", eff_model), ("head", linear_head)):
        try:
            m, u = module.load_state_dict(other_sd, strict=False)
            tried_any = True
            useful = (len(other_sd) > 0) and (len(u) < len(other_sd))
            print(
                f"[ckpt:{os.path.basename(ckpt_path)}] direct->{name} attempted. missing={len(m)} unexpected={len(u)} useful={useful}"
            )
            direct_loaded = direct_loaded or useful
        except Exception as e:
            print(
                f"[ckpt:{os.path.basename(ckpt_path)}] direct->{name} failed: {type(e).__name__}: {e}"
            )

    if direct_loaded:
        return True

    if not tried_any:
        print(
            f"[ckpt:{os.path.basename(ckpt_path)}] no usable dict-like state found after cleaning."
        )
        return False

    return False


def _gather_ckpts(root: str):
    pats = [os.path.join(root, "**", "*.pth"), os.path.join(root, "**", "*.pt")]
    files = set()
    for p in pats:
        files.update(glob.glob(p, recursive=True))
    return sorted(files)


def _score_ckpt_name(p: str) -> int:
    name = os.path.basename(p).lower()
    score = 0
    if "best" in name:
        score += 10
    if "final" in name or "last" in name:
        score += 7
    if "epoch" in name:
        score += 2
    if "fold" in name:
        score += 1
    if "ckpt" in name or "checkpoint" in name:
        score += 2
    try:
        score += min(int(os.path.getsize(p) / (50 * 1024 * 1024)), 5)
    except Exception:
        pass
    return score


all_ckpt_files = _gather_ckpts("/kaggle/working") + _gather_ckpts("/kaggle/input")
all_ckpt_files = sorted(all_ckpt_files, key=lambda x: _score_ckpt_name(x), reverse=True)

vit_model.eval()
eff_model.eval()
linear_head.eval()

loaded_any = False

common_candidates = [
    "/kaggle/working/vit_model.pth",
    "/kaggle/working/eff_model.pth",
    "/kaggle/working/linear_head.pth",
    "/kaggle/working/vit.pth",
    "/kaggle/working/eff.pth",
    "/kaggle/working/head.pth",
    "/kaggle/working/model.pth",
    "/kaggle/working/model.pt",
    f"{DATA_DIR}/vit_model.pth",
    f"{DATA_DIR}/eff_model.pth",
    f"{DATA_DIR}/linear_head.pth",
    f"{DATA_DIR}/model.pth",
    f"{DATA_DIR}/model.pt",
]
for p in common_candidates:
    if _load_any_checkpoint(vit_model, eff_model, linear_head, p):
        print("Loaded checkpoint from:", p)
        loaded_any = True
        break

if not loaded_any:
    for p in all_ckpt_files[:120]:
        if _load_any_checkpoint(vit_model, eff_model, linear_head, p):
            print("Loaded checkpoint from scan:", p)
            loaded_any = True
            break

if not loaded_any:
    print(
        "No trained weight files found; running with ImageNet-pretrained backbones and randomly initialized heads."
    )




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test data (image_id only)."""

    def __init__(
        self,
        data_dir,
        vit_transform=None,
        eff_transform=None,
        ttas=None,
    ):
        super().__init__(root=data_dir)

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        self.images = sorted(
            [f for f in os.listdir(data_dir) if f.lower().endswith(exts)]
        )
        self.ttas = ttas

        self.vit_transform = vit_transform
        self.eff_transform = eff_transform

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        vit_img = img
        eff_img = img

        if self.ttas is not None:
            vit_img = (
                [self.vit_transform(t(vit_img)) for t in self.ttas]
                if self.vit_transform
                else [t(vit_img) for t in self.ttas]
            )
            eff_img = (
                [self.eff_transform(t(eff_img)) for t in self.ttas]
                if self.eff_transform
                else [t(eff_img) for t in self.ttas]
            )
        else:
            if self.vit_transform:
                vit_img = self.vit_transform(vit_img)
            if self.eff_transform:
                eff_img = self.eff_transform(eff_img)

        return vit_img, eff_img, filename

    def __len__(self):
        return len(self.images)




## === cell 3
def _resolve_image_dir(base_dir: str) -> str:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if os.path.isdir(base_dir):
        files = [f for f in os.listdir(base_dir) if f.lower().endswith(exts)]
        if len(files) > 0:
            return base_dir
        nested = os.path.join(base_dir, os.path.basename(os.path.normpath(base_dir)))
        if os.path.isdir(nested):
            files2 = [f for f in os.listdir(nested) if f.lower().endswith(exts)]
            if len(files2) > 0:
                return nested
    return base_dir


test_dir = _resolve_image_dir(test_dir)
print("Using test_dir:", test_dir)

vit_preprocess = weights_vit.transforms()
eff_preprocess = weights_eff.transforms()

if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

test_dataset = CassavaDataset(
    test_dir,
    vit_transform=vit_preprocess,
    eff_transform=eff_preprocess,
    ttas=ttas,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 4
all_names = []
all_preds = []

vit_model.eval()
eff_model.eval()
linear_head.eval()

with torch.no_grad():
    for vit_inputs, eff_inputs, filenames in test_loader:
        batch_n = len(filenames)

        if tta:
            vit_inputs = torch.cat(vit_inputs, dim=0).to(device)
            eff_inputs = torch.cat(eff_inputs, dim=0).to(device)
            filenames = list(filenames)

            vit_outputs = vit_model(vit_inputs)
            eff_outputs = eff_model(eff_inputs)

            vit_batch_logits = torch.stack(torch.split(vit_outputs, batch_n), dim=0)
            vit_mean_logits = torch.mean(vit_batch_logits, dim=0)

            eff_batch_logits = torch.stack(torch.split(eff_outputs, batch_n), dim=0)
            eff_mean_logits = torch.mean(eff_batch_logits, dim=0)

            logit_inputs = torch.cat([vit_mean_logits, eff_mean_logits], dim=1)
            outputs = linear_head(logit_inputs)

            mean_preds = normalizer(outputs)
            pred_labels = torch.argmax(mean_preds, 1).tolist()
        else:
            vit_inputs = vit_inputs.to(device, non_blocking=True)
            eff_inputs = eff_inputs.to(device, non_blocking=True)
            filenames = list(filenames)

            vit_outputs = vit_model(vit_inputs)
            eff_outputs = eff_model(eff_inputs)

            logit_inputs = torch.cat([vit_outputs, eff_outputs], dim=1)
            outputs = linear_head(logit_inputs)

            preds = normalizer(outputs)
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

assert len(all_names) == len(test_dataset), (len(all_names), len(test_dataset))
assert len(all_preds) == len(test_dataset), (len(all_preds), len(test_dataset))



## === cell 5
sample_sub = pd.read_csv(sample_sub_path)
pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})

my_submission = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
missing = my_submission["label"].isna().sum()
assert missing == 0, f"Missing predictions for {missing} test images."

my_submission["label"] = my_submission["label"].astype(int)
my_submission.to_csv("submission.csv", index=False)

print(my_submission.shape)
print(my_submission.head())
print("Wrote submission.csv")
