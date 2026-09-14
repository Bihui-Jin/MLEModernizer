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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.13

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
timm==1.0.19
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
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.4700460829493088

# 6. Current score

0.14028

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.09108) has done: 'I fix the immediate runtime failure by making checkpoint loading robust to missing Kaggle input paths: the script automatically fall back to a timm pretrained model when the specified `.pth` file is not present, so it always runs end-to-end. I also remove the checkpoint-specific assumptions (`ckpt["meta"]["num_classes"]`, `ckpt["model"]`) by supporting multiple common checkpoint formats, while keeping the same inference logic (argmax over logits and the same 4→10 label mapping). Finally, I ensure a valid submission file named `submission.csv` is always written with the required columns from `sample_submission.csv`.'
- What this solution (achieved 0.15027) has done: 'Your current score is low mainly because the inference pipeline predicts only 4 classes and then maps them into the 10 required labels, which guarantees systematic misclassification for the 6 unmapped diseases. To move toward the target accuracy with minimal disruption, I keep the same timm-based inference flow but switch to a 10-class classifier head and use the dataset’s native class names (so no lossy 4→10 mapping). I also make the checkpoint loader robust to both 4-class and 10-class heads by loading with `strict=False` when shapes don’t match, so it still runs even if a provided checkpoint was trained for a different number of classes. Finally, the submission still be aligned to `sample_submission.csv` and always written as `submission.csv`.'
- What this solution (achieved 0.14028) has done: 'Your score is far below the target, and the biggest likely cause (without changing the overall inference approach) is a label-index mismatch: the order in `CLASSES_10` must exactly match the class-to-index mapping used by the training data/model. I fix this by deriving `label2id/id2label` directly from `train.csv` in a stable sorted order, and use that mapping for decoding predictions, which typically yields a large accuracy jump with minimal code change. I also apply the matching train-time resize/crop pattern commonly used with timm (Resize + CenterCrop) to reduce train/test preprocessing mismatch while keeping the same model and argmax inference semantics. The rest of the flow (timm model, checkpoint loading fallback, argmax logits, submission aligned to `sample_submission.csv`) stays the same.'

# 9. Code solution

## === cell 0
import os
import timm
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image




## === cell 1
class CFG:
    comp_root: str = "/kaggle/input/paddy-disease-classification"
    test_dir: str = "/kaggle/input/paddy-disease-classification/test_images"
    sample_csv: str = "/kaggle/input/paddy-disease-classification/sample_submission.csv"
    train_csv: str = "/kaggle/input/paddy-disease-classification/train.csv"

    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 2
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def infer_tfms(img_size: int):
        return transforms.Compose(
            [
                transforms.Resize(int(img_size * 256 / 224)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )


cfg = CFG()

train_df = pd.read_csv(cfg.train_csv)
CLASSES_10 = sorted(train_df["label"].unique().tolist())
label2id_10 = {c: i for i, c in enumerate(CLASSES_10)}
id2label_10 = {i: c for c, i in label2id_10.items()}

print("Derived classes (sorted):", CLASSES_10)




## === cell 2
class TestDataset(Dataset):
    def __init__(self, folder, transform):
        self.ids = sorted(
            [
                f
                for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        self.paths = [os.path.join(folder, f) for f in self.ids]
        self.tfm = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tfm(img), self.ids[i]




## === cell 3
def create_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(
        backbone, pretrained=pretrained, num_classes=num_classes, in_chans=3
    )
    return model


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
    if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        return ckpt
    return None


def _strip_module_prefix(sd: dict) -> dict:
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _filter_mismatched_classifier_keys(model: nn.Module, sd: dict) -> dict:
    """
    Change (stability + score): allow loading checkpoints trained with a different head size
    by dropping only mismatched-shaped tensors, keeping the backbone weights.
    """
    model_sd = model.state_dict()
    out = {}
    for k, v in sd.items():
        if k in model_sd and hasattr(v, "shape") and hasattr(model_sd[k], "shape"):
            if tuple(v.shape) != tuple(model_sd[k].shape):
                continue
        out[k] = v
    return out


def load_model(backbone: str, ckpt_path: str, num_classes: int) -> nn.Module:
    """
    Always build a num_classes model for correct competition labels.
    If a checkpoint exists but has a different head, load backbone weights and ignore mismatched head tensors.
    """
    model = create_model(backbone, num_classes=num_classes, pretrained=True)

    if ckpt_path is not None and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = _extract_state_dict(ckpt)
        if sd is None:
            raise ValueError(f"Unsupported checkpoint format at: {ckpt_path}")
        sd = _strip_module_prefix(sd)
        sd = _filter_mismatched_classifier_keys(model, sd)
        model.load_state_dict(sd, strict=False)

    model.to(cfg.device).eval()
    return model




## === cell 4
BACKBONES = [
    (
        "vit_small_patch16_224",
        "/kaggle/input/model-tk_deeplearning/pytorch/default/1/vit_small_patch16_224_best.pth",
    ),
]



## === cell 5
ds = TestDataset(cfg.test_dir, CFG.infer_tfms(cfg.img_size))
dl = DataLoader(
    ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=torch.cuda.is_available(),
)



## === cell 6
all_preds_map = {}

for backbone, ckpt_path in BACKBONES:
    model = load_model(backbone, ckpt_path, num_classes=len(CLASSES_10))

    preds_map = {}
    with torch.inference_mode():
        for x, ids in dl:
            x = x.to(cfg.device, non_blocking=True)
            logits = model(x)
            pred_ids = logits.argmax(1).detach().cpu().tolist()
            for img_id, pid in zip(ids, pred_ids):
                preds_map[img_id] = id2label_10[int(pid)]

    all_preds_map.update(preds_map)

sub = pd.read_csv(cfg.sample_csv)  # image_id, label
sub["label"] = sub["image_id"].map(all_preds_map).fillna("normal")
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(sub.head())
print("Rows:", len(sub), "Unique predicted labels:", sub["label"].nunique())
print("Any missing after fill:", int(sub["label"].isna().sum()))
