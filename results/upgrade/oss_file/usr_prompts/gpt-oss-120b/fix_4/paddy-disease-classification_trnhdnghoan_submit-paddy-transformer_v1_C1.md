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

0.17294

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.17487) has done: 'I wrap the model loading in a try‑except block so the script no longer crashes when the checkpoint file is missing. If the checkpoint cannot be loaded, a dummy model that always predicts the “normal” class is used, and the whole pipeline still write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.17487) has done: 'I replace the hard‑coded checkpoint path with the one defined in the configuration (`cfg.ckpt_path`) and use the matching backbone name. This ensures the real model checkpoint is loaded instead of falling back to the dummy model that always predicts “normal”, which raise the validation accuracy toward the target value.'
- What this solution (achieved 0.17294) has done: 'I keep the existing pipeline but improve the fallback when the checkpoint file is missing. Instead of using a dummy model that always predicts “Normal”, the code now instantiate the specified backbone with ImageNet‑pretrained weights and the correct number of classes, which provides much more informative predictions and moves the validation accuracy toward the target. The change is limited to the error‑handling block in the inference loop, preserving all core logic.'

# 9. Code solution

## === cell 0
import os
import timm
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from dataclasses import dataclass
from PIL import Image




## === cell 1
class CFG:
    comp_root: str = "/kaggle/input/paddy-disease-classification"
    test_dir: str = "/kaggle/input/paddy-disease-classification/test_images"
    sample_csv: str = "/kaggle/input/paddy-disease-classification/sample_submission.csv"
    ckpt_path: str = (
        "/kaggle/input/transformer_our_data/pytorch/default/1/mobilevit_s_best_our_data.pth"
    )

    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 2
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CLASSES_4 = ["Brow_Spot", "Leaf_Blast", "Leaf_Blight", "Normal"]
    id2label_4 = {i: c for i, c in enumerate(CLASSES_4)}

    MAP_4_TO_10 = {
        "Brow_Spot": "brown_spot",
        "Leaf_Blast": "blast",
        "Leaf_Blight": "bacterial_leaf_blight",
        "Normal": "normal",
    }

    @staticmethod
    def infer_tfms(img_size):
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )


cfg = CFG()




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
def create_model(backbone: str, num_classes: int, pretrained=True) -> nn.Module:
    model = timm.create_model(
        backbone, pretrained=pretrained, num_classes=num_classes, in_chans=3
    )
    return model




## === cell 4
def load_model(backbone, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
    model = create_model(backbone, num_classes=ckpt["meta"]["num_classes"])
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(cfg.device).eval()
    return model




## === cell 5
class DummyModel(nn.Module):
    """
    Simple fallback model that always predicts the index corresponding to
    the 'Normal' class in the 4‑class scheme.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "logits", torch.tensor([-1.0, -1.0, -1.0, 10.0]).unsqueeze(0)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        return self.logits.expand(batch_size, -1)




## === cell 6
BACKBONES = [("mobilevit_s", cfg.ckpt_path)]




## === cell 7
ds = TestDataset(cfg.test_dir, CFG.infer_tfms(cfg.img_size))
dl = DataLoader(
    ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
)

for backbone, ckpt_path in BACKBONES:
    try:
        model = load_model(backbone, ckpt_path)
    except FileNotFoundError:
        print(
            f"Checkpoint not found at {ckpt_path}. Using pretrained '{backbone}' model instead of dummy."
        )
        model = (
            create_model(
                backbone,
                num_classes=len(cfg.CLASSES_4),
                pretrained=True,
            )
            .to(cfg.device)
            .eval()
        )
    preds_map = {}
    with torch.inference_mode():
        for x, ids in dl:
            x = x.to(cfg.device, non_blocking=True)
            pred_ids = model(x).argmax(1).cpu().tolist()
            for img_id, pid in zip(ids, pred_ids):
                four = cfg.id2label_4[pid]
                preds_map[img_id] = cfg.MAP_4_TO_10[four]

    sub = pd.read_csv(cfg.sample_csv)  # image_id, label
    sub["label"] = sub["image_id"].map(preds_map).fillna("normal")
    out_path = os.path.join("/kaggle/working", "submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")
