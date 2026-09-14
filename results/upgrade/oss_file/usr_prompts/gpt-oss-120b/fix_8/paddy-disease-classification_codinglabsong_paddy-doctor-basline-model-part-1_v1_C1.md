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

No external packages required in the script and installed.

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

0.8951612903225806

# 6. Current score

0.12952

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.13259) has done: 'I correct the dataset paths so they point to the actual Kaggle input directory (falling back to the relative path if needed). This fixes the `FileNotFoundError` and the `TypeError` that arose from an empty folder list. I also adjust the sample‑submission read/write to use the same resolved base path. No core modeling logic is changed.'
- What this solution (achieved 0.12952) has done: 'I increase the model capacity and training duration to raise validation accuracy, which should translate into a higher leaderboard score. Specifically, I switch the backbone from **resnet34** to **resnet50**, extend fine‑tuning to **10 epochs**, and keep a modest learning rate. I also add a quick validation print so you can see the improvement before generating the submission.'
- What this solution (achieved 0.1299) has done: 'I remove the fp16 conversion (which can degrade training on CPU) and extend fine‑tuning to 15 epochs so the model can learn more from the data while keeping the same architecture and loss. This modest change should raise validation accuracy markedly, moving the score toward the target.'
- What this solution (achieved 0.12952) has done: 'The update keeps the exact training loop, model architecture, loss, and callbacks, but speeds up data loading and GPU work by using a much smaller image resolution for both the initial resize and the augmentation size.  Reducing the image dimensions cuts the amount of tensor work per batch dramatically, allowing the 30‑epoch fine‑tune to finish well under the 600‑second limit while preserving all other logic and reproducibility settings.'

# 9. Code solution

## === cell 0
import subprocess, sys


def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-qU", pkg])


try:
    import fastai
except ImportError:
    install("fastai")
    install("timm")




## === cell 1
import torch

torch.backends.cudnn.benchmark = True

from pathlib import Path
import pandas as pd
import numpy as np
import torch
from fastai.vision.all import *

set_seed(42, reproducible=True)




## === cell 2
possible_base = Path("/kaggle/input/paddy-disease-classification")
if possible_base.is_dir():
    base_path = possible_base
else:
    base_path = Path("data/paddy-disease-classification")

train_path = base_path / "train_images"
test_path = base_path / "test_images"
assert train_path.is_dir(), f"Training images folder not found: {train_path}"
assert test_path.is_dir(), f"Test images folder not found: {test_path}"




## === cell 3
dls = ImageDataLoaders.from_folder(
    train_path,
    valid_pct=0.2,
    seed=42,
    item_tfms=Resize(224, method="squish"),  # smaller initial resize
    batch_tfms=aug_transforms(size=128, min_scale=0.75),  # smaller augmentation size
    bs=64,
    num_workers=8,
)




## === cell 4
learn = vision_learner(
    dls, arch=resnet50, metrics=accuracy, loss_func=CrossEntropyLossFlat()
)

learn = learn.to_fp16()

mixup_c = MixUp(0.2)
save_c = SaveModelCallback(monitor="accuracy", comp=np.greater, fname="best_model")
learn.fine_tune(epochs=30, base_lr=1e-4, cbs=[mixup_c, save_c])

learn.load("best_model")

val_loss, val_acc = learn.validate()
print(f"Validation accuracy after training: {val_acc:.4f}")




## === cell 5
test_files = get_image_files(test_path).sorted()
test_dl = dls.test_dl(test_files)
preds, _ = learn.get_preds(dl=test_dl)
pred_labels = preds.argmax(dim=1)
label_names = np.array(dls.vocab)[pred_labels.cpu().numpy()]




## === cell 6
sample_sub = pd.read_csv(base_path / "sample_submission.csv")
sample_sub["label"] = label_names
submission_path = Path("submission.csv")
sample_sub.to_csv(submission_path, index=False)

print(sample_sub.head())
print(f"Submission saved to {submission_path.resolve()}")
