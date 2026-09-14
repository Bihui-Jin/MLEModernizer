# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.12

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

# 5. Code solution

## === cell 0
from pathlib import Path


def setup_comp(comp, install=None):
    p1 = Path("/kaggle/input") / comp
    if p1.exists():
        return p1
    p2 = Path("/kaggle/data") / comp
    if p2.exists():
        return p2
    return p1




## === cell 1
comp = "paddy-disease-classification"
path = setup_comp(comp, install='fastai "timm>=0.6.2.dev0"')



## === cell 2
path



## === cell 3
from fastai.vision.all import *
import os, torch

set_seed(42, reproducible=True)

try:
    ncpu = os.cpu_count() or 2
    torch.set_num_threads(min(8, ncpu))
    torch.set_num_interop_threads(min(4, ncpu))
except Exception:
    pass

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True



## === cell 4
trn_path = path / "train_images"



## === cell 5
pass



## === cell 6
sizes = None



## === cell 7
ncpu = os.cpu_count() or 2
num_workers = min(
    4, ncpu
)  # lower overhead; fastai/torch often saturates with fewer workers on JPEG decode
prefetch_factor = 2

dls = ImageDataLoaders.from_folder(
    trn_path,
    valid_pct=0.2,
    seed=42,
    item_tfms=Resize(
        128, method="squish"
    ),  # was 480; final training size is 128 via batch_tfms
    batch_tfms=aug_transforms(size=128, min_scale=0.75),
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=prefetch_factor,
)



## === cell 8
learn = vision_learner(dls, "resnet26d", metrics=error_rate, path=".").to_fp16()



## === cell 9
pass



## === cell 10
learn.fine_tune(3, 0.01)



## === cell 11
ss = pd.read_csv(path / "sample_submission.csv")
ss



## === cell 12
tst_files = get_image_files(path / "test_images")
tst_files = sorted(tst_files, key=lambda p: p.name)

tst_dl = dls.test_dl(
    tst_files,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=prefetch_factor,
)



## === cell 13
probs, _, idxs = learn.get_preds(dl=tst_dl, with_decoded=True)
idxs



## === cell 14
dls.vocab



## === cell 15
vocab = np.array(dls.vocab, dtype=object)
results = vocab[idxs.cpu().numpy()]
results



## === cell 16
ss["label"] = results
ss.to_csv("submission.csv", index=False)

print(ss.head())



## === cell 17
try:
    iskaggle
except NameError:
    iskaggle = False

if not iskaggle:
    try:
        from kaggle import api

        api.competition_submit_cli("submission.csv", "initial rn26d 128px", comp)
    except Exception as e:
        print(f"Skipping kaggle submission API call: {e}")



## === cell 18
if not iskaggle:
    try:
        push_notebook(
            "jhoward",
            "first-steps-road-to-the-top-part-1",
            title="First Steps: Road to the Top, Part 1",
            file="first-steps-road-to-the-top-part-1.ipynb",
            competition=comp,
            private=False,
            gpu=True,
        )
    except Exception as e:
        print(f"Skipping push_notebook: {e}")
