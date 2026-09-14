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

0.8917

# 6. Current score

0.12414

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.12759) has done: 'Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.

The crash happens because `google.colab.drive.mount()` is being called in a non-Colab environment where it raises `NotImplementedError`. The minimal fix is to guard the mount call so it only runs when the runtime is actually Colab-capable (i.e., `google.colab` is importable and the Colab host marker exists). This preserves the original intent (mount Drive on Colab) while preventing the unsupported call here. No other cells are changed, and `competition_name` remains defined when applicable.'
- What this solution (achieved 0.12414) has done: 'The crash happens because `setup_comp(...)` tries to authenticate with the Kaggle API, but `kaggle.json` is not present in this environment (and `COLAB_RELEASE_TAG` may still be set), causing an `OSError`. Since the dataset is already available locally under the provided `/kaggle/input/...` and `/kaggle/data/...` paths, the minimal fix is to only call `setup_comp` when `kaggle.json` exists. This preserves the original behavior on Colab (where credentials are mounted) while preventing the unnecessary Kaggle API call (and crash) elsewhere. No downstream variables are introduced/changed, so cell 6 remains compatible.'

# 9. Code solution

## === cell 0
import os

if os.getenv("COLAB_RELEASE_TAG") and os.path.exists("/var/colab/hostname"):
    competition_name = "paddy-disease-classification"
    from google.colab import drive

    drive.mount("/content/drive")


## === cell 1
! pip install kaggle --quiet


## === cell 2
if os.getenv("COLAB_RELEASE_TAG"):
    kaggle_creds_path = "/content/drive/MyDrive/kaggle.json"
    ! mkdir ~/.kaggle
    ! cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
    ! chmod 600 ~/.kaggle/kaggle.json


## === cell 3
!pip install fastkaggle --quiet


## === cell 4
from fastai.vision.all import *
from fastkaggle import *  # for easy Kaggle dataset access


## === cell 5
if os.getenv("COLAB_RELEASE_TAG") and os.path.exists(
    os.path.expanduser("~/.kaggle/kaggle.json")
):
    setup_comp("paddy-disease-classification", "train.csv")


## === cell 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


## === cell 7
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


## === cell 8
!ls ../input/paddy-disease-classification


## === cell 9
if os.getenv("COLAB_RELEASE_TAG"):
    path = Path('paddy-disease-classification')
    train_path = path/'train_images'
    test_path = path/'test_images'
else:
    path = Path('../input/paddy-disease-classification')
    train_path = path/'train_images'
    test_path = path/'test_images'


## === cell 10
def get_subset_items(path):
    files = get_image_files(path)
    sample_file_count = 10407
    print(sample_file_count, len(files))
    return L(files).shuffle()[:sample_file_count]


## === cell 11
paddy_block = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items= get_subset_items, #get_image_files
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(460),  # resize to slightly larger than target
    batch_tfms=[*aug_transforms(size=224, min_scale=0.9),
                Normalize.from_stats(*imagenet_stats)]
)


## === cell 12
dls = paddy_block.dataloaders(train_path, bs=200)


## === cell 13

learn = vision_learner(dls, 'resnet26d', metrics=error_rate).to_fp16()

learn.fine_tune(5, 3e-3)

learn.save('paddy_model')


## === cell 14
learn.show_results()


## === cell 15
test_files = get_image_files(test_path)
test_files.sort()
test_dl = learn.dls.test_dl(test_files, with_labels=False, bs=200)


## === cell 16
probs,_,idxs = learn.get_preds(dl=test_dl, with_decoded=True)


## === cell 17
mapping = dict(enumerate(dls.vocab))
results = pd.Series(idxs.numpy(), name="idxs").map(mapping)


## === cell 18
ss = pd.read_csv(path/'sample_submission.csv')
ss['label'] = results
ss.to_csv('submission.csv', index=False)
!head submission.csv
