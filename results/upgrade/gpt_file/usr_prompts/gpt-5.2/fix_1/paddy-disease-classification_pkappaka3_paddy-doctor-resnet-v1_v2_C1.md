# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.13182

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
if os.getenv("COLAB_RELEASE_TAG"):
    competition_name = "paddy-disease-classification"
    from google.colab import drive
    drive.mount("/content/drive")


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_11/1696710646.py in <cell line: 0>()
      4     # Mount your Google Drive.
      5     from google.colab import drive
----> 6     drive.mount("/content/drive")

/usr/local/lib/python3.11/dist-packages/google/colab/drive.py in mount(mountpoint, force_remount, timeout_ms, readonly)
     98 def mount(mountpoint, force_remount=False, timeout_ms=120000, readonly=False):
     99   """Mount your Google Drive at the specified mountpoint path."""
--> 100   return _mount(
    101       mountpoint,
    102       force_remount=force_remount,

/usr/local/lib/python3.11/dist-packages/google/colab/drive.py in _mount(mountpoint, force_remount, timeout_ms, ephemeral, readonly)
    116   """Internal helper to mount Google Drive."""
    117   if not _os.path.exists('/var/colab/hostname'):
--> 118     raise NotImplementedError(
    119         'Mounting drive is unsupported in this environment. Use PyDrive2'
    120         ' instead. See examples at'

NotImplementedError: Mounting drive is unsupported in this environment. Use PyDrive2 instead. See examples at https://colab.research.google.com/notebooks/io.ipynb#scrollTo=7taylj9wpsA2.

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
if os.getenv("COLAB_RELEASE_TAG"):
    setup_comp('paddy-disease-classification', 'train.csv')


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/4276927122.py in <cell line: 0>()
      1 if os.getenv("COLAB_RELEASE_TAG"):
      2     # Set up Kaggle credentials (you'll need to provide your kaggle.json file)
----> 3     setup_comp('paddy-disease-classification', 'train.csv')

/usr/local/lib/python3.11/dist-packages/fastkaggle/core.py in setup_comp(competition, install)
     36     else:
     37         path = Path(competition)
---> 38         api = import_kaggle()
     39         if not path.exists():
     40             import zipfile

/usr/local/lib/python3.11/dist-packages/fastkaggle/core.py in import_kaggle()
     24         if not os.environ['KAGGLE_USERNAME']: raise Exception("Please insert your Kaggle username and key into Kaggle secrets")
     25         os.environ['KAGGLE_KEY'] = sec.get_secret("kaggle_key")
---> 26     from kaggle import api
     27     return api
     28 

/usr/local/lib/python3.11/dist-packages/kaggle/__init__.py in <module>
      4 
      5 api = KaggleApi()
----> 6 api.authenticate()

/usr/local/lib/python3.11/dist-packages/kaggle/api/kaggle_api_extended.py in authenticate(self)
    432         return
    433       else:
--> 434         raise IOError('Could not find {}. Make sure it\'s located in'
    435                       ' {}. Or use the environment method. See setup'
    436                       ' instructions at'

OSError: Could not find kaggle.json. Make sure it's located in /root/.kaggle. Or use the environment method. See setup instructions at https://github.com/Kaggle/kaggle-api/

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
