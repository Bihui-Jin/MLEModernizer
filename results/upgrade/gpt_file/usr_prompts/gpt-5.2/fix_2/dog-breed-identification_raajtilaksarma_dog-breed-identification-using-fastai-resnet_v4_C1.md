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
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.7

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.38103

# 6. Current score

0.67075

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.67075) has done: 'I update the notebook from deprecated fastai v0.7 imports/APIs to the installed fastai v2 equivalents so it actually runs in this environment. I keep the same core approach (ResNet101 transfer learning, image resizing/augmentation, fit, then test-time augmentation and averaged probabilities) while fixing dataset paths, CUDA handling, and submission-column alignment to `sample_submission.csv`. I also remove the unsafe symlink/cleanup logic that’s causing filesystem errors and instead read images directly from the provided Kaggle input folders. Finally, I guarantee that a valid `submission.csv` is written with the exact required header/order.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

BASE_INPUT = "/kaggle/input/dog-breed-identification"
TRAIN_DIR = os.path.join(BASE_INPUT, "train")
TEST_DIR = os.path.join(BASE_INPUT, "test")
LABELS_CSV = os.path.join(BASE_INPUT, "labels.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_INPUT, "sample_submission.csv")

print("Exists TRAIN_DIR:", os.path.exists(TRAIN_DIR))
print("Exists TEST_DIR :", os.path.exists(TEST_DIR))
print("Exists LABELS_CSV:", os.path.exists(LABELS_CSV))
print("Exists SAMPLE_SUB_CSV:", os.path.exists(SAMPLE_SUB_CSV))
print(
    "Train images:", len(os.listdir(TRAIN_DIR)) if os.path.exists(TRAIN_DIR) else "NA"
)
print("Test images :", len(os.listdir(TEST_DIR)) if os.path.exists(TEST_DIR) else "NA")

labels_df = pd.read_csv(LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

print(labels_df.head())
print(sample_sub.head())
print("Num classes (sample submission):", sample_sub.shape[1] - 1)



## === cell 1
from fastai.vision.all import *
import torch

set_seed(42, reproducible=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available(), "| device:", device)



## === cell 2
sz = 224
bs = 128
arch = resnet101

assert labels_df["id"].is_unique, "labels.csv ids must be unique"
assert set(sample_sub.columns[1:]).issuperset(
    set(labels_df["breed"].unique())
), "Mismatch between labels and sample submission classes"



## === cell 3
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock(vocab=list(sample_sub.columns[1:]))),
    get_x=ColReader("fname"),
    get_y=ColReader("breed"),
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=Resize(sz),
    batch_tfms=aug_transforms(max_zoom=1.1),  # similar spirit to original max_zoom=1.1
)

train_df = labels_df.copy()
train_df["fname"] = train_df["id"].apply(lambda x: os.path.join(TRAIN_DIR, f"{x}.jpg"))

missing = train_df.loc[~train_df["fname"].map(os.path.exists)]
if len(missing) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing)} train image files, e.g. {missing.iloc[0]['fname']}"
    )

dls = dblock.dataloaders(train_df, bs=bs, shuffle=True)

print("dls vocab size:", len(dls.vocab))
print("Example vocab head:", dls.vocab[:5])



## === cell 4
learn = vision_learner(dls, arch, metrics=[accuracy]).to_fp32()

lr_min, lr_steep = learn.lr_find(suggest_funcs=(minimum, steep), show_plot=False)
lr = float(lr_min) if lr_min is not None else 1e-2
print("Suggested lr (minimum):", lr)



## === cell 5
learn.fine_tune(5, base_lr=lr)



## === cell 6
test_ids = sample_sub["id"].tolist()
test_files = [os.path.join(TEST_DIR, f"{i}.jpg") for i in test_ids]

missing_test = [p for p in test_files if not os.path.exists(p)]
if missing_test:
    raise FileNotFoundError(
        f"Missing {len(missing_test)} test image files, e.g. {missing_test[0]}"
    )

test_dl = learn.dls.test_dl(test_files, with_labels=False)



## === cell 7
tta_preds, _ = learn.tta(
    dl=test_dl, n=4, beta=0.0
)  # beta=0 to avoid mixing with non-TTA preds
probs = tta_preds.cpu().numpy()

print("probs shape:", probs.shape)



## === cell 8
probs = np.clip(probs, 1e-12, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)

sub = pd.DataFrame(probs, columns=dls.vocab)
sub.insert(0, "id", test_ids)

sub = sub[sample_sub.columns]

print(sub.head())
print("Submission shape:", sub.shape)



## === cell 9
out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "| bytes:", os.path.getsize(out_path))
print("Columns match sample submission:", list(sub.columns) == list(sample_sub.columns))
