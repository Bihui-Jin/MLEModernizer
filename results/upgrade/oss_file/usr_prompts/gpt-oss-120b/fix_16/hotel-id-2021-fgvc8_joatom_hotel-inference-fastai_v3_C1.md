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
Identify hotels from images.

## Metric
Mean Average Precision @ 5 (MAP@5)

## Submission Format
For each image in the test set, you must predict a space-delimited list of hotel IDs that could match that image. The first ID should be the most relevant one and the last the least relevant one. The file should contain a header and have the following format:

```
image,hotel_id
99e91ad5f2870678.jpg,36363 53586 18807 64314 60181
b5cc62ab665591a9.jpg,36363 53586 18807 64314 60181
d5664a972d5a644b.jpg,36363 53586 18807 64314 60181
```

## Dataset
**train.csv** - The training set metadata.

- `image` - The image ID.

- `chain` - An ID code for the hotel chain. A `chain` of zero (0) indicates that the hotel is either not part of a chain or the chain is not known. This field is not available for the test set. The number of hotels per chain varies widely.

- `hotel_id` - The hotel ID. The target class.

- `timestamp` - When the image was taken. Provided for the training set only.

**sample_submission.csv** - A sample submission file in the correct format.

- `image` The image ID

- `hotel_id` The hotel ID. The target class.

**train_images** - The training set contains 97000+ images from around 7700 hotels from across the globe. All of the images for each hotel chain are in a dedicated subfolder for that chain.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 13,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

dill==0.4.0
fastai==2.8.5
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
            train/
                train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        input/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
                    test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
            train/
                train/
                    train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        working/
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
```

-> data/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> data/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> input/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> input/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> (stopped after 10 files for performance)

# 5. Target score

0.5748358171255361

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from fastai.vision.all import *
import dill, torch, warnings, os, pathlib, concurrent.futures, PIL.Image as Image

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
torch.set_num_threads(os.cpu_count() or 1)



## === cell 1
DATA_ROOT = Path("../input/hotel-id-2021-fgvc8")
TRAIN_CSV = DATA_ROOT / "train.csv"
TEST_SUBMIT = DATA_ROOT / "sample_submission.csv"
TRAIN_IMG_ROOT = DATA_ROOT / "train_images"
TEST_IMG_ROOT = DATA_ROOT / "test_images"

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_SUBMIT)

train_df["img_path"] = (
    TRAIN_IMG_ROOT / train_df["chain"].astype(str) / train_df["image"]
).astype(str)
test_df["img_path"] = (TEST_IMG_ROOT / test_df["image"]).astype(str)



## === cell 2
workers = min(16, os.cpu_count() or 1)  # more workers for parallel I/O
BATCH_SIZE = 1024 if torch.cuda.is_available() else 512
IMG_SIZE = 128

vocab = sorted(train_df["hotel_id"].unique())
label2idx = {lbl: i for i, lbl in enumerate(vocab)}
train_labels = torch.tensor(
    train_df["hotel_id"].map(label2idx).values, dtype=torch.long
)


def load_image(path: str) -> torch.Tensor:
    """Read JPEG, resize, and return a float tensor in C×H×W format."""
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # H×W×C, 0‑1 float
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # C×H×W
    return tensor


with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
    train_tensors = list(ex.map(load_image, train_df["img_path"].tolist()))

train_images = torch.stack(train_tensors)  # (N, C, H, W)

full_dataset = TensorDataset(train_images, train_labels)


def random_split(dataset, valid_pct=0.05, seed=42):
    """Return two SubsetRandomSamplers for train/valid."""
    rng = torch.Generator().manual_seed(seed)
    n = len(dataset)
    indices = torch.randperm(n, generator=rng).tolist()
    split = int(valid_pct * n)
    return indices[split:], indices[:split]  # train_idx, valid_idx


train_idx, valid_idx = random_split(full_dataset, valid_pct=0.05, seed=42)

train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

train_dl = DataLoader(
    full_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=workers,
    pin_memory=True,
    persistent_workers=True,
)
valid_dl = DataLoader(
    full_dataset,
    batch_size=BATCH_SIZE,
    sampler=valid_sampler,
    num_workers=workers,
    pin_memory=True,
    persistent_workers=True,
)

dls = DataLoaders(train_dl, valid_dl)
dls.c = len(vocab)  # number of classes
dls.vocab = vocab  # needed for decoding predictions later
dls.after_batch.add(Normalize.from_stats(*imagenet_stats))



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4222432924.py in <cell line: 0>()
     28 
     29 # create a TensorDataset and split into train/valid
---> 30 full_dataset = TensorDataset(train_images, train_labels)
     31 
     32 

NameError: name 'TensorDataset' is not defined

## === cell 3
learn = cnn_learner(dls, resnet34, metrics=top_k_accuracy, pretrained=True)
if torch.cuda.is_available():
    learn.to_fp16()  # mixed precision on GPU only
learn.fine_tune(1, freeze_epochs=0)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2099087042.py in <cell line: 0>()
----> 1 learn = cnn_learner(dls, resnet34, metrics=top_k_accuracy, pretrained=True)
      2 if torch.cuda.is_available():
      3     learn.to_fp16()  # mixed precision on GPU only
      4 learn.fine_tune(1, freeze_epochs=0)
      5 

NameError: name 'dls' is not defined

## === cell 4
test_dl = learn.dls.test_dl(
    test_df["img_path"],
    bs=BATCH_SIZE,
    num_workers=workers,
    persistent_workers=True,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2891234640.py in <cell line: 0>()
      1 # test DataLoader – still loads from disk (only ~10k images)
----> 2 test_dl = learn.dls.test_dl(
      3     test_df["img_path"],
      4     bs=BATCH_SIZE,
      5     num_workers=workers,

NameError: name 'learn' is not defined

## === cell 5
probs, _ = learn.get_preds(dl=test_dl, with_loss=False)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1328127129.py in <cell line: 0>()
----> 1 probs, _ = learn.get_preds(dl=test_dl, with_loss=False)
      2 

NameError: name 'learn' is not defined

## === cell 6
preds_idx = probs.topk(5)[1]



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3593509576.py in <cell line: 0>()
----> 1 preds_idx = probs.topk(5)[1]
      2 

NameError: name 'probs' is not defined

## === cell 7
preds = [" ".join(map(str, [learn.dls.vocab[p] for p in row])) for row in preds_idx]



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3234044973.py in <cell line: 0>()
----> 1 preds = [" ".join(map(str, [learn.dls.vocab[p] for p in row])) for row in preds_idx]
      2 

NameError: name 'preds_idx' is not defined

## === cell 8
submission = pd.read_csv(TEST_SUBMIT)  # keep original ordering and headers
submission["hotel_id"] = preds
submission.to_csv("submission.csv", index=False)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2166833458.py in <cell line: 0>()
      1 submission = pd.read_csv(TEST_SUBMIT)  # keep original ordering and headers
----> 2 submission["hotel_id"] = preds
      3 submission.to_csv("submission.csv", index=False)
      4 

NameError: name 'preds' is not defined

## === cell 9
submission.head()
