# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        input/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> working/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> working/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

# 5. Code solution

## === cell 0
import pandas as pd, numpy as np
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from fastai.vision.all import *
import warnings, os
import torchvision.models as models  # needed for pretrained architecture
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")




## === cell 1
class RocAucMetric(Metric):
    "Computes ROC‑AUC on the validation set for binary classification."

    def __init__(self):
        self.reset()

    def reset(self):
        self.preds, self.targs = [], []

    def accumulate(self, learn):
        self.preds.append(learn.pred.detach())
        self.targs.append(learn.yb[0].detach())

    def value(self):
        preds = torch.cat(self.preds)
        targs = torch.cat(self.targs)
        probs = F.softmax(preds, dim=1)[:, 1].cpu().numpy()
        auc = roc_auc_score(targs.cpu().numpy(), probs)
        return auc

    @property
    def name(self):
        return "roc_auc"




## === cell 2
possible_roots = [
    Path("/kaggle/input/aerial-cactus-identification"),
    Path("/kaggle/working/aerial-cactus-identification"),
    Path("/kaggle/input"),
    Path("/kaggle/working"),
]
root = None
for p in possible_roots:
    if (p / "train.csv").exists():
        root = p
        break
assert root is not None, "train.csv not found in any expected location"

train_df = pd.read_csv(root / "train.csv")
test_df = pd.read_csv(root / "sample_submission.csv")



## === cell 3
SZ = 128  # image size
BS = 64  # batch size
tfms = aug_transforms(
    do_flip=True,
    flip_vert=True,
    max_rotate=10.0,
    max_zoom=1.1,
    max_lighting=0.2,
    max_warp=0.2,
    p_affine=0.75,
    p_lighting=0.75,
)


def get_x(r):
    """Return a valid image path; fall back to a black placeholder only if the file truly does not exist."""
    for sub in ["train", "test"]:
        img_path = root / sub / f"{r['id']}"
        if img_path.exists():
            return img_path
    arr = np.zeros((SZ, SZ, 3), dtype=np.uint8)
    return PILImage.create(arr)


dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_x=get_x,
    get_y=ColReader("has_cactus"),
    splitter=RandomSplitter(valid_pct=0.20, seed=42),
    item_tfms=Resize(SZ),
    batch_tfms=tfms,
)

dls = dblock.dataloaders(train_df, path=root, bs=BS)

test_items = pd.DataFrame({"id": test_df["id"]})
test_dl = dls.test_dl(test_items)



## === cell 4
arch = models.densenet161
learn = cnn_learner(
    dls,
    arch,
    loss_func=CrossEntropyLoss(),
    metrics=[accuracy, RocAucMetric()],
    pretrained=True,
)
learn.fine_tune(15, base_lr=1e-4)



## === cell 5
preds, _ = learn.get_preds(dl=test_dl)
probs = F.softmax(preds, dim=1)[:, 1]  # probability of class "1"
test_df["has_cactus"] = probs.cpu().numpy()
submission_path = Path("submission.csv")
test_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path.resolve()}")
