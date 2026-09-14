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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
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
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.7004940282523127

# 6. Current score

0.39293

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.09193) has done: 'I remove the broken dependency on a missing `../input/alexnet` checkpoint and instead instantiate an AlexNet classifier directly from `torchvision` so inference can run end-to-end. I also fix notebook-only syntax (`%matplotlib inline`) so the script runs as a Kaggle Python script, and make test image loading deterministic by reading IDs from `test.csv` rather than `os.listdir()` ordering. Finally, I ensure predictions are generated on the available GPU if present, and that the output `submission.csv` has exactly the required columns and 367 rows.'
- What this solution (achieved 0.29038) has done: 'Your score is very low because the model’s final layer was replaced with a new random 5-class head, so predictions are essentially random with respect to DR severity. To move toward the target with minimal change and without altering the core inference approach, I keep AlexNet and ImageNet preprocessing but stop reinitializing the classifier head. Instead, I use the pretrained 1000-class logits and map them deterministically into 5 ordinal classes via fixed percentile binning on a scalar “severity proxy” derived from the logits (expected class index), which typically gives a materially better-than-random ordering. I also ensure the submission is strictly aligned to `test.csv` order and write `submission.csv` with the required columns.'
- What this solution (achieved 0.39293) has done: 'Your current mapping from ImageNet outputs to 5 DR classes uses *test-set quantile binning*, which forces an artificial uniform class distribution and typically harms quadratic weighted kappa because APTOS labels are highly imbalanced toward class 0. To move the score upward toward the 0.70 target while keeping the same model/inference core, I keep AlexNet and the same “severity proxy” computation, but calibrate the 5-class bin thresholds using the *training label distribution* (no leakage: labels only used to set global priors). Concretely, we compute severity scores for train images too, then choose four cutpoints so predicted class proportions on train match the true label proportions, and apply those cutpoints to test. This is a minimal change (only post-processing/calibration) and is expected to improve kappa substantially versus uniform binning.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms, models

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1
INPUT_DIR = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train_images")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(TEST_CSV), f"Missing {TEST_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

assert {"id_code", "diagnosis"}.issubset(train_df.columns)
assert "id_code" in test_df.columns

print("Train rows:", len(train_df), "Test rows:", len(test_df))
print("Train label distribution:")
print(train_df["diagnosis"].value_counts().sort_index())



## === cell 2
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
model = model.to(device)
model.eval()




## === cell 3
class ImageIdDataset(torch.utils.data.Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_code = self.df.loc[idx, "id_code"]
        path = os.path.join(self.root_dir, f"{id_code}.png")
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, id_code




## === cell 4
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = ImageIdDataset(train_df[["id_code"]], TRAIN_IMG_DIR, test_transform)
test_dataset = ImageIdDataset(test_df[["id_code"]], TEST_IMG_DIR, test_transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




## === cell 5
def compute_severity_scores(data_loader):
    id_codes = []
    severity_scores = []

    idx_vec = torch.arange(1000, device=device, dtype=torch.float32).view(1, -1)

    with torch.no_grad():
        for imgs, ids in data_loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)  # [B, 1000]
            probs = torch.softmax(logits, dim=1)
            sev = (probs * idx_vec).sum(dim=1)  # [B], in [0,999] approximately

            id_codes.extend(list(ids))
            severity_scores.extend(
                sev.detach().cpu().numpy().astype(np.float32).tolist()
            )

    return np.asarray(id_codes), np.asarray(severity_scores, dtype=np.float32)


train_ids, train_sev = compute_severity_scores(train_loader)
test_ids, test_sev = compute_severity_scores(test_loader)

train_sev_df = pd.DataFrame({"id_code": train_ids, "severity": train_sev})
train_sev_df = train_df[["id_code", "diagnosis"]].merge(
    train_sev_df, on="id_code", how="left"
)
assert (
    train_sev_df["severity"].notna().all()
), "Missing train severity scores (image load/inference issue?)"



## === cell 6
label_counts = train_sev_df["diagnosis"].value_counts().sort_index()
label_props = (
    (label_counts / label_counts.sum()).reindex([0, 1, 2, 3, 4]).fillna(0.0).to_numpy()
)

cum_props = np.cumsum(label_props)  # length 5, ends at 1.0
cut_quantiles = cum_props[:4]  # q at boundaries after class 0..3

cut_quantiles = np.clip(cut_quantiles, 1e-6, 1 - 1e-6)

bins = np.quantile(
    train_sev_df["severity"].to_numpy(dtype=np.float32), cut_quantiles
).astype(np.float32)

test_diags = np.digitize(test_sev, bins, right=False).astype(int)

pred_df = pd.DataFrame({"id_code": test_ids, "diagnosis": test_diags})

sub = test_df[["id_code"]].merge(pred_df, on="id_code", how="left")
assert len(sub) == len(test_df), "Submission row count mismatch vs test.csv"
assert sub["diagnosis"].notna().all(), "Found missing predictions"
sub["diagnosis"] = sub["diagnosis"].astype(int)

sub_path = "./submission.csv"
sub.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(sub.head())



## === cell 7
sample_path = os.path.join(INPUT_DIR, "sample_submission.csv")
sample = pd.read_csv(sample_path)
print("Sample columns:", list(sample.columns), "rows:", len(sample))
print("Submission columns:", list(sub.columns), "rows:", len(sub))

print("Bins (train-calibrated):", bins)
print("Test predicted label distribution:")
print(sub["diagnosis"].value_counts().sort_index())
