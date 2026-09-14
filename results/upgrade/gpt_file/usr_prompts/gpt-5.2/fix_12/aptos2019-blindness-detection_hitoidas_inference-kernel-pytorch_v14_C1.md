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

fastai==2.8.5
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.8534645190996363

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the path logic that incorrectly tries to load non-existent `model*.pth` files and remove the brittle `os.listdir()` indexing that points to `train.zip` instead of a folder. Because this environment has fastai v2 (where `ImageList/ImageDataBunch` no longer exist), I replace the deprecated fastai v1 inference pipeline with a minimal PyTorch + torchvision inference loop that preserves the core idea: run four models and average their predicted classes. To ensure end-to-end execution, I add a safe fallback: if pretrained model files are not present in the input folder, the code still generate a valid `submission.csv` by predicting class `0` for all test images (score be low but it yield a valid submission). The script always write `./submission.csv` with the required `id_code,diagnosis` columns aligned to `test.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with the fallback path that predicts all zeros because no `model1.pth`…`model4.pth` files exist in the dataset, so the smallest score-improving change is to remove dependency on missing checkpoints. To keep the same overall approach (image classifier inference → class labels), I add a lightweight, deterministic baseline that uses a pretrained ImageNet model (available via `torchvision`) to produce non-constant predictions, which should move QWK meaningfully above 0.0. I also fix the “average 4 argmaxes then floor” aggregation to a majority-vote scheme when multiple models exist (same core ensemble idea, but less biased downward). The script still always write a valid `submission.csv` aligned to `test.csv`.'
- What this solution (achieved 0.01936) has done: 'Your 0.0 score is consistent with the fallback path using an ImageNet-pretrained model whose final layer is randomly initialized (so predictions are effectively random wrt DR classes). To move the score toward your target with minimal changes and without changing the overall “single-image CNN inference → class 0–4” core logic, I keep your pipeline but replace the baseline with a deterministic, label-free severity heuristic that maps each image’s brightness/contrast statistics to 5 ordinal bins (better aligned with QWK than random). This keeps end-to-end execution, requires no training/checkpoints, and still writes a valid `submission.csv` in the required format. If the `model*.pth` files ever exist, your original ensemble path is preserved unchanged.'
- What this solution (achieved 0.07985) has done: 'Your current score is low because the fallback heuristic is only weakly correlated with DR severity; the smallest way to move toward your target (without training or changing the overall “image → 5-class label” logic) is to replace that heuristic with a slightly more DR-relevant, still label-free proxy: estimate how much of the fundus is “bright lesions / glare” and how much is “dark/vessel texture,” then bin that severity score into 5 quantiles. I keep your existing model-loading/ensemble path unchanged (so if model*.pth ever exists, it behaves as before), and only adjust the missing-checkpoint fallback. I also make the heuristic run faster and more stable by using grayscale, center-cropping, and a small blur + edge/contrast measure. The script still run end-to-end and write `./submission.csv` with the required columns aligned to `test.csv`.'
- What this solution (achieved 0.11978) has done: 'Your current score (0.07985) is far below the target (0.85346), so we should improve the fallback path (used when `model*.pth` are missing) while keeping your overall “image → 5-class label” inference logic intact. The smallest meaningful gain is to make the heuristic more fundus-specific: mask out the black background, normalize within the circular field-of-view, and then compute a severity proxy from (a) vessel/lesion texture (high-frequency energy) and (b) bright lesion fraction. To keep it stable and label-free, we still map a single continuous severity score to 5 classes using fixed quantiles computed on the test set (same semantics as your current binning, just a better score). No changes are made to the model-loading ensemble path; we only replace the heuristic feature computation.'
- What this solution (achieved 0.16455) has done: 'Your current score (0.11978) is far below the target (0.85346), so we should improve the fallback path that’s actually being used when `model*.pth` files are missing. To keep core logic unchanged (image → scalar severity proxy → fixed 5-bin discretization), I replace the heuristic score with a more fundus-specific, label-free measure: (1) mask the circular field-of-view, (2) normalize within-mask, and (3) compute a severity proxy from edge/texture energy plus bright-lesion fraction and dark-structure fraction. I keep the same quantile binning on the test set (same semantics as your current approach) and keep the existing model-loading/majority-vote path untouched. This is a minimal change localized to the heuristic and should move QWK upward without introducing training or external files.'
- What this solution (achieved -0.23047) has done: 'I fix the runtime error by importing `ImageFilter` (used by the heuristic path) so the pipeline can complete inference when the `model*.pth` files are missing. I also add a small safety fallback so `finalPreds` is always defined even if an unexpected image read error occurs, ensuring a valid `submission.csv` is always written. These changes are score-neutral in intent (they don’t alter the model/heuristic logic) and are strictly to unblock end-to-end execution and submission generation.'
- What this solution (achieved -0.23248) has done: 'Your current score is far below the target, so we should improve the fallback path that’s actually being used (missing `model*.pth`). To keep the same core logic (single-image deterministic inference → continuous severity proxy → discretize into 0–4), I keep your existing fundus masking/texture features but change only the final discretization: instead of binning by the test-set quantiles, we bin to match the **train label distribution** using a monotonic rank-based mapping, which is typically much better aligned with QWK for ordinal problems. This is label-free w.r.t. test and doesn’t change the overall approach; it just calibrates class frequencies to be realistic. I also make the heuristic slightly more stable by adding a tiny epsilon guard and ensuring images are always read in RGB/L consistently (no change to model path).'
- What this solution (achieved 0.0) has done: 'Main runtime is dominated by the heuristic fallback path: it extracts a heavy image feature for every test image and (worse) for every train image, then does a 4-threshold brute-force grid search with repeated QWK computation. To keep identical semantics while fitting in 600s, the changes cache computed features to disk (so the expensive extraction runs at most once), parallelize feature extraction across CPU cores, and replace the nested Python-loop QWK with a fully vectorized confusion-matrix implementation that is mathematically identical. Additionally, a few safe I/O/data-loader tweaks reduce overhead (vectorized path joins, persistent workers), without changing model inference logic or outputs beyond negligible FP differences.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

INPUT_ROOT = "/kaggle/input"
COMP_DIR = os.path.join(INPUT_ROOT, "aptos2019-blindness-detection")

print("Listing /kaggle/input:", os.listdir(INPUT_ROOT))
print("Competition dir exists:", os.path.exists(COMP_DIR))
if os.path.exists(COMP_DIR):
    print("Listing competition dir (head):", sorted(os.listdir(COMP_DIR))[:30])



## === cell 1
test_csv_path = os.path.join(COMP_DIR, "test.csv")
test_img_dir = os.path.join(COMP_DIR, "test_images")

if not os.path.exists(test_csv_path):
    test_csv_path = os.path.join(INPUT_ROOT, "test.csv")
if not os.path.exists(test_img_dir):
    test_img_dir = os.path.join(INPUT_ROOT, "test_images")

assert os.path.exists(test_csv_path), f"test.csv not found at {test_csv_path}"
assert os.path.exists(test_img_dir), f"test_images folder not found at {test_img_dir}"

test_df = pd.read_csv(test_csv_path)
assert "id_code" in test_df.columns

ids = test_df["id_code"].astype(str).values
test_df["image_path"] = np.char.add(np.char.add(test_img_dir + os.sep, ids), ".png")

missing = (~pd.Series(test_df["image_path"]).map(os.path.exists)).sum()
print("Test rows:", len(test_df), "Missing images:", int(missing))



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_54/3560306849.py in <cell line: 0>()
     15 
     16 ids = test_df["id_code"].astype(str).values
---> 17 test_df["image_path"] = np.char.add(np.char.add(test_img_dir + os.sep, ids), ".png")
     18 
     19 missing = (~pd.Series(test_df["image_path"]).map(os.path.exists)).sum()

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U56' and 'object' (the few cases where this used to work often lead to incorrect results).

## === cell 2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

infer_tfms = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class APTOSDataset(Dataset):
    def __init__(self, df, tfms):
        self.paths = df["image_path"].values
        self.tfms = tfms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        img = self.tfms(img)
        return img


ds = APTOSDataset(test_df, infer_tfms)

num_workers = min(8, (os.cpu_count() or 2))
dl = DataLoader(
    ds,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2,
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'image_path'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_54/2141753254.py in <cell line: 0>()
     31 
     32 
---> 33 ds = APTOSDataset(test_df, infer_tfms)
     34 
     35 # Speed: use more workers + persistent workers to reduce dataloader startup overhead each epoch/iteration.

/tmp/ipykernel_54/2141753254.py in __init__(self, df, tfms)
     18 class APTOSDataset(Dataset):
     19     def __init__(self, df, tfms):
---> 20         self.paths = df["image_path"].values
     21         self.tfms = tfms
     22 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'image_path'

## === cell 3
def try_load_model(path):
    if not os.path.exists(path):
        return None
    try:
        m = torch.load(path, map_location=device)
        return m
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None


model_paths = [
    os.path.join(COMP_DIR, "model1.pth"),
    os.path.join(COMP_DIR, "model2.pth"),
    os.path.join(COMP_DIR, "model3.pth"),
    os.path.join(COMP_DIR, "model4.pth"),
]

models = []
for mp in model_paths:
    m = try_load_model(mp)
    models.append(m)

all_present = all(m is not None for m in models)
print("Models found:", [m is not None for m in models])



## === cell 4
import torch.nn as nn
import torchvision
from PIL import ImageFilter  # heuristic path uses ImageFilter.BoxBlur
import hashlib


@torch.no_grad()
def predict_logits_single_model(model, dataloader):
    model.eval()
    model.to(device)
    logits_all = []
    for xb in dataloader:
        xb = xb.to(device, non_blocking=True)
        out = model(xb)
        if isinstance(out, (tuple, list)):
            out = out[0]
        logits_all.append(out.detach().float().cpu())
    return torch.cat(logits_all, dim=0)


def majority_vote(preds_2d):
    preds_2d = np.asarray(preds_2d, dtype=np.int64)
    n_models, n_samples = preds_2d.shape
    out = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        counts = np.bincount(preds_2d[:, i], minlength=5)
        out[i] = int(np.argmax(counts))
    return out.tolist()


def _otsu_threshold_u8(u8):
    hist = np.bincount(u8.ravel(), minlength=256).astype(np.float64)
    total = u8.size
    if total == 0:
        return 0
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]

    eps = 1e-12
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + eps)
    t = int(np.argmax(sigma_b2))
    return t


def _load_train_df_and_paths():
    train_csv_path = os.path.join(COMP_DIR, "train.csv")
    train_img_dir = os.path.join(COMP_DIR, "train_images")
    if not os.path.exists(train_csv_path):
        train_csv_path = os.path.join(INPUT_ROOT, "train.csv")
    if not os.path.exists(train_img_dir):
        train_img_dir = os.path.join(INPUT_ROOT, "train_images")

    if not (os.path.exists(train_csv_path) and os.path.exists(train_img_dir)):
        return None, None

    tr = pd.read_csv(train_csv_path)
    if "id_code" not in tr.columns or "diagnosis" not in tr.columns:
        return None, None

    tr = tr.copy()
    ids = tr["id_code"].astype(str).values
    tr["image_path"] = np.char.add(np.char.add(train_img_dir + os.sep, ids), ".png")
    tr = tr[pd.Series(tr["image_path"]).map(os.path.exists).values].reset_index(
        drop=True
    )
    return tr, train_img_dir


def _qwk(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape
    N = n_classes

    mask = (y_true >= 0) & (y_true < N) & (y_pred >= 0) & (y_pred < N)
    yt = y_true[mask]
    yp = y_pred[mask]
    if yt.size == 0:
        return 0.0

    O = np.zeros((N, N), dtype=np.float64)
    np.add.at(O, (yt, yp), 1.0)

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    sO = O.sum()
    sE = E.sum()
    if sE > 0:
        E = E / sE * sO

    idx = np.arange(N, dtype=np.float64)
    W = (idx[:, None] - idx[None, :]) ** 2 / ((N - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den <= 0:
        return 0.0
    return 1.0 - num / den


def _extract_severity_feature(image_path):
    img = Image.open(image_path).convert("RGB")

    w, h = img.size
    s = int(min(w, h) * 0.92)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    img = img.resize((384, 384), resample=Image.BILINEAR)

    arr_rgb = np.asarray(img, dtype=np.uint8)
    g = (
        0.299 * arr_rgb[..., 0] + 0.587 * arr_rgb[..., 1] + 0.114 * arr_rgb[..., 2]
    ).astype(np.uint8)

    t = _otsu_threshold_u8(g)
    mask = g > max(10, t)  # avoid background
    if mask.mean() < 0.10:
        mask = g > 12
    if not mask.any():
        return 0.0

    g_img = Image.fromarray(g, mode="L")
    bg = np.asarray(g_img.filter(ImageFilter.BoxBlur(15)), dtype=np.float32) + 1.0
    gf = g.astype(np.float32)
    corr = gf / bg  # roughly flattens illumination (unitless)

    cm = corr[mask]
    med = float(np.median(cm))
    mad = float(np.median(np.abs(cm - med)) + 1e-6)
    z = (corr - med) / (1.4826 * mad)

    yy, xx = np.mgrid[0 : g.shape[0], 0 : g.shape[1]]
    cy = (g.shape[0] - 1) / 2.0
    cx = (g.shape[1] - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (min(cx, cy) + 1e-6)
    radial = np.clip(1.0 - rr, 0.0, 1.0)
    wmask = mask.astype(np.float32) * (0.25 + 0.75 * radial)

    a = corr.astype(np.float32)
    gx = a[:, 1:] - a[:, :-1]
    gy = a[1:, :] - a[:-1, :]
    gmag = np.zeros_like(a, dtype=np.float32)
    gmag[:, 1:] += np.abs(gx)
    gmag[1:, :] += np.abs(gy)
    tex = float((gmag * wmask).sum() / (wmask.sum() + 1e-6))

    bright = float(((z > 2.0).astype(np.float32) * wmask).sum() / (wmask.sum() + 1e-6))
    dark = float(((z < -1.8).astype(np.float32) * wmask).sum() / (wmask.sum() + 1e-6))

    gf01 = gf / 255.0
    glare = float(
        (((gf01 > 0.93).astype(np.float32)) * wmask).sum() / (wmask.sum() + 1e-6)
    )

    score = 3.2 * tex + 2.8 * bright + 1.6 * dark + 0.8 * glare
    if not np.isfinite(score):
        score = 0.0
    return float(score)


def _predict_from_thresholds(scores, thr4):
    s = np.asarray(scores, dtype=np.float32)
    t = np.asarray(thr4, dtype=np.float32)
    y = np.zeros(len(s), dtype=np.int64)
    y[s > t[0]] = 1
    y[s > t[1]] = 2
    y[s > t[2]] = 3
    y[s > t[3]] = 4
    return y


def _fit_thresholds_max_qwk(scores, y_true, n_grid=80):
    scores = np.asarray(scores, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int64)

    def quantile_split(sc):
        qs = np.quantile(sc, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
        return _predict_from_thresholds(sc, qs)

    pred_pos = quantile_split(scores)
    pred_neg = quantile_split(-scores)
    qwk_pos = _qwk(y_true, pred_pos)
    qwk_neg = _qwk(y_true, pred_neg)
    use_neg = qwk_neg > qwk_pos
    sc = -scores if use_neg else scores

    q = np.linspace(0.05, 0.95, n_grid)
    cand = np.unique(np.quantile(sc, q).astype(np.float32))
    if cand.size < 10:
        cand = np.unique(sc)

    best_qwk = -1e9
    best_thr = None

    max_cand = 60
    if cand.size > max_cand:
        cand = np.unique(
            np.quantile(sc, np.linspace(0.05, 0.95, max_cand)).astype(np.float32)
        )

    m = cand.size
    for i in range(0, m - 3):
        t0 = cand[i]
        for j in range(i + 1, m - 2):
            t1 = cand[j]
            for k in range(j + 1, m - 1):
                t2 = cand[k]
                for l in range(k + 1, m):
                    t3 = cand[l]
                    pred = _predict_from_thresholds(sc, (t0, t1, t2, t3))
                    val = _qwk(y_true, pred)
                    if val > best_qwk:
                        best_qwk = val
                        best_thr = (float(t0), float(t1), float(t2), float(t3))

    if best_thr is None:
        best_thr = tuple(
            np.quantile(sc, [0.2, 0.4, 0.6, 0.8]).astype(np.float32).tolist()
        )

    return {"use_neg": bool(use_neg), "thr": best_thr, "train_qwk": float(best_qwk)}


def _feature_cache_path(tag, paths):
    h = hashlib.md5()
    h.update(tag.encode("utf-8"))
    h.update(str(len(paths)).encode("utf-8"))
    for p in paths[:3].tolist() if hasattr(paths, "tolist") else list(paths)[:3]:
        h.update(str(p).encode("utf-8"))
    return os.path.join("/kaggle/working", f"severity_feat_cache_{h.hexdigest()}.npz")


def _extract_features_cached(paths, tag):
    paths = np.asarray(paths, dtype=object)
    cache_path = _feature_cache_path(tag, paths)
    if os.path.exists(cache_path):
        try:
            z = np.load(cache_path, allow_pickle=False)
            scores = z["scores"].astype(np.float32, copy=False)
            if scores.shape[0] == paths.shape[0]:
                return scores
        except Exception:
            pass

    from concurrent.futures import ProcessPoolExecutor

    n_workers = min(8, max(1, (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        scores_list = list(ex.map(_extract_severity_feature, paths, chunksize=16))
    scores = np.asarray(scores_list, dtype=np.float32)

    try:
        np.savez_compressed(cache_path, scores=scores)
    except Exception as e:
        print("WARNING: failed to write cache:", e)
    return scores


def heuristic_predict_from_paths(df_paths):
    scores = _extract_features_cached(df_paths, tag="test")

    tr, _ = _load_train_df_and_paths()
    if tr is not None and len(tr) > 0:
        tr_scores = _extract_features_cached(tr["image_path"].values, tag="train")
        y_true = tr["diagnosis"].astype(int).values
        calib = _fit_thresholds_max_qwk(tr_scores, y_true, n_grid=80)
        print("Heuristic calibration:", calib)
    else:
        qs = np.quantile(scores, [0.2, 0.4, 0.6, 0.8]).astype(np.float32).tolist()
        calib = {"use_neg": False, "thr": tuple(qs), "train_qwk": None}
        print(
            "WARNING: train.csv/images not found; using test-quantile thresholds:",
            calib["thr"],
        )

    sc = -scores if calib["use_neg"] else scores
    preds = _predict_from_thresholds(sc, calib["thr"]).astype(np.int64)
    return preds.tolist()


finalPreds = None

try:
    if all_present:
        labels_list = []
        for i, m in enumerate(models, start=1):
            print(f"Predicting from model{i}...")
            logits = predict_logits_single_model(m, dl)
            pred = torch.argmax(logits, dim=1).numpy().astype(int).tolist()
            labels_list.append(pred)

        finalPreds = majority_vote(np.array(labels_list))
    else:
        print(
            "WARNING: One or more model*.pth files are missing. Using a deterministic heuristic baseline."
        )
        finalPreds = heuristic_predict_from_paths(test_df["image_path"].values)
except Exception as e:
    print(
        "ERROR during inference; falling back to all-zeros to still produce a submission:",
        repr(e),
    )
    finalPreds = [0] * len(test_df)

finalPreds = [int(min(4, max(0, p))) for p in finalPreds]
print("Preds length:", len(finalPreds), "Unique:", sorted(set(finalPreds)))



## === cell 5
submission = pd.DataFrame(
    {"id_code": test_df["id_code"].astype(str).values, "diagnosis": finalPreds}
)
assert submission.shape[0] == test_df.shape[0]
assert list(submission.columns) == ["id_code", "diagnosis"]

out_path = "./submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())
