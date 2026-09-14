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

-0.23248

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the path logic that incorrectly tries to load non-existent `model*.pth` files and remove the brittle `os.listdir()` indexing that points to `train.zip` instead of a folder. Because this environment has fastai v2 (where `ImageList/ImageDataBunch` no longer exist), I replace the deprecated fastai v1 inference pipeline with a minimal PyTorch + torchvision inference loop that preserves the core idea: run four models and average their predicted classes. To ensure end-to-end execution, I add a safe fallback: if pretrained model files are not present in the input folder, the code still generate a valid `submission.csv` by predicting class `0` for all test images (score be low but it yield a valid submission). The script always write `./submission.csv` with the required `id_code,diagnosis` columns aligned to `test.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with the fallback path that predicts all zeros because no `model1.pth`…`model4.pth` files exist in the dataset, so the smallest score-improving change is to remove dependency on missing checkpoints. To keep the same overall approach (image classifier inference → class labels), I add a lightweight, deterministic baseline that uses a pretrained ImageNet model (available via `torchvision`) to produce non-constant predictions, which should move QWK meaningfully above 0.0. I also fix the “average 4 argmaxes then floor” aggregation to a majority-vote scheme when multiple models exist (same core ensemble idea, but less biased downward). The script still always write a valid `submission.csv` aligned to `test.csv`.'
- What this solution (achieved 0.01936) has done: 'Your 0.0 score is consistent with the fallback path using an ImageNet-pretrained model whose final layer is randomly initialized (so predictions are effectively random wrt DR classes). To move the score toward your target with minimal changes and without changing the overall “single-image CNN inference → class 0–4” core logic, I keep your pipeline but replace the baseline with a deterministic, label-free severity heuristic that maps each image’s brightness/contrast statistics to 5 ordinal bins (better aligned with QWK than random). This keeps end-to-end execution, requires no training/checkpoints, and still writes a valid `submission.csv` in the required format. If the `model*.pth` files ever exist, your original ensemble path is preserved unchanged.'
- What this solution (achieved 0.07985) has done: 'Your current score is low because the fallback heuristic is only weakly correlated with DR severity; the smallest way to move toward your target (without training or changing the overall “image → 5-class label” logic) is to replace that heuristic with a slightly more DR-relevant, still label-free proxy: estimate how much of the fundus is “bright lesions / glare” and how much is “dark/vessel texture,” then bin that severity score into 5 quantiles. I keep your existing model-loading/ensemble path unchanged (so if model*.pth ever exists, it behaves as before), and only adjust the missing-checkpoint fallback. I also make the heuristic run faster and more stable by using grayscale, center-cropping, and a small blur + edge/contrast measure. The script still run end-to-end and write `./submission.csv` with the required columns aligned to `test.csv`.'
- What this solution (achieved 0.11978) has done: 'Your current score (0.07985) is far below the target (0.85346), so we should improve the fallback path (used when `model*.pth` are missing) while keeping your overall “image → 5-class label” inference logic intact. The smallest meaningful gain is to make the heuristic more fundus-specific: mask out the black background, normalize within the circular field-of-view, and then compute a severity proxy from (a) vessel/lesion texture (high-frequency energy) and (b) bright lesion fraction. To keep it stable and label-free, we still map a single continuous severity score to 5 classes using fixed quantiles computed on the test set (same semantics as your current binning, just a better score). No changes are made to the model-loading ensemble path; we only replace the heuristic feature computation.'
- What this solution (achieved 0.16455) has done: 'Your current score (0.11978) is far below the target (0.85346), so we should improve the fallback path that’s actually being used when `model*.pth` files are missing. To keep core logic unchanged (image → scalar severity proxy → fixed 5-bin discretization), I replace the heuristic score with a more fundus-specific, label-free measure: (1) mask the circular field-of-view, (2) normalize within-mask, and (3) compute a severity proxy from edge/texture energy plus bright-lesion fraction and dark-structure fraction. I keep the same quantile binning on the test set (same semantics as your current approach) and keep the existing model-loading/majority-vote path untouched. This is a minimal change localized to the heuristic and should move QWK upward without introducing training or external files.'
- What this solution (achieved -0.23047) has done: 'I fix the runtime error by importing `ImageFilter` (used by the heuristic path) so the pipeline can complete inference when the `model*.pth` files are missing. I also add a small safety fallback so `finalPreds` is always defined even if an unexpected image read error occurs, ensuring a valid `submission.csv` is always written. These changes are score-neutral in intent (they don’t alter the model/heuristic logic) and are strictly to unblock end-to-end execution and submission generation.'
- What this solution (achieved -0.23248) has done: 'Your current score is far below the target, so we should improve the fallback path that’s actually being used (missing `model*.pth`). To keep the same core logic (single-image deterministic inference → continuous severity proxy → discretize into 0–4), I keep your existing fundus masking/texture features but change only the final discretization: instead of binning by the test-set quantiles, we bin to match the **train label distribution** using a monotonic rank-based mapping, which is typically much better aligned with QWK for ordinal problems. This is label-free w.r.t. test and doesn’t change the overall approach; it just calibrates class frequencies to be realistic. I also make the heuristic slightly more stable by adding a tiny epsilon guard and ensuring images are always read in RGB/L consistently (no change to model path).'

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
test_df["image_path"] = (
    test_df["id_code"]
    .astype(str)
    .apply(lambda x: os.path.join(test_img_dir, f"{x}.png"))
)

missing = (~test_df["image_path"].apply(os.path.exists)).sum()
print("Test rows:", len(test_df), "Missing images:", int(missing))



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
dl = DataLoader(
    ds,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




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


def _load_train_label_distribution():
    train_csv_path = os.path.join(COMP_DIR, "train.csv")
    if not os.path.exists(train_csv_path):
        train_csv_path = os.path.join(INPUT_ROOT, "train.csv")
    if not os.path.exists(train_csv_path):
        print(
            "WARNING: train.csv not found; falling back to uniform label distribution."
        )
        return np.ones(5, dtype=np.float64) / 5.0

    tr = pd.read_csv(train_csv_path)
    if "diagnosis" not in tr.columns:
        print(
            "WARNING: train.csv has no diagnosis; falling back to uniform label distribution."
        )
        return np.ones(5, dtype=np.float64) / 5.0

    y = tr["diagnosis"].astype(int).values
    counts = np.bincount(y, minlength=5).astype(np.float64)
    p = counts / max(1.0, counts.sum())
    return p


def _rank_to_train_distribution_classes(scores, train_p):
    scores = np.asarray(scores, dtype=np.float32)
    n = scores.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)

    order = np.argsort(scores, kind="mergesort")  # stable for ties
    target = np.maximum(0, np.floor(train_p * n + 1e-9).astype(int))
    rem = n - int(target.sum())
    if rem != 0:
        frac = (train_p * n) - np.floor(train_p * n)
        idx = np.argsort(-frac)  # descending
        step = 1 if rem > 0 else -1
        for k in range(abs(rem)):
            target[idx[k % 5]] += step
    target = np.clip(target, 0, n)
    diff = n - int(target.sum())
    if diff != 0:
        target[0] = int(np.clip(target[0] + diff, 0, n))

    pred = np.zeros(n, dtype=np.int64)
    start = 0
    for c in range(5):
        cnt = int(target[c])
        if cnt <= 0:
            continue
        idxs = order[start : start + cnt]
        pred[idxs] = c
        start += cnt
    if start < n:
        pred[order[start:]] = 4
    return pred


def heuristic_predict_from_paths(df_paths):
    vals = []

    for p in df_paths:
        img = Image.open(p).convert("RGB")

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
            vals.append(0.0)
            continue

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

        bright = float(
            ((z > 2.0).astype(np.float32) * wmask).sum() / (wmask.sum() + 1e-6)
        )
        dark = float(
            ((z < -1.8).astype(np.float32) * wmask).sum() / (wmask.sum() + 1e-6)
        )

        gf01 = gf / 255.0
        glare = float(
            (((gf01 > 0.93).astype(np.float32)) * wmask).sum() / (wmask.sum() + 1e-6)
        )

        score = 3.2 * tex + 2.8 * bright + 1.6 * dark + 0.8 * glare
        if not np.isfinite(score):
            score = 0.0
        vals.append(score)

    vals = np.asarray(vals, dtype=np.float32)

    train_p = _load_train_label_distribution()
    preds = _rank_to_train_distribution_classes(vals, train_p).astype(np.int64)
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
