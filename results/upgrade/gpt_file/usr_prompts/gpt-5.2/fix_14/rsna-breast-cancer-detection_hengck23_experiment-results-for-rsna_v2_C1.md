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
Detect breast cancer in mammograms.

## Metric
[Probabilistic F1 score](https://aclanthology.org/2020.eval4nlp-1.9.pdf) (pF1). This extension of the traditional F score accepts probabilities instead of binary classifications. 

With pX as the probabilistic version of X:

$$
pF_1 = 2 \frac{pPrecision \cdot pRecall}{pPrecision + pRecall}
$$

where:

$$
pPrecision = \frac{pTP}{pTP + pFP}
$$

$$
pRecall = \frac{pTP}{TP + FN}
$$

## Submission Format
For each `prediction_id`, you should predict the likelihood of cancer in the corresponding `cancer` column. The submission file should have the following format:

```
prediction_id,cancer
0-L,0
0-R,0.5
0-R,0.5
1-L,1
...
# Dataset

**[train/test]_images/[patient_id]/[image_id].dcm** The mammograms, in dicom format. You can expect roughly 8,000 patients in the hidden test set. There are usually but not always 4 images per patient. Note that many of the images use the jpeg 2000 format which may you may need special libraries to load.

**sample_submission.csv** A valid sample submission.

**[train/test].csv** Metadata for each patient and image. Only the first few rows of the test set are available for download.

- `site_id` - ID code for the source hospital.
- `patient_id` - ID code for the patient.
- `image_id` - ID code for the image.
- `laterality` - Whether the image is of the left or right breast.
- `view` - The orientation of the image. The default for a screening exam is to capture two views per breast.
- `age` - The patient's age in years.
- `implant` - Whether or not the patient had breast implants. Site 1 only provides breast implant information at the patient level, not at the breast level.
- `density` - A rating for how dense the breast tissue is, with A being the least dense and D being the most dense. Extremely dense tissue can make diagnosis more difficult. Only provided for train.
- `machine_id` - An ID code for the imaging device.
- `cancer` - Whether or not the breast was positive for malignant cancer. The target value. Only provided for train.
- `biopsy` - Whether or not a follow-up biopsy was performed on the breast. Only provided for train.
- `invasive` - If the breast is positive for cancer, whether or not the cancer proved to be invasive. Only provided for train.
- `BIRADS` - 0 if the breast required follow-up, 1 if the breast was rated as negative for cancer, and 2 if the breast was rated as normal. Only provided for train.
- `prediction_id` - The ID for the matching submission row. Multiple images will share the same prediction ID. Test only.
- `difficult_negative_case` - True if the case was unusually difficult. Only provided for train.

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pydicom==3.0.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
timm==1.0.19
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
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        input/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        working/
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
```

-> data/rsna-breast-cancer-detection/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/rsna-breast-cancer-detection/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/rsna-breast-cancer-detection/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> data/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> (stopped after 10 files for performance)

# 5. Target score

0.0555775173506362

# 6. Current score

0.02212

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the immediate failure by making checkpoint discovery robust (searching `/kaggle/input` for the expected `.pth` file) and gracefully falling back if it isn’t available. To keep core logic intact, the model, preprocessing, and inference aggregation stay the same; only checkpoint loading is made reliable, plus a small state-dict key normalization to handle common `module.`/`state_dict` packaging variants. Finally, I ensure the written `submission.csv` exactly matches `sample_submission.csv` ordering and row count (one row per `prediction_id`), which prevents invalid submissions due to missing/extra IDs.'
- What this solution (achieved 0.02212) has done: 'Your 0.0 score is coming from the fallback path that writes all-zero predictions when no checkpoint is found/loaded. To move toward the target with minimal changes, I (1) make checkpoint discovery deterministic and more likely to find the intended RSNA weight file by searching for the exact filename first, then RSNA-related `.pth` files, and (2) if no checkpoint exists, write a calibrated constant probability (the train positive rate) instead of zeros (still valid and typically >0 pF1). I also keep your submission alignment logic but add a safety clip to `[0,1]` to avoid any rare numeric issues that could invalidate scoring. Core model, preprocessing, aggregation, and inference loop remain unchanged.'
- What this solution (achieved 0.02212) has done: 'Your current score (0.02212) is below the target (0.05558), so we should improve toward the target by making the smallest changes that legitimately increase pF1 without changing your model/inference core. The biggest likely win with minimal risk is post-processing calibration: your per-`prediction_id` probabilities are currently a plain mean of per-image probabilities, which is often miscalibrated for pF1; we add a tiny, validation-based calibration step using the provided train.csv metadata only (no images) by optimizing a single scalar transform (power + scale) and optional floor, then apply it to test predictions. This preserves the architecture, checkpoint loading, preprocessing, and inference loop; it only changes how probabilities are aggregated/calibrated to better match the metric. If no checkpoint is found, we keep your constant-probability fallback unchanged.'
- What this solution (achieved 0.02212) has done: 'Your current score (0.02212) is well below the target (0.05558), so we should nudge performance upward with minimal, low-risk changes that don’t alter the model or inference loop. The biggest likely issue is that your train-metadata calibration currently merges the test `prediction_id` strings (like `"10116_L"`) with a train-derived `prediction_id` built as `"10116_L"` (underscore), while the competition uses hyphen format (`"10116-L"`), so calibration likely does nothing. I fix calibration to build train `prediction_id` with the same hyphen convention as the official submission IDs, and also add a safe fallback: if no merge matches, rebuild using underscore and try again (covers either format). Everything else (checkpoint loading, DICOM preprocessing, model forward, and per-prediction_id mean aggregation) remains unchanged.'
- What this solution (achieved 0.02212) has done: 'Your current score (0.02212) is below the target (0.05558), so we should improve pF1 with the smallest change that’s likely to help without touching the model/inference core. The most impactful low-risk fix is to make calibration actually operate in the correct “space”: pF1 is computed per-breast (`prediction_id`), but your calibration currently learns from train labels only and can still be mismatched by how multiple images map into one `prediction_id`. I (1) build a “train-side” proxy predictor using only train metadata (no images) to learn an optimal single scalar mapping from your model’s per-`prediction_id` probabilities to pF1 targets, and (2) expand the calibration search slightly (still tiny) and add a deterministic tie-break to avoid accidental worse choices. Everything else (DICOM read, model, weights, inference loop, mean aggregation) stays the same, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.02212) has done: 'Your current score (0.02212) is below the target (0.05558), so we should improve pF1 with the smallest, low-risk change that doesn’t touch the model/inference core. The biggest likely win is fixing the aggregation mismatch: pF1 is evaluated per `prediction_id`, but your predictions are currently a plain mean across images; switching to a calibrated “noisy-or” aggregation (still a simple probability aggregation) typically increases recall-sensitive pF1 for multi-view exams. To stay minimal and stable, I keep your existing metadata-based calibration step, but feed it the new per-`prediction_id` probability and also fall back to mean if a `prediction_id` has only one image. Everything else (checkpoint discovery/loading, DICOM preprocessing, network forward, dataloader loop, submission alignment) remains unchanged, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.02212) has done: 'To move your pF1 upward toward the target with minimal risk, I keep your model/inference loop unchanged and focus on two low-impact post-processing fixes: (1) make the per-`prediction_id` aggregation slightly less aggressive than pure noisy-or by introducing a single mixing parameter `alpha` (no new model, just aggregation), and (2) slightly tighten the calibration grid around near-identity transforms so we don’t over-warp probabilities when train/test differ. This should improve recall-sensitive pF1 versus mean-only while avoiding the common overconfidence that can hurt precision with full noisy-or. Everything still runs end-to-end, uses the same inputs/paths, and writes a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.02212) has done: 'We keep your model, DICOM preprocessing, and inference loop unchanged, and only adjust the post-processing in a way that typically improves pF1 (your score is below target, so we want a modest lift). The minimal, highest-impact tweak is to make the aggregation mixing parameter `alpha` data-driven using only train labels (per-breast) to choose among a tiny set of plausible values, instead of using a fixed `alpha=0.65` that may be mismatched. To preserve evaluation semantics and avoid overfitting, we select `alpha` by maximizing pF1 on train breast-level labels using your already-computed per-image probabilities distribution only indirectly (we don’t have train image predictions), so we instead pick a conservative `alpha` using only the number-of-images-per-breast distribution (which we can compute from train/test metadata) and a small heuristic grid; then keep your existing metadata-based calibration grid tight. This is a small, stable change that should move the score upward toward the target while staying within Kaggle constraints and still producing a valid `submission.csv`.'
- What this solution (achieved 0.02212) has done: 'Your current score (0.02212) is well below the target (0.05558), so we should move upward with the smallest change that can materially improve pF1 without touching the model or inference loop. The highest-impact minimal fix is to correct the pF1 computation used for calibration: the current `pfbeta_np` has an incorrect `pFP` term (it double-counts positives), which makes the calibration choose suboptimal transforms. I replace it with the standard RSNA pF1 implementation (still purely post-processing) and slightly broaden the *near-identity* calibration grid so it can find a better mapping while remaining stable. Everything else (checkpoint loading, DICOM reading, network forward, aggregation, submission alignment) stays the same.'
- What this solution (achieved 0.02212) has done: 'Your current score (0.02212) is below the target (0.05558), so we want a modest, low-risk lift without touching the model/inference core. The biggest minimal win for pF1 is usually probability calibration plus a slightly more recall-friendly aggregation; you already do both, so the smallest next step is to tune the aggregation strength `alpha` using train breast-level labels by simulating multi-image aggregation via grouping within each breast (still metadata-only, no image access, no loop changes). I keep your tempered noisy-or aggregation and your calibration grid intact, but make `alpha` chosen from a tiny grid by maximizing correct pF1 on train after applying the same aggregation+calibration pipeline. This preserves architecture, preprocessing, checkpoint logic, and the inference loop; it only improves the deterministic post-processing choice that directly affects the metric.'
- What this solution (achieved 0.02212) has done: 'Your current score (0.02212) is below the target (0.05558), so we need a modest, low-risk lift without touching the model, DICOM preprocessing, or inference loop. The largest likely issue is that `_choose_alpha_by_train_pf1()` is currently “tuning” alpha using **test predictions merged with train labels**, but train and test `prediction_id` sets don’t overlap, so it silently falls back to a default alpha and gives no benefit. I keep your tempered noisy-or + calibration approach intact, but change alpha selection to use only information that actually overlaps: choose alpha by maximizing pF1 on a **patient-level holdout split of train labels using only train metadata**, then apply that alpha to test aggregation. This is a minimal post-processing-only change that should move pF1 upward toward the target while preserving core logic.'
- What this solution (achieved 0.02212) has done: 'Your current score (0.02212) is below the target (0.05558), so we want a modest uplift with minimal risk and without touching the model, DICOM preprocessing, or inference loop. The largest likely low-impact win for pF1 is reducing systematic under-confidence: your train-metadata calibration currently optimizes a transform but doesn’t allow a simple global “logit shift” that often improves recall-sensitive pF1 while keeping ranking similar. I keep your existing aggregation and power/scale/floor grid intact, but add a tiny extra 1D logit-shift search (near zero) during calibration and apply it to all predictions. Everything still writes a valid `submission.csv` aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os, sys, glob
import pandas as pd
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
from timeit import default_timer as timer

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
import torch.nn as nn
import torch.nn.functional as F

import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__)
print("timm :", timm.__version__)
print("device:", device)


def time_to_str(t, mode="min"):
    if mode == "min":
        t = int(t) / 60
        hr = t // 60
        min_ = t % 60
        return "%2d hr %02d min" % (hr, min_)
    elif mode == "sec":
        t = int(t)
        min_ = t // 60
        sec = t % 60
        return "%2d min %02d sec" % (min_, sec)
    else:
        raise NotImplementedError


try:
    from timm.models.senet import seresnext26d_32x4d
except Exception:
    seresnext26d_32x4d = None




## === cell 1
image_size = 256

is_local = False
if is_local:
    csv_file = "/kaggle/input/rsna-breast-mammography-00/valid_df.fold0.csv"
    dcm_dir = "/kaggle/input/rsna-breast-cancer-detection/train_images"
else:
    csv_file = "/kaggle/input/rsna-breast-cancer-detection/test.csv"
    dcm_dir = "/kaggle/input/rsna-breast-cancer-detection/test_images"

test_df = pd.read_csv(csv_file)
test_df.loc[:, "i"] = np.arange(len(test_df))
if is_local:
    test_df.loc[:, "prediction_id"] = (
        test_df.patient_id.astype(str) + "_" + test_df.laterality
    )

print("test_df", test_df.shape)
print(test_df.head())


def read_dicom(dcm_file):
    dicom = pydicom.dcmread(dcm_file)

    img = (
        apply_voi_lut(dicom.pixel_array, dicom)
        if hasattr(dicom, "VOILUTSequence") or hasattr(dicom, "WindowWidth")
        else dicom.pixel_array
    )
    img = img.astype(np.float32)

    mn, mx = float(np.min(img)), float(np.max(img))
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img = np.zeros_like(img, dtype=np.float32)

    if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = 1.0 - img

    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return img


def read_data(df):
    image = []
    for _, d in df.iterrows():
        m = read_dicom(f"{dcm_dir}/{d.patient_id}/{d.image_id}.dcm")
        image.append(m)
    image = np.stack(image)
    return image


class RsnaDataset(Dataset):
    def __init__(self, df):
        patient_id = sorted(df.patient_id.unique())
        self.patient_id = patient_id
        self.length = len(patient_id)
        self.df = df

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        patient_id = self.patient_id[index]
        df = self.df[self.df.patient_id == patient_id].reset_index(drop=True)
        image = read_data(df)

        r = {}
        r["index"] = index
        r["df"] = df
        r["patient_id"] = patient_id
        r["image"] = torch.from_numpy(image).float()
        r["num"] = len(df)
        return r


tensor_key = ["image"]


def null_collate(batch):
    d = {}
    key = batch[0].keys()
    for k in key:
        v = [b[k] for b in batch]
        if k in tensor_key:
            v = torch.cat(v, 0)
        d[k] = v
    return d




## === cell 2
class RGB(nn.Module):
    IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
    IMAGE_RGB_STD = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("std", torch.ones(1, 3, 1, 1))
        self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(self.mean.shape)
        self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(self.std.shape)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb = RGB()

        if seresnext26d_32x4d is not None:
            self.encoder = seresnext26d_32x4d(pretrained=False)
        else:
            self.encoder = timm.create_model("seresnext26d_32x4d", pretrained=False)

        self.weight = nn.Linear(2048, 1)
        self.cancer = nn.Linear(2048 * 2, 1)

    def forward(self, batch):
        x = batch["image"]
        x = x.unsqueeze(1).expand(-1, 3, -1, -1)
        x = self.rgb(x)

        e = self.encoder
        x = e.conv1(x)
        x = e.bn1(x)
        x = e.act1(x)
        x = e.maxpool(x)
        for blk in [e.layer1, e.layer2, e.layer3, e.layer4]:
            x = blk(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1, 3)

        num = batch["num"]
        batch_size = len(num)
        s = torch.split_with_sizes(x, num)

        feature = []
        for b in range(batch_size):
            w = self.weight(s[b])
            w = F.softmax(w, 0)
            pool = (s[b] * w).sum(0, keepdims=True)
            pool = pool.expand(num[b], -1)
            f = torch.cat([s[b], pool], -1)
            feature.append(f)
        feature = torch.cat(feature, 0)

        cancer = self.cancer(feature).reshape(-1)
        output = {"cancer": torch.sigmoid(cancer)}
        return output




## === cell 3
def _find_checkpoint(preferred_path: str):
    """
    Change rationale (score improvement toward target):
    - Missing checkpoint can cause weak fallback predictions.
    - Keep your robust checkpoint discovery as-is.
    """
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    filename = os.path.basename(preferred_path) if preferred_path else ""
    candidates = []

    if filename:
        candidates = glob.glob(f"/kaggle/input/**/{filename}", recursive=True)
        candidates = [c for c in candidates if os.path.isfile(c)]
        if candidates:
            candidates = sorted(candidates, key=lambda p: (len(p), p))
            return candidates[0]

    rsna_candidates = glob.glob("/kaggle/input/**/*.pth", recursive=True)
    rsna_candidates = [
        c
        for c in rsna_candidates
        if os.path.isfile(c)
        and (
            "rsna" in c.lower()
            or "breast" in c.lower()
            or "mamm" in c.lower()
            or "cancer" in c.lower()
            or "model" in c.lower()
        )
    ]
    if rsna_candidates:
        rsna_candidates = sorted(rsna_candidates, key=lambda p: (len(p), p))
        return rsna_candidates[0]

    candidates = glob.glob("/kaggle/input/**/*.pth", recursive=True)
    candidates = [c for c in candidates if os.path.isfile(c)]
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda p: (len(p), p))
    return candidates[0]


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError("Unsupported checkpoint format")


def _strip_prefix_from_state_dict(state_dict, prefix):
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix) :]: v for k, v in state_dict.items()}


def _train_positive_rate_default():
    train_csv = "/kaggle/input/rsna-breast-cancer-detection/train.csv"
    if os.path.exists(train_csv):
        df = pd.read_csv(train_csv, usecols=["cancer"])
        p = float(df["cancer"].mean())
        return float(np.clip(p, 1e-6, 1 - 1e-6))
    return 0.01


def pfbeta_np(labels, predictions, beta=1.0):
    labels = np.asarray(labels).astype(np.int64)
    predictions = np.asarray(predictions).astype(np.float64)
    predictions = np.clip(predictions, 0.0, 1.0)

    y_true_count = labels.sum()
    if y_true_count == 0:
        return 0.0

    ctp = predictions[labels == 1].sum()
    cfp = predictions[labels == 0].sum()
    beta2 = beta * beta

    c_precision = ctp / (ctp + cfp + 1e-12)
    c_recall = ctp / (y_true_count + 1e-12)

    if c_precision > 0 and c_recall > 0:
        return (
            (1.0 + beta2)
            * (c_precision * c_recall)
            / (beta2 * c_precision + c_recall + 1e-12)
        )
    return 0.0


def _logit(p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_with_train_metadata(pred_pid_df: pd.DataFrame):
    """
    Change rationale (score improvement toward target):
    - pF1 is recall-sensitive; many RSNA mammography models are systematically under-confident.
    - Minimal, stable boost: in addition to your existing near-identity (power,scale,floor),
      also allow a tiny global logit-shift (bias) chosen by the same train-metadata pF1 grid search.
    - This does not change the model/inference core; it's pure post-processing and stays near-identity.
    """
    train_csv = "/kaggle/input/rsna-breast-cancer-detection/train.csv"
    if (not os.path.exists(train_csv)) or pred_pid_df is None or pred_pid_df.empty:
        return pred_pid_df, None

    train = pd.read_csv(train_csv, usecols=["patient_id", "laterality", "cancer"])
    train["prediction_id_dash"] = (
        train["patient_id"].astype(str) + "-" + train["laterality"].astype(str)
    )
    train["prediction_id_us"] = (
        train["patient_id"].astype(str) + "_" + train["laterality"].astype(str)
    )

    y_dash = train.groupby("prediction_id_dash", as_index=False)["cancer"].max()
    y_dash = y_dash.rename(columns={"prediction_id_dash": "prediction_id"})

    calib = pred_pid_df.merge(
        y_dash, on="prediction_id", how="inner", suffixes=("", "_y")
    )
    join_used = "dash"
    if calib.empty:
        y_us = train.groupby("prediction_id_us", as_index=False)["cancer"].max()
        y_us = y_us.rename(columns={"prediction_id_us": "prediction_id"})
        calib = pred_pid_df.merge(
            y_us, on="prediction_id", how="inner", suffixes=("", "_y")
        )
        join_used = "underscore"

    if calib.empty:
        print(
            "Calibration skipped: no matching prediction_id between preds and train (dash/underscore)."
        )
        return pred_pid_df, None

    y_true = calib["cancer_y"].astype(np.int64).values
    p = calib["cancer"].astype(np.float64).values

    powers = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35]
    scales = [0.85, 0.90, 0.97, 1.00, 1.03, 1.10, 1.20]
    floors = [0.0, 0.001, 0.002, 0.004, 0.006]

    shifts = [-0.40, -0.25, -0.10, 0.0, 0.10, 0.25, 0.40]

    def tie_breaker(a, s, f, b):
        return abs(a - 1.0) + abs(s - 1.0) + 5.0 * abs(f - 0.0) + 0.5 * abs(b - 0.0)

    best_score = -1.0
    best_params = (1.0, 1.0, 0.0, 0.0)
    best_tb = 1e9

    p_clip = np.clip(p, 0.0, 1.0)
    for a in powers:
        pa = np.power(p_clip, a)
        for s in scales:
            ps = np.clip(pa * s, 0.0, 1.0)
            for f in floors:
                pf = np.clip(ps + f, 0.0, 1.0)
                l = _logit(pf)
                for b in shifts:
                    pc = np.clip(_sigmoid(l + b), 0.0, 1.0)
                    score = pfbeta_np(y_true, pc, beta=1.0)
                    tb = tie_breaker(a, s, f, b)
                    if (score > best_score + 1e-12) or (
                        abs(score - best_score) <= 1e-12 and tb < best_tb
                    ):
                        best_score = score
                        best_params = (a, s, f, b)
                        best_tb = tb

    a, s, f, b = best_params
    params = {
        "join_used": join_used,
        "power": float(a),
        "scale": float(s),
        "floor": float(f),
        "logit_shift": float(b),
        "train_pf1": float(best_score),
        "calib_points": int(len(calib)),
    }

    p_all = pred_pid_df["cancer"].astype(np.float64).values
    p_all = np.clip(p_all, 0.0, 1.0)
    p_all = np.power(p_all, a)
    p_all = np.clip(p_all * s + f, 0.0, 1.0)
    p_all = np.clip(_sigmoid(_logit(p_all) + b), 0.0, 1.0)

    out = pred_pid_df.copy()
    out["cancer"] = p_all
    print("Calibration params selected:", params)
    return out, params


def _aggregate_pred_per_prediction_id(
    pred_img_df: pd.DataFrame, alpha: float
) -> pd.DataFrame:
    """
    Change rationale (score improvement toward target):
    - Keep your tempered noisy-or aggregation unchanged.
    """
    df = pred_img_df.copy()
    p = np.clip(df["cancer"].astype(np.float64).values, 0.0, 1.0)
    df["_q"] = np.clip(1.0 - p, 1e-9, 1.0)

    g = df.groupby("prediction_id", as_index=False)

    agg = g.agg(
        q_log_sum=(
            "_q",
            lambda x: float(np.log(np.asarray(x, dtype=np.float64)).sum()),
        ),
        n=("cancer", "size"),
    )
    log_prod_q = agg["q_log_sum"].astype(np.float64).values
    p_tempered_noisy_or = 1.0 - np.exp(alpha * log_prod_q)

    agg_out = pd.DataFrame(
        {
            "prediction_id": agg["prediction_id"].values,
            "cancer": np.clip(p_tempered_noisy_or, 0.0, 1.0),
        }
    )
    return agg_out


def _choose_alpha_by_train_holdout_pf1() -> float:
    """
    Change rationale (score improvement toward target):
    - Keep your current meaningful alpha-selection (no test/train leakage attempt).
    """
    train_csv = "/kaggle/input/rsna-breast-cancer-detection/train.csv"
    if not os.path.exists(train_csv):
        return 0.70

    train = pd.read_csv(train_csv, usecols=["patient_id", "laterality", "cancer"])
    train["prediction_id"] = (
        train["patient_id"].astype(str) + "-" + train["laterality"].astype(str)
    )
    y = train.groupby("prediction_id", as_index=False)["cancer"].max()
    y["patient_id"] = (
        y["prediction_id"].str.split("-", n=1, expand=True)[0].astype(np.int64)
    )

    patients = np.array(sorted(y["patient_id"].unique()), dtype=np.int64)
    rng = np.random.default_rng(12345)
    rng.shuffle(patients)
    n_valid = max(1, int(0.20 * len(patients)))
    valid_pat = set(patients[:n_valid].tolist())

    y_valid = y[y["patient_id"].isin(valid_pat)].copy()
    y_train = y[~y["patient_id"].isin(valid_pat)].copy()
    if y_valid.empty or y_train.empty:
        return 0.70

    base_p = float(np.clip(y_train["cancer"].mean(), 1e-6, 1 - 1e-6))

    pred_img_df = pd.DataFrame(
        {
            "prediction_id": np.repeat(y_valid["prediction_id"].values, 2),
            "cancer": base_p,
        }
    )

    alpha_grid = [0.60, 0.65, 0.70, 0.75, 0.80]
    best_alpha = 0.70
    best_score = -1.0
    best_tb = 1e9

    for a in alpha_grid:
        pid_pred = _aggregate_pred_per_prediction_id(pred_img_df, alpha=float(a))
        m = pid_pred.merge(
            y_valid[["prediction_id", "cancer"]],
            on="prediction_id",
            how="inner",
            suffixes=("", "_y"),
        )
        score = pfbeta_np(
            m["cancer_y"].values.astype(np.int64),
            m["cancer"].values.astype(np.float64),
            beta=1.0,
        )
        tb = abs(a - 0.70)
        if (score > best_score + 1e-12) or (
            abs(score - best_score) <= 1e-12 and tb < best_tb
        ):
            best_score = score
            best_alpha = float(a)
            best_tb = tb

    print(
        f"Alpha selected by train holdout prevalence proxy: alpha={best_alpha:.3f} (proxy_pf1={best_score:.6f}, base_p={base_p:.6f})"
    )
    return best_alpha


def run_submit():
    checkpoint = "/kaggle/input/rsna-breast-mammography-weight-00/00003426.model.pth"
    checkpoint = _find_checkpoint(checkpoint)

    net = Net()

    if checkpoint is None:
        print(
            "WARNING: No checkpoint found under /kaggle/input. Writing constant-probability predictions."
        )
        sample = pd.read_csv(
            "/kaggle/input/rsna-breast-cancer-detection/sample_submission.csv"
        )
        default_p = _train_positive_rate_default()
        print("Using default constant probability:", default_p)
        sample["cancer"] = default_p
        sample.to_csv("submission.csv", index=False)
        print("submission.csv written:", sample.shape)
        print(sample.head())
        return

    print("Using checkpoint:", checkpoint)
    f = torch.load(checkpoint, map_location="cpu")
    state_dict = _extract_state_dict(f)

    state_dict = _strip_prefix_from_state_dict(state_dict, "module.")
    state_dict = _strip_prefix_from_state_dict(state_dict, "net.")

    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if len(missing) or len(unexpected):
        print("WARNING: load_state_dict strict=False")
        print("  missing keys   :", len(missing))
        print("  unexpected keys:", len(unexpected))

    net = net.to(device)
    net = net.eval()

    test_dataset = RsnaDataset(test_df)
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=16,
        drop_last=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        collate_fn=null_collate,
    )

    result = {"i": [], "probability": []}
    test_num = 0

    autocast_ctx = torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")
    )

    start_timer = timer()
    for t, batch in enumerate(test_loader):
        batch_size = len(batch["index"])
        batch["image"] = batch["image"].to(device, non_blocking=(device.type == "cuda"))

        with torch.no_grad():
            with autocast_ctx:
                output = net(batch)

        result["probability"].append(output["cancer"].detach().cpu().numpy())
        result["i"].append(pd.concat(batch["df"])["i"].values)

        test_num += batch_size
        print(
            "\r %8d / %d  %s"
            % (test_num, len(test_dataset), time_to_str(timer() - start_timer, "sec")),
            end="",
            flush=True,
        )
    print("")

    probability = np.concatenate(result["probability"])
    i = np.concatenate(result["i"])

    argsort = np.argsort(i)
    i = i[argsort]
    probability = probability[argsort]

    pred_img_df = pd.DataFrame(
        {"prediction_id": test_df.prediction_id.values, "cancer": probability}
    )

    alpha = _choose_alpha_by_train_holdout_pf1()

    pred_pid_df = _aggregate_pred_per_prediction_id(pred_img_df, alpha=alpha)
    pred_pid_df, _ = _calibrate_with_train_metadata(pred_pid_df)

    sample = pd.read_csv(
        "/kaggle/input/rsna-breast-cancer-detection/sample_submission.csv"
    )
    submit_df = sample.merge(
        pred_pid_df, on="prediction_id", how="left", suffixes=("", "_pred")
    )
    if "cancer_pred" in submit_df.columns:
        submit_df["cancer"] = submit_df["cancer_pred"]
        submit_df = submit_df.drop(columns=["cancer_pred"])

    submit_df["cancer"] = submit_df["cancer"].fillna(0.0).astype(float)
    submit_df["cancer"] = np.clip(submit_df["cancer"].values, 0.0, 1.0)

    submit_df.to_csv("submission.csv", index=False)
    print("submission.csv written:", submit_df.shape)
    print(submit_df.head())

    if is_local:

        def pfbeta(labels, predictions, beta=1):
            y_true_count = 0
            ctp = 0
            cfp = 0
            for idx in range(len(labels)):
                prediction = min(max(predictions[idx], 0), 1)
                if labels[idx]:
                    y_true_count += 1
                    ctp += prediction
                else:
                    cfp += prediction
            beta_squared = beta * beta
            c_precision = ctp / (ctp + cfp + 1e-12)
            c_recall = ctp / (y_true_count + 1e-12)
            if c_precision > 0 and c_recall > 0:
                return (
                    (1 + beta_squared)
                    * (c_precision * c_recall)
                    / (beta_squared * c_precision + c_recall + 1e-12)
                )
            return 0

        truth_df = (
            test_df[["prediction_id", "cancer"]]
            .groupby("prediction_id")
            .mean(numeric_only=True)
            .sort_index()
        )
        truth = truth_df.cancer.values
        predict = submit_df.set_index("prediction_id").loc[truth_df.index].cancer.values
        lb_score = pfbeta(truth, predict)
        print("lb_score", lb_score)


run_submit()
