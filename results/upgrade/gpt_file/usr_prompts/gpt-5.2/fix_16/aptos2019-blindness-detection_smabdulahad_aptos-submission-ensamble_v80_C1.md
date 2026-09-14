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

3.12

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
pillow==11.3.0
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
tqdm==4.67.1

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

0.8907307753605649

# 6. Current score

0.60208

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00504) has done: 'I fix the immediate runtime blockers by (1) making the model checkpoint loading robust to missing Kaggle input files and (2) preventing the ensemble inference from crashing when no external models are available. To preserve the core ensemble logic, if checkpoints are present they be loaded and used exactly as before; otherwise the script fall back to a single standard timm model (untrained) just to produce a valid submission CSV end-to-end. I also fix the mismatch between `model_paths` keys and the `weights` dictionary (which previously caused incorrect/failed weighting) by computing weights only for the actually loaded models. Finally, I ensure `final_predictions` is always defined and aligned to `test.csv` ordering, then write `submission.csv`.'
- What this solution (achieved 0.35249) has done: 'Your current score is near-random because the script usually falls back to an untrained model when the external checkpoints aren’t available in your environment. The smallest change that legitimately improves score toward your target is to switch that fallback to a pretrained ImageNet backbone (same timm model family and same argmax-over-5-classes semantics), so predictions become meaningfully correlated with DR severity even without extra files. I also make the dataset path selection robust to both `/kaggle/input/...` and your listed `/kaggle/data/...` locations so it reliably finds `test.csv` and images. No changes are made to the ensemble logic when checkpoints are present; they still be loaded and weighted exactly as before.'
- What this solution (achieved 0.35249) has done: 'Your score gap is large (0.35249 vs target 0.89073), and the main limiter is that the fallback path uses an ImageNet-pretrained model with a randomly initialized 5-class head, which is not aligned to DR classes and yields near-random argmax labels. To improve toward the target without changing the overall ensemble/inference semantics, I keep the same timm backbone and argmax-over-5-classes logic but switch the fallback to a DR-specific pretrained checkpoint that is already available locally (the `pretrained-models-pytorch` package in your Kaggle data). I also make the checkpoint search robust by checking multiple known local paths, and ensure we always produce a valid `submission.csv` with correct ordering and columns. If your original external ensemble checkpoints are present, the code still uses them exactly as before.'
- What this solution (achieved 0.40811) has done: 'Your current score (0.35249) is far below the target (0.89073), and the biggest issue is that the inference post-processing uses `argmax` on softmax probabilities, which is poorly aligned with quadratic weighted kappa for ordinal labels. Keeping the same models, same forward pass, same softmax+weighted ensembling core logic, I change only the final mapping from probabilities to {0,1,2,3,4} by using an expected-value regression (`sum(p*c)`) followed by simple rounding and clipping—this is a standard minimal calibration for ordinal targets that typically raises QWK substantially. I also add an optional (safe) threshold-optimization step using the training set and the current model outputs (no extra training, no architecture changes) to tune the rounding cutpoints for QWK; if train images aren’t found, it falls back to plain rounding. The script still run end-to-end and always write a valid `submission.csv` with correct ordering and columns.'
- What this solution (achieved 0.45188) has done: 'I make one targeted change to better align your inference-time “continuous severity” with the ordinal QWK metric: apply a monotonic calibration (simple 1D least-squares fit) from model expected-value outputs to label space using train predictions, then optimize thresholds on this calibrated scale. This keeps your core logic intact (same models, same softmax, same expected value, same thresholding idea) but fixes a common issue where raw expected values are badly scaled/shifted, which can cap QWK around the level you’re seeing. I also guard the calibration so it only runs when train inference succeeds; otherwise it falls back to your current behavior and still produces `submission.csv`. No architecture/training changes are introduced, and the submission format/path stays the same.'
- What this solution (achieved 0.44069) has done: 'Your current gap to the target is large (0.45188 vs 0.89073), and the most likely limiter is that you’re optimizing calibration/thresholds on the *training set in-sample*, which can overfit and not transfer to the test distribution under QWK. I keep the exact same models, softmax ensembling, expected-value continuous score, linear calibration, and threshold-search logic, but change the threshold tuning to use a deterministic out-of-fold (OOF) procedure: get train predictions once, fit calibration + thresholds on one fold and evaluate on the held-out fold, then average the learned thresholds across folds. This is a minimal semantic change (still “calibrate + threshold” without any extra training) that typically improves generalization for QWK versus in-sample tuning, moving your score upward toward the target. I also enable deterministic CUDA behavior to reduce run-to-run drift while keeping inference/training logic unchanged, and still write the same `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.45396) has done: 'Your score gap to the target is still very large (0.44069 vs 0.89073), so we should improve generalization in the only place we’re currently “learning” at inference time: the calibration/threshold fitting. I keep your exact ensemble/softmax/expected-value logic intact, but change the OOF procedure to (1) fit the linear calibration on each fold’s training split, (2) optimize thresholds on that fold’s *validation* split (not the training split), and (3) pick the thresholds that maximize mean OOF QWK—this avoids overfitting thresholds to in-fold data. I also compute a single final calibration (a,b) on the full training set after choosing thresholds (still no training, just a deterministic 1D fit), then apply it to test as before. These are minimal changes limited to the calibration/threshold block and should move QWK upward toward your target without changing the model architecture or inference semantics.'
- What this solution (achieved 0.44961) has done: 'Your current score (0.45396) is far below the target (0.89073), so we should improve generalization in the only “learned at inference time” part: calibration/threshold selection. Keeping your exact ensemble → softmax → expected-value continuous prediction logic intact, I change the OOF selection from “pick best single fold’s thresholds” to a more stable “average thresholds across folds”, which typically transfers better to test for QWK with minimal semantic change. I also make the threshold optimizer enforce sensible ordering bounds between thresholds (to avoid pathological/too-close cutpoints) without changing the underlying search approach. Everything else (models, transforms, loaders, submission format/path) remains the same.'
- What this solution (achieved 0.44849) has done: 'I make a minimal change to the threshold-learning block to reduce overfitting and better align the learned cutpoints to the test distribution under QWK. Specifically, I replace the current “weighted average of per-fold thresholds” with a more stable approach: generate full out-of-fold (OOF) calibrated continuous predictions for every train sample (each sample calibrated by a model fit without that sample), then optimize a single set of thresholds on these OOF predictions. This keeps your ensemble/softmax/expected-value logic identical and only changes how thresholds are selected, which should legitimately improve generalization and move your score upward toward the 0.89 target. Everything else (paths, transforms, model loading, submission writing) remains unchanged.'
- What this solution (achieved 0.31827) has done: 'Your current score (0.44849) is far below the target (0.89073), so we should make the smallest changes that improve generalization without altering your ensemble/model logic. The biggest, safe gain under QWK usually comes from (1) using a DR-appropriate preprocessing (center-crop around the fundus + remove black borders) while keeping the same 224×224 + ImageNet normalization, and (2) making the threshold optimization a bit less coarse (finer grid) without changing the optimization approach. I add a deterministic, fast OpenCV “circle-crop” transform inside the dataset (same images/labels, just better input conditioning) and slightly increase the threshold search resolution while keeping the same coordinate-descent structure and n_iter. Everything else (model list, weighting, softmax ensembling, expected value, calibration, submission formatting) stays the same and it still write `submission.csv`.'
- What this solution (achieved 0.40307) has done: 'Your score is far below the target, so we should make the smallest changes that improve the quality/consistency of the inputs and the calibration step without altering your ensemble architecture or inference semantics. The biggest issue in your last change is that the “circle crop” is not a true fundus crop (it mostly masks corners but keeps black borders), which can hurt pretrained DR checkpoints; I replace it with a standard fast Ben Graham-style crop (estimate radius, crop to fundus ROI, then optional mild blur/contrast) while still outputting the same 224×224 normalized tensor. Next, I make threshold optimization slightly more stable by using more coordinate-descent iterations (same optimizer, same semantics) and by optimizing thresholds on OOF *labels aligned with OOF calibrated predictions* (already intended) while keeping deterministic behavior. Everything else (model list, softmax-weighted ensembling, expected-value continuous prediction, linear calibration, thresholding, submission format/path) remains the same and it still write `submission.csv`.'
- What this solution (achieved 0.36162) has done: 'I keep your ensemble/inference and “expected value → linear calibration → thresholds → class” semantics unchanged, and focus only on two small score-relevant fixes. First, the Ben Graham preprocessing is currently over-aggressive (can distort lesions) and uses a fragile radius estimate; I replace it with the standard, fast, widely-used APTOS crop+resize based on image intensity and radius estimation, and make the enhancement optional/mild so it doesn’t destroy signal. Second, your threshold search grid is too narrow/coarse for calibrated continuous outputs; I keep the same coordinate-descent optimizer but use an adaptive grid centered on the current thresholds (per coordinate) with a finer step, which improves the chance of finding better cutpoints without changing the optimization approach.'
- What this solution (achieved 0.55897) has done: 'Your score (0.36162) is far below the target (0.89073), so we should make a small change that legitimately improves generalization without touching the ensemble/model forward-pass logic. The biggest low-risk gain is to fix a mismatch between your Ben Graham preprocessing and the pretrained DR checkpoints you may load: those checkpoints are typically trained on a *simple crop-to-fundus + resize* pipeline, and your current sharpening can distort lesion texture and hurt QWK. I keep the same dataset class and transforms, but replace the enhancement with the standard APTOS “trim black border by mask, then center-crop-square” preprocessing (no sharpening), and I make threshold optimization a bit more robust by using a slightly larger coordinate-descent iteration count while keeping the same optimizer semantics. Everything else (model loading, softmax weighting, expected value, linear calibration, OOF threshold fit, submission format/path) remains unchanged.'
- What this solution (achieved 0.56019) has done: 'I keep your exact ensemble → softmax → expected-value → linear calibration → single-threshold-optimization-on-OOF → apply-to-test pipeline intact, and make only two score-relevant stability improvements aimed at better generalization under QWK. First, I ensure the OOF predictions used for threshold optimization are *truly OOF* by also generating them fold-by-fold for the calibration fit (already done) and by optimizing thresholds on those OOF values but using a slightly stronger, more appropriate constraint: enforce thresholds within the empirical calibrated prediction range to avoid wasted search space. Second, I make the threshold optimizer’s search bounds adaptive per-threshold (based on percentiles of OOF predictions in the neighboring class regions) while keeping the same coordinate-descent + grid search semantics; this typically improves QWK without changing model logic. Everything else (model loading, preprocessing, dataloaders, submission format/path) stays the same and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.60208) has done: 'Your current score (0.56019) is far below the target (0.89073), so we should cautiously increase performance with minimal, metric-aligned changes while keeping your ensemble/softmax/expected-value → linear calibration → thresholding pipeline intact. The biggest low-risk issue is that your OOF continuous predictions are calibrated fold-by-fold but you then fit the final linear calibration on the full train set using *raw* (uncalibrated) continuous outputs, which can misalign the calibrated scale used to pick thresholds vs the calibrated scale applied at test time. I keep the same linear calibration model and threshold optimizer, but refit the final calibration using an OOF-style “average of fold calibrators” so the calibration applied to test matches the calibration used to create OOF predictions, improving generalization under QWK. I also make one tiny, score-relevant adjustment to threshold initialization by deriving it from OOF prediction quantiles (still the same thresholding semantics), which typically helps the coordinate-descent search converge to better cutpoints without changing the core logic.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
import cv2

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
try:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass




## === cell 1
class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

    @staticmethod
    def _fundus_crop_ben_graham(pil_img):
        """
        Score-relevant preprocessing kept as in your current best run:
        - detect non-black fundus area via grayscale mask
        - trim black borders
        - center-crop to square
        """
        img = np.asarray(pil_img)  # RGB uint8
        if img.ndim != 3 or img.shape[2] != 3:
            return pil_img

        h, w = img.shape[:2]
        if h < 64 or w < 64:
            return pil_img

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        thr = max(5, int(np.percentile(gray, 5)))
        mask = gray > thr

        if mask.sum() < (h * w * 0.05):
            return pil_img

        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1

        if (y1 - y0) < 32 or (x1 - x0) < 32:
            return pil_img

        cropped = img[y0:y1, x0:x1]
        ch, cw = cropped.shape[:2]

        side = min(ch, cw)
        cy, cx = ch // 2, cw // 2
        y0s = max(cy - side // 2, 0)
        x0s = max(cx - side // 2, 0)
        square = cropped[y0s : y0s + side, x0s : x0s + side]

        if square.size == 0 or square.shape[0] < 32 or square.shape[1] < 32:
            return pil_img

        return Image.fromarray(square)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".png")
        image = Image.open(img_name).convert("RGB")

        image = self._fundus_crop_ben_graham(image)

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 2
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)




## === cell 3
def _pick_first_existing(paths):
    for p in paths:
        if p is not None and os.path.exists(p):
            return p
    return None


test_csv_file = _pick_first_existing(
    [
        "/kaggle/input/aptos2019-blindness-detection/test.csv",
        "/kaggle/data/aptos2019-blindness-detection/test.csv",
        "/kaggle/input/test.csv",
        "/kaggle/data/test.csv",
    ]
)
test_root_dir = _pick_first_existing(
    [
        "/kaggle/input/aptos2019-blindness-detection/test_images",
        "/kaggle/data/aptos2019-blindness-detection/test_images",
        "/kaggle/input/test_images",
        "/kaggle/data/test_images",
    ]
)

if test_csv_file is None or test_root_dir is None:
    raise FileNotFoundError(
        f"Could not locate test.csv or test_images. test_csv_file={test_csv_file}, test_root_dir={test_root_dir}"
    )

test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 4
model_paths = {
    "resnet18": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/resnet18(WD_1e-3)_aptos.pth",
    "efficientnet_b1": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b1.pth",
    "efficientnet_b2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b2.pth",
    "seresnext101_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/seresnext101_32x4d.pth",
}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b1": "efficientnet_b1",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnet_b3": "efficientnet_b3",
    "efficientnet_b4": "efficientnet_b4",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}



## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_list = []
loaded_model_keys = []

for model_key, path in model_paths.items():
    if not os.path.exists(path):
        print(f"[WARN] Checkpoint not found for {model_key}: {path} (skipping)")
        continue

    model_name = model_names[model_key]
    model = timm.create_model(model_name, pretrained=False, num_classes=5)

    state = torch.load(path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            new_state[nk] = v
        state = new_state

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    models_list.append(model)
    loaded_model_keys.append(model_key)

if len(models_list) == 0:
    fallback_key = "efficientnet_b0"
    fallback_arch = model_names[fallback_key]

    dr_ckpt = _pick_first_existing(
        [
            "/kaggle/input/pretrained-models-pytorch/aptos2019/effnetb0/effnetb0_dr_0_9728.pth",
            "/kaggle/data/pretrained-models-pytorch/aptos2019/effnetb0/effnetb0_dr_0_9728.pth",
            "/kaggle/input/pretrained-models-pytorch/aptos2019/effnetb0/effnetb0_dr_0_9732.pth",
            "/kaggle/data/pretrained-models-pytorch/aptos2019/effnetb0/effnetb0_dr_0_9732.pth",
            "/kaggle/input/pretrained-models-pytorch/aptos2019/effnetb0/effnetb0_dr_0_9779.pth",
            "/kaggle/data/pretrained-models-pytorch/aptos2019/effnetb0/effnetb0_dr_0_9779.pth",
        ]
    )

    if dr_ckpt is not None:
        print(
            f"[WARN] No external ensemble checkpoints loaded. Falling back to DR checkpoint: {dr_ckpt}"
        )
        model = timm.create_model(fallback_arch, pretrained=False, num_classes=5)
        state = torch.load(dr_ckpt, map_location="cpu")
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        if isinstance(state, dict):
            new_state = {}
            for k, v in state.items():
                nk = k
                if nk.startswith("module."):
                    nk = nk[len("module.") :]
                if nk.startswith("model."):
                    nk = nk[len("model.") :]
                new_state[nk] = v
            state = new_state
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        models_list = [model]
        loaded_model_keys = [fallback_key]
    else:
        print(
            f"[WARN] No external checkpoints and no DR checkpoint found. Falling back to '{fallback_key}' with pretrained=True (ImageNet backbone) to still produce a valid submission."
        )
        model = timm.create_model(fallback_arch, pretrained=True, num_classes=5)
        model.to(device)
        model.eval()
        models_list = [model]
        loaded_model_keys = [fallback_key]



## === cell 6
validation_scores = {
    "resnet18": 0.887,
    "efficientnet_b0": 0.8922,
    "efficientnet_b1": 0.894,
    "efficientnet_b2": 0.898,
    "efficientnet_b3": 0.9127,
    "efficientnet_b4": 0.893,
    "efficientnet_b5": 0.870,
    "inception_resnet_v2": 0.896,
    "inception_v4": 0.8875,
    "seresnext50_32x4d": 0.8652,
    "seresnext101_32x4d": 0.9083,
}

used_scores = {k: validation_scores.get(k, 1.0) for k in loaded_model_keys}
total_score = float(sum(used_scores.values()))
weights = {k: (v / total_score) for k, v in used_scores.items()}

print("Using models:", loaded_model_keys)
print("Weights:", weights)



## === cell 7
all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="Infer"):
        images = images.to(device, non_blocking=True)

        per_model = []
        for model_key, model in zip(loaded_model_keys, models_list):
            probs = nn.functional.softmax(model(images), dim=1)
            per_model.append(weights[model_key] * probs)

        weighted_outputs = torch.stack(per_model, dim=0).sum(dim=0)
        all_outputs.append(weighted_outputs.detach().cpu().numpy())

all_outputs = np.concatenate(all_outputs, axis=0)

classes = np.arange(5, dtype=np.float32)
test_pred_cont = (all_outputs * classes[None, :]).sum(axis=1)

test_ids = pd.read_csv(test_csv_file)["id_code"].values
if len(test_pred_cont) != len(test_ids):
    raise RuntimeError(
        f"Prediction length mismatch: preds={len(test_pred_cont)} vs test={len(test_ids)}"
    )




## === cell 8
def _quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0
    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - num / den if den > 0 else 0.0


def _apply_thresholds(x, thr):
    x = np.asarray(x, dtype=np.float32)
    thr = np.asarray(thr, dtype=np.float32)
    y = np.zeros_like(x, dtype=np.int64)
    y[x >= thr[0]] = 1
    y[x >= thr[1]] = 2
    y[x >= thr[2]] = 3
    y[x >= thr[3]] = 4
    return y


def _enforce_thr_constraints(thr, lo=0.0, hi=4.0, min_gap=0.05):
    thr = np.sort(np.asarray(thr, dtype=np.float32))
    thr = np.clip(thr, lo, hi)
    for i in range(1, 4):
        if thr[i] < thr[i - 1] + min_gap:
            thr[i] = thr[i - 1] + min_gap
    thr = np.clip(thr, lo, hi)
    for i in range(2, -1, -1):
        if thr[i] > thr[i + 1] - min_gap:
            thr[i] = thr[i + 1] - min_gap
    thr = np.clip(thr, lo, hi)
    return np.sort(thr)


def _fit_linear_calibration(x, y, clip=(0.0, 4.0)):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    xm = float(x.mean())
    ym = float(y.mean())
    xv = float(((x - xm) ** 2).mean())
    if not np.isfinite(xv) or xv < 1e-12:
        a = 1.0
        b = 0.0
    else:
        cov = float(((x - xm) * (y - ym)).mean())
        a = cov / xv
        if not np.isfinite(a):
            a = 1.0
        if a <= 0:
            a = 1.0
        b = ym - a * xm
        if not np.isfinite(b):
            b = 0.0

    def transform_fn(z):
        zz = a * np.asarray(z, dtype=np.float64) + b
        if clip is not None:
            zz = np.clip(zz, float(clip[0]), float(clip[1]))
        return zz.astype(np.float32)

    return (float(a), float(b)), transform_fn


def _optimize_thresholds(y_true, x_pred, init_thr=None, n_iter=4):
    """
    Keep same coordinate-descent + grid search semantics, but use adaptive bounds
    based on OOF prediction range to focus search where it matters for QWK.
    """
    y_true = np.asarray(y_true, dtype=int)
    x_pred = np.asarray(x_pred, dtype=np.float32)
    if init_thr is None:
        init_thr = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    thr = init_thr.astype(np.float32).copy()

    lo = float(np.percentile(x_pred, 0.5))
    hi = float(np.percentile(x_pred, 99.5))
    lo = min(lo, 0.0)
    hi = max(hi, 4.0)

    thr = _enforce_thr_constraints(thr, lo=lo, hi=hi, min_gap=0.05)
    best = _quadratic_weighted_kappa(y_true, _apply_thresholds(x_pred, thr))

    base_width = max(0.35, 0.12 * (hi - lo))

    for _ in range(n_iter):
        for t in range(4):
            best_local = best
            best_thr = thr.copy()

            left_cap = lo if t == 0 else float(thr[t - 1] + 0.05)
            right_cap = hi if t == 3 else float(thr[t + 1] - 0.05)

            center = float(thr[t])
            left = max(left_cap, center - base_width)
            right = min(right_cap, center + base_width)

            if not (right > left):
                continue

            grid = np.linspace(left, right, 161, dtype=np.float32)

            for cand in grid:
                thr_c = thr.copy()
                thr_c[t] = cand
                thr_c = _enforce_thr_constraints(thr_c, lo=lo, hi=hi, min_gap=0.05)
                k = _quadratic_weighted_kappa(y_true, _apply_thresholds(x_pred, thr_c))
                if k > best_local:
                    best_local = k
                    best_thr = thr_c

            thr = best_thr
            best = best_local

    return thr, best


def _init_thresholds_from_quantiles(x, clip_lo=0.0, clip_hi=4.0):
    """
    Score-relevant minimal change:
    initialize thresholds from OOF calibrated prediction quantiles (still 4 cutpoints),
    which usually helps QWK threshold search find better local optima.
    """
    x = np.asarray(x, dtype=np.float32)
    qs = np.quantile(x, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    qs = np.clip(qs, clip_lo, clip_hi)
    return _enforce_thr_constraints(qs, lo=clip_lo, hi=clip_hi, min_gap=0.05).astype(
        np.float32
    )


train_csv_file = _pick_first_existing(
    [
        "/kaggle/input/aptos2019-blindness-detection/train.csv",
        "/kaggle/data/aptos2019-blindness-detection/train.csv",
        "/kaggle/input/train.csv",
        "/kaggle/data/train.csv",
    ]
)
train_root_dir = _pick_first_existing(
    [
        "/kaggle/input/aptos2019-blindness-detection/train_images",
        "/kaggle/data/aptos2019-blindness-detection/train_images",
        "/kaggle/input/train_images",
        "/kaggle/data/train_images",
    ]
)

thresholds = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
calib_params = (1.0, 0.0)
calib_fn = lambda z: np.asarray(z, dtype=np.float32)

if train_csv_file is not None and train_root_dir is not None:
    try:
        train_dataset = BlindnessDataset(
            train_csv_file, train_root_dir, transform=transform, test=False
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

        train_outputs = []
        train_labels = []

        with torch.no_grad():
            for images, labels in tqdm(train_loader, desc="Infer(train for thr)"):
                images = images.to(device, non_blocking=True)
                per_model = []
                for model_key, model in zip(loaded_model_keys, models_list):
                    probs = nn.functional.softmax(model(images), dim=1)
                    per_model.append(weights[model_key] * probs)
                weighted_outputs = torch.stack(per_model, dim=0).sum(dim=0)
                train_outputs.append(weighted_outputs.detach().cpu().numpy())
                train_labels.append(labels.numpy())

        train_outputs = np.concatenate(train_outputs, axis=0)
        train_labels = np.concatenate(train_labels, axis=0).astype(int)
        train_pred_cont = (train_outputs * classes[None, :]).sum(axis=1)

        n = len(train_labels)
        idx = np.arange(n)
        rng = np.random.RandomState(42)
        rng.shuffle(idx)

        K = 5
        folds = np.array_split(idx, K)

        oof_cont_cal = np.empty(n, dtype=np.float32)
        fold_calib_params = []

        for k in range(K):
            val_idx = folds[k]
            tr_idx = np.concatenate([folds[j] for j in range(K) if j != k])

            x_tr = train_pred_cont[tr_idx]
            y_tr = train_labels[tr_idx]
            x_val = train_pred_cont[val_idx]

            (a_k, b_k), fn_k = _fit_linear_calibration(x_tr, y_tr, clip=(0.0, 4.0))
            fold_calib_params.append((a_k, b_k))
            oof_cont_cal[val_idx] = fn_k(x_val)

        base_thr = _init_thresholds_from_quantiles(
            oof_cont_cal, clip_lo=0.0, clip_hi=4.0
        )
        thresholds, oof_best = _optimize_thresholds(
            train_labels, oof_cont_cal, init_thr=base_thr, n_iter=4
        )

        lo_oof = float(np.percentile(oof_cont_cal, 0.5))
        hi_oof = float(np.percentile(oof_cont_cal, 99.5))
        lo_oof = min(lo_oof, 0.0)
        hi_oof = max(hi_oof, 4.0)
        thresholds = _enforce_thr_constraints(
            thresholds, lo=lo_oof, hi=hi_oof, min_gap=0.05
        ).astype(np.float32)

        a_avg = float(np.mean([ab[0] for ab in fold_calib_params]))
        b_avg = float(np.mean([ab[1] for ab in fold_calib_params]))
        calib_params = (a_avg, b_avg)

        def calib_fn_full(z):
            zz = a_avg * np.asarray(z, dtype=np.float64) + b_avg
            zz = np.clip(zz, 0.0, 4.0)
            return zz.astype(np.float32)

        calib_fn = calib_fn_full

        y_oof_hat = _apply_thresholds(oof_cont_cal, thresholds)
        oof_qwk = _quadratic_weighted_kappa(train_labels, y_oof_hat)

        print("[INFO] OOF QWK (single thr on OOF):", float(oof_qwk))
        print("[INFO] Linear calibration avg-fold (a,b):", calib_params)
        print("[INFO] Selected thresholds (opt on OOF):", thresholds)
    except Exception as e:
        print(
            "[WARN] Calibration/threshold optimization failed; using default rounding. Error:",
            repr(e),
        )
else:
    print("[WARN] train.csv/train_images not found; using default rounding thresholds.")

test_pred_cont_cal = calib_fn(test_pred_cont)
final_predictions = _apply_thresholds(test_pred_cont_cal, thresholds).astype(int)



## === cell 9
submission_df = pd.DataFrame({"id_code": test_ids, "diagnosis": final_predictions})

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission_df.head())
