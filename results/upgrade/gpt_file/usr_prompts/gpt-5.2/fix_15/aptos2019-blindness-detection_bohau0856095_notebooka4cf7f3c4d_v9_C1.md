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

3.9

# 3. Installed packages

geopandas==0.14.4
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

0.8923503909317514

# 6. Current score

-0.00475

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.2545) has done: 'I fix the immediate runtime blockers so the notebook runs end-to-end: the missing `Parameter` import, the misspelled `tranforms` variable, the undefined `classes_num`, and the bad weight path (it currently contains an extra space before `.pth`). To keep core logic intact, I still load the same saved model if it exists; if it doesn’t (common in this environment), I fall back to constructing the same architecture and running it with pretrained backbone weights so a valid submission is always produced. I also make device selection robust (CPU fallback) and ensure the test IDs are read as strings and the submission is written with the exact required columns and `.csv` suffix.'
- What this solution (achieved 0.10673) has done: 'Your score is far below the target, so we should improve predictions without changing the model architecture or training (you’re doing pure inference). The main issue is that you’re ignoring the model’s regression head and directly argmax’ing the classifier logits, which often performs poorly for quadratic weighted kappa on this task. I keep the exact same network and weights-loading logic, but change only the prediction post-processing to use the regressor output (rounded and clipped to 0–4), with a simple average ensemble between classifier-expected-class and regressed value to stabilize results. I also fix the `tranforms` typo safely by using a correctly named `transforms_fn` variable (no functional change).'
- What this solution (achieved -0.06528) has done: 'Your current score is far below the target, so we should improve predictions while keeping the same model and pure-inference workflow. The largest likely issue is a mismatch between the training-time pooling/feature dimension and your inference-time head definition: `tf_efficientnet_b4_ns`’s classifier feature dim is not 1000, so loading weights with `strict=False` can silently leave your heads randomly initialized, crushing performance. I keep the exact architecture pieces (EfficientNet-B4 + GeM + linear classifier/regressor) but set the head input dim from the backbone (`num_features`) and replace the backbone’s classifier with identity so GeM actually feeds the heads. Finally, I keep your blended prediction idea but slightly bias toward the regressor (commonly better aligned with QWK) as a minimal, low-risk calibration tweak.'
- What this solution (achieved -0.00664) has done: 'We need to move the score upward toward the target, and your current score suggests the model weights likely aren’t being applied correctly at inference (heads left randomly initialized or backbone feature shape mismatch). I keep the same EfficientNet-B4 + GeM + (classifier, regressor) core, but make weight loading deterministic and strict about matching keys/shapes by constructing `Model()` first and loading a cleaned state_dict into it (instead of `torch.load()` potentially returning a different object). I also ensure we use `forward_features()` so GeM is applied to the final conv features (not to an already-pooled vector), which fixes a common mismatch with EfficientNet in timm while preserving the same architecture intent. Finally, I keep your blended post-processing but make it conditional: if regressor outputs look out-of-range (a sign of bad weights), fall back to classifier-expected-class to avoid catastrophic negative kappa.'
- What this solution (achieved -0.31853) has done: 'Your current score is still far below the target, so the most likely cause is that the competition weights are not being loaded correctly (because `strict=False` silently leaves mismatched layers randomly initialized). I keep your exact EfficientNet-B4 + GeM + (classifier, regressor) design and pure-inference approach, but make weight loading “strict when possible” and explicitly verify that a meaningful fraction of parameters actually matched; if not, we fall back to loading the entire saved `nn.Module` object when available. I also ensure the backbone head is set to identity so `forward_features()` and GeM always feed your heads consistently, which avoids common timm EfficientNet head-shape traps without changing the intended architecture. Finally, I keep your blended post-processing but make it slightly more robust by computing everything on CPU-safe tensors and keeping deterministic settings.'
- What this solution (achieved -0.01427) has done: 'The negative kappa strongly suggests the loaded weights are not actually being used as intended, so the smallest change with the biggest expected gain is to make weight loading strict and fail fast into a known-good fallback. I keep your exact EfficientNet-B4 + GeM + (classifier, regressor) core and the same inference-only workflow, but I (1) search for weights in common Kaggle locations (including the dataset folder itself), (2) only accept a state_dict load if it matches the model shapes with a high ratio, and otherwise (3) fall back to the pretrained backbone and use the classifier-expected-class only (more stable than blending with a likely-untrained regressor). This should move the score upward toward the target by avoiding catastrophic random-head predictions while preserving your overall approach and producing the same submission format.'
- What this solution (achieved -0.01427) has done: 'Your score is far below the target, so we should increase it with the smallest possible change that improves quadratic weighted kappa without altering the model or training. The main improvement is to keep your same logits/regressor computation, but replace the fixed rounding with an “optimized rounding thresholds” step fitted on the training set using the model’s own out-of-fold-like predictions (via a simple train/val split) and directly maximizing QWK. This preserves your inference-only workflow and architecture, and only changes the final mapping from continuous predictions to classes—often a large win for QWK on APTOS. I also keep your existing “weights_ok” safety checks and only apply threshold optimization when weights look usable; otherwise it falls back to your current stable behavior. The script still runs end-to-end and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.04019) has done: 'Your current QWK is far below the target, so we should improve it with the smallest change that most directly affects QWK while keeping your model/inference core intact. The biggest issue is that threshold fitting is currently both (a) very noisy (only 1 random split) and (b) extremely slow (hundreds of single-image GPU/CPU calls), which can also cause timeouts/incomplete runs; we make it stable and faster by doing batched inference on a small fixed validation subset. We also add a tiny “fail-safe” so threshold optimization is skipped unless weights truly look usable (to avoid optimizing on random/untrained outputs, which can worsen QWK). Finally, we keep your exact model, blending logic, and submission format; only inference batching and threshold-fitting procedure are tightened to move score upward toward the target.'
- What this solution (achieved 0.04019) has done: 'I keep your EfficientNet+GeM inference core unchanged and focus on the post-processing that directly drives QWK. The biggest controllable lever is threshold fitting: instead of fitting on just one small random split (high variance), we fit thresholds on a deterministic stratified validation split and use a slightly larger fixed validation size for stability. I also make the threshold search a bit more thorough but still fast (small extra compute only on the val subset), and add a monotonic safety clamp to keep thresholds ordered and sane. These minimal changes should move your score upward toward the target by producing more reliable class binning for QWK.'
- What this solution (achieved 0.04019) has done: 'Your score is far below the target, so we should push it upward with the smallest change that most directly improves QWK without touching your model architecture or training loop. The biggest likely issue is that your “stratified” validation construction is not actually randomized across classes (it concatenates per-class blocks), so threshold fitting is biased and can generalize poorly to test. I change the validation sampling to a deterministic stratified *sample* (fixed per-class counts, then shuffled) and add a tiny safety fallback: if the fitted thresholds don’t beat the default thresholds on the same validation predictions, keep the defaults. This keeps your inference core, blending logic, and threshold-search method intact, but makes threshold optimization materially more reliable (and thus should increase QWK toward the target).'
- What this solution (achieved 0.04019) has done: 'Your current score (0.04019) is far below the target (0.89235), so we should increase QWK with the smallest change that directly affects the metric while keeping your EfficientNet+GeM inference core intact. The most effective low-risk lever here is threshold fitting, but it currently uses a single split and can overfit/noise; I make it more stable by fitting thresholds via deterministic K-fold out-of-fold predictions on a capped subset (so it stays within time) and only applying the fitted thresholds if they improve mean OOF QWK over defaults. This keeps your same model, weights-loading, continuous blended prediction, and grid-search method; we only change how the validation data for threshold tuning is constructed (OOF instead of one split). The script still runs end-to-end and writes a valid `submission.csv` with required columns.'
- What this solution (achieved 0.04019) has done: 'We should move your QWK upward toward the target by fixing a likely major inference mismatch: your EfficientNet backbone still has `drop_rate` active during `forward_features()` unless set to 0, which makes predictions noisy and can collapse QWK even in `eval()`; setting it to 0 is a minimal, architecture-preserving inference-only change. Next, your “OOF” threshold fitting currently leaks because you never re-fit per fold (you use the same net for all folds), so it’s not actually OOF and can produce thresholds that generalize poorly; we keep the same idea but instead fit thresholds on a single deterministic stratified validation split (no extra training, much faster, less noisy). Finally, we add a very small safety check that skips threshold fitting if the model’s continuous predictions on the val set have near-zero variance (a common sign weights didn’t load properly), preventing harmful thresholds. These changes keep your model, weights loading, blending logic, and submission format intact, and stay within time.'
- What this solution (achieved 0.04019) has done: 'Your score is far below the target, so we should increase it with the smallest change that most directly affects QWK while preserving your model and inference flow. The biggest likely issue is that `cls_exp` is being computed from a softmax over the classifier head even when the head weights may not match (or may be poorly calibrated), which can destabilize the continuous predictions used for thresholding. I keep the exact architecture and weight-loading logic, but add a minimal “head sanity” check to decide whether to trust the classifier head; when it looks unreliable, we use the regressor-only continuous output (clipped) for threshold fitting and test prediction. This is a post-processing / routing change only, and it should move QWK upward by preventing noisy classifier logits from dominating the continuous target used by threshold optimization.'
- What this solution (achieved -0.00475) has done: 'We need to move your QWK upward toward the target, and the current 0.04019 suggests the biggest remaining issue is not the architecture but the input pipeline mismatch with how these APTOS models are typically trained. I keep your EfficientNet-B4+GeM+(classifier, regressor) inference exactly as-is and only adjust preprocessing to include a standard center-crop after resize (to reduce black borders/background that can dominate features) while keeping resolution the same. I also make threshold fitting use a stratified *fixed-size* validation with per-class floor and an overall cap, plus a small robustness check that prefers thresholds that improve QWK on a held-out slice of the validation predictions (still no training, just post-processing). These are minimal, metric-aligned changes that should materially increase QWK without changing your model, weights, or inference semantics.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import torchvision.transforms as transforms
from PIL import Image
import timm

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda:0" if torch.cuda.is_available() else "cpu"

DATA_DIR = "../input/aptos2019-blindness-detection"
TEST_CSV = f"{DATA_DIR}/test.csv"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
TEST_IMG_DIR = f"{DATA_DIR}/test_images"
TRAIN_IMG_DIR = f"{DATA_DIR}/train_images"

test_df = pd.read_csv(TEST_CSV, dtype={"id_code": str})
test_ids = test_df["id_code"].tolist()

transforms_fn = transforms.Compose(
    [
        transforms.Resize((400, 400)),
        transforms.CenterCrop((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

classes_num = 5


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)

        if hasattr(self.backbone, "drop_rate"):
            self.backbone.drop_rate = 0.0

        feat_dim = getattr(self.backbone, "num_features", 1000)

        if hasattr(self.backbone, "classifier"):
            self.backbone.classifier = nn.Identity()
        if hasattr(self.backbone, "global_pool"):
            self.backbone.global_pool = nn.Identity()

        self.pool = GeM(flatten=True)
        self.classifier = nn.Linear(feat_dim, classes_num)
        self.regressor = nn.Linear(feat_dim, 1)

    def forward(self, x):
        feats = self.backbone.forward_features(x)  # (B, C, H, W)
        pooled = self.pool(feats)  # (B, C)
        out1 = self.classifier(pooled)
        out2 = self.regressor(pooled)
        return out1, out2


def _clean_state_dict(state):
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if not isinstance(state, dict):
        return None
    cleaned = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        cleaned[nk] = v
    return cleaned


def _try_load_state_dict_strictish(
    model: nn.Module, weight_path: str, min_match_ratio: float = 0.95
):
    """
    Rationale (score-up): only accept weights if they match tensor shapes well; otherwise heads may be random -> bad QWK.
    """
    state = torch.load(weight_path, map_location="cpu")

    if isinstance(state, nn.Module):
        return state, True

    cleaned = _clean_state_dict(state)
    if cleaned is None:
        return model, False

    model_sd = model.state_dict()
    loadable = {}
    matched = 0
    for k, v in cleaned.items():
        if k in model_sd and hasattr(v, "shape") and model_sd[k].shape == v.shape:
            loadable[k] = v
            matched += 1

    match_ratio = matched / max(1, len(model_sd))
    if match_ratio < min_match_ratio:
        return model, False

    model.load_state_dict(loadable, strict=False)
    return model, True


weight_candidates = [
    "../input/weights/tf_efficientnet_b4_ns_regress.pth",
    "../input/weights/tf_efficientnet_b4_ns_regress .pth",
    "../input/aptos2019-blindness-detection/tf_efficientnet_b4_ns_regress.pth",
    "../input/aptos2019-blindness-detection/tf_efficientnet_b4_ns_regress .pth",
    "/kaggle/input/weights/tf_efficientnet_b4_ns_regress.pth",
]

weight_path = None
for p in weight_candidates:
    if os.path.exists(p):
        weight_path = p
        break

net = Model()
weights_ok = False
if weight_path is not None:
    net, weights_ok = _try_load_state_dict_strictish(
        net, weight_path, min_match_ratio=0.95
    )

net = net.to(device)
net.eval()


def qwk(y_true, y_pred, n_classes=5):
    """Quadratic weighted kappa (numpy), to optimize thresholding toward the competition metric."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        O[a, b] += 1.0

    act_hist = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=n_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    E = E / E.sum() * O.sum()

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - num / den


def apply_thresholds(x, thr):
    x = np.asarray(x, dtype=np.float64)
    thr = np.asarray(thr, dtype=np.float64)
    return np.digitize(x, thr).astype(int)


def _sanitize_thresholds(thr, n_classes=5):
    """
    Rationale (score-up, minimal): ensure strictly increasing thresholds after search to avoid invalid binning.
    """
    thr = np.asarray(thr, dtype=np.float64).copy()
    for k in range(1, n_classes - 1):
        if thr[k] <= thr[k - 1] + 1e-3:
            thr[k] = thr[k - 1] + 1e-3
    thr = np.clip(thr, -2.0, 6.0)
    return thr


def fit_thresholds_grid(x, y, init_thr=None, n_classes=5, span=0.8, steps=40, iters=2):
    """
    Rationale (score-up, minimal): QWK is very sensitive to how continuous predictions are binned.
    We keep your same continuous prediction and tune 4 cutpoints on val to maximize QWK.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=int)

    if init_thr is None:
        thr = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    else:
        thr = np.array(init_thr, dtype=np.float64)

    thr = _sanitize_thresholds(thr, n_classes=n_classes)
    best_thr = thr.copy()
    best_score = qwk(y, apply_thresholds(x, best_thr), n_classes=n_classes)

    for _ in range(iters):
        for k in range(n_classes - 1):
            left_bound = best_thr[k - 1] + 1e-3 if k > 0 else -10.0
            right_bound = best_thr[k + 1] - 1e-3 if k < n_classes - 2 else 10.0

            center = best_thr[k]
            grid = np.linspace(center - span, center + span, steps)
            grid = grid[(grid > left_bound) & (grid < right_bound)]
            if grid.size == 0:
                continue

            local_best_t = best_thr[k]
            local_best_s = best_score
            for t in grid:
                cand = best_thr.copy()
                cand[k] = t
                cand = _sanitize_thresholds(cand, n_classes=n_classes)
                s = qwk(y, apply_thresholds(x, cand), n_classes=n_classes)
                if s > local_best_s:
                    local_best_s = s
                    local_best_t = t
            best_thr[k] = local_best_t
            best_thr = _sanitize_thresholds(best_thr, n_classes=n_classes)
            best_score = local_best_s

    return best_thr, best_score


def _estimate_head_quality(sample_n=96, bs=16):
    if not os.path.exists(TRAIN_CSV):
        return False, None

    df = pd.read_csv(TRAIN_CSV, dtype={"id_code": str})
    df = df.sample(n=min(sample_n, len(df)), random_state=0).reset_index(drop=True)
    paths = [f"{TRAIN_IMG_DIR}/{idx}.png" for idx in df["id_code"].tolist()]

    cls_exps = []
    regs = []
    with torch.inference_mode():
        for i in range(0, len(paths), bs):
            imgs = []
            for p in paths[i : i + bs]:
                img = Image.open(p).convert("RGB")
                imgs.append(transforms_fn(img))
            x = torch.stack(imgs, dim=0).to(device)

            logits, reg = net(x)
            probs = torch.softmax(logits, dim=1)
            arng = torch.arange(classes_num, device=device).float()
            cls_exp = (probs * arng).sum(dim=1)
            cls_exps.append(cls_exp.detach().cpu().numpy())
            regs.append(reg.view(-1).detach().cpu().numpy())

    cls_exps = np.concatenate(cls_exps).astype(np.float64)
    regs = np.concatenate(regs).astype(np.float64)

    cls_std = float(np.std(cls_exps))
    reg_std = float(np.std(regs))
    cls_ok = np.isfinite(cls_std) and cls_std > 0.15
    reg_ok = np.isfinite(reg_std) and reg_std > 0.05
    return bool(cls_ok and reg_ok), {"cls_std": cls_std, "reg_std": reg_std}


head_ok, head_stats = (False, None)
if weights_ok:
    head_ok, head_stats = _estimate_head_quality(
        sample_n=96, bs=16 if device.startswith("cuda") else 8
    )


def _infer_continuous_batch(image_paths):
    """
    Rationale (score-up): batching makes threshold-fitting feasible and deterministic within time limits.
    Rationale (score-up, minimal): if classifier head looks unreliable, avoid mixing its softmax into the continuous target.
    """
    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        imgs.append(transforms_fn(img))
    x = torch.stack(imgs, dim=0).to(device)

    logits, reg = net(x)

    probs = torch.softmax(logits, dim=1)
    arng = torch.arange(classes_num, device=device).float()
    cls_exp = (probs * arng).sum(dim=1)

    if weights_ok:
        reg_val = reg.view(-1)
        finite = torch.isfinite(reg_val)
        reasonable = reg_val.abs() <= 20.0

        if head_ok:
            use_reg = (finite & reasonable).float()
            blended = 0.35 * cls_exp + 0.65 * reg_val
            blended = use_reg * blended + (1.0 - use_reg) * cls_exp
        else:
            blended = torch.clamp(reg_val, -1.0, 5.0)
            blended = torch.where(finite & reasonable, blended, cls_exp)
    else:
        blended = cls_exp

    return blended.detach().cpu().numpy().astype(np.float64)


thresholds_default = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
thresholds = thresholds_default.copy()

do_threshold_opt = bool(weights_ok and os.path.exists(TRAIN_CSV))

if do_threshold_opt:
    train_df = pd.read_csv(TRAIN_CSV, dtype={"id_code": str})
    y_all = train_df["diagnosis"].astype(int).values

    rng = np.random.RandomState(0)

    max_val_n = 1800  # still bounded for runtime
    per_class_floor = 120  # ensure each class contributes
    per_class_cap = max(per_class_floor, max_val_n // classes_num)

    val_parts = []
    for c in range(classes_num):
        idxs = np.where(y_all == c)[0]
        rng.shuffle(idxs)
        take = min(per_class_cap, len(idxs))
        val_parts.append(idxs[:take])

    val_idx = np.concatenate(val_parts, axis=0)
    rng.shuffle(val_idx)

    if len(val_idx) > max_val_n:
        val_idx = val_idx[:max_val_n]

    val_df = train_df.iloc[val_idx].reset_index(drop=True)
    y_val = val_df["diagnosis"].astype(int).values
    val_paths = [f"{TRAIN_IMG_DIR}/{idx}.png" for idx in val_df["id_code"].tolist()]

    bs = 16 if device.startswith("cuda") else 8
    preds_val = []
    with torch.inference_mode():
        for i in range(0, len(val_paths), bs):
            preds_val.append(_infer_continuous_batch(val_paths[i : i + bs]))
    x_val = np.concatenate(preds_val, axis=0)

    if float(np.std(x_val)) < 1e-6:
        thresholds = thresholds_default.copy()
        thr_qwk = None
        do_threshold_opt = False
    else:
        thresholds_fit, thr_qwk_fit = fit_thresholds_grid(
            x_val,
            y_val,
            init_thr=thresholds_default,
            n_classes=classes_num,
            span=1.1,
            steps=71,
            iters=3,
        )
        thresholds_fit = _sanitize_thresholds(thresholds_fit, n_classes=classes_num)

        thr_qwk_default = qwk(
            y_val, apply_thresholds(x_val, thresholds_default), n_classes=classes_num
        )

        cut = int(0.8 * len(x_val))
        x_a, y_a = x_val[:cut], y_val[:cut]
        x_b, y_b = x_val[cut:], y_val[cut:]

        fit_a = qwk(y_a, apply_thresholds(x_a, thresholds_fit), n_classes=classes_num)
        fit_b = qwk(y_b, apply_thresholds(x_b, thresholds_fit), n_classes=classes_num)
        def_a = qwk(
            y_a, apply_thresholds(x_a, thresholds_default), n_classes=classes_num
        )
        def_b = qwk(
            y_b, apply_thresholds(x_b, thresholds_default), n_classes=classes_num
        )

        if (thr_qwk_fit >= thr_qwk_default + 1e-6) and (fit_b >= def_b - 1e-6):
            thresholds = thresholds_fit
            thr_qwk = float(thr_qwk_fit)
        else:
            thresholds = thresholds_default.copy()
            thr_qwk = float(thr_qwk_default)
else:
    thr_qwk = None

submission = []
with torch.inference_mode():
    test_paths = [f"{TEST_IMG_DIR}/{idx}.png" for idx in test_ids]
    bs = 16 if device.startswith("cuda") else 8

    preds_cont = []
    for i in range(0, len(test_paths), bs):
        preds_cont.append(_infer_continuous_batch(test_paths[i : i + bs]))
    preds_cont = np.concatenate(preds_cont, axis=0)

    preds_cls = apply_thresholds(preds_cont, thresholds)
    preds_cls = np.clip(preds_cls, 0, classes_num - 1).astype(int)

    for idx, pred in zip(test_ids, preds_cls.tolist()):
        submission.append([idx, int(pred)])

submission = np.array(submission, dtype=object)



## === cell 1
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

df.to_csv("submission.csv", index=False)

print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print(
    "weights_ok:",
    weights_ok,
    "| head_ok:",
    head_ok,
    "| head_stats:",
    head_stats,
    "| used_threshold_opt:",
    do_threshold_opt,
    "| thr_qwk:",
    thr_qwk,
)
print("Using thresholds:", thresholds.tolist())
