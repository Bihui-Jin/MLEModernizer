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

0.9024844612384112

# 6. Current score

0.32821

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.01643) has done: 'I fix the missing weight-file issue by loading the provided EfficientNet-B4 model with ImageNet pretrained weights via `timm` when the competition-specific `.pth` file is not present, so the notebook runs end-to-end and writes a valid `submission.csv`. I also fix a couple of runtime bugs: missing `Parameter` import for the GeM layer, a hardcoded CUDA device that can crash on CPU-only runs, and a `torch.load` compatibility issue by using `map_location`. Finally, I ensure predictions are generated safely (no grad, robust tensor-to-scalar conversion) and that the submission dataframe has the exact required columns.'
- What this solution (achieved -0.11977) has done: 'Your current score is extremely low because the model is effectively an ImageNet-pretrained EfficientNet with randomly initialized ordinal heads, so predictions are close to random. To move the score toward your target while keeping the same core architecture and inference semantics, I (1) load the backbone with `num_classes=0` so it outputs proper features, and (2) attach the four ordinal heads on top of those features with the correct input dimension (matching EfficientNet-B4’s feature size). This is a minimal, directly relevant fix that makes the model’s forward pass coherent (features → heads) and should substantially improve kappa relative to the current near-random setup, while still producing the same `submission.csv` format. If your competition-specific weights file exists, the code still try to load it (with key-cleaning) and work whether it contains a full model or just a state_dict.'
- What this solution (achieved 0.01693) has done: 'Your score is very low because you’re effectively doing random ordinal classification: the four linear “ordinal heads” are randomly initialized (since your competition weights file isn’t present), and the hard 0.5 thresholds make predictions unstable and poorly calibrated for quadratic kappa. To move the score upward toward your target while preserving the same model/backbone/heads and inference semantics, I (1) add a tiny, deterministic calibration step that fits the four ordinal thresholds on the training set using the *same* forward pass (no training, no architecture change), optimizing quadratic weighted kappa, and then (2) apply those fitted thresholds at test-time. I also vectorize inference (batching) to keep runtime within limits and ensure the submission rows align exactly with `test.csv`. This is a minimal change that directly improves the metric alignment without altering your core modeling approach.'
- What this solution (achieved 0.07427) has done: 'Your current score is far below the target, and the biggest issue is that you’re calibrating thresholds using predictions from an essentially untrained ordinal head (ImageNet backbone + random linear layers), so the calibrated thresholds won’t generalize and kappa stays near random. Keeping the same model, loss-free approach, and prediction semantics, I (1) compute a closed-form, deterministic least-squares fit from backbone features to the ordinal targets on the full training set to set the four linear heads (no training loop, no architecture change), then (2) redo the same small grid threshold calibration on train as you already do. This preserves the core logic (EfficientNet-B4 backbone → 4 sigmoid ordinal heads → thresholding) but makes the heads meaningful, which should move QWK substantially upward toward your target while still writing a valid `submission.csv`. I also keep runtime safe by caching features for train/test so we don’t re-run the backbone multiple times.'
- What this solution (achieved 0.18993) has done: 'Your current score (0.07427) is far below the target (0.90248), so we should improve generalization while keeping your exact backbone→4 ordinal heads→thresholding pipeline intact. The biggest issue is that you’re fitting the ordinal heads and thresholds using in-sample predictions from the same data, which overfits and won’t translate to test QWK; we instead fit heads on a training split and choose thresholds on a held-out validation split (same closed-form ridge fit, same threshold grid). We then refit the heads on all training data using the chosen thresholds (no training loop added) and generate test predictions. This is a minimal change that directly aligns your calibration with the evaluation metric and typically yields a large jump in QWK versus in-sample threshold tuning, while still producing a valid `submission.csv`.'
- What this solution (achieved 0.13893) has done: 'Your gap to the target is large (0.18993 vs 0.90248, higher-is-better), so the most direct minimal improvement is to reduce overfitting and better align predictions with quadratic kappa without changing the backbone→4 ordinal heads→thresholding core. I keep your closed-form ridge head fitting, but replace the single 80/20 split + coarse grid with (1) stratified K-fold out-of-fold (OOF) predictions for robust threshold calibration and (2) a small coordinate-descent refinement of thresholds (still deterministic, fast) starting from the coarse grid optimum. Then I refit the ordinal heads on all training data (same closed-form) and apply the calibrated thresholds to test, ensuring `submission.csv` is valid and aligned to `test.csv`. Runtime stays bounded by extracting train/test features once and reusing them across folds.'
- What this solution (achieved 0.30477) has done: 'Your current pipeline is still far below the target, so the smallest high-impact fix is to make the closed-form ordinal head fit less brittle and better aligned with QWK without changing the backbone→4 sigmoid heads→thresholding semantics. I (1) standardize the extracted features (fit on train, apply to val/test) so the ridge solve is numerically better conditioned, and (2) tune the ridge regularization strength `lam` via the same OOF QWK proxy you already use, then refit on all data with the selected `lam`. I also enforce monotonic thresholds `t1<=t2<=t3<=t4` during search/refinement (same thresholding, just avoids invalid orderings that hurt kappa). These are minimal, deterministic changes that typically move QWK upward while keeping your core logic and submission format unchanged.'
- What this solution (achieved 0.30477) has done: 'Your current score (0.30477) is far below the target (0.90248), so we should improve generalization without changing the backbone→4 ordinal sigmoid heads→thresholding core. The biggest low-risk gain is to make threshold calibration more faithful by generating out-of-fold (OOF) probabilities from heads trained on *training folds only*, using a standardizer fit on each fold (to avoid subtle leakage), and then searching/refining thresholds on those OOF probabilities. This keeps the same closed-form ridge head fitting and the same thresholding semantics, but reduces overfitting in the calibration step that directly drives QWK. Finally, we refit on all training data (as you already do) and write a valid `submission.csv` with correct alignment.'
- What this solution (achieved 0.30477) has done: 'Your current score is far below the target, so we should improve generalization while keeping your exact pipeline (EfficientNet features → closed-form ridge fit for 4 ordinal sigmoid heads → thresholding) unchanged. The highest-impact minimal fix is to remove a subtle but important fold-to-fold contamination: you reuse the same `net` across folds, so later folds start from weights fit on earlier folds rather than a clean state, which distorts OOF probabilities and hurts threshold/lambda selection. I snapshot the initial head weights once and restore them at the start of every fold before the closed-form fit (backbone stays untouched), making OOF calibration faithful and typically improving QWK. I also make the ridge “bias” unregularized (standard practice) to slightly improve fit without changing the approach, and keep everything deterministic and within runtime limits.'
- What this solution (achieved 0.33239) has done: 'Your current score (0.30477) is far below the target (0.90248), so we should increase performance while keeping your backbone→features→closed-form ridge ordinal heads→thresholding pipeline intact. The biggest minimal gain is to calibrate thresholds in a way that better matches quadratic kappa by allowing each threshold to move independently around a sensible starting point derived from the data, while still enforcing monotonicity. I keep your OOF setup and ridge fitting unchanged, but (1) expand the coarse threshold search adaptively around class-conditional percentiles of the OOF probabilities and (2) slightly strengthen coordinate refinement (more but still small iterations/step schedule) to better land near the kappa optimum without changing semantics. These changes are deterministic, low-risk, and directly improve the kappa-aligned post-processing that dominates your final score.'
- What this solution (achieved 0.32777) has done: 'Your current score (0.33239) is far below the target (0.90248), so we should increase performance while keeping your exact pipeline (EfficientNet features → closed-form ridge heads → sigmoid ordinal probs → thresholding) unchanged. The biggest low-risk issue is that your OOF threshold search uses a very wide grid (9^4) that is both expensive and still coarse in the “right” local neighborhood, and the refinement can get stuck because it explores too few candidate points around each threshold. I keep the same data-driven initialization and monotonic constraint, but (1) replace the 9-point-per-threshold grid with a denser *local* grid around the initial thresholds (fewer useless combinations, more resolution where it matters) and (2) strengthen coordinate refinement slightly with a smaller initial step and a few more iterations to better optimize QWK on OOF probabilities. These are pure post-processing/calibration changes (no architecture/training-loop change), deterministic, and should move QWK upward toward your target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.31695) has done: 'Your score is still far below the target (0.32777 vs 0.90248, higher-is-better), so the most direct minimal improvement is to make the *post-processing for QWK* less brittle without changing your backbone/features/closed-form head-fit/thresholding pipeline. I keep your exact model and closed-form ridge fitting, but replace the single global threshold set with a **fold-averaged (ensemble) set of thresholds** computed per-fold from truly out-of-fold probabilities, which typically generalizes better to test than one threshold vector tuned on all OOF at once. I also make the fold split more robust by using a deterministic, dependency-free stratification that evenly distributes each class across folds (your current split can create small imbalances for rare classes), and keep everything deterministic and within the 600s runtime by reusing the already-extracted features. The submission writing remains identical (`submission.csv` with `id_code,diagnosis` aligned to `test.csv`).'
- What this solution (achieved 0.32081) has done: 'Your current score (0.31695) is far below the target (0.90248), so we should make small, directly metric-relevant changes that improve generalization without changing your backbone→features→closed-form ridge ordinal heads→sigmoid→thresholding pipeline. The biggest low-risk issue is that your fold “threshold ensemble” computes per-fold thresholds on *in-fold* validation data and then averages them, which can be unstable; we instead calibrate a single threshold vector on true out-of-fold (OOF) probabilities, which is more faithful to test behavior for QWK. To keep changes minimal, we keep your feature extraction, ridge fitting, lambda grid, and coordinate refinement, but switch threshold selection to (coarse+refine) on the full OOF probabilities after each lambda. We also add a small guard to ensure monotonic thresholds are enforced everywhere and keep runtime bounded.'
- What this solution (achieved 0.32821) has done: 'I keep your backbone→features→closed-form ridge ordinal heads→sigmoid→thresholding pipeline unchanged, and focus on one minimal but high-impact metric-alignment fix: directly optimize the 4 thresholds for **quadratic weighted kappa** using a deterministic 1D optimizer (golden-section search) per-threshold with monotonicity constraints, instead of a coarse grid + local coordinate steps that can miss good optima. This change only affects post-processing (the thresholds) and should move the score upward toward your target without altering the model, feature extraction, or head fitting. I also add a very small safety improvement: compute OOF probabilities once for the chosen lambda and then run the optimizer on that OOF set (still no leakage, still deterministic). The script still runs end-to-end and writes a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import timm
from torch.nn.parameter import Parameter

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_DIR = "../input/aptos2019-blindness-detection"
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).values

transforms_tf = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)


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
        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=True, num_classes=0
        )

        self.backbone.global_pool = GeM(flatten=True)

        in_features = getattr(self.backbone, "num_features", None)
        if in_features is None:
            in_features = 1792

        self.one = nn.Linear(in_features, 1)
        self.two = nn.Linear(in_features, 1)
        self.three = nn.Linear(in_features, 1)
        self.four = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        out1 = self.sigmoid(self.one(x))
        out2 = self.sigmoid(self.two(x))
        out3 = self.sigmoid(self.three(x))
        out4 = self.sigmoid(self.four(x))
        return out1, out2, out3, out4


def quadratic_weighted_kappa(y_true, y_pred, num_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape
    N = num_classes

    O = np.zeros((N, N), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < N and 0 <= b < N:
            O[a, b] += 1.0

    hist_true = O.sum(axis=1)
    hist_pred = O.sum(axis=0)
    E = np.outer(hist_true, hist_pred)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            W[i, j] = ((i - j) ** 2) / ((N - 1) ** 2)

    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    num = (W * O).sum()
    return 1.0 - (num / denom)


def ordinal2class_with_thresholds(o, th=(0.5, 0.5, 0.5, 0.5)):
    t = np.clip(np.array(th, dtype=np.float64), 0.05, 0.95)
    t = np.maximum.accumulate(t)
    t1, t2, t3, t4 = float(t[0]), float(t[1]), float(t[2]), float(t[3])

    y = np.zeros((o.shape[0],), dtype=int)
    y[o[:, 0] >= t1] = 1
    y[o[:, 1] >= t2] = 2
    y[o[:, 2] >= t3] = 3
    y[o[:, 3] >= t4] = 4
    return y


def load_image_tensor(img_path):
    img = Image.open(img_path).convert("RGB")
    return transforms_tf(img)


def extract_features(net, ids, img_dir, batch_size=16):
    net.eval()
    feats = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size]
            imgs = []
            for idx in batch_ids:
                image_name = os.path.join(img_dir, f"{idx}.png")
                imgs.append(load_image_tensor(image_name))
            x = torch.stack(imgs, dim=0).to(device)
            f = net.backbone(x).detach().cpu().numpy()
            feats.append(f)
    return np.concatenate(feats, axis=0)


def predict_ordinal_probs_from_features(net, features_np, batch_size=512):
    net.eval()
    probs_all = []
    with torch.no_grad():
        for start in range(0, features_np.shape[0], batch_size):
            f = torch.from_numpy(features_np[start : start + batch_size]).to(
                device=device, dtype=torch.float32
            )
            out1 = net.sigmoid(net.one(f))
            out2 = net.sigmoid(net.two(f))
            out3 = net.sigmoid(net.three(f))
            out4 = net.sigmoid(net.four(f))
            probs = torch.cat([out1, out2, out3, out4], dim=1).detach().cpu().numpy()
            probs_all.append(probs)
    return np.concatenate(probs_all, axis=0)


def fit_ordinal_heads_closed_form(net, features_np, y_int, lam=1e-2):
    """
    Deterministic ridge least-squares fit that sets the 4 ordinal linear heads
    to match (y>=k) targets. (Core logic preserved.)

    Bias term is not regularized (common and usually improves fit).
    """
    Y_ord = np.stack(
        [(y_int >= k).astype(np.float32) for k in [1, 2, 3, 4]], axis=1
    )  # (N,4)

    X = features_np.astype(np.float64)
    N, D = X.shape
    X1 = np.concatenate([X, np.ones((N, 1), dtype=np.float64)], axis=1)  # add bias

    reg = lam * np.eye(D + 1, dtype=np.float64)
    reg[D, D] = 0.0

    A = X1.T @ X1 + reg
    B = X1.T @ Y_ord.astype(np.float64)
    W = np.linalg.solve(A, B)  # (D+1, 4)
    W_w = W[:D, :].T.astype(np.float32)  # (4, D)
    W_b = W[D, :].astype(np.float32)  # (4,)

    with torch.no_grad():
        net.one.weight.copy_(torch.from_numpy(W_w[0:1, :]).to(device))
        net.one.bias.copy_(torch.tensor([W_b[0]], device=device))
        net.two.weight.copy_(torch.from_numpy(W_w[1:2, :]).to(device))
        net.two.bias.copy_(torch.tensor([W_b[1]], device=device))
        net.three.weight.copy_(torch.from_numpy(W_w[2:3, :]).to(device))
        net.three.bias.copy_(torch.tensor([W_b[2]], device=device))
        net.four.weight.copy_(torch.from_numpy(W_w[3:4, :]).to(device))
        net.four.bias.copy_(torch.tensor([W_b[3]], device=device))


def make_stratified_folds(y, n_splits=5, seed=0):
    """
    Ensure each fold gets as equal-as-possible class counts.
    Returns: list of (train_idx, val_idx)
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y, dtype=int)
    idx_all = np.arange(len(y))

    folds = [[] for _ in range(n_splits)]
    for c in range(5):
        idx_c = idx_all[y == c].copy()
        rng.shuffle(idx_c)
        for i, ix in enumerate(idx_c):
            folds[i % n_splits].append(ix)

    out = []
    for f in range(n_splits):
        val_idx = np.array(folds[f], dtype=int)
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[val_idx] = False
        train_idx = idx_all[train_mask]
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        out.append((train_idx, val_idx))
    return out


def fit_feature_standardizer(X):
    mu = X.mean(axis=0, keepdims=True).astype(np.float32)
    sd = X.std(axis=0, keepdims=True).astype(np.float32)
    sd = np.maximum(sd, 1e-6)
    return mu, sd


def apply_feature_standardizer(X, mu, sd):
    return ((X.astype(np.float32) - mu) / sd).astype(np.float32)


def _initial_thresholds_from_data(probs, y_true):
    y_true = np.asarray(y_true, dtype=int)
    th = []
    for k in range(4):
        prev = float((y_true >= (k + 1)).mean())
        q = np.clip(1.0 - prev, 0.05, 0.95)
        t = float(np.quantile(probs[:, k].astype(np.float64), q))
        th.append(t)
    th = np.maximum.accumulate(np.array(th, dtype=np.float64))
    th = np.clip(th, 0.05, 0.95)
    return (float(th[0]), float(th[1]), float(th[2]), float(th[3]))


def _qwk_for_thresholds(probs, y_true, th):
    pred = ordinal2class_with_thresholds(probs, th)
    return quadratic_weighted_kappa(y_true, pred, num_classes=5)


def _golden_section_maximize_1d(f, a, b, tol=1e-4, max_iter=80):
    gr = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    it = 0
    while (b - a) > tol and it < max_iter:
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
        it += 1
    x = (a + b) / 2.0
    fx = f(x)
    return float(x), float(fx)


def optimize_thresholds_golden(probs, y_true, th_start, n_passes=4, tol=2e-4):
    th = np.array(th_start, dtype=np.float64)
    th = np.clip(th, 0.05, 0.95)
    th = np.maximum.accumulate(th)

    best_k = _qwk_for_thresholds(probs, y_true, tuple(th))

    for _ in range(n_passes):
        for j in range(4):
            lo = 0.05 if j == 0 else float(th[j - 1])
            hi = 0.95 if j == 3 else float(th[j + 1])

            if hi - lo < 1e-6:
                continue

            def f1d(tj):
                th_try = th.copy()
                th_try[j] = float(tj)
                th_try = np.maximum.accumulate(np.clip(th_try, 0.05, 0.95))
                return _qwk_for_thresholds(probs, y_true, tuple(th_try))

            tj_opt, k_opt = _golden_section_maximize_1d(
                f1d, lo, hi, tol=tol, max_iter=80
            )
            if k_opt >= best_k - 1e-12:
                th[j] = tj_opt
                th = np.maximum.accumulate(np.clip(th, 0.05, 0.95))
                best_k = k_opt

    th = np.maximum.accumulate(np.clip(th, 0.05, 0.95))
    return (float(th[0]), float(th[1]), float(th[2]), float(th[3])), float(best_k)


WEIGHTS_PATH = "../input/weights/efficientd4_ns_Krank.pth"

if os.path.exists(WEIGHTS_PATH):
    loaded = torch.load(WEIGHTS_PATH, map_location=device)
    if isinstance(loaded, nn.Module):
        net = loaded
    else:
        net = Model()
        state = loaded.get("state_dict", loaded) if isinstance(loaded, dict) else loaded
        cleaned = {}
        if isinstance(state, dict):
            for k, v in state.items():
                ck = k
                if ck.startswith("model."):
                    ck = ck[len("model.") :]
                if ck.startswith("net."):
                    ck = ck[len("net.") :]
                if ck.startswith("module."):
                    ck = ck[len("module.") :]
                cleaned[ck] = v
            net.load_state_dict(cleaned, strict=False)
else:
    net = Model()

net = net.to(device)
net.eval()

initial_head_state = {
    "one_w": net.one.weight.detach().clone(),
    "one_b": net.one.bias.detach().clone(),
    "two_w": net.two.weight.detach().clone(),
    "two_b": net.two.bias.detach().clone(),
    "three_w": net.three.weight.detach().clone(),
    "three_b": net.three.bias.detach().clone(),
    "four_w": net.four.weight.detach().clone(),
    "four_b": net.four.bias.detach().clone(),
}


def restore_initial_heads(net, state):
    with torch.no_grad():
        net.one.weight.copy_(state["one_w"])
        net.one.bias.copy_(state["one_b"])
        net.two.weight.copy_(state["two_w"])
        net.two.bias.copy_(state["two_b"])
        net.three.weight.copy_(state["three_w"])
        net.three.bias.copy_(state["three_b"])
        net.four.weight.copy_(state["four_w"])
        net.four.bias.copy_(state["four_b"])


train_df = pd.read_csv(TRAIN_CSV)
train_ids = train_df["id_code"].astype(str).values
y_train = train_df["diagnosis"].astype(int).values

train_features_all = extract_features(net, train_ids, TRAIN_IMG_DIR, batch_size=16)
test_features = extract_features(net, test_ids, TEST_IMG_DIR, batch_size=16)

folds = make_stratified_folds(y_train, n_splits=5, seed=0)

lam_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
best_lam = 1e-2
best_lam_k = -1e9
best_lam_th = (0.5, 0.5, 0.5, 0.5)

for lam in lam_grid:
    oof_probs = np.zeros((len(train_ids), 4), dtype=np.float32)

    for tr_idx, va_idx in folds:
        mu_fold, sd_fold = fit_feature_standardizer(train_features_all[tr_idx])
        X_tr = apply_feature_standardizer(train_features_all[tr_idx], mu_fold, sd_fold)
        X_va = apply_feature_standardizer(train_features_all[va_idx], mu_fold, sd_fold)

        restore_initial_heads(net, initial_head_state)
        fit_ordinal_heads_closed_form(net, X_tr, y_train[tr_idx], lam=lam)

        va_probs = predict_ordinal_probs_from_features(
            net, X_va, batch_size=512
        ).astype(np.float32)
        oof_probs[va_idx] = va_probs

    th0 = _initial_thresholds_from_data(oof_probs, y_train)
    th1, k1 = optimize_thresholds_golden(oof_probs, y_train, th0, n_passes=4, tol=2e-4)

    oof_pred = ordinal2class_with_thresholds(oof_probs, th1)
    k_oof = quadratic_weighted_kappa(y_train, oof_pred, num_classes=5)

    if k_oof > best_lam_k:
        best_lam_k = float(k_oof)
        best_lam = lam
        best_lam_th = th1

best_th = best_lam_th
best_k = best_lam_k

mu_all, sd_all = fit_feature_standardizer(train_features_all)
train_features_all_std = apply_feature_standardizer(train_features_all, mu_all, sd_all)
test_features_std = apply_feature_standardizer(test_features, mu_all, sd_all)

restore_initial_heads(net, initial_head_state)
fit_ordinal_heads_closed_form(net, train_features_all_std, y_train, lam=best_lam)

test_probs = predict_ordinal_probs_from_features(net, test_features_std, batch_size=512)
test_pred = ordinal2class_with_thresholds(test_probs, best_th)

submission = pd.DataFrame({"id_code": test_ids, "diagnosis": test_pred.astype(int)})



## === cell 1
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Selected lambda (OOF):", best_lam)
print("Selected thresholds (OOF, golden-opt):", best_th, "OOF QWK (proxy):", best_k)
print("Wrote submission.csv with shape:", submission.shape)
