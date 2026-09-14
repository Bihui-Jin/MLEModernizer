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

0.27171

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.09193) has done: 'I remove the broken dependency on a missing `../input/alexnet` checkpoint and instead instantiate an AlexNet classifier directly from `torchvision` so inference can run end-to-end. I also fix notebook-only syntax (`%matplotlib inline`) so the script runs as a Kaggle Python script, and make test image loading deterministic by reading IDs from `test.csv` rather than `os.listdir()` ordering. Finally, I ensure predictions are generated on the available GPU if present, and that the output `submission.csv` has exactly the required columns and 367 rows.'
- What this solution (achieved 0.29038) has done: 'Your score is very low because the model’s final layer was replaced with a new random 5-class head, so predictions are essentially random with respect to DR severity. To move toward the target with minimal change and without altering the core inference approach, I keep AlexNet and ImageNet preprocessing but stop reinitializing the classifier head. Instead, I use the pretrained 1000-class logits and map them deterministically into 5 ordinal classes via fixed percentile binning on a scalar “severity proxy” derived from the logits (expected class index), which typically gives a materially better-than-random ordering. I also ensure the submission is strictly aligned to `test.csv` order and write `submission.csv` with the required columns.'
- What this solution (achieved 0.39293) has done: 'Your current mapping from ImageNet outputs to 5 DR classes uses *test-set quantile binning*, which forces an artificial uniform class distribution and typically harms quadratic weighted kappa because APTOS labels are highly imbalanced toward class 0. To move the score upward toward the 0.70 target while keeping the same model/inference core, I keep AlexNet and the same “severity proxy” computation, but calibrate the 5-class bin thresholds using the *training label distribution* (no leakage: labels only used to set global priors). Concretely, we compute severity scores for train images too, then choose four cutpoints so predicted class proportions on train match the true label proportions, and apply those cutpoints to test. This is a minimal change (only post-processing/calibration) and is expected to improve kappa substantially versus uniform binning.'
- What this solution (achieved -0.41353) has done: 'Your current pipeline is already end-to-end and deterministic, but the “expected ImageNet class index” severity proxy is a weak ordering signal for DR, which caps kappa well below your 0.70 target. To move the score upward with minimal semantic change, I keep the exact same AlexNet inference and data pipeline, but compute the severity proxy from the **penultimate-layer embedding** (AlexNet `classifier[:-1]`) and use its **L2 norm** as a more stable, domain-agnostic “image abnormality/complexity” score than the arbitrary 0–999 class index. I keep your **train-label-distribution calibration** (quantile cutpoints on train severity) unchanged, just swapping the severity definition, and I also ensure images are fully loaded (`.convert("RGB")`) and files are closed to avoid intermittent I/O issues. This is a pure post-processing/feature hook change (no training, no architecture edits) and is typically a meaningful lift over the current proxy, moving closer to the target.'
- What this solution (achieved 0.31527) has done: 'Your current negative score strongly suggests the severity-to-class mapping is effectively inverted or miscalibrated for this feature, so the smallest reliable step is to learn the *direction* (ascending vs descending severity) from train labels and apply the better one. I keep the exact same AlexNet feature extraction and L2-norm severity proxy, and the same “match train label proportions via quantile cutpoints” calibration, but compute bins on **ascending severity** and choose whether to flip the severity sign based on which yields higher train quadratic weighted kappa. This is a minimal post-processing change (no training, no architecture changes) and should move the score upward toward your 0.70 target. I also add a deterministic sanity check (train kappa printout) while still producing the same `submission.csv` format and order.'
- What this solution (achieved 0.373) has done: 'Your current score (0.31527) is far below the target (0.70049), so we should make a small but meaningful post-processing change that improves alignment with quadratic weighted kappa without changing the model or adding training. The biggest issue is that “match train label proportions via quantiles” does not optimize kappa; instead, we can learn 4 ordinal thresholds on the train severity scores that directly maximize train QWK (using out-of-fold predictions to reduce overfitting) while keeping the same AlexNet feature extraction and the same severity definition. Concretely, we keep your L2-norm severity proxy, but replace the quantile cutpoints with a tiny coordinate-ascent threshold search on OOF severity-to-label mapping, then apply the learned thresholds to test (still optionally allowing flip based on OOF QWK). This is still just calibration/post-processing, runs quickly, and is much more likely to move the score upward toward your 0.70 target than fixed-proportion binning.'
- What this solution (achieved 0.38234) has done: 'Your current score (0.373) is far below the target (0.7005), so we should increase performance with the smallest change that better aligns the post-processing with quadratic weighted kappa while keeping AlexNet feature extraction and the “severity score → 5 classes via thresholds” core intact. The main issue is that the current threshold search optimizes thresholds on in-fold train severities (and only uses OOF to choose flip), which can overfit and generalize poorly to test. I change the calibration to **select the final thresholds purely by maximizing OOF QWK** (averaging fold-specific thresholds), and also make the coordinate-ascent search **more stable** by using candidate thresholds from the *current fold’s sev distribution* rather than global quantiles, without changing the model or inference. This stays within the same approach (no training, same network, same severity proxy, same metric) but should move the score upward toward the target.'
- What this solution (achieved 0.39655) has done: 'Your current pipeline’s biggest limitation is that the severity score is extracted from `model.features` without the AlexNet input normalization that `model.forward()` expects, which can distort embeddings and weaken the severity ordering signal; I keep the exact same architecture and “embedding L2 norm → thresholds” logic, but run the image through the full AlexNet up to `classifier[:-1]` so preprocessing is consistent. Next, the fold split is currently deterministic by `id_code` sort, which can create systematic fold bias; I keep 5-fold OOF calibration but switch to a seeded shuffled fold assignment for a more representative OOF threshold fit (same semantics, less bias). Finally, I keep your coordinate-ascent threshold optimizer unchanged in spirit, but slightly increase candidate resolution/iterations in a controlled way to better match QWK without changing the modeling approach, still staying well within the 600s budget.'
- What this solution (achieved 0.40216) has done: 'Your current score (0.39655) is far below the target (0.70049), so we need a small but meaningful lift without changing the model or introducing training. The safest improvement is to calibrate the 4 ordinal thresholds in a way that better matches quadratic weighted kappa by (a) using stratified folds (reduces OOF bias/variance vs purely shuffled folds) and (b) choosing final thresholds by directly maximizing OOF QWK via a lightweight search around the averaged fold thresholds (same thresholding approach, just a more metric-aligned selection). I keep your AlexNet embedding L2 “severity” proxy and the coordinate-ascent optimizer, but make the fold assignment stratified and add a tiny post-OOF refinement step for the final thresholds (and still allow flip/no-flip selection based on OOF QWK). This remains fast and deterministic, and still writes a valid `submission.csv` with the correct schema and order.'
- What this solution (achieved 0.40551) has done: 'Your score gap to the target is large (0.40216 vs 0.70049), but we still should keep the same AlexNet feature-based severity + thresholding calibration core and only make small calibration changes that better match QWK. The biggest low-risk lift here is to make the out-of-fold threshold fitting less “self-inflated” by selecting the final thresholds using OOF QWK (not full-train QWK), and to ensemble fold thresholds in a QWK-aligned way (choose the fold-threshold set that gives best overall OOF when applied globally, then do a tiny local refinement scored on OOF). This keeps the same severity proxy, same coordinate-ascent optimizer, same stratified folds, but makes the “which thresholds do we actually use” decision match the evaluation metric more directly. The submission writing, ordering, and schema remain unchanged.'
- What this solution (achieved 0.40551) has done: 'Your current pipeline is already producing a valid submission and the main limiter is that the threshold calibration is being scored on the same data used to fit thresholds (even though it’s OOF-predicted, the final “refinement” still scores on all points), which can overfit and generalize poorly to test. I keep the exact AlexNet embedding L2 severity proxy and the same coordinate-ascent threshold fitting, but I generate true OOF predictions once, then choose/refine the single global threshold set by maximizing QWK on those fixed OOF predictions only. This is a minimal, metric-aligned calibration change (no model/training/feature changes) and is expected to lift QWK toward your 0.7005 target while staying deterministic and within time. The submission writing, ordering, and schema remain unchanged.'
- What this solution (achieved 0.27171) has done: 'Your current score (0.40551) is far below the target (0.70049), so we should improve the QWK signal with the smallest change that preserves the same “AlexNet feature → scalar severity → learned ordinal thresholds” core. The main issue is that using only the embedding L2 norm is a very weak severity proxy; we can strengthen it without changing architecture or training by deriving severity from the pretrained logits as well (still pure inference) and blending the two proxies. Then we keep your exact OOF threshold-fitting logic, but fit thresholds on the blended severity (and still learn whether to flip) so the calibration is more metric-aligned. This is a minimal, legitimate change (no training, same model, same thresholding approach) that is expected to move the score upward toward the target.'

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

embed_extractor = torch.nn.Sequential(
    model.features,
    model.avgpool,
    torch.nn.Flatten(start_dim=1),
    *list(model.classifier.children())[:-1],
).to(device)
embed_extractor.eval()




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
        with Image.open(path) as im:
            image = im.convert("RGB")
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

    with torch.no_grad():
        for imgs, ids in data_loader:
            imgs = imgs.to(device, non_blocking=True)

            emb = embed_extractor(imgs)  # [B, 4096]
            sev = torch.linalg.vector_norm(emb, ord=2, dim=1)  # [B]

            id_codes.extend(list(ids))
            severity_scores.extend(
                sev.detach().cpu().numpy().astype(np.float32).tolist()
            )

    return np.asarray(id_codes), np.asarray(severity_scores, dtype=np.float32)


def compute_blended_severity_scores(data_loader, alpha=0.70):
    """
    alpha: weight on embedding-L2 severity; (1-alpha) weight on logits-based proxy.
    """
    id_codes = []
    sev_blend = []

    with torch.no_grad():
        for imgs, ids in data_loader:
            imgs = imgs.to(device, non_blocking=True)

            emb = embed_extractor(imgs)  # [B, 4096]
            sev_emb = torch.linalg.vector_norm(emb, ord=2, dim=1)  # [B]

            logits = model(imgs)  # [B, 1000]
            probs = torch.softmax(logits, dim=1)
            cls_idx = torch.arange(
                probs.shape[1], device=probs.device, dtype=probs.dtype
            )
            sev_logits = (probs * cls_idx[None, :]).sum(dim=1)  # expected class index

            def z(x):
                m = x.mean()
                s = x.std(unbiased=False).clamp_min(1e-6)
                return (x - m) / s

            sev = alpha * z(sev_emb) + (1.0 - alpha) * z(sev_logits)

            id_codes.extend(list(ids))
            sev_blend.extend(sev.detach().cpu().numpy().astype(np.float32).tolist())

    return np.asarray(id_codes), np.asarray(sev_blend, dtype=np.float32)


train_ids, train_sev = compute_blended_severity_scores(train_loader, alpha=0.70)
test_ids, test_sev = compute_blended_severity_scores(test_loader, alpha=0.70)

train_sev_df = pd.DataFrame({"id_code": train_ids, "severity": train_sev})
train_sev_df = train_df[["id_code", "diagnosis"]].merge(
    train_sev_df, on="id_code", how="left"
)
assert (
    train_sev_df["severity"].notna().all()
), "Missing train severity scores (image load/inference issue?)"




## === cell 6
def quadratic_weighted_kappa(y_true, y_pred, num_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape

    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < num_classes and 0 <= b < num_classes:
            O[a, b] += 1.0

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    return 1.0 - (W * O).sum() / denom


def predict_with_thresholds(severity, thresholds):
    s = np.asarray(severity, dtype=np.float32)
    t = np.asarray(thresholds, dtype=np.float32)
    return np.digitize(s, t, right=False).astype(int)


def make_strictly_increasing(t):
    t = np.asarray(t, dtype=np.float32).copy()
    eps = 1e-6
    for i in range(1, len(t)):
        if not (t[i] > t[i - 1]):
            t[i] = t[i - 1] + eps
            eps *= 2.0
    return t


def init_thresholds_from_label_priors(sev, y):
    label_counts = pd.Series(y).value_counts().sort_index()
    label_props = (
        (label_counts / label_counts.sum())
        .reindex([0, 1, 2, 3, 4])
        .fillna(0.0)
        .to_numpy()
    )
    cum_props = np.cumsum(label_props)
    cut_quantiles = np.clip(cum_props[:4], 1e-6, 1 - 1e-6)
    init_t = np.quantile(sev, cut_quantiles).astype(np.float32)
    return make_strictly_increasing(init_t)


def stratified_fold_id(y, n_folds=5, seed=42):
    """
    Use stratified fold assignment by label to reduce OOF calibration variance/bias.
    """
    y = np.asarray(y, dtype=int)
    rng = np.random.RandomState(seed)
    fold_id = np.empty(len(y), dtype=int)

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        fold_id[idx] = np.arange(len(idx), dtype=int) % n_folds
    return fold_id


def optimize_thresholds_coordinate_ascent(
    sev, y, init_thresholds, n_iter=3, grid_size=90
):
    sev = np.asarray(sev, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    t = make_strictly_increasing(init_thresholds)

    qs = np.linspace(0.005, 0.995, grid_size, dtype=np.float32)
    cand_vals = np.quantile(sev, qs).astype(np.float32)

    best_t = t.copy()
    best_k = quadratic_weighted_kappa(
        y, predict_with_thresholds(sev, best_t), num_classes=5
    )

    for _ in range(n_iter):
        for j in range(4):
            low = -np.inf if j == 0 else best_t[j - 1] + 1e-6
            high = np.inf if j == 3 else best_t[j + 1] - 1e-6

            candidates = cand_vals[(cand_vals > low) & (cand_vals < high)]
            if candidates.size == 0:
                candidates = np.asarray([best_t[j]], dtype=np.float32)

            local_best_k = best_k
            local_best_val = best_t[j]

            for v in candidates:
                t_try = best_t.copy()
                t_try[j] = float(v)
                t_try = make_strictly_increasing(t_try)
                pred = predict_with_thresholds(sev, t_try)
                k = quadratic_weighted_kappa(y, pred, num_classes=5)
                if k > local_best_k:
                    local_best_k = k
                    local_best_val = float(v)

            best_t[j] = local_best_val
            best_t = make_strictly_increasing(best_t)
            best_k = quadratic_weighted_kappa(
                y, predict_with_thresholds(sev, best_t), num_classes=5
            )

    return best_t, best_k


def refine_thresholds_around_center(
    fit_sev,
    center_t,
    span_q=0.08,
    grid_size=35,
    n_iter=2,
    score_sev=None,
    score_y=None,
):
    """
    We score candidate thresholds ONLY on fixed OOF predictions (score_sev/score_y)
    to reduce overfitting, keeping the same threshold refinement logic.
    """
    fit_sev = np.asarray(fit_sev, dtype=np.float32)
    t = make_strictly_increasing(center_t)

    assert score_sev is not None and score_y is not None, "score_sev/score_y required"
    score_sev = np.asarray(score_sev, dtype=np.float32)
    score_y = np.asarray(score_y, dtype=int)

    base_qs = np.linspace(0.0, 1.0, 1001, dtype=np.float32)
    fit_sev_q = np.quantile(fit_sev, base_qs).astype(np.float32)

    def q_of_value(v):
        pos = int(np.searchsorted(fit_sev_q, v, side="left"))
        pos = max(0, min(pos, len(base_qs) - 1))
        return float(base_qs[pos])

    center_qs = [q_of_value(v) for v in t]

    best_t = t.copy()
    best_k = quadratic_weighted_kappa(
        score_y, predict_with_thresholds(score_sev, best_t), num_classes=5
    )

    for _ in range(n_iter):
        for j in range(4):
            q0 = center_qs[j]
            loq = max(0.001, q0 - span_q)
            hiq = min(0.999, q0 + span_q)

            qs = np.linspace(loq, hiq, grid_size, dtype=np.float32)
            cand_vals = np.quantile(fit_sev, qs).astype(np.float32)

            low = -np.inf if j == 0 else best_t[j - 1] + 1e-6
            high = np.inf if j == 3 else best_t[j + 1] - 1e-6
            cand_vals = cand_vals[(cand_vals > low) & (cand_vals < high)]
            if cand_vals.size == 0:
                continue

            local_best_k = best_k
            local_best_val = best_t[j]
            for v in cand_vals:
                t_try = best_t.copy()
                t_try[j] = float(v)
                t_try = make_strictly_increasing(t_try)
                k = quadratic_weighted_kappa(
                    score_y, predict_with_thresholds(score_sev, t_try), num_classes=5
                )
                if k > local_best_k:
                    local_best_k = k
                    local_best_val = float(v)

            best_t[j] = local_best_val
            best_t = make_strictly_increasing(best_t)
            best_k = quadratic_weighted_kappa(
                score_y, predict_with_thresholds(score_sev, best_t), num_classes=5
            )

    return best_t, float(best_k)


def fit_thresholds_from_oof(sev, y, fold_id):
    """
    Build fixed OOF predictions first, then select/refine the single global threshold
    set by maximizing QWK on those fixed OOF points only.
    """
    sev = np.asarray(sev, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    fold_id = np.asarray(fold_id, dtype=int)

    init_t = init_thresholds_from_label_priors(sev, y)

    oof_pred = np.zeros_like(y, dtype=int)

    fold_thresholds = []
    fold_val_kappas = []
    uniq_folds = np.unique(fold_id)

    for f in uniq_folds:
        tr_idx = fold_id != f
        va_idx = fold_id == f

        t_f, _k_f_tr = optimize_thresholds_coordinate_ascent(
            sev[tr_idx], y[tr_idx], init_thresholds=init_t, n_iter=3, grid_size=90
        )
        fold_thresholds.append(make_strictly_increasing(t_f))

        oof_pred[va_idx] = predict_with_thresholds(sev[va_idx], t_f)
        k_f_va = quadratic_weighted_kappa(y[va_idx], oof_pred[va_idx], num_classes=5)
        fold_val_kappas.append(k_f_va)

    oof_k = float(quadratic_weighted_kappa(y, oof_pred, num_classes=5))

    mean_t = np.mean(np.stack(fold_thresholds, axis=0), axis=0).astype(np.float32)
    mean_t = make_strictly_increasing(mean_t)

    global_k_by_fold_t = []
    for t_f in fold_thresholds:
        k = quadratic_weighted_kappa(
            y, predict_with_thresholds(sev, t_f), num_classes=5
        )
        global_k_by_fold_t.append(float(k))

    best_fold_idx = int(np.argmax(global_k_by_fold_t))
    best_fold_t = fold_thresholds[best_fold_idx]

    k_mean = float(
        quadratic_weighted_kappa(y, predict_with_thresholds(sev, mean_t), num_classes=5)
    )
    k_bestfold = float(
        quadratic_weighted_kappa(
            y, predict_with_thresholds(sev, best_fold_t), num_classes=5
        )
    )

    center_t = best_fold_t if k_bestfold >= k_mean else mean_t

    refined_t, refined_k = refine_thresholds_around_center(
        fit_sev=sev,
        center_t=center_t,
        span_q=0.08,
        grid_size=35,
        n_iter=2,
        score_sev=sev,
        score_y=y,
    )

    return (
        init_t,
        oof_k,
        mean_t,
        k_mean,
        best_fold_t,
        k_bestfold,
        refined_t,
        refined_k,
        np.asarray(fold_val_kappas, dtype=np.float32),
        np.asarray(global_k_by_fold_t, dtype=np.float32),
        best_fold_idx,
    )


y_train = train_sev_df["diagnosis"].to_numpy(dtype=int)
sev_all = train_sev_df["severity"].to_numpy(dtype=np.float32)

fold_id = stratified_fold_id(y_train, n_folds=5, seed=SEED)

(
    init_pos,
    oof_k_pos,
    mean_t_pos,
    mean_k_pos,
    best_fold_t_pos,
    best_fold_k_pos,
    ref_t_pos,
    ref_oof_k_pos,
    fold_k_pos,
    global_k_pos,
    best_fold_idx_pos,
) = fit_thresholds_from_oof(sev_all, y_train, fold_id)

(
    init_neg,
    oof_k_neg,
    mean_t_neg,
    mean_k_neg,
    best_fold_t_neg,
    best_fold_k_neg,
    ref_t_neg,
    ref_oof_k_neg,
    fold_k_neg,
    global_k_neg,
    best_fold_idx_neg,
) = fit_thresholds_from_oof(-sev_all, y_train, fold_id)

use_flip = oof_k_neg > oof_k_pos
chosen_t = ref_t_neg if use_flip else ref_t_pos

print(
    f"OOF QWK (no flip): {oof_k_pos:.5f} | (flip): {oof_k_neg:.5f} | using_flip={use_flip}"
)
print(f"Mean-threshold OOF QWK (no flip): {mean_k_pos:.5f} | (flip): {mean_k_neg:.5f}")
print(
    f"Best-fold-threshold OOF QWK (no flip): {best_fold_k_pos:.5f} (fold {best_fold_idx_pos})"
    f" | (flip): {best_fold_k_neg:.5f} (fold {best_fold_idx_neg})"
)
print(
    f"Refined-threshold OOF QWK (no flip): {ref_oof_k_pos:.5f} | (flip): {ref_oof_k_neg:.5f}"
)
print("Per-fold val kappas (no flip):", np.round(fold_k_pos, 5))
print("Per-fold val kappas (flip):   ", np.round(fold_k_neg, 5))
print("Global kappa by fold-threshold (no flip):", np.round(global_k_pos, 5))
print("Global kappa by fold-threshold (flip):   ", np.round(global_k_neg, 5))
print("Chosen thresholds:", chosen_t)

test_sev_used = (-test_sev) if use_flip else test_sev
test_diags = predict_with_thresholds(test_sev_used, chosen_t)

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

print("Test predicted label distribution:")
print(sub["diagnosis"].value_counts().sort_index())
