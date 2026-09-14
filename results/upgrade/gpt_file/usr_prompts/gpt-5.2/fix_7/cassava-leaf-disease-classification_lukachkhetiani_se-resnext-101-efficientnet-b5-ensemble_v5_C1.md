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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8618918102145663

# 6. Current score

0.55082

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.23057) has done: 'I remove the hard failure on a missing checkpoint and instead fall back to using EfficientNet-B5 ImageNet weights when `b5.pth` is not available, so the notebook can run end-to-end and generate a valid `submission.csv`. I also fix the device/dtype mismatch by ensuring the loaded model weights are moved to the same device as the input tensor after loading. These changes preserve the existing inference pipeline (same architecture, same preprocessing, same argmax-over-softmax decision) while making it runnable and typically much higher-scoring than an untrained random model when no checkpoint is present. Finally, I add a small amount of defensive I/O checking so missing images don’t crash submission generation.'
- What this solution (achieved 0.11996) has done: 'Your current score is low because when the checkpoint is missing, the model’s final classifier layer is randomly initialized, so predictions are essentially random despite using an ImageNet backbone. To move the score toward the target while preserving your core inference pipeline (EfficientNet-B5 + CLAHE + argmax over softmax), I keep the architecture identical but, in the fallback case, I replace the random head with the pretrained 1000-class ImageNet head and map its logits to 5 cassava classes via a fixed grouping (no training, no extra data). I also normalize inputs using the official EfficientNet-B5 ImageNet mean/std so the pretrained weights behave as intended; this is a minimal preprocessing correction that typically yields a large accuracy gain versus unnormalized uint8 tensors. If your `b5.pth` exists, the code path remains effectively unchanged (still loads your checkpoint and uses the 5-class head).'
- What this solution (achieved 0.30605) has done: 'Your score is far below the target, so we should improve accuracy without changing the core inference logic (EfficientNet-B5 + CLAHE preprocessing + argmax decision). The biggest issue is the “missing checkpoint” fallback: the current fixed 1000→5 grouping is essentially arbitrary and produces poor labels. I keep the exact same model and preprocessing, but replace that mapping with a small, deterministic, **self-supervised** test-time adaptation: build 5 prototypes from the test-set image embeddings and classify by nearest prototype (no labels, no training loop changes, and still argmax-based). If the checkpoint exists, the behavior remains unchanged (uses your 5-class head directly), so this only improves the weak fallback path.'
- What this solution (achieved 0.35277) has done: 'Your current score is far below the target, and the main limiting factor is the ImageNet-fallback path: k-means cluster IDs are arbitrary and don’t correspond to the cassava label IDs, so accuracy stays low. Keeping your core logic (EfficientNet-B5 + CLAHE + argmax decision) intact, I only change the fallback to add a tiny, deterministic “cluster→label” calibration step using the provided `train.csv` labels (no training loop, no loss changes, no architecture changes). Concretely: extract embeddings for a small, fixed number of training images per class, assign them to the same 5 prototypes, then map each prototype to the majority class among the assigned labeled samples; inference then uses argmax over prototype similarity and outputs the mapped label. This preserves the existing checkpoint path exactly, only improving the weak fallback path in a legitimate way.'
- What this solution (achieved 0.55082) has done: 'Your score is far below the target, so the safest way to move toward it (without changing your model/loop) is to fix a key mismatch between how prototypes are built and how inference assigns clusters: you currently run k-means on **test** embeddings but learn the cluster→label mapping using **train** embeddings assigned to those test-derived centers, which is unstable and tends to collapse accuracy. I keep the exact same EfficientNet-B5 feature extractor + k-means + argmax logic, but instead build prototypes from a small, balanced labeled subset of **train** images (same per_class budget), then map each prototype to its majority class (with a deterministic tie-break). Inference remains “embed → cosine-sim to 5 prototypes → argmax → mapped label”, and the checkpoint path remains unchanged. This should substantially increase accuracy toward your target while staying within runtime and preserving core semantics.'

# 9. Code solution

## === cell 0
import os
import glob
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import tqdm

warnings.filterwarnings("ignore")

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")

CKPT_PATH = "/kaggle/working/b5.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



## === cell 1
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def build_model(
    num_classes: int = 5,
    use_imagenet_weights: bool = True,
    keep_imagenet_head: bool = False,
):
    weights = (
        torchvision.models.EfficientNet_B5_Weights.IMAGENET1K_V1
        if use_imagenet_weights
        else None
    )
    model = torchvision.models.efficientnet_b5(weights=weights)

    if not keep_imagenet_head:
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model


use_imagenet_fallback = not os.path.exists(CKPT_PATH)

keep_imagenet_head = use_imagenet_fallback
model = build_model(
    num_classes=5,
    use_imagenet_weights=True,
    keep_imagenet_head=keep_imagenet_head,
)

if os.path.exists(CKPT_PATH):
    state = torch.load(CKPT_PATH, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state_dict = state["state_dict"]
    else:
        state_dict = state

    clean_state = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        clean_state[nk] = v

    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    print(
        "Loaded checkpoint. Missing keys:",
        len(missing),
        "Unexpected keys:",
        len(unexpected),
    )
else:
    print(
        f"WARNING: Checkpoint not found at {CKPT_PATH}. Using ImageNet-pretrained EfficientNet-B5 fallback."
    )

model = model.to(device).eval()

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)




## === cell 2
def process(image_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.resize(image_bgr, (512, 512))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    x = x.to(device)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x


if not os.path.exists(SAMPLE_SUB_PATH):
    raise FileNotFoundError(f"sample_submission.csv not found at: {SAMPLE_SUB_PATH}")
if not os.path.isdir(TEST_IMG_DIR):
    raise FileNotFoundError(f"test_images directory not found at: {TEST_IMG_DIR}")

sub_df = pd.read_csv(SAMPLE_SUB_PATH)
test_names = sub_df["image_id"].tolist()
test_paths = [os.path.join(TEST_IMG_DIR, n) for n in test_names]

if not os.path.exists(TRAIN_CSV_PATH):
    raise FileNotFoundError(f"train.csv not found at: {TRAIN_CSV_PATH}")
if not os.path.isdir(TRAIN_IMG_DIR):
    raise FileNotFoundError(f"train_images directory not found at: {TRAIN_IMG_DIR}")


@torch.no_grad()
def extract_embeddings(paths, batch_size=16):
    feats = []
    valid_idx = []
    feature_extractor = model.features
    avgpool = model.avgpool

    for i in tqdm.tqdm(
        range(0, len(paths), batch_size), desc="Extract feats", leave=False
    ):
        batch_paths = paths[i : i + batch_size]
        xs = []
        idxs = []
        for j, p in enumerate(batch_paths):
            img = cv2.imread(p)
            if img is None:
                continue
            xs.append(process(img))
            idxs.append(i + j)
        if not xs:
            continue
        x = torch.cat(xs, dim=0)
        f = feature_extractor(x)
        f = avgpool(f)
        f = torch.flatten(f, 1)
        f = F.normalize(f, dim=1)
        feats.append(f.detach().cpu())
        valid_idx.extend(idxs)

    if feats:
        feats = torch.cat(feats, dim=0)
    else:
        feats = torch.empty((0, 2048), dtype=torch.float32)
    return feats, valid_idx


@torch.no_grad()
def kmeans_5(feats: torch.Tensor, iters: int = 12, seed: int = 123):
    N, C = feats.shape
    if N == 0:
        return torch.zeros((5, C), dtype=torch.float32)

    g = torch.Generator()
    g.manual_seed(seed)

    centers = torch.empty((5, C), dtype=torch.float32)
    first = torch.randint(0, N, (1,), generator=g).item()
    centers[0] = feats[first]
    d2 = torch.full((N,), float("inf"))
    for k in range(1, 5):
        dist = 1.0 - (feats @ centers[k - 1].unsqueeze(1)).squeeze(1)
        d2 = torch.minimum(d2, dist**2)
        probs = d2 / (d2.sum() + 1e-12)
        idx = torch.multinomial(probs, 1, generator=g).item()
        centers[k] = feats[idx]

    for _ in range(iters):
        sims = feats @ centers.T
        assign = torch.argmax(sims, dim=1)

        new_centers = torch.zeros_like(centers)
        for k in range(5):
            m = assign == k
            cnt = int(m.sum().item())
            if cnt > 0:
                new_centers[k] = feats[m].mean(dim=0)
            else:
                idx = torch.randint(0, N, (1,), generator=g).item()
                new_centers[k] = feats[idx]
        centers = F.normalize(new_centers, dim=1)
    return centers


def build_balanced_calib_set(
    train_df: pd.DataFrame, per_class: int = 80, seed: int = 123
):
    rng = np.random.RandomState(seed)
    rows = []
    for y in range(5):
        cls = train_df[train_df["label"] == y]
        if len(cls) == 0:
            continue
        take = min(per_class, len(cls))
        idxs = rng.choice(cls.index.values, size=take, replace=False)
        rows.append(train_df.loc[idxs])
    calib_df = pd.concat(rows, axis=0).reset_index(drop=True)
    return calib_df


def build_cluster_to_label_map_from_counts(count: np.ndarray, seed: int = 123):
    rng = np.random.RandomState(seed)
    mapping = np.zeros((5,), dtype=np.int64)
    for c in range(5):
        row = count[c]
        mx = row.max()
        best = np.flatnonzero(row == mx)
        if len(best) == 0:
            mapping[c] = 0
        elif len(best) == 1:
            mapping[c] = int(best[0])
        else:
            mapping[c] = int(rng.choice(best, 1)[0])
    return mapping


prototypes = None
cluster_to_label = None

if use_imagenet_fallback:
    train_df = pd.read_csv(TRAIN_CSV_PATH)

    calib_df = build_balanced_calib_set(train_df, per_class=80, seed=123)
    calib_paths = [
        os.path.join(TRAIN_IMG_DIR, n) for n in calib_df["image_id"].tolist()
    ]
    calib_labels = calib_df["label"].astype(int).values

    feats_calib, valid_idx_calib = extract_embeddings(calib_paths, batch_size=16)
    if feats_calib.shape[0] == 0:
        prototypes = torch.zeros((5, 2048), dtype=torch.float32)
        cluster_to_label = np.arange(5, dtype=np.int64)
        print("WARNING: No calibration embeddings extracted; using identity mapping.")
    else:
        prototypes = kmeans_5(feats_calib, iters=12, seed=123)

        valid_labels = calib_labels[np.array(valid_idx_calib, dtype=np.int64)]
        sims = feats_calib @ prototypes.T
        assign = torch.argmax(sims, dim=1).numpy()

        count = np.zeros((5, 5), dtype=np.int64)
        for c, y in zip(assign, valid_labels):
            count[int(c), int(y)] += 1

        cluster_to_label = build_cluster_to_label_map_from_counts(count, seed=123)
        print("Cluster->label mapping:", cluster_to_label.tolist())

labels = [0] * len(test_paths)

for idx, path in enumerate(tqdm.tqdm(test_paths, total=len(test_paths), desc="Infer")):
    img = cv2.imread(path)
    if img is None:
        labels[idx] = 0
        continue

    topred = process(img)
    with torch.no_grad():
        out = model(topred)

        if use_imagenet_fallback:
            f = model.features(topred)
            f = model.avgpool(f)
            f = torch.flatten(f, 1)
            f = F.normalize(f, dim=1).detach().cpu()

            sims = (f @ prototypes.T).squeeze(0)
            cluster = int(torch.argmax(sims, dim=0).item())
            pred = int(cluster_to_label[cluster])
        else:
            pred = int(torch.argmax(F.softmax(out, dim=1), dim=1).detach().cpu().item())

    labels[idx] = pred

sub_df["label"] = labels
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote", sub_path, "with shape:", sub_df.shape)
print(sub_df.head())
