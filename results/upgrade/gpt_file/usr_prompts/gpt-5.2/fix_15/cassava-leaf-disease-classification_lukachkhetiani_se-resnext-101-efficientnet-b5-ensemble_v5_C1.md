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

0.60164

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.23057) has done: 'I remove the hard failure on a missing checkpoint and instead fall back to using EfficientNet-B5 ImageNet weights when `b5.pth` is not available, so the notebook can run end-to-end and generate a valid `submission.csv`. I also fix the device/dtype mismatch by ensuring the loaded model weights are moved to the same device as the input tensor after loading. These changes preserve the existing inference pipeline (same architecture, same preprocessing, same argmax-over-softmax decision) while making it runnable and typically much higher-scoring than an untrained random model when no checkpoint is present. Finally, I add a small amount of defensive I/O checking so missing images don’t crash submission generation.'
- What this solution (achieved 0.11996) has done: 'Your current score is low because when the checkpoint is missing, the model’s final classifier layer is randomly initialized, so predictions are essentially random despite using an ImageNet backbone. To move the score toward the target while preserving your core inference pipeline (EfficientNet-B5 + CLAHE + argmax over softmax), I keep the architecture identical but, in the fallback case, I replace the random head with the pretrained 1000-class ImageNet head and map its logits to 5 cassava classes via a fixed grouping (no training, no extra data). I also normalize inputs using the official EfficientNet-B5 ImageNet mean/std so the pretrained weights behave as intended; this is a minimal preprocessing correction that typically yields a large accuracy gain versus unnormalized uint8 tensors. If your `b5.pth` exists, the code path remains effectively unchanged (still loads your checkpoint and uses the 5-class head).'
- What this solution (achieved 0.30605) has done: 'Your score is far below the target, so we should improve accuracy without changing the core inference logic (EfficientNet-B5 + CLAHE preprocessing + argmax decision). The biggest issue is the “missing checkpoint” fallback: the current fixed 1000→5 grouping is essentially arbitrary and produces poor labels. I keep the exact same model and preprocessing, but replace that mapping with a small, deterministic, **self-supervised** test-time adaptation: build 5 prototypes from the test-set image embeddings and classify by nearest prototype (no labels, no training loop changes, and still argmax-based). If the checkpoint exists, the behavior remains unchanged (uses your 5-class head directly), so this only improves the weak fallback path.'
- What this solution (achieved 0.35277) has done: 'Your current score is far below the target, and the main limiting factor is the ImageNet-fallback path: k-means cluster IDs are arbitrary and don’t correspond to the cassava label IDs, so accuracy stays low. Keeping your core logic (EfficientNet-B5 + CLAHE + argmax decision) intact, I only change the fallback to add a tiny, deterministic “cluster→label” calibration step using the provided `train.csv` labels (no training loop, no loss changes, no architecture changes). Concretely: extract embeddings for a small, fixed number of training images per class, assign them to the same 5 prototypes, then map each prototype to the majority class among the assigned labeled samples; inference then uses argmax over prototype similarity and outputs the mapped label. This preserves the existing checkpoint path exactly, only improving the weak fallback path in a legitimate way.'
- What this solution (achieved 0.55082) has done: 'Your score is far below the target, so the safest way to move toward it (without changing your model/loop) is to fix a key mismatch between how prototypes are built and how inference assigns clusters: you currently run k-means on **test** embeddings but learn the cluster→label mapping using **train** embeddings assigned to those test-derived centers, which is unstable and tends to collapse accuracy. I keep the exact same EfficientNet-B5 feature extractor + k-means + argmax logic, but instead build prototypes from a small, balanced labeled subset of **train** images (same per_class budget), then map each prototype to its majority class (with a deterministic tie-break). Inference remains “embed → cosine-sim to 5 prototypes → argmax → mapped label”, and the checkpoint path remains unchanged. This should substantially increase accuracy toward your target while staying within runtime and preserving core semantics.'
- What this solution (achieved 0.59641) has done: 'Your current score (0.55082) is far below the target (0.86189), so we should cautiously improve accuracy without changing the model architecture or adding training. The biggest low-risk gain in your fallback path is to stop using k-means (which can mix classes) and instead build 5 class prototypes directly as the mean embedding per known label from a small balanced labeled calibration subset; inference remains “embed → cosine similarity → argmax → label”. This preserves the same EfficientNet-B5 feature extractor, the same preprocessing (CLAHE + ImageNet normalization), and the same argmax decision semantics, but makes the fallback mapping stable and label-aligned. I also keep the checkpoint path unchanged; these changes only affect the ImageNet-fallback path.'
- What this solution (achieved 0.59492) has done: 'Your current score (0.59641) is well below the target (0.86189), and the main weakness is the ImageNet-fallback path: using raw EfficientNet features without matching the backbone’s expected input resolution and inference settings reduces embedding quality and hurts the prototype classifier. I keep your exact model and prototype/argmax logic, but make the preprocessing resolution match EfficientNet-B5’s default (456) so the pretrained features are more consistent. I also run the backbone in true inference mode (disable autograd, use autocast on CUDA) and extract features in mini-batches during test inference (same computations, just batched), which tends to reduce numeric noise and improves throughput without changing semantics. These are minimal changes focused specifically on improving fallback accuracy toward your target while still producing the same required `submission.csv`.'
- What this solution (achieved 0.60277) has done: 'Your current gap to the target is large (0.59492 → 0.86189), and the bottleneck is the ImageNet-fallback classifier: it uses only a small labeled subset (per_class=120) to form prototypes, which is too noisy for this 5-class problem. Keeping the exact same backbone, preprocessing, embedding extraction, cosine-similarity, and argmax decision logic, I only increase the labeled calibration budget to use more train images per class (still just computing mean embeddings; no training and no new model). I also make prototype building more robust by doing a streaming mean (avoids holding all calibration embeddings in memory) and by consistently running the feature extractor in the same batched/AMP inference mode. These are minimal, directly score-relevant changes to improve class prototype quality and should move accuracy upward toward your target.'
- What this solution (achieved 0.60688) has done: 'Your current score (0.60277) is far below the target (0.86189), so we should improve the fallback path while preserving the same backbone, preprocessing, cosine-similarity + argmax decision, and no-training constraint. The smallest high-impact change is to build prototypes from a larger, more representative labeled calibration set (use more images per class) and to make those prototypes more robust by averaging multiple deterministic “views” (lightweight test-time augmentation) of each calibration image during embedding extraction. This keeps your core logic identical (EfficientNet-B5 features → normalized embeddings → cosine similarity to 5 prototypes → argmax) but reduces prototype noise, typically improving accuracy. The checkpoint path remains effectively unchanged.'
- What this solution (achieved 0.59865) has done: 'Your current score (0.60688) is far below the target (0.86189), so we should improve accuracy in the weakest path (missing-checkpoint ImageNet fallback) while keeping the same backbone, preprocessing, cosine-similarity + argmax decision, and no-training constraint. The most minimal, high-impact fix is to make prototype building closer to the true class centers by (1) removing heavy TTA during prototype construction (it can blur fine-grained class differences in feature space) and (2) using a simple, deterministic “trimmed mean” aggregation per class to reduce the influence of outliers/mislabeled/noisy images without changing the classifier logic. Inference remains identical (embed → cosine sim to 5 prototypes → argmax), and the checkpoint path is unchanged. These changes are directly score-relevant and should move the score upward toward your target without altering the core approach.'
- What this solution (achieved 0.60164) has done: 'Your current score (0.59865) is far below the target (0.86189), so we should improve accuracy in the ImageNet-fallback path while keeping the same backbone, preprocessing (CLAHE + ImageNet norm), and the same “prototype cosine-similarity + argmax” decision. The smallest high-impact fix is to make prototype building more representative by (1) using a deterministic, class-stratified larger calibration subset and (2) computing prototypes as a robust mean of *multiple* embeddings per image (light TTA) but only during prototype construction (not changing the inference logic). This keeps the core semantics identical (EffNet-B5 features → normalized embedding → cosine similarity → argmax label) while reducing prototype noise and class overlap. The checkpoint path remains unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.60164) has done: 'I fix the crash by ensuring the EfficientNet preset transform receives a supported input type (PIL Image) instead of a NumPy array. This is a minimal, score-positive correction because it restores the intended ImageNet preprocessing (resize/crop/normalize) that your fallback prototype pipeline relies on. I also keep your existing CLAHE + 456 resize logic intact and add a small safety conversion to `uint8` before creating the PIL image to avoid dtype edge cases. No model architecture, feature extraction, cosine-similarity argmax logic, or file paths are changed, and the script write a valid `submission.csv`.'

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

EFFNET_B5_RES = 456



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

EFFNET_WEIGHTS = torchvision.models.EfficientNet_B5_Weights.IMAGENET1K_V1
EFFNET_PREPROCESS = EFFNET_WEIGHTS.transforms()

INFER_CTX = torch.inference_mode



## === cell 2
from PIL import Image


def _apply_clahe_bgr(image_bgr: np.ndarray) -> np.ndarray:
    img = image_bgr
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img


def process(image_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.resize(image_bgr, (EFFNET_B5_RES, EFFNET_B5_RES))
    img = _apply_clahe_bgr(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img)

    x = EFFNET_PREPROCESS(img_pil).unsqueeze(0).to(device)  # [1,3,H,W]
    return x


def embed_from_bgr(image_bgr: np.ndarray, tta: bool = False) -> torch.Tensor:
    x = process(image_bgr)  # [1,3,H,W]

    if tta:
        x = torch.cat([x, torch.flip(x, dims=[3])], dim=0)  # [2,3,H,W]

    use_amp = device.type == "cuda"
    autocast_ctx = torch.cuda.amp.autocast if use_amp else torch.cpu.amp.autocast

    with INFER_CTX():
        with autocast_ctx(enabled=use_amp):
            f = model.features(x)
            f = model.avgpool(f)
            f = torch.flatten(f, 1)
        f = F.normalize(f.float(), dim=1)

    if tta:
        f = F.normalize(f.mean(dim=0, keepdim=True), dim=1)  # [1,D]
    return f  # on device


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


@torch.no_grad()
def build_class_prototypes_from_labeled_subset_streaming(
    train_df: pd.DataFrame,
    per_class: int = 1800,
    seed: int = 123,
    batch_size: int = 16,
    use_tta: bool = True,
    trim_q: float = 0.06,
):
    calib_df = build_balanced_calib_set(train_df, per_class=per_class, seed=seed)
    calib_paths = [
        os.path.join(TRAIN_IMG_DIR, n) for n in calib_df["image_id"].tolist()
    ]
    calib_labels = calib_df["label"].astype(int).values

    feat_dim = 2048  # EfficientNet-B5 pooled feature dim
    per_class_feats = [[] for _ in range(5)]

    for i in tqdm.tqdm(
        range(0, len(calib_paths), batch_size),
        desc="Build prototypes",
        leave=False,
    ):
        batch_paths = calib_paths[i : i + batch_size]
        batch_labels = calib_labels[i : i + batch_size]

        for p, y in zip(batch_paths, batch_labels):
            img = cv2.imread(p)
            if img is None:
                continue
            f = embed_from_bgr(img, tta=use_tta)  # [1,D] on device
            f = F.normalize(f.float(), dim=1).detach().cpu()[0]  # [D] CPU
            per_class_feats[int(y)].append(f)

    prototypes = torch.zeros((5, feat_dim), dtype=torch.float32)
    counts = torch.zeros((5,), dtype=torch.float32)

    for y in range(5):
        if len(per_class_feats[y]) == 0:
            prototypes[y] = torch.zeros((feat_dim,), dtype=torch.float32)
            counts[y] = 0.0
            continue

        feats = torch.stack(per_class_feats[y], dim=0)  # [N,D]
        counts[y] = float(feats.shape[0])

        mean0 = F.normalize(feats.mean(dim=0, keepdim=True), dim=1)  # [1,D]
        sims = (F.normalize(feats, dim=1) @ mean0.T).squeeze(1)  # [N]
        n = sims.numel()
        k = int(n * trim_q)
        if 2 * k < n and k > 0:
            keep_idx = torch.argsort(sims, descending=True)[k : n - k]
            feats = feats[keep_idx]

        proto = feats.mean(dim=0, keepdim=True)  # [1,D]
        prototypes[y] = proto[0]

    prototypes = F.normalize(prototypes, dim=1)
    ok = bool(torch.any(counts > 0).item())
    return prototypes, ok


prototypes = None

if use_imagenet_fallback:
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    prototypes, ok = build_class_prototypes_from_labeled_subset_streaming(
        train_df,
        per_class=1800,
        seed=123,
        batch_size=16,
        use_tta=True,
        trim_q=0.06,
    )
    if not ok:
        print(
            "WARNING: No calibration embeddings extracted; prototypes are zeros (will default to label 0)."
        )

labels = [0] * len(test_paths)

use_amp = device.type == "cuda"
autocast_ctx = torch.cuda.amp.autocast if use_amp else torch.cpu.amp.autocast

if use_imagenet_fallback:
    batch_size = 16
    for i in tqdm.tqdm(
        range(0, len(test_paths), batch_size),
        total=(len(test_paths) + batch_size - 1) // batch_size,
        desc="Infer",
    ):
        batch_paths = test_paths[i : i + batch_size]
        fs = []
        idxs = []
        for j, p in enumerate(batch_paths):
            img = cv2.imread(p)
            if img is None:
                labels[i + j] = 0
                continue
            f = embed_from_bgr(img, tta=True)  # keep inference TTA
            fs.append(f)
            idxs.append(i + j)

        if not fs:
            continue

        f = torch.cat(fs, dim=0)  # [B,D] on device
        f = F.normalize(f.float(), dim=1).detach().cpu()

        sims = f @ prototypes.T  # [B,5]
        preds = torch.argmax(sims, dim=1).numpy().astype(int).tolist()
        for k, pred in enumerate(preds):
            labels[idxs[k]] = int(pred)
else:
    for idx, path in enumerate(
        tqdm.tqdm(test_paths, total=len(test_paths), desc="Infer")
    ):
        img = cv2.imread(path)
        if img is None:
            labels[idx] = 0
            continue

        topred = process(img)
        with INFER_CTX():
            with autocast_ctx(enabled=use_amp):
                out = model(topred)
            pred = int(
                torch.argmax(F.softmax(out.float(), dim=1), dim=1).detach().cpu().item()
            )

        labels[idx] = pred

sub_df["label"] = labels
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote", sub_path, "with shape:", sub_df.shape)
print(sub_df.head())
