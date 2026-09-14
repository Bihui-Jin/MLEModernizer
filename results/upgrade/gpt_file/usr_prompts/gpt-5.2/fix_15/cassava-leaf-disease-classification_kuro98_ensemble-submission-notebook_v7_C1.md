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

3.12

# 3. Installed packages

geopandas==0.14.4
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

0.8981565427621638

# 6. Current score

0.53513

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.54709) has done: 'I fix the pipeline so it always runs end-to-end and writes a valid `submission.csv`. The main blocker is missing external model files under `/kaggle/input/...`, so I add a safe fallback that uses a pretrained torchvision model when those files aren’t present, without changing the overall “image → model → argmax label” semantics. I also fix inference bugs that cause invalid submission length: the test loader must not be shuffled, and test filenames must be returned in the exact sample submission order. Finally, I make TTA deterministic at inference by using non-random augmentations (flips/rotations), keeping the intended “average across TTAs” logic while ensuring stability.'
- What this solution (achieved 0.10613) has done: 'Your current score is far below the target, and the biggest (minimal) win without changing your “image → model → argmax label” core logic is to make the fallback path actually use a meaningful classifier instead of a randomly-initialized 5-class head. I keep the exact same preprocessing/inference flow and simply (1) keep EfficientNet-B0’s pretrained 1000-class head, then (2) map its ImageNet predictions to your 5 cassava classes using a tiny “nearest-label” rule built from the provided disease names in `label_num_to_disease_map.json`. This preserves evaluation semantics (still argmax over 5 classes) while giving you a much stronger baseline than random weights, moving accuracy toward the target. I also replace `Random*` TTA ops with deterministic equivalents (same transforms, no RNG) to avoid any residual nondeterminism at inference.'
- What this solution (achieved 0.05531) has done: 'Your score (0.10613) is far below the target (0.8982), so we should improve performance with the smallest change that doesn’t alter your core “image → model → argmax label” inference semantics. The main issue is the current fallback mapping from ImageNet-1000 logits to 5 cassava classes is effectively unrelated to the task, so it behaves close to guessing. I keep the same fallback EfficientNet-B0 inference and TTA averaging, but replace only the mapping step with a deterministic, task-aligned approach: extract EfficientNet features and use a lightweight nearest-centroid classifier built from the provided training images (no training loop, just computing class mean embeddings). This stays within the same pipeline structure (pretrained model forward pass + deterministic post-processing) and should move accuracy substantially toward the target while still writing a valid `submission.csv`.'
- What this solution (achieved 0.62145) has done: 'I fix the immediate runtime error by importing `create_feature_extractor` from the correct torchvision module (`torchvision.models.feature_extraction`) instead of the non-existent `torchvision.models.feature_extraction` attribute. This also ensure the fallback functions (`_build_class_centroids`, `_extract_feats_from_eff_batch`, etc.) are defined, eliminating the downstream `NameError`s in later cells. I keep the pipeline and “EfficientNet feature extractor + nearest-centroid argmax” logic unchanged, only making compatibility fixes needed for torchvision 0.21 and ensuring everything runs end-to-end to write a valid `submission.csv`. No score-decreasing changes are introduced; this should restore the intended fallback behavior (and thus improve score vs the broken run).'
- What this solution (achieved 0.61547) has done: 'Your current score (0.62145) is well below the target (0.89816), so we should improve accuracy while keeping the same “EfficientNet feature extractor → nearest-centroid argmax” fallback core logic intact. The biggest low-risk gain is to build stronger class centroids by (1) using EfficientNet’s recommended preprocessing weights (not generic ImageNet mean/std alone) and (2) computing centroids over more (ideally all) training images per class, since centroid quality is currently bottlenecked by `max_per_class=256`. I also switch the test-time resize/crop ordering to the standard “resize then center-crop” for both centroid building and test inference, keeping the same semantics but improving feature consistency. These changes should move the score upward toward the target without changing model architecture, training loops, or the nearest-centroid approach.'
- What this solution (achieved 0.61734) has done: 'The timeout is dominated by expensive per-sample CPU preprocessing (decode → resize(600) → center crop → two resizes) done inside `__getitem__`, plus heavy TTA list handling and (in fallback) an extremely expensive full-train centroid build with TTA. To keep core logic identical while speeding up, the main changes are: move all geometric ops into batched, vectorized GPU transforms applied once per batch (same ops, same order), make the Dataset return only the raw image tensor, and use a fast `collate_fn` to avoid Python overhead. For fallback centroids, keep the exact centroid definition but compute features once per image (no TTA during centroid build) and reuse the same deterministic TTA only at test-time; this preserves the classifier semantics (nearest-centroid on the same feature space) and greatly cuts compute. Finally, reduce unnecessary tensor stacking/splitting overhead in TTA aggregation by reshaping instead of `torch.split`/`torch.stack` while keeping identical averages.'
- What this solution (achieved 0.61734) has done: 'I fix the immediate runtime crash (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by preventing the TensorFlow import from executing in this PyTorch notebook, since TF is only used as an optional centroid-building acceleration and is currently breaking the run in this environment. Then I make the TFRecord centroid path automatically disable itself when TensorFlow isn’t available, cleanly falling back to the existing image-directory centroid builder (same nearest-centroid core logic). Finally, I keep submission ordering and non-shuffled test inference intact so the generated `submission.csv` is valid and aligned with `sample_submission.csv`, without changing the model/inference semantics.'
- What this solution (achieved 0.60912) has done: 'We keep your exact “EfficientNet feature extractor → L2-normalized embeddings → nearest-centroid argmax (with deterministic TTA on test)” core logic, but strengthen the centroids without changing the model or adding any training. The main accuracy limiter is that centroids are currently computed from *only one* deterministic view per train image while test uses 4-view TTA, creating a feature-space mismatch; we compute centroids as the **mean embedding over the same deterministic TTAs** (still just averaging, no learning). To stay within the 600s budget, we do this efficiently in batches by expanding each batch to (T*B) and reshaping for averaging, mirroring your fast inference path. This should move the score upward toward the target while preserving evaluation semantics and still writing a valid `submission.csv`.'
- What this solution (achieved 0.65321) has done: 'Your current score (0.60912) is far below the target (0.89816), so we should improve accuracy with the smallest changes that keep your existing “EfficientNet feature extractor → L2-normalized embeddings → nearest-centroid argmax (with deterministic TTA on test)” fallback logic intact. The biggest low-risk gain is to make the centroid classifier less sensitive to class imbalance by using **per-class whitening (standardization) computed from train embeddings**, then computing centroids in that whitened space; this is still nearest-centroid argmax, just in a better-scaled feature space. We also fix a small but important issue: your current TTA uses `Random*` transforms (even with p=1), which can still introduce RNG-dependent behavior; we replace them with deterministic v2 functional transforms (flip/rotate) to stabilize both centroid-building and test inference. These changes preserve the overall pipeline and output semantics while plausibly moving accuracy upward toward the target.'
- What this solution (achieved 0.53513) has done: 'To move accuracy upward toward your target while keeping the exact same core “EfficientNet feature extractor → (optional whitening) → nearest-centroid argmax with deterministic TTA” logic, the smallest high-impact fix is to correct the whitening step: you currently compute per-class mean/std but then apply a *global* mean/std averaged across classes, which weakens separation. I change whitening to be consistent with the centroid classifier by using a single global mean/std computed over all train embeddings (same features, no learning), and apply that same global whitening to both centroids and test embeddings so the similarity is computed in the same space. This is a minimal, metric-aligned calibration change (still nearest-centroid argmax) and should improve accuracy without touching the model architecture or adding any training loops. I keep submission ordering and deterministic TTA unchanged and still write `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import io
import json
import warnings
import random
from glob import glob

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

tf = None

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
random.seed(3407)

cudnn.deterministic = True
cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
test_dir = f"{DATA_ROOT}/test_images/"
train_dir = f"{DATA_ROOT}/train_images/"
train_csv_path = f"{DATA_ROOT}/train.csv"
sample_sub_path = f"{DATA_ROOT}/sample_submission.csv"
label_map_path = f"{DATA_ROOT}/label_num_to_disease_map.json"
train_tfrecords_dir = f"{DATA_ROOT}/train_tfrecords/"

eff_img_size = 528
vit_img_size = 384
batch_size = 16
num_workers = 2
num_classes = 5
tta = True


def _try_load_model(path: str):
    if path and os.path.exists(path):
        m = torch.load(path, map_location=device)
        try:
            m = m.to(device)
        except Exception:
            pass
        try:
            m.eval()
        except Exception:
            pass
        return m
    return None


vit_model = _try_load_model("/kaggle/input/vit-v1-update/vit_v1_1.pt")
eff_model = _try_load_model("/kaggle/input/efficient-net/vit_cont_3.pt")
linear_head = _try_load_model("/kaggle/input/linear-head/linear_cls.pt")

fallback_mode = (vit_model is None) or (eff_model is None) or (linear_head is None)
if fallback_mode:
    warnings.warn(
        "One or more external model files were not found under /kaggle/input/. "
        "Falling back to torchvision EfficientNet_B0 pretrained ImageNet backbone + "
        "a deterministic nearest-centroid classifier built from train.csv embeddings "
        "(still argmax over 5 classes)."
    )

    import torchvision
    from torchvision.models.feature_extraction import create_feature_extractor

    eff_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    fallback_model = torchvision.models.efficientnet_b0(weights=eff_weights).to(device)
    fallback_model.eval()

    fallback_extractor = create_feature_extractor(
        fallback_model, return_nodes={"avgpool": "feat"}
    ).to(device)
    fallback_extractor.eval()

    with open(label_map_path, "r") as f:
        label_name_map = json.load(f)

    def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return x / (x.norm(dim=1, keepdim=True) + eps)

    def _extract_feats_from_eff_batch(eff_batch: torch.Tensor) -> torch.Tensor:
        out = fallback_extractor(eff_batch)["feat"]  # (B, 1280, 1, 1)
        out = out.flatten(1)  # (B, 1280)
        return _l2_normalize(out)

    cassava_centroids = None
    cassava_whiten_mu = None  # (D,)
    cassava_whiten_std = None  # (D,)



## === cell 1
try:
    from torchvision.io import read_image

    _HAS_TV_READ_IMAGE = True
except Exception:
    _HAS_TV_READ_IMAGE = False


class CassavaDataset(VisionDataset):
    """Custom dataset for Cassava test images.

    Submission requires preserving exact test image order given by sample_submission.csv.
    """

    def __init__(self, data_dir, image_ids):
        super().__init__(root=data_dir)
        self.images = list(image_ids)  # ordered list from sample_submission

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root, filename)

        if _HAS_TV_READ_IMAGE:
            img = read_image(img_path)  # uint8, [C,H,W]
            if img.shape[0] == 1:
                img = img.expand(3, -1, -1)
            elif img.shape[0] > 3:
                img = img[:3]
        else:
            img = Image.open(img_path).convert("RGB")
        return img, filename

    def __len__(self):
        return len(self.images)




## === cell 2
if fallback_mode:
    import torchvision

    eff_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    eff_mean = list(eff_weights.transforms().mean)
    eff_std = list(eff_weights.transforms().std)
else:
    eff_mean = [0.485, 0.456, 0.406]
    eff_std = [0.229, 0.224, 0.225]

resize_shorter_600 = v2.Resize(
    600, interpolation=InterpolationMode.BICUBIC, antialias=True
)
cc_600 = v2.CenterCrop((600, 600))

resize_vit = v2.Resize(
    (vit_img_size, vit_img_size),
    interpolation=InterpolationMode.BICUBIC,
    antialias=True,
)
resize_eff = v2.Resize(
    (eff_img_size, eff_img_size),
    interpolation=InterpolationMode.BICUBIC,
    antialias=True,
)

norm_tf = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=eff_mean, std=eff_std),
    ]
)

if tta:

    def _tta_identity(x):
        return x

    def _tta_hflip(x):
        return v2.functional.horizontal_flip(x)

    def _tta_vflip(x):
        return v2.functional.vertical_flip(x)

    def _tta_rot90(x):
        return v2.functional.rotate(
            x, angle=90, interpolation=InterpolationMode.BICUBIC, expand=False
        )

    ttas = [_tta_identity, _tta_hflip, _tta_vflip, _tta_rot90]
else:
    ttas = None

sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].tolist()

test_dataset = CassavaDataset(test_dir, image_ids=test_image_ids)


def _collate_images_and_names(batch):
    imgs, names = zip(*batch)
    return list(imgs), list(names)


_cpu_cnt = os.cpu_count() or 2
_tuned_workers = min(8, max(num_workers, _cpu_cnt // 2))
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=_tuned_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_tuned_workers > 0),
    prefetch_factor=8 if _tuned_workers > 0 else None,
    collate_fn=_collate_images_and_names,
)

normalizer = torch.nn.Softmax(dim=1)

if fallback_mode:
    import torchvision
    from torchvision.models.feature_extraction import create_feature_extractor

    def _build_class_centroids_from_tfrecords(
        train_csv: str,
        tfrecord_dir: str,
        transform,
    ):
        if tf is None:
            raise RuntimeError("tensorflow is not available for TFRecord fallback.")

        df = pd.read_csv(train_csv).sort_values("image_id", kind="mergesort")

        tfrec_files = sorted(glob(os.path.join(tfrecord_dir, "*.tfrec")))
        if not tfrec_files:
            raise RuntimeError(f"No TFRecord files found under: {tfrecord_dir}")

        feature_desc = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_id": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }

        def _parse(x):
            ex = tf.io.parse_single_example(x, feature_desc)
            return ex["image"], ex["target"]

        AUTOTUNE = getattr(tf.data, "AUTOTUNE", None)
        ds = (
            tf.data.TFRecordDataset(
                tfrec_files,
                compression_type="GZIP",
                num_parallel_reads=AUTOTUNE if AUTOTUNE is not None else 2,
            )
            .map(_parse, num_parallel_calls=AUTOTUNE if AUTOTUNE is not None else 2)
            .prefetch(AUTOTUNE if AUTOTUNE is not None else 1)
        )

        D = 1280
        feats_sum = torch.zeros((num_classes, D), device=device)
        feats_sumsq = torch.zeros((num_classes, D), device=device)
        counts = torch.zeros((num_classes,), dtype=torch.long, device=device)

        g_sum = torch.zeros((D,), device=device)
        g_sumsq = torch.zeros((D,), device=device)
        g_count = torch.zeros((), dtype=torch.long, device=device)

        batch_imgs = []
        batch_lbls = []

        T = len(ttas) if (tta and ttas is not None) else 1

        def _flush():
            nonlocal batch_imgs, batch_lbls, feats_sum, feats_sumsq, counts, g_sum, g_sumsq, g_count
            if not batch_imgs:
                return
            eff_batch = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)
            feats = _extract_feats_from_eff_batch(eff_batch)  # (B,1280) already L2

            lbls = torch.tensor(batch_lbls, device=device, dtype=torch.long)
            feats_sum.index_add_(0, lbls, feats)
            feats_sumsq.index_add_(0, lbls, feats * feats)
            counts.index_add_(0, lbls, torch.ones_like(lbls, dtype=torch.long))

            g_sum += feats.sum(dim=0)
            g_sumsq += (feats * feats).sum(dim=0)
            g_count += feats.shape[0]

            batch_imgs, batch_lbls = [], []

        with torch.inference_mode():
            for img_bytes, target in ds.as_numpy_iterator():
                c = int(target)
                if c < 0 or c >= num_classes:
                    continue
                try:
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception:
                    continue

                img = resize_shorter_600(img)
                img = cc_600(img)

                if tta and ttas is not None:
                    views = []
                    for t in ttas:
                        x = t(img)
                        x = resize_eff(x)
                        views.append(transform(x))
                    views = torch.stack(views, dim=0)  # (T,3,H,W)
                    img_t = views.mean(dim=0)
                else:
                    img = resize_eff(img)
                    img_t = transform(img)

                batch_imgs.append(img_t)
                batch_lbls.append(c)
                if len(batch_imgs) >= batch_size:
                    _flush()

        _flush()

        eps = 1e-6

        mu_c = torch.zeros((num_classes, D), device=device)
        std_c = torch.ones((num_classes, D), device=device)
        centroids = torch.zeros((num_classes, D), device=device)

        centroid_counts = [0] * num_classes
        fallback_counts = [0] * num_classes

        for c in range(num_classes):
            n = int(counts[c].item())
            centroid_counts[c] = n
            if n == 0:
                fallback_counts[c] = int((df["label"] == c).sum())
                continue
            mu_c[c] = feats_sum[c] / n
            var = feats_sumsq[c] / n - mu_c[c] * mu_c[c]
            std_c[c] = torch.sqrt(torch.clamp(var, min=eps))
            centroids[c] = _l2_normalize(mu_c[c].unsqueeze(0)).squeeze(0)

        g_n = int(g_count.item())
        if g_n > 0:
            mu_g = g_sum / g_n
            var_g = g_sumsq / g_n - mu_g * mu_g
            std_g = torch.sqrt(torch.clamp(var_g, min=eps))
        else:
            mu_g = torch.zeros((D,), device=device)
            std_g = torch.ones((D,), device=device)

        return centroids, mu_g, std_g, centroid_counts, fallback_counts

    try:
        (
            cassava_centroids,
            cassava_whiten_mu,
            cassava_whiten_std,
            centroid_counts,
            centroid_fallback_counts,
        ) = _build_class_centroids_from_tfrecords(
            train_csv=train_csv_path,
            tfrecord_dir=train_tfrecords_dir,
            transform=norm_tf,
        )
        print(
            "Fallback centroids built from TFRecords. Counts per class:",
            centroid_counts,
        )
    except Exception as e:
        warnings.warn(
            f"TFRecord centroid build unavailable/failed ({e}). Falling back to image-dir build."
        )

        class CassavaTrainDataset(VisionDataset):
            def __init__(self, image_dir, df):
                super().__init__(root=image_dir)
                self.df = df.reset_index(drop=True)

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                fname = row["image_id"]
                y = int(row["label"])
                img_path = os.path.join(self.root, fname)
                if _HAS_TV_READ_IMAGE:
                    img = read_image(img_path)
                    if img.shape[0] == 1:
                        img = img.expand(3, -1, -1)
                    elif img.shape[0] > 3:
                        img = img[:3]
                else:
                    img = Image.open(img_path).convert("RGB")
                return img, y

        def _collate_train(batch):
            imgs, ys = zip(*batch)
            return list(imgs), torch.tensor(ys, dtype=torch.long)

        def _to_batched_tensor(imgs):
            imgs = [v2.ToImage()(im) for im in imgs]
            return torch.stack(imgs, dim=0)

        def _build_class_centroids(
            train_csv: str,
            image_dir: str,
            transform,
        ):
            df = pd.read_csv(train_csv).sort_values("image_id", kind="mergesort")
            train_ds = CassavaTrainDataset(image_dir, df)

            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=_tuned_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=(_tuned_workers > 0),
                prefetch_factor=8 if _tuned_workers > 0 else None,
                collate_fn=_collate_train,
            )

            D = 1280
            feats_sum = torch.zeros((num_classes, D), device=device)
            feats_sumsq = torch.zeros((num_classes, D), device=device)
            counts = torch.zeros((num_classes,), dtype=torch.long, device=device)

            g_sum = torch.zeros((D,), device=device)
            g_sumsq = torch.zeros((D,), device=device)
            g_count = torch.zeros((), dtype=torch.long, device=device)

            T = len(ttas) if (tta and ttas is not None) else 1

            fallback_extractor.eval()
            with torch.inference_mode():
                for raw_imgs, ys in train_loader:
                    bsz = len(raw_imgs)
                    base = _to_batched_tensor(raw_imgs)
                    if torch.cuda.is_available():
                        base = base.pin_memory()
                    base = base.to(device, non_blocking=True)

                    base600 = resize_shorter_600(base)
                    base600 = cc_600(base600)

                    if tta and ttas is not None:
                        eff_list = []
                        for t in ttas:
                            x = t(base600)
                            eff_list.append(transform(resize_eff(x)))
                        eff_inputs_cat = torch.cat(eff_list, dim=0)  # (T*B,3,H,W)
                        feats = _extract_feats_from_eff_batch(
                            eff_inputs_cat
                        )  # (T*B,1280)
                        feats = feats.view(T, bsz, -1).mean(dim=0)  # (B,1280)
                        feats = _l2_normalize(feats)
                    else:
                        eff_inputs = transform(resize_eff(base600))
                        feats = _extract_feats_from_eff_batch(eff_inputs)  # (B,1280)

                    ys = ys.to(device, non_blocking=True)
                    feats_sum.index_add_(0, ys, feats)
                    feats_sumsq.index_add_(0, ys, feats * feats)
                    counts.index_add_(0, ys, torch.ones_like(ys, dtype=torch.long))

                    g_sum += feats.sum(dim=0)
                    g_sumsq += (feats * feats).sum(dim=0)
                    g_count += feats.shape[0]

            eps = 1e-6
            mu_c = torch.zeros((num_classes, D), device=device)
            std_c = torch.ones((num_classes, D), device=device)
            centroids = torch.zeros((num_classes, D), device=device)
            centroid_counts = [0] * num_classes
            fallback_counts = [0] * num_classes

            for c in range(num_classes):
                n = int(counts[c].item())
                centroid_counts[c] = n
                if n == 0:
                    centroids[c].zero_()
                    fallback_counts[c] = int((df["label"] == c).sum())
                else:
                    mu_c[c] = feats_sum[c] / n
                    var = feats_sumsq[c] / n - mu_c[c] * mu_c[c]
                    std_c[c] = torch.sqrt(torch.clamp(var, min=eps))
                    centroids[c] = _l2_normalize(mu_c[c].unsqueeze(0)).squeeze(0)

            g_n = int(g_count.item())
            if g_n > 0:
                mu_g = g_sum / g_n
                var_g = g_sumsq / g_n - mu_g * mu_g
                std_g = torch.sqrt(torch.clamp(var_g, min=eps))
            else:
                mu_g = torch.zeros((D,), device=device)
                std_g = torch.ones((D,), device=device)

            return centroids, mu_g, std_g, centroid_counts, fallback_counts

        (
            cassava_centroids,
            cassava_whiten_mu,
            cassava_whiten_std,
            centroid_counts,
            centroid_fallback_counts,
        ) = _build_class_centroids(
            train_csv=train_csv_path,
            image_dir=train_dir,
            transform=norm_tf,
        )
        print(
            "Fallback centroids built from image files. Counts per class:",
            centroid_counts,
        )



## === cell 3
all_names = []
all_preds = []

if fallback_mode:
    fallback_model.eval()
    fallback_extractor.eval()
else:
    vit_model.eval()
    eff_model.eval()
    linear_head.eval()


def _to_batched_tensor(imgs):
    imgs = [v2.ToImage()(im) for im in imgs]
    return torch.stack(imgs, dim=0)


def _apply_whitening_global(
    feats: torch.Tensor, mu_g: torch.Tensor, std_g: torch.Tensor
) -> torch.Tensor:
    std_g = std_g.clamp_min(1e-6)
    wf = (feats - mu_g.unsqueeze(0)) / std_g.unsqueeze(0)
    return _l2_normalize(wf)


if (
    fallback_mode
    and (cassava_centroids is not None)
    and (cassava_whiten_mu is not None)
    and (cassava_whiten_std is not None)
):
    cassava_centroids = _apply_whitening_global(
        cassava_centroids, cassava_whiten_mu, cassava_whiten_std
    )

with torch.inference_mode():
    for batch_idx, (raw_imgs, filenames) in enumerate(test_loader):
        bsz = len(filenames)

        base = _to_batched_tensor(raw_imgs)
        if torch.cuda.is_available():
            base = base.pin_memory()
        base = base.to(device, non_blocking=True)

        base600 = resize_shorter_600(base)
        base600 = cc_600(base600)

        if tta:
            vit_list = []
            eff_list = []
            for t in ttas:
                x = t(base600)
                vit_list.append(norm_tf(resize_vit(x)))
                eff_list.append(norm_tf(resize_eff(x)))
            vit_inputs_cat = torch.cat(vit_list, dim=0)  # (T*B,3,H,W)
            eff_inputs_cat = torch.cat(eff_list, dim=0)
        else:
            vit_inputs_cat = norm_tf(resize_vit(base600))
            eff_inputs_cat = norm_tf(resize_eff(base600))

        if fallback_mode:
            if tta:
                feats = _extract_feats_from_eff_batch(eff_inputs_cat)  # (T*B,1280)
                T = len(ttas)
                feats_tta = feats.view(T, bsz, -1)  # (T,B,1280)
                mean_feats = feats_tta.mean(dim=0)
                mean_feats = _l2_normalize(mean_feats)

                if (cassava_whiten_mu is not None) and (cassava_whiten_std is not None):
                    mean_feats = _apply_whitening_global(
                        mean_feats, cassava_whiten_mu, cassava_whiten_std
                    )

                logits = mean_feats @ cassava_centroids.T
                pred_labels = torch.argmax(logits, 1).tolist()
            else:
                feats = _extract_feats_from_eff_batch(eff_inputs_cat)  # (B,1280)

                if (cassava_whiten_mu is not None) and (cassava_whiten_std is not None):
                    feats = _apply_whitening_global(
                        feats, cassava_whiten_mu, cassava_whiten_std
                    )

                logits = feats @ cassava_centroids.T
                pred_labels = torch.argmax(logits, 1).tolist()
        else:
            if tta:
                vit_outputs = vit_model(vit_inputs_cat)
                eff_outputs = eff_model(eff_inputs_cat)

                T = len(ttas)
                vit_mean_logits = vit_outputs.view(T, bsz, -1).mean(dim=0)
                eff_mean_logits = eff_outputs.view(T, bsz, -1).mean(dim=0)

                logit_inputs = torch.cat([vit_mean_logits, eff_mean_logits], dim=1)
                outputs = linear_head(logit_inputs)

                probs = normalizer(outputs)
                pred_labels = torch.argmax(probs, 1).tolist()
            else:
                vit_outputs = vit_model(vit_inputs_cat)
                eff_outputs = eff_model(eff_inputs_cat)

                logit_inputs = torch.cat([vit_outputs, eff_outputs], dim=1)
                outputs = linear_head(logit_inputs)

                probs = normalizer(outputs)
                pred_labels = torch.argmax(probs, 1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)



## === cell 4
pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})

pred_df = pred_df.set_index("image_id").reindex(sample_sub["image_id"])
if pred_df["label"].isna().any():
    pred_df["label"] = pred_df["label"].fillna(0).astype(int)
else:
    pred_df["label"] = pred_df["label"].astype(int)

submission = pred_df.reset_index()
submission.to_csv("submission.csv", index=False)

print("submission.csv written:", submission.shape)
print(submission.head())
