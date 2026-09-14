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

2.7

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8700513750377757

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'The DataLoader error is caused by `os.listdir()` picking up a nested `test_images/` directory inside the provided path, so `io.imread()` is trying to read a folder as an image; we fix this by filtering to files with image extensions and sorting for determinism. The “Invalid submission length” then comes from predicting the wrong number of rows (because the dataset included that directory); using `sample_submission.csv` as the authoritative test list guarantees the exact required 2676 image_ids and correct ordering. These changes are execution/format fixes and keep the model/inference logic the same; they should also improve score versus random because the code now successfully load the provided weights (if present) and run inference over the correct test set. The script always write a valid `submission.csv` with the required columns and length.'
- What this solution (achieved 0.09492) has done: 'Your current score (0.11024) is consistent with either missing/incorrect pretrained weights (random-ish predictions) or a silent mismatch when loading the checkpoint into the modified classifier. To move accuracy toward the target with minimal change, I (1) initialize EfficientNet-B4 with ImageNet weights when the competition checkpoint isn’t found, (2) load checkpoints more robustly (handle common key prefixes like `model.`) and require a strict load when a checkpoint exists so we don’t silently run with partially-loaded/random heads, and (3) ensure deterministic test ordering and a safe fallback prediction if any unexpected NaNs appear (shouldn’t trigger). This keeps the same model, transforms, and inference flow, but avoids the “random init” failure mode that tanks accuracy. It still write a valid `submission.csv` matching `sample_submission.csv` exactly.'
- What this solution (achieved 0.23318) has done: 'Your current score indicates the model is effectively guessing; the most likely cause is that the competition checkpoint is not being loaded (wrong `/kaggle/input/{folder_name}` path), so you’re using an ImageNet-pretrained EfficientNet with a randomly-initialized 5-class head. To move accuracy toward the target with minimal change and identical inference semantics, I (1) make checkpoint discovery robust by searching all `/kaggle/input/**` subfolders for the expected `efficientnet-b4-e10.pt`, and (2) if a checkpoint still isn’t found, use a safe fallback that at least avoids a random head by using the ImageNet head and mapping its 1000 classes to 5 labels deterministically (this won’t hit the target, but should move well above ~0.09). I also enforce evaluation mode and deterministic ordering as you already do, without changing the model architecture or transforms. The script still write a valid `submission.csv` matching `sample_submission.csv` exactly.'
- What this solution (achieved 0.61099) has done: 'Your score (0.23318) is far below the target (0.87005), so the smallest likely reason is still that the intended cassava-trained checkpoint isn’t being used for inference. I keep your model/inference logic identical, but make checkpoint discovery more robust (support common filename variants and pick the best match), and add a strict safety check that the loaded classifier really is 5-class when a checkpoint is found. If no checkpoint is found, I remove the ImageNet “modulo mapping” fallback (which is effectively random for this task) and instead output a stable majority-class fallback computed from `train.csv` (still not great, but typically better than random and moves accuracy upward without changing model core logic). The submission still be generated from `sample_submission.csv` to guarantee correct ordering and row count.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is still well below the target (0.87005), so we should make a minimal, score-relevant change that improves inference quality without changing the model/training approach. The most likely issue is a preprocessing mismatch: your test transform uses `CenterCrop(512)` even though EfficientNet-B4 is typically trained/inferred at 380px, and cassava checkpoints are commonly trained with a `Resize(380,380)` pipeline; this mismatch can materially hurt accuracy. I change only the test-time spatial preprocessing to `Resize(380,380)` (keeping normalization, model, checkpoint loading, and submission alignment identical) and also make cuDNN deterministic to avoid run-to-run drift while keeping semantics the same. This should move accuracy upward toward the target while staying within the “minimal changes” constraint.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is still far below the target (0.87005), and the most likely minimal fix is improving test-time inference (not changing training) to better match common cassava EfficientNet checkpoints. I keep the same model, checkpoint loading, dataset, and submission alignment, but add standard test-time augmentation (horizontal flip) and average the probabilities across the two views, which typically gives a meaningful accuracy lift for this task. I also switch inference to softmax-probability averaging (instead of argmax per-view) while still outputting the same final hard labels, preserving evaluation semantics. These changes are small, deterministic, and should move the score upward toward the target without altering architecture or training loops.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.87005), so the smallest likely improvement is to make test-time preprocessing match how cassava EfficientNet-B4 checkpoints are typically trained: include a center-crop after a slightly larger resize, and use a slightly larger inference batch size for more stable BatchNorm behavior (still in eval) and faster throughput. I keep your model, checkpoint loading, and TTA logic identical, but adjust only the spatial transform (Resize->CenterCrop) and DataLoader batch_size/num_workers to reduce input distribution mismatch and improve accuracy. Submission generation remain anchored to `sample_submission.csv` for exact ordering/length. These are minimal inference-only changes, expected to move accuracy upward toward the target without altering training/architecture/loss.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.87005), so we should make a small inference-only change that commonly yields a meaningful accuracy gain for cassava EfficientNet checkpoints without altering the model or training logic. I keep your checkpoint loading, EfficientNet-B4 architecture, dataset, and submission alignment identical, but expand test-time augmentation from 2 views (original + horizontal flip) to 4 views (add vertical flip and both flips) and average softmax probabilities across all views. This preserves evaluation semantics (still outputs hard labels) while typically improving accuracy toward the target. I also keep determinism settings and ensure the submission remains exactly aligned to `sample_submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is still far below the target (0.87005), so we should make a minimal, inference-only change that is likely to improve accuracy without changing the model architecture or any training logic. The biggest low-risk gain now is to add a small “multi-crop” test-time augmentation: run the model on center + 4 corner crops (using the same 420→380 crop geometry you already use), average softmax probabilities, then apply your existing flip TTA on top. This keeps the same EfficientNet-B4 checkpoint usage, same normalization, same argmax labeling, and still writes a submission aligned exactly to `sample_submission.csv`. I also keep determinism settings unchanged so results are stable run-to-run.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is still far below the target (0.87005), so we need a small inference-only change that can improve accuracy without changing the model architecture or any training logic. The most likely remaining mismatch is normalization: many cassava EfficientNet-B4 checkpoints were trained with *dataset-specific* mean/std rather than ImageNet mean/std, and using the wrong normalization can significantly hurt accuracy. I keep your exact model, checkpoint-loading, multi-crop + flip TTA, and submission alignment, but (1) compute cassava train-set mean/std once from a limited number of training images for speed and (2) use that mean/std in the test transforms only when a cassava checkpoint is found (falling back to ImageNet stats otherwise). This is a minimal, deterministic change aimed to move the score upward toward the target.'

# 9. Code solution

## === cell 0
from __future__ import print_function, division

import os
import warnings
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from skimage import io
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_full_name = "efficientnet-b4-e10"
model_name = "efficientnet-b4"
folder_name = "effnetmodelv18"




## === cell 1
class ToTensor(object):
    def __call__(self, image, force_apply=True):
        output = image.transpose((2, 0, 1))
        return torch.from_numpy(output)




## === cell 2
class TestDataset(Dataset):
    def __init__(self, root_dir, image_ids, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = list(image_ids)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)

        image = io.imread(img_path)

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        if self.transform:
            out = self.transform(image=image)
            image = out["image"]

        return img_name, image




## === cell 3
data_root = "/kaggle/input/cassava-leaf-disease-classification"
test_root = os.path.join(data_root, "test_images")

sample_path = os.path.join(data_root, "sample_submission.csv")
sample_sub = pd.read_csv(sample_path)
test_image_ids = sample_sub["image_id"].tolist()

test_image = TestDataset(root_dir=test_root, image_ids=test_image_ids, transform=None)

testloader = DataLoader(
    test_image, batch_size=32, shuffle=False, num_workers=4, pin_memory=use_cuda
)

expected_ckpt_name = model_full_name + ".pt"
candidate_names = [
    expected_ckpt_name,
    model_full_name + ".pth",
    model_name + ".pt",
    model_name + ".pth",
]

candidate_paths = []
for root, dirs, files in os.walk("/kaggle/input"):
    for fname in candidate_names:
        if fname in files:
            candidate_paths.append(os.path.join(root, fname))

intended_path = os.path.join("/kaggle/input", folder_name, expected_ckpt_name)
if os.path.exists(intended_path):
    PATH = intended_path
elif len(candidate_paths) > 0:

    def _rank(p):
        base = os.path.basename(p)
        exact = 0 if base == expected_ckpt_name else 1
        return (exact, len(p), p)

    candidate_paths = sorted(candidate_paths, key=_rank)
    PATH = candidate_paths[0]
else:
    PATH = intended_path  # stable string for logging

ckpt_found = os.path.exists(PATH)

train_path = os.path.join(data_root, "train.csv")
majority_class = 0
if os.path.exists(train_path):
    train_df = pd.read_csv(train_path)
    if "label" in train_df.columns and len(train_df) > 0:
        majority_class = int(train_df["label"].value_counts().idxmax())




## === cell 4
def estimate_mean_std(train_csv_path, train_images_root, n_samples=512, seed=42):
    if (not os.path.exists(train_csv_path)) or (not os.path.exists(train_images_root)):
        return None

    df = pd.read_csv(train_csv_path)
    if "image_id" not in df.columns or len(df) == 0:
        return None

    ids = df["image_id"].tolist()
    rng = np.random.RandomState(seed)
    if len(ids) > n_samples:
        ids = list(rng.choice(ids, size=n_samples, replace=False))

    c_sum = np.zeros(3, dtype=np.float64)
    c_sqsum = np.zeros(3, dtype=np.float64)
    pix_count = 0

    for img_id in ids:
        p = os.path.join(train_images_root, img_id)
        if not os.path.exists(p):
            continue
        im = io.imread(p)
        if im.ndim == 2:
            im = np.stack([im] * 3, axis=-1)
        elif im.shape[2] == 4:
            im = im[:, :, :3]

        im = im.astype(np.float32) / 255.0
        flat = im.reshape(-1, 3)
        c_sum += flat.sum(axis=0)
        c_sqsum += (flat * flat).sum(axis=0)
        pix_count += flat.shape[0]

    if pix_count == 0:
        return None

    mean = c_sum / float(pix_count)
    var = (c_sqsum / float(pix_count)) - (mean * mean)
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)
    return tuple(mean.tolist()), tuple(std.tolist())


train_images_root = os.path.join(data_root, "train_images")
cassava_stats = None
if ckpt_found:
    cassava_stats = estimate_mean_std(
        train_path, train_images_root, n_samples=512, seed=seed
    )

norm_mean = (0.485, 0.456, 0.406)
norm_std = (0.229, 0.224, 0.225)

if ckpt_found and cassava_stats is not None:
    norm_mean, norm_std = cassava_stats




## === cell 5
transform_center = A.Compose(
    [
        A.Resize(height=420, width=420, interpolation=1, p=1.0),
        A.CenterCrop(height=380, width=380, p=1.0),
        A.Normalize(
            mean=norm_mean,
            std=norm_std,
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(),
    ]
)

transform_tl = A.Compose(
    [
        A.Resize(height=420, width=420, interpolation=1, p=1.0),
        A.Crop(x_min=0, y_min=0, x_max=380, y_max=380, p=1.0),  # top-left
        A.Normalize(
            mean=norm_mean,
            std=norm_std,
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(),
    ]
)

transform_tr = A.Compose(
    [
        A.Resize(height=420, width=420, interpolation=1, p=1.0),
        A.Crop(x_min=40, y_min=0, x_max=420, y_max=380, p=1.0),  # top-right
        A.Normalize(
            mean=norm_mean,
            std=norm_std,
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(),
    ]
)

transform_bl = A.Compose(
    [
        A.Resize(height=420, width=420, interpolation=1, p=1.0),
        A.Crop(x_min=0, y_min=40, x_max=380, y_max=420, p=1.0),  # bottom-left
        A.Normalize(
            mean=norm_mean,
            std=norm_std,
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(),
    ]
)

transform_br = A.Compose(
    [
        A.Resize(height=420, width=420, interpolation=1, p=1.0),
        A.Crop(x_min=40, y_min=40, x_max=420, y_max=420, p=1.0),  # bottom-right
        A.Normalize(
            mean=norm_mean,
            std=norm_std,
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(),
    ]
)

if ckpt_found:
    weights = None
    model = efficientnet_b4(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
else:
    weights = EfficientNet_B4_Weights.DEFAULT
    model = efficientnet_b4(weights=weights)

model = model.to(device)

if ckpt_found:
    state = torch.load(PATH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
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

    model.load_state_dict(state, strict=True)

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        if (
            hasattr(model.classifier[1], "out_features")
            and model.classifier[1].out_features != 5
        ):
            raise RuntimeError(
                "Checkpoint loaded but model head is not 5-class; out_features=%s"
                % str(model.classifier[1].out_features)
            )

model.eval()

names = []
predicted = []

crop_transforms = [
    transform_center,
    transform_tl,
    transform_tr,
    transform_bl,
    transform_br,
]

with torch.no_grad():
    for names_batch, images_batch_np in testloader:
        if ckpt_found:
            probs_accum = None
            n_views = 0

            if isinstance(images_batch_np, torch.Tensor):
                images_list = [
                    img.permute(1, 2, 0).cpu().numpy() for img in images_batch_np
                ]
            else:
                images_list = list(images_batch_np)

            for tform in crop_transforms:
                crop_tensors = []
                for im in images_list:
                    if im.ndim == 2:
                        im = np.stack([im] * 3, axis=-1)
                    elif im.shape[2] == 4:
                        im = im[:, :, :3]
                    out = tform(image=im)
                    crop_tensors.append(out["image"])
                x0 = (
                    torch.stack(crop_tensors, dim=0)
                    .to(device, non_blocking=True)
                    .float()
                )

                logits0 = model(x0)
                prob0 = torch.softmax(logits0, dim=1)

                x_h = torch.flip(x0, dims=[3])
                logits_h = model(x_h)
                prob_h = torch.softmax(logits_h, dim=1)

                x_v = torch.flip(x0, dims=[2])
                logits_v = model(x_v)
                prob_v = torch.softmax(logits_v, dim=1)

                x_hv = torch.flip(x0, dims=[2, 3])
                logits_hv = model(x_hv)
                prob_hv = torch.softmax(logits_hv, dim=1)

                probs_view = 0.25 * (prob0 + prob_h + prob_v + prob_hv)

                if probs_accum is None:
                    probs_accum = probs_view
                else:
                    probs_accum = probs_accum + probs_view
                n_views += 1

            probs = probs_accum / float(n_views)
            preds = torch.argmax(probs, dim=1).detach().cpu().numpy()
        else:
            preds = np.full((len(names_batch),), majority_class, dtype=np.int64)

        names.extend(list(names_batch))
        predicted.extend(list(preds))




## === cell 6
pred_map = dict(zip(names, predicted))
result = sample_sub.copy()
result["label"] = result["image_id"].map(pred_map)

if result["label"].isna().any():
    result["label"] = result["label"].fillna(0)

result["label"] = result["label"].astype(int)

assert os.path.exists(sample_path)
assert list(result.columns) == ["image_id", "label"]
assert len(result) == len(sample_sub)
assert result["label"].notna().all()

result.to_csv("submission.csv", index=False)

assert os.path.exists("submission.csv")
print(result.head())
print("Wrote submission.csv with", len(result), "rows")
print(
    "Checkpoint used:",
    (
        PATH
        if ckpt_found
        else "None found (used majority-class fallback=%d)" % majority_class
    ),
)
if ckpt_found:
    print("Normalization mean/std used:", norm_mean, norm_std)
else:
    print("Searched for:", candidate_names)
    print("Found", len(candidate_paths), "candidates:", candidate_paths[:10])
